"""DDP-safe checkpoint save / load.

Design notes
------------
- rank-0-only save; atomic write (temp file + os.replace in the same dir)
- every tensor is moved to CPU before saving -> checkpoints are portable
  across devices
- the payload is primitives + tensors only (see _to_serializable), so
  torch.load(weights_only=True) -- the safe default since torch 2.6 --
  works. Unsafe pickle loading is opt-in (cfg.trust_checkpoint); by
  default a checkpoint file cannot execute code during load.
- load: every rank reads weights independently (no NCCL collective around
  file I/O -> no deadlock); only tiny metadata goes through
  broadcast_object_list
- failure policy: missing file -> fresh start; broadcast / model-state
  failures are FATAL (a silent resume onto random weights corrupts the
  run); optimizer / scheduler / scaler problems log a warning and continue
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from bleu_control import default_bleu_state
from utils import is_master, log_info

logger = logging.getLogger(__name__)


def default_history() -> dict:
    return {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "bleu_scores": [],
        "lr": [],
    }


# -------------------------
# Serialization helpers
# -------------------------
def _to_serializable(obj: Any) -> Any:
    """Recursively convert an object into JSON/pickle-safe primitives.

    Keeps the checkpoint payload loadable with weights_only=True:
    Path -> str, dataclasses/objects -> dict of their __dict__, anything
    exotic -> str.
    """
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return str(obj)
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_serializable(v) for k, v in obj.__dict__.items()}
        except Exception:
            return str(obj)
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _recursive_apply(obj: Any, fn) -> Any:
    """Apply fn to every tensor leaf inside nested dict/list/tuple/set."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _recursive_apply(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_apply(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_recursive_apply(v, fn) for v in obj)
    if isinstance(obj, set):
        return {_recursive_apply(v, fn) for v in obj}
    return obj


def _state_dict_to_cpu(state_dict: Any) -> Any:
    return _recursive_apply(
        state_dict, lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t
    )


def _move_state_dict_to_device(state_dict: Any, device: torch.device) -> Any:
    return _recursive_apply(
        state_dict, lambda t: t.to(device) if isinstance(t, torch.Tensor) else t
    )


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _torch_load(path: str, map_location, unsafe: bool):
    if unsafe:
        logger.warning(
            "Loading checkpoint with weights_only=False "
            "(executes arbitrary pickle): %s", path,
        )
        return torch.load(path, map_location=map_location, weights_only=False)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # older checkpoints may pickle TorchVersion; allow-list exactly that
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            return torch.load(path, map_location=map_location, weights_only=True)


# -------------------------
# Save
# -------------------------
def save_checkpoint(
    rank: int,
    model,
    scheduler,
    epoch: int,
    path: str,
    bleu_state: dict,
    history: dict,
    cfg,
    save_train_state: bool = True,
    amp_scaler=None,
) -> None:
    """Save a full training checkpoint (rank 0 only, atomic write).

    model may be a raw module or a DDP wrapper (unwrapped via .module).
    save_train_state=False stores weights + metadata only (smaller; no
    optimizer/scheduler/scaler state, so not resumable mid-schedule).
    """
    if not is_master(rank):
        return

    raw_model = model.module if hasattr(model, "module") else model
    model_state = _state_dict_to_cpu(raw_model.state_dict())

    optimizer_state = scheduler_state = scaler_state = None
    if save_train_state:
        optimizer = getattr(scheduler, "optimizer", None)
        if optimizer is not None:
            optimizer_state = _state_dict_to_cpu(optimizer.state_dict())
        if hasattr(scheduler, "state_dict"):
            scheduler_state = scheduler.state_dict()
        if amp_scaler is not None:
            scaler_state = _state_dict_to_cpu(amp_scaler.state_dict())

    state = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "bleu_state": _to_serializable(bleu_state),
        "history": _to_serializable(history),
        "config": _to_serializable(cfg.__dict__ if hasattr(cfg, "__dict__") else cfg),
        "_meta": {
            "pytorch_version": str(torch.__version__),
            "save_train_state": save_train_state,
            "saved_by": "save_checkpoint",
        },
    }

    path = str(path)
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)

    # atomic save: temp file in the target dir, then os.replace
    fd, tmp_path = tempfile.mkstemp(dir=dir_name)
    os.close(fd)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    log_info(rank, f"Checkpoint saved: {path} (save_train_state={save_train_state})")


# -------------------------
# Load
# -------------------------
def load_checkpoint(
    model,
    scheduler,
    checkpoint_path: str,
    device: torch.device,
    rank: int,
    cfg,
    unsafe: bool = False,
    amp_scaler=None,
    map_location=None,
):
    """Restore a checkpoint on EVERY rank; returns (start_epoch, bleu_state, history).

    Missing file -> (1, defaults, defaults) so a fresh start is silent.
    Model-state mismatch is fatal; auxiliary state problems only warn.
    """
    bleu_default = default_bleu_state(cfg)
    history_default = default_history()

    _map = map_location if map_location is not None else device
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    # ---- Step 1: rank 0 loads and broadcasts tiny metadata ----
    ckpt = None
    meta = {"exists": False}
    if is_master(rank):
        if os.path.exists(checkpoint_path):
            ckpt = _torch_load(checkpoint_path, _map, unsafe)
            meta = {"exists": True, "epoch": ckpt.get("epoch", 0)}
        else:
            log_info(rank, f"Checkpoint not found: {checkpoint_path} - starting fresh")

    if distributed:
        meta_list = [meta]
        # FATAL on failure: rank desync here means a collective hang later
        dist.broadcast_object_list(meta_list, src=0)
        meta = meta_list[0]

    if not meta.get("exists", False):
        return 1, bleu_default, history_default

    # ---- Step 2: every other rank loads its own copy ----
    if ckpt is None:
        ckpt = _torch_load(checkpoint_path, _map, unsafe)

    # ---- Step 3: model weights (FATAL on mismatch) ----
    model_state = ckpt.get("model_state")
    if model_state is None:
        raise RuntimeError(f"Checkpoint has no model_state: {checkpoint_path}")
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(model_state, strict=True)

    # ---- Step 4: optimizer (recoverable) ----
    opt_state = ckpt.get("optimizer_state")
    if opt_state is not None:
        optimizer = getattr(scheduler, "optimizer", None)
        if optimizer is not None:
            try:
                optimizer.load_state_dict(opt_state)
                _move_optimizer_state_to_device(optimizer, device)
            except Exception as e:
                logger.warning("Could not restore optimizer state: %s", e)

    # ---- Step 5: LR schedule (recoverable) ----
    sched_state = ckpt.get("scheduler_state")
    if sched_state is not None and hasattr(scheduler, "load_state_dict"):
        try:
            scheduler.load_state_dict(sched_state)
        except Exception as e:
            logger.warning("Could not restore scheduler state: %s", e)

    # ---- Step 6: AMP scaler (recoverable) ----
    scaler_state = ckpt.get("scaler_state")
    if scaler_state is not None and amp_scaler is not None:
        try:
            amp_scaler.load_state_dict(_move_state_dict_to_device(scaler_state, device))
        except Exception as e:
            logger.warning("Could not restore AMP scaler state: %s", e)

    # ---- Step 7: training-progress metadata ----
    start_epoch = ckpt.get("epoch", 0) + 1
    bleu_state = ckpt.get("bleu_state") or dict(bleu_default)
    for k, v in bleu_default.items():
        bleu_state.setdefault(k, v)
    history = ckpt.get("history") or history_default

    log_info(
        rank,
        f"Resumed: epoch={start_epoch}, "
        f"best_bleu={bleu_state.get('best', -1.0):.2f}, "
        f"patience={bleu_state.get('patience', 0)}",
    )
    return start_epoch, bleu_state, history
