"""BLEU-driven training controller.

Every `bleu_interval` epochs, rank 0 decodes a fixed probe slice of the
test set and computes corpus BLEU. The controller reacts to stagnation:

- improvement            -> reset all counters, record new best
- 2+ consecutive misses  -> halve label smoothing (floor 0.0)
- every missed eval      -> halve the LR scale (floor cfg.lr_min_scale),
                            applied inside NoamOpt.rate() via set_lr_scale
- `patience` misses      -> early stop

The state dict is broadcast to all ranks in a single collective so DDP
workers stay in lockstep. The pure update logic (everything except
broadcast_bleu_state) is unit-tested without a process group.
"""

import logging
from typing import Any, Dict

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

STATE_KEYS = ("best", "no_improve", "plateau", "patience", "label_smoothing", "lr_scale")


def default_bleu_state(cfg) -> Dict[str, Any]:
    return {
        "best": -1.0,
        "no_improve": 0,
        "plateau": 0,
        "patience": 0,
        "label_smoothing": cfg.label_smoothing,
        "lr_scale": 1.0,
    }


def update_bleu_state(state: Dict[str, Any], bleu: float, cfg) -> bool:
    """Mutate `state` in place; returns True if `bleu` set a new best."""
    improved = bleu > state["best"] + 1e-4

    if improved:
        state["best"] = bleu
        state["no_improve"] = 0
        state["plateau"] = 0
        state["patience"] = 0
        logger.info("[BLEU] New best: %.2f", bleu)
    else:
        state["no_improve"] += 1
        state["plateau"] += 1
        state["patience"] += 1
        logger.info("[BLEU] No improvement (%d/%d)", state["patience"], cfg.patience)

    # decay label smoothing after 2 consecutive misses
    if state["no_improve"] >= 2 and state["label_smoothing"] > 0.0:
        new_ls = max(state["label_smoothing"] * 0.5, 0.0)
        logger.info("[LS] label_smoothing %.3f -> %.3f", state["label_smoothing"], new_ls)
        state["label_smoothing"] = new_ls
        state["no_improve"] = 0

    # halve the LR scale on every missed eval (floored at cfg.lr_min_scale)
    if state["plateau"] >= 1:
        new_scale = max(state["lr_scale"] * 0.5, cfg.lr_min_scale)
        if new_scale < state["lr_scale"]:
            logger.info("[LR] lr_scale %.3f -> %.3f", state["lr_scale"], new_scale)
        state["lr_scale"] = new_scale
        state["plateau"] = 0

    return improved


def should_stop(state: Dict[str, Any], cfg) -> bool:
    return state["patience"] >= cfg.patience


def broadcast_bleu_state(state: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Sync controller state from rank 0 to all ranks (single collective).

    No-op copy when no process group is initialized (single-process
    runs).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return dict(state)

    tensor = torch.tensor([state[k] for k in STATE_KEYS], dtype=torch.float32, device=device)
    dist.broadcast(tensor, src=0)
    return dict(zip(STATE_KEYS, tensor.tolist()))
