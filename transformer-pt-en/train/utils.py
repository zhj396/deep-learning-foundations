"""Shared helpers: seeding, rank-aware logging, barriers, probe sharding, plots."""

import random
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist


def is_master(rank: int) -> bool:
    return rank == 0


def log_info(rank: int, msg: str) -> None:
    """Print from rank 0 only (keeps multi-GPU logs readable)."""
    if is_master(rank):
        print(msg, flush=True)


def set_seed(seed: int = 41) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def barrier(device: torch.device) -> None:
    """dist.barrier() that passes device_ids on NCCL (required there).

    No-op when no process group is initialized (single-process runs).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index or 0])
    else:
        dist.barrier()


# -------------------------
# BLEU-probe sharding helpers
# -------------------------
def probe_shard_indices(rank: int, world_size: int, n_total: int) -> List[int]:
    """This rank's indices of a fixed n_total-sentence probe set.

    Strided rather than blocked: a block split leaves the last rank with the
    remainder (and with nothing at all once n_total < world_size), while a
    stride keeps every shard within one sentence of every other one. Corpus
    BLEU does not depend on hypothesis order, so the union of the shards
    scores exactly the whole slice.
    """
    return list(range(rank, n_total, max(1, world_size)))


def gather_by_rank(payload: Any) -> List[Any]:
    """Collect one small payload from every rank, ordered by rank.

    Object collectives pickle their argument, so payloads stay tiny (token-id
    lists, not tensors). Without a process group the payload is returned as
    the only element, which is what a single-process run decoded.
    """
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return [payload]
    gathered: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    return gathered


def all_ranks_ok(ok: torch.Tensor) -> bool:
    """True only when every rank still holds a nonzero vote in `ok` (uint8).

    The MIN-reduce is the collective that decides, so a rank whose decode
    failed cannot leave the others scoring a partial probe set: all ranks
    see the same answer and all ranks abort on the same epoch.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


def plot_training_history(history: dict, save_dir) -> None:
    """Save a 4-panel training summary (loss / accuracy / BLEU / LR)."""
    import matplotlib

    matplotlib.use("Agg")  # headless backends (Kaggle, CI)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Val")
    axes[0, 1].set_title("Token Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    bleu_scores = history.get("bleu_scores") or []
    if bleu_scores:
        epochs, bleus = zip(*bleu_scores)
        axes[1, 0].plot(epochs, bleus, "o-")
    axes[1, 0].set_title("BLEU (validation probe)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True)

    lr = history.get("lr") or []
    if lr:
        axes[1, 1].plot(lr)
    axes[1, 1].set_title("Learning Rate (after scale)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True)

    plt.tight_layout()
    out_path = Path(save_dir) / "training_history.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Training history saved: {out_path}", flush=True)
