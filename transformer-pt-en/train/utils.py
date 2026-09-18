"""Shared helpers: seeding, rank-aware logging, barriers, training plots."""

import random
from pathlib import Path

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
    axes[1, 0].set_title("BLEU (test probe)")
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
