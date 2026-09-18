"""Entry point: python main.py [--flags]

Multi-GPU (CUDA devices > 1) -> torch.multiprocessing.spawn DDP (NCCL).
Single GPU or CPU           -> one process group (NCCL / gloo), world_size=1.

Rendezvous uses a FileStore in a unique temp file (env PTEN_DIST_STORE):
single-node runs need no TCP port, which avoids firewall prompts and
libuv-dependent TCPStore builds on Windows.
"""

import os
import sys
import tempfile

import torch
import torch.multiprocessing as mp

from config import Config
from train import train


def main() -> None:
    # training logs may contain Portuguese diacritics; make legacy Windows
    # console code pages tolerant instead of crashing on print
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = Config.from_argparse()
    print(f"Config:\n{cfg}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    # unique rendezvous file for this run; inherited by spawned workers
    fd, store_path = tempfile.mkstemp(prefix="pten_dist_")
    os.close(fd)
    os.environ["PTEN_DIST_STORE"] = store_path

    try:
        use_cuda = torch.cuda.is_available() and not cfg.no_cuda
        n_gpus = torch.cuda.device_count() if use_cuda else 0

        if n_gpus > 1:
            print(f"Launching DDP with {n_gpus} GPUs")
            mp.spawn(train, args=(n_gpus, cfg), nprocs=n_gpus, join=True)
        else:
            print("Single-process run (CPU or 1 GPU), world_size=1")
            train(0, 1, cfg)
    finally:
        try:
            os.remove(store_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
