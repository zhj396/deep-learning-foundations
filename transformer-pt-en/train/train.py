"""Training orchestration: single-process (CPU / 1 GPU) or multi-GPU DDP.

Launched from main.py as train(rank, world_size, cfg). Rank-aware logging
goes through utils.is_master / utils.log_info; validation metrics are
all-reduced inside evaluate_epoch and controller state is broadcast after
every BLEU probe, so all ranks stay in lockstep through early stopping.
"""

import contextlib
import datetime
import glob
import logging
import os
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from bleu_control import (
    broadcast_bleu_state,
    default_bleu_state,
    should_stop,
    update_bleu_state,
)
from checkpoint import default_history, load_checkpoint, save_checkpoint
from config import Config
from dataset import Collator, TranslationDataset
from decoding import beam_decode_batch, greedy_decode_batch
from evaluate import compute_bleu, evaluate_epoch
from loss import LabelSmoothingLoss, accuracy
from model import Transformer
from scheduler import NoamOpt
from utils import barrier, is_master, log_info, plot_training_history, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -------------------------
# One epoch (DDP-safe, AMP + gradient accumulation)
# -------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    scheduler: NoamOpt,
    criterion: nn.Module,
    device: torch.device,
    rank: int = 0,
    scaler: torch.amp.GradScaler = None,
    accumulation_steps: int = 4,
    clip_grad: float = 1.0,
    use_amp: bool = False,
):
    """Run one training epoch; returns (avg_loss, avg_token_accuracy).

    Gradient accumulation skips the DDP all-reduce on non-boundary steps
    via model.no_sync() (only when the model actually is DDP-wrapped).
    """
    model.train()
    optimizer = scheduler.optimizer
    optimizer.zero_grad(set_to_none=True)

    is_ddp = isinstance(model, DDP)
    use_amp = use_amp and device.type == "cuda" and scaler is not None
    autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    n_steps = len(dataloader)

    pbar = tqdm(dataloader, desc="Training", leave=False) if is_master(rank) else dataloader

    for i, (src, tgt) in enumerate(pbar):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

        # optimizer-step boundary: accumulate or flush
        is_boundary = ((i + 1) % accumulation_steps == 0) or ((i + 1) == n_steps)
        sync_ctx = (
            contextlib.nullcontext()
            if (is_boundary or not is_ddp)
            else model.no_sync()
        )

        with sync_ctx:
            if scaler is not None:
                scaler.scale(loss / accumulation_steps).backward()
            else:
                (loss / accumulation_steps).backward()

        if is_boundary:
            if scaler is not None:
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()
            # schedule step AFTER the optimizer step (Noam counts updates)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            total_loss += loss.item()
            total_acc += accuracy(logits, tgt[:, 1:]).item()
        n_batches += 1

        if is_master(rank):
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_acc / n_batches:.4f}")

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


# -------------------------
# BLEU probe (rank 0)
# -------------------------
def _run_bleu_probe(model, bleu_loader, cfg, sp, device: torch.device) -> float:
    """Decode the probe set and return corpus BLEU (raises on failure)."""
    model.eval()
    all_preds, all_refs = [], []

    common = dict(
        max_len=cfg.max_len,
        bos_idx=cfg.bos_idx,
        eos_idx=cfg.eos_idx,
        pad_idx=cfg.pad_idx,
        device=device,
    )
    for src, tgt in bleu_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        if cfg.use_beam:
            preds = beam_decode_batch(
                model, src, beam_size=cfg.beam_size,
                length_penalty=cfg.length_penalty, **common,
            )
        else:
            preds = greedy_decode_batch(model, src, **common)

        all_preds.extend(preds.cpu().tolist())
        all_refs.extend(tgt[:, 1:].cpu().tolist())

    return compute_bleu(
        all_preds, all_refs,
        pad_idx=cfg.pad_idx,
        eos_idx=cfg.eos_idx,
        sp=sp,
        print_samples=cfg.print_samples,
    )


# -------------------------
# Full training run
# -------------------------
def train(rank: int, world_size: int, cfg: Config) -> None:
    # =======================
    # Distributed / device init
    # =======================
    os.environ.setdefault("MASTER_ADDR", cfg.master_addr)
    os.environ.setdefault("MASTER_PORT", str(cfg.master_port))

    use_cuda = torch.cuda.is_available() and not cfg.no_cuda
    backend = "nccl" if use_cuda else "gloo"
    local_rank = rank  # single-node assumption: local rank == global rank

    if use_cuda:
        # bind the device BEFORE init_process_group so NCCL picks the right GPU
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Rendezvous: single-node runs use a FileStore created by main.py (no TCP
    # port -> no firewall / libuv issues on Windows, no port conflicts).
    # Falls back to env:// (torchrun-style MASTER_ADDR/MASTER_PORT) when the
    # store file is not provided.
    store_path = os.environ.get("PTEN_DIST_STORE")
    if store_path is not None:
        store = dist.FileStore(store_path, world_size)
        store.set_timeout(datetime.timedelta(seconds=cfg.timeout))
        init_kwargs = dict(backend=backend, store=store, rank=rank, world_size=world_size)
    else:
        init_kwargs = dict(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=cfg.timeout),
        )

    dist.init_process_group(**init_kwargs)
    log_info(
        rank,
        f"[rank {rank}] Process group ready: backend={backend}, "
        f"world_size={world_size}, device={device}",
    )

    set_seed(cfg.seed + rank)

    # =======================
    # Data
    # =======================
    if cfg.data_vocab is None:
        raise ValueError(
            "--data_vocab is required: directory with {train,valid,test}.{en,pt} "
            "and spm_en_pt.model (see ../data_prep)"
        )
    vocab_dir = Path(cfg.data_vocab)

    sp = spm.SentencePieceProcessor(model_file=str(vocab_dir / "spm_en_pt.model"))
    cfg.vocab_size = sp.get_piece_size()
    log_info(rank, f"Vocab: {cfg.vocab_size} pieces (from {vocab_dir})")

    # src = Portuguese, tgt = English (pt -> en)
    train_dataset = TranslationDataset(
        vocab_dir / "train.pt", vocab_dir / "train.en", sp_processor=sp)
    val_dataset = TranslationDataset(
        vocab_dir / "valid.pt", vocab_dir / "valid.en", sp_processor=sp)
    test_dataset = TranslationDataset(
        vocab_dir / "test.pt", vocab_dir / "test.en", sp_processor=sp)
    log_info(
        rank,
        f"Data: train={len(train_dataset)} valid={len(val_dataset)} "
        f"test={len(test_dataset)}",
    )

    collator = Collator(
        pad_idx=train_dataset.pad_idx,
        eos_idx=train_dataset.eos_idx,
        max_len=cfg.max_len,
    )

    # num_workers=0 is the safe default under DDP; single-process may use more
    num_workers = 0 if world_size > 1 else cfg.num_workers
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

    # BLEU probe set: a fixed slice of the test split in a plain, NON-sharded
    # loader so the control signal is deterministic and rank-independent
    n_probe = min(cfg.bleu_batches * cfg.batch_size, len(test_dataset))
    bleu_loader = DataLoader(
        Subset(test_dataset, range(n_probe)),
        batch_size=cfg.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )

    # =======================
    # Model
    # =======================
    # build + init BEFORE the DDP wrap (DDP broadcasts params at construction)
    model = Transformer(
        src_vocab=cfg.vocab_size,
        tgt_vocab=cfg.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        tie_weights=cfg.tie_weights,
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)

    # static_graph is incompatible with find_unused_parameters; prefer the
    # latter only if the architecture ever grows unused branches
    static_graph = not cfg.find_unused_parameters
    if use_cuda:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            gradient_as_bucket_view=True,  # memory optimisation
            static_graph=static_graph,
            find_unused_parameters=cfg.find_unused_parameters,
        )
    else:
        model = DDP(model, find_unused_parameters=cfg.find_unused_parameters)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(rank, f"Parameters: {total_params:,} total, "
                   f"{trainable_params:,} trainable (tied: {cfg.tie_weights})")

    # lr starts at 0: NoamOpt sets it on the first step()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = NoamOpt(cfg.d_model, cfg.factor, cfg.warmup_steps, optimizer)

    # AMP scaler only makes sense on CUDA
    scaler = torch.amp.GradScaler("cuda") if (cfg.use_amp and use_cuda) else None

    # =======================
    # BLEU controller state
    # =======================
    bleu_state = default_bleu_state(cfg)

    # =======================
    # Checkpoints / resume
    # =======================
    start_epoch = 1
    history = None
    ckpt_dir = Path(cfg.ckpt_dir)
    latest_path = ckpt_dir / "latest_model.pt"

    if is_master(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history = default_history()

    resume_path = None
    if cfg.resume_ckpt and Path(cfg.resume_ckpt).exists():
        resume_path = str(cfg.resume_ckpt)
    elif cfg.resume and latest_path.exists():
        resume_path = str(latest_path)

    if resume_path is not None:
        log_info(rank, f"Resuming from {resume_path}")
        # every rank loads independently (see checkpoint.py design notes)
        start_epoch, bleu_state, loaded_history = load_checkpoint(
            model, scheduler, resume_path, device, rank, cfg,
            unsafe=cfg.trust_checkpoint, amp_scaler=scaler,
        )
        if is_master(rank):
            history = loaded_history
    else:
        log_info(rank, "No checkpoint specified/found - training from scratch")

    # =======================
    # Training loop
    # =======================
    for epoch in range(start_epoch, cfg.epochs + 1):
        # a resumed run may already be in early-stop state
        if epoch == start_epoch and should_stop(bleu_state, cfg):
            log_info(rank, "Resume landed in an early-stop state; exiting")
            break

        log_info(rank, f"=== Epoch {epoch}/{cfg.epochs} ===")
        train_sampler.set_epoch(epoch)

        # rebuilt every epoch: the controller may have decayed label smoothing
        criterion = LabelSmoothingLoss(
            bleu_state["label_smoothing"], cfg.vocab_size, cfg.pad_idx
        ).to(device)

        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, scheduler, criterion, device, rank,
            scaler=scaler,
            accumulation_steps=cfg.accumulation_steps,
            clip_grad=cfg.clip_grad,
            use_amp=cfg.use_amp,
        )
        log_info(
            rank,
            f"Epoch {epoch} train: loss={train_loss:.4f} "
            f"acc={train_acc * 100:.2f}% ({time.time() - t0:.1f}s)",
        )

        t0 = time.time()
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device,
            pad_idx=cfg.pad_idx, use_amp=cfg.use_amp, is_master=is_master(rank),
        )
        # evaluate_epoch already all-reduces; no second reduction here
        log_info(
            rank,
            f"Epoch {epoch} valid: loss={val_loss:.4f} "
            f"acc={val_acc * 100:.2f}% ({time.time() - t0:.1f}s)",
        )

        barrier(device)

        current_lr = scheduler.get_lr()
        log_info(rank, f"LR: {current_lr:.6f} (plateau scale {bleu_state['lr_scale']:.3f})")

        if is_master(rank):
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

        # =======================
        # BLEU probe (rank 0 only, result broadcast to all ranks)
        # =======================
        bleu = None
        bleu_ok = torch.ones(1, dtype=torch.uint8, device=device)
        if is_master(rank) and epoch % cfg.bleu_interval == 0:
            try:
                t0 = time.time()
                bleu = _run_bleu_probe(model, bleu_loader, cfg, sp, device)
                history["bleu_scores"].append((epoch, bleu))
                log_info(rank, f"[BLEU] Epoch {epoch}: {bleu:.2f} ({time.time() - t0:.1f}s)")
            except Exception as e:
                bleu_ok[0] = 0
                log_info(rank, f"BLEU evaluation failed: {e}")

        dist.broadcast(bleu_ok, src=0)
        if not bool(bleu_ok.item()):
            log_info(rank, "Aborting: BLEU evaluation failed on rank 0")
            break

        barrier(device)

        # =======================
        # BLEU-driven control
        # =======================
        if is_master(rank) and bleu is not None:
            if update_bleu_state(bleu_state, bleu, cfg):
                # rotate the best-BLEU checkpoint (keep only the latest best)
                for old in glob.glob(str(ckpt_dir / "best_model_bleu_epoch_*.pt")):
                    try:
                        os.remove(old)
                        log_info(rank, f"Removed old best checkpoint: {old}")
                    except OSError as e:
                        log_info(rank, f"Failed to remove {old}: {e}")
                best_path = ckpt_dir / f"best_model_bleu_epoch_{epoch}.pt"
                save_checkpoint(
                    rank, model, scheduler, epoch, str(best_path),
                    bleu_state=bleu_state, history=history, cfg=cfg,
                    amp_scaler=scaler,
                )

        bleu_state = broadcast_bleu_state(bleu_state, device)
        # NoamOpt applies lr_scale inside rate(), so the plateau decay
        # survives subsequent scheduler.step() calls
        scheduler.set_lr_scale(bleu_state["lr_scale"])

        if is_master(rank):
            save_checkpoint(
                rank, model, scheduler, epoch, str(latest_path),
                bleu_state=bleu_state, history=history, cfg=cfg,
                amp_scaler=scaler,
            )
        barrier(device)

        # =======================
        # Early stop (all ranks exit together)
        # =======================
        stop_flag = torch.zeros(1, dtype=torch.uint8, device=device)
        if is_master(rank):
            stop_flag[0] = 1 if should_stop(bleu_state, cfg) else 0
        dist.broadcast(stop_flag, src=0)
        if bool(stop_flag.item()):
            log_info(rank, f"Early stopping at epoch {epoch}")
            break

    barrier(device)
    dist.destroy_process_group()

    if is_master(rank):
        cfg.save_json(str(ckpt_dir / "config.json"))
        plot_training_history(history, ckpt_dir)
        log_info(rank, "Training finished.")
