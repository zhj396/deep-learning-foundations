"""Training orchestration: single-process (CPU / 1 GPU) or multi-GPU DDP.

Launched from main.py as train(rank, world_size, cfg). Rank-aware logging
goes through utils.is_master / utils.log_info; validation metrics are
all-reduced inside evaluate_epoch, the BLEU probe is decoded shard-by-shard
by every rank over the split named by cfg.bleu_split and scored on rank 0,
and controller state is broadcast after every probe, so all ranks stay in
lockstep through early stopping. Selection (best checkpoint, LS/LR decay,
early stop) reads the validation probe; the test split is decoded once after
training, on the checkpoint the probe picked, and only reported.
"""

import contextlib
import datetime
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
from checkpoint import (
    default_history,
    load_checkpoint,
    load_weights_only,
    save_checkpoint,
)
from config import Config
from dataset import Collator, TranslationDataset
from decoding import beam_decode_batch, greedy_decode_batch
from evaluate import compute_bleu, evaluate_epoch
from loss import LabelSmoothingLoss, accuracy
from model import Transformer
from scheduler import NoamOpt
from utils import (
    all_ranks_ok,
    barrier,
    gather_by_rank,
    is_master,
    log_info,
    plot_training_history,
    probe_shard_indices,
    set_seed,
)

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
# Sharded BLEU decoding (probe + the final test pass)
# -------------------------
def _sharded_loader(dataset, rank, world_size, batch_size, collator, n_take) -> DataLoader:
    """Loader over the first n_take rows, this rank's strided share of them.

    No sampler and no shuffling: the slice must be identical across epochs for
    the control signal to be comparable, and identical across runs.
    """
    return DataLoader(
        Subset(dataset, probe_shard_indices(rank, world_size, n_take)),
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )


def _decode_shard(
    model, loader, cfg, device: torch.device, decode_len: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """Decode this rank's shard -> (pred_id rows, ref_id rows).

    Both decode passes share this function so the served/selected model and
    the reported number go through the same code. Decoding has no KV cache
    (see decoding.py), so cost per sentence grows with the square of the
    generated length: the control probe caps that with bleu_max_len, the final
    test pass does not. Both run under the same autocast as training, which on
    a sentence count this size is the difference between seconds and minutes.
    Raises on failure; the caller votes on it across ranks.
    """
    model.eval()
    all_preds: List[List[int]] = []
    all_refs: List[List[int]] = []

    common = dict(
        max_len=decode_len,
        bos_idx=cfg.bos_idx,
        eos_idx=cfg.eos_idx,
        pad_idx=cfg.pad_idx,
        device=device,
    )
    use_amp = cfg.use_amp and device.type == "cuda"

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if cfg.use_beam:
                preds = beam_decode_batch(
                    model, src, beam_size=cfg.beam_size,
                    length_penalty=cfg.length_penalty, **common,
                )
            else:
                preds = greedy_decode_batch(model, src, **common)

        all_preds.extend(preds.cpu().tolist())
        all_refs.extend(tgt[:, 1:].cpu().tolist())

    return all_preds, all_refs


def _sharded_bleu(
    model, loader, cfg, sp, device: torch.device, rank: int, decode_len: int
) -> Tuple[Optional[float], bool]:
    """Corpus BLEU over a loader sharded across ranks -> (bleu, ok).

    One collective sequence for both decode passes, identical on every rank:
    decode the local shard, MIN-vote any decode failure, gather the id rows,
    score on rank 0, broadcast the scorer's status. Ranks therefore never
    diverge on a failure -- either everyone continues or everyone stops.
    `bleu` is only non-None on rank 0 (that is where history lives).
    """
    decode_ok = torch.ones(1, dtype=torch.uint8, device=device)
    try:
        preds, refs = _decode_shard(model, loader, cfg, device, decode_len)
    except Exception as e:
        decode_ok[0] = 0
        preds, refs = [], []
        log_info(rank, f"BLEU decoding failed: {e}")

    if not all_ranks_ok(decode_ok):
        return None, False

    shards = gather_by_rank((preds, refs))

    score_ok = torch.ones(1, dtype=torch.uint8, device=device)
    bleu = None
    if is_master(rank):
        try:
            bleu = compute_bleu(
                [row for part in shards for row in part[0]],
                [row for part in shards for row in part[1]],
                pad_idx=cfg.pad_idx,
                eos_idx=cfg.eos_idx,
                sp=sp,
                print_samples=cfg.print_samples,
            )
        except Exception as e:
            score_ok[0] = 0
            log_info(rank, f"BLEU evaluation failed: {e}")

    dist.broadcast(score_ok, src=0)
    return bleu, bool(score_ok.item())


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

    # Control probe: a fixed slice of the split named by cfg.bleu_split, handed
    # to the ranks in strided shards so the decode is the same total work every
    # epoch and no rank decodes alone while the others wait on it. Sharding by
    # stride (not by block) keeps the shards balanced, and corpus BLEU does not
    # depend on hypothesis order, so the union scores exactly these n_probe
    # sentences. A prefix is already an unbiased sample: data_prep shuffles
    # before it slices the splits.
    probe_source = {"valid": val_dataset, "test": test_dataset}[cfg.bleu_split]
    n_probe = min(cfg.bleu_batches * cfg.batch_size, len(probe_source))
    bleu_loader = _sharded_loader(
        probe_source, rank, world_size, cfg.batch_size, collator, n_probe)
    log_info(
        rank,
        f"BLEU probe: {n_probe} of {len(probe_source)} {cfg.bleu_split} sentences "
        f"(selection split; test is decoded once, after training)",
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
        # BLEU probe (selection split, shards decoded in parallel, rank 0 scores)
        # =======================
        bleu = None
        if epoch % cfg.bleu_interval == 0:
            t0 = time.time()
            bleu, ok = _sharded_bleu(
                model, bleu_loader, cfg, sp, device, rank,
                decode_len=min(cfg.bleu_max_len, cfg.max_len),
            )
            if not ok:
                log_info(rank, "Aborting: BLEU probe failed on some rank")
                break
            if bleu is not None:
                history["bleu_scores"].append((epoch, bleu))
                log_info(rank, f"[BLEU] Epoch {epoch}: {bleu:.2f} ({time.time() - t0:.1f}s)")

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

    # =======================
    # Final test evaluation - reporting only, never part of selection
    # =======================
    # The probe above decides checkpoints, label smoothing and early stopping
    # on the validation split; this is the single pass over the test split that
    # produces the number the README quotes, so the reported score is not the
    # maximum of a search. It evaluates the checkpoint the probe picked, not
    # whatever the last epoch happened to leave in memory.
    if cfg.final_test:
        barrier(device)  # every rank globs the same dir; see the same finished writes
        best_files = sorted(glob.glob(str(ckpt_dir / "best_model_bleu_epoch_*.pt")))
        eval_weights = best_files[-1] if best_files else None
        if eval_weights is not None:
            log_info(rank, f"Final test pass evaluates {Path(eval_weights).name} "
                           f"(the checkpoint the probe selected)")
            load_weights_only(eval_weights, model, device, rank, unsafe=cfg.trust_checkpoint)
        else:
            log_info(rank, "Final test pass evaluates the weights as they stand "
                           "(no best-probe checkpoint was written)")

        n_test = len(test_dataset)
        t0 = time.time()
        test_loader = _sharded_loader(
            test_dataset, rank, world_size, cfg.batch_size, collator, n_test)
        # no bleu_max_len here: a reported number decodes as far as the model may go
        test_bleu, ok = _sharded_bleu(
            model, test_loader, cfg, sp, device, rank, decode_len=cfg.max_len)

        if not ok:
            log_info(rank, "Final test evaluation failed (the training run itself completed)")
        elif is_master(rank):
            record = {
                "bleu": test_bleu,
                "sentences": n_test,
                "weights": eval_weights,
                "decode": "beam" if cfg.use_beam else "greedy",
                "decode_max_len": cfg.max_len,
                "beam_size": cfg.beam_size if cfg.use_beam else None,
                "length_penalty": cfg.length_penalty if cfg.use_beam else None,
                "seconds": round(time.time() - t0, 1),
            }
            with open(ckpt_dir / "test_bleu.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=4)
            log_info(
                rank,
                f"[TEST] {n_test} sentences, {record['decode']} decoding to "
                f"{cfg.max_len} steps: BLEU {test_bleu:.2f} "
                f"({record['seconds']}s) -> {ckpt_dir / 'test_bleu.json'}",
            )

    barrier(device)
    dist.destroy_process_group()

    if is_master(rank):
        cfg.save_json(str(ckpt_dir / "config.json"))
        plot_training_history(history, ckpt_dir)
        log_info(rank, "Training finished.")
