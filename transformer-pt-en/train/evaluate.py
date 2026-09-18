"""Epoch-level evaluation (loss/accuracy) and corpus BLEU via sacrebleu."""

from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import accuracy


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = 0,
    use_amp: bool = False,
    is_master: bool = True,
) -> Tuple[float, float]:
    """Token-weighted (avg_loss, avg_accuracy), globally reduced under DDP.

    The returned values are already averaged across ALL ranks -- callers
    must not reduce them a second time (that would double-count the
    collective).
    """
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_acc = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)

    autocast_ctx = torch.amp.autocast(
        "cuda", enabled=use_amp and device.type == "cuda"
    )

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False, disable=not is_master):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(src, tgt[:, :-1])  # (B, T-1, V)
                target_tokens = tgt[:, 1:].reshape(-1)
                logits_flat = logits.reshape(-1, logits.size(-1))
                # criterion already averages over non-pad tokens; weight by
                # this batch's valid-token count for exact corpus-level means
                loss = criterion(logits_flat, target_tokens)

            n_valid = (target_tokens != pad_idx).sum()
            total_loss += loss.float() * n_valid
            total_acc += accuracy(logits, tgt[:, 1:], pad_idx=pad_idx).float() * n_valid
            total_tokens += n_valid

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    denom = total_tokens.clamp(min=1.0)
    return (total_loss / denom).item(), (total_acc / denom).item()


def compute_bleu(
    preds: List[List[int]],
    refs: List[List[int]],
    pad_idx: int,
    eos_idx: int,
    sp,
    print_samples: int = 5,
) -> float:
    """Corpus BLEU (sacrebleu) over decoded id sequences.

    preds/refs are cut at the first EOS; special tokens are dropped before
    detokenization. `sp` is a SentencePiece-like processor with .decode().
    """

    def strip(seq: List[int]) -> List[int]:
        out = []
        for t in seq:
            if t == eos_idx:
                break
            if t != pad_idx:
                out.append(t)
        return out

    preds_clean = [strip(p) for p in preds]
    refs_clean = [strip(r) for r in refs]

    preds_text = [sp.decode(p) for p in preds_clean]
    refs_text = [sp.decode(r) for r in refs_clean]

    n = min(print_samples, len(preds_text))
    if n > 0:
        print(f"===== first {n} samples =====")
        for i in range(n):
            print(f"sample {i + 1}:")
            print(f"  pred: {preds_text[i]}")
            print(f"  ref:  {refs_text[i]}")
        print()

    import sacrebleu

    bleu = sacrebleu.corpus_bleu(preds_text, [[r] for r in refs_text])
    return bleu.score
