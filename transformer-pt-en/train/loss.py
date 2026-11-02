"""Label-smoothing cross-entropy (KL form) and token-level accuracy."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Label smoothing as in the paper (epsilon_ls = 0.1), in KL-div form.

    PAD positions contribute zero loss; the mean is taken over non-pad
    positions only.
    """

    def __init__(self, smoothing: float, vocab_size: int, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        x: (N, vocab_size) logits (NOT log-probs)
        target: (N,) token ids
        """
        log_probs = F.log_softmax(x, dim=-1)

        with torch.no_grad():
            # smooth distribution: confidence on the true class, the rest
            # spread uniformly over the other vocab entries
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))

            pad_mask = target == self.pad_idx
            valid_target = target.clone()
            valid_target[pad_mask] = 0  # legal index; the row is zeroed below

            true_dist.scatter_(1, valid_target.view(-1, 1), self.confidence)
            # zero out pad rows so they contribute nothing to the KL sum
            true_dist[pad_mask] = 0.0

        loss = F.kl_div(log_probs, true_dist, reduction="none").sum(dim=1)
        loss = loss.masked_fill(pad_mask, 0.0)

        non_pad = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / non_pad


def accuracy(logits: torch.Tensor, target: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Token-level accuracy over non-pad positions.

    logits: (B, T, V), target: (B, T). Returns a scalar tensor.
    """
    preds = logits.argmax(dim=-1)
    mask = target != pad_idx
    correct = (preds == target) & mask
    denom = mask.sum().clamp(min=1.0).float()
    return correct.sum().float() / denom
