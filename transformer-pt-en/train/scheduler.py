"""Noam learning-rate schedule with an integrated plateau scale.

lr(step) = lr_scale * factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

lr_scale lives INSIDE rate(): plateau decay applied by overwriting the
optimizer param-group lr after the fact would be erased by the next
step() call. Scaling must be part of the schedule itself.
"""

from typing import Optional

import torch


class NoamOpt:
    """Wraps an optimizer with the paper's warmup-then-decay schedule."""

    def __init__(
        self,
        model_size: int,
        factor: float,
        warmup: int,
        optimizer: torch.optim.Optimizer,
        lr_scale: float = 1.0,
    ):
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.lr_scale = lr_scale
        self._step = 0
        self._lr = 0.0

    def step(self) -> None:
        """Advance one optimizer step and refresh the learning rate."""
        self._step += 1
        self._lr = self.rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._lr

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def rate(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step
        if step <= 0:
            return 0.0
        return (
            self.lr_scale
            * self.factor
            * (self.model_size ** -0.5)
            * min(step ** -0.5, step * (self.warmup ** -1.5))
        )

    def set_lr_scale(self, lr_scale: float) -> None:
        """Apply a new plateau scale now AND to all subsequent steps."""
        self.lr_scale = lr_scale
        self._lr = self.rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._lr

    def get_lr(self) -> float:
        return self._lr

    def get_step(self) -> int:
        return self._step

    def state_dict(self) -> dict:
        """Schedule state only -- the optimizer has its own state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "lr_scale": self.lr_scale,
            "_lr": self._lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._step = state_dict.get("_step", 0)
        self.warmup = state_dict.get("warmup", self.warmup)
        self.factor = state_dict.get("factor", self.factor)
        self.model_size = state_dict.get("model_size", self.model_size)
        self.lr_scale = state_dict.get("lr_scale", self.lr_scale)
        # recompute lr from the restored step count (single source of truth)
        self._lr = self.rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._lr
