# -*- coding: utf-8 -*-
"""
Learning Rate Scheduler Module:
1. Cosine Decay with Restarts (schedule attached to the optimizer)
2. ReduceLROnPlateau (returned as an extra callback)
"""

from typing import List, Tuple

import tensorflow as tf

from config import Config


def create_optimizer_and_scheduler(
    cfg: Config,
    steps_per_epoch: int,
) -> Tuple[tf.keras.optimizers.Optimizer, List[tf.keras.callbacks.Callback]]:
    """Create the optimizer plus any LR-scheduling callbacks.

    Returns a tuple instead of mutating a caller-owned callback list, so the
    data flow is explicit and the function has no side effects.

    Args:
        cfg: Global configuration object (uses lr_mode, init_lr, min_lr,
            factor, patience, cosine_restart_epochs).
        steps_per_epoch: Number of optimizer steps per epoch (used to express
            the cosine restart period in steps).

    Returns:
        (optimizer, extra_callbacks): extra_callbacks is empty for cosine mode
        and contains ReduceLROnPlateau for plateau mode.
    """
    mode = cfg.lr_mode.lower()

    if mode == "cosine":
        schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=cfg.init_lr,
            first_decay_steps=steps_per_epoch * cfg.cosine_restart_epochs,
            t_mul=2.0,                                   # each cycle is 2x longer
            m_mul=0.9,                                   # peak LR decays 10% per cycle
            alpha=cfg.min_lr / cfg.init_lr,              # floor as a fraction of init LR
        )
        cfg.logger.info(
            f"LR schedule: cosine decay with restarts every "
            f"{cfg.cosine_restart_epochs} epochs (x2 cycle growth)"
        )
        return tf.keras.optimizers.Adam(learning_rate=schedule), []

    if mode == "plateau":
        plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",   # loss reacts earlier than accuracy when stalled
            factor=cfg.factor,
            patience=cfg.patience,
            verbose=1,
            min_lr=cfg.min_lr,
        )
        cfg.logger.info(
            f"LR schedule: ReduceLROnPlateau (factor={cfg.factor}, "
            f"patience={cfg.patience}, min_lr={cfg.min_lr})"
        )
        return tf.keras.optimizers.Adam(learning_rate=cfg.init_lr), [plateau_cb]

    raise ValueError(f"Unsupported lr_mode: {mode}. Supported: ['cosine', 'plateau']")
