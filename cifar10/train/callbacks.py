# -*- coding: utf-8 -*-
"""
Training callbacks:
1. TrainingMonitor: per-epoch metrics + timing through the project logger
2. ModelCheckpoint: saves the best model (monitors val_accuracy)
3. EarlyStopping: stops when validation accuracy plateaus (monitors val_accuracy)
"""

import time
from typing import Dict, List

import numpy as np
from tensorflow.keras import callbacks

from config import Config


class TrainingMonitor(callbacks.Callback):
    """Logs per-epoch metrics and timing through the project logger.

    model.fit() should run with verbose=0 so this callback is the single
    source of per-epoch output (console + log file).
    """

    def __init__(self, logger):
        super().__init__()  # required: Keras Callback base sets internal state
        self.logger = logger
        self.start_time: float = 0.0
        self.epoch_start: float = 0.0
        self.epoch_times: List[float] = []

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        self.logger.info("Training started...")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        self.epoch_times.append(duration)
        logs = logs or {}

        lr = logs.get("learning_rate")  # Keras 3 exposes the LR under this key
        lr_str = f", lr={lr:.2e}" if lr is not None else ""
        self.logger.info(
            f"Epoch {epoch + 1}/{self.params['epochs']} - {duration:.1f}s | "
            f"loss={logs.get('loss', float('nan')):.4f}, "
            f"acc={logs.get('accuracy', float('nan')):.4f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.4f}, "
            f"val_acc={logs.get('val_accuracy', float('nan')):.4f}{lr_str}"
        )

        self._check_overfitting(epoch, logs)

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        avg_time = np.mean(self.epoch_times) if self.epoch_times else 0.0
        self.logger.info(
            f"Training completed! Total time: {total_time:.1f}s, "
            f"average epoch: {avg_time:.1f}s"
        )

    def _check_overfitting(self, epoch: int, logs: Dict[str, float]) -> None:
        """Warn when validation loss drifts far above training loss."""
        if epoch < 5:  # give the model a few epochs to warm up
            return

        val_loss = logs.get("val_loss")
        train_loss = logs.get("loss")
        if val_loss and train_loss and val_loss > train_loss * 1.5:
            self.logger.warning(
                "Possible overfitting detected (val_loss > 1.5 * train_loss)"
            )


def create_training_callbacks(cfg: Config, best_model_path: str) -> List[callbacks.Callback]:
    """Create the callback list for one training phase.

    Called separately for each phase (initial training, fine-tuning) so each
    phase starts with fresh EarlyStopping/ModelCheckpoint state.
    """
    callbacks_list: List[callbacks.Callback] = [
        TrainingMonitor(logger=cfg.logger),
        callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=0,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            min_delta=0.0005,
            patience=cfg.patience + 3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    cfg.logger.info(f"Created {len(callbacks_list)} training callbacks")
    return callbacks_list
