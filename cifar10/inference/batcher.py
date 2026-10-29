# -*- coding: utf-8 -*-
"""
Dynamic batching queue for the inference service.

Merges items from concurrent requests (and multi-image requests) into single
backend calls. Backend-agnostic: wraps any predict_fn(batch) -> probabilities.

Robustness contract:
- The worker thread never dies: inference exceptions are attached to the
  affected items (item.error) and their events are always set, so waiting
  clients get an immediate error instead of hanging forever.
- waiters use a timeout as a second safety net.
- submit() takes a whole list so one request with N images becomes as few
  backend calls as possible (true intra-request batching).
"""

import logging
import threading
import time
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger("inference")


class RequestItem:
    """One image waiting for (or carrying) its prediction."""

    __slots__ = ("img", "filename", "result", "error", "event")

    def __init__(self, img: np.ndarray, filename: str):
        self.img = img
        self.filename = filename
        self.result: Optional[np.ndarray] = None
        self.error: Optional[Exception] = None
        self.event = threading.Event()


class DynamicBatcher:
    """Batches RequestItems from concurrent clients into single predict calls."""

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray],
                 max_batch_size: int = 32,
                 poll_interval_s: float = 0.005,
                 wait_timeout_s: float = 30.0):
        self.predict_fn = predict_fn
        self.max_batch_size = max_batch_size
        self.poll_interval_s = poll_interval_s
        self.wait_timeout_s = wait_timeout_s
        self._queue: List[RequestItem] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="dynamic-batcher")
        self._thread.start()

    def submit(self, items: List[RequestItem]) -> None:
        """Enqueue all items of a request at once."""
        with self._lock:
            self._queue.extend(items)

    def wait(self, item: RequestItem) -> np.ndarray:
        """Block until the item's prediction is ready; raise on error/timeout."""
        if not item.event.wait(self.wait_timeout_s):
            raise TimeoutError(
                f"Prediction for {item.filename} timed out after {self.wait_timeout_s}s"
            )
        if item.error is not None:
            raise item.error
        return item.result

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _worker(self) -> None:
        while True:
            # Short sleep lets items from concurrent requests accumulate
            time.sleep(self.poll_interval_s)
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    batch = self._queue[: self.max_batch_size]
                    self._queue = self._queue[self.max_batch_size:]

                try:
                    imgs = np.vstack([item.img for item in batch])
                    preds = self.predict_fn(imgs)
                except Exception as e:  # keep the worker alive no matter what
                    logger.exception(f"Batch inference failed for {len(batch)} item(s)")
                    for item in batch:
                        item.error = RuntimeError(f"Batch inference failed: {e}")
                        item.event.set()
                    continue

                for item, pred in zip(batch, preds):
                    item.result = pred
                    item.event.set()
