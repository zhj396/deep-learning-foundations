# -*- coding: utf-8 -*-
"""Dynamic batching queue for the translation service.

Merges sentences from concurrent requests (and multi-sentence requests) into
single backend decode calls -- autoregressive decoding amortises especially
well over batches, since every sequence in the batch shares each decoder
step. Backend-agnostic: wraps any translate_fn(src_ids (B,S) int64) ->
list[list[int]].

Robustness contract:
- the worker thread never dies: decode exceptions are attached to the
  affected items (item.error) and their events are always set, so waiting
  clients get an immediate error instead of hanging forever
- waiters use a timeout as a second safety net (generous by default:
  CPU beam search over long batches can be slow)
- submit() takes a whole list, so one request with N sentences becomes as
  few backend calls as the batch size allows
"""

import logging
import threading
import time
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger("inference")


class RequestItem:
    """One sentence waiting for (or carrying) its translation."""

    __slots__ = ("src_ids", "text", "result", "error", "event")

    def __init__(self, src_ids: List[int], text: str = ""):
        self.src_ids = src_ids
        self.text = text
        self.result: Optional[List[int]] = None
        self.error: Optional[Exception] = None
        self.event = threading.Event()


class DynamicBatcher:
    """Batches RequestItems from concurrent clients into single decode calls."""

    def __init__(
        self,
        translate_fn: Callable[[np.ndarray], List[List[int]]],
        pad_idx: int = 0,
        max_batch_size: int = 32,
        poll_interval_s: float = 0.005,
        wait_timeout_s: float = 120.0,
    ):
        self.translate_fn = translate_fn
        self.pad_idx = pad_idx
        self.max_batch_size = max_batch_size
        self.poll_interval_s = poll_interval_s
        self.wait_timeout_s = wait_timeout_s
        self._queue: List[RequestItem] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="translate-batcher"
        )
        self._thread.start()

    def submit(self, items: List[RequestItem]) -> None:
        """Enqueue all items of a request at once."""
        with self._lock:
            self._queue.extend(items)

    def wait(self, item: RequestItem) -> List[int]:
        """Block until the item's translation is ready; raise on error/timeout."""
        if not item.event.wait(self.wait_timeout_s):
            raise TimeoutError(
                f"Translation timed out after {self.wait_timeout_s}s: {item.text[:60]!r}"
            )
        if item.error is not None:
            raise item.error
        return item.result

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _collate(self, batch: List[RequestItem]) -> np.ndarray:
        """Pad token-id lists to a rectangular (B, S) int64 array."""
        width = max(len(it.src_ids) for it in batch)
        arr = np.full((len(batch), width), self.pad_idx, dtype=np.int64)
        for i, it in enumerate(batch):
            arr[i, : len(it.src_ids)] = it.src_ids
        return arr

    def _worker(self) -> None:
        while True:
            # short sleep lets items from concurrent requests accumulate
            time.sleep(self.poll_interval_s)
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    batch = self._queue[: self.max_batch_size]
                    self._queue = self._queue[self.max_batch_size:]

                try:
                    preds = self.translate_fn(self._collate(batch))
                except Exception as e:  # keep the worker alive no matter what
                    logger.exception(f"Batch decode failed for {len(batch)} item(s)")
                    for item in batch:
                        item.error = RuntimeError(f"Batch decode failed: {e}")
                        item.event.set()
                    continue

                for item, pred in zip(batch, preds):
                    item.result = list(pred)
                    item.event.set()
