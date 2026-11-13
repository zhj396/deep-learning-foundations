"""DynamicBatcher contract: merging, worker error handling, timeouts, collation."""

import time

import pytest

from batcher import DynamicBatcher, RequestItem


def test_merges_items_into_few_calls():
    calls = []

    def fake(arr):
        calls.append(arr.shape[0])
        return [[1] for _ in range(arr.shape[0])]

    b = DynamicBatcher(fake, pad_idx=0, max_batch_size=4, poll_interval_s=0.05)
    items = [RequestItem([1, 2, 3]) for _ in range(10)]
    b.submit(items)
    for it in items:
        b.wait(it)

    assert sum(calls) == 10            # every item processed exactly once
    assert all(n <= 4 for n in calls)  # respects max_batch_size
    assert len(calls) <= 3             # merged: 10 items need >= 3 calls, not 10


def test_worker_survives_predict_failure():
    state = {"fail": True}

    def flaky(arr):
        if state["fail"]:
            state["fail"] = False
            raise RuntimeError("boom")
        return [[7] for _ in range(arr.shape[0])]

    b = DynamicBatcher(flaky, pad_idx=0, poll_interval_s=0.01)

    bad = RequestItem([1])
    b.submit([bad])
    with pytest.raises(RuntimeError, match="boom"):
        b.wait(bad)

    # the worker thread must still be alive and serving
    assert b.alive
    good = RequestItem([2])
    b.submit([good])
    assert b.wait(good) == [7]


def test_wait_timeout_raises():
    def slow(arr):
        time.sleep(1.0)
        return [[0] for _ in range(arr.shape[0])]

    b = DynamicBatcher(slow, pad_idx=0, poll_interval_s=0.01, wait_timeout_s=0.05)
    item = RequestItem([1])
    b.submit([item])
    with pytest.raises(TimeoutError):
        b.wait(item)


def test_collate_pads_and_preserves_order():
    b = DynamicBatcher(lambda arr: arr.tolist(), pad_idx=9, poll_interval_s=0.01)
    items = [RequestItem([1, 2]), RequestItem([3]), RequestItem([4, 5, 6])]
    b.submit(items)

    outs = [b.wait(it) for it in items]
    assert outs[0] == [1, 2, 9]
    assert outs[1] == [3, 9, 9]
    assert outs[2] == [4, 5, 6]
