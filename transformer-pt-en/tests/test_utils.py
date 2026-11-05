"""BLEU-probe helpers in utils: shard arithmetic and the collective wrappers.

The wrappers are checked twice: without a process group (the single-process
path must issue nothing) and against a stubbed one, which pins the rank
ordering of the gather and the MIN semantics of the failure vote. No real
ranks are spawned -- the suite stays CPU-only and process-group-free.
"""

import pytest
import torch

import utils


class FakePG:
    """Stand-in for an initialized process group, recording what was called."""

    def __init__(self, monkeypatch, world_size=2, remote_votes=None):
        self.world_size = world_size
        self.remote_votes = remote_votes or []
        self.collectives = []
        self.reduced_with = None

        dist = utils.dist
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda *a, **kw: world_size)
        monkeypatch.setattr(dist, "all_gather_object", self.all_gather_object)
        monkeypatch.setattr(dist, "all_reduce", self.all_reduce)

    def all_gather_object(self, out, payload):
        self.collectives.append("all_gather_object")
        for rank in range(self.world_size):
            out[rank] = (f"rank{rank}", payload)

    def all_reduce(self, tensor, op=None):
        self.collectives.append("all_reduce")
        self.reduced_with = op
        tensor[0] = min([int(tensor[0])] + list(self.remote_votes))


# -------------------------
# Shard arithmetic
# -------------------------
@pytest.mark.parametrize("n_total, world_size", [(10, 2), (3, 8)])
def test_shards_cover_the_slice_once(n_total, world_size):
    shards = [utils.probe_shard_indices(r, world_size, n_total) for r in range(world_size)]
    flat = [i for shard in shards for i in shard]

    assert sorted(flat) == list(range(n_total)), "every sentence exactly once"
    assert all(shard == sorted(shard) for shard in shards), "strided, ascending"


@pytest.mark.parametrize("n_total, world_size", [(640, 2), (7, 3), (3, 8)])
def test_shards_stay_balanced(n_total, world_size):
    sizes = [len(utils.probe_shard_indices(r, world_size, n_total)) for r in range(world_size)]
    assert max(sizes) - min(sizes) <= 1


def test_single_rank_gets_the_whole_slice():
    assert utils.probe_shard_indices(0, 1, 5) == [0, 1, 2, 3, 4]
    # a degenerate world_size still yields a usable slice rather than nothing
    assert utils.probe_shard_indices(0, 0, 4) == [0, 1, 2, 3]


# -------------------------
# Gather wrapper
# -------------------------
def test_gather_by_rank_without_process_group():
    payload = ([[1, 2]], [[3, 4]])
    assert utils.gather_by_rank(payload) == [payload]


def test_gather_by_rank_keeps_rank_order(monkeypatch):
    pg = FakePG(monkeypatch, world_size=2)
    payload = ([[7], [8]], [[9], [10]])

    gathered = utils.gather_by_rank(payload)

    assert gathered == [("rank0", payload), ("rank1", payload)]
    assert pg.collectives == ["all_gather_object"]


def test_gather_by_rank_single_rank_decodes_alone(monkeypatch):
    pg = FakePG(monkeypatch, world_size=1)
    assert utils.gather_by_rank([1, 2]) == [[1, 2]]
    assert pg.collectives == []


# -------------------------
# Failure vote
# -------------------------
def test_all_ranks_ok_uses_a_min_reduce(monkeypatch):
    pg = FakePG(monkeypatch, world_size=2, remote_votes=[0])
    flag = torch.ones(1, dtype=torch.uint8)

    assert utils.all_ranks_ok(flag) is False
    assert pg.collectives == ["all_reduce"]
    assert pg.reduced_with is torch.distributed.ReduceOp.MIN


def test_all_ranks_ok_true_when_every_vote_survives(monkeypatch):
    pg = FakePG(monkeypatch, world_size=3, remote_votes=[1, 1])

    assert utils.all_ranks_ok(torch.ones(1, dtype=torch.uint8)) is True
    assert pg.collectives == ["all_reduce"]


def test_all_ranks_ok_without_process_group():
    assert utils.all_ranks_ok(torch.ones(1, dtype=torch.uint8)) is True
    assert utils.all_ranks_ok(torch.zeros(1, dtype=torch.uint8)) is False
