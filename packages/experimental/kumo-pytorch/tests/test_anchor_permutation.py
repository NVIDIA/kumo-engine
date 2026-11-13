# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default anchor shuffle keeps its scalar sequence with bounded work."""

from itertools import islice

import pytest

from kumo.pytorch.anchors import _mix, _Permutation, shard_positions


@pytest.mark.parametrize("size", [*range(66), 127, 128, 129, 4095, 4096, 4097, 10003])
@pytest.mark.parametrize("seed", [0, -1, 2**64 + 17])
def test_vector_permutation_is_the_existing_scalar_permutation(size, seed):
    order = _Permutation(size, seed)
    expected = [order(i) for i in range(size)]
    for width in (17, 4096):
        actual = [
            int(value)
            for start in range(0, size, width)
            for value in order.many(range(start, min(size, start + width)))
        ]
        assert actual == expected
    assert sorted(expected) == list(range(size))


@pytest.mark.parametrize("size", [2**32 - 1, 2**32 + 1, 2**63, 2**64 - 1, 2**64, 2**65 + 1])
def test_large_domains_are_pointwise_and_do_not_allocate_an_epoch(size):
    positions = [0, 1, size // 2, size - 2, size - 1]
    order = _Permutation(size, 77)
    assert list(order.many(positions)) == [order(i) for i in positions]
    assert list(islice(shard_positions(size, seed=7, epoch=3, shuffle=True), 5)) == [
        _Permutation(size, _mix(7) ^ _mix(3))(i) for i in range(5)
    ]


@pytest.mark.parametrize("epoch,rank,world", [(0, 0, 1), (3, 2, 7), (9, 11, 20)])
def test_vector_shuffle_keeps_each_rank_sequence(epoch, rank, world):
    size = 8209
    order = _Permutation(size, _mix(31) ^ _mix(epoch))
    assert list(
        shard_positions(size, seed=31, epoch=epoch, shuffle=True, rank=rank, world_size=world)
    ) == [order(i) for i in range(rank, size, world)]


def test_first_outputs_only_evaluate_bounded_chunks(monkeypatch):
    evaluated = []
    original = _Permutation.many

    def bounded(self, positions):
        assert len(positions) <= 4096
        evaluated.append(len(positions))
        return original(self, positions)

    monkeypatch.setattr(_Permutation, "many", bounded)
    iterator = shard_positions(2**40, shuffle=True, rank=2, world_size=7)
    assert evaluated == []
    assert len(list(islice(iterator, 4100))) == 4100
    assert evaluated == [4096, 4096]


@pytest.mark.parametrize("from_table", [False, True])
@pytest.mark.parametrize("rank,world", [(0, 1), (2, 3), (10, 11)])
def test_task_delivery_keeps_scalar_order(from_table, rank, world):
    from kumo.pytorch.api import InMemoryEngine, prepare
    from tests.test_anchor_tables import DATASET, _contract, _requests_for, _scores, _table

    requests = _requests_for("train")
    seed, epoch = 31, 3
    scalar = _Permutation(len(requests), _mix(seed) ^ _mix(epoch))
    expected = [float(requests[scalar(i)].entity_id) for i in range(rank, len(requests), world)]
    with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
        with plan.materialize(_table(), fit_split="train") as task:
            selection = {"split": "train"} if from_table else {"requests": requests}
            delivered = task.batches(
                **selection,
                batch_size=3,
                seed=seed,
                epoch=epoch,
                shuffle=True,
                rank=rank,
                world_size=world,
            )
            assert _scores(delivered) == expected
