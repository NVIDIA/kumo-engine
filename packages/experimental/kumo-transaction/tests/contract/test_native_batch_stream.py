# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded native tensor delivery, ordering, prefetch and producer ownership."""

from threading import Event, get_ident

import pytest
import torch

from tfm.contracts.batch import TransactionBatch
from tfm.data.native_batches import NativeBatches


class Task:
    def __init__(self):
        self.calls = []
        self.closed = []

    def count(self, split):
        return 6

    def batches(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield from range(3)
        finally:
            self.closed.append(get_ident())


def adapt(value):
    return TransactionBatch(torch.full((2, 2), value), torch.ones((2, 2), dtype=torch.bool))


def test_one_continuous_native_iterator_per_epoch():
    task = Task()
    with NativeBatches(task, adapt, batch_size=2, seed=41) as batches:
        assert len(batches) == 3
        assert [b.input_ids[0, 0].item() for b in batches] == [0, 1, 2]
        assert [b.input_ids[0, 0].item() for b in batches] == [0, 1, 2]
    assert [c["epoch"] for c in task.calls] == [0, 1]
    assert all(c["shuffle"] and c["seed"] == 41 for c in task.calls)
    assert len(task.closed) == 2
    assert all(thread != get_ident() for thread in task.closed)


def test_next_cpu_batch_is_adapted_before_consumer_resumes():
    task, started, release = Task(), Event(), Event()

    def adapter(value):
        if value == 1:
            started.set()
            assert release.wait(3), "test did not release lookahead"
        return adapt(value)

    batches = NativeBatches(task, adapter, batch_size=2, seed=0)
    iterator = iter(batches)
    try:
        assert next(iterator).input_ids[0, 0] == 0
        assert started.wait(3), "lookahead did not start while consumer held the first batch"
        assert not task.closed
    finally:
        release.set()
        batches.close()
    assert len(task.closed) == 1
    with pytest.raises(ValueError, match="closed"):
        iter(batches)


def test_adapter_failure_closes_native_iterator_and_propagates():
    task = Task()

    def bad(value):
        raise RuntimeError("tensor adapter failed")

    with NativeBatches(task, bad, batch_size=2, seed=0) as batches:
        with pytest.raises(RuntimeError, match="tensor adapter failed"):
            next(iter(batches))
    assert len(task.closed) == 1


def test_rank_selection_and_validation_order_are_native():
    task = Task()
    with NativeBatches(task, adapt, batch_size=2, seed=9, shuffle=False) as batches:
        with batches.for_rank(1, 2) as ranked:
            assert len(ranked) == 2
            list(ranked)
            list(ranked)
    assert all(c["rank"] == 1 and c["world_size"] == 2 for c in task.calls)
    assert all(c["epoch"] == 0 and not c["shuffle"] for c in task.calls)


def test_active_sources_reject_concurrent_consumers():
    with NativeBatches(Task(), adapt, batch_size=2, seed=0) as batches:
        iterator = iter(batches)
        with pytest.raises(ValueError, match="active iterator"):
            iter(batches)
        next(iterator)
        iterator.close()
        assert len(list(batches)) == 3


def test_prefetch_uses_normal_no_grad_tensors_inside_inference_mode():
    def adapter(value):
        assert not torch.is_grad_enabled()
        batch = adapt(value)
        assert not batch.input_ids.is_inference()
        assert batch.input_ids._version == 0
        return batch

    with torch.inference_mode():
        with NativeBatches(Task(), adapter, batch_size=2, seed=0) as batches:
            assert len(list(batches)) == 3


def test_rank_length_accounts_for_native_padding():
    task = Task()
    task.count = lambda split: 5
    with NativeBatches(task, adapt, batch_size=2, seed=0) as batches:
        with batches.for_rank(1, 2) as ranked:
            assert ranked.owned_samples == 2
            assert ranked.samples == 3
            assert len(ranked) == 2
