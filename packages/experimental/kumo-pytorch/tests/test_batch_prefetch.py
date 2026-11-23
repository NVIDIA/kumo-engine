# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefetch changes execution placement, not batch semantics or resume positions."""

import contextvars
import gc
import threading
import weakref

import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.task import TaskBatches
from tests.test_anchor_tables import DATASET, _contract, _table
from tests.test_preview import _equal


def batches(produce, depth=1):
    return TaskBatches(produce, task_fingerprint="task", config={}, prefetch_batches=depth)


def test_default_is_lazy_and_stays_on_calling_thread():
    calls = []

    def produce():
        calls.append(threading.get_ident())
        yield 1

    iterator = batches(produce, depth=0)
    assert calls == []
    assert next(iterator) == 1
    assert calls == [threading.get_ident()]
    iterator.close()


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_buffer_bounds_production_and_close_unblocks_a_full_buffer(depth):
    ready, overflow, closed = threading.Event(), threading.Event(), threading.Event()
    produced = []
    owner = []

    def produce():
        owner.append(threading.get_ident())
        try:
            for i in range(100):
                produced.append(i)
                if len(produced) == depth + 1:
                    ready.set()
                if len(produced) > depth + 1:
                    overflow.set()
                yield i
        finally:
            assert threading.get_ident() == owner[0]
            closed.set()

    iterator = batches(produce, depth)
    try:
        assert next(iterator) == 0
        assert ready.wait(3)
        assert not overflow.wait(0.03)
        assert iterator.state_dict()["batches"] == 1
    finally:
        iterator.close()
    assert closed.is_set()
    assert owner[0] != threading.get_ident()
    assert list(iterator) == []
    iterator.close()


@pytest.mark.parametrize("save_depth,resume_depth", [(0, 1), (1, 4), (4, 0)])
def test_resume_tracks_consumption_and_can_change_prefetch_depth(save_depth, resume_depth):
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        with plan.materialize(_table(), fit_split="train") as task:
            config = dict(batch_size=2, shuffle=True, seed=7, epoch=2)
            expected = list(task.batches(prefetch_batches=0, **config))
            with task.batches(prefetch_batches=save_depth, **config) as iterator:
                next(iterator)
                next(iterator)
                state = iterator.state_dict()
            assert state["batches"] == 2
            with task.batches(prefetch_batches=resume_depth, **config) as resumed:
                resumed.load_state_dict(state)
                actual = list(resumed)
                assert resumed.state_dict()["batches"] == len(expected)
            for old, new in zip(expected[2:], actual, strict=True):
                _equal(old.inputs, new.inputs)
                _equal(old.targets, new.targets)
                _equal(old.metadata, new.metadata)


@pytest.mark.parametrize("depth", [0, 3])
@pytest.mark.parametrize("error", [RuntimeError("producer failed"), KeyboardInterrupt()])
def test_producer_errors_follow_queued_batches_and_close_the_iterator(error, depth):
    closed = threading.Event()

    def produce():
        try:
            yield 0
            yield 1
            raise error
        finally:
            closed.set()

    with batches(produce, depth) as iterator:
        assert next(iterator) == 0
        assert next(iterator) == 1
        with pytest.raises(type(error)):
            next(iterator)
        assert list(iterator) == []
        assert iterator.state_dict()["batches"] == 2
        assert closed.is_set()
        with pytest.raises(ValueError, match="before iteration starts"):
            iterator.load_state_dict(iterator.state_dict())


def test_empty_source_and_factory_failure():
    assert list(batches(lambda: iter(()))) == []

    def fail():
        raise ValueError("factory failed")

    with batches(fail) as iterator, pytest.raises(ValueError, match="factory failed"):
        next(iterator)


def test_abandoning_iterator_signals_without_waiting(monkeypatch):
    preparing, release = threading.Event(), threading.Event()

    def produce(*args, **kwargs):
        yield 1
        preparing.set()
        assert release.wait(5)

    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        with plan.materialize(_table(), fit_split="train") as task:
            monkeypatch.setattr(task, "_anchor_batches", produce)
            iterator = task.batches(batch_size=1, prefetch_batches=1)
            assert next(iterator) == 1
            thread = iterator._iterator._thread
            try:
                assert preparing.wait(3)
                reference = weakref.ref(iterator)
                del iterator
                gc.collect()
                assert reference() is None
                assert thread.is_alive()
            finally:
                release.set()
                thread.join(timeout=3)
            assert not thread.is_alive()
            assert not task._prefetch_group._active


@pytest.mark.parametrize("inference", [False, True])
def test_producer_copies_contextvars_without_inheriting_torch_modes(inference):
    trace = contextvars.ContextVar("trace", default="unset")
    trace.set("caller")

    def produce():
        yield (
            torch.is_inference_mode_enabled(),
            torch.is_grad_enabled(),
            torch.is_autocast_enabled("cpu"),
            trace.get(),
        )

    iterator = batches(produce)
    with torch.inference_mode(inference), torch.no_grad(), torch.autocast("cpu"):
        assert next(iterator) == (False, True, False, "caller")
        assert torch.is_inference_mode_enabled() == inference
        assert not torch.is_grad_enabled()
        assert torch.is_autocast_enabled("cpu")
    iterator.close()
    assert torch.is_grad_enabled()
    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("depth", [0, 1, 4])
def test_inference_warmup_does_not_make_later_batches_inference_tensors(depth):
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        with plan.materialize(_table(), fit_split="train") as task:
            with task.batches(batch_size=1, prefetch_batches=depth) as iterator:
                with torch.inference_mode():
                    next(iterator)
                for batch in iterator:
                    values = batch.inputs["entity"]["score"]["values"]
                    assert not values.is_inference()
                    values.add_(1)


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_invalid_prefetch_depth_is_refused_before_work(value):
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        with plan.materialize(_table(), fit_split="train") as task:
            with pytest.raises(ValueError, match="prefetch_batches"):
                task.batches(prefetch_batches=value)


def test_task_close_stops_prefetch_before_closing_anchors():
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        task = plan.materialize(_table(), fit_split="train")
        iterator = task.batches(batch_size=1, prefetch_batches=1)
        next(iterator)
        task.close()
        assert list(iterator) == []


@pytest.mark.parametrize("owner", ["batches", "task", "plan"])
def test_interrupted_close_waits_for_producer_before_releasing_resources(monkeypatch, owner):
    plan = api.prepare(_contract(), DATASET, api.InMemoryEngine())
    task = plan.materialize(_table(), fit_split="train")
    preparing, release, finished = threading.Event(), threading.Event(), threading.Event()
    events = []
    produce = task._anchor_batches

    def blocked_produce(*args, **kwargs):
        producer = produce(*args, **kwargs)
        try:
            yield next(producer)
            preparing.set()
            assert release.wait(5), "producer was not released"
            yield next(producer)
        finally:
            producer.close()
            events.append("producer")
            finished.set()

    monkeypatch.setattr(task, "_anchor_batches", blocked_produce)
    iterator = task.batches(batch_size=1, prefetch_batches=1)
    owners = {"batches": iterator, "task": task, "plan": plan}
    try:
        next(iterator)
        assert preparing.wait(3)
        producer = iterator._iterator
        resource = plan._source if owner == "plan" else task._anchors
        close_resource = resource.close

        def checked_close():
            assert finished.is_set()
            events.append("resource")
            close_resource()

        monkeypatch.setattr(resource, "close", checked_close)

        def interrupt_wait(predicate):
            assert not predicate()
            raise KeyboardInterrupt("interrupted while waiting for cleanup")

        with monkeypatch.context() as interrupted:
            interrupted.setattr(producer._condition, "wait_for", interrupt_wait)
            with pytest.raises(KeyboardInterrupt, match="interrupted while waiting"):
                owners[owner].close()
        assert not events
        assert producer in task._prefetch_group._active
        release.set()
        owners[owner].close()
        assert finished.is_set()
        task.close()
        plan.close()
        assert events[:2] == ["producer", "resource"]
    finally:
        release.set()
        task.close()
        plan.close()


@pytest.mark.parametrize("launched", [False, True])
def test_interrupted_start_cannot_access_released_anchors(monkeypatch, launched):
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        task = plan.materialize(_table(), fit_split="train")
        iterator = task.batches(batch_size=1, prefetch_batches=1)
        opened = threading.Event()
        monkeypatch.setattr(task, "_anchor_batches", lambda *args, **kwargs: opened.set())
        start = threading.Thread.start
        threads = []

        def interrupted_start(thread):
            assert iterator._iterator in task._prefetch_group._active
            if launched:
                start(thread)
                threads.append(thread)
            raise KeyboardInterrupt("interrupted during startup")

        with monkeypatch.context() as interrupted:
            interrupted.setattr(threading.Thread, "start", interrupted_start)
            with pytest.raises(KeyboardInterrupt, match="interrupted during startup"):
                next(iterator)
        task.close()
        for thread in threads:
            thread.join(timeout=3)
            assert not thread.is_alive()
        assert not opened.is_set()
        assert not task._prefetch_group._active


def test_owner_close_blocks_unstarted_and_buffered_iterators():
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        task = plan.materialize(_table(), fit_split="train")
        lazy = task.batches(batch_size=1, prefetch_batches=1)
        buffered = task.batches(batch_size=1, prefetch_batches=100)
        next(buffered)
        producer = buffered._iterator
        with producer._condition:
            assert producer._condition.wait_for(lambda: producer._done, timeout=3)
        task.close()
        assert list(lazy) == []
        assert list(buffered) == []


def test_close_before_iteration_does_not_open_source():
    def fail():
        pytest.fail("closed iterator opened its source")

    iterator = batches(fail)
    iterator.close()
    assert list(iterator) == []


def test_plan_close_stops_prefetch_before_closing_the_source(monkeypatch):
    plan = api.prepare(_contract(), DATASET, api.InMemoryEngine())
    task = plan.materialize(_table(), fit_split="train")
    iterator = task.batches(batch_size=1, prefetch_batches=1)
    next(iterator)
    source = plan._source
    close = source.close

    def checked_close():
        assert iterator._iterator._done
        close()

    monkeypatch.setattr(source, "close", checked_close)
    plan.close()
    assert list(iterator) == []
    task.close()
