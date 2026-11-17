# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefetch changes execution placement, not batch semantics or resume positions."""

import contextvars
import gc
import threading
import weakref
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.task import TaskBatches
from tests.test_anchor_tables import DATASET, _contract, _table
from tests.test_preview import _equal, _fixture


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


@pytest.mark.parametrize("save_depth", [0, 1, 3])
@pytest.mark.parametrize("resume_depth", [0, 1, 3])
def test_resume_tracks_consumption_and_can_change_prefetch_depth(save_depth, resume_depth):
    with batches(lambda: iter(range(20)), save_depth) as iterator:
        assert [next(iterator), next(iterator)] == [0, 1]
        state = iterator.state_dict()
    with batches(lambda: iter(range(20)), resume_depth) as resumed:
        resumed.load_state_dict(state)
        assert list(resumed) == list(range(2, 20))
        assert resumed.state_dict()["batches"] == 20


@pytest.mark.parametrize("error", [RuntimeError("producer failed"), KeyboardInterrupt()])
def test_producer_errors_follow_queued_batches_and_close_the_iterator(error):
    closed = threading.Event()

    def produce():
        try:
            yield 0
            yield 1
            raise error
        finally:
            closed.set()

    with batches(produce, 3) as iterator:
        assert next(iterator) == 0
        assert next(iterator) == 1
        with pytest.raises(type(error)):
            next(iterator)
        assert list(iterator) == []
        assert iterator.state_dict()["batches"] == 2
    assert closed.is_set()


def test_empty_source_and_factory_failure():
    assert list(batches(lambda: iter(()))) == []

    def fail():
        raise ValueError("factory failed")

    with batches(fail) as iterator, pytest.raises(ValueError, match="factory failed"):
        next(iterator)


def test_abandoned_iterator_stops_its_producer():
    closed = threading.Event()

    def produce():
        try:
            yield from range(100)
        finally:
            closed.set()

    iterator = batches(produce)
    next(iterator)
    reference = weakref.ref(iterator)
    del iterator
    gc.collect()
    assert reference() is None
    assert closed.wait(3)


@pytest.mark.parametrize("inference", [False, True])
def test_producer_inherits_start_context_without_changing_consumer_context(inference):
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
        assert next(iterator) == (inference, False, True, "caller")
    iterator.close()
    assert torch.is_grad_enabled()
    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_invalid_prefetch_depth_is_refused_before_work(value):
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        with plan.materialize(_table(), fit_split="train") as task:
            with pytest.raises(ValueError, match="prefetch_batches"):
                task.batches(prefetch_batches=value)


def parquet_dataset(directory, contract, tables):
    sources = {}
    for table in contract.schema.tables:
        schema = pa.schema([(name, dtype.to_arrow()) for name, dtype in table.columns.items()])
        path = directory / f"{table.name}.parquet"
        pq.write_table(pa.Table.from_pylist(tables[table.name], schema=schema), path)
        sources[table.name] = api.Parquet(str(path))
    return api.Dataset(sources)


@pytest.mark.parametrize(
    "engine,output_format",
    [("memory", "inspectable"), ("diskgraph", "inspectable"), ("diskgraph", "model")],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_entity_batches_keep_values_order_shuffling_and_rank_padding(
    tmp_path, engine, output_format, explicit
):
    contract, dataset, anchors = _contract(), DATASET, _table()
    backend = api.InMemoryEngine()
    if engine == "diskgraph":
        pytest.importorskip("diskgraph")
        dataset = parquet_dataset(
            tmp_path, contract, {name: source.rows for name, source in DATASET.tables.items()}
        )
        path = tmp_path / "anchors.parquet"
        pq.write_table(pa.Table.from_pylist(anchors.source.rows), path)
        anchors = replace(anchors, source=api.Parquet(str(path)))
        backend = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with api.prepare(contract, dataset, backend) as plan:
        with plan.materialize(anchors, fit_split="train") as task:
            requests = task._anchors.fetch(api.ALL_ANCHORS, range(12)) if explicit else None
            config = dict(
                batch_size=3,
                shuffle=True,
                seed=7,
                epoch=2,
                rank=1,
                world_size=5,
                output_format=output_format,
            )
            expected = list(task.batches(requests, **config))
            with task.batches(requests, prefetch_batches=2, **config) as iterator:
                actual = list(iterator)
            for old, new in zip(expected, actual, strict=True):
                _equal(old.inputs, new.inputs)
                _equal(old.targets, new.targets)
                _equal(old.metadata, new.metadata)


@pytest.mark.parametrize("kind", ["fields", "masked", "next"])
@pytest.mark.parametrize("inference", [False, True])
def test_native_graph_entity_and_sequence_contracts_are_model_independent(
    tmp_path, kind, inference
):
    pytest.importorskip("diskgraph")
    contract, tables = _fixture()
    inputs = dict(contract.batch.inputs)
    spaces, names, targets = {}, {}, {}
    if kind != "fields":
        spaces = {
            "tokens": api.IdSpace(
                specials={
                    "pad": "[PAD]",
                    "unknown": "[UNK]",
                    "missing": "[MISSING]",
                    "mask": "[MASK]",
                }
            )
        }
        fields = api.Fields(
            (api.Encode("kind", api.Vocabulary(), id_space="tokens"),),
            row=api.Tokens(),
        )
        if kind == "masked":
            names = {"fields": api.FieldNames()}
            fields = replace(fields, field_names="fields", field_space="tokens")
            layout = api.Nested(fields=api.Packed(), events=api.Padded())
            targets = {
                "masked": api.MaskedValuesTarget(
                    input="history",
                    token_probability=0.5,
                    event_probability=0.1,
                    key_probability=0.1,
                )
            }
        else:
            fields = replace(fields, history=api.Flatten(max_tokens=16, separator=None))
            layout = api.Padded()
            targets = {"next": api.NextTokenTarget(input="history")}
        inputs["history"] = api.Input(api.EventSource(("history",)), fields, layout)
    contract = replace(
        contract,
        batch=replace(
            contract.batch,
            inputs=inputs,
            id_spaces=spaces,
            field_names=names,
            targets=targets,
            batching=api.Batching(max_samples=2),
        ),
    )
    dataset = parquet_dataset(tmp_path, contract, tables)
    path = tmp_path / "anchors.parquet"
    pq.write_table(pa.table({"id": [1] * 7, "time": [100] * 7, "sid": list("abcdefg")}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "id", "time", sample_id="sid")
    with api.prepare(
        contract, dataset, api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    ) as plan:
        with plan.materialize(anchors) as fitted_task:
            fitted = fitted_task.fitted_state
        with plan.materialize(anchors, fitted=fitted, inference=inference) as task:
            config = dict(output_format="model", shuffle=True, seed=9, epoch=2)
            expected = list(task.batches(**config))
            with task.batches(prefetch_batches=2, **config) as iterator:
                actual = list(iterator)
            for old, new in zip(expected, actual, strict=True):
                _equal(old.inputs, new.inputs)
                _equal(old.targets, new.targets)
                _equal(old.metadata, new.metadata)
            if inference:
                assert all(not batch.targets for batch in actual)


def test_task_close_stops_prefetch_before_closing_anchors():
    with api.prepare(_contract(), DATASET, api.InMemoryEngine()) as plan:
        task = plan.materialize(_table(), fit_split="train")
        iterator = task.batches(batch_size=1, prefetch_batches=1)
        next(iterator)
        task.close()
        assert list(iterator) == []


@pytest.mark.parametrize("interrupted_owner", ["batches", "task", "plan"])
@pytest.mark.parametrize("retry_owner", ["batches", "task", "plan"])
def test_interrupted_close_waits_for_producer_before_releasing_resources(
    monkeypatch, interrupted_owner, retry_owner
):
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
    thread = None
    try:
        next(iterator)
        assert preparing.wait(3)
        thread = iterator._iterator._thread
        join = thread.join
        source = plan._source
        close_anchors, close_source = task._anchors.close, source.close

        def checked_anchor_close():
            assert finished.is_set()
            assert not thread.is_alive()
            events.append("anchors")
            close_anchors()

        def checked_source_close():
            assert finished.is_set()
            assert not thread.is_alive()
            events.append("source")
            close_source()

        def interrupt_join():
            assert threading.current_thread() is threading.main_thread()
            raise KeyboardInterrupt("consumer join interrupted")

        monkeypatch.setattr(task._anchors, "close", checked_anchor_close)
        monkeypatch.setattr(source, "close", checked_source_close)
        with monkeypatch.context() as interrupted:
            interrupted.setattr(thread, "join", interrupt_join)
            for _ in range(2):
                with pytest.raises(KeyboardInterrupt, match="consumer join interrupted"):
                    owners[interrupted_owner].close()
                assert thread.is_alive()
                assert events == []
                assert plan._source is source
                assert list(iterator) == []
                assert iterator._finalizer.alive

        def release_and_join():
            release.set()
            join(timeout=3)
            assert not thread.is_alive()

        with monkeypatch.context() as retry:
            retry.setattr(thread, "join", release_and_join)
            owners[retry_owner].close()
            assert finished.is_set()
            assert not thread.is_alive()
        assert not iterator._finalizer.alive
        task.close()
        plan.close()
        assert events[0] == "producer"
        assert "anchors" in events and "source" in events
    finally:
        release.set()
        if thread is not None:
            thread.join(timeout=3)
        task.close()
        plan.close()


def test_close_confirms_producer_cleanup_when_join_returns_early(monkeypatch):
    preparing, release, finished = threading.Event(), threading.Event(), threading.Event()

    def produce():
        try:
            yield 0
            preparing.set()
            assert release.wait(5), "producer was not released"
            yield 1
        finally:
            finished.set()

    iterator = batches(produce)
    thread = None
    try:
        assert next(iterator) == 0
        assert preparing.wait(3)
        prefetch = iterator._iterator
        thread = prefetch._thread
        wait_for = prefetch._condition.wait_for

        def release_and_wait(predicate):
            assert not finished.is_set()
            assert not predicate()
            release.set()
            return wait_for(predicate, timeout=3)

        with monkeypatch.context() as interrupted_state:
            # Model join's premature return after an interrupted wait without
            # mutating Python's private thread bookkeeping.
            interrupted_state.setattr(thread, "join", lambda: None)
            interrupted_state.setattr(prefetch._condition, "wait_for", release_and_wait)
            iterator.close()
            assert finished.is_set()
            assert prefetch._done
    finally:
        release.set()
        if thread is not None:
            thread.join(timeout=3)
        iterator.close()


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
        assert not iterator._iterator._thread.is_alive()
        close()

    monkeypatch.setattr(source, "close", checked_close)
    plan.close()
    assert list(iterator) == []
    task.close()
