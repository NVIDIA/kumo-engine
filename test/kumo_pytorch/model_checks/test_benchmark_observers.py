# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify measured boundaries, native ownership and scoped restoration."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from test.kumo_pytorch.benchmarking.kumo_observers import KumoObservers
from test.kumo_pytorch.benchmarking.recording import Recorder
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.lifecycle import input_digest

import pytest

from kumo.pytorch import api as r
from kumo.pytorch import assembly

from .fixtures import build_fixture, probe_anchor_table


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_restore_after_failure_and_do_not_observe_other_threads():
    stream = io.StringIO()
    recorder = Recorder(stream)
    original = assembly.fit_encoder

    def fit():
        return assembly.fit_encoder("price", r.ZScore(), r.DType("float64"), [1.0, 3.0])

    with pytest.raises(ValueError, match="deliberate"), KumoObservers(recorder):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(fit).result().mean == 2
        assert not records(stream)
        assert fit().mean == 2
        raise ValueError("deliberate")
    assert assembly.fit_encoder is original
    assert [row["metric"] for row in records(stream)] == ["statistics.fit.elapsed_ns"]
    # The process guard is released even after failure.
    with KumoObservers(recorder):
        pass


def test_reentrant_collector_rejected_without_corrupting_outer():
    recorder = Recorder(io.StringIO())
    original = assembly.fit_encoder
    with KumoObservers(recorder):
        wrapper = assembly.fit_encoder
        with (
            pytest.raises(RuntimeError, match="already active"),
            KumoObservers(recorder),
        ):
            pass
        assert assembly.fit_encoder is wrapper
    assert assembly.fit_encoder is original


def test_native_profile_counts_are_not_inferred_from_other_fields():
    stream = io.StringIO()
    observer = KumoObservers(Recorder(stream))
    observer.capture(
        {
            "native_timings_ns": {
                "traversal": 5,
                "decode_and_select": 11,
                "aggregation": 7,
                "total": 99,
            },
            "native_stats": {"rows_decoded": 13, "edges_visited": 50},
        },
        "build",
        "last_temporal_profile",
    )
    measured = {
        row["metric"]: row["value"]
        for row in records(stream)
        if row["kind"] == "measurement" and row["status"] == "available"
    }
    assert measured == {
        "native.sample.elapsed_ns": 5,
        "native.retrieve.elapsed_ns": 11,
        "native.aggregate.elapsed_ns": 7,
        "rows.decoded": 13,
    }
    assert "rows.examined" not in measured
    unavailable = [row for row in records(stream) if row.get("status") == "unavailable"]
    assert [row["metric"] for row in unavailable] == ["rows.examined"]


def test_stale_profile_not_counted_again():
    class Source:
        def __init__(self):
            self.last_temporal_profile = None

        def build(self, fresh):
            if fresh:
                self.last_temporal_profile = {"native_timings_ns": {"aggregation": 17}}

    source = Source()
    stream = io.StringIO()
    observers = KumoObservers(Recorder(stream))
    observers.attach_source(source)
    try:
        source.build(True)
        source.build(False)
    finally:
        observers.close()
    assert (
        len([row for row in records(stream) if row["kind"] == "backend_profile"]) == 1
    )
    assert "build" not in vars(source)


def test_runtime_graph_cannot_borrow_temporal_measurement_coverage():
    class Source:
        last_graph_profile = None

        def sample_graph_batch(self):
            return "unchanged"

    source = Source()
    stream = io.StringIO()
    observer = KumoObservers(Recorder(stream))
    observer.attach_source(source)
    try:
        assert source.sample_graph_batch() == "unchanged"
    finally:
        observer.close()
    metrics = {
        row["metric"]: row for row in records(stream) if row["kind"] == "measurement"
    }
    assert set(metrics) == {
        "native.sample.elapsed_ns",
        "native.retrieve.elapsed_ns",
        "rows.examined",
    }
    assert all(
        row["status"] == "unavailable" and row["value"] is None
        for row in metrics.values()
    )
    assert "sample_graph_batch" not in vars(source)


def test_multiple_graph_inputs_do_not_repeat_native_profile():
    class Source:
        last_graph_profile = None

        def sample_graph_batch(self, runtime):
            return runtime["answer"]

    stream = io.StringIO()
    observer = KumoObservers(Recorder(stream))
    source = Source()
    observer.attach_source(source)
    profile = {"sampling_ns": 11, "retrieval_ns": 17, "rows_decoded": 3, "rows_examined": 5}
    try:
        for answer in ("first", "next"):
            runtime = {"answer": answer, "native_profile": dict(profile)}
            # Native delivery records one profile; two Graph inputs consume the same work.
            observer._capture_runtime_profile(runtime, "native_iterator")
            assert source.sample_graph_batch(runtime) == answer
            assert source.sample_graph_batch(runtime) == answer
    finally:
        observer.close()
    measured = [row for row in records(stream) if row["kind"] == "measurement"]
    assert len([row for row in records(stream) if row["kind"] == "backend_profile"]) == 2
    assert sum(row["value"] for row in measured if row["metric"] == "rows.decoded") == 6
    assert sum(row["value"] for row in measured if row["metric"] == "native.sample.elapsed_ns") == 22


def test_real_native_queries_and_python_phases(tmp_path):
    fixture = build_fixture()
    dataset = fixture.parquet_dataset(tmp_path / "source")
    contract = build_contract(fixture.schema, "combined", max_events=32)
    stream = io.StringIO()
    recorder = Recorder(stream)
    digests = []
    for action in ("build", "reopen"):
        with KumoObservers(recorder) as observers:  # noqa: SIM117 -- keep observer and plan lifetimes visible
            with r.prepare(
                contract,
                dataset,
                r.DiskGraphEngine(index=tmp_path / "index", num_threads=1),
            ) as plan:
                observers.attach_source(plan._source)
                with plan.materialize(
                    probe_anchor_table(plan, fixture.training), fit_split="train"
                ) as task:
                    observers.attach_source(task._assembler._source)
                    with observers.native_epoch(0, history=True):
                        for batch in task.batches(batch_size=1, output_format="model"):
                            assert len(batch.metadata["sample_ids"]) == 1
                            digests.append(input_digest(batch.inputs))
        observed = {
            row["metric"] for row in records(stream) if row["kind"] == "measurement"
        }
        assert f"index.{action}.elapsed_ns" in observed
    with (
        r.prepare(
            contract,
            dataset,
            r.DiskGraphEngine(index=tmp_path / "index", num_threads=1),
        ) as plan,
        plan.materialize(
            probe_anchor_table(plan, fixture.training), fit_split="train"
        ) as task,
    ):
        baseline = [
            input_digest(batch.inputs)
            for batch in task.batches(batch_size=1, output_format="model")
        ]
    assert digests == baseline * 2
    observed = {
        row["metric"] for row in records(stream) if row["kind"] == "measurement"
    }
    assert {
        "input.fit.elapsed_ns",
        "vocabulary.fit.elapsed_ns",
        "python.plan.elapsed_ns",
        "python.postprocess.elapsed_ns",
        "python.encode.elapsed_ns",
        "python.tokenize.elapsed_ns",
        "native.aggregate.elapsed_ns",
        "native.sample.elapsed_ns",
        "native.retrieve.elapsed_ns",
        "rows.decoded",
        "index.bytes",
    } <= observed
    profiles = [row for row in records(stream) if row["kind"] == "backend_profile"]
    assert any(row["profile_name"] == "native_graph" for row in profiles)
    for row in records(stream):
        if row["kind"] == "measurement" and "backend_profile_id" in row["details"]:
            assert any(
                p["id"] == row["details"]["backend_profile_id"] for p in profiles
            )
