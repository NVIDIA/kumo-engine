# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify measured boundaries, native ownership and scoped restoration."""

import io
import json
from test.kumo_pytorch.benchmarking.kumo_observers import KumoObservers
from test.kumo_pytorch.benchmarking.recording import Recorder
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.lifecycle import input_digest

from kumo.pytorch import api as r

from .fixtures import build_fixture, probe_anchor_table


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


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
