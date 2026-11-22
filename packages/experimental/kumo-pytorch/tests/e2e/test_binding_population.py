# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliver each eligible original source row through the selected DiskGraph binding."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from e2e.datasets.binding_sources import full_source
from kumo.pytorch import api
from tests.e2e.binding_plan import BINDINGS, NIGHTLY_HISTORY_EVENTS
from tests.e2e.binding_report import BindingReport
from tests.e2e.full_binding_run import (
    full_batches,
    full_tables,
    source_population,
    training_contract,
)
from tests.helpers import parquet_anchors
from tests.test_history_cap_pushdown import _history_case


def test_training_history_keeps_recent_merged_rows(tmp_path, native_anchor_delivery):
    original, data = _history_case(
        tmp_path, None, "ascending", api.Packed(), False, extended=True, budget=False, graph=True
    )
    # 127 physical events, with both families tied at the boundary of the most recent 64.
    events = {
        name: replace(
            event,
            rows=replace(event.rows, time=api.TimeWindow("time", stop=-38 if name == "web" else 0)),
        )
        for name, event in original.sample.events.items()
    }
    original = replace(original, sample=replace(original.sample, events=events))
    anchors = parquet_anchors(
        tmp_path / "anchors.parquet", {"id": [1, 2], "at": [100, 100]}, "id", "at"
    )
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    for spec, count in ((training_contract(original), NIGHTLY_HISTORY_EVENTS), (original, 127)):
        with (
            api.prepare(spec, data, engine) as plan,
            plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task,
        ):
            batch = next(iter(native_anchor_delivery(task)))
        expected = [v for t in range(64) for v in ((t, 1000 + t) if t < 63 else (t,))][-count:]
        history = batch.inputs["history"]
        assert history["offsets"].tolist() == [0, count, count]
        assert history["value"]["values"].tolist() == expected
        assert batch.inputs["ages"]["age"]["values"].tolist() == [100 - v % 1000 for v in expected]
        assert batch.targets["labels"]["label"]["values"].tolist() == [v % 1000 for v in expected]
    assert all(event.limit is None for event in original.sample.events.values())


def _check_population(binding, files, native, index):
    spec, data, table = native
    _, _, selector, representation = BINDINGS[binding]
    column, counts, targets, cutoff = source_population(binding, files)
    report = BindingReport(
        binding,
        selector,
        representation,
        counts,
        targets,
        spec.batch.inputs,
        mode="full",
        scope="full_source",
    )
    engine = api.DiskGraphEngine(index=index, num_threads=1, num_shards=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=cutoff) as task,
    ):
        for split, count in counts.items():
            assert task.count(split) == count
            for batch in full_batches(
                task, files["anchors"], column, 32 if binding == "L01" else 4, split
            ):
                report.observe(batch)
    report.publish()


@pytest.mark.parametrize("binding", BINDINGS)
@pytest.mark.binding_report("full")
def test_full_binding(binding, tmp_path):
    if os.getenv("KUMO_PYTORCH_FULL_BINDING") != binding:
        pytest.skip("select the registered full-population binding on manual dispatch")
    if not os.environ.get("KUMO_PYTORCH_DIAGNOSTICS"):
        pytest.fail("set KUMO_PYTORCH_DIAGNOSTICS to validate full-population evidence")
    files = full_source(binding, Path(os.environ["KUMO_PYTORCH_DATA_ROOT"]))
    native = full_tables(binding, files, tmp_path / "prepared")
    _check_population(binding, files, native, tmp_path / "index")
