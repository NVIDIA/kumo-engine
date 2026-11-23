# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliver each eligible original source row through the selected DiskGraph binding."""

import os
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from e2e.datasets.binding_sources import full_source
from kumo.pytorch import api
from tests.e2e.binding_plan import BINDINGS, NIGHTLY_HISTORY_EVENTS
from tests.e2e.binding_report import BindingReport
from tests.e2e.full_binding_run import (
    _native_training,
    full_batches,
    full_tables,
    source_population,
    training_contract,
)
from tests.helpers import parquet_anchors
from tests.test_history_cap_pushdown import _history_case


def test_training_windows_continue_after_warmup_and_ignore_ineligible_batches(
    tmp_path, monkeypatch
):
    from e2e.datasets.mbd import CATEGORIES, prepare_mbd
    from tests.e2e import binding_training

    anchors = pa.table(
        {
            "sample_id": list("abcdef"),
            "entity_id": ["client"] * 6,
            "anchor_row_position": range(6),
            "split": ["train"] * 4 + ["valid", "test"],
            "future_count": [None, 1, 2, 3, 4, 5],
            **dict.fromkeys(
                ("as_of_time", "graph_as_of_time"), pa.array([20] * 6, pa.timestamp("s"))
            ),
        }
    )
    events = pa.table(
        {
            "client_id": ["client"],
            "event_id": [0],
            "event_time": pa.array([10], pa.timestamp("ns")),
            "amount": [1.0],
            **dict.fromkeys(CATEGORIES, [1]),
        }
    )
    spec, data, table = prepare_mbd((anchors, events, {"client": 0}), tmp_path / "prepared")
    monkeypatch.setattr(binding_training, "WARMUP_UPDATES", 1)
    monkeypatch.setattr(binding_training, "MEASURED_UPDATES", 2)
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1, num_shards=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train") as task,
        _native_training(task),
    ):
        result = binding_training.train_full_binding(
            task,
            "M04",
            batch_size=1,
            split_batches=partial(full_batches, task, anchors, 1),
            device="cpu",
            train_window=nullcontext,
            bounded=True,
        )
    rows = (result["warmup"], result["measurement"], *result["validation"], result["test"])
    assert [r["samples"] for r in rows] == [2, 2, 1, 1]
    assert [r["optimizer_updates"] for r in rows] == [1, 2, 0, 0]
    assert [r["heads"]["count"]["eligible"] for r in rows] == [1, 2, 1, 1]


def test_training_verifies_reordered_native_rows_exactly_once(tmp_path):
    from tests.test_native_anchor_handoff import _native_inputs

    spec, data, table = _native_inputs(tmp_path)
    path = tmp_path / "anchors.parquet"
    rows = pq.read_table(path)
    source = rows.select(["sid", "split"]).rename_columns(["sample_id", "split"])
    rows = rows.append_column("anchor_row_position", pa.array(range(len(rows))))
    pq.write_table(rows.take([5, 2, 3, 1, 4, 0, 6]), path)
    table = replace(table, metadata=("anchor_row_position",))
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train") as task,
    ):
        batches = list(full_batches(task, source, 3, "train"))
    assert sorted(s for batch in batches for s in batch.metadata["sample_ids"]) == list("abdf")
    replay = Mock()
    replay.batches.return_value = batches
    with pytest.raises(ValueError, match="arrived after"):
        list(full_batches(replay, source, 3, "train", ordered=True))
    with pytest.raises(ValueError, match="expected sample ID"):
        list(full_batches(replay, source.take([1, 0, 2, 3, 4, 5, 6]), 3, "train"))
    replay.batches.return_value = batches + batches[:1]
    with pytest.raises(ValueError, match="repeated source row"):
        list(full_batches(replay, source, 3, "train"))
    replay.batches.return_value = batches[:-1]
    with pytest.raises(ValueError, match="omitted source row"):
        list(full_batches(replay, source, 3, "train"))


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
    _, counts, targets, cutoff, identities = source_population(binding, files)
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
                task, identities, 32 if binding == "L01" else 4, split, ordered=True
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
