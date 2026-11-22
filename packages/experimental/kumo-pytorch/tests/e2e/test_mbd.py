# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transaction forecasts checked against original MBD rows."""

import math
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from e2e.datasets.mbd import CATEGORIES, M03_MANIFEST, M04_MANIFEST, load_mbd, prepare_mbd
from kumo.pytorch import api
from tests.e2e.binding_report import BindingReport
from tests.e2e.history import check_mbd_packed_history, check_source_graph, source_vocabulary
from tests.e2e.loader_timing import correctness_stage
from tests.test_inference_lifecycle import assert_same, batch_location

_HEADS = ("event_type", "event_subtype")


@pytest.mark.parametrize(
    ("manifest", "verified"),
    [
        (M04_MANIFEST, {3899798: 709138, 9371016: 178273, 26822: 533358}),
        (M03_MANIFEST, {3950478: 709138, 3583375: 709138, 9371016: 178273, 2528648: 937042}),
    ],
    ids=["m04-count", "m03-category"],
)
@pytest.mark.binding_report("correctness")
def test_mbd_transaction_forecast(tmp_path, manifest, verified):
    root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
    if root is None:
        pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
    _check_forecast(load_mbd(Path(root), manifest), manifest, tmp_path, verified)


@pytest.mark.binding_report("correctness")
def test_mbd_category_ties_and_independent_masks(tmp_path):
    pytest.importorskip("diskgraph")
    tick, window = 10**9, 30 * 86400
    start = 1_640_995_200

    def clocks(offsets):
        return pa.array([start * tick + offset for offset in offsets], pa.timestamp("ns"))

    # The first physical row wins each forecast tie, even when read second.
    events = pa.table(
        {
            **dict.fromkeys(CATEGORIES, [1] * 7),
            "client_id": ["client"] * 7,
            "event_id": [0, 1, 3, 2, 5, 4, 6],
            "event_time": clocks([-2 * tick, -1, 0, 0, 10 * tick, 10 * tick, (10 + window) * tick]),
            "event_type": [1, 2, 8, None, 9, 3, 1],
            "event_subtype": [1, 2, 8, 7, 9, None, 1],
            "amount": [-1.0, 0.0, 1.0, None, 1.0, 1.0, 1.0],
        }
    )
    anchors = pa.table(
        {
            "sample_id": ["first", "second"],
            "entity_id": ["client"] * 2,
            "anchor_row_position": [0, 1],
            "split": ["train"] * 2,
            "source_fold": [0, 0],
            "as_of_time": clocks([0, 10 * tick]),
            "outcome_end": clocks([window * tick, (window + 10) * tick]),
            "next_event_type": [None, 3],
            "next_event_subtype": [7, None],
            "next_event_type_valid": [False, True],
            "next_event_subtype_valid": [True, False],
        }
    )
    _check_forecast(
        (anchors, events, {"client": 0}),
        M03_MANIFEST,
        tmp_path,
        {0: 0, 1: 0},
        selector="ties fixture",
    )


def _check_forecast(tables, manifest, tmp_path, verified, selector=None):
    anchors, raw, ranks = tables
    category = manifest == M03_MANIFEST
    clocks = {"as_of_time", "outcome_end", "graph_as_of_time"}
    for name in clocks.intersection(anchors.column_names):
        anchors = anchors.set_column(
            anchors.schema.get_field_index(name), name, anchors[name].cast(pa.int64())
        )
    registered = {r["sample_id"]: r for r in anchors.to_pylist()}
    assert {
        r["anchor_row_position"]: ranks[r["entity_id"]] for r in registered.values()
    } == verified

    categories = (
        "currency",
        "event_type",
        "event_subtype",
        "src_type11",
        "src_type12",
        "dst_type11",
        "dst_type12",
        "src_type21",
        "src_type22",
        "src_type31",
        "src_type32",
    )
    fields = {name: api.Vocabulary() for name in categories}
    fields.update(amount=api.ZScore(), event_time_seconds=api.Raw())
    events = {client: [] for client in ranks}
    for row in raw.append_column("ns", raw["event_time"].cast(pa.int64())).to_pylist():
        features = {"event_time_seconds": row["ns"] // 10**9}
        for name in (*categories, "amount"):
            value = row[name]
            if value is None or not math.isfinite(value):
                value = None
            elif name != "amount":
                value = str(int(value)) if value == int(value) else repr(value)
            features[name] = value
        events[row["client_id"]].append((row["ns"], row["event_id"], features))

    spec, data, table = prepare_mbd(tables, tmp_path / "prepared")
    window = 30 * 86400
    if category:
        forecasts = _category_forecasts(registered, events)
        projected = pq.read_table(table.source.path, columns=["sample_id", *_HEADS])
        for name in _HEADS:
            assert anchors["next_" + name].type == pa.int64()
            assert anchors["next_" + name + "_valid"].type == pa.bool_()
            assert projected[name].type == pa.string()
            assert projected[name].to_pylist() == [
                forecasts[s][name] for s in projected["sample_id"].to_pylist()
            ]
    else:
        originals = pq.read_table(table.source.path, columns=["sample_id", "original_count"])
        assert originals["original_count"].type == pa.int64()
        assert {r["sample_id"]: r["original_count"] for r in originals.to_pylist()} == {
            s: row["future_count"] for s, row in registered.items()
        }
        assert spec.sample.targets["generated_count"] == api.Aggregate(
            api.Rows(("client_transactions",), api.TimeWindow("event_time", 0, window)),
            "count",
            dtype=api.DType("int64"),
        )
    cutoff = (
        max(row["as_of_time"] for row in registered.values() if row["split"] == "train") // 10**9
        - 1
    )
    population = Counter(row["split"] for row in registered.values())
    if category:
        heads = {
            name: Counter(
                row["split"] for row in registered.values() if row["next_" + name + "_valid"]
            )
            for name in _HEADS
        }
    else:
        heads = dict.fromkeys(("count", "generated_count"), population)
    evidence = BindingReport(
        "M03" if category else "M04",
        selector or manifest.name,
        "nuformer" if category else "graphsage",
        population,
        heads,
        ("history", "identity") if category else ("graph",),
    )
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=cutoff) as task,
        correctness_stage("delivery"),
    ):
        delivered = []
        for batch in task.batches(batch_size=3, output_format="model"):
            meta = batch.metadata
            delivered.extend(meta["sample_ids"])
            assert not any(meta["is_padding"])
            expected, counts = {}, []
            for owner, sample in enumerate(meta["sample_ids"]):
                row = registered[sample]
                start, end = row["as_of_time"], row["outcome_end"]
                assert end == start + window * 10**9
                execution = start if category else start - 10**9
                assert meta["as_of_time"][owner].item() == execution // 10**9
                assert row["source_fold"] == {"train": 0, "valid": 3, "test": 4}[row["split"]]
                assert meta["splits"][owner] == row["split"]
                assert meta["entity_ids"][owner] == ranks[row["entity_id"]]
                metadata = meta["request_metadata"][owner]
                assert metadata["anchor_row_position"] == row["anchor_row_position"]
                assert metadata["registered_entity_id"] == row["entity_id"]
                assert metadata["prediction_time_ns"] == start
                if not category:
                    assert row["graph_as_of_time"] == execution
                    source_events = events[row["entity_id"]]
                    times = [time for time, _, _ in source_events]
                    assert min(times) < start and end <= max(times)  # Includes the covered zero.
                    expected.update(
                        ((owner, "transactions", i), f)
                        for time, i, f in source_events
                        if time < start
                    )
                    counts.append(sum(start <= time < end for time in times))
                    assert row["future_count"] == counts[-1]
            if category:
                _check_category_targets(task, batch, registered, forecasts, ranks)
                check_mbd_packed_history(task, batch, registered, events, fields)
                evidence.observe(batch)
                continue
            assert set(batch.inputs) == {"graph"}
            check_source_graph(
                task, batch, expected, {"transactions": ("client_transactions", fields)}
            )
            assert_same(
                batch.targets,
                {
                    name: {
                        "values": torch.tensor(counts, dtype=dtype),
                        "valid": torch.ones(len(counts), dtype=torch.bool),
                    }
                    for name, dtype in (("count", torch.float32), ("generated_count", torch.int64))
                },
                "targets",
                locate=batch_location(batch),
            )
            evidence.observe(batch)
        assert sorted(delivered) == sorted(registered)
        evidence.publish()


def _category_forecasts(registered, events):
    answers = {}
    times = [ns for history in events.values() for ns, _, _ in history]
    extent = min(times), max(times)
    for sample, row in registered.items():
        start, end = row["as_of_time"], row["outcome_end"]
        assert extent[0] <= start and end <= extent[1]
        # The global physical ordinal breaks equal-nanosecond ties in original file/row order.
        first = min(
            (event for event in events[row["entity_id"]] if start <= event[0] < end), default=None
        )
        answers[sample] = {}
        for name in _HEADS:
            value = first[2][name] if first else None
            stored = row["next_" + name]
            assert row["next_" + name + "_valid"] is (value is not None)
            assert (str(stored) if stored is not None else None) == value
            answers[sample][name] = value
    return answers


def _check_category_targets(task, batch, registered, forecasts, ranks):
    samples = batch.metadata["sample_ids"]
    train = [s for s, row in registered.items() if row["split"] == "train"]
    assert set(batch.targets) == {*_HEADS, "aligned", "matching"}
    for name in _HEADS:
        vocab = source_vocabulary(forecasts[s][name] for s in train)
        assert dict(task.target(name).field(name="value").encoder.entries) == vocab
        values = [forecasts[s][name] for s in samples]
        assert_same(
            batch.targets[name],
            {
                "values": torch.tensor([0 if v is None else vocab.get(v, 1) for v in values]),
                "valid": torch.tensor([v is not None for v in values]),
            },
            f"targets.{name}",
            locate=batch_location(batch),
        )
    train_roots = sorted({ranks[registered[s]["entity_id"]] for s in train})
    group_ids = {root: i for i, root in enumerate(train_roots)}
    roots = [ranks[registered[s]["entity_id"]] for s in samples]
    groups = torch.tensor([group_ids.get(root, -1) for root in roots])
    assert_same(
        batch.targets["matching"],
        {"values": groups, "valid": groups >= 0},
        "targets.matching",
        locate=batch_location(batch),
    )
    assert_same(
        batch.inputs["identity"],
        {
            "root_id": {
                "values": torch.tensor(roots),
                "valid": torch.ones_like(groups, dtype=torch.bool),
            }
        },
        "inputs.identity",
        locate=batch_location(batch),
    )
