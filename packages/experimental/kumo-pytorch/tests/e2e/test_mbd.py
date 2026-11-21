# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M04 transaction forecasts checked against original MBD rows."""

import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from e2e.datasets.mbd import M04_MANIFEST, prepare_m04, read_m04
from e2e.datasets.staging import verify
from kumo.pytorch import api
from tests.e2e.history import check_source_graph
from tests.test_inference_lifecycle import assert_same


def test_m04_transaction_forecast(tmp_path):
    root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
    if root is None:
        pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
    tables = read_m04(verify(Path(root), M04_MANIFEST))
    anchors, raw, ranks = tables
    # Ranks from the five original membership files in UTF-8 client-ID order.
    verified = {3899798: 709138, 9371016: 178273, 26822: 533358}
    for name in ("as_of_time", "graph_as_of_time", "outcome_end"):
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

    spec, data, table = prepare_m04(tables, tmp_path / "prepared")
    originals = pq.read_table(table.source.path, columns=["sample_id", "original_count"])
    assert originals["original_count"].type == pa.int64()
    assert {r["sample_id"]: r["original_count"] for r in originals.to_pylist()} == {
        s: row["future_count"] for s, row in registered.items()
    }
    window = 30 * 86400
    assert spec.sample.targets["generated_count"] == api.Aggregate(
        api.Rows(("client_transactions",), api.TimeWindow("event_time", 0, window)),
        "count",
        dtype=api.DType("int64"),
    )
    cutoff = (
        max(row["as_of_time"] for row in registered.values() if row["split"] == "train") // 10**9
        - 1
    )
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=cutoff) as task,
    ):
        delivered = []
        for batch in task.batches(
            split=("train", "valid", "test"), batch_size=3, output_format="model"
        ):
            meta = batch.metadata
            delivered.extend(meta["sample_ids"])
            assert not any(meta["is_padding"])
            expected, counts = {}, []
            for owner, sample in enumerate(meta["sample_ids"]):
                row = registered[sample]
                start, end = row["as_of_time"], row["outcome_end"]
                assert end == start + window * 10**9
                assert row["graph_as_of_time"] == start - 10**9
                assert meta["as_of_time"][owner].item() == start // 10**9 - 1
                assert row["source_fold"] == {"train": 0, "valid": 3, "test": 4}[row["split"]]
                assert meta["splits"][owner] == row["split"]
                assert meta["entity_ids"][owner] == ranks[row["entity_id"]]
                metadata = meta["request_metadata"][owner]
                assert metadata["anchor_row_position"] == row["anchor_row_position"]
                assert metadata["registered_entity_id"] == row["entity_id"]
                assert metadata["prediction_time_ns"] == start
                source_events = events[row["entity_id"]]
                times = [time for time, _, _ in source_events]
                assert min(times) < start and end <= max(times)  # Includes the covered zero.
                expected.update(
                    ((owner, "transactions", i), f) for time, i, f in source_events if time < start
                )
                counts.append(sum(start <= time < end for time in times))
                assert row["future_count"] == counts[-1]
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
            )
        assert sorted(delivered) == sorted(registered)
