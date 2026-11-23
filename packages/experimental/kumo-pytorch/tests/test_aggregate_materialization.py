# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pyarrow.parquet as pq
import pytest

from kumo.pytorch.aggregate_materialization import MaterializedValues
from kumo.pytorch.api import (
    Aggregate,
    AnchorTable,
    DType,
    ExternalTarget,
    InMemoryEngine,
    InMemoryRows,
    Parquet,
    Rows,
    TimeWindow,
    prepare,
)
from tests.test_anchor_tables import ANCHORS, DATASET, _contract, _table


def test_materializes_in_chunks_and_reuses_values_without_reading_rows(tmp_path, monkeypatch):
    with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
        source = plan._require_open()
        sizes = []
        build = source.build

        def counted(spec, requests):
            sizes.append(len(requests))
            return build(spec, requests)

        monkeypatch.setattr(source, "build", counted)
        values = plan._materialize_values(_table(), tmp_path / "values.parquet", batch_size=3)
        assert sizes and max(sizes) <= 3
        assert sum(sizes) == len(ANCHORS)
        assert values.rows == len(ANCHORS)
        saved = pq.ParquetFile(tmp_path / "values.parquet" / "part-00000.parquet")
        assert 1 <= saved.metadata.num_row_groups <= 4
        assert saved.read().column(values.features["seen"]).to_pylist() == [1] * len(ANCHORS)
        reopened = MaterializedValues.open(tmp_path / "values.parquet")
        assert reopened == values
        base = plan._require_open()
        original = base.build
        seen = []

        def no_aggregates(spec, requests):
            assert not spec.entity_values
            seen.append(len(requests))
            return original(spec, requests)

        monkeypatch.setattr(base, "build", no_aggregates)
        task = plan.materialize(values.anchors, values=reopened, fit_split="train")
        got = [b for b in task.batches(batch_size=4)]
        assert sum(b.batch_size for b in got) == len(ANCHORS)
        assert seen
        assert all(
            not any(key.startswith("_value_") for key in m)
            for b in got
            for m in b.metadata["request_metadata"]
        )


def test_duplicates_and_distinct_labels_stay_aligned(tmp_path):
    contract = _contract()
    contract = replace(
        contract,
        sample=replace(contract.sample, targets={"manual": ExternalTarget(DType("float64"))}),
    )
    rows = [
        {"sid": "first", "user_id": 1, "at": 100, "label": 3.0},
        {"sid": "second", "user_id": 1, "at": 100, "label": 9.0},
    ]
    anchors = AnchorTable(
        InMemoryRows(rows),
        entity_id="user_id",
        as_of_time="at",
        sample_id="sid",
        targets={"manual": "label"},
    )
    with prepare(contract, DATASET, InMemoryEngine()) as plan:
        values = plan._materialize_values(anchors, tmp_path / "duplicates.parquet", batch_size=1)
    saved = pq.read_table(values.anchors.source.path)
    assert saved.column("_anchor_id").to_pylist() == ["first", "second"]
    assert saved.column(values.anchors.targets["manual"]).to_pylist() == [3.0, 9.0]


def test_forward_targets_and_all_null_first_chunk_keep_declared_types(tmp_path):
    contract = _contract()
    target = Aggregate(
        Rows(("user_events",), time=TimeWindow("occurred_at", 0, 100)),
        "sum",
        "amount",
        dtype=DType("float64"),
    )
    contract = replace(contract, sample=replace(contract.sample, targets={"future": target}))
    rows = [{"sid": "before", "user_id": 1, "at": 0}, {"sid": "after", "user_id": 1, "at": 100}]
    anchors = AnchorTable(InMemoryRows(rows), entity_id="user_id", as_of_time="at", sample_id="sid")
    with prepare(contract, DATASET, InMemoryEngine()) as plan:
        values = plan._materialize_values(anchors, tmp_path / "targets.parquet", batch_size=1)
        saved = pq.read_table(values.anchors.source.path)
        assert saved.column(values.targets["future"]).to_pylist() == [1.0, 0.0]
        assert saved.column(values.features["seen"]).to_pylist() == [0, 1]
        # Materialization never fits; all splits can be computed, while fitting
        # still requires its original explicit selection.
        assert values.contract == plan.contract.fingerprint


def test_failure_never_publishes_partial_values(tmp_path, monkeypatch):
    with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
        source = plan._require_open()
        original = source.build
        calls = 0

        def fail_second(spec, requests):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected computation failure")
            return original(spec, requests)

        monkeypatch.setattr(source, "build", fail_second)
        with pytest.raises(RuntimeError, match="injected"):
            plan._materialize_values(_table(), tmp_path / "partial.parquet", batch_size=3)
    assert not (tmp_path / "partial.parquet").exists()
    assert not list(tmp_path.glob(".materialize-*"))


def test_parquet_scan_is_bounded_and_positional_ids_do_not_restart(tmp_path, monkeypatch):
    from kumo.pytorch.aggregate_materialization import _anchor_chunks
    from kumo.pytorch.engines.parquet_anchors import ParquetAnchors

    path = tmp_path / "anchors.parquet"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(ANCHORS), path, row_group_size=5)
    table = replace(
        _table(),
        source=Parquet(str(path)),
        sample_id=None,
    )
    with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
        monkeypatch.setattr(
            ParquetAnchors, "__init__", lambda *a, **k: pytest.fail("full anchor read")
        )
        chunks = list(_anchor_chunks(plan, table, 3))
    assert all(len(chunk) <= 3 for chunk in chunks)
    assert [request.sample_id for chunk in chunks for request in chunk] == list(range(len(ANCHORS)))
    assert [request.split for chunk in chunks for request in chunk] == [
        row["split"] for row in ANCHORS
    ]


def test_actual_diskgraph_scalar_materialization_reuses_model_values(tmp_path):
    pytest.importorskip("diskgraph")
    import pyarrow as pa
    import torch

    from kumo.pytorch.api import Dataset, DiskGraphEngine
    from tests.test_anchor_tables import SCHEMA

    sources = {}
    for table in SCHEMA.tables:
        path = tmp_path / (table.name + ".parquet")
        schema = pa.schema([(name, dtype.to_arrow()) for name, dtype in table.columns.items()])
        pq.write_table(pa.Table.from_pylist(DATASET.tables[table.name].rows, schema=schema), path)
        sources[table.name] = Parquet(str(path))
    path = tmp_path / "anchors.parquet"
    pq.write_table(pa.Table.from_pylist(ANCHORS), path)
    anchors = replace(_table(), source=Parquet(str(path)))
    with prepare(
        _contract(), Dataset(sources), DiskGraphEngine(index=tmp_path / "index", num_threads=2)
    ) as plan:
        baseline = plan.materialize(anchors, fit_split="train")
        values = plan._materialize_values(anchors, tmp_path / "computed.parquet", batch_size=3)
        cached = plan.materialize(values.anchors, values=values, fit_split="train")
        for before, after in zip(
            baseline.batches(output_format="model"),
            cached.batches(output_format="model"),
            strict=True,
        ):
            assert before.metadata["sample_ids"] == after.metadata["sample_ids"]
            for field in ["score", "seen"]:
                assert torch.equal(
                    before.inputs["entity"][field]["values"],
                    after.inputs["entity"][field]["values"],
                )


def test_nullable_profile_field_materializes_as_null(tmp_path):
    from kumo.pytorch.api import Dataset
    from tests.test_anchor_tables import USERS

    users = [dict(row, score=None if row["user_id"] == 1 else row["score"]) for row in USERS]
    dataset = Dataset({**DATASET.tables, "users": InMemoryRows(users)})
    with prepare(_contract(), dataset, InMemoryEngine()) as plan:
        values = plan._materialize_values(_table(), tmp_path / "nullable.parquet", batch_size=2)
    assert pq.read_table(values.anchors.source.path)[values.features["score"]].to_pylist() == [
        None,
        *[float(i) for i in range(2, 13)],
    ]
