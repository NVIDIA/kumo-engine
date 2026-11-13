# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected anchor statistics use the shared collectors/fitter without histories."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from math import sqrt

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.aggregate_materialization import MaterializedValues
from kumo.pytorch.column_stats import ColumnStats, TableColumnEncoder, fit_from_stats
from tests.test_anchor_tables import ANCHORS, DATASET, EVENTS, _contract, _table
from tests.test_model_history_limits import history_data as history_data

diskgraph = pytest.importorskip("diskgraph")


@pytest.mark.parametrize(
    "dtype,labels",
    [
        (api.DType("bool"), [False, True]),
        (api.DType("uint64"), [2**63, 2**63 + 1]),
        (api.DType("date32"), [date(2024, 1, 1), date(2024, 1, 2)]),
        *(
            (
                api.DType("timestamp", unit=unit, timezone="UTC"),
                [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
            )
            for unit in ("s", "ms", "us", "ns")
        ),
        (api.DType("decimal128", precision=20, scale=2), [Decimal("1.25"), Decimal("2.50")]),
        (api.DType("decimal256", precision=40, scale=2), [Decimal("1.25"), Decimal("2.50")]),
    ],
)
def test_materialized_label_stats_accept_declared_types(tmp_path, dtype, labels):
    spec = aggregate_contract()
    spec = replace(
        spec,
        sample=replace(spec.sample, targets={"label": api.ExternalTarget(dtype)}),
        batch=replace(spec.batch, targets={"label": api.LabelTarget("label", api.Vocabulary())}),
    )
    path = tmp_path / "anchors.parquet"
    rows = pa.Table.from_pylist(ANCHORS).append_column(
        "label", pa.array([labels[i % 2] for i in range(len(ANCHORS))], type=dtype.to_arrow())
    )
    pq.write_table(rows, path)
    table = _table(source=api.Parquet(str(path)), targets={"label": "label"})
    sources = {}
    for declared in spec.schema.tables:
        source = tmp_path / f"{declared.name}.parquet"
        schema = pa.schema([(name, dtype.to_arrow()) for name, dtype in declared.columns.items()])
        pq.write_table(
            pa.Table.from_pylist(aggregate_dataset().tables[declared.name].rows, schema=schema),
            source,
        )
        sources[declared.name] = api.Parquet(str(source))
    with api.prepare(
        spec, api.Dataset(sources), api.DiskGraphEngine(index=tmp_path / "index", num_threads=2)
    ) as plan:
        values = plan._materialize_values(table, tmp_path / "values", stats_split="train")
        with plan.materialize(values.anchors, fit_split="train") as task:
            encoder = task.target("label").field(name="value").encoder
            assert encoder.dtype == dtype
            assert len(encoder.entries) == 2
            assert {value for value, _ in encoder.entries} == set(labels)


def test_generated_bool_target_statistics(tmp_path):
    spec = aggregate_contract()
    spec = replace(
        spec,
        sample=replace(
            spec.sample,
            targets={
                "label": api.Compare(
                    api.Aggregate(
                        api.Rows(("user_events",), api.TimeWindow("occurred_at", 0, 100)),
                        "sum",
                        "amount",
                        dtype=api.DType("float64"),
                    ),
                    "greater_than",
                    5,
                    dtype=api.DType("bool"),
                )
            },
        ),
        batch=replace(spec.batch, targets={"label": api.LabelTarget("label", api.Vocabulary())}),
    )
    with api.prepare(spec, aggregate_dataset(), api.InMemoryEngine()) as plan:
        table = _table(source=api.InMemoryRows([dict(row, at=0) for row in ANCHORS]))
        values = plan._materialize_values(table, tmp_path / "values", stats_split="train")
        with plan.materialize(values.anchors, fit_split="train") as task:
            assert len(task.target("label").field(name="value").encoder.entries) == 2


def aggregate_contract():
    spec = _contract()
    score = api.Aggregate(
        api.Rows(("user_events",), api.TimeWindow("occurred_at", stop=0)),
        "sum",
        "amount",
        dtype=api.DType("float64"),
    )
    sample = replace(spec.sample, entity_values={"score": score})
    inputs = {
        "entity": api.Input(
            api.EntitySource(), api.Fields((api.Encode("score", api.ZScore()),)), api.Separate()
        ),
        "tokens": api.Input(
            api.EntitySource(), api.Fields((api.Encode("score", api.Vocabulary()),)), api.Separate()
        ),
    }
    return replace(spec, sample=sample, batch=replace(spec.batch, inputs=inputs))


def aggregate_dataset():
    return replace(
        DATASET,
        tables={
            **DATASET.tables,
            "events": api.InMemoryRows([dict(row, amount=float(row["user_id"])) for row in EVENTS]),
        },
    )


def test_selected_stats_reuse_shared_fitter_without_sample_collection(tmp_path, monkeypatch):
    with api.prepare(aggregate_contract(), aggregate_dataset(), api.InMemoryEngine()) as plan:
        values = plan._materialize_values(
            _table(), tmp_path / "values", batch_size=3, stats_split="train"
        )
        assert values.rows == len(ANCHORS)
        assert values.feature_stats["rows"] == 8
        stats = values.column_statistics("score")
        assert stats["count"] == 8
        assert stats["mean"] == 6.0
        assert stats["std"] == pytest.approx(sqrt(92 / 7))
        assert len(stats["category_counts"]) == 8
        assert MaterializedValues.open(values.anchors.source.path) == values
        metadata = json.loads((tmp_path / "values" / "feature-stats.json").read_text())
        assert "stats" not in metadata
        stats_path = tmp_path / "values" / metadata["columns_file"]
        assert json.loads(stats_path.read_text())["anchors"][values.features["score"]] == stats

        def forbidden(*args, **kwargs):
            pytest.fail("stats-backed scalar fitting fetched model samples/history")

        monkeypatch.setattr(plan._require_open(), "build", forbidden)
        monkeypatch.setattr(plan._require_open(), "build_model", forbidden, raising=False)
        with plan.materialize(values.anchors, fit_split="train") as task:
            encoder = task.output("entity").field(name="score").encoder
            assert encoder.mean == 6.0
            assert encoder.scale == pytest.approx(sqrt(92 / 8))
            assert len(task.output("tokens").field(name="score").encoder.entries) == 8
        assert not list(tmp_path.glob(".materialize-*"))


def test_selection_is_train_membership_not_a_global_cutoff(tmp_path):
    selected = [
        dict(row, split="train" if i in (0, 11) else "valid") for i, row in enumerate(ANCHORS)
    ]
    with api.prepare(aggregate_contract(), aggregate_dataset(), api.InMemoryEngine()) as plan:
        values = plan._materialize_values(
            _table(source=api.InMemoryRows(selected)),
            tmp_path / "values",
            stats_split="train",
            batch_size=1,
        )
        stats = values.column_statistics("score")
        assert stats["count"] == 2 and stats["mean"] == 6.5
        with pytest.raises(ValueError, match="different fit split"):
            plan.materialize(values.anchors, fit_split="valid")


def test_missing_split_publishes_nothing(tmp_path):
    path = tmp_path / "values"
    with api.prepare(aggregate_contract(), aggregate_dataset(), api.InMemoryEngine()) as plan:
        with pytest.raises(ValueError, match="no anchors"):
            plan._materialize_values(_table(), path, stats_split="missing")
    assert not path.exists()
    assert not list(tmp_path.glob(".materialize-*"))


def test_native_mixed_stat_sets_keep_count_null_and_sample_std_conventions(tmp_path):
    data = tmp_path / "x.parquet"
    pq.write_table(pa.table({"x": [1.0, None, 3.0, 5.0]}), data)
    directory = tmp_path / "stats"
    directory.mkdir()
    diskgraph.write_column_stats_for_parquet(
        [str(data)],
        json.dumps([{"name": "x", "ingested_dtype": "float64"}]),
        json.dumps([["categorical", "numerical"]]),
        str(directory),
        quantiles=[0, 0.5, 1],
    )
    entry = json.loads((directory / "col_stats.json").read_text())["anchors"]["x"]
    assert entry["count"] == 4 and entry["null_count"] == 1 and entry["invalid_count"] == 0
    assert entry["std"] == 2.0 and entry["mean"] == 3.0
    assert entry["quantiles"] == pytest.approx([1, 3, 5])
    for config, kind in [(api.ZScore(), "numerical"), (api.Vocabulary(), "categorical")]:
        request = ("anchors", "x", kind)
        fitted = fit_from_stats(
            "x",
            TableColumnEncoder(config, api.DType("float64"), (("anchors", "x"),)),
            ColumnStats({request: entry}, {"value": None, "unit": "seconds"}),
        )
        if kind == "numerical":
            assert fitted.scale == pytest.approx(sqrt(8 / 3))
        else:
            assert len(fitted.entries) == 3


def test_generated_label_encoder_uses_selected_column_statistics(tmp_path, monkeypatch):
    spec = aggregate_contract()
    sample = replace(spec.sample, targets={"label": api.ExternalTarget(api.DType("float64"))})
    batch = replace(spec.batch, targets={"label": api.LabelTarget("label", api.ZScore())})
    with api.prepare(
        replace(spec, sample=sample, batch=batch), aggregate_dataset(), api.InMemoryEngine()
    ) as plan:
        table = _table(
            source=api.InMemoryRows([dict(row, label=float(row["user_id"])) for row in ANCHORS]),
            targets={"label": "label"},
        )
        values = plan._materialize_values(table, tmp_path / "values", stats_split="train")

        def forbidden(*args, **kwargs):
            pytest.fail("target statistics fitting fetched labels/history")

        monkeypatch.setattr(plan._require_open(), "build", forbidden)
        with plan.materialize(values.anchors, fit_split="train") as task:
            encoder = task.target("label").field(name="value").encoder
            assert encoder.mean == 6
            assert encoder.scale == pytest.approx(sqrt(92 / 8))


@pytest.mark.parametrize("zero", [False, True])
def test_materialized_buckets_use_shared_statistics(tmp_path, zero, monkeypatch):
    spec = aggregate_contract()
    config = api.Buckets(bins=4, separate_zero=True)
    batch = replace(
        spec.batch,
        inputs={
            "entity": api.Input(
                api.EntitySource(), api.Fields((api.Encode("score", config),)), api.Separate()
            )
        },
    )
    dataset = aggregate_dataset()
    expected_values = [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0]
    if zero:
        dataset = replace(
            dataset,
            tables={
                **dataset.tables,
                "events": api.InMemoryRows([dict(row, amount=0.0) for row in EVENTS]),
            },
        )
        expected_values = [0.0] * 8
    with api.prepare(replace(spec, batch=batch), dataset, api.InMemoryEngine()) as plan:
        calls = []
        shared_stats = diskgraph.write_column_stats_for_parquet

        def collect(*args, **kwargs):
            calls.append(args)
            return shared_stats(*args, **kwargs)

        def forbidden(*args, **kwargs):
            pytest.fail("materialized Buckets performed a second exact-quantile pass")

        monkeypatch.setattr(diskgraph, "write_column_stats_for_parquet", collect)
        monkeypatch.setattr(diskgraph, "compute_col_stats_for_parquet", forbidden)
        values = plan._materialize_values(_table(), tmp_path / "values", stats_split="train")
        with plan.materialize(values.anchors, fit_split="train") as task:
            boundaries = task.output("entity").field(name="score").encoder.boundaries
            if zero:
                assert boundaries == ()
            else:
                assert len(boundaries) == 2
                for cut, probability in zip(boundaries, (1 / 3, 2 / 3), strict=True):
                    rank = sum(value <= cut for value in expected_values) / len(expected_values)
                    assert rank == pytest.approx(probability, abs=1.5 / len(expected_values))
        requests = [
            api.SampleRequest(row["sid"], row["user_id"], row["at"], split=row["split"])
            for row in ANCHORS
        ]
        with plan.materialize(requests, fit_split="train") as task:
            row_boundaries = task.output("entity").field(name="score").encoder.boundaries
            if zero:
                assert row_boundaries == ()
            else:
                assert len(row_boundaries) == 2
                for cut, probability in zip(row_boundaries, (1 / 3, 2 / 3), strict=True):
                    rank = sum(value <= cut for value in expected_values) / len(expected_values)
                    assert rank == pytest.approx(probability, abs=1.5 / len(expected_values))
        assert len(calls) == 1
        assert "bucket_cuts" not in values.feature_stats


def _target_free_diskgraph_case(history_data, tmp_path):
    schema, dataset, _, _, _ = history_data
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        entity_values={
            "spent": api.Aggregate(
                api.Rows(("orders",), api.TimeWindow("at", stop=0)),
                "sum",
                "amount",
                dtype=api.DType("float64"),
            )
        },
    )
    spec = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            {
                "entity": api.Input(
                    api.EntitySource(),
                    api.Fields((api.Encode("spent", api.ZScore()),)),
                    api.Separate(),
                )
            },
            batching=api.Batching(max_samples=2),
        ),
    )
    path = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table({"entity": [1, 3, 1], "at": [5, 5, 4], "split": ["train", "train", "valid"]}), path
    )
    return spec, dataset, api.AnchorTable(api.Parquet(str(path)), "entity", "at", split="split")


def test_fitted_target_free_task_reuse_needs_no_fit_split(history_data, tmp_path, monkeypatch):
    spec, dataset, anchors = _target_free_diskgraph_case(history_data, tmp_path)
    with api.prepare(
        spec, dataset, api.DiskGraphEngine(index=tmp_path / "index", num_threads=2)
    ) as plan:
        with plan.materialize(anchors, fit_split="train") as initial:
            fitted = initial.fitted_state.inputs
            expected = [
                batch.inputs["entity"]["spent"]["values"]
                for batch in initial.batches(output_format="model", shuffle=False)
            ]

        def forbidden(*args, **kwargs):
            pytest.fail("reusing all fitted state collected new encoder statistics")

        monkeypatch.setattr(diskgraph, "write_column_stats_for_parquet", forbidden)
        with plan.materialize(anchors, fitted=fitted) as reused:
            actual = [
                batch.inputs["entity"]["spent"]["values"]
                for batch in reused.batches(output_format="model", shuffle=False)
            ]
            assert len(actual) == len(expected)
            for got, want in zip(actual, expected, strict=True):
                assert got.equal(want)


def test_saved_fit_record_omits_temporary_values_paths(history_data, tmp_path):
    from pathlib import Path

    spec, dataset, anchors = _target_free_diskgraph_case(history_data, tmp_path)
    index = tmp_path / "index"
    with api.prepare(spec, dataset, api.DiskGraphEngine(index=index, num_threads=2)) as plan:
        with plan.materialize(anchors, fit_split="train") as task:
            (directory,) = (index / "tasks").glob("task-*")
            assert directory.exists()
            fit = task.fitted_state.fit["inputs"]
            assert fit["fit_split"] == ["train"]
            files = fit["statistics"]["col_stats"]
            assert all(not Path(file).is_absolute() for file in files)
            assert all((index / "col_stats" / file).exists() for file in files)
            artifact = task.to_artifact()
            artifact.save(tmp_path / "artifact")
        assert not directory.exists()
    restored = api.Artifact.load(tmp_path / "artifact")
    assert restored.fitted_state.fit["inputs"] == artifact.fitted_state.fit["inputs"]
