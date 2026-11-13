# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected anchor statistics use the shared collectors/fitter without histories."""

import json
from dataclasses import replace
from math import sqrt

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.aggregate_materialization import MaterializedValues
from kumo.pytorch.column_stats import ColumnStats, TableColumnEncoder, fit_from_stats
from tests.test_anchor_tables import ANCHORS, DATASET, EVENTS, _contract, _table

diskgraph = pytest.importorskip("diskgraph")


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
def test_materialized_bucket_cuts_preserve_row_fitter_quantiles(tmp_path, zero):
    from kumo.pytorch.encoding import fit_encoder

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
        values = plan._materialize_values(_table(), tmp_path / "values", stats_split="train")
        with plan.materialize(values.anchors, fit_split="train") as task:
            assert (
                task.output("entity").field(name="score").encoder.boundaries
                == fit_encoder(
                    "reference", config, api.DType("float64"), expected_values
                ).boundaries
            )
