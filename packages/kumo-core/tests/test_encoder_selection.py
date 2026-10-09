# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from kumo_core.encoder_selection import (
    _EncoderStat,
    select_encoder,
    select_text_encoder,
)
from kumo_core.model_plan import ModelPlan
from kumo_core.pquery import (
    NativeGraphColumn,
    PredictiveQueryPlanError,
    _column_encoder_stats,
    _infer_encoder,
    build_native_pquery_task_plan,
    normalize_graph,
    parse_and_validate_query,
)
from kumoapi.typing import ColStatType, Dtype, Stype


def _graph_config():
    return {
        "tables": [{
            "table_name":
            "users",
            "source_table_name":
            "main.schema.users",
            "pkey": {
                "name": "id",
                "dtype": "int64",
                "stype": "ID",
            },
            "time_col": {
                "name": "event_time",
                "dtype": "date",
                "stype": "timestamp",
            },
            "cols": [
                {
                    "name": "label",
                    "dtype": "bool",
                    "stype": "categorical",
                },
                {
                    "name": "age",
                    "dtype": "int64",
                    "stype": "numerical",
                },
                {
                    "name": "score",
                    "dtype": "float",
                    "stype": "numerical",
                },
                {
                    "name": "repeated_id",
                    "dtype": "int64",
                    "stype": "ID",
                },
                {
                    "name": "mostly_unique_id",
                    "dtype": "int64",
                    "stype": "ID",
                },
                {
                    "name": "observed_at",
                    "dtype": "date",
                    "stype": "timestamp",
                },
            ],
            "fkeys": [],
        }]
    }


def _stat(value, *, approx=True, num_entries=None):
    return {
        "value": value,
        "approx": approx,
        "num_entries": num_entries,
    }


def test_native_plan_uses_all_query_operands_and_preserves_ordinary_roles(
        tmp_path, monkeypatch):
    raw_stats = {
        "label": {
            "NA_FRACTION": _stat(0.0),
        },
        "repeated_id": {
            "NA_FRACTION": _stat(0.0),
            "NUM_UNIQUE": _stat(8_000, num_entries=10_000),
        },
        "mostly_unique_id": {
            "NA_FRACTION": _stat(0.0),
            "NUM_UNIQUE": _stat(8_001, num_entries=10_000),
        },
        "observed_at": {
            "NA_FRACTION": _stat(0.0),
        },
        "event_time": {
            "NA_FRACTION": _stat(0.0),
        },
    }
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, _table, _kind: json.dumps(raw_stats),
    )
    graph = normalize_graph(_graph_config())
    query = parse_and_validate_query(
        "PREDICT users.age < 20 OR users.score > 60 FOR EACH users.id",
        graph,
    )

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(),
        run_id="shared-selection",
        graph_index_path=str(tmp_path / "index"),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    users = plan.task_dataset.feature_schema["node_types"]["users"]
    assert users["feature_columns"] == [
        "label",
        "repeated_id",
        "mostly_unique_id",
        "observed_at",
        "event_time",
    ]
    assert {
        spec["column_name"]: spec["encoder"]["name"]
        for spec in users["column_specs"]
    } == {
        "label": "Index",
        "repeated_id": "Hash",
        "mostly_unique_id": "Null",
        "observed_at": "Datetime",
        "event_time": "Datetime",
    }


def test_flat_stats_preserve_selection_metadata_and_prefer_exact_values():
    stats = {
        "account_id__kumo.ID__NUM_UNIQUE__True":
        _stat(8_000, approx=True, num_entries=10_000),
        "account_id__kumo.ID__NUM_UNIQUE__False":
        _stat(7_500, approx=False, num_entries=10_000),
    }

    result = _column_encoder_stats(stats, "account_id")

    assert result == {
        ColStatType.NUM_UNIQUE:
        _EncoderStat(value=7_500, approx=False, num_entries=10_000),
    }


def test_flat_stats_do_not_cross_column_name_prefixes():
    stats = {
        "account__kumo.ID__NUM_UNIQUE__False":
        _stat(200, approx=False, num_entries=1_000),
        "account_id__kumo.ID__NUM_UNIQUE__False":
        _stat(750, approx=False, num_entries=1_000),
    }

    assert _column_encoder_stats(stats, "account") == {
        ColStatType.NUM_UNIQUE:
        _EncoderStat(value=200, approx=False, num_entries=1_000),
    }


def test_id_selection_requires_the_statistic_population():
    stats = {
        ColStatType.NA_FRACTION: _EncoderStat(0.0, approx=True),
        ColStatType.NUM_UNIQUE: _EncoderStat(8_000, approx=True),
    }

    with pytest.raises(ValueError, match="must carry num_entries"):
        select_encoder("account_id", Dtype.int64, Stype.ID, stats)


def test_portable_selector_rejects_approximate_exact_stats():
    column = NativeGraphColumn("category", "string", "categorical")
    stats = {
        ColStatType.NA_FRACTION:
        _EncoderStat(0.0, approx=True),
        ColStatType.COUNT:
        _EncoderStat(100, approx=True),
        ColStatType.NUM_NA:
        _EncoderStat(0, approx=False),
        ColStatType.NUM_UNIQUE:
        _EncoderStat(2, approx=False),
        ColStatType.CATEGORY_COUNTS:
        _EncoderStat([["a", 60], ["b", 40]], approx=False),
    }

    with pytest.raises(PredictiveQueryPlanError, match="must be exact"):
        _infer_encoder(column, stats)


def test_text_left_unencoded_is_reported_by_every_inference(caplog):
    stats = {
        stat_type: _EncoderStat(value=value, approx=True)
        for stat_type, value in (
            (ColStatType.STRING_GLOVE_OVERLAP, 0.0),
            (ColStatType.STRING_AVG_TOKENS, 2.0),
            (ColStatType.STRING_MAX_TOKENS, 3.0),
            (ColStatType.STRING_AVG_NON_CHAR, 0.0),
        )
    }
    # Each job (in a long-lived process) is told, not only the first.
    for _ in range(2):
        select_text_encoder("short", stats)
    assert [record.getMessage() for record in caplog.records] == [
        "Text column 'short' will not be encoded by default: its values are "
        "too short or have too few known words. To encode it anyway, set its "
        "encoder to TextEmbedding()."
    ] * 2
