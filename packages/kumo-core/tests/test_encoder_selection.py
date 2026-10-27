# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from kumo_core.encoder_selection import (
    _EncoderStat,
    select_encoder,
    select_target_encoder,
    select_text_encoder,
)
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.pquery import (
    NativeGraphColumn,
    _column_encoder_stats,
    _feature_schema,
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
        training_table_generation_plan=TrainingTableGenerationPlan(),
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


def test_id_selection_warns_without_the_statistic_population(caplog):
    stats = {
        ColStatType.NA_FRACTION: _EncoderStat(0.0, approx=True),
        ColStatType.NUM_UNIQUE: _EncoderStat(8_000, approx=True),
    }

    selected = select_encoder("users.account_id", Dtype.int64, Stype.ID, stats)

    assert selected.name == "Null"
    assert "users.account_id" in caplog.text
    assert "must carry num_entries" in caplog.text


@pytest.mark.parametrize("table_name", [None, "users"])
def test_portable_selector_warns_for_approximate_exact_stats(table_name, caplog):
    column = NativeGraphColumn("category", "string", "categorical",
                               table_name=table_name)
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

    assert _infer_encoder(column, stats)["name"] == "Null"
    assert column.qualified_name in caplog.text
    assert "COUNT must be exact" in caplog.text


@pytest.mark.parametrize("missing_stat", [
    ColStatType.STRING_GLOVE_OVERLAP,
    ColStatType.STRING_AVG_TOKENS,
    ColStatType.STRING_MAX_TOKENS,
    ColStatType.STRING_AVG_NON_CHAR,
])
def test_missing_text_stats_warn_with_the_exact_reason(missing_stat, caplog):
    column = NativeGraphColumn("description", "string", "text",
                               table_name="users")
    stats = {
        ColStatType.NA_FRACTION: _EncoderStat(0.0, approx=True),
        ColStatType.STRING_GLOVE_OVERLAP: _EncoderStat(0.8, approx=True),
        ColStatType.STRING_AVG_TOKENS: _EncoderStat(5.0, approx=True),
        ColStatType.STRING_MAX_TOKENS: _EncoderStat(20.0, approx=True),
        ColStatType.STRING_AVG_NON_CHAR: _EncoderStat(0.0, approx=True),
    }
    del stats[missing_stat]
    reason = (f"Column Stat {missing_stat} is required for encoder inferral "
              "but was not provided.")

    selected = select_encoder(column.qualified_name, Dtype.string, Stype.text,
                              stats)
    assert selected.name == "Null"
    assert _infer_encoder(column, stats)["name"] == "Null"
    assert len(caplog.records) == 2
    for record in caplog.records:
        assert "users.description" in record.getMessage()
        assert reason in record.getMessage()


@pytest.mark.parametrize("dtype,stype,missing_stat", [
    ("float", "numerical", "MEAN"),
    ("string", "categorical", "COUNT"),
    ("string", "multicategorical", "NUM_UNIQUE_MULTI"),
    ("int64", "ID", "NUM_UNIQUE"),
    ("floatlist", "sequence", "SEQUENCE_MIN_LENGTH"),
])
def test_missing_feature_stats_return_null_and_warn(
        dtype, stype, missing_stat, caplog):
    column = NativeGraphColumn("feature", dtype, stype, table_name="users")

    selected = _infer_encoder(column, {
        ColStatType.NA_FRACTION: _EncoderStat(0.0, approx=True),
    })

    assert selected["name"] == "Null"
    assert "users.feature" in caplog.text
    assert f"Column Stat {missing_stat} is required" in caplog.text


def test_missing_na_fraction_returns_null_and_warns(caplog):
    column = NativeGraphColumn("feature", "float", "numerical",
                               table_name="users")

    assert _infer_encoder(column, {})["name"] == "Null"
    assert "users.feature" in caplog.text
    assert "NA_FRACTION is required" in caplog.text


@pytest.mark.parametrize("count,num_na,num_unique,reason", [
    (100, 101, 2, "NUM_NA (101) exceeds COUNT (100)"),
    (100, 0, -1, "NUM_UNIQUE must be a finite nonnegative count"),
    (100, 0, float("inf"), "NUM_UNIQUE must be a finite nonnegative count"),
])
def test_invalid_category_counts_warn_instead_of_raising(
        count, num_na, num_unique, reason, caplog):
    stats = {
        ColStatType.NA_FRACTION: _EncoderStat(0.0, approx=True),
        ColStatType.COUNT: _EncoderStat(count, approx=False),
        ColStatType.NUM_NA: _EncoderStat(num_na, approx=False),
        ColStatType.NUM_UNIQUE: _EncoderStat(num_unique, approx=False),
        ColStatType.CATEGORY_COUNTS: _EncoderStat([], approx=False),
    }
    column = NativeGraphColumn("category", "string", "categorical",
                               table_name="users")

    assert _infer_encoder(column, stats)["name"] == "Null"
    assert "users.category" in caplog.text
    assert reason in caplog.text


def test_target_selection_still_requires_numerical_stats():
    with pytest.raises(ValueError, match="MEAN is required"):
        select_target_encoder(Stype.numerical, {})


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


@pytest.mark.parametrize("num_unique", [None, "null", 4_000, 4_001])
@pytest.mark.parametrize("approx", [False, True])
def test_text_warnings_identify_tables_and_explain_high_cardinality(
        num_unique, approx, monkeypatch, caplog):
    config = _graph_config()
    table = config["tables"][0]
    table.pop("time_col")
    table["cols"] = [
        {
            "name": "age",
            "dtype": "int64",
            "stype": "numerical"
        },
        {
            "name": "category_01",
            "dtype": "string",
            "stype": "text"
        },
    ]
    config["tables"].append({
        **table,
        "table_name": "orders",
        "cols": [table["cols"][1]],
    })
    raw_stats = {
        "category_01": {
            "NA_FRACTION": _stat(0.0),
            "STRING_GLOVE_OVERLAP": _stat(0.0),
            "STRING_AVG_TOKENS": _stat(1.0),
            "STRING_MAX_TOKENS": _stat(1.0),
            "STRING_AVG_NON_CHAR": _stat(0.5),
        },
    }
    if num_unique is not None:
        raw_stats["category_01"]["NUM_UNIQUE"] = _stat(
            None if num_unique == "null" else num_unique, approx=approx)
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, _table, _kind: json.dumps(raw_stats),
    )
    graph = normalize_graph(config)
    query = parse_and_validate_query("PREDICT users.age > 20 FOR EACH users.id",
                                     graph)

    schema = _feature_schema(graph, query, "index", lag_timesteps=None,
                             year_over_year=False)

    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 2
    for table_name, message in zip(("users", "orders"), warnings):
        qualified_name = f"{table_name}.category_01"
        assert f"Text column {qualified_name!r}" in message
        assert "too short or have too few known words" in message
        if isinstance(num_unique, int) and num_unique > 4_000:
            assert "4001 distinct values, exceeding MAX_CATEGORIES=4000" in message
            assert "Hash() or Index()" in message
            assert ("approximately" in message) == approx
            assert (
                f"model_plan.column_processing.encoder_overrides[{qualified_name!r}]"
                in message)
        else:
            assert "TextEmbedding()" in message
            assert "MAX_CATEGORIES" not in message
        spec = next(spec
                    for spec in schema["node_types"][table_name]["column_specs"]
                    if spec["column_name"] == "category_01")
        assert spec["encoder"]["name"] == "Null"
