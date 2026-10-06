# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.pquery import (
    PredictiveQueryPlanError,
    _column_stats,
    _default_split,
    _split_options,
    _temporal_timestamps,
    build_generated_training_task_dataset,
    build_native_pquery_task_plan,
    latest_temporal_timestamp,
    normalize_graph,
    parse_and_validate_query,
    query_train_table_time,
    static_link_fact,
    table_row_count,
    target_to_entity_paths,
    to_api_graph,
    to_plan_graph,
)
from kumo_core.training.splits import TrainTableTime
from kumoapi.typing import Stype


def graph_config():
    return {
        "tables": [
            {
                "table_name":
                "users",
                "source_table_name":
                "main.schema.users",
                "pkey": {
                    "name": "id",
                    "dtype": "int64",
                    "stype": "ID"
                },
                "cols": [
                    {
                        "name": "label",
                        "dtype": "bool",
                        "stype": "categorical",
                        "encoder": {
                            "name": "Index"
                        },
                    },
                    {
                        "name": "age",
                        "dtype": "int64",
                        "stype": "numerical",
                        "encoder": {
                            "name": "Numerical"
                        },
                    },
                ],
                "fkeys": [],
            },
            {
                "table_name":
                "products",
                "source_table_name":
                "main.schema.products",
                "pkey": {
                    "name": "id",
                    "dtype": "int64",
                    "stype": "ID"
                },
                "cols": [{
                    "name": "price",
                    "dtype": "float64",
                    "stype": "numerical",
                    "encoder": {
                        "name": "Numerical"
                    },
                }],
                "fkeys": [],
            },
            {
                "table_name":
                "orders",
                "source_table_name":
                "main.schema.orders",
                "pkey": {
                    "name": "id",
                    "dtype": "int64",
                    "stype": "ID"
                },
                "fkeys": [
                    {
                        "name": "user_id",
                        "dtype": "int64",
                        "stype": "ID",
                        "dst_table": "users",
                        "dst_key": "id",
                    },
                    {
                        "name": "product_id",
                        "dtype": "int64",
                        "stype": "ID",
                        "dst_table": "products",
                        "dst_key": "id",
                    },
                ],
                "cols": [],
            },
        ]
    }


def forecasting_graph_config():
    return {
        "tables": [
            {
                "table_name": "articles",
                "source_table_name": "main.schema.articles",
                "pkey": {
                    "name": "aid",
                    "dtype": "int64",
                    "stype": "ID",
                },
                "cols": [],
                "fkeys": [],
            },
            {
                "table_name":
                "events",
                "source_table_name":
                "main.schema.events",
                "pkey": {
                    "name": "event_id",
                    "dtype": "int64",
                    "stype": "ID",
                },
                "time_col": {
                    "name": "event_time",
                    "dtype": "date",
                    "stype": "timestamp",
                },
                "cols": [],
                "fkeys": [{
                    "name": "aid",
                    "dtype": "int64",
                    "stype": "ID",
                    "dst_table": "articles",
                    "dst_key": "aid",
                }],
            },
        ]
    }


def test_real_parser_accepts_an_unrelated_scalar_multicategorical_column():
    config = graph_config()
    users = next(table for table in config["tables"]
                 if table["table_name"] == "users")
    users["cols"].append({
        "name": "tags",
        "dtype": "string",
        "stype": "multicategorical",
        "multicategorical_delimiter": ",",
    })

    query = parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                     config)

    assert query.task_type == "binary_classification"


def test_scalar_string_multicategorical_without_delimiter_is_rejected():
    config = graph_config()
    users = next(table for table in config["tables"]
                 if table["table_name"] == "users")
    users["cols"].append({
        "name": "tags",
        "dtype": "string",
        "stype": "multicategorical",
    })

    with pytest.raises(ValueError, match="requires multicategorical_delimiter"):
        parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                 config)


@pytest.mark.parametrize(
    "query,family,task_type,rhs",
    [
        (
            "PREDICT users.label FOR EACH users.id",
            "node_prediction",
            "binary_classification",
            None,
        ),
        (
            "PREDICT LIST_DISTINCT(orders.product_id) RANK TOP 3 "
            "FOR EACH users.id",
            "link_prediction",
            "static_link_prediction",
            "products",
        ),
    ],
)
def test_real_parser_validator_builds_selected_task(query, family, task_type,
                                                    rhs):
    result = parse_and_validate_query(query, graph_config())

    assert result.task_family == family
    assert result.task_type == task_type
    assert result.rhs_node_type == rhs


def test_timestamp_format_survives_the_portable_graph_round_trip():
    config = graph_config()
    config["tables"][0]["cols"].append({
        "name": "event_time",
        "dtype": "string",
        "stype": "timestamp",
        "timestamp_format": "%Y-%m-%d",
    })

    native = normalize_graph(config)
    restored = to_api_graph(native)

    assert native.table("users").column("event_time").timestamp_format == (
        "%Y-%m-%d")
    columns = {column.name: column for column in restored.tables["users"].cols}
    assert columns["event_time"].timestamp_format == "%Y-%m-%d"


def test_normalize_graph_preserves_direct_portable_encoder_payload():
    native = normalize_graph(graph_config())

    assert native.table("users").column("age").encoder == {"name": "Numerical"}


def test_duplicate_portable_columns_keep_the_last_definition():
    config = graph_config()
    config["tables"][0]["cols"].append({
        "name": "age",
        "dtype": "string",
        "stype": "categorical",
        "encoder": {
            "name": "Hash",
            "num_buckets": 8,
        },
    })

    age = normalize_graph(config).table("users").column("age")

    assert (age.dtype, age.stype, age.encoder) == (
        "string",
        "categorical",
        {
            "name": "Hash",
            "num_buckets": 8,
        },
    )


def test_plan_derives_train_and_task_contracts_from_spec_and_ingest(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users": '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products": '{"price__MEAN": 10}',
            "orders": '{"placeholder": 0}',
        }[table]),
    )
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                     graph)
    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(),
        run_id="native-1",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert plan.train_table.query is not None
    assert plan.train_table.index_path == str(index)
    assert plan.train_table.tt_id == "tt-native-1"
    assert plan.task_dataset.seed_node_type == "users"
    assert plan.task_dataset.feature_schema["node_types"]["users"][
        "feature_columns"] == ["age"]
    assert plan.task_dataset.artifact_root == str(
        Path(tmp_path / "artifacts" / "task_dataset"))
    assert plan.train_table.split_family == "RandomSplit"
    assert plan.train_table.split_config == {"ratio": [0.8, 0.1, 0.1]}


def test_plan_split_comes_from_the_model_plan_when_the_caller_sets_one(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users": '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products": '{"price__MEAN": 10}',
            "orders": '{"placeholder": 0}',
        }[table]),
    )
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                     graph)

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                split="RandomSplit(ratio=[0.7, 0.2, 0.1], seed=5)"), ),
        run_id="native-split-1",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert plan.train_table.split_family == "RandomSplit"
    assert plan.train_table.split_config == {
        "ratio": [0.7, 0.2, 0.1],
        "rnd_seed": 5,
    }


def test_plan_split_converts_range_boundaries_to_epoch_seconds(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users": '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products": '{"price__MEAN": 10}',
            "orders": '{"placeholder": 0}',
        }[table]),
    )
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                     graph)

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                split="TimeRangeSplit(ranges=[['2024-01-01', '2024-02-01'], "
                "['2024-02-01', '2024-03-01']])"), ),
        run_id="native-split-3",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert plan.train_table.split_family == "TimeRangeSplit"
    # Every value the constructor resolves reaches the split config, not just
    # the arguments the caller happened to type.
    assert plan.train_table.split_config == {
        "ranges": [[1_704_067_200, 1_706_745_600],
                   [1_706_745_600, 1_709_251_200]],
        "unit": "days",
        "anchor_start_offset": 0,
        "flexible_boundaries": True,
    }
    assert plan.task_dataset.required_stages == ("train", "val")


@pytest.mark.parametrize("end_time", ["inferred", "2026-06-01"])
def test_temporal_time_range_split_uses_query_window_and_range_anchors(
        tmp_path, monkeypatch, end_time):
    index = tmp_path / "index"
    index.mkdir()
    minimum = "2025-06-28T00:01:30+00:00"
    maximum = "2026-06-27T23:59:59.999+00:00"
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users":
            '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products":
            '{"price__MEAN": 10}',
            "orders":
            json.dumps({
                "ts__NA_FRACTION__False": 0.0,
                "ts__kumo.timestamp__MIN__False": {
                    "value": minimum
                },
                "ts__kumo.timestamp__MAX__False": {
                    "value": maximum
                },
            }),
        }[table]),
    )
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(
        "PREDICT LIST_DISTINCT(orders.product_id, 0, 1, days) "
        "RANK TOP 50 FOR EACH users.id",
        graph,
    )
    ranges = [
        ["2026-05-27", "2026-06-26"],
        ["2026-06-26", "2026-06-27"],
        ["2026-06-27", "2026-06-28"],
    ]

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                split=f"TimeRangeSplit(ranges={ranges!r})",
                end_time=end_time), ),
        run_id="native-t02-range",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    def timestamp_ns(value: str) -> int:
        return int(
            datetime.fromisoformat(value).replace(
                tzinfo=timezone.utc).timestamp() * 1_000_000_000)

    assert plan.train_table.timestamps[:3] == (
        timestamp_ns("2026-06-27"),
        timestamp_ns("2026-06-26"),
        timestamp_ns("2026-06-25"),
    )
    assert plan.train_table.timestamps[-1] == timestamp_ns("2026-05-27")
    assert len(plan.train_table.timestamps) == 32
    assert plan.train_table.split_config == {
        "ranges": [[
            timestamp_ns(start) // 1_000_000_000,
            timestamp_ns(end) // 1_000_000_000
        ] for start, end in ranges],
        "unit":
        "days",
        "timeframe":
        1,
        "anchor_start_offset":
        0,
        "flexible_boundaries":
        False,
    }
    assert plan.task_dataset.required_stages == ("train", "val", "test")


def test_temporal_time_range_split_uses_nondefault_query_offsets(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind:
        ({
            "users": '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products": '{"price__MEAN": 10}',
            "orders": ('{"ts__NA_FRACTION__False": 0.0, "ts__MIN__False": 0, '
                       '"ts__MAX__False": 100}'),
        }[table]),
    )
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(orders.*, 2, 5, hours) FOR EACH users.id",
        graph,
    )

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                split="TimeRangeSplit(ranges=[['2026-06-26', "
                "'2026-06-27'], ['2026-06-27', '2026-06-28']])"), ),
        run_id="native-query-window",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
        timestamps=(1, ),
    )

    assert plan.train_table.split_config["unit"] == "hours"
    assert plan.train_table.split_config["timeframe"] == 3
    assert plan.train_table.split_config["anchor_start_offset"] == 2
    assert plan.train_table.split_config["flexible_boundaries"] is False


@pytest.mark.parametrize(
    ("config_name", "pql", "options", "expected_family", "expected_config"),
    [
        (
            "temporal",
            "PREDICT COUNT(orders.*, 0, 7, days) FOR EACH users.id",
            TrainingTableGenerationPlan(),
            "ApproxDateOffsetSplit",
            {
                "ratio": [0.8, 0.1, 0.1]
            },
        ),
        (
            "forecasting",
            "PREDICT COUNT(events.*, 0, 1, days) FORECAST 3 TIMEFRAMES "
            "FOR EACH articles.aid",
            TrainingTableGenerationPlan(timeframe_step=2),
            "DateOffsetSplit",
            {
                "offsets": [-5, -2],
                "unit": "days",
                "offset_unit": "days",
                "timeframe": 1,
                "anchor_start_offset": 0,
                "flexible_boundaries": True,
                "timeframe_step": 2,
            },
        ),
        (
            "temporal",
            "PREDICT COUNT(orders.*, 2, 7, days) FOR EACH users.id",
            TrainingTableGenerationPlan(timeframe_step=2),
            "DateOffsetSplit",
            {
                "offsets": [-7, 0],
                "unit": "days",
                "offset_unit": "days",
                "timeframe": 5,
                "anchor_start_offset": 2,
                "flexible_boundaries": False,
                "timeframe_step": 2,
            },
        ),
        (
            "temporal",
            "PREDICT LIST_DISTINCT(orders.product_id, 0, 7, days) "
            "RANK TOP 10 FOR EACH users.id",
            TrainingTableGenerationPlan(),
            "DateOffsetSplit",
            {
                "offsets": [-7, 0],
                "unit": "days",
                "offset_unit": "days",
                "timeframe": 7,
                "anchor_start_offset": 0,
                "flexible_boundaries": False,
            },
        ),
    ],
)
def test_inferred_temporal_split_matches_hosted_legacy_defaults(
    config_name,
    pql,
    options,
    expected_family,
    expected_config,
):
    config = (forecasting_graph_config()
              if config_name == "forecasting" else temporal_graph_config())
    graph = normalize_graph(config)
    query = parse_and_validate_query(pql, graph)

    split, family = _split_options(
        query,
        graph,
        ModelPlan(training_table_generation=options),
    )

    assert family == expected_family
    assert split == expected_config


def test_default_split_keeps_two_argument_diagnostic_call_compatible():
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(orders.*, 0, 7, days) FOR EACH users.id",
        graph,
    )

    split, family = _default_split(query, graph)

    assert family == "ApproxDateOffsetSplit"
    assert split == {"ratio": [0.8, 0.1, 0.1]}


@pytest.mark.parametrize(
    "source_days,expected_days",
    [
        (365, (7, 14, 21, 28, 35, 42, 49)),
        (21, (7, 14)),
    ],
)
def test_temporal_plan_honours_train_start_offset_in_query_units(
        tmp_path, monkeypatch, source_days, expected_days):
    index = tmp_path / "index"
    index.mkdir()
    maximum = datetime(2026, 6, 28, tzinfo=timezone.utc)
    minimum = maximum - timedelta(days=source_days)
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users":
            '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products":
            '{"price__MEAN": 10}',
            "orders":
            json.dumps({
                "ts__NA_FRACTION__False": 0.0,
                "ts__kumo.timestamp__MIN__False": {
                    "value": minimum.isoformat()
                },
                "ts__kumo.timestamp__MAX__False": {
                    "value": maximum.isoformat()
                },
            }),
        }[table]),
    )
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(
        "PREDICT LIST_DISTINCT(orders.product_id, 0, 7, days) "
        "RANK TOP 100 FOR EACH users.id",
        graph,
    )

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                train_start_offset=49), ),
        run_id="native-offset-01",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    day_ns = 86_400 * 1_000_000_000
    maximum_ns = int(maximum.timestamp() * 1_000_000_000)
    assert plan.train_table.timestamps == tuple(maximum_ns - days * day_ns
                                                for days in expected_days)
    assert plan.train_table.split_family == "DateOffsetSplit"


def test_plan_rejects_explicit_timestamps_on_a_static_query(tmp_path):
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                     graph)

    with pytest.raises(PredictiveQueryPlanError, match="temporal"):
        build_native_pquery_task_plan(
            query=query,
            graph=graph,
            model_plan=ModelPlan(),
            run_id="native-static-1",
            graph_index_path=str(tmp_path / "index"),
            scratch_dir=str(tmp_path / "scratch"),
            artifact_dir=str(tmp_path / "artifacts"),
            timestamps=(1_659_304_801_388_000_000, ),
        )


@pytest.mark.parametrize("unit", ["days", "hours"])
@pytest.mark.parametrize("options, first_day, last_day", [
    ({}, 1, 39),
    ({
        "train_end_offset": 10
    }, 1, 29),
    ({
        "train_start_offset": 20,
        "train_end_offset": 10
    }, 20, 29),
    ({
        "start_time": -20,
        "end_time": -10
    }, 20, 29),
    ({
        "start_time": "2024-01-11",
        "end_time": "2024-01-21"
    }, 10, 19),
    ({
        "start_time": "2024-01-11T00:00:00Z",
        "end_time": "2024-01-21T00:00:00Z"
    }, 10, 19),
    ({
        "start_time": "-INF",
        "end_time": -10
    }, 1, 29),
    ({
        "train_end_offset": 10,
        "start_time": "2024-01-11",
        "end_time": "2024-01-21"
    }, 0, 29),
])
def test_generation_plan_time_bounds(tmp_path, monkeypatch, options, first_day,
                                     last_day, unit):
    start = 1_704_067_200_000_000_000  # 2024-01-01 UTC
    day = (86_400 if unit == "days" else 3_600) * 1_000_000_000
    if unit == "hours":
        options = {
            key: {
                "2024-01-11": "2024-01-01T10:00:00",
                "2024-01-21": "2024-01-01T20:00:00",
                "2024-01-11T00:00:00Z": "2024-01-01T10:00:00Z",
                "2024-01-21T00:00:00Z": "2024-01-01T20:00:00Z",
            }.get(value, value)
            for key, value in options.items()
        }
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda *args: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": start,
            "event_time__MAX__False": start + 40 * day
        }))
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        f"PREDICT COUNT(events.*, 0, 1, {unit}) FOR EACH articles.aid", graph)
    plan = build_native_pquery_task_plan(
        query=query, graph=graph, model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(**options)),
        run_id="bounds", graph_index_path=str(tmp_path / "index"),
        scratch_dir=str(tmp_path), artifact_dir=str(tmp_path))
    assert plan.train_table.timestamps == tuple(
        start + day * index for index in range(last_day, first_day - 1, -1))


def test_latest_temporal_timestamp_uses_source_maximum(tmp_path, monkeypatch):
    minimum = 1_704_067_200_000_000_000
    maximum = 1_711_843_200_000_000_000
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda *args: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": minimum,
            "event_time__MAX__False": maximum,
        }))
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 7, days) FOR EACH articles.aid", graph)

    assert latest_temporal_timestamp(query, str(tmp_path / "index")) == maximum


@pytest.mark.parametrize("order", [
    ("first", "second"),
    ("second", "first"),
])
def test_latest_temporal_timestamp_is_order_independent(monkeypatch, order):
    maxima = {
        "first": 1_700_000_000_000_000_000,
        "second": 1_800_000_000_000_000_000,
    }

    calls = []

    def stats(_index_path, table_name, _scope):
        calls.append(table_name)
        return json.dumps({
            "ts__NA_FRACTION__False": 0.0,
            "ts__MIN__False": 1_600_000_000_000_000_000,
            "ts__MAX__False": maxima[table_name],
        })

    monkeypatch.setattr("kumo_core.pquery.read_col_stats", stats)
    query = SimpleNamespace(
        query_type="temporal",
        query=SimpleNamespace(
            get_all_target_aggregations=lambda: [
                SimpleNamespace(time_col=f"{table_name}.ts")
                for table_name in order
            ]),
    )

    assert latest_temporal_timestamp(query, "/index") == maxima["second"]
    assert calls == list(order)


def test_generation_plan_keeps_earliest_anchor_needed_for_filter_history(
        tmp_path, monkeypatch):
    from pandas import Timedelta, Timestamp

    origin = Timestamp("2024-01-01")
    day = Timedelta(days=1)
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats", lambda *args: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": origin.value,
            "event_time__MAX__False": (origin + 40 * day).value,
        }))
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 1, days) FOR EACH articles.aid "
        "WHERE COUNT(events.*, -3, 0, days) > 0",
        graph,
    )
    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(),
        run_id="filtered",
        graph_index_path=str(tmp_path / "index"),
        scratch_dir=str(tmp_path),
        artifact_dir=str(tmp_path),
    )

    assert plan.train_table.timestamps[-1] == (origin + 3 * day).value


def test_generation_plan_keeps_its_only_complete_anchor(tmp_path, monkeypatch):
    from pandas import Timedelta, Timestamp

    origin = Timestamp("2024-01-01")
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats", lambda *args: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": origin.value,
            "event_time__MAX__False": (origin + Timedelta(days=1)).value,
        }))
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 1, days) FOR EACH articles.aid",
        graph,
    )
    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(),
        run_id="single-anchor",
        graph_index_path=str(tmp_path / "index"),
        scratch_dir=str(tmp_path),
        artifact_dir=str(tmp_path),
    )

    assert plan.train_table.timestamps == (origin.value, )


@pytest.mark.parametrize("unit", ["days", "hours"])
def test_generation_plan_non_aligned_step_anchors_at_end(
        tmp_path, monkeypatch, unit):
    from pandas import Timedelta, Timestamp

    origin = Timestamp("2024-01-01")
    tick = Timedelta(**{unit: 1})
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats", lambda *args: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False":
            origin.value,
            "event_time__MAX__False": (origin + 40 * tick).value,
        }))
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        f"PREDICT COUNT(events.*, 0, 1, {unit}) FOR EACH articles.aid", graph)
    plan = build_native_pquery_task_plan(
        query=query, graph=graph, model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                start_time=(origin + 10 * tick).isoformat(),
                end_time=(origin + 21 * tick).isoformat(),
                timeframe_step=3,
            )), run_id="non-aligned", graph_index_path=str(tmp_path / "index"),
        scratch_dir=str(tmp_path), artifact_dir=str(tmp_path))

    # Hosted generation includes the newest complete window, then moves back.
    # Starting from the lower bound instead gives offsets 10,13,16,19.
    assert plan.train_table.timestamps == tuple(
        (origin + index * tick).value for index in (20, 17, 14, 11))


def test_forecast_date_offset_split_restarts_step_at_each_boundary(
        tmp_path, monkeypatch):
    from pandas import Timedelta, Timestamp

    origin = Timestamp("2026-01-01")
    day = Timedelta(days=1)
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats", lambda *args: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": origin.value,
            "event_time__MAX__False": (origin + 11 * day).value,
        }))
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 1, days) FORECAST 3 TIMEFRAMES "
        "FOR EACH articles.aid", graph)
    plan = build_native_pquery_task_plan(
        query=query, graph=graph, model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                timeframe_step=2)), run_id="forecast-date-offset-phase",
        graph_index_path=str(tmp_path / "index"), scratch_dir=str(tmp_path),
        artifact_dir=str(tmp_path))

    assert plan.train_table.split_family == "DateOffsetSplit"
    assert plan.train_table.split_config["flexible_boundaries"] is False
    assert plan.train_table.timestamps == tuple(
        (origin + index * day).value for index in (10, 8, 7, 5, 4, 2))


def test_date_offset_split_uses_bounded_range_for_each_phase(
        tmp_path, monkeypatch):
    from pandas import Timedelta, Timestamp

    origin = Timestamp("2026-01-01")
    day = Timedelta(days=1)
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats", lambda *args: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": origin.value,
            "event_time__MAX__False": (origin + 11 * day).value,
        }))
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 1, days) FORECAST 3 TIMEFRAMES "
        "FOR EACH articles.aid", graph)
    plan = build_native_pquery_task_plan(
        query=query, graph=graph, model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                start_time=(origin + day).isoformat(),
                end_time=(origin + 10 * day).isoformat(), timeframe_step=2)),
        run_id="bounded-date-offset-phase",
        graph_index_path=str(tmp_path / "index"), scratch_dir=str(tmp_path),
        artifact_dir=str(tmp_path))

    assert plan.train_table.timestamps == tuple(
        (origin + index * day).value for index in (9, 7, 6, 4, 3, 1))


def test_nonzero_target_offset_restarts_step_at_each_split_boundary(
        tmp_path, monkeypatch):
    from pandas import Timedelta, Timestamp

    origin = Timestamp("2026-01-01")
    day = Timedelta(days=1)
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users": '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products": '{"price__MEAN": 10}',
            "orders": json.dumps({
                "ts__NA_FRACTION__False": 0.0,
                "ts__MIN__False": origin.value,
                "ts__MAX__False": (origin + 40 * day).value,
            }),
        }[table]))
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(orders.*, 2, 7, days) FOR EACH users.id", graph)
    plan = build_native_pquery_task_plan(
        query=query, graph=graph, model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                timeframe_step=2)), run_id="target-offset-phase",
        graph_index_path=str(tmp_path / "index"), scratch_dir=str(tmp_path),
        artifact_dir=str(tmp_path))

    assert plan.train_table.split_family == "DateOffsetSplit"
    assert plan.train_table.split_config["flexible_boundaries"] is False
    assert plan.train_table.timestamps == tuple(
        (origin + index * day).value
        for index in (33, 26, 19, 17, 15, 13, 11, 9, 7, 5, 3))


@pytest.mark.parametrize("pql_horizon, plan_horizon, expected", [
    (None, 4, 4),
    (3, 5, 5),
    (3, None, 3),
    (3, 1, 1),
])
def test_generation_plan_forecast_horizon(tmp_path, monkeypatch, pql_horizon,
                                          plan_horizon, expected):
    from kumo_core.training.workflow import (
        PreparedGraph,
        TrainingWorkflowRequest,
    )
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda *args: json.dumps({"event_time__NA_FRACTION__False": 0.0}))
    clause = (f" FORECAST {pql_horizon} TIMEFRAMES" if pql_horizon else "")
    pql = (f"PREDICT COUNT(events.*, 0, 1, days){clause} "
           "FOR EACH articles.aid")
    request = TrainingWorkflowRequest.from_spec(
        graph=forecasting_graph_config(), predictive_query=pql,
        prepared_graph=PreparedGraph("graph", str(tmp_path / "index")),
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                forecast_length=plan_horizon, lag_timesteps=3,
                year_over_year=True)), run_id="forecast",
        scratch_dir=str(tmp_path), artifact_dir=str(tmp_path),
        timestamps=(1_704_067_200_000_000_000, ))
    assert request.query.task_type == "forecasting"
    assert request.query.forecast_length == expected
    plan = build_native_pquery_task_plan(
        query=parse_and_validate_query(pql, request.graph), graph=request.graph,
        model_plan=request.model_plan, run_id=request.run_id,
        graph_index_path=request.prepared_graph.index_path,
        scratch_dir=request.scratch_dir, artifact_dir=request.artifact_dir,
        timestamps=request.timestamps)
    assert plan.train_table.forecast_length == expected
    assert plan.train_table.compiler_options["future_offsets_secs"] == tuple(
        index * 86_400 for index in range(expected))
    assert plan.task_dataset.task_type == "forecasting"
    columns = plan.task_dataset.feature_schema["node_types"]["train_table"]
    assert columns["feature_columns"] == [
        "future_labels", "autoregression_labels", "year_over_year_labels"
    ]


def test_plan_rejects_a_split_family_portable_training_cannot_run(tmp_path):
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                     graph)

    with pytest.raises(PredictiveQueryPlanError, match="TimeSplit"):
        build_native_pquery_task_plan(
            query=query,
            graph=graph,
            model_plan=ModelPlan(
                training_table_generation=TrainingTableGenerationPlan(
                    split="TimeSplit(fractions=[0.8, 0.1, 0.1])"), ),
            run_id="native-split-2",
            graph_index_path=str(tmp_path / "index"),
            scratch_dir=str(tmp_path / "scratch"),
            artifact_dir=str(tmp_path / "artifacts"),
        )


def test_plan_rejects_an_untyped_model_plan_mapping(tmp_path):
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query("PREDICT users.label FOR EACH users.id",
                                     graph)

    with pytest.raises(TypeError, match="ModelPlan"):
        build_native_pquery_task_plan(
            query=query,
            graph=graph,
            model_plan={"training_table_generation": {
                "lag_timesteps": 2
            }},
            run_id="native-1",
            graph_index_path=str(tmp_path / "index"),
            scratch_dir=str(tmp_path / "scratch"),
            artifact_dir=str(tmp_path / "artifacts"),
        )


def test_forecasting_plan_carries_lag_window_and_diskgraph_offsets(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    day_ns = 86_400 * 1_000_000_000
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, _table, _kind: json.
        dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": 1_659_304_801_388_000_000,
            "event_time__MAX__False": 1_659_304_801_388_000_000 + 35 * day_ns,
        }),
    )
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 1, days) FORECAST 2 TIMEFRAMES "
        "FOR EACH articles.aid",
        graph,
    )

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                lag_timesteps=2), ),
        run_id="native-forecast-1",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert plan.train_table.lag_timesteps == 2
    assert plan.train_table.compiler_options == {
        "future_offsets_secs": (0, 86_400),
        "lag_offsets_secs": (-172_800, -86_400),
    }
    assert plan.task_dataset.task_family == "forecasting"
    train_table_schema = plan.task_dataset.feature_schema["node_types"][
        "train_table"]
    assert train_table_schema["feature_columns"] == [
        "future_labels",
        "autoregression_labels",
    ]
    assert train_table_schema["column_specs"] == [
        {
            "column_name": "future_labels",
            "dtype": "floatlist",
            "stype": "sequence",
            "encoder": {
                "name": "NumericalList"
            },
            "stats": {
                "SEQUENCE_MIN_LENGTH": 2,
                "SEQUENCE_MAX_LENGTH": 2,
            },
        },
        {
            "column_name": "autoregression_labels",
            "dtype": "floatlist",
            "stype": "sequence",
            "encoder": {
                "name": "NumericalList"
            },
            "stats": {
                "SEQUENCE_MIN_LENGTH": 2,
                "SEQUENCE_MAX_LENGTH": 2,
            },
        },
    ]


def test_explicit_timestamps_replace_the_column_stat_derived_window(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    day_ns = 86_400 * 1_000_000_000
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, _table, _kind: json.
        dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": 1_659_304_801_388_000_000,
            "event_time__MAX__False": 1_659_304_801_388_000_000 + 35 * day_ns,
        }),
    )
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 1, days) FORECAST 2 TIMEFRAMES "
        "FOR EACH articles.aid",
        graph,
    )

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(),
        run_id="native-forecast-2",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
        timestamps=(1_659_304_801_388_000_000, ),
    )

    assert plan.train_table.timestamps == (1_659_304_801_388_000_000, )


def test_existing_forecast_rows_build_selected_graph_task_without_planning(
        tmp_path, monkeypatch):
    day_ns = 86_400 * 1_000_000_000
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, _table, _kind: json.dumps({
            "event_time__NA_FRACTION__False": 0.0,
            "event_time__MIN__False": 1_659_304_801_388_000_000,
            "event_time__MAX__False":
            1_659_304_801_388_000_000 + 400 * day_ns,
        }),
    )
    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(events.*, 0, 1, days) FORECAST 2 TIMEFRAMES "
        "FOR EACH articles.aid",
        graph,
    )
    anchors = (1_659_304_801_388_000_000, )
    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                lag_timesteps=3,
                year_over_year=True,
            ), ),
        run_id="generated",
        graph_index_path=str(tmp_path / "old-index"),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
        timestamps=anchors,
    )

    def must_not_plan(*args, **kwargs):
        raise AssertionError("existing rows must not plan table generation")

    monkeypatch.setattr("kumo_core.pquery._temporal_timestamps", must_not_plan)
    supplied = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                lag_timesteps=3,
                year_over_year=True,
            ), ),
        run_id="supplied",
        graph_index_path=str(tmp_path / "selected-index"),
        scratch_dir=str(tmp_path / "supplied-scratch"),
        artifact_dir=str(tmp_path / "supplied-artifacts"),
        supplied_rows=True,
    )
    assert supplied.train_table.timestamps == ()
    assert supplied.train_table.lag_timesteps == 3
    assert "yoy_offsets_secs" in supplied.train_table.compiler_options

    monkeypatch.setattr("kumo_core.pquery._split_options", must_not_plan)
    monkeypatch.setattr("kumo_core.pquery._forecast_options", must_not_plan)

    task_dataset = build_generated_training_task_dataset(
        query=query,
        graph=graph,
        graph_index_path=str(tmp_path / "selected-index"),
        run_id="fit-run",
        artifact_dir=str(tmp_path / "fit-artifacts"),
        tt_id="tt-fit-run",
        required_stages=plan.task_dataset.required_stages,
        lag_timesteps=3,
        year_over_year=True,
    )

    assert task_dataset.required_stages == plan.task_dataset.required_stages
    assert task_dataset.job_id == "fit-run"
    assert task_dataset.artifact_root == str(tmp_path / "fit-artifacts" /
                                             "task_dataset")
    assert task_dataset.tt_manifest_path == str(
        tmp_path / "selected-index/train_tables/tt-fit-run/manifest.json")
    assert task_dataset.feature_schema["node_types"]["train_table"][
        "feature_columns"] == [
            "future_labels",
            "autoregression_labels",
            "year_over_year_labels",
        ]


def test_link_plan_carries_fk_free_entity_and_rhs_resolution(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users": '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products": '{"price__MEAN": 10}',
            "orders": '{"placeholder": 0}',
        }[table]),
    )
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query(
        "PREDICT LIST_DISTINCT(orders.product_id) RANK TOP 3 "
        "FOR EACH users.id",
        graph,
    )

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(),
        run_id="native-link-1",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert plan.train_table.entity_node_type == "users"
    assert plan.train_table.rhs_node_type == "products"
    assert plan.task_dataset.seed_node_type == "users"
    assert plan.task_dataset.rhs_node_type == "products"


def test_plan_rejects_unresolved_query_and_graph_inputs(tmp_path):
    with pytest.raises(TypeError, match="resolved"):
        build_native_pquery_task_plan(
            query="PREDICT users.label FOR EACH users.id",
            graph=graph_config(),
            model_plan=ModelPlan(),
            run_id="native-1",
            graph_index_path=str(tmp_path / "index"),
            scratch_dir=str(tmp_path),
            artifact_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    "graph",
    [
        {
            "tables": {
                "users": {}
            }
        },
        {
            "tables": [{
                "name": "users",
                "source_table_name": "main.schema.users",
            }]
        },
        {
            "tables": [{
                "table_name": "users",
                "source_table_name": "users",
            }]
        },
        {
            "tables": [{
                "table_name": "users",
                "source_table_name": "main.schema.users",
                "source_name": "users",
            }]
        },
        {
            "tables": [{
                "table_name": "users",
                "source_table_name": "main.schema.users",
                "pkey": "id",
            }]
        },
    ],
)
def test_portable_graph_rejects_fallback_and_unresolved_shapes(graph):
    with pytest.raises(PredictiveQueryPlanError):
        parse_and_validate_query(
            "PREDICT users.label FOR EACH users.id",
            graph,
        )


def test_plan_graph_exposes_the_shape_model_plan_validation_reads():
    graph = to_plan_graph(normalize_graph(graph_config()))

    assert "orders" in graph
    assert "missing" not in graph
    with pytest.raises(KeyError):
        graph["missing"]

    users = graph["users"]
    assert users.table_name == "users"
    assert users.pkey.name == "id"
    # Keys are excluded from `cols`, exactly as `TableConfig.cols` excludes
    # them: the metric-validation stype map is built from that list.
    assert [(col.name, col.stype) for col in users.cols] == [
        ("label", Stype.categorical),
        ("age", Stype.numerical),
    ]
    assert users.get_col("age").stype == Stype.numerical
    assert users.get_col("id") is None

    orders = graph["orders"]
    assert orders.cols == ()
    assert [(fkey.name, fkey.dst_table, fkey.dst_key)
            for fkey in orders.fkeys] == [
                ("user_id", "users", "id"),
                ("product_id", "products", "id"),
            ]


def test_table_row_count_reads_the_ingested_count_statistic(monkeypatch):
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users":
            json.dumps({
                "id__kumo.ID__COUNT__False": {
                    "value": 120
                },
                "age__kumo.numerical__COUNT__False": {
                    "value": 120
                },
                "age__kumo.numerical__NUM_NA__False": {
                    "value": 7
                },
            }),
            "orders":
            json.dumps({"id__kumo.ID__COUNT__False": {
                "value": 4800
            }}),
            "products":
            json.dumps({"price__kumo.numerical__MEAN__False": {
                "value": 3
            }}),
        }).get(table),
    )

    assert table_row_count("/index", "users") == 120
    assert table_row_count("/index", "orders") == 4800
    # A table whose statistics carry no `COUNT`, and one the index has never
    # seen, both report no row count rather than a wrong one.
    assert table_row_count("/index", "products") is None
    assert table_row_count("/index", "absent") is None


def temporal_graph_config():
    config = graph_config()
    for table in config["tables"]:
        if table["table_name"] == "orders":
            table["time_col"] = {
                "name": "ts",
                "dtype": "date",
                "stype": "timestamp",
            }
    return config


def shared_destination_graph_config():
    config = temporal_graph_config()
    config["tables"].append({
        "table_name":
        "video_views",
        "source_table_name":
        "main.schema.video_views",
        "pkey": {
            "name": "id",
            "dtype": "int64",
            "stype": "ID",
        },
        "fkeys": [{
            "name": "user_id",
            "dtype": "int64",
            "stype": "ID",
            "dst_table": "users",
            "dst_key": "id",
        }],
        "cols": [],
    })
    return config


def test_real_validator_keeps_all_fact_relationships_to_one_entity():
    query = parse_and_validate_query(
        "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.id",
        shared_destination_graph_config(),
    )

    assert query.task_type == "binary_classification"
    assert query.entity_table == "users"


def test_api_graph_serializes_shared_destination_as_one_column_group():
    graph = to_api_graph(normalize_graph(shared_destination_graph_config()))
    user_group = next(group for group in graph.col_groups
                      if any(key.table_name == "users" and key.col_name == "id"
                             for key in group.columns))

    assert {(key.table_name, key.col_name)
            for key in user_group.columns} == {
                ("users", "id"),
                ("orders", "user_id"),
                ("video_views", "user_id"),
            }


@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 10 "
            "FOR EACH users.id",
            # Link prediction names both edges: the fact table to the seed, then
            # the fact table to the RHS -- `PredictiveQueryV2`'s own order.
            [[("orders", "user_id", "users")],
             [("orders", "product_id", "products")]],
        ),
        (
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.id",
            [[("orders", "user_id", "users")]],
        ),
        (
            # A static task over the entity's own column has no target->entity
            # path at all.
            "PREDICT users.label FOR EACH users.id",
            [],
        ),
    ])
def test_target_to_entity_paths_match_the_predictive_query(query, expected):
    graph = normalize_graph(temporal_graph_config())

    result = target_to_entity_paths(parse_and_validate_query(query, graph),
                                    graph)

    assert result == expected


def test_plan_records_the_target_to_entity_paths_for_materialization(
        tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users":
            '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products":
            '{"price__MEAN": 10}',
            "orders":
            json.dumps({
                "ts__NA_FRACTION__False": 0.0,
                "ts__kumo.timestamp__MIN__False": {
                    "value": 0
                },
                "ts__kumo.timestamp__MAX__False": {
                    "value": 10**18
                },
            }),
        }[table]),
    )
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(
        "PREDICT COUNT(orders.*, 0, 30, days) "
        "FOR EACH users.id", graph)

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(),
        run_id="native-paths-1",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert plan.task_dataset.target_to_entity_paths == ((("orders", "user_id",
                                                          "users"), ), )
    assert plan.train_table.include_frequency is False


def test_temporal_link_prediction_plans_a_supplied_split_column(
        tmp_path, monkeypatch):
    # The combination a supplied training table exists to serve: a temporal
    # link-prediction query whose rows the caller has already placed in
    # train/val/test. The split column has to survive into the resolved
    # plan, not merely past validation.
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, table, _kind: ({
            "users":
            '{"age__MEAN": 42, "label__CATEGORY_COUNTS": []}',
            "products":
            '{"price__MEAN": 10}',
            "orders":
            json.dumps({
                "ts__NA_FRACTION__False": 0.0,
                "ts__kumo.timestamp__MIN__False": {
                    "value": 0
                },
                "ts__kumo.timestamp__MAX__False": {
                    "value": 10**18
                },
            }),
        }[table]),
    )
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(
        "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 10 "
        "FOR EACH users.id", graph)
    assert query.task_type == "temporal_link_prediction"

    plan = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                split="ManualSplit(col='SPLIT', train_value=0, val_value=1, "
                "test_value=2)"), ),
        run_id="native-tlp-manual-1",
        graph_index_path=str(index),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
        timestamps=(10**18, ),
    )

    assert plan.train_table.task_type == "temporal_link_prediction"
    assert plan.train_table.include_frequency is True
    assert plan.task_dataset.target_to_entity_paths == (
        (("orders", "user_id", "users"), ),
        (("orders", "product_id", "products"), ),
    )
    assert plan.train_table.split_family == "ManualSplit"
    assert plan.train_table.split_config == {
        "split_column": "SPLIT",
        "manual_train_values": ["0"],
        "manual_val_values": ["1"],
        "manual_test_values": ["2"],
    }


def test_table_row_count_reads_the_nested_statistics_layout(monkeypatch):
    # `_column_stats` accepts a nested per-column mapping; reading only the
    # flat keys would report "no statistics" for the same artifact.
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, _table, _kind: json.dumps({
            "age": {
                "COUNT": 120,
                "NUM_NA": 3
            },
            "label": {
                "COUNT": 120
            },
        }),
    )

    assert table_row_count("/index", "users") == 120


def test_column_stats_does_not_parse_a_stat_name_out_of_the_column_name():
    stats = {
        "DISTINCT_ITEM_COUNT__kumo.numerical__COUNT__False": {
            "value": 100
        },
        "DISTINCT_ITEM_COUNT__kumo.numerical__NA_FRACTION__False": {
            "value": 0.01
        },
        "DISTINCT_ITEM_COUNT__kumo.numerical__MEAN__False": {
            "value": 2.5
        },
        "DISTINCT_ITEM_COUNT__kumo.numerical__MIN__False": {
            "value": 1
        },
        "DISTINCT_ITEM_COUNT__kumo.numerical__MAX__False": {
            "value": 9
        },
        "DISTINCT_ITEM_COUNT__kumo.numerical__STD__False": {
            "value": 1.2
        },
    }

    assert _column_stats(stats, "DISTINCT_ITEM_COUNT") == {
        "COUNT": 100,
        "NA_FRACTION": 0.01,
        "MEAN": 2.5,
        "MIN": 1,
        "MAX": 9,
        "STD": 1.2,
    }


@pytest.mark.parametrize('source,destination,accepted', [
    ('string', 'int32', False),
    ('int64', 'string', False),
    ('int64', 'int32', False),
    ('int32', 'int64', True),
    ('int', 'int64', True),
    ('int64', 'int', True),
    ('string', 'string', True),
    ('float64', 'int64', False),
    ('int64', 'float64', False),
    ('float32', 'float64', False),
])
def test_graph_key_dtype_validation(source, destination, accepted):
    config = graph_config()
    config['tables'][0]['pkey']['dtype'] = destination
    config['tables'][2]['fkeys'][0]['dtype'] = source
    if accepted:
        normalize_graph(config)
    else:
        with pytest.raises(PredictiveQueryPlanError, match='key'):
            normalize_graph(config)


@pytest.mark.parametrize("value", [
    "RandomSplit([0.6, 0.3, 0.1], seed=17)",
    "TemporalSplit([0.6, 0.3, 0.1])",
])
@pytest.mark.parametrize("array_mode", [False, True])
def test_explicit_static_link_split_uses_hosted_per_link_policy(
        value, array_mode):
    graph = normalize_graph(graph_config())
    query = parse_and_validate_query(
        "PREDICT LIST_DISTINCT(orders.product_id) RANK TOP 3 "
        "FOR EACH users.id", graph)
    query = replace(query, list_distinct_target_is_array=array_mode)
    config, family = _split_options(
        query, graph,
        ModelPlan(training_table_generation=TrainingTableGenerationPlan(
            split=value)))
    assert family == ("RandomLinkSplit" if array_mode else value.split("(")[0])
    assert config["ratio"] == [0.6, 0.3, 0.1]
    # Hosted carries the ratio but drops RandomSplit's entity-level seed.
    if array_mode:
        assert config == {"ratio": [0.6, 0.3, 0.1]}
    elif family == "RandomSplit":
        assert config["rnd_seed"] == 17


STATIC_LINK_PQL = ("PREDICT LIST_DISTINCT(orders.product_id) RANK TOP 3 "
                   "FOR EACH users.id")


def timestamped_users_graph_config(base=graph_config):
    config = base()
    config["tables"][0]["time_col"] = {
        "name": "signup",
        "dtype": "date",
        "stype": "timestamp",
    }
    return config


def test_static_link_prediction_rejects_a_timestamped_entity_table():
    graph = normalize_graph(timestamped_users_graph_config())
    with pytest.raises(PredictiveQueryPlanError,
                       match="time columns users.signup"):
        parse_and_validate_query(STATIC_LINK_PQL, graph)


@pytest.mark.parametrize("config, pql, task_type", [
    (graph_config, STATIC_LINK_PQL, "static_link_prediction"),
    (lambda: timestamped_users_graph_config(temporal_graph_config),
     "PREDICT LIST_DISTINCT(orders.product_id, 0, 7, days) RANK TOP 3 "
     "FOR EACH users.id", "temporal_link_prediction"),
])
def test_link_prediction_entity_time_rule_accepts(config, pql, task_type):
    query = parse_and_validate_query(pql, normalize_graph(config()))
    assert query.task_type == task_type


@pytest.mark.parametrize("config, pql, expected", [
    (graph_config, STATIC_LINK_PQL, TrainTableTime.NONE),
    (lambda: temporal_graph_config(), "PREDICT LIST_DISTINCT("
     "orders.product_id, 0, 7, "
     "days) RANK TOP 3 FOR EACH users.id", TrainTableTime.REAL),
    (graph_config, "PREDICT users.label FOR EACH users.id",
     TrainTableTime.NONE),
    (timestamped_users_graph_config, "PREDICT users.label FOR EACH users.id",
     TrainTableTime.REAL),
])
def test_query_train_table_time_reads_the_query_and_graph(
        config, pql, expected):
    graph = normalize_graph(config())
    query = parse_and_validate_query(pql, graph)
    assert query_train_table_time(query, graph) == expected


@pytest.mark.parametrize("pql", [
    "PREDICT users.label FOR EACH users.id",
    "PREDICT LIST_DISTINCT(orders.product_id, 0, 7, days) "
    "RANK TOP 3 FOR EACH users.id",
])
def test_explicit_random_split_preserved_outside_static_link_prediction(pql):
    graph = normalize_graph(temporal_graph_config())
    query = parse_and_validate_query(pql, graph)
    config, family = _split_options(
        query, graph,
        ModelPlan(training_table_generation=TrainingTableGenerationPlan(
            split="RandomSplit([0.8, 0.2], seed=9)")))
    assert family == "RandomSplit"
    assert config == {"ratio": [0.8, 0.2], "rnd_seed": 9}


@pytest.mark.parametrize("item_table", ["products", "orders"])
def test_static_link_fact_requires_separate_item_table(item_table):
    config = graph_config()
    config["tables"][2]["fkeys"][1]["dst_table"] = item_table
    graph = normalize_graph(config)
    query = parse_and_validate_query(STATIC_LINK_PQL, graph)
    if item_table == "products":
        assert static_link_fact(query.query, query.rhs_node_type) == {
            "table": "orders",
            "entity_col": "user_id",
            "target_col": "product_id",
        }
    else:
        with pytest.raises(ValueError, match="item table distinct"):
            static_link_fact(query.query, query.rhs_node_type)


@pytest.mark.parametrize(
    "maximum,options,window,expected",
    [
        ("2024-04-29", {}, (0, 1), ["2024-03-29", "2024-02-29"]),
        ("2023-04-29", {}, (0, 1), ["2023-03-29", "2023-02-28"]),
        ("2024-06-30", {"timeframe_step": 2}, (0, 1),
         ["2024-05-30", "2024-03-30"]),
        ("2024-06-30", {"start_time": -3, "end_time": -1}, (0, 1),
         ["2024-04-30", "2024-03-30"]),
        ("2024-06-30", {"train_start_offset": 3, "train_end_offset": 1},
         (0, 1), ["2024-04-30", "2024-03-30"]),
        ("2024-06-30", {"start_time": "2024-01-31"}, (1, 2),
         ["2024-04-30", "2024-03-30", "2024-02-29"]),
    ],
)
def test_month_planning_preserves_calendar_windows(
        monkeypatch, maximum, options, window, expected):
    from pandas import Timestamp

    graph = normalize_graph(forecasting_graph_config())
    query = parse_and_validate_query(
        f"PREDICT COUNT(events.*, {window[0]}, {window[1]}, months) "
        "FOR EACH articles.aid", graph)
    minimum = Timestamp(f"{maximum[:4]}-01-01")
    monkeypatch.setattr(
        "kumo_core.pquery._temporal_target_time_range",
        lambda *_args: (minimum, Timestamp(maximum), None))
    timestamps, _ = _temporal_timestamps(
        query, '/synthetic-index', TrainingTableGenerationPlan(**options),
        {}, 'RandomSplit', ())
    assert timestamps == tuple(Timestamp(value).value for value in expected)
