# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An index clock changes only the stored tick: every clock gives main's results."""

import datetime
import decimal
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph
from diskgraph import (DiskGraphTTBuilder, EdgeRef, RowFilter, SamplerEngine,
                       SplitSpec, TimeUnit, execute_tt_gen)

CLOCKS = ("nanoseconds", "microseconds", "milliseconds", "seconds", "hours", "days")
TICK_NS = {
    "nanoseconds": 1,
    "microseconds": 1_000,
    "milliseconds": 1_000_000,
    "seconds": 1_000_000_000,
    "hours": 3_600_000_000_000,
    "days": 86_400_000_000_000,
}
DAY = 86_400
EPOCH_DAY = 19_700  # 2023-12-09
USERS = 6


def _ingest(index_dir: pathlib.Path, clock: str, tables: list) -> str:
    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "clocks",
            "num_shards": 1,
            "timestamp_unit": clock,
            "tables": tables,
        }),
        output_dir=str(index_dir),
    )
    return str(index_dir)


def _as_nanos(array: pa.Array) -> list:
    unit = array.type.unit
    factor = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[unit]
    return [None if v is None else v * factor for v in array.cast(pa.int64()).to_pylist()]


def _features(index: str, table: str, column: str, num_nodes: int) -> list:
    engine = SamplerEngine(index)
    events = engine.fetch_features([
        {"node_type": table, "node_ids": list(range(num_nodes)), "feature_names": [column]},
    ])
    batches = [e["batch"] for e in events if e["type"] == "feature"]
    return _as_nanos(pa.concat_arrays([b.column(column) for b in batches]))


@pytest.fixture(scope="module")
def whole_day_data(tmp_path_factory) -> pathlib.Path:
    data_dir = tmp_path_factory.mktemp("whole_days")
    pq.write_table(
        pa.table({
            "user_id": pa.array(range(USERS), pa.int64()),
            "signup": pa.array([EPOCH_DAY + u for u in range(USERS)], pa.date32()),
        }),
        data_dir / "users.parquet",
    )
    orders = [(u * 10 + k, u, (EPOCH_DAY + u + 3 * k) * DAY) for u in range(USERS)
              for k in range(5)]
    pq.write_table(
        pa.table({
            "order_id": pa.array([o[0] for o in orders], pa.int64()),
            "user_id": pa.array([o[1] for o in orders], pa.int64()),
            "ordered_at": pa.array([o[2] for o in orders], pa.timestamp("s")),
        }),
        data_dir / "orders.parquet",
    )
    return data_dir


def _whole_day_tables(data_dir: pathlib.Path) -> list:
    return [
        {
            "name": "users",
            "columns": [
                {"name": "user_id", "ingested_dtype": "int64"},
                {"name": "signup", "ingested_dtype": "timestamp"},
            ],
            "primary_key": "user_id",
            "create_time": "signup",
            "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
        },
        {
            "name": "orders",
            "columns": [
                {"name": "order_id", "ingested_dtype": "int64"},
                {"name": "user_id", "ingested_dtype": "int64"},
                {"name": "ordered_at", "ingested_dtype": "timestamp",
                 "semantic_type": "timestamp"},
            ],
            "primary_key": "order_id",
            "create_time": "ordered_at",
            "fkeys": [{
                "src_column": "user_id",
                "dst_table": "users",
                "reverse": {"time": {"table": "orders", "column": "ordered_at",
                                     "unit": "s"}},
            }],
            "source": {"type": "parquet", "paths": [str(data_dir / "orders.parquet")]},
        },
    ]


def _sampled_edges(index: str) -> dict:
    plan = {"layers": [{"fanouts": {}, "default_fanout": {
        "fanout": 100, "mode": "temporal_most_recent", "absent_key_policy": "reject_plan"}}]}
    anchors = [(EPOCH_DAY + 7 + u) * DAY * 10**9 for u in range(USERS)]
    events = SamplerEngine(index, random_seed=0).sample(
        {"users": list(range(USERS))}, json.dumps(plan), fused=True,
        seed_timestamps={"users": anchors}, seed_timestamp_unit=TimeUnit.Nanoseconds)
    edges = []
    for event in events:
        if event["type"] == "edge":
            batch = event["batch"]
            edges += zip(batch.column("id_a").to_pylist(), batch.column("id_b").to_pylist())
    return sorted(edges)


def _training_table(index: str, clock: str, window_days: int = 7) -> list:
    builder = DiskGraphTTBuilder(clock)
    builder.temporal([(EPOCH_DAY + d) * DAY for d in (4, 8, 12, 16)], forecast_length=1,
                     unit="seconds")
    builder.node_prediction()
    builder.for_each("users", "user_id")
    builder.entity_time_col("signup")
    builder.set_target_stype("numerical")
    builder.aggregate(
        "orders_next_week", EdgeRef("users", "user_id", "orders", "user_id"), "count",
        time_col="ordered_at", window=(0, window_days), window_unit="days",
        row_filter=RowFilter.compare(
            "ordered_at", ">", datetime.datetime(2023, 12, 10, tzinfo=datetime.timezone.utc),
            clock),
    )
    builder.split_ratio(SplitSpec(
        kind="TimeRangeSplit",
        ranges=[((EPOCH_DAY + 4) * DAY, (EPOCH_DAY + 9) * DAY),
                ((EPOCH_DAY + 9) * DAY, (EPOCH_DAY + 13) * DAY),
                ((EPOCH_DAY + 13) * DAY, (EPOCH_DAY + 20) * DAY)],
        is_temporal=True,
        timestamp_unit="seconds",
        offset_unit="seconds",
        unit="seconds",
    ))
    generated = execute_tt_gen(builder.build(), index, "clock_tt")
    staging = pathlib.Path(generated["staging_dir"])
    rows = []
    for part in sorted(staging.rglob("*.parquet")):
        table = pq.read_table(part)
        rows += [(part.relative_to(staging).as_posix(), row) for row in table.to_pylist()]
    return sorted(rows, key=repr)


def _timestamp_stats(index: str, clock: str) -> dict:
    stats = json.loads((pathlib.Path(index) / "col_stats.json").read_text())["orders"]
    stats = stats["ordered_at"]
    suffix = {"nanoseconds": "ns", "microseconds": "us", "milliseconds": "ms",
              "seconds": "secs", "hours": "hours", "days": "days"}[clock]
    return {
        "count": stats["count"],
        "null_count": stats["null_count"],
        "min_ns": stats[f"min_{suffix}"] * TICK_NS[clock],
        "max_ns": stats[f"max_{suffix}"] * TICK_NS[clock],
    }


def _whole_day_results(data_dir: pathlib.Path, index_dir: pathlib.Path, clock: str) -> dict:
    index = _ingest(index_dir, clock, _whole_day_tables(data_dir))
    return {
        "orders": _features(index, "orders", "ordered_at", USERS * 5),
        "signups": _features(index, "users", "signup", USERS),
        "sampled": _sampled_edges(index),
        "training_table": _training_table(index, clock),
        "stats": _timestamp_stats(index, clock),
    }


@pytest.mark.parametrize("clock", [c for c in CLOCKS if c != "seconds"])
def test_every_clock_matches_the_seconds_index_on_whole_days(whole_day_data, tmp_path,
                                                             clock):
    seconds = _whole_day_results(whole_day_data, tmp_path / "seconds", "seconds")
    assert _whole_day_results(whole_day_data, tmp_path / clock, clock) == seconds


def test_index_metadata_and_config_record_every_clock(whole_day_data, tmp_path):
    for clock in CLOCKS:
        index = _ingest(tmp_path / clock, clock, _whole_day_tables(whole_day_data))
        metadata = json.loads(diskgraph.read_index_metadata(index))
        assert metadata["timestamp_unit"] == clock
        config = json.loads((pathlib.Path(index) / "diskgraph_config.json").read_text())
        assert config["timestamp_unit"] == clock


# Instant 2024-01-15 10:00:00.750 UTC, written in each source representation.
INSTANT_NS = 1_705_312_800_750_000_000


def _floor(ns: int, clock: str) -> int:
    return ns // TICK_NS[clock]


def _ceil(ns: int, clock: str) -> int:
    return -(-ns // TICK_NS[clock])


def _decimal(ns: int, unit_ns: int, clock: str) -> int:
    step = min(unit_ns, TICK_NS[clock])
    truncated = ns // step * step if ns >= 0 else -(-ns // step * step)
    return _ceil(truncated, clock)


@pytest.fixture(scope="module")
def messy_data(tmp_path_factory) -> pathlib.Path:
    data_dir = tmp_path_factory.mktemp("messy")
    pq.write_table(
        pa.table({
            "event_id": pa.array([0, 1, 2], pa.int64()),
            "text_with_format": ["2024-01-15 10:00:00.750", None, "not a time"],
            "text_without_format": ["2024-01-15T10:00:00.750Z", "2024-01-15T10:00:00",
                                    "2024-01-15 10:00:00"],
            "epoch_millis": pa.array([1_705_312_800_750, None, -1], pa.int64()),
            "epoch_decimal": pa.array([decimal.Decimal("1705312800.75"), None,
                                       decimal.Decimal("-0.75")], pa.decimal128(20, 2)),
            "epoch_millis_decimal": pa.array([decimal.Decimal("1705312800750.5"), None, None],
                                             pa.decimal128(20, 1)),
            "typed_nanos": pa.array([INSTANT_NS, INSTANT_NS + 1, None],
                                    pa.timestamp("ns")),
            "date": pa.array([19_737, 29_999, 2_932_896], pa.date32()),
        }),
        data_dir / "events.parquet",
    )
    return data_dir


def _messy_tables(data_dir: pathlib.Path) -> list:
    timestamp = {"ingested_dtype": "timestamp"}
    return [{
        "name": "events",
        "columns": [
            {"name": "event_id", "ingested_dtype": "int64"},
            {"name": "text_with_format", **timestamp,
             "timestamp_format": "%Y-%m-%d %H:%M:%S%.f"},
            {"name": "text_without_format", **timestamp},
            {"name": "epoch_millis", **timestamp, "timestamp_unit": "millis"},
            {"name": "epoch_decimal", **timestamp},
            {"name": "epoch_millis_decimal", **timestamp, "timestamp_unit": "millis"},
            {"name": "typed_nanos", **timestamp},
            {"name": "date", **timestamp},
        ],
        "primary_key": "event_id",
        "source": {"type": "parquet", "paths": [str(data_dir / "events.parquet")]},
    }]


@pytest.mark.parametrize("clock", CLOCKS)
def test_messy_values_follow_mains_rules_in_every_clock(messy_data, tmp_path, clock):
    index = _ingest(tmp_path / clock, clock, _messy_tables(messy_data))
    jan_15 = 19_737 * DAY * 10**9
    expected = {
        # Text truncates; epochs and typed values round up; decimals truncate to
        # the finer of their unit and the clock, then round up.
        "text_with_format": [_floor(INSTANT_NS, clock), None, None],
        "text_without_format": [_floor(INSTANT_NS, clock), None, _floor(jan_15 + 36_000 * 10**9, clock)],
        "epoch_millis": [_ceil(INSTANT_NS, clock), None, _ceil(-10**6, clock)],
        "epoch_decimal": [_decimal(INSTANT_NS, 10**9, clock), None,
                          _decimal(-750_000_000, 10**9, clock)],
        "epoch_millis_decimal": [_decimal(INSTANT_NS + 500_000, 10**6, clock), None, None],
        "typed_nanos": [_ceil(INSTANT_NS, clock), _ceil(INSTANT_NS + 1, clock), None],
        # Dates outside main's range (after 2038-01-19) are null.
        "date": [_floor(jan_15, clock), None, None],
    }
    for column, ticks in expected.items():
        decoded = _features(index, "events", column, 3)
        # Hours and days decode as Arrow seconds, which are exact multiples.
        assert [None if v is None else _floor(v, clock) for v in decoded] == ticks, column


def test_values_beyond_the_clock_are_null_and_drop_required_rows(tmp_path):
    year_9999 = 253_402_300_799
    pq.write_table(
        pa.table({
            "event_id": pa.array([0, 1], pa.int64()),
            "happened_at": pa.array([1_705_312_800, 1_705_312_800], pa.timestamp("s")),
            "noted": ["9999-12-31 23:59:59", "2024-01-15 10:00:00"],
            "expires_at": pa.array([year_9999, 1_705_312_800], pa.timestamp("s")),
        }),
        tmp_path / "events.parquet",
    )
    timestamp = {"ingested_dtype": "timestamp"}
    tables = [{
        "name": "events",
        "columns": [
            {"name": "event_id", "ingested_dtype": "int64"},
            {"name": "happened_at", **timestamp},
            {"name": "noted", **timestamp},
            {"name": "expires_at", **timestamp},
        ],
        "primary_key": "event_id",
        "source": {"type": "parquet", "paths": [str(tmp_path / "events.parquet")]},
    }]
    index = _ingest(tmp_path / "ns", "nanoseconds", tables)
    assert _features(index, "events", "noted", 2) == [None, 1_705_312_800 * 10**9]
    assert _features(index, "events", "expires_at", 2) == [None, 1_705_312_800 * 10**9]

    tables[0]["create_time"] = "expires_at"
    index = _ingest(tmp_path / "ns_required", "nanoseconds", tables)
    nodes = json.loads(diskgraph.read_index_metadata(index))["node_types"]
    assert [node["max_id"] for node in nodes if node["name"] == "events"] == [0]


def test_rows_whose_window_leaves_the_clock_are_skipped(whole_day_data, tmp_path):
    index = _ingest(tmp_path / "ns", "nanoseconds", _whole_day_tables(whole_day_data))
    last_day_in_clock = (2**63 - 1) // (DAY * 10**9)
    window_days = last_day_in_clock - (EPOCH_DAY + 8)
    rows = _training_table(index, "nanoseconds", window_days)
    anchors = {row["TIMESTAMP"].date() for part, row in rows if part.startswith("tt.parquet")}
    assert anchors == {datetime.date(2023, 12, 13), datetime.date(2023, 12, 17)}
