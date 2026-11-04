# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integer time columns are read in their declared unit or format."""

import json
from datetime import UTC, datetime

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

CUTOFF = int(datetime(2024, 3, 10, tzinfo=UTC).timestamp())
DAY = 86_400
# Seconds before the cutoff; the window keeps (-3 days, 0].
OFFSETS = [5 * DAY, 2 * DAY, DAY, 0, -DAY]
EXPECTED_IDS = [4, 3, 2]


def encode_seconds(value, hint):
    if "timestamp_format" in hint:
        return int(
            datetime.strptime(str(value), "%Y%m%d").replace(tzinfo=UTC).timestamp()
        )
    scale = {"seconds": 1, "millis": 10**3, "micros": 10**6, "nanos": 10**9}
    return value // scale[hint["timestamp_unit"]]


def encode(seconds, hint):
    if "timestamp_format" in hint:
        return int(datetime.fromtimestamp(seconds, UTC).strftime("%Y%m%d"))
    scale = {"seconds": 1, "millis": 10**3, "micros": 10**6, "nanos": 10**9}
    return seconds * scale[hint["timestamp_unit"]]


def open_engine(tmp_path, event_times, hint, sort_key):
    """`event_times` are raw `event_time` values; `time` is their floored second."""
    seconds = [encode_seconds(value, hint) for value in event_times]
    pq.write_table(pa.table({"id": [1]}), tmp_path / "users.parquet")
    pq.write_table(
        pa.table(
            {
                "id": list(range(1, len(event_times) + 1)),
                "owner": [1] * len(event_times),
                "time": pa.array(seconds, type=pa.timestamp("s")),
                "event_time": pa.array(event_times, pa.int64()),
            }
        ),
        tmp_path / "events.parquet",
    )
    config = {
        "dataset_name": "int_time_units",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "source": {
                    "type": "parquet",
                    "paths": [str(tmp_path / "users.parquet")],
                },
            },
            {
                "name": "events",
                "primary_key": "id",
                "create_time": "time",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "owner", "ingested_dtype": "int64"},
                    {"name": "time", "ingested_dtype": "timestamp"},
                    {"name": "event_time", "ingested_dtype": "int64", **hint},
                ],
                "fkeys": [
                    {
                        "src_column": "owner",
                        "dst_table": "users",
                        "reverse": {
                            "time": {
                                "table": "events",
                                "column": "event_time" if sort_key else "time",
                                "unit": "s",
                            }
                        },
                    }
                ],
                "source": {
                    "type": "parquet",
                    "paths": [str(tmp_path / "events.parquet")],
                },
            },
        ],
    }
    diskgraph.ingest(
        json.dumps(config),
        str(tmp_path / "index"),
        col_stats=False,
        context_threads=1,
        encode_threads=1,
        pk_map_threads=1,
        edges_map_threads=1,
        index_threads=1,
    )
    engine = diskgraph.SamplerEngine(str(tmp_path / "index"))
    relation = next(
        relation["name"]
        for relation in json.loads(engine.get_type_mapping())["edge_relations"]
        if relation["src_type"] == "users"
    )
    return engine, relation


def run(engine, relation, aggregates=()):
    plan = json.dumps(
        {
            "frames": [
                {
                    "name": "f",
                    "root_type": "users",
                    "path": [{"relation": relation, "destination_type": "events"}],
                    "time_column": "event_time",
                    "start_seconds": -3 * DAY,
                    "stop_seconds": 0,
                    "distinct": True,
                    "columns": ["id", "owner"],
                }
            ],
            "sequences": [
                {
                    "name": "s",
                    "frame": "f",
                    "columns": ["id"],
                    "order": "descending",
                    "limit": None,
                }
            ],
            "aggregates": [{"frame": "f", **aggregate} for aggregate in aggregates],
        }
    )
    roots = {"users": engine.lookup_node_ids("users", pa.array([1], pa.int64()))}
    cutoffs = {"users": [CUTOFF]}
    result = engine.query_temporal(roots, cutoffs, plan, output_format="inspectable")
    lengths = engine.query_temporal_lengths(roots, cutoffs, plan)
    return result, lengths


@pytest.mark.parametrize("sort_key", [True, False], ids=["sort_key", "rowpack"])
@pytest.mark.parametrize(
    "hint",
    [
        {"timestamp_unit": "seconds"},
        {"timestamp_unit": "millis"},
        {"timestamp_unit": "micros"},
        {"timestamp_unit": "nanos"},
        {"timestamp_format": "%Y%m%d"},
    ],
    ids=["s", "ms", "us", "ns", "yyyymmdd"],
)
def test_integer_time_column_uses_declared_unit(tmp_path, hint, sort_key):
    times = [CUTOFF - offset for offset in OFFSETS]
    engine, relation = open_engine(
        tmp_path, [encode(t, hint) for t in times], hint, sort_key
    )
    recency = {"name": "r", "operation": "recency", "column": "event_time"}
    result, lengths = run(engine, relation, [recency])

    sequence = result["sequences"][0]
    assert sequence["batch"].column("id").to_pylist() == EXPECTED_IDS
    assert list(sequence["timestamps"]) == [times[i - 1] for i in EXPECTED_IDS]
    assert lengths["sequences"][0]["lengths"].to_pylist() == [len(EXPECTED_IDS)]
    path = "sort_key_timestamps" if sort_key else "rowpack_timestamps"
    assert lengths["stats"][path] == len(times)
    assert list(result["aggregates"][0]["values"]) == [0.0]


@pytest.mark.parametrize("sort_key", [True, False], ids=["sort_key", "rowpack"])
def test_sub_second_times_round_up_like_ingestion(tmp_path, sort_key):
    lower = (CUTOFF - 3 * DAY) * 1000
    upper = CUTOFF * 1000
    event_times = [lower - 400, lower, lower + 400, upper - 400, upper + 400]
    engine, relation = open_engine(
        tmp_path, event_times, {"timestamp_unit": "millis"}, sort_key
    )
    result, lengths = run(engine, relation)

    sequence = result["sequences"][0]
    assert sequence["batch"].column("id").to_pylist() == [4, 3]
    assert list(sequence["timestamps"]) == [CUTOFF, CUTOFF - 3 * DAY + 1]
    assert lengths["sequences"][0]["lengths"].to_pylist() == [2]
    path = "sort_key_timestamps" if sort_key else "rowpack_timestamps"
    assert lengths["stats"][path] == len(event_times)


def test_grouped_recency_uses_declared_unit(tmp_path):
    hint = {"timestamp_unit": "millis"}
    times = [CUTOFF - offset for offset in OFFSETS]
    engine, relation = open_engine(
        tmp_path, [encode(t, hint) for t in times], hint, sort_key=False
    )
    count = {"name": "n", "operation": "count", "column": None}
    latest = {"name": "m", "operation": "max", "column": "event_time"}
    result, _ = run(
        engine,
        relation,
        [
            {
                "name": "by_time",
                "groups": [{"keys": ["event_time"], "values": [count]}],
                "operation": "recency",
                "column": "event_time",
            },
            {
                "name": "of_max",
                "groups": [{"keys": ["owner"], "values": [latest]}],
                "operation": "recency",
                "column": "m",
            },
            {
                "name": "grouped",
                "groups": [
                    {
                        "keys": ["owner"],
                        "values": [{**latest, "name": "r", "operation": "recency"}],
                    }
                ],
                "operation": "max",
                "column": "r",
            },
            {
                "name": "two_stages",
                "groups": [
                    {"keys": ["id", "owner"], "values": [latest]},
                    {"keys": ["owner"], "values": [{**latest, "column": "m"}]},
                ],
                "operation": "recency",
                "column": "m",
            },
            {
                "name": "overwritten_key",
                "groups": [
                    {
                        "keys": ["event_time"],
                        "values": [{**count, "name": "event_time"}],
                    }
                ],
                "operation": "recency",
                "column": "event_time",
            },
            {"name": "max", "operation": "max", "column": "event_time"},
        ],
    )

    values = {item["name"]: list(item["values"]) for item in result["aggregates"]}
    assert values == {
        "by_time": [0.0],
        "of_max": [0.0],
        "grouped": [0.0],
        "two_stages": [0.0],
        "overwritten_key": [float(CUTOFF - 1)],
        "max": [encode(CUTOFF, hint)],
    }
