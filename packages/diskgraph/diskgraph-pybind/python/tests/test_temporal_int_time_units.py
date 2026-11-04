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


def encode(seconds, hint):
    if "timestamp_format" in hint:
        return int(datetime.fromtimestamp(seconds, UTC).strftime("%Y%m%d"))
    scale = {"seconds": 1, "millis": 10**3, "micros": 10**6, "nanos": 10**9}
    return seconds * scale[hint["timestamp_unit"]]


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
    pq.write_table(pa.table({"id": [1]}), tmp_path / "users.parquet")
    pq.write_table(
        pa.table(
            {
                "id": list(range(1, len(times) + 1)),
                "owner": [1] * len(times),
                "time": pa.array(times, type=pa.timestamp("s")),
                "event_time": pa.array([encode(t, hint) for t in times], pa.int64()),
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
                    "columns": ["id"],
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
            "aggregates": [
                {
                    "name": "r",
                    "frame": "f",
                    "operation": "recency",
                    "column": "event_time",
                }
            ],
        }
    )
    roots = {"users": engine.lookup_node_ids("users", pa.array([1], pa.int64()))}
    cutoffs = {"users": [CUTOFF]}

    result = engine.query_temporal(roots, cutoffs, plan, output_format="inspectable")
    lengths = engine.query_temporal_lengths(roots, cutoffs, plan)

    sequence = result["sequences"][0]
    assert sequence["batch"].column("id").to_pylist() == EXPECTED_IDS
    assert list(sequence["timestamps"]) == [times[i - 1] for i in EXPECTED_IDS]
    assert lengths["sequences"][0]["lengths"].to_pylist() == [len(EXPECTED_IDS)]
    path = "sort_key_timestamps" if sort_key else "rowpack_timestamps"
    assert lengths["stats"][path] == len(times)
    assert list(result["aggregates"][0]["values"]) == [0.0]
