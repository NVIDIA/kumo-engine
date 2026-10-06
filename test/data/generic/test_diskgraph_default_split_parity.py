# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kumo_core.training.table import build_split_spec

from kumo.task.split import ApproxDateOffsetSplit, DateOffsetSplit


def _native_membership(tmp_path, timestamps, split, family):
    import diskgraph

    entity_ids = list(range(len(timestamps)))
    users_path = tmp_path / "users.parquet"
    pq.write_table(pa.table({"id": pa.array(entity_ids, type=pa.int64())}),
                   users_path)
    index_path = tmp_path / "index"
    index_path.mkdir()
    diskgraph.ingest(
        json.dumps({
            "dataset_name":
            "default_split_parity",
            "num_shards":
            1,
            "tables": [{
                "name": "users",
                "columns": [{
                    "name": "id",
                    "ingested_dtype": "int64"
                }],
                "primary_key": "id",
                "source": {
                    "type": "parquet",
                    "paths": [str(users_path)],
                },
            }],
        }),
        str(index_path),
        col_stats=True,
        cat_count_strategy="hybrid",
    )

    table_path = tmp_path / "train_table.parquet"
    pq.write_table(
        pa.table({
            "ENTITY1": pa.array(entity_ids, type=pa.int64()),
            "TIMESTAMP": pa.array(timestamps, type=pa.timestamp("s")),
            "TARGET1": pa.array([float(i % 2) for i in entity_ids]),
        }),
        table_path,
    )
    result = diskgraph.build_train_table_artifacts_for_existing_table(
        output_path=str(table_path),
        split_config=build_split_spec(split, family, diskgraph.SplitSpec),
        input_timestamp_unit="seconds",
        tt_id="default-split-parity",
        index_path=str(index_path),
        split_stats_config={
            "entity_table": "users",
            "entity_pk_column": "id",
        },
        train_table_col_stats_config={
            "target_stype": "numerical",
            "time_col": "TIMESTAMP",
        },
        task_type="regression",
    )

    membership = {}
    for stage, stage_id in (("train", 0), ("val", 1), ("test", 2)):
        for path in sorted((tmp_path / "train_table.artifacts" /
                            f"{stage}_seeds").glob("*.parquet")):
            for value in pq.read_table(path,
                                       columns=["TIMESTAMP"
                                                ])["TIMESTAMP"].to_pylist():
                timestamp = int(value.timestamp())
                assert membership.setdefault(timestamp, stage_id) == stage_id
    assert sum(result["split_counts"].values()) == len(timestamps)
    return membership


def _hosted_membership(split, timestamps):
    minimum = pd.Timestamp(min(timestamps), unit="s")
    maximum = pd.Timestamp(max(timestamps), unit="s")
    boundaries = split.get_adjusted_boundaries(maximum, minimum)
    membership = {}
    for timestamp in timestamps:
        value = pd.Timestamp(timestamp, unit="s")
        stages = [
            stage for stage, (start, end) in enumerate(boundaries)
            if value >= start and value + split.timeframe_offset +
            split.anchor_start_date_offset <= end
        ]
        if stages:
            membership[timestamp] = stages[-1]
    return membership


def test_native_approx_default_matches_hosted_skewed_membership(tmp_path):
    timestamps = [10] * 60 + [20] * 25 + [30] * 10 + [40] * 5
    counts = Counter(timestamps)
    hosted = ApproxDateOffsetSplit([0.8, 0.1, 0.1])
    hosted.set_date_offset_boundary(
        train_table_stats=SimpleNamespace(
            num_train_rows=len(timestamps),
            stats_per_timeframe=[
                SimpleNamespace(
                    anchor_date=pd.Timestamp(timestamp, unit="s"),
                    num_train_rows=counts[timestamp],
                    index=index,
                )
                for index, timestamp in enumerate(sorted(counts, reverse=True))
            ],
        ),
        timeframe=10,
        unit="seconds",
        timeframe_step=10,
    )
    assert hosted.split is not None

    native = _native_membership(
        tmp_path,
        timestamps,
        {"ratio": [0.8, 0.1, 0.1]},
        "ApproxDateOffsetSplit",
    )

    assert native == _hosted_membership(hosted.split, timestamps)
    assert native == {10: 0, 20: 0, 30: 1, 40: 2}


def test_native_forecast_default_matches_hosted_flexible_membership(tmp_path):
    timestamps = list(range(11))
    hosted = DateOffsetSplit(
        [-5, -2],
        offset_unit="seconds",
        unit="seconds",
        timeframe=1,
        anchor_start_offset=0,
        timeframe_step=2,
        flexible_boundaries=True,
    )

    native = _native_membership(
        tmp_path,
        timestamps,
        hosted.to_split_config(),
        "DateOffsetSplit",
    )

    assert native == _hosted_membership(hosted, timestamps)
    assert native == {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 2,
        9: 2,
        10: 2,
    }
