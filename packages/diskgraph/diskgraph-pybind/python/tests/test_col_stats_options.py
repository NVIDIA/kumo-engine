# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq

SPECS = json.dumps([
    {"name": "SCORE", "stype": "numerical", "dtype": "float64"},
    {"name": "CLASS", "stype": "categorical", "dtype": "string"},
])


def test_prediction_options_through_python(tmp_path):
    path = tmp_path / "part-00000.parquet"
    pq.write_table(
        pa.table({
            "ENTITY": [1, 1, 2, 2, 3, 3],
            "SCORE": [0.9, 0.2, 0.7, 0.6, 0.4, 0.1],
            "CLASS": ["a", "b", "a", "c", "b", "a"],
        }), path, row_group_size=3)

    full, _ = diskgraph.compute_col_stats_for_parquet(
        [str(path)], SPECS, "v1", percentiles=True,
        summary_stats_by_class=False, histograms=False,
        category_counts_limit=1)
    stats = json.loads(full)
    assert len(stats["SCORE__kumo.numerical__PERCENTILES__True"]["value"]) == 99
    assert stats["CLASS__kumo.categorical__CATEGORY_COUNTS__False"][
        "value"] == [["a", 3]]
    assert stats["CLASS__kumo.categorical__CATEGORY_COUNTS_CAPPED__False"][
        "value"] is False
    assert not any("HISTOGRAM" in key or "BY_CLASS" in key for key in stats)

    full, _ = diskgraph.compute_col_stats_for_parquet(
        [str(path)], json.dumps(json.loads(SPECS)[:1]), "v1",
        first_row_per=["ENTITY"])
    stats = json.loads(full)
    assert stats["SCORE__kumo.numerical__COUNT__False"]["value"] == 3
    assert stats["SCORE__kumo.numerical__MIN__False"]["value"] == 0.4


def test_default_output_is_unchanged(tmp_path):
    path = tmp_path / "part-00000.parquet"
    pq.write_table(
        pa.table({
            "SCORE": [0.9, 0.2, 0.7, 0.6, 0.4, 0.1, 0.35, 0.8],
            "CLASS": ["a", "b", "a", "c", "b", "a", "d", "a"],
            "AT": pa.array([1_600_000_000 + 86_400 * i for i in range(8)],
                           pa.timestamp("s")),
        }), path)
    specs = json.dumps(json.loads(SPECS) + [
        {"name": "AT", "stype": "timestamp", "dtype": "date"}])

    full, min_ = diskgraph.compute_col_stats_for_parquet([str(path)], specs,
                                                         "v1")

    expected = ("7e7c026d875b5eb33be64227ee539fa4"
                "fa558d0618f794a7ae3ddb25c6764d9a")
    assert hashlib.sha256(full.encode()).hexdigest() == expected
    assert hashlib.sha256(min_.encode()).hexdigest() == expected
