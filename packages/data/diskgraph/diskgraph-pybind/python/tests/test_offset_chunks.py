# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph as diskgraph_py
from conftest import _append_portable_rustflags

NUM_ITEMS = 1000


@pytest.mark.parametrize("pad_len,delta_bytes", [(0, 2), (300_000, 3), (17_000_000, 4)])
def test_fetch_features_across_offset_chunks(tmp_path, pad_len, delta_bytes):
    ids = list(range(NUM_ITEMS))
    pad = random.Random(0).randbytes(pad_len // 2).hex()
    pads = [pad if i == 500 else f"p{i}" for i in ids]
    data = tmp_path / "items.parquet"
    pq.write_table(pa.table({
        "item_id": pa.array(ids, type=pa.int64()),
        "score": pa.array([i * 3 + 1 for i in ids], type=pa.int64()),
        "pad": pa.array(pads),
    }), data)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "dataset_name": "offset_chunks",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {"name": "score", "ingested_dtype": "int64"},
                {"name": "pad", "ingested_dtype": "string"},
            ],
            "primary_key": "item_id",
            "source": {"type": "parquet", "paths": [str(data)]},
        }],
    }))
    index = tmp_path / "index"
    subprocess.run(
        ["cargo", "run", "--release", "--bin", "ingest", "--", "--source", str(config),
         "--dest", "local", "--output", str(index)],
        cwd=Path(__file__).resolve().parents[3], check=True,
        env=_append_portable_rustflags(dict(os.environ)),
    )
    [offsets] = list(index.rglob("offsets.bin"))
    assert offsets.read_bytes()[24] == delta_bytes

    dg = diskgraph_py.SamplerEngine(str(index))
    req = [{"node_type": "items", "node_ids": ids, "feature_names": []}]
    rows = pa.Table.from_batches(
        [e["batch"] for e in dg.fetch_features(req) if e["type"] == "feature"]
    ).to_pylist()
    assert len(rows) == NUM_ITEMS
    for row in rows:
        assert row["score"] == row["item_id"] * 3 + 1
        assert row["pad"] == pads[row["item_id"]]
