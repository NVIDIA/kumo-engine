# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise an installed wheel's default mmap reader on every supported OS."""

import json
import os
import sys

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope="module")
def mmap_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("mmap_portability")
    parquet = root / "users.parquet"
    pq.write_table(pa.table({
        "id": pa.array(range(8), type=pa.int64()),
        "parent": pa.array([None] + [0] * 7, type=pa.int64()),
        "score": pa.array([index * 0.5 for index in range(8)]),
    }), parquet)
    config = {
        "dataset_name": "mmap_portability",
        "num_shards": 1,
        "tables": [{
            "name": "users",
            "columns": [
                {"name": "id", "ingested_dtype": "int64"},
                {"name": "parent", "ingested_dtype": "int64"},
                {"name": "score", "ingested_dtype": "float64"},
            ],
            "primary_key": "id",
            "fkeys": [{"src_column": "parent", "dst_table": "users", "reverse": {}}],
            "source": {"type": "parquet", "paths": [str(parquet)]},
        }],
    }
    index = str(root / "index")
    with pytest.MonkeyPatch.context() as environment:
        environment.delenv("DG_EDGE_BACKEND", raising=False)
        diskgraph.ingest(json.dumps(config), index, col_stats=False)
    return index


@pytest.mark.parametrize("backend", (None, "mmap"))
def test_installed_mmap_wheel(mmap_graph, monkeypatch, backend):
    assert diskgraph._c.__file__.startswith(sys.prefix + os.sep), diskgraph._c.__file__
    if backend is None:
        monkeypatch.delenv("DG_EDGE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    engine = diskgraph.SamplerEngine(mmap_graph, random_seed=42, num_threads=1)
    plan = {"layers": [{
        "fanouts": {},
        "default_fanout": {
            "fanout": 4,
            "mode": "uniform",
            "absent_key_policy": "uniform_fallback",
        },
    }]}
    events = list(engine.sample({"users": [0]}, json.dumps(plan), fused=True))
    stats = json.loads(next(event["stats"] for event in events if event["type"] == "stats"))
    assert stats["total_edges"] == 4
    features = next(event["batch"] for event in events if event["type"] == "feature")
    assert features.num_rows == 5
    assert "score" in features.schema.names


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="Linux supports native explicit I/O")
@pytest.mark.parametrize("backend", ("buffered", "direct"))
def test_unsupported_backend(mmap_graph, monkeypatch, backend):
    monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    with pytest.raises(diskgraph.DiskgraphSystemError, match="requires Linux io_uring"):
        diskgraph.SamplerEngine(mmap_graph)


@pytest.mark.parametrize("backend", ("", "MMAP", "Direct", " buffered", "direct ", "unknown"))
def test_invalid_backend(mmap_graph, monkeypatch, backend):
    monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    with pytest.raises(diskgraph.DiskgraphSystemError, match="DG_EDGE_BACKEND"):
        diskgraph.SamplerEngine(mmap_graph)


@pytest.mark.skipif(os.name != "posix", reason="requires Unix byte environment values")
def test_invalid_backend_bytes(mmap_graph, monkeypatch):
    monkeypatch.setitem(os.environb, b"DG_EDGE_BACKEND", b"\xff")
    with pytest.raises(diskgraph.DiskgraphSystemError, match="DG_EDGE_BACKEND.*valid UTF-8"):
        diskgraph.SamplerEngine(mmap_graph)
