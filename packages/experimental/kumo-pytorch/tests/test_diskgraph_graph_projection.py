# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest

from kumo.pytorch.engines import graph_projection
from kumo.pytorch.engines.graph_projection import GraphProjectionCache, contiguous_node_counts


def test_graph_projection_duplicate_overlapping_colliding_and_null_nodes():
    cache = GraphProjectionCache()
    vec = pa.list_(pa.float32(), 2)
    rows = {0: ([1, 2], 0), 1: (None, 10), 2: ([2, 4], None), 8192: ([8, 9], 20)}
    calls = []

    def read(ids):
        calls.append(ids)
        return {
            "vec": pa.array([rows[i][0] for i in ids], type=vec),
            "key": pa.array([rows[i][1] for i in ids], type=pa.int64()),
        }

    def check(ids):
        out = cache.read("pins", ids, read)
        assert out["vec"].to_pylist() == [rows[i][0] for i in ids]
        assert out["key"].to_pylist() == [rows[i][1] for i in ids]
        return out

    check([8192, 1, 0, 1])
    assert calls == [[0, 1, 8192]]
    check([0, 1])
    assert len(calls) == 1
    check([0, 2, 8192, 1, 2])
    assert calls[-1] == [2, 8192]
    returned = check([8192, 1, 2])
    # A torch tensor sharing this Arrow buffer may be modified by its consumer.
    buf = returned["vec"].values.buffers()[1]
    np.frombuffer(buf, dtype=np.float32)[0] = 99
    check([8192, 2, 1, 8192])
    assert len(calls) == 2


@pytest.mark.parametrize(
    "scalar",
    [
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.float32(),
        pa.float64(),
    ],
)
def test_gathered_vectors_match_arrow_storage_and_have_private_writable_buffers(scalar):
    cache = GraphProjectionCache()
    width = 2

    def read(ids):
        # Include a hidden leading row to exercise nonzero Arrow offsets.
        rows = [[0, 0], *(None if i == 1 else [i, i + 1] for i in ids)]
        return {"vec": pa.array(rows, type=pa.list_(scalar, width)).slice(1)}

    for ids in ([0, 0, 1], [1, 2, 2, 0], [2, 0, 1]):
        gathered = cache.read("table", ids, read)
        values, valid = gathered.vectors["vec"]
        assert values.dtype == scalar.to_pandas_dtype()
        assert values.tolist() == [([0, 0] if i == 1 else [i, i + 1]) for i in ids]
        assert valid.tolist() == [i != 1 for i in ids]
        assert gathered["vec"].to_pylist() == [None if i == 1 else [i, i + 1] for i in ids]
        values.fill(77)
        valid.fill(False)


def test_only_proven_contiguous_physical_shard_ranges_are_eligible():
    def table(name, ranges, maximum, *, stored=True):
        return dict(
            name=name,
            has_node_data=stored,
            max_id=maximum,
            shards=[{"min_id": lo, "max_id": hi} for lo, hi in ranges],
        )

    metadata = {
        "node_types": [
            table("sharded", [(0, 2), (1, 0), (3, 9)], 9),
            table("sparse", [(0, 2), (4, 9)], 9),
            table("overlap", [(0, 5), (5, 9)], 9),
            table("wrong_end", [(0, 2)], 9),
            table("empty", [(1, 0)], 0),
            table("topology_only", [(0, 2)], 2, stored=False),
            table("large", [(0, 32768)], 32768),
        ]
    }
    assert contiguous_node_counts(metadata) == {"sharded": 10}


def test_dense_tables_fetch_only_demanded_rows_and_obey_byte_limit(monkeypatch):
    cache = GraphProjectionCache()
    monkeypatch.setattr(graph_projection, "_CACHE_BYTES", 800)
    monkeypatch.setattr(graph_projection, "_DENSE_BYTES", 400)
    monkeypatch.setattr(graph_projection, "_MAPPED_BYTES", 300)
    calls = []

    def fetch(ids):
        calls.append(ids)
        values = [None if i == 3 else [i, -i] for i in ids]
        return {"vec": pa.array(values, type=pa.list_(pa.int32(), 2))}

    # 20 rows at 18 bytes/row fit within the per-projection bound.
    for ids in ([0], [0, 0], [1], [0]):
        cache.read("pins", ids, fetch, dense_count=20)
    assert calls == [[0], [1]]
    rows = cache.read("pins", [3, 1, 3], fetch, dense_count=20)
    assert calls[-1] == [3]
    assert rows["vec"].to_pylist() == [None, [1, -1], None]
    values, valid = rows.vectors["vec"]
    values.fill(99)
    valid.fill(False)
    rows = cache.read("pins", [1, 3], fetch, dense_count=20)
    assert len(calls) == 3
    assert rows.vectors["vec"][0].tolist() == [[1, -1], [0, 0]]
    assert rows.vectors["vec"][1].tolist() == [True, False]
    # A view with additional physical IDs must leave the contiguous fast path.
    assert cache.read("pins", [20, 0], fetch, dense_count=20)["vec"].to_pylist() == [
        [20, -20],
        [0, 0],
    ]
    assert calls[-1] == [20]
    for key in ("boards", "products"):
        cache.read(key, [0, 1], fetch, dense_count=20)
    assert "pins" not in cache.entries
    cache.read("wide", [0, 1], fetch, dense_count=30)
    cache.read("wide", [1, 5, 8, 10], fetch, dense_count=30)
    assert calls[-1] == [5, 8, 10]
    assert sum(p.nbytes for p in cache.entries.values()) <= 800


def test_mapping_grows_after_conflicts_within_unused_budget(monkeypatch):
    cache = GraphProjectionCache()
    monkeypatch.setattr(graph_projection, "_CACHE_BYTES", 18 * 16)
    monkeypatch.setattr(graph_projection, "_MAPPED_BYTES", 18 * 4)
    calls = []

    def fetch(ids):
        calls.append(ids)
        return {
            "vec": pa.array(
                [None if i == 0 else [i, -i] for i in ids], type=pa.list_(pa.int32(), 2)
            )
        }

    def check(ids):
        out = cache.read("pins", ids, fetch)
        assert out["vec"].to_pylist() == [None if i == 0 else [i, -i] for i in ids]
        out.vectors["vec"][0].fill(99)

    check([0, 1, 2])
    check([0, 1, 1, 2])
    assert len(calls) == 1 and cache.entries["pins"].capacity == 4
    check([5, 1, 5])
    before = cache.entries["pins"].nbytes
    check([5, 0, 2])
    grown = cache.entries["pins"]
    assert grown.capacity == 12 and grown.nbytes + before == graph_projection._CACHE_BYTES
    assert len(calls) == 2  # rehash reads only resident data
    check([1, 0, 13, 5, 2, 1])
    assert calls[-1] == [1, 13]
