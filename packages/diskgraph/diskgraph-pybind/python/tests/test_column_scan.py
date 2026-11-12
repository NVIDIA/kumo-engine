# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import random
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph

T0 = 1_700_000_000
CUTOFF = T0 + 500


def _ingest(data_dir: pathlib.Path, out: pathlib.Path, threads: int, shards: int) -> str:
    def column(name, dtype):
        return {"name": name, "ingested_dtype": dtype}

    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "scan",
            "num_shards": shards,
            "timestamp_unit": "seconds",
            "tables": [
                {"name": "docs", "primary_key": "id", "create_time": "ts",
                 "columns": [column("id", "int64"), column("ts", "timestamp"),
                             column("body", "string"), column("n", "int64")],
                 "source": {"type": "parquet", "paths": [str(data_dir / "docs.parquet")]}},
                {"name": "tags", "primary_key": "id",
                 "columns": [column("id", "int64"), column("label", "string")],
                 "source": {"type": "parquet", "paths": [str(data_dir / "tags.parquet")]}},
            ],
        }),
        output_dir=str(out),
        encode_threads=threads,
        index_threads=threads,
    )
    return str(out)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> pathlib.Path:
    rng = random.Random(0)
    root = tmp_path_factory.mktemp("scan_data")
    ids = list(range(2_000)) + [5]  # a duplicate key is stored too
    ts = [T0 + rng.randrange(-1_000, 1_000) for _ in ids]
    ts[1:4] = [CUTOFF, CUTOFF + 1, None]
    body = [None if i % 9 == 0 else f"doc {i} " + "é" * (i % 5) for i in ids]
    body[1:3] = ["at #", "after %"]
    pq.write_table(pa.table({
        "id": pa.array(ids, pa.int64()),
        "ts": pa.array(ts, pa.timestamp("s")),
        "body": pa.array(body, pa.string()),
        "n": pa.array(ids, pa.int64()),
    }), root / "docs.parquet", row_group_size=300)
    pq.write_table(pa.table({
        "id": pa.array(range(500), pa.int64()),
        "label": pa.array([f"tag{i % 17}" for i in range(500)], pa.string()),
    }), root / "tags.parquet")
    return root


@pytest.fixture(scope="module")
def indexes(data_dir, tmp_path_factory) -> dict:
    return {(t, s): _ingest(data_dir, tmp_path_factory.mktemp(f"index_t{t}_s{s}"), t, s)
            for t in (1, 4) for s in (1, 3)}


def _stored(index: str, table: str, columns: list) -> dict:
    metadata = json.loads((pathlib.Path(index) / "index_metadata.json").read_text())
    nodes = next(n["max_id"] + 1 for n in metadata["node_types"] if n["name"] == table)
    events = diskgraph.SamplerEngine(index).fetch_features(
        [{"node_type": table, "node_ids": list(range(nodes)), "feature_names": columns}])
    batches = [e["batch"] for e in events if e["type"] == "feature"]
    return {c: _plain(pa.concat_arrays([b.column(c) for b in batches])) for c in columns}


def _plain(array: pa.Array) -> list:
    if pa.types.is_timestamp(array.type):
        array = array.cast(pa.timestamp("s")).cast(pa.int64())
    return array.to_pylist()


def _scan(index, table, column, cutoff, **kwargs) -> list:
    chunks = list(diskgraph.scan_column(index, table, column, cutoff, **kwargs))
    chunk_rows = kwargs.get("chunk_rows", 65536)
    assert all(c.type == pa.large_string() and c.null_count == 0 for c in chunks)
    assert all(len(c) == chunk_rows for c in chunks[:-1])
    assert all(0 < len(c) <= chunk_rows for c in chunks[-1:])
    return [v for c in chunks for v in c.to_pylist()]


def test_scan_yields_stored_values_in_order(data_dir, indexes):
    table = pq.read_table(data_dir / "docs.parquet")
    source = {c: _plain(table[c].combine_chunks()) for c in ("body", "ts")}
    for index in indexes.values():
        stored = _stored(index, "docs", ["body", "ts"])
        for cutoff in (CUTOFF, None):
            expected = [b for b, t in zip(stored["body"], stored["ts"])
                        if b is not None and (cutoff is None or t <= cutoff)]
            kept = [b for b, t in zip(source["body"], source["ts"])
                    if b is not None and t is not None and (cutoff is None or t <= cutoff)]
            assert sorted(expected) == sorted(kept)
            for threads in (1, 4):
                for chunk_rows in (1, 7, 65536):
                    assert _scan(index, "docs", "body", cutoff, chunk_rows=chunk_rows,
                                 num_threads=threads) == expected
    first, last = source["body"][1:3]
    boundary = _scan(indexes[1, 1], "docs", "body", CUTOFF)
    assert first in boundary and last not in boundary
    request = {"cutoffs": [CUTOFF], "dirs": ["counts"], "columns": {"docs": {"body": "categorical"}}}
    diskgraph.cutoff_col_stats(indexes[1, 1], json.dumps(request))
    counted = json.loads((pathlib.Path(indexes[1, 1]) / "col_stats/counts/col_stats.json").read_text())
    assert len(boundary) == counted["docs"]["body"]["count"] - counted["docs"]["body"]["null_count"]
    assert _scan(indexes[1, 1], "docs", "body", T0 - 10_000) == []


def test_non_temporal_table_yields_every_row(indexes):
    index = indexes[4, 3]
    assert _scan(index, "tags", "label", CUTOFF) == _stored(index, "tags", ["label"])["label"]


def test_rejects_non_string_columns(indexes):
    with pytest.raises(Exception, match="not strings"):
        diskgraph.scan_column(indexes[1, 1], "docs", "n", None)
    with pytest.raises(Exception, match="no column"):
        diskgraph.scan_column(indexes[1, 1], "docs", "missing", None)
    with pytest.raises(Exception, match="num_threads must be positive"):
        diskgraph.scan_column(indexes[1, 1], "docs", "body", None, num_threads=0)


def _train(index, cutoff, columns=(("docs", "body"),), **kwargs) -> str:
    return diskgraph.train_tokenizer(index, list(columns), cutoff, vocab_size=300, min_frequency=0,
                                     special_tokens=["[MISSING]", "[UNK]"], unk_token="[UNK]",
                                     **kwargs)


def test_tokenizer_learns_the_rows_counted_at_the_cutoff(indexes):
    for cutoff, seen in ((CUTOFF, {"#"}), (None, {"#", "%"})):
        trained = {_train(index, cutoff, num_threads=threads)
                   for index in indexes.values() for threads in (1, 4)}
        assert len(trained) == 1
        vocab = json.loads(trained.pop())["model"]["vocab"]
        assert {c for c in "#%" if c in vocab} == seen


def test_tokenizer_counts_every_row_of_a_table_without_a_create_time(indexes):
    both = [("docs", "body"), ("tags", "label")]
    assert _train(indexes[4, 3], T0 - 10_000, both) == _train(indexes[4, 3], None, both[1:])


def test_tokenizer_rejects_non_string_columns(indexes):
    with pytest.raises(Exception, match="not strings"):
        _train(indexes[1, 1], None, [("docs", "n")])
    with pytest.raises(Exception, match="no column"):
        _train(indexes[1, 1], None, [("docs", "missing")])
    with pytest.raises(Exception, match="num_threads must be positive"):
        _train(indexes[1, 1], None, num_threads=0)


def test_scan_memory_stays_bounded(tmp_path):
    rows, width = 3_000_000, 100
    pq.write_table(pa.table({
        "id": pa.array(range(rows), pa.int64()),
        "ts": pa.array([T0] * rows, pa.timestamp("s")),
        "body": pa.array([f"{i:0{width}d}" for i in range(rows)], pa.string()),
        "n": pa.array(range(rows), pa.int64()),
    }), tmp_path / "docs.parquet")
    pq.write_table(pa.table({"id": pa.array([0], pa.int64()), "label": ["x"]}),
                   tmp_path / "tags.parquet")
    index = _ingest(tmp_path, tmp_path / "index", 4, 3)
    script = """
import sys, pyarrow, diskgraph
def peak_kib():
    return next(int(line.split()[1]) for line in open("/proc/self/status") if line.startswith("VmHWM"))
before = peak_kib()
rows = sum(len(c) for c in diskgraph.scan_column(sys.argv[1], "docs", "body", None,
                                                 chunk_rows=16384, num_threads=64))
print(rows, peak_kib() - before)
"""
    out = subprocess.run([sys.executable, "-c", script, index], check=True,
                         capture_output=True, text=True).stdout.split()
    scanned, growth_kib = int(out[0]), int(out[1])
    assert scanned == rows
    assert growth_kib * 1024 < rows * width
