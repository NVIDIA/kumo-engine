# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import random

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
    ids = list(range(2_000))
    ts = [T0 + rng.randrange(-1_000, 1_000) for _ in ids]
    ts[1:3] = [CUTOFF, CUTOFF + 1]
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
