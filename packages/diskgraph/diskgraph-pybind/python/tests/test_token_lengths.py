# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph

WORDS = json.dumps({
    "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
    "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": None,
    "decoder": None, "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
})
ROWS = 3_000
NO_DATA = 1  # holds more node data than a node can store
SATURATED = 2


def _body(key):
    if key == NO_DATA:
        return "x " * (33 << 20)
    return None if key % 7 == 0 else " ".join(["word"] * (key % 11))


def _tags(key):
    if key == SATURATED:
        return ["t"] * 70_000
    return None if key % 5 == 0 else ["t"] * (key % 4)


def _expected(key):
    """Words of the body and elements of the tags, 0 for NULL."""
    if key == NO_DATA:
        return 0
    return min(len((_body(key) or "").split()) + len(_tags(key) or []), 65_535)


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("lengths")
    keys = range(ROWS)
    pq.write_table(pa.table({
        "id": pa.array(keys, pa.int64()),
        "body": pa.array([_body(key) for key in keys], pa.string()),
        "tags": pa.array([_tags(key) for key in keys], pa.list_(pa.string())),
        "vec": pa.array([None if key % 3 == 0 else [0.5] * 3 for key in keys], pa.list_(pa.float32())),
        "n": pa.array([None if key % 2 else key for key in keys], pa.int64()),
    }), root / "docs.parquet", row_group_size=500)

    def column(name, dtype, **extra):
        return {"name": name, "ingested_dtype": dtype, **extra}

    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "lengths",
            "num_shards": 3,
            "tables": [{
                "name": "docs", "primary_key": "id",
                "columns": [column("id", "int64"), column("body", "string"),
                            column("tags", "string_array"),
                            column("vec", "float32_array", fixed_len=3), column("n", "int64")],
                "source": {"type": "parquet", "paths": [str(root / "docs.parquet")]},
            }],
        }),
        output_dir=str(root / "index"),
        encode_threads=4,
        index_threads=4,
    )
    return str(root / "index")


def test_each_node_counts_its_list_elements_and_pieces(index, tmp_path):
    nodes = diskgraph.SamplerEngine(index).lookup_node_ids("docs", pa.array(range(ROWS), pa.int64()))
    expected = np.zeros(ROWS, dtype="<u2")
    expected[nodes] = [_expected(key) for key in range(ROWS)]
    assert expected[nodes[NO_DATA]] == 0 and expected[nodes[SATURATED]] == 65_535
    path = tmp_path / "docs.lengths.bin"
    for threads in (1, 4):
        diskgraph.token_lengths(index, "docs", [("body", WORDS), ("tags", None)], str(path),
                                num_threads=threads)
        assert np.memmap(path, dtype="<u2", mode="r").tolist() == expected.tolist()
    assert [p.name for p in tmp_path.iterdir()] == ["docs.lengths.bin"]


def test_token_lengths_refuses_unknown_fixed_cost_and_non_string_columns(index, tmp_path):
    path = str(tmp_path / "docs.lengths.bin")
    with pytest.raises(diskgraph.DiskgraphConfigError, match="no column"):
        diskgraph.token_lengths(index, "docs", [("missing", None)], path)
    for column in ("n", "vec"):
        with pytest.raises(diskgraph.DiskgraphConfigError, match="fixed token cost"):
            diskgraph.token_lengths(index, "docs", [(column, None)], path)
    with pytest.raises(diskgraph.DiskgraphConfigError, match="not strings"):
        diskgraph.token_lengths(index, "docs", [("n", WORDS)], path)
    with pytest.raises(diskgraph.DiskgraphConfigError, match="num_threads must be positive"):
        diskgraph.token_lengths(index, "docs", [("tags", None)], path, num_threads=0)
