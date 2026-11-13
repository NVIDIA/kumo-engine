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


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> tuple[str, dict]:
    root = tmp_path_factory.mktemp("lengths")
    rows = 3_000
    body = [None if i % 7 == 0 else " ".join(["word"] * (i % 11)) for i in range(rows)]
    tags = [None if i % 5 == 0 else ["t"] * (i % 4) for i in range(rows)]
    body[1] = "x " * (33 << 20)  # over the 64 MiB of node data a node can hold
    tags[2] = ["t"] * 70_000
    pq.write_table(pa.table({
        "id": pa.array(range(rows), pa.int64()),
        "body": pa.array(body, pa.string()),
        "tags": pa.array(tags, pa.list_(pa.string())),
        "n": pa.array(range(rows), pa.int64()),
    }), root / "docs.parquet", row_group_size=500)

    def column(name, dtype):
        return {"name": name, "ingested_dtype": dtype}

    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "lengths",
            "num_shards": 3,
            "tables": [{
                "name": "docs", "primary_key": "id",
                "columns": [column("id", "int64"), column("body", "string"),
                            column("tags", "string_array"), column("n", "int64")],
                "source": {"type": "parquet", "paths": [str(root / "docs.parquet")]},
            }],
        }),
        output_dir=str(root / "index"),
        encode_threads=4,
        index_threads=4,
    )
    expected = {key: min(len((b or "").split()) + len(t or []) + 1, 65_535)
                for key, (b, t) in enumerate(zip(body, tags, strict=True))}
    expected[1] = 0
    return str(root / "index"), expected


def test_each_node_counts_its_scalars_list_elements_and_pieces(table, tmp_path):
    index, expected = table
    nodes = diskgraph.SamplerEngine(index).lookup_node_ids("docs", pa.array(expected, pa.int64()))
    by_node = [0] * len(nodes)
    for node, length in zip(nodes, expected.values(), strict=True):
        by_node[node] = length
    path = tmp_path / "docs.lengths.bin"
    for threads in (1, 4):
        diskgraph.token_lengths(index, "docs", [("body", WORDS), ("tags", None), ("n", None)],
                                str(path), num_threads=threads)
        assert np.memmap(path, dtype="<u2", mode="r").tolist() == by_node
    assert by_node[nodes[1]] == 0 and by_node[nodes[2]] == 65_535
    assert [p.name for p in tmp_path.iterdir()] == ["docs.lengths.bin"]


def test_token_lengths_refuses_unknown_and_non_string_columns(table, tmp_path):
    index, _ = table
    path = str(tmp_path / "docs.lengths.bin")
    with pytest.raises(diskgraph.DiskgraphConfigError, match="no column"):
        diskgraph.token_lengths(index, "docs", [("missing", None)], path)
    with pytest.raises(diskgraph.DiskgraphConfigError, match="not strings"):
        diskgraph.token_lengths(index, "docs", [("n", WORDS)], path)
    with pytest.raises(diskgraph.DiskgraphConfigError, match="num_threads must be positive"):
        diskgraph.token_lengths(index, "docs", [("n", None)], path, num_threads=0)
