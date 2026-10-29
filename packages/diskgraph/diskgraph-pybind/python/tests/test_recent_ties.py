# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recent fanout must select logical identities independently of storage order."""

import hashlib
import json

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _index(root, keys, dtype, layout, shards, *, distinct_times=False, weighted=False):
    root.mkdir()
    accounts = root / "accounts.parquet"
    pq.write_table(pa.table({"account_id": [1, 2]}), accounts)
    rows = pa.table({
        "transaction_id": pa.array(keys, type=dtype),
        "account_id": [1] * len(keys),
        "time": [70, 80, 90, 110] if distinct_times else [80, 80, 80, 110],
        "amount": [1, 2, 3, 4],
    })
    if layout == "forward":
        chunks = [rows]
    elif layout == "reverse":
        chunks = [rows.take([3, 2, 1, 0])]
    else:
        chunks = [rows.take([2, 0]), rows.take([3, 1])]
    paths = []
    for index, chunk in enumerate(chunks):
        path = root / f"transactions-{index}.parquet"
        pq.write_table(chunk, path, row_group_size=1)
        paths.append(str(path))
    config = {
        "dataset_name": "recent_ties",
        "num_shards": shards,
        "edge_format": "binary",
        "tables": [
            {"name": "accounts", "primary_key": "account_id",
             "columns": [{"name": "account_id", "ingested_dtype": "int64"}],
             "source": {"type": "parquet", "paths": [str(accounts)]}},
            {"name": "transactions", "primary_key": "transaction_id",
             "columns": [
                 {"name": "transaction_id", "ingested_dtype": "u_int64" if pa.types.is_uint64(dtype) else str(dtype)},
                 {"name": "account_id", "ingested_dtype": "int64"},
                 {"name": "time", "ingested_dtype": "int64"},
                 {"name": "amount", "ingested_dtype": "int64"},
             ],
             "source": {"type": "parquet", "paths": paths},
             "fkeys": [{
                 "src_column": "account_id", "dst_table": "accounts", "bidirectional": True,
                 "reverse": {"time": {"table": "transactions", "column": "time", "unit": "s"}},
             }]},
        ],
    }
    if weighted:
        config["tables"][1]["fkeys"][0]["reverse"]["weight"] = {
            "table": "transactions", "expr": "5 - amount", "dtype": "uint32",
        }
    output = root / "index"
    diskgraph.ingest(json.dumps(config), str(output), col_stats=False, cat_count_strategy="none")
    return output


def _sample(index, fanout, cutoff=100, account=1, *, fused=True, deterministic_ties=True):
    # Each call reopens the same persisted index; the fix must not require a
    # newly sorted adjacency representation or mutate any index bytes.
    engine = diskgraph.SamplerEngine(str(index))
    matches = engine.canonical_node_mask("accounts", [0, 1], pa.array([account, account]))
    seed = next(node_id for node_id, matches_key in enumerate(matches) if matches_key)
    plan = {"layers": [{"fanouts": {
        "transactions_account_id_to_accounts_account_id_reverse": {
            "fanout": fanout, "mode": "temporal_most_recent",
            "absent_key_policy": "uniform_fallback",
        }}, "default_fanout": None}]}
    options = {} if deterministic_ties is None else {"deterministic_ties": deterministic_ties}
    result = []
    for event in engine.sample(
        {"accounts": [seed]}, json.dumps(plan), fused=fused,
        **options, seed_timestamps={"accounts": [cutoff]}, seed_timestamp_unit=diskgraph.TimeUnit.Seconds,
    ):
        if event["type"] == "feature" and "transaction_id" in event["batch"].schema.names:
            result.extend(zip(
                event["batch"].column("transaction_id").to_pylist(),
                event["batch"].column("amount").to_pylist(),
            ))
    return sorted(result)


def _hashes(index):
    return {str(path.relative_to(index)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in index.rglob("*") if path.is_file()}


@pytest.mark.parametrize("keys,dtype", [
    ([2**53 + i for i in range(1, 5)], pa.int64()),
    ([2**63 + i for i in range(1, 5)], pa.uint64()),
    (["tx-101", "tx-102", "tx-103", "tx-104"], pa.string()),
])
@pytest.mark.parametrize("shards", [1, 3])
def test_recent_ties_select_exact_primary_keys_across_layouts(tmp_path, keys, dtype, shards):
    for layout in ["forward", "reverse", "files"]:
        index = _index(tmp_path / layout, keys, dtype, layout, shards)
        before = _hashes(index)
        assert _sample(index, 1) == [(keys[2], 3)]
        assert _sample(index, 2) == [(keys[1], 2), (keys[2], 3)]
        assert _sample(index, 3) == list(zip(keys[:3], [1, 2, 3]))
        assert _sample(index, 0) == []
        assert _sample(index, 1, cutoff=79) == []
        assert _sample(index, 1, account=2) == []
        assert _hashes(index) == before


def test_recent_distinct_timestamps_and_inclusive_cutoff(tmp_path):
    keys = [101, 102, 103, 104]
    for layout in ["forward", "reverse", "files"]:
        index = _index(tmp_path / layout, keys, pa.int64(), layout, 1, distinct_times=True)
        assert _sample(index, 1) == [(103, 3)]
        assert _sample(index, 1, cutoff=80) == [(102, 2)]
        assert _sample(index, 2, cutoff=80) == [(101, 1), (102, 2)]


@pytest.mark.parametrize("fused", [False, True])
def test_recent_ties_reject_unreadable_declared_primary_key(tmp_path, fused):
    index = _index(tmp_path / "data", [101, 102, 103, 104], pa.int64(), "forward", 1)
    path = index / "index_metadata.json"
    metadata = json.loads(path.read_text())
    table = next(table for table in metadata["node_types"] if table["name"] == "transactions")
    table["primary_key_map"]["column"] = "missing_pk"
    path.write_text(json.dumps(metadata))
    with pytest.raises(diskgraph.DiskgraphSystemError, match="missing primary key column missing_pk"):
        _sample(index, 1, fused=fused)


def test_recent_configured_weight_precedes_primary_key(tmp_path):
    index = _index(tmp_path / "weighted", [101, 102, 103, 104], pa.int64(), "reverse", 1,
                   weighted=True)
    # All three eligible timestamps match. Their descending weights are 4, 3, 2,
    # so PK 103 must not overtake PK 101 or PK 102 merely because it is larger.
    assert _sample(index, 1) == [(101, 1)]
    assert _sample(index, 2) == [(101, 1), (102, 2)]


@pytest.mark.parametrize("fused", [False, True])
def test_default_and_false_do_not_read_tie_keys(tmp_path, fused):
    index = _index(tmp_path / "data", [101, 102, 103, 104], pa.int64(), "forward", 1)
    default = _sample(index, 1, fused=fused, deterministic_ties=None)
    assert default == _sample(index, 1, fused=fused, deterministic_ties=False)
    assert default == _sample(index, 1, fused=fused, deterministic_ties=None)
    path = index / "index_metadata.json"
    metadata = json.loads(path.read_text())
    table = next(table for table in metadata["node_types"] if table["name"] == "transactions")
    table["primary_key_map"]["column"] = "missing_pk"
    path.write_text(json.dumps(metadata))
    assert default == _sample(index, 1, fused=fused, deterministic_ties=None)
    assert default == _sample(index, 1, fused=fused, deterministic_ties=False)
    with pytest.raises(diskgraph.DiskgraphSystemError, match="missing primary key column"):
        _sample(index, 1, fused=fused, deterministic_ties=True)
