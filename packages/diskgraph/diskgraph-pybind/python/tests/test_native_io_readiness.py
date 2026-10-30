# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native backend checks over a real parent and train-table index.

Run with DG_TEST_NATIVE_IO=1 on a Linux host permitting io_uring. These
checks deliberately fail on EPERM, rather than treating denied I/O as success.
"""

import datetime
import json
import os

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_runtime_batch_iterator_e2e as runtime_tests

seed_table_fixture = runtime_tests.seed_table_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("DG_TEST_NATIVE_IO") != "1",
    reason="requires explicit DG_TEST_NATIVE_IO=1 and native Linux io_uring",
)


def snapshot(value):
    """Keep row and topology ordering while comparing exact Arrow values."""
    if isinstance(value, pa.RecordBatch):
        return {"schema": str(value.schema), "columns": snapshot(value.to_pydict())}
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return snapshot(value.to_pylist())
    if isinstance(value, dict):
        result = {key: snapshot(item) for key, item in value.items()}
        # These lists encode dictionaries of edge types. Their iteration
        # order already varies between repeated mmap requests; edge rows,
        # node IDs, feature rows, and task order remain compared exactly.
        for key in ("edge_index_dict", "num_sampled_edges_dict"):
            if key in result:
                result[key].sort(
                    key=lambda entry: tuple(
                        entry["edge_type"][name] for name in ("src", "rel", "dst")
                    )
                )
        return result
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unhandled runtime output: {type(value).__name__}")


def iterator_options():
    fanout = {
        "fanout": 2,
        "mode": "deterministic_uniform",
        "absent_key_policy": "uniform_fallback",
    }
    return {
        "stage": "train",
        "sampling_plan": {
            "layers": [
                {"fanouts": {}, "default_fanout": fanout},
                {"fanouts": {}, "default_fanout": fanout},
            ],
        },
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "epoch": 0,
        "batch_size": 3,
        "drop_last": False,
        "entity_node_type": "users",
        "swap_edge_direction": False,
    }


def test_parent_tt_backend_parity(seed_table_fixture, monkeypatch):
    reference = None
    for backend in ("mmap", "buffered", "direct"):
        monkeypatch.setenv("DG_EDGE_BACKEND", backend)
        index = diskgraph.RuntimeIndex(
            seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
        )
        for repeat in range(3):
            iterator, count = index.iter(**iterator_options(), fused=False)
            batches = list(iterator)
            assert count == 6
            assert len(batches) == 2
            assert sum(len(batch["task"]["input_ids"]) for batch in batches) == 6
            assert any(batch["raw_features"] for batch in batches)
            assert any(batch["graphs"]["main"]["edge_index_dict"] for batch in batches)
            observed = snapshot(batches)
            if reference is None:
                reference = observed
            assert observed == reference, f"backend={backend}, repeat={repeat}"


@pytest.mark.parametrize("backend", ("buffered", "direct"))
@pytest.mark.xfail(
    strict=True,
    raises=diskgraph.DiskgraphConfigError,
    reason="PR424 auto-fused heuristic selects unsupported explicit-I/O dispatch",
)
def test_auto_fused_backend(seed_table_fixture, monkeypatch, backend):
    monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    index = diskgraph.RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
    )
    try:
        iterator, count = index.iter(**iterator_options())
        assert len(list(iterator)) == 2
        assert count == 6
    except diskgraph.DiskgraphConfigError as error:
        assert "fused feature fetching is not yet implemented for explicit edge I/O" in str(error)
        raise


def test_projected_backend_parity(seed_table_fixture, monkeypatch):
    reference = None
    for backend in ("mmap", "buffered", "direct"):
        monkeypatch.setenv("DG_EDGE_BACKEND", backend)
        index = diskgraph.RuntimeIndex(
            seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
        )
        iterator, count = index.iter(
            **iterator_options(),
            projections={"users": [], "orders": ["order_id", "order_time"]},
        )
        batches = list(iterator)
        assert count == 6
        assert len(batches) == 2
        for batch in batches:
            assert "users" not in batch["raw_features"]
        observed = snapshot(batches)
        if reference is None:
            reference = observed
        assert observed == reference, backend


def test_high_degree_seeded_parity(tmp_path, monkeypatch):
    """Pin upstream's seeded sample at a degree exercising partial selection."""
    data = tmp_path / "users.parquet"
    pq.write_table(
        pa.table({
            "user_id": pa.array(range(10001), type=pa.int64()),
            "root_id": pa.array([None] + [0] * 10000, type=pa.int64()),
        }),
        data,
    )
    config = {
        "dataset_name": "native_io_seeded_golden",
        "num_shards": 1,
        "tables": [{
            "name": "users",
            "columns": [
                {"name": "user_id", "ingested_dtype": "int64"},
                {"name": "root_id", "ingested_dtype": "int64"},
            ],
            "primary_key": "user_id",
            "fkeys": [{
                "src_column": "root_id",
                "dst_table": "users",
                "forward": {},
                "reverse": {},
            }],
            "source": {"type": "parquet", "paths": [str(data)]},
        }],
    }
    index_path = str(tmp_path / "parent")
    diskgraph.ingest(json.dumps(config), index_path, col_stats=False)
    plan = {
        "layers": [{
            "fanouts": {
                "users_root_id_to_users_user_id_reverse": {
                    "fanout": 10,
                    "mode": "uniform",
                    "absent_key_policy": "uniform_fallback",
                },
            },
            "default_fanout": None,
        }],
    }
    expected = [0, 1463, 1644, 2885, 3080, 3145, 3786, 4273, 5365, 7368, 9496]
    reference = None
    for backend in ("mmap", "buffered", "direct"):
        monkeypatch.setenv("DG_EDGE_BACKEND", backend)
        # Reverse relation ID 1 plus root global ID 0 gives effective seed 42.
        engine = diskgraph.SamplerEngine(index_path, random_seed=41, num_threads=1)
        events = list(engine.sample({"users": [0]}, json.dumps(plan), fused=False))
        nodes = next(event["batch"] for event in events if event["type"] == "node")
        assert nodes.column("id_b").to_pylist() == expected, backend
        observed = snapshot(events)
        if reference is None:
            reference = observed
        assert observed == reference, backend
