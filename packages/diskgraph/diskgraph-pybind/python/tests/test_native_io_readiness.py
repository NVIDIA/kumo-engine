# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native backend checks over a real parent and train-table index.

Run with DG_TEST_NATIVE_IO=1 on a Linux host permitting io_uring. These
checks deliberately fail on EPERM, rather than treating denied I/O as success.
"""

import datetime
import gc
import json
import multiprocessing
import os
from functools import partial

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


def _check_inherited_handles(handles, connection):
    """Exercise actual native owners, then destroy them before child exit."""
    index, iterator = handles
    operations = (
        partial(index.iter, **iterator_options(), fused=False),
        iterator.num_samples,
        iterator.worker_batch_count,
        iterator.stage_size,
        partial(len, iterator),
        partial(next, iterator),
    )
    errors = []
    try:
        for operation in operations:
            try:
                operation()
            except diskgraph.DiskgraphSystemError as error:
                assert "another process" in str(error), str(error)
                errors.append(str(error))
            else:
                raise AssertionError("inherited native handle accepted an operation")
        del operations, operation, index, iterator
        handles.clear()
        gc.collect()
        connection.send({"errors": len(errors), "dropped": True})
    except Exception as error:
        connection.send({"failed": repr(error)})
    finally:
        connection.close()


@pytest.mark.parametrize("backend", ("mmap", "direct"))
def test_inherited_runtime_handles(seed_table_fixture, monkeypatch, backend):
    monkeypatch.setenv("DG_EDGE_BACKEND", backend)

    def open_handles():
        index = diskgraph.RuntimeIndex(
            seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
        )
        iterator, count = index.iter(**iterator_options(), fused=False)
        assert count == 6
        assert next(iterator)["task"]["input_ids"].to_pylist()
        return [index, iterator]

    handles = open_handles()
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_check_inherited_handles, args=(handles, sender))
    child.start()
    sender.close()
    try:
        child.join(timeout=5)
        assert not child.is_alive(), "inherited runtime access or destruction hung"
        assert child.exitcode == 0
        assert receiver.poll(), "child exited without validating inherited handles"
        assert receiver.recv() == {"errors": 6, "dropped": True}
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        receiver.close()
    # Fork does not invalidate the creator's handles or ordinary epoch reuse.
    assert len(list(handles[1])) == 1
    iterator, count = handles[0].iter(**iterator_options(), fused=False)
    assert count == 6 and len(list(iterator)) == 2


def _identity(batch):
    return batch


@pytest.mark.skipif(
    os.environ.get("DG_TEST_CORE_IO") != "1",
    reason="requires checkout kumo_core and its real Torch dependencies",
)
@pytest.mark.parametrize("backend", ("mmap", "direct"))
def test_warmed_dataset_worker(seed_table_fixture, monkeypatch, backend):
    from kumo_core.diskgraph.lightning_data import (
        DiskgraphTaskBatchIterable,
        _ShuffleStreamState,
    )
    from torch.utils.data import DataLoader

    monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    options = iterator_options()
    dataset = DiskgraphTaskBatchIterable(
        manifest_path=seed_table_fixture["manifest_path"],
        stage="train",
        sampling_plan=options["sampling_plan"],
        batch_size=3,
        shuffle=False,
        drop_last=False,
        base_seed=42,
        shuffle_state=_ShuffleStreamState(),
        num_seeds=6,
        replay=False,
        runtime_options={
            "num_threads": 1,
            "fused": False,
            "entity_node_type": "users",
            "swap_edge_direction": False,
        },
    )
    reference = snapshot(list(dataset))
    assert len(reference) == 2
    owner = dataset._runtime_index()
    assert dataset._runtime_index() is owner
    assert snapshot(list(dataset)) == reference
    state = dataset.__getstate__()
    assert state["_runtime_index_handle"] is None
    assert state["_runtime_index_pid"] is None
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        multiprocessing_context="fork",
        collate_fn=_identity,
        timeout=3,
    )
    iterator = iter(loader)
    try:
        assert snapshot(list(iterator)) == reference
    finally:
        iterator._shutdown_workers()
    assert dataset._runtime_index() is owner
    assert snapshot(list(dataset)) == reference


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
def test_auto_fused_backend(seed_table_fixture, monkeypatch, backend):
    monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    index = diskgraph.RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
    )
    iterator, count = index.iter(**iterator_options())
    observed = snapshot(list(iterator))
    explicit, explicit_count = index.iter(**iterator_options(), fused=False)
    assert observed == snapshot(list(explicit))
    assert len(observed) == 2
    assert count == explicit_count == 6


@pytest.mark.parametrize("backend", ("buffered", "direct"))
@pytest.mark.parametrize("layers", ([], iterator_options()["sampling_plan"]["layers"]))
def test_explicit_fused_rejected(seed_table_fixture, monkeypatch, backend, layers):
    monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    engine = diskgraph.SamplerEngine(seed_table_fixture["parent_root"], num_threads=1)
    with pytest.raises(diskgraph.DiskgraphConfigError, match="fused feature fetching is not yet implemented"):
        list(engine.sample({"users": [0]}, json.dumps({"layers": layers}), fused=True))


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
