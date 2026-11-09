# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local diagnostics over real parent/train-table reads."""

import json
import multiprocessing
import os

import diskgraph
import pytest
import test_runtime_batch_iterator_e2e as runtime_tests
from test_native_io_readiness import iterator_options

seed_table_fixture = runtime_tests.seed_table_fixture


def test_stats_are_readonly():
    snapshot = diskgraph.page_io_stats()
    values = snapshot.to_dict()
    assert values["pid"] == os.getpid()
    assert json.loads(json.dumps(values)) == values
    with pytest.raises(AttributeError):
        snapshot.pid = 0
    values["pid"] = 0
    assert snapshot.to_dict()["pid"] == os.getpid()


def _read_backend_stats(manifest, backend, connection):
    os.environ["DG_EDGE_BACKEND"] = backend
    os.environ["DG_DIRECT_CACHE_GB"] = "1" if backend == "direct" else "0"
    os.environ["DG_IO_RINGS"] = "1"
    try:
        before = diskgraph.page_io_stats().to_dict()
        index = diskgraph.RuntimeIndex(manifest, rnd_seed=42, num_threads=1)
        snapshots = [before]
        for _ in range(2):
            iterator, count = index.iter(**iterator_options(), fused=False)
            batches = list(iterator)
            assert count == 6 and len(batches) == 2
            snapshots.append(diskgraph.page_io_stats().to_dict())
        connection.send({"snapshots": snapshots})
    except Exception as error:
        connection.send({"error": repr(error)})
    finally:
        connection.close()


@pytest.mark.skipif(
    os.environ.get("DG_TEST_NATIVE_IO") != "1",
    reason="requires Linux io_uring and explicit DG_TEST_NATIVE_IO=1",
)
@pytest.mark.parametrize("backend", ("mmap", "buffered", "direct"))
def test_stats_track_actual_reads(seed_table_fixture, backend):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(
        target=_read_backend_stats,
        args=(seed_table_fixture["manifest_path"], backend, sender),
    )
    child.start()
    sender.close()
    try:
        child.join(timeout=30)
        assert child.exitcode == 0
        assert receiver.poll(), "diagnostics child returned no snapshots"
        result = receiver.recv()
        assert "error" not in result, result
        before, first, second = result["snapshots"]
        assert not before["runtime_initialized"]
        assert not before["cache_initialized"]
        assert before["configured_backend"] == backend
        for name in ("io_ops", "bytes", "runtime_jobs", "cache_hits", "cache_misses"):
            assert before[name] <= first[name] <= second[name]
        if backend == "mmap":
            assert second["io_ops"] == 0
            assert not second["runtime_initialized"]
        else:
            assert second["io_ops"] > 0
            assert second["bytes"] > 0
            assert second["runtime_initialized"]
            assert second["runtime_rings"] == 1
        if backend == "direct":
            assert second["cache_initialized"]
            assert second["cache_capacity_pages"] > 0
            assert second["cache_hits"] > first["cache_hits"]
        else:
            assert not second["cache_initialized"]
            assert second["cache_capacity_pages"] == 0
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        receiver.close()


def _query_inherited_stats(connection):
    try:
        diskgraph.page_io_stats()
    except RuntimeError as error:
        connection.send(str(error))
    else:
        connection.send("unexpected success")
    finally:
        connection.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_stats_reject_inherited_counters():
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_query_inherited_stats, args=(sender,))
    child.start()
    sender.close()
    try:
        child.join(timeout=5)
        assert child.exitcode == 0
        assert receiver.poll()
        assert "inherited from another process" in receiver.recv()
        assert diskgraph.page_io_stats().pid == os.getpid()
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        receiver.close()
