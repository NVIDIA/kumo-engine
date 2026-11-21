# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime records preserve Arrow owners and standard pickle transport."""

import gc
import pickle
from dataclasses import fields, is_dataclass

import pyarrow as pa
import pytest
from diskgraph import (
    RuntimeBatch,
    RuntimeEdge,
    RuntimeEdgeCounts,
    RuntimeEdgeType,
    RuntimeFeatureBlock,
    RuntimeGraph,
    RuntimeTask,
)


@pytest.mark.parametrize(
    "record",
    [
        RuntimeBatch(),
        RuntimeTask(),
        RuntimeGraph(),
        RuntimeEdgeType("users", "follows", "users"),
        RuntimeEdge(
            RuntimeEdgeType("users", "follows", "users"),
            pa.array([0], type=pa.uint32()),
            pa.array([1], type=pa.uint32()),
        ),
        RuntimeEdgeCounts(
            RuntimeEdgeType("users", "follows", "users"),
            pa.array([1], type=pa.uint64()),
        ),
        RuntimeFeatureBlock(record_batches=[pa.record_batch({"age": [21]})]),
    ],
)
def test_runtime_records_are_mutable_pickleable_dataclasses(record: object) -> None:
    """The public records require no dictionary adapter or custom pickle hooks."""
    assert is_dataclass(record) and not isinstance(record, type)
    assert not isinstance(record, dict)
    restored = pickle.loads(pickle.dumps(record))
    assert type(restored) is type(record)
    for item in fields(record):
        expected = getattr(record, item.name)
        actual = getattr(restored, item.name)
        if isinstance(expected, pa.Array):
            assert expected.equals(actual)
        else:
            assert actual == expected
        setattr(restored, item.name, actual)


def test_batch_defaults_do_not_share_mutable_containers() -> None:
    """Independent batches, graphs and feature blocks own their own containers."""
    first, second = RuntimeBatch(), RuntimeBatch()
    first.graphs["main"] = RuntimeGraph()
    first.raw_features["users"] = RuntimeFeatureBlock()
    first.history_merges["events"] = (pa.array([0]), pa.array([1]))
    first.metadata["consumer"] = "trainer"
    first.task.num_positives = 3
    assert not second.graphs and not second.raw_features and not second.history_merges
    assert second.task.num_positives == 0
    assert not second.metadata
    graph = RuntimeGraph()
    first.graphs["main"].node_ids["users"] = pa.array([7])
    assert not graph.node_ids
    block = RuntimeFeatureBlock()
    first.raw_features["users"].record_batches.append(pa.record_batch({"age": [21]}))
    assert not block.record_batches


def test_record_keeps_arrow_slices_alive_without_copying() -> None:
    """Record construction retains the existing Arrow arrays and their buffers."""
    source = pa.array([10, 11, 12], type=pa.uint64())
    view = source.slice(1, 2)
    address = source.buffers()[1].address
    batch = RuntimeBatch(task=RuntimeTask(seed_node_ids=view))
    assert batch.task.seed_node_ids is view
    del view, source
    gc.collect()
    assert batch.task.seed_node_ids is not None
    assert batch.task.seed_node_ids.buffers()[1].address == address
    assert batch.task.seed_node_ids.to_pylist() == [11, 12]


def test_ray_response_roundtrip_preserves_runtime_record_types() -> None:
    """A sampler response transports typed records and their Arrow arrays."""
    ray = pytest.importorskip("ray")

    def produce() -> RuntimeBatch:
        """Return the same public objects constructed by the native producer."""
        import pyarrow as pa
        from diskgraph import (
            RuntimeBatch,
            RuntimeFeatureBlock,
            RuntimeGraph,
            RuntimeTask,
        )

        ids = pa.array([7, 9], type=pa.uint64())
        return RuntimeBatch(
            task=RuntimeTask(seed_node_ids=ids),
            graphs={"main": RuntimeGraph(node_ids={"users": ids})},
            raw_features={
                "users": RuntimeFeatureBlock(
                    key="users",
                    num_rows=2,
                    record_batches=[pa.record_batch({"age": [21, 32]})],
                )
            },
        )

    ray.init(num_cpus=1, include_dashboard=False, object_store_memory=100 * 1024**2)
    try:
        batch = ray.get(ray.remote(produce).remote())
        assert isinstance(batch, RuntimeBatch)
        assert isinstance(batch.task, RuntimeTask)
        assert isinstance(batch.graphs["main"], RuntimeGraph)
        assert isinstance(batch.raw_features["users"], RuntimeFeatureBlock)
        assert batch.task.seed_node_ids.to_pylist() == [7, 9]
        assert batch.graphs["main"].node_ids["users"].to_pylist() == [7, 9]
        assert batch.raw_features["users"].record_batches[0].column(0).to_pylist() == [
            21,
            32,
        ]
    finally:
        ray.shutdown()
