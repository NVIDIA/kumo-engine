# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local/native Flight parity using synthetic prebuilt graph and task files."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from typing import Any

import diskgraph
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.diskgraph.remote_sampling import RemoteRuntimeIndex


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Use synthetic local artifacts without application services."""
    return False


@pytest.fixture(scope="module")
def prepared_task(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Ingest a graph and task once and serve the resulting immutable snapshot."""
    root = tmp_path_factory.mktemp("remote_task")
    graph = root / "graph"
    users = root / "users.parquet"
    pq.write_table(pa.table({"id": range(8), "age": range(20, 28)}), users)
    orders = root / "orders.parquet"
    pq.write_table(
        pa.table(
            {
                "id": range(16),
                "user_id": [i // 2 for i in range(16)],
                "amount": [float(i) for i in range(16)],
            }
        ),
        orders,
    )
    diskgraph.ingest(
        config_json=orjson.dumps(
            {
                "dataset_name": "remote_task",
                "num_shards": 1,
                "tables": [
                    {
                        "name": "users",
                        "primary_key": "id",
                        "columns": [
                            {"name": "id", "ingested_dtype": "int64"},
                            {"name": "age", "ingested_dtype": "int64"},
                        ],
                        "source": {"type": "parquet", "paths": [str(users)]},
                    },
                    {
                        "name": "orders",
                        "primary_key": "id",
                        "columns": [
                            {"name": "id", "ingested_dtype": "int64"},
                            {"name": "user_id", "ingested_dtype": "int64"},
                            {"name": "amount", "ingested_dtype": "float64"},
                        ],
                        "fkeys": [
                            {
                                "src_column": "user_id",
                                "dst_table": "users",
                                "reverse": {},
                            }
                        ],
                        "source": {"type": "parquet", "paths": [str(orders)]},
                    },
                ],
            }
        ).decode(),
        output_dir=str(graph),
        col_stats=False,
    )
    seeds = root / "seeds.parquet"
    pq.write_table(
        pa.table(
            {
                "ENTITY1": pa.array(range(6), pa.int64()),
                "entity_node_id": pa.array(range(6), pa.uint64()),
                "TARGET1": pa.array([0.1 * i for i in range(6)], pa.float64()),
            }
        ),
        seeds,
    )
    extension = graph / "train_tables" / "remote_test"
    diskgraph.ingest(
        config_json=orjson.dumps(
            {
                "dataset_name": "remote_seeds",
                "num_shards": 1,
                "tables": [
                    {
                        "name": "train_seeds",
                        "primary_key": None,
                        "columns": [
                            {"name": "ENTITY1", "ingested_dtype": "int64"},
                            {"name": "entity_node_id", "ingested_dtype": "u_int64"},
                            {"name": "TARGET1", "ingested_dtype": "float64"},
                        ],
                        "fkeys": [],
                        "source": {"type": "parquet", "paths": [str(seeds)]},
                    }
                ],
            }
        ).decode(),
        output_dir=str(extension),
        parent_index_path=str(graph),
        col_stats=False,
    )
    manifest = root / "runtime_manifest.json"
    manifest.write_bytes(
        orjson.dumps(
            {
                "kind": "diskgraph_task_dataset",
                "graph": {"index_path": str(graph)},
                "task": {"is_temporal": False},
                "stages": {
                    "train": {
                        "table_name": "train_seeds",
                        "index_path": str(extension),
                        "row_count": 6,
                    }
                },
            }
        )
    )
    return str(manifest)


@pytest.fixture(scope="module")
def remote_task(prepared_task: str) -> Iterator[tuple[str, str]]:
    """Serve an independent graph replica, receiving the task through Flight."""
    from pathlib import Path

    from kumo_core.diskgraph.sampling_service import (
        SamplingService,
        SamplingServiceConfig,
    )

    manifest = Path(prepared_task)
    graph = manifest.parent / "service_graph"
    shutil.copytree(
        manifest.parent / "graph", graph, ignore=shutil.ignore_patterns("train_tables")
    )
    service = SamplingService(SamplingServiceConfig(graph))
    try:
        yield prepared_task, service.endpoint()
    finally:
        service.close()


def _assert_equal(local: Any, actual: Any) -> None:
    if isinstance(local, (pa.Array, pa.RecordBatch)):
        assert actual.equals(local)
    elif isinstance(local, dict):
        assert local.keys() == actual.keys()
        for key in local:
            _assert_equal(local[key], actual[key])
    elif isinstance(local, list):
        assert len(local) == len(actual)
        if local and isinstance(local[0], dict) and "edge_type" in local[0]:
            local = sorted(
                local, key=lambda item: str(sorted(item["edge_type"].items()))
            )
            actual = sorted(
                actual, key=lambda item: str(sorted(item["edge_type"].items()))
            )
        for expected, value in zip(local, actual, strict=True):
            _assert_equal(expected, value)
    else:
        assert local == actual


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"shuffle": True, "epoch": 3, "num_workers": 2, "worker_id": 1},
        {"selected_stage_node_ids": [4, 0, 2]},
        {"batch_size": 4, "drop_last": True},
        {"batch_size": 4},
    ],
)
def test_local_remote_parity(
    remote_task: tuple[str, str], options: dict[str, Any]
) -> None:
    """Compare complete batches for natural, sharded, selected and partial input."""
    manifest, endpoint = remote_task
    local = diskgraph.RuntimeIndex(manifest, rnd_seed=42, num_threads=1)
    remote = RemoteRuntimeIndex(manifest, endpoint, rnd_seed=42, num_threads=1)
    kwargs = {
        "stage": "train",
        "sampling_plan": {
            "layers": [
                {
                    "fanouts": {},
                    "default_fanout": {
                        "fanout": 2,
                        "mode": "uniform",
                        "absent_key_policy": "uniform_fallback",
                    },
                },
                {
                    "fanouts": {},
                    "default_fanout": {
                        "fanout": 2,
                        "mode": "uniform",
                        "absent_key_policy": "uniform_fallback",
                    },
                },
            ]
        },
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "epoch": 0,
        "batch_size": 2,
        "entity_node_type": "users",
        "fused": False,
        **options,
    }
    try:
        expected, local_count = local.iter(**kwargs)
        actual, remote_count = remote.iter(**kwargs)
        assert local_count == remote_count
        _assert_equal(list(expected), list(actual))
    finally:
        remote.close()


def test_cancellation_releases_gateway_streams(remote_task: tuple[str, str]) -> None:
    """Repeated early exits must release connections and server stream permits."""
    manifest, endpoint = remote_task
    source = RemoteRuntimeIndex(manifest, endpoint, rnd_seed=42, num_threads=1)
    for _ in range(8):
        iterator, _ = source.iter(
            stage="train",
            sampling_plan={"layers": []},
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=1,
            entity_node_type="users",
            fused=False,
        )
        try:
            next(iterator)
        finally:
            source.close()
        assert not source._streams
