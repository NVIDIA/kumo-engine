# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Ray actors preserve native batch contents and seed scheduling."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from diskgraph.runtime_batch import RuntimeBatch
    from kumo_core.diskgraph.ray_sampling import RayRuntimeIndex
    from torch_geometric.data import HeteroData

import pytest
from diskgraph import RuntimeIndex, SamplingOptions
from test_runtime_batch_iterator_e2e import (
    _sampling_plan,
    seed_table_fixture,  # noqa: F401 -- shared native ingest fixture
)


class SeedTableFixture(TypedDict):
    """Paths and sizes provided by the shared native ingest fixture."""

    manifest_path: str
    parent_root: str
    n_users: int
    n_seeds: int


@pytest.fixture(scope="module")
def remote_index(
    seed_table_fixture: SeedTableFixture,  # noqa: F811 -- imported pytest fixture
) -> Iterator[RayRuntimeIndex]:
    """Open two native samplers on a local Ray cluster; close only owned actors."""
    backend = pytest.importorskip("kumo_core.diskgraph.ray_sampling")
    from kumo_core.diskgraph.batch_contract import ArtifactIdentity

    ray = pytest.importorskip("ray")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    manifest = seed_table_fixture["manifest_path"]
    snapshot = str(Path(seed_table_fixture["parent_root"]) / "index_metadata.json")
    artifacts = ArtifactIdentity(
        sha256(Path(snapshot).read_bytes()).hexdigest(),
        sha256(Path(manifest).read_bytes()).hexdigest(),
        "native-test-mapping",
    )
    ray.init(
        num_cpus=2,
        namespace="native-sampling-tests",
        include_dashboard=False,
        object_store_memory=100 * 1024**2,
        _node_ip_address="127.0.0.1",
    )
    try:
        placement = NodeAffinitySchedulingStrategy(
            ray.get_runtime_context().get_node_id(), soft=False
        )
        import sys

        ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
        replica = backend.ReplicaConfig(
            manifest, snapshot, artifacts, 42, 1, batch_encoder=_encode_native_runtime
        )
        actors = backend.start_sampling_replicas(
            [replica, replica], [placement, placement]
        )
        try:
            config = backend.RayBackendConfig(
                actors,
                artifacts,
                42,
                1,
            )
            yield backend.RayRuntimeIndex(manifest, config)
        finally:
            backend.stop_sampling_replicas(actors)
    finally:
        ray.shutdown()


@pytest.mark.parametrize("mode", ["flat", "selected", "rhs"])
def test_remote_encoded_batches_match_local(
    remote_index: RayRuntimeIndex,
    seed_table_fixture: SeedTableFixture,  # noqa: F811 -- imported pytest fixture
    mode: str,
) -> None:
    """Sample and encode on actors, preserving local collation semantics."""
    options = SamplingOptions(
        stage="train",
        sampling_plan=_sampling_plan(),
        entity_node_type="users",
        num_workers=2,
        worker_id=1,
        shuffle=mode == "flat",
        rnd_seed=42,
        epoch=3,
        batch_size=2,
    )
    if mode == "selected":
        options = replace(options, selected_stage_node_ids=[4, 1, 4, 2, 0])
    elif mode == "rhs":
        options = replace(options, rhs_seed_node_type="users", rhs_seed_time=1000)
    local = RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
    )
    expected, expected_count = local.iter(options)
    actual, count = remote_index.iter(options)
    assert count == expected_count
    received = list(actual)
    references = list(expected)
    assert len(received) == len(references) > 0
    for batch, reference in zip(received, references, strict=True):
        _assert_graph(batch, _encode_native_runtime(reference))


def _encode_native_runtime(batch: RuntimeBatch) -> HeteroData:
    from kumo_core.diskgraph.runtime_adapter import (
        FeatureSchema,
        MapperBank,
        runtime_batch_to_hetero,
    )

    names = set(batch.raw_features) | set(batch.graphs["main"].node_ids) | {"users"}
    mappers: MapperBank = {name: {} for name in names}
    schema: FeatureSchema = {
        "node_types": {name: {"mapper_key": name} for name in names}
    }
    return runtime_batch_to_hetero(
        batch,
        train_table_name="users",
        seed_node_type="users",
        mapper_bank=mappers,
        feature_schema=schema,
        attach_pkeys=False,
    )[0]


def _assert_graph(batch: HeteroData, expected: HeteroData) -> None:
    import torch
    from torch_frame import TensorFrame

    assert set(batch.node_types) == set(expected.node_types)
    assert set(batch.edge_types) == set(expected.edge_types)
    for name in batch.node_types + batch.edge_types:
        assert set(batch[name].keys()) == set(expected[name].keys())
        for key, value in expected[name].items():
            actual = batch[name][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value, equal_nan=True)
            elif isinstance(value, TensorFrame):
                assert actual.num_rows == value.num_rows
                assert actual.col_names_dict == value.col_names_dict
            else:
                assert actual == value
