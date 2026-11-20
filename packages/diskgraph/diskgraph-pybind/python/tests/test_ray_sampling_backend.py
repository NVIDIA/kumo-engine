# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Ray actors preserve native batch contents and seed scheduling."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from kumo_core.diskgraph.batch_contract import SamplingArguments
    from kumo_core.diskgraph.ray_sampling import RayRuntimeIndex

import pytest
from diskgraph import RuntimeIndex
from test_runtime_batch_iterator_e2e import (
    _sampling_plan,
    seed_table_fixture,  # noqa: F401 -- shared native ingest fixture
)
from test_stateless_sampling import _assert_equal


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
    from kumo_core.diskgraph.batch_contract import (
        ArtifactIdentity,
        ConsumerToken,
        content_digest,
    )

    ray = pytest.importorskip("ray")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    manifest = seed_table_fixture["manifest_path"]
    snapshot = str(Path(seed_table_fixture["parent_root"]) / "index_metadata.json")
    artifacts = ArtifactIdentity(
        content_digest(Path(snapshot).read_bytes()),
        content_digest(Path(manifest).read_bytes()),
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
        replica = backend.ReplicaConfig(manifest, snapshot, artifacts, 42, 1)
        actors = backend.start_sampling_replicas(
            [replica, replica], [placement, placement]
        )
        try:
            config = backend.RayBackendConfig(
                backend.RayConnection("auto", "native-sampling-tests"),
                actors,
                artifacts,
                42,
                1,
                "native-test",
            )
            yield backend.RayRuntimeIndex(
                manifest, config, ConsumerToken("native-test", 0, "train", 0)
            )
        finally:
            backend.stop_sampling_replicas(actors)
    finally:
        ray.shutdown()


@pytest.mark.parametrize("mode", ["flat", "selected", "rhs"])
def test_remote_native_batches_match_local(
    remote_index: RayRuntimeIndex,
    seed_table_fixture: SeedTableFixture,  # noqa: F811 -- imported pytest fixture
    mode: str,
) -> None:
    """Transport complete Arrow batches with identical local sampling semantics."""
    options: SamplingArguments = {
        "stage": "train",
        "sampling_plan": _sampling_plan(),
        "entity_node_type": "users",
        "num_workers": 2,
        "worker_id": 1,
        "shuffle": mode == "flat",
        "rnd_seed": 42,
        "epoch": 3,
        "batch_size": 2,
    }
    if mode == "selected":
        options["selected_stage_node_ids"] = [4, 1, 4, 2, 0]
    elif mode == "rhs":
        options["rhs_seed_node_type"] = "users"
        options["rhs_seed_time"] = 1000
    local = RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
    )
    expected, expected_count = local.iter(**options)
    actual, count = remote_index.iter(**options)
    assert count == expected_count
    received = list(actual)
    references = list(expected)
    assert len(received) == len(references) > 0
    for batch, reference in zip(received, references, strict=True):
        _assert_equal(batch, reference)
