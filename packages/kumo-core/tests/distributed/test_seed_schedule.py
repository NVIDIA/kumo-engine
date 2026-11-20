# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata-only seed scheduling without opening trainer-side graph readers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack
from unittest.mock import Mock

import pytest
from diskgraph import DiskgraphConfigError
from kumo_core.diskgraph.batch_contract import (
    SamplingArguments,
    SamplingOptions,
    WeightedSeedSampling,
)
from kumo_core.diskgraph.seed_schedule import SeedIndexMetadata, make_seed_schedule

if TYPE_CHECKING:
    from diskgraph.seed_schedule import SeedScheduleConfig


class StageMetadata(TypedDict):
    """Synthetic stage metadata whose row count can be changed by a test."""

    table_name: str
    index_path: str
    row_count: int


class ManifestMetadata(TypedDict):
    """Synthetic task manifest containing only scheduling metadata."""

    kind: str
    graph: dict[str, str]
    stages: dict[str, StageMetadata]


@dataclass
class MetadataFixture:
    """Typed synthetic metadata and graph-reader guards used by schedule tests."""

    manifest: ManifestMetadata
    seed_index: SeedIndexMetadata
    rhs_index: SeedIndexMetadata
    reads: Mock
    forbidden: Mock


@pytest.fixture
def metadata(monkeypatch: pytest.MonkeyPatch) -> MetadataFixture:
    """Provide only task/shard metadata; fail if a graph reader is constructed."""
    import diskgraph
    import diskgraph.runtime_batch_iterator as runtime

    manifest: ManifestMetadata = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": "/snapshot"},
        "stages": {
            "train": {
                "table_name": "seeds",
                "index_path": "/snapshot/train_tables/task",
                "row_count": 5,
            }
        },
    }
    seed_index: SeedIndexMetadata = {
        "node_types": [
            {
                "name": "seeds",
                "shards": [
                    {"min_id": 10, "max_id": 12},
                    {"min_id": 20, "max_id": 21},
                ],
            }
        ],
    }
    rhs_index: SeedIndexMetadata = {
        "node_types": [
            {
                "name": "users",
                "shards": [
                    {"min_id": 5, "max_id": 6},
                    {"min_id": 8, "max_id": 9},
                ],
            }
        ],
    }
    files: dict[str, ManifestMetadata | SeedIndexMetadata] = {
        "runtime.json": manifest,
        "/snapshot/train_tables/task/index_metadata.json": seed_index,
        "/snapshot/index_metadata.json": rhs_index,
    }
    reads = Mock(side_effect=files.__getitem__)
    monkeypatch.setattr(runtime, "_load_json", reads)
    forbidden = Mock(side_effect=AssertionError("Trainer must not open graph data"))
    monkeypatch.setattr(diskgraph, "RuntimeIndex", forbidden)
    monkeypatch.setattr(diskgraph, "SamplerEngine", forbidden)
    return MetadataFixture(manifest, seed_index, rhs_index, reads, forbidden)


def _options(**changes: Unpack[SamplingArguments]) -> SamplingOptions:
    arguments: SamplingArguments = {
        "stage": "train",
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "rnd_seed": 42,
        "epoch": 0,
        "batch_size": 2,
        **changes,
    }
    return SamplingOptions.from_kwargs(**arguments)


def test_stage_metadata_produces_canonical_ids(metadata: MetadataFixture) -> None:
    """Sparse stage ranges map to canonical ids without opening graph data."""
    schedule, count = make_seed_schedule("runtime.json", _options())
    assert count == 5
    assert [schedule.next_seed_ids() for _ in range(4)] == [
        [10, 11],
        [12, 20],
        [21],
        None,
    ]
    assert [call.args[0] for call in metadata.reads.call_args_list] == [
        "runtime.json",
        "/snapshot/train_tables/task/index_metadata.json",
    ]
    metadata.forbidden.assert_not_called()


def test_selected_ids_keep_order_duplicates_and_sharding(
    metadata: MetadataFixture,
) -> None:
    """Logical worker sharding preserves the caller's explicit seed selection."""
    schedule, count = make_seed_schedule(
        "runtime.json",
        _options(
            num_workers=2,
            worker_id=1,
            selected_stage_node_ids=[12, 20, 12, 21, 10],
        ),
    )
    assert count == 2
    assert schedule.next_seed_ids() == [20, 21]
    assert schedule.next_seed_ids() is None


def test_rhs_uses_parent_metadata_and_dense_candidate_domain(
    metadata: MetadataFixture,
) -> None:
    """RHS scheduling uses the parent table's enclosing native node-id range."""
    schedule, count = make_seed_schedule(
        "runtime.json", _options(rhs_seed_node_type="users")
    )
    assert count == 5
    assert [schedule.next_seed_ids() for _ in range(4)] == [
        [5, 6],
        [7, 8],
        [9],
        None,
    ]
    assert metadata.reads.call_args.args == ("/snapshot/index_metadata.json",)


@pytest.mark.parametrize("failure", ["stage", "empty", "table", "rhs"])
def test_missing_seed_metadata_fails_before_sampling(
    metadata: MetadataFixture, failure: str
) -> None:
    """Invalid task/table metadata cannot silently select a different seed domain."""
    options = _options()
    if failure == "stage":
        options = replace(options, stage="missing")
    elif failure == "empty":
        metadata.manifest["stages"]["train"]["row_count"] = 0
    elif failure == "table":
        metadata.seed_index["node_types"] = []
    else:
        options = replace(options, rhs_seed_node_type="users")
        metadata.rhs_index["node_types"][0]["shards"] = []
    with pytest.raises(DiskgraphConfigError):
        make_seed_schedule("runtime.json", options)
    metadata.forbidden.assert_not_called()


@pytest.mark.parametrize("grouped", [False, True])
def test_sidecar_schedules_use_qualified_stage_identity(
    monkeypatch: pytest.MonkeyPatch, metadata: MetadataFixture, grouped: bool
) -> None:
    """Bucket and weighted schedules match native namespaced shuffle identities."""
    import diskgraph._c as native
    import diskgraph.runtime_batch_iterator as runtime

    captured: list[SeedScheduleConfig] = []
    schedule = Mock()
    schedule.num_samples.return_value = 5

    def construct(config: SeedScheduleConfig) -> Mock:
        captured.append(config)
        return schedule

    monkeypatch.setattr(native, "SeedSchedule", construct)
    weighted = Mock(return_value=None if grouped else '{"buckets":[]}')
    monkeypatch.setattr(runtime, "_weighted_seed_sampling_json", weighted)
    options = _options(group_by_timestamp=grouped)
    if not grouped:
        options = replace(
            options, weighted_seed_sampling=WeightedSeedSampling(num_samples=5)
        )
    actual, count = make_seed_schedule("runtime.json", options)
    assert actual is schedule and count == 5
    assert captured[0].stage_key == "task/seeds"
    assert captured[0].bucket_extension_root == (
        Path("/snapshot/train_tables/task") if grouped else None
    )
    assert captured[0].weighted_seed_sampling_json == weighted.return_value
