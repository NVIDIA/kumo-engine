# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata-only seed scheduling without opening trainer-side graph readers."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import orjson
import pytest
from diskgraph import DiskgraphConfigError
from kumo_core.diskgraph.seed_schedule import make_seed_schedule


@pytest.fixture
def metadata(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide only task/shard metadata; fail if a graph reader is constructed."""
    import diskgraph
    import diskgraph.runtime_batch_iterator as runtime

    manifest = {
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
    files = {
        "runtime.json": manifest,
        "/snapshot/train_tables/task/index_metadata.json": {
            "node_types": [
                {
                    "name": "seeds",
                    "shards": [
                        {"min_id": 10, "max_id": 12},
                        {"min_id": 20, "max_id": 21},
                    ],
                }
            ],
        },
        "/snapshot/index_metadata.json": {
            "node_types": [
                {
                    "name": "users",
                    "shards": [
                        {"min_id": 5, "max_id": 6},
                        {"min_id": 8, "max_id": 9},
                    ],
                }
            ],
        },
    }
    reads = Mock(side_effect=files.__getitem__)
    monkeypatch.setattr(runtime, "_load_json", reads)
    forbidden = Mock(side_effect=AssertionError("Trainer must not open graph data"))
    monkeypatch.setattr(diskgraph, "RuntimeIndex", forbidden)
    monkeypatch.setattr(diskgraph, "SamplerEngine", forbidden)
    return {
        "manifest": manifest,
        "files": files,
        "reads": reads,
        "forbidden": forbidden,
    }


def _options(**changes: object) -> dict[str, Any]:
    return {
        "stage": "train",
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "rnd_seed": 42,
        "epoch": 0,
        "batch_size": 2,
        **changes,
    }


def test_stage_metadata_produces_canonical_ids(metadata: dict[str, Any]) -> None:
    """Sparse stage ranges map to canonical ids without opening graph data."""
    schedule, count = make_seed_schedule("runtime.json", _options())
    assert count == 5
    assert [schedule.next_seed_ids() for _ in range(4)] == [
        [10, 11],
        [12, 20],
        [21],
        None,
    ]
    assert [call.args[0] for call in metadata["reads"].call_args_list] == [
        "runtime.json",
        "/snapshot/train_tables/task/index_metadata.json",
    ]
    metadata["forbidden"].assert_not_called()


def test_selected_ids_keep_order_duplicates_and_sharding(
    metadata: dict[str, Any],
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
    metadata: dict[str, Any],
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
    assert metadata["reads"].call_args.args == ("/snapshot/index_metadata.json",)


@pytest.mark.parametrize("failure", ["stage", "empty", "table", "rhs"])
def test_missing_seed_metadata_fails_before_sampling(
    metadata: dict[str, Any], failure: str
) -> None:
    """Invalid task/table metadata cannot silently select a different seed domain."""
    options = _options()
    if failure == "stage":
        options["stage"] = "missing"
    elif failure == "empty":
        metadata["manifest"]["stages"]["train"]["row_count"] = 0
    elif failure == "table":
        metadata["files"]["/snapshot/train_tables/task/index_metadata.json"][
            "node_types"
        ] = []
    else:
        options["rhs_seed_node_type"] = "users"
        metadata["files"]["/snapshot/index_metadata.json"]["node_types"][0][
            "shards"
        ] = []
    with pytest.raises(DiskgraphConfigError):
        make_seed_schedule("runtime.json", options)
    metadata["forbidden"].assert_not_called()


@pytest.mark.parametrize("grouped", [False, True])
def test_sidecar_schedules_use_qualified_stage_identity(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, Any], grouped: bool
) -> None:
    """Bucket and weighted schedules match native namespaced shuffle identities."""
    import diskgraph._c as native
    import diskgraph.runtime_batch_iterator as runtime

    captured: list[dict[str, Any]] = []
    schedule = Mock()
    schedule.num_samples.return_value = 5

    def construct(config: str) -> Mock:
        captured.append(orjson.loads(config))
        return schedule

    monkeypatch.setattr(native, "SeedSchedule", construct)
    weighted = Mock(return_value=None if grouped else '{"buckets":[]}')
    monkeypatch.setattr(runtime, "_weighted_seed_sampling_json", weighted)
    options = _options(group_by_timestamp=grouped)
    if not grouped:
        options["weighted_seed_sampling"] = {"num_samples": 5}
    actual, count = make_seed_schedule("runtime.json", options)
    assert actual is schedule and count == 5
    assert captured[0]["stage_key"] == "task/seeds"
    assert captured[0]["bucket_extension_root"] == (
        "/snapshot/train_tables/task" if grouped else None
    )
    assert captured[0]["weighted_seed_sampling_json"] == weighted.return_value
