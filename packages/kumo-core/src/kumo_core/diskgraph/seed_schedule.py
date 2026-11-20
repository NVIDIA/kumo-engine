# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trainer-side native seed schedules using metadata and seed sidecars only."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypedDict

from kumo_core.diskgraph.batch_contract import SamplingOptions

if TYPE_CHECKING:
    from diskgraph._c import SeedSchedule


class SeedShard(TypedDict):
    """Inclusive seed IDs stored in one index shard."""

    min_id: int
    max_id: int


class SeedTableMetadata(TypedDict):
    """Seed table name and shard ranges read from index metadata."""

    name: str
    shards: list[SeedShard]


class SeedIndexMetadata(TypedDict):
    """Subset of native index metadata needed by seed scheduling."""

    node_types: list[SeedTableMetadata]


class SeedStageRef(TypedDict):
    """Task manifest paths identifying the stage's seed table."""

    index_path: str
    table_name: str


def make_seed_schedule(
    dataset_ref: str,
    options: SamplingOptions,
) -> tuple[SeedSchedule, int]:
    """Build the native seed schedule using metadata and seed sidecars only.

    Args:
        dataset_ref: Runtime manifest with local metadata/seed-sidecar paths.
        options: The same stage, epoch, shard, and seed options as ``iter``.

    Returns:
        Native seed schedule and its worker-local sample/bucket count.

    Raises:
        DiskgraphConfigError: If the stage, metadata, or seed options are invalid.
        OSError: If required seed metadata or sidecars are unavailable.
    """
    from diskgraph import DiskgraphConfigError
    from diskgraph._c import SeedSchedule
    from diskgraph.runtime_batch_iterator import (
        _extension_ref_from_stage_path,
        _graph_index_path,
        _load_json,
        _manifest_view,
        _stage_tables,
        _weighted_seed_sampling_json,
    )
    from diskgraph.seed_schedule import SeedScheduleConfig

    manifest, split_info = _manifest_view(dataset_ref)
    stage = options["stage"]
    ref: SeedStageRef | None = _stage_tables(manifest).get(stage)
    if ref is None or not split_info["stage_counts"].get(stage):
        raise DiskgraphConfigError(f"stage {stage!r} not available on this dataset")
    parent = _graph_index_path(manifest)
    extension = _extension_ref_from_stage_path(parent, ref["index_path"])
    root = os.path.join(parent, extension)
    rhs_type = options.get("rhs_seed_node_type")
    table_name = (rhs_type or ref["table_name"]).rsplit("/", 1)[-1]
    metadata: SeedIndexMetadata = _load_json(
        os.path.join(parent if rhs_type else root, "index_metadata.json")
    )
    table = next(
        (node for node in metadata["node_types"] if node["name"] == table_name), None
    )
    if table is None:
        raise DiskgraphConfigError(f"missing seed metadata for {table_name!r}")
    ranges = [(shard["min_id"], shard["max_id"]) for shard in table["shards"]]
    if rhs_type:
        nonempty = [(lo, hi) for lo, hi in ranges if hi >= lo]
        if not nonempty:
            raise DiskgraphConfigError("empty RHS candidate domain")
        ranges = [(min(lo for lo, _ in nonempty), max(hi for _, hi in nonempty))]
    weighted_sampling = _weighted_seed_sampling_json(
        ref, stage, options.get("weighted_seed_sampling")
    )
    stage_key = rhs_type or ref["table_name"]
    if (
        not rhs_type
        and (weighted_sampling is not None or options.get("group_by_timestamp"))
        and "/" not in stage_key
    ):
        stage_key = f"{extension.rsplit('/', 1)[-1]}/{stage_key}"
    config = SeedScheduleConfig(
        stage_key=stage_key,
        ranges=ranges,
        num_workers=options["num_workers"],
        worker_id=options["worker_id"],
        shuffle=options["shuffle"],
        rnd_seed=options["rnd_seed"],
        epoch=options["epoch"],
        batch_size=options["batch_size"],
        drop_last=options.get("drop_last", False),
        rhs=rhs_type is not None,
        selected_stage_node_ids=options.get("selected_stage_node_ids"),
        weighted_seed_sampling_json=weighted_sampling,
        bucket_extension_root=root if options.get("group_by_timestamp") else None,
    )
    schedule = SeedSchedule(config)
    return schedule, schedule.num_samples()
