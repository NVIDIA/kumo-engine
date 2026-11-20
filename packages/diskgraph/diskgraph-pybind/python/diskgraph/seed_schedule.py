# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration for graph-independent native seed scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class SeedScheduleConfig:
    """Configure a graph-independent native seed schedule.

    Args:
        stage_key: Exact stage-table, catalog, or RHS key for this mode's RNG.
        ranges: Disjoint inclusive ID ranges in traversal order.
        batch_size: Maximum positive number of seeds per batch.
        num_workers: Positive worker count.
        worker_id: Zero-based worker shard identity.
        shuffle: Shuffle flat/RHS seed order.
        rnd_seed: Base random seed, required when shuffling.
        epoch: Epoch mixed into the random seed.
        drop_last: Drop partial flat/RHS batches.
        rhs: Use one nonempty RHS candidate range.
        selected_stage_node_ids: Ordered stage IDs, preserving duplicates.
        weighted_seed_sampling_json: Weighted-sampling settings and sidecars.
        bucket_extension_root: Extension containing timestamp buckets.
    """

    stage_key: str
    ranges: list[tuple[int, int]]
    batch_size: int
    num_workers: int = 1
    worker_id: int = 0
    shuffle: bool = False
    rnd_seed: int | None = None
    epoch: int = 0
    drop_last: bool = False
    rhs: bool = False
    selected_stage_node_ids: list[int] | None = None
    weighted_seed_sampling_json: str | None = None
    bucket_extension_root: str | None = None
