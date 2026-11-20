# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed options shared by local and remote DiskGraph sampling."""

from dataclasses import dataclass, field
from enum import Enum

from .sampling_plan import SamplingPlan


class InheritEngineSeed(Enum):
    """Use the random seed configured when the index was opened."""

    DEFAULT = "inherit_engine_seed"


@dataclass(frozen=True, kw_only=True)
class WeightedSeedSampling:
    """Weighted seed draws, with optional class and per-row sample weights.

    Args:
        num_samples: Number of draws, with replacement.
        false_weight: Optional multiplier for the false class.
        true_weight: Optional multiplier for the true class.
        use_sample_weights: Whether to use per-row sample-weight sidecars.
    """

    num_samples: int
    false_weight: float | None = None
    true_weight: float | None = None
    use_sample_weights: bool = False


@dataclass(frozen=True, kw_only=True)
class SamplingOptions:
    """Configure one stage, epoch and worker shard of a runtime index.

    Attribute assignment is frozen. Projections and selected IDs retain their
    dictionary/list formats; callers must not mutate them during iteration.

    Args:
        stage: Task manifest stage to sample.
        batch_size: Maximum number of seeds in each batch.
        sampling_plan: Native graph traversal plan.
        num_workers: Number of logical shards.
        worker_id: Zero-based logical shard ordinal.
        shuffle: Whether to shuffle seed rows deterministically.
        epoch: Logical epoch used for seed ordering.
        rnd_seed: Shuffle seed; omitted inherits the opened index seed. An explicit
            None disables the seed and cannot be combined with shuffle.
        drop_last: Whether to omit the final incomplete seed batch.
        fused: Whether to fuse traversal and feature materialization.
        swap_edge_direction: Whether output edges point toward sampled seeds.
        neg_rhs_node_type: Optional negative-sampling destination type.
        neg_amount_m: Number of negatives per positive seed.
        neg_rnd_seed: Optional negative-sampling seed.
        replace: Whether graph neighbors are sampled with replacement.
        disjoint: Whether seed neighborhoods are kept separate.
        adaptive: Whether to use adaptive graph sampling.
        entity_node_type: Optional entity node type override.
        target_node_type: Optional target node type override.
        rhs_seed_node_type: Optional RHS candidate-domain seed type.
        rhs_seed_time: Optional RHS cutoff timestamp.
        group_by_timestamp: Whether to group seeds into timestamp buckets.
        weighted_seed_sampling: Optional weighted seed draw configuration.
        selected_stage_node_ids: Optional ordered seed IDs, retaining duplicates.
        pseudo_seed_time: Optional static-link pseudo-time cutoff.
        disable_pseudo_time: Whether to disable static-link pseudo-time filtering.
        projections: Feature columns by node type. None selects all features;
            an empty column sequence skips that type's features, not its edges.
    """

    stage: str
    batch_size: int
    sampling_plan: SamplingPlan = field(default_factory=SamplingPlan)
    num_workers: int = 1
    worker_id: int = 0
    shuffle: bool = False
    epoch: int = 0
    rnd_seed: int | None | InheritEngineSeed = InheritEngineSeed.DEFAULT
    drop_last: bool = False
    fused: bool | None = None
    swap_edge_direction: bool = True
    neg_rhs_node_type: str | None = None
    neg_amount_m: int | None = None
    neg_rnd_seed: int | None = None
    replace: bool = False
    disjoint: bool = True
    adaptive: bool = False
    entity_node_type: str | None = None
    target_node_type: str | None = None
    rhs_seed_node_type: str | None = None
    rhs_seed_time: int | None = None
    group_by_timestamp: bool = False
    weighted_seed_sampling: WeightedSeedSampling | None = None
    selected_stage_node_ids: list[int] | None = None
    pseudo_seed_time: int | None = None
    disable_pseudo_time: bool = False
    projections: dict[str, list[str]] | None = None
