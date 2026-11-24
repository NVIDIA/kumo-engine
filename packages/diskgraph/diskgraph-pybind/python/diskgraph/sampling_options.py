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
        num_samples: Required number of seed draws, with replacement.
        false_weight: Defaults to None: no explicit false-class weighting;
            its effective multiplier is 1.0.
        true_weight: Defaults to None: no explicit true-class weighting;
            its effective multiplier is 1.0.
        use_sample_weights: Defaults to False: draw uniformly within each
            binary-class bucket. True uses per-row sample-weight sidecars.
            With both class weights None, use the all-rows weight sidecar;
            setting either class weight selects binary-class weight sidecars.
            Thus explicit 1.0 and None can require different sidecars.
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
        stage: Required task manifest stage to sample.
        batch_size: Required positive maximum number of seeds per batch.
        sampling_plan: Defaults to a fresh SamplingPlan(): zero traversal
            layers, so only seeds are sampled. Pass layers to sample neighbors.
        num_workers: Defaults to 1 logical shard: process the entire stage.
        worker_id: Defaults to 0: the first and, by default, only shard.
        shuffle: Defaults to False: do not shuffle seed ordering.
        epoch: Defaults to 0: the initial seed-schedule epoch.
        rnd_seed: Defaults to InheritEngineSeed.DEFAULT: inherit the opened
            index's seed. Explicit None disables the shuffle seed and is invalid
            with shuffle=True. Zero is an explicit seed, not inheritance.
        drop_last: Defaults to False: retain the final incomplete seed batch.
        fused: Defaults to None: let the engine choose fused feature fetching;
            non-empty projections select non-fused mode. Explicit True ignores
            non-empty projections with a warning.
        swap_edge_direction: Defaults to True: output edges point toward seeds.
        neg_rhs_node_type: Defaults to None: disable global negative sampling.
        neg_amount_m: Defaults to None: no negative count. Required when
            neg_rhs_node_type is set; specifies negatives per positive seed.
        neg_rnd_seed: Defaults to None: negative draws use thread-local random
            state, independently of rnd_seed. Set a seed for reproducible draws.
        replace: Defaults to False: sample graph neighbors without replacement.
            This does not control weighted seed draws, which use replacement.
        disjoint: Defaults to True: keep seed neighborhoods separate.
        adaptive: Defaults to False: do not enable adaptive graph sampling.
        entity_node_type: Defaults to None: no entity type is supplied.
            Normal stage graph sampling requires an explicit type to resolve
            the entity_node_id rowpack column; None raises a native config
            error on batch consumption. RHS catalog mode uses its own type.
        target_node_type: Defaults to None: pass through raw targets instead
            of resolving ragged link-prediction labels for this node type.
        rhs_seed_node_type: Defaults to None: seed from the stage table.
            Setting a type instead seeds its full RHS catalog (NodeRange mode).
        rhs_seed_time: Defaults to None: no fixed RHS anchor timestamp.
            Required when rhs_seed_node_type is set.
        group_by_timestamp: Defaults to False: batch by seed count rather
            than timestamp buckets.
        weighted_seed_sampling: Defaults to None: visit stage seeds without
            weighted replacement draws. Set WeightedSeedSampling to enable them.
        selected_stage_node_ids: Defaults to None: select all stage rows.
            A list restricts selection and retains duplicates; its order is
            preserved when shuffle=False.
        pseudo_seed_time: Defaults to None: infer static-link pseudo-time from
            the stage when sidecars exist (train/val=1, test=2, predict=none).
        disable_pseudo_time: Defaults to False: retain automatic pseudo-time
            filtering when present. True disables it and cannot be combined
            with an explicit pseudo_seed_time.
        projections: Defaults to None: select all feature columns for all types.
            Missing types also select all columns; an empty column list skips
            that type's features but preserves its edges and graph traversal.
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
