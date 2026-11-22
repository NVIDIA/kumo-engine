# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable sampling requests and encoded graph payload types."""

from __future__ import annotations

from dataclasses import dataclass

from diskgraph.sampling_options import SamplingOptions

from kumo_core.diskgraph.hetero_wire import HeteroMetadata, TensorBuffer

WIRE_VERSION = 2
SAMPLING_VERSION = 1


class SamplingContractError(RuntimeError):
    """A request or response violates the sampling identity or batch contract."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """Snapshot, task-manifest, and fitted-mapping digests shared by replicas."""

    snapshot_digest: str
    task_digest: str
    mapping_digest: str


@dataclass(frozen=True)
class BatchRequest:
    """An explicit logical batch; actor placement never participates in its RNG."""

    epoch: int
    logical_shard: int
    logical_shard_count: int
    batch_ordinal: int
    seed_ids: tuple[int, ...]
    options: SamplingOptions
    engine_seed: int
    num_threads: int
    wire_version: int = WIRE_VERSION
    sampling_version: int = SAMPLING_VERSION

    def validate(self) -> None:
        """Validate request metadata and sampling options.

        Raises:
            SamplingContractError: If versions, seeds, or RNG settings are invalid.
        """
        if (
            self.wire_version != WIRE_VERSION
            or self.sampling_version != SAMPLING_VERSION
        ):
            raise SamplingContractError("unsupported sampling or wire version")
        integers = (
            self.engine_seed,
            self.logical_shard,
            self.logical_shard_count,
            self.num_threads,
            self.epoch,
            self.batch_ordinal,
        )
        if (
            not isinstance(self.seed_ids, tuple)
            or not isinstance(self.options, SamplingOptions)
            or any(type(value) is not int for value in (*integers, *self.seed_ids))
        ):
            raise SamplingContractError(
                "request requires tuple seeds, SamplingOptions, and integers"
            )
        if (
            not self.seed_ids
            or any(not 0 <= seed < 2**64 for seed in self.seed_ids)
            or not 0 <= self.engine_seed < 2**64
            or not 0 <= self.logical_shard < self.logical_shard_count
            or self.num_threads < 1
            or min(
                self.epoch,
                self.logical_shard,
                self.batch_ordinal,
            )
            < 0
        ):
            raise SamplingContractError("invalid seed or logical RNG identity")
        options = self.options
        if type(options.batch_size) is not int or options.batch_size < 1:
            raise SamplingContractError("native batch size must be a positive integer")
        if (
            options.num_workers != 1
            or options.worker_id != 0
            or options.shuffle
            or options.epoch != 0
            or options.drop_last
            or options.group_by_timestamp
            or options.weighted_seed_sampling is not None
            or options.selected_stage_node_ids is not None
        ):
            raise SamplingContractError("feeder options cannot own a seed schedule")
        if len(self.seed_ids) > options.batch_size:
            raise SamplingContractError("request exceeds native batch size")
        if options.neg_rhs_node_type and options.neg_rnd_seed is None:
            raise SamplingContractError("remote negatives require neg_rnd_seed")


HeteroBatchPayload = tuple[HeteroMetadata, list[TensorBuffer]]
