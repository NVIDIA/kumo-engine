# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable identities and complete native batch envelopes for sampling backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diskgraph.sampling_options import SamplingOptions

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from kumo_core.diskgraph.hetero_wire import TensorBuffer


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
class ConsumerToken:
    """Independently fenced consumer within a job and trainer rank."""

    job: str
    trainer_rank: int
    consumer_kind: str
    generation: int


@dataclass(frozen=True)
class BatchRequest:
    """An explicit logical batch; actor placement never participates in its RNG."""

    artifacts: ArtifactIdentity
    consumer: ConsumerToken
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
            self.consumer.generation,
            self.consumer.trainer_rank,
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
                self.consumer.generation,
                self.consumer.trainer_rank,
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


@dataclass(frozen=True)
class HeteroBatchPayload:
    """Actor-encoded graph metadata and separate NumPy tensor buffers.

    Args:
        request: Logical request and artifact identity for the encoded batch.
        metadata: Versioned MessagePack graph layout and tensor descriptors.
        buffers: Contiguous NumPy views owning the tensor data.
        sampling_seconds: Actor sampling, encoding, and packing duration.
    """

    request: BatchRequest
    metadata: bytes
    buffers: list[TensorBuffer]
    sampling_seconds: float

    def decode(self, expected: BatchRequest) -> HeteroData:
        """Verify request identity and reconstruct the encoded graph.

        Args:
            expected: Request whose response is being consumed.

        Returns:
            HeteroData retaining the received NumPy buffer owners.

        Raises:
            SamplingContractError: If request identity differs.
            TypeError: If buffer types are unsupported.
            ValueError: If wire metadata or tensor layout is invalid.
        """
        from kumo_core.diskgraph.hetero_wire import unpack_python_hetero

        self.validate(expected)
        return unpack_python_hetero(self.metadata, self.buffers)

    def validate(self, expected: BatchRequest) -> None:
        """Check response identity before decoding its metadata and buffers.

        Args:
            expected: Logical request whose response is being consumed.

        Raises:
            SamplingContractError: If request identity differs.
        """
        expected.validate()
        if self.request != expected:
            raise SamplingContractError("wrong request, artifact, or phase generation")
