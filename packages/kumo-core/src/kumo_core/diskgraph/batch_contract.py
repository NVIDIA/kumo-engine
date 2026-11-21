# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable identities and complete native batch envelopes for sampling backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import NotRequired, TypedDict

import pyarrow as pa
from diskgraph.sampling_options import SamplingOptions


class NativeTask(TypedDict):
    """Required seed identity arrays in a complete native task block."""

    input_ids: pa.Array
    tt_stage_node_ids: pa.Array
    seed_node_ids: pa.Array


class NativeEdge(TypedDict):
    """Typed relation and aligned source/destination arrays."""

    edge_type: dict[str, str]
    src: pa.Array
    dst: pa.Array


class NativeEdgeCounts(TypedDict):
    """Per-hop sampling counts for one relation."""

    edge_type: dict[str, str]
    counts: pa.Array


class NativeGraph(TypedDict):
    """Required topology and provenance arrays for one graph channel."""

    node_ids: dict[str, pa.Array]
    batch_dict: dict[str, pa.Array]
    edge_index_dict: list[NativeEdge]
    num_sampled_nodes_dict: dict[str, pa.Array]
    num_sampled_edges_dict: list[NativeEdgeCounts]


class NativeFeatureBlock(TypedDict):
    """Arrow feature chunks and the row count used to verify completeness."""

    num_rows: int
    record_batches: list[pa.RecordBatch]
    node_ids: NotRequired[pa.Array]


class NativeBatch(TypedDict):
    """Required sections of the complete native batch transported by this backend.

    Native dictionaries retain additional task/history fields without repacking.
    """

    task: NativeTask
    graphs: dict[str, NativeGraph]
    raw_features: dict[str, NativeFeatureBlock]


WIRE_VERSION = 2
SAMPLING_VERSION = 1


class SamplingContractError(RuntimeError):
    """A request or response violates the sampling identity or batch contract."""


def content_digest(content: bytes) -> str:
    """Return the SHA-256 identity of configuration or manifest bytes.

    Args:
        content: Exact immutable artifact bytes.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return sha256(content).hexdigest()


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
class BatchPayload:
    """One complete batch retaining Arrow/native buffer owners until conversion."""

    request: BatchRequest
    batch: NativeBatch
    sampling_seconds: float

    def validate(self, expected: BatchRequest) -> NativeBatch:
        """Check identity and required complete-batch fields before tensor mapping.

        Args:
            expected: Issued logical request.

        Returns:
            Existing native batch mapping, without repacking its Arrow buffers.

        Raises:
            SamplingContractError: If identity or native batch structure differs.
        """
        expected.validate()
        options = expected.options
        if self.request != expected:
            raise SamplingContractError("wrong request, artifact, or phase generation")
        if not all(key in self.batch for key in ("task", "graphs", "raw_features")):
            raise SamplingContractError("incomplete native batch")
        sections: Mapping[str, object] = self.batch
        if not all(
            isinstance(sections[key], Mapping)
            for key in ("task", "graphs", "raw_features")
        ):
            raise SamplingContractError("native batch sections must be mappings")
        task = self.batch["task"]
        graph = self.batch["graphs"].get("main")
        if graph is None or not all(
            key in graph
            for key in (
                "node_ids",
                "batch_dict",
                "edge_index_dict",
                "num_sampled_nodes_dict",
                "num_sampled_edges_dict",
            )
        ):
            raise SamplingContractError("incomplete graph channel")
        for name in ("input_ids", "tt_stage_node_ids", "seed_node_ids"):
            if not isinstance(task.get(name), pa.Array):
                raise SamplingContractError(f"missing native task array: {name}")
        seed_ids = (
            task["seed_node_ids"]
            if options.rhs_seed_node_type
            else task["tt_stage_node_ids"]
        )
        if not seed_ids.equals(pa.array(expected.seed_ids, type=pa.uint64())):
            raise SamplingContractError("response seed membership/order differs")
        for channel in self.batch["graphs"].values():
            for node_type, ids in channel["node_ids"].items():
                provenance = channel["batch_dict"].get(node_type)
                if provenance is None or len(ids) != len(provenance):
                    raise SamplingContractError("node provenance length differs")
            for edge in channel["edge_index_dict"]:
                if len(edge["src"]) != len(edge["dst"]):
                    raise SamplingContractError("edge endpoint length differs")
        for bank in self.batch["raw_features"].values():
            batches = bank.get("record_batches")
            if (
                not isinstance(batches, list)
                or any(not isinstance(batch, pa.RecordBatch) for batch in batches)
                or sum(batch.num_rows for batch in batches) != bank.get("num_rows")
            ):
                raise SamplingContractError("incomplete native feature bank")
        return self.batch
