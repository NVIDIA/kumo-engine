# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable identities and complete native batch envelopes for sampling backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import cached_property
from hashlib import sha256
from typing import NotRequired, TypedDict, Unpack, cast

import orjson
import pyarrow as pa

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class WeightedSeedSamplingArguments(TypedDict):
    """Weights and draw count accepted by the native weighted seed scheduler."""

    false_weight: NotRequired[float]
    true_weight: NotRequired[float]
    num_samples: int
    use_sample_weights: NotRequired[bool]


class SamplingArguments(TypedDict, total=False):
    """Native keyword argument dictionaries accepted at the compatibility boundary."""

    stage: str
    sampling_plan: dict[str, JsonValue] | str
    num_workers: int
    worker_id: int
    shuffle: bool
    epoch: int
    batch_size: int
    rnd_seed: int | None
    drop_last: bool
    fused: bool | None
    swap_edge_direction: bool
    neg_rhs_node_type: str | None
    neg_amount_m: int | None
    neg_rnd_seed: int | None
    replace: bool
    disjoint: bool
    adaptive: bool
    entity_node_type: str | None
    target_node_type: str | None
    rhs_seed_node_type: str | None
    rhs_seed_time: int | None
    group_by_timestamp: bool
    weighted_seed_sampling: WeightedSeedSamplingArguments | None
    selected_stage_node_ids: list[int] | None
    pseudo_seed_time: int | None
    disable_pseudo_time: bool
    projections: dict[str, list[str]] | None


@dataclass(frozen=True)
class FrozenJsonObject:
    """An immutable JSON object whose array children are tuples.

    Args:
        entries: Object keys and values; nested containers are copied and frozen.

    Raises:
        TypeError: If keys or values are not JSON-compatible.
        ValueError: If object keys are duplicated.
    """

    entries: tuple[tuple[str, FrozenJsonValue], ...]

    def __post_init__(self) -> None:
        if any(not isinstance(key, str) for key, _ in self.entries):
            raise TypeError("sampling plan keys must be strings")
        if len({key for key, _ in self.entries}) != len(self.entries):
            raise ValueError("sampling plan keys must be unique")
        object.__setattr__(
            self,
            "entries",
            tuple(sorted((key, _freeze_json(value)) for key, value in self.entries)),
        )


FrozenJsonValue = (
    None | bool | int | float | str | tuple["FrozenJsonValue", ...] | FrozenJsonObject
)


def _freeze_json(value: object) -> FrozenJsonValue:
    if isinstance(value, FrozenJsonObject):
        return value
    if isinstance(value, dict):
        return FrozenJsonObject(
            tuple((key, _freeze_json(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("sampling plans require JSON-compatible values")


def _thaw_json(value: FrozenJsonValue) -> JsonValue:
    if isinstance(value, FrozenJsonObject):
        return {key: _thaw_json(item) for key, item in value.entries}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, kw_only=True)
class WeightedSeedSampling:
    """Immutable class/sample weights and draw count for native scheduling."""

    num_samples: int
    false_weight: float | None = None
    true_weight: float | None = None
    use_sample_weights: bool = False

    def to_native_kwargs(self) -> WeightedSeedSamplingArguments:
        """Return native settings, preserving whether class weights were omitted.

        Returns:
            Fresh native keyword dictionary with absent weights omitted.
        """
        return cast(
            WeightedSeedSamplingArguments,
            {
                item.name: value
                for item in fields(self)
                if (value := getattr(self, item.name)) is not None
            },
        )


@dataclass(frozen=True, kw_only=True)
class SamplingOptions:
    """Owned immutable sampling configuration, independent of native dictionaries.

    Use ``from_kwargs`` to snapshot existing native iterator arguments. Nested
    plans, projections, selections, and weights cannot change after construction.

    Raises:
        TypeError: If a plan or weight configuration is not immutable.
    """

    stage: str
    batch_size: int
    sampling_plan: FrozenJsonObject | str = FrozenJsonObject((("layers", ()),))
    num_workers: int = 1
    worker_id: int = 0
    shuffle: bool = False
    epoch: int = 0
    rnd_seed: int | None = None
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
    selected_stage_node_ids: tuple[int, ...] | None = None
    pseudo_seed_time: int | None = None
    disable_pseudo_time: bool = False
    projections: tuple[tuple[str, tuple[str, ...]], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sampling_plan, (FrozenJsonObject, str)):
            raise TypeError(
                "use SamplingOptions.from_kwargs to snapshot a mutable plan"
            )
        if self.weighted_seed_sampling is not None and not isinstance(
            self.weighted_seed_sampling, WeightedSeedSampling
        ):
            raise TypeError("weights must be an immutable WeightedSeedSampling")
        if self.selected_stage_node_ids is not None:
            object.__setattr__(
                self, "selected_stage_node_ids", tuple(self.selected_stage_node_ids)
            )
        if self.projections is not None:
            object.__setattr__(
                self,
                "projections",
                tuple(
                    sorted((name, tuple(columns)) for name, columns in self.projections)
                ),
            )

    @classmethod
    def from_kwargs(cls, **arguments: Unpack[SamplingArguments]) -> SamplingOptions:
        """Copy native keyword arguments into an immutable owned configuration.

        Args:
            **arguments: Existing native iterator keyword arguments.

        Returns:
            Configuration independent of the caller's mutable containers.

        Raises:
            KeyError: If stage or batch size is missing.
            TypeError: If the sampling plan is not a JSON object or string.
        """
        plan = _freeze_json(arguments.get("sampling_plan", {"layers": []}))
        if not isinstance(plan, (FrozenJsonObject, str)):
            raise TypeError("sampling_plan must be an object or string")
        weights = arguments.get("weighted_seed_sampling")
        selected = arguments.get("selected_stage_node_ids")
        projections = arguments.get("projections")
        return cls(
            stage=arguments["stage"],
            batch_size=arguments["batch_size"],
            sampling_plan=plan,
            weighted_seed_sampling=WeightedSeedSampling(**weights)
            if weights is not None
            else None,
            selected_stage_node_ids=tuple(selected) if selected is not None else None,
            projections=tuple(
                (name, tuple(columns)) for name, columns in projections.items()
            )
            if projections is not None
            else None,
            num_workers=arguments.get("num_workers", 1),
            worker_id=arguments.get("worker_id", 0),
            shuffle=arguments.get("shuffle", False),
            epoch=arguments.get("epoch", 0),
            rnd_seed=arguments.get("rnd_seed", None),
            drop_last=arguments.get("drop_last", False),
            fused=arguments.get("fused", None),
            swap_edge_direction=arguments.get("swap_edge_direction", True),
            neg_rhs_node_type=arguments.get("neg_rhs_node_type", None),
            neg_amount_m=arguments.get("neg_amount_m", None),
            neg_rnd_seed=arguments.get("neg_rnd_seed", None),
            replace=arguments.get("replace", False),
            disjoint=arguments.get("disjoint", True),
            adaptive=arguments.get("adaptive", False),
            entity_node_type=arguments.get("entity_node_type", None),
            target_node_type=arguments.get("target_node_type", None),
            rhs_seed_node_type=arguments.get("rhs_seed_node_type", None),
            rhs_seed_time=arguments.get("rhs_seed_time", None),
            group_by_timestamp=arguments.get("group_by_timestamp", False),
            pseudo_seed_time=arguments.get("pseudo_seed_time", None),
            disable_pseudo_time=arguments.get("disable_pseudo_time", False),
        )

    def to_native_kwargs(self) -> SamplingArguments:
        """Return fresh native dictionaries and lists without exposing owned state.

        Returns:
            Native iterator keyword arguments; mutating them cannot alter this object.
        """
        arguments = cast(
            SamplingArguments,
            {item.name: getattr(self, item.name) for item in fields(self) if item.init},
        )
        plan = _thaw_json(self.sampling_plan)
        arguments["sampling_plan"] = cast(dict[str, JsonValue] | str, plan)
        arguments["selected_stage_node_ids"] = (
            list(self.selected_stage_node_ids)
            if self.selected_stage_node_ids is not None
            else None
        )
        arguments["projections"] = (
            {name: list(columns) for name, columns in self.projections}
            if self.projections is not None
            else None
        )
        arguments["weighted_seed_sampling"] = (
            self.weighted_seed_sampling.to_native_kwargs()
            if self.weighted_seed_sampling is not None
            else None
        )
        return arguments

    @cached_property
    def digest(self) -> str:
        """Canonical identity, computed only for configurations needing a cache key."""
        return content_digest(canonical_json(self.to_native_kwargs()))

    @property
    def has_seed_schedule(self) -> bool:
        """Whether this configuration selects or shards a trainer-owned schedule."""
        return (
            self.num_workers != 1
            or self.worker_id != 0
            or self.shuffle
            or self.epoch != 0
            or self.drop_last
            or self.group_by_timestamp
            or self.weighted_seed_sampling is not None
            or self.selected_stage_node_ids is not None
        )

    def sampling_only(self) -> SamplingOptions:
        """Return immutable traversal settings without trainer-owned seed selection.

        Returns:
            This object when already sampling-only, otherwise a normalized copy.
        """
        if not self.has_seed_schedule:
            return self
        return replace(
            self,
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            drop_last=False,
            group_by_timestamp=False,
            weighted_seed_sampling=None,
            selected_stage_node_ids=None,
        )


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


def canonical_json(value: object) -> bytes:
    """Return sorted JSON bytes for an immutable request configuration.

    Args:
        value: JSON-compatible configuration.

    Returns:
        Canonical UTF-8 JSON bytes.

    Raises:
        TypeError: If the configuration cannot be represented as JSON.
    """
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


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

    def __post_init__(self) -> None:
        if isinstance(self.options, SamplingOptions):
            # Carry the cached key through Ray, avoiding per-batch plan serialization.
            _ = self.options.digest

    @property
    def request_id(self) -> str:
        """Stable identity retained by every attempt of this logical batch."""
        return content_digest(
            canonical_json(
                [
                    self.artifacts.snapshot_digest,
                    self.artifacts.task_digest,
                    self.artifacts.mapping_digest,
                    self.consumer.job,
                    self.consumer.trainer_rank,
                    self.consumer.consumer_kind,
                    self.consumer.generation,
                    self.epoch,
                    self.logical_shard,
                    self.logical_shard_count,
                    self.batch_ordinal,
                    self.seed_ids,
                    self.sampling_plan_digest,
                    self.engine_seed,
                    self.num_threads,
                    self.wire_version,
                    self.sampling_version,
                ]
            )
        )

    @property
    def sampling_plan_digest(self) -> str:
        """Identity of traversal, projection, supervision, and output options."""
        return self.options.digest

    def validate(self) -> SamplingOptions:
        """Validate immutable metadata and return typed native sampling options.

        Returns:
            Sampling options suitable for native context construction.

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
                "request requires immutable seeds/options and integers"
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
        if options.has_seed_schedule:
            raise SamplingContractError("feeder options cannot own a seed schedule")
        if len(self.seed_ids) > options.batch_size:
            raise SamplingContractError("request exceeds native batch size")
        if options.neg_rhs_node_type and options.neg_rnd_seed is None:
            raise SamplingContractError("remote negatives require neg_rnd_seed")
        return options


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
        options = expected.validate()
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
