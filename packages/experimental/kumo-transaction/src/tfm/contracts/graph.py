# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared/disjoint graph inputs with separate per-anchor future targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import ClassVar, Literal

import torch
from torch import Tensor

from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.validation import ValidatedBatch, ValidationMetadata, require_cpu

EdgeType = tuple[str, str, str]


def _edge_properties(index: Tensor, source_size: int, target_size: int) -> tuple[bool, bool]:
    """Return (duplicate pairs, at most one parent per receiver) for bounded CPU edges."""
    if index.shape[1] < 2:
        return False, True
    # Bounds were checked by GraphBatch. Receiver-first ordering makes both
    # duplicate pairs and repeated receivers adjacent, sharing the sorting work.
    if source_size * target_size > torch.iinfo(torch.int64).max:
        order = torch.argsort(index[0], stable=True)
        order = order[torch.argsort(index[1, order], stable=True)]
        receivers, sources = index[1, order], index[0, order]
        repeated_receiver = receivers[1:] == receivers[:-1]
        duplicate = repeated_receiver & (sources[1:] == sources[:-1])
    else:
        keys, order = torch.sort(index[1] * source_size + index[0])
        receivers = index[1, order]
        repeated_receiver = receivers[1:] == receivers[:-1]
        duplicate = keys[1:] == keys[:-1]
    return bool(duplicate.any()), not bool(repeated_receiver.any())


def _has_duplicate_edges(index: Tensor, source_size: int, target_size: int) -> bool:
    """Check duplicate pairs using the shared receiver-first classifier."""
    return _edge_properties(index, source_size, target_size)[0]


STATIC_TIME = -(2**63)


def utc_us(value: datetime) -> int:
    """Exact UTC microseconds; reject naive clocks instead of guessing availability."""
    if value.tzinfo is None:
        raise ValueError("Graph clocks must be timezone-aware")
    from datetime import UTC

    delta = value - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


@dataclass(frozen=True)
class GraphAnchor(ValidationMetadata):
    """Complete identity of one cutoff-level graph/sequence example."""

    cache_validation_layout: ClassVar[bool] = True

    sample_id: str
    entity_id: str
    cutoff: datetime
    observation_start: datetime | None = None

    def __post_init__(self) -> None:
        utc_us(self.cutoff)
        if not self.sample_id or not self.entity_id:
            raise ValueError("Graph sample and entity IDs must be nonempty")
        if self.observation_start is not None and utc_us(self.observation_start) > utc_us(
            self.cutoff
        ):
            raise ValueError("Observation start must not follow cutoff")

    def as_dict(self) -> dict[str, str | datetime | None]:
        """Semantic sample identity, excluding ephemeral validation caches."""
        return {
            "sample_id": self.sample_id,
            "entity_id": self.entity_id,
            "cutoff": self.cutoff,
            "observation_start": self.observation_start,
        }

    def __reduce__(self) -> tuple[type[GraphAnchor], tuple[str, str, datetime, datetime | None]]:
        # Runtime layout caches contain weak references; workers rebuild them.
        cached = self._metadata_layout
        if cached is not None:
            cached.record.check()
        return type(self), (self.sample_id, self.entity_id, self.cutoff, self.observation_start)


@dataclass(frozen=True)
class GraphBatch(ValidatedBatch):
    """Explicit shared or disjoint heterogeneous graphs, edges source -> receiver.

    Features are finite floats [nodes, fields]. node_sample assigns each node to
    its component: anchor index when disjoint, zero for a shared component.
    Shared anchors must have identical cutoff/observation windows. Roots remain
    in sample order and may repeat when shared; targets are never deduplicated. Availability clocks include links as well as nodes; static rows
    use STATIC_TIME only after the data schema declares the table immutable.
    A root per anchor is mandatory, including empty-neighborhood/cold roots.
    Optional trimming metadata is all-or-none: node_hop gives ascending minimum
    root-distance shells, num_sampled_nodes counts shells 0..depth per node type,
    and num_sampled_edges counts receiver shells 0..depth-1 per directed role.
    Counts include zeroes, and roots occupy the hop-zero prefix. Native inputs
    may omit metadata only when the consuming encoder explicitly disables trim.
    Construct on CPU, then use .to(device). Mappings are read-only snapshots;
    tensor mutation invalidates this batch. single_parent_relations caches the
    roles with unique receivers, independent of Engine's declared FK directions.
    Untracked external/.data writes are forbidden; rebuild on CPU after edits.
    """

    execution_fields: ClassVar[frozenset[str]] = frozenset(
        ("node_features", "edge_index", "root_index")
    )

    anchors: tuple[GraphAnchor, ...]
    root_type: str
    node_features: Mapping[str, Tensor]
    node_sample: Mapping[str, Tensor]
    node_available_us: Mapping[str, Tensor]
    edge_index: Mapping[EdgeType, Tensor]
    edge_available_us: Mapping[EdgeType, Tensor]
    root_index: Tensor
    sharing: Literal["disjoint", "shared"] = "disjoint"
    node_hop: Mapping[str, Tensor] | None = None
    num_sampled_nodes: Mapping[str, tuple[int, ...]] | None = None
    num_sampled_edges: Mapping[EdgeType, tuple[int, ...]] | None = None
    window_exempt_types: frozenset[str] = frozenset()

    single_parent_relations: frozenset[EdgeType] = field(
        init=False, repr=False, default=frozenset()
    )

    def __post_init__(self) -> None:
        require_cpu(self)
        if not self.anchors or len({a.sample_id for a in self.anchors}) != self.batch_size:
            raise ValueError("Graph anchors must have unique sample IDs and be nonempty")
        if self.sharing not in {"disjoint", "shared"}:
            raise ValueError("Graph sharing must be disjoint or shared")
        if (
            self.sharing == "shared"
            and len(
                {
                    (
                        utc_us(a.cutoff),
                        None if a.observation_start is None else utc_us(a.observation_start),
                    )
                    for a in self.anchors
                }
            )
            != 1
        ):
            raise ValueError("Shared graphs require identical cutoffs and observation windows")
        types = set(self.node_features)
        if not self.window_exempt_types.issubset(types):
            raise ValueError(
                "Observation-window exemptions must name declared structural node types"
            )
        if types != set(self.node_sample) or types != set(self.node_available_us):
            raise ValueError("Graph node dictionaries must have identical types")
        if self.root_type not in types or self.root_index.shape != (self.batch_size,):
            raise ValueError("Graph needs one root index per anchor")
        if set(self.edge_index) != set(self.edge_available_us):
            raise ValueError("Every edge needs an availability clock")
        cutoffs = torch.tensor([utc_us(a.cutoff) for a in self.anchors], dtype=torch.long)
        starts = torch.tensor(
            [
                STATIC_TIME if a.observation_start is None else utc_us(a.observation_start)
                for a in self.anchors
            ],
            dtype=torch.long,
        )
        for name, x in self.node_features.items():
            owners, times = self.node_sample[name], self.node_available_us[name]
            if x.ndim != 2 or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
                raise ValueError("Node features must be finite floating [nodes, fields]")
            if (
                owners.dtype != torch.long
                or times.dtype != torch.long
                or owners.shape != (len(x),)
                or times.shape != owners.shape
            ):
                raise ValueError("Node owners and clocks must be int64 [nodes]")
            if owners.numel() and (
                int(owners.min()) < 0
                or int(owners.max()) >= (1 if self.sharing == "shared" else self.batch_size)
            ):
                raise ValueError("Node owner outside anchor range")
            if bool((times > cutoffs[owners]).any()):
                raise ValueError("Future node availability in graph inputs")
            if name not in self.window_exempt_types and bool(
                ((times != STATIC_TIME) & (times < starts[owners])).any()
            ):
                raise ValueError("Node predates observation window")
        self._indices(self.root_index, len(self.node_features[self.root_type]))
        owners = self.node_sample[self.root_type][self.root_index]
        expected_owners = (
            torch.zeros(self.batch_size, dtype=torch.long, device=owners.device)
            if self.sharing == "shared"
            else torch.arange(self.batch_size, device=owners.device)
        )
        if not torch.equal(owners, expected_owners):
            raise ValueError("Root order must match anchor order")
        single_parent = set()
        for (source, role, target), edges in self.edge_index.items():
            if (
                source not in types
                or target not in types
                or not role
                or edges.ndim != 2
                or edges.shape[0] != 2
            ):
                raise ValueError("Unknown edge type or invalid edge shape")
            self._indices(edges[0], len(self.node_features[source]))
            self._indices(edges[1], len(self.node_features[target]))
            source_owner = self.node_sample[source][edges[0]]
            if not torch.equal(source_owner, self.node_sample[target][edges[1]]):
                raise ValueError("An edge crosses anchor graphs")
            times = self.edge_available_us[(source, role, target)]
            if times.dtype != torch.long or times.shape != (edges.shape[1],):
                raise ValueError("Edge clocks must be int64 [edges]")
            duplicate, single_receiver_parent = _edge_properties(
                edges, len(self.node_features[source]), len(self.node_features[target])
            )
            if duplicate:
                raise ValueError("Duplicate graph edges")
            if single_receiver_parent:
                single_parent.add((source, role, target))
            if bool((times > cutoffs[source_owner]).any()):
                raise ValueError("Future edge availability in graph inputs")

        self.validate_hops()
        object.__setattr__(self, "single_parent_relations", frozenset(single_parent))
        for name in (
            "node_features",
            "node_sample",
            "node_available_us",
            "edge_index",
            "edge_available_us",
            "node_hop",
            "num_sampled_nodes",
            "num_sampled_edges",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        self._seal()

    def validate_hops(self) -> None:
        """Validate Engine prefix trimming, ordering and all layer index bounds."""
        source = self._execution_source
        if source is not None:
            self.assert_validated()
            source.validate_hops()
            return
        metadata = (self.node_hop, self.num_sampled_nodes, self.num_sampled_edges)
        if all(value is None for value in metadata):
            return
        if any(value is None for value in metadata):
            raise ValueError("Hop metadata requires node hops and both count dictionaries")
        assert (
            self.node_hop is not None
            and self.num_sampled_nodes is not None
            and self.num_sampled_edges is not None
        )
        if (
            set(self.node_hop) != set(self.node_features)
            or set(self.num_sampled_nodes) != set(self.node_features)
            or set(self.num_sampled_edges) != set(self.edge_index)
        ):
            raise ValueError("Hop-count schema differs from graph schema")
        lengths = {len(c) for c in self.num_sampled_nodes.values()}
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise ValueError("Node counts need one root shell plus at least one hop")
        depth = next(iter(lengths)) - 1
        connected = {
            name: torch.zeros(len(x), dtype=torch.bool, device=x.device)
            for name, x in self.node_features.items()
        }
        for name, counts in self.num_sampled_nodes.items():
            if any(type(c) is not int or c < 0 for c in counts) or sum(counts) != len(
                self.node_features[name]
            ):
                raise ValueError("Invalid node hop counts")
            hops = self.node_hop[name]
            expected = torch.repeat_interleave(
                torch.arange(depth + 1, device=hops.device),
                torch.tensor(counts, device=hops.device),
            )
            if hops.dtype != torch.long or not torch.equal(hops, expected):
                raise ValueError("Nodes are not in declared hop-prefix order")
            if name != self.root_type and counts[0] != 0:
                raise ValueError("Only prediction roots may occupy hop zero")
            connected[name] |= hops == 0
        n_roots = self.num_sampled_nodes[self.root_type][0]
        if not torch.equal(
            self.root_index.unique(sorted=True),
            torch.arange(n_roots, device=self.root_index.device),
        ):
            raise ValueError("Prediction roots must exactly occupy the hop-zero prefix")
        for (source, role, target), counts in self.num_sampled_edges.items():
            edges = self.edge_index[source, role, target]
            if (
                len(counts) != depth
                or any(type(c) is not int or c < 0 for c in counts)
                or sum(counts) != edges.shape[1]
            ):
                raise ValueError("Invalid edge hop counts")
            src_hop = self.node_hop[source][edges[0]]
            dst_hop = self.node_hop[target][edges[1]]
            expected = torch.repeat_interleave(
                torch.arange(depth, device=dst_hop.device),
                torch.tensor(counts, device=dst_hop.device),
            )
            if not torch.equal(dst_hop, expected) or bool((src_hop > dst_hop + 1).any()):
                raise ValueError("Edges are not valid in every declared trim prefix")
            inward = src_hop == dst_hop + 1
            connected[source][edges[0, inward]] = True
        if any(not bool(value.all()) for value in connected.values()):
            raise ValueError("Every sampled non-root needs an inward path to a root")

    @staticmethod
    def _indices(indices: Tensor, size: int) -> None:
        if indices.dtype != torch.long or (
            indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= size)
        ):
            raise ValueError("Graph indices must be int64 and within local node bounds")

    @property
    def batch_size(self) -> int:
        return len(self.anchors)


@dataclass(frozen=True)
class FutureSubgraphTargets(ValidatedBatch):
    """Future labels [B,K], never encoder inputs. available_us includes source delay.

    A target partition is closed only when every whole horizon and its label
    availability lie within partition_end_us. False mask entries may hold NaNs;
    true entries must be finite. No target tensor may require gradients.
    """

    execution_fields: ClassVar[frozenset[str]] = frozenset(("values", "valid"))

    anchors: tuple[GraphAnchor, ...]
    values: Tensor
    valid: Tensor
    columns: tuple[str, ...]
    horizon_us: int
    label_delay_us: int
    available_us: Tensor
    partition_end_us: int

    _valid_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        require_cpu(self)
        if self.horizon_us <= 0 or self.label_delay_us < 0:
            raise ValueError("Future horizon must be positive and label delay nonnegative")
        if not self.anchors or len({a.sample_id for a in self.anchors}) != len(self.anchors):
            raise ValueError("Targets need unique, nonempty anchor identities")
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Target columns must be nonempty and unique")
        if (
            self.values.shape != (len(self.anchors), len(self.columns))
            or self.valid.shape != self.values.shape
            or self.valid.dtype != torch.bool
        ):
            raise ValueError("Target values/masks must match anchors and columns")
        if (
            not self.values.is_floating_point()
            or self.values.requires_grad
            or not bool(torch.isfinite(self.values[self.valid]).all())
        ):
            raise ValueError("Valid targets must be finite floats without gradients")
        if self.available_us.shape != (len(self.anchors),) or self.available_us.dtype != torch.long:
            raise ValueError("Target availability must be int64 [B]")
        earliest = torch.tensor(
            [utc_us(a.cutoff) + self.horizon_us + self.label_delay_us for a in self.anchors],
            dtype=torch.long,
        )
        if bool(
            ((self.available_us < earliest) | (self.available_us > self.partition_end_us)).any()
        ):
            raise ValueError("Incomplete future horizon or unavailable target in partition")
        object.__setattr__(self, "_valid_count", int(self.valid.sum()))
        self._seal()

    @property
    def valid_count(self) -> int:
        """CPU-validated target count, preserved by value-only device transfer."""
        self.assert_validated()
        return self._valid_count


@dataclass(frozen=True)
class GraphTrainingBatch(ValidatedBatch):
    """Aligned graph inputs and detached self-supervised future labels."""

    execution_fields: ClassVar[frozenset[str]] = frozenset(("graph", "targets"))

    graph: GraphBatch
    targets: FutureSubgraphTargets

    def __post_init__(self) -> None:
        require_cpu(self)
        self.graph.assert_validated()
        self.targets.assert_validated()
        if self.graph.anchors != self.targets.anchors:
            raise ValueError("Graph and target sample identities/cutoffs differ")
        self._seal()

    @property
    def has_supervision(self) -> bool:
        return self.targets.valid_count > 0


@dataclass(frozen=True)
class GraphSupervisedBatch(ValidatedBatch):
    """Graph inputs and available binary task labels [B], with explicit target level.

    This is direct task supervision, not an event label or future-target vector.
    Labels and their availability remain outside the encoder inputs.
    """

    execution_fields: ClassVar[frozenset[str]] = frozenset(("graph", "labels"))

    graph: GraphBatch
    labels: Tensor
    label_available_us: Tensor
    label_cutoff_us: int
    target_level: Literal["sequence", "entity"] = "sequence"

    def __post_init__(self) -> None:
        require_cpu(self)
        if self.target_level not in {"sequence", "entity"}:
            raise ValueError("Binary target_level must explicitly name sequence or entity")
        self.graph.assert_validated()
        if (
            self.labels.shape != (self.graph.batch_size,)
            or self.labels.requires_grad
            or not bool(((self.labels == 0) | (self.labels == 1)).all())
        ):
            raise ValueError("Expected detached binary task labels [B]")
        if (
            self.label_available_us.shape != self.labels.shape
            or self.label_available_us.dtype != torch.long
            or bool((self.label_available_us > self.label_cutoff_us).any())
        ):
            raise ValueError("Task labels are not available at the declared fitting cutoff")
        self._seal()


@dataclass(frozen=True)
class GraphHistoryBatch(ValidatedBatch):
    """Cutoff-aligned graph/history inputs and optional binary task labels.

    history_anchors certify the cutoffs used by the history builder. Labels
    require their own availability clocks, independently of feature cutoffs.
    """

    execution_fields: ClassVar[frozenset[str]] = frozenset(("graph", "history", "labels"))

    graph: GraphBatch
    history: KeyValueTimeBatch
    history_anchors: tuple[GraphAnchor, ...]
    labels: Tensor | None = None
    label_available_us: Tensor | None = None
    label_cutoff_us: int | None = None
    target_level: Literal["sequence", "entity"] = "sequence"

    _history_projection: SequenceSupervisedBatch | None = field(
        default=None, init=False, repr=False, compare=False, metadata={"validation_cache": True}
    )

    def __post_init__(self) -> None:
        require_cpu(self)
        if self.target_level not in {"sequence", "entity"}:
            raise ValueError("Binary target_level must explicitly name sequence or entity")
        self.graph.assert_validated()
        if self.graph.anchors != self.history_anchors:
            raise ValueError("Graph and history anchors/cutoffs differ")
        if self.history.sequence_ids != tuple(
            a.sample_id for a in self.history_anchors
        ) or self.history.entity_ids != tuple(a.entity_id for a in self.history_anchors):
            raise ValueError("History sample order differs from graph order")
        if self.history.batch_size != self.graph.batch_size:
            raise ValueError("Graph/history batch sizes differ")
        if self.labels is not None:
            if (
                self.labels.shape != (self.graph.batch_size,)
                or not bool(((self.labels == 0) | (self.labels == 1)).all())
                or self.labels.requires_grad
            ):
                raise ValueError("Expected detached binary task labels [B]")
            if (
                self.label_available_us is None
                or self.label_cutoff_us is None
                or self.label_available_us.shape != self.labels.shape
                or self.label_available_us.dtype != torch.long
                or bool((self.label_available_us > self.label_cutoff_us).any())
            ):
                raise ValueError("Task labels are not available at the declared fitting cutoff")
        elif self.label_available_us is not None or self.label_cutoff_us is not None:
            raise ValueError("Label availability provided without labels")
        self._seal()


@dataclass(frozen=True)
class SequencePredictionBatch(ValidatedBatch):
    """Unlabeled history inputs with exact anchor identities and mutation guards."""

    execution_fields: ClassVar[frozenset[str]] = frozenset(("history",))

    history: KeyValueTimeBatch
    anchors: tuple[GraphAnchor, ...]
    target_level: Literal["sequence", "entity"] = "sequence"

    def __post_init__(self) -> None:
        require_cpu(self)
        if self.target_level not in {"sequence", "entity"}:
            raise ValueError("Binary target_level must explicitly name sequence or entity")
        if not self.anchors or len({a.sample_id for a in self.anchors}) != len(self.anchors):
            raise ValueError("History predictions need unique, nonempty anchors")
        if (
            self.history.sequence_ids != tuple(a.sample_id for a in self.anchors)
            or self.history.entity_ids != tuple(a.entity_id for a in self.anchors)
            or self.history.batch_size != len(self.anchors)
        ):
            raise ValueError("Prediction history sample order differs from anchors")
        self._seal()


@dataclass(frozen=True)
class SequenceSupervisedBatch(ValidatedBatch):
    """History-only binary control, with the exact graph recipe sample/label clocks.

    Build with from_graph_history before device transfer; no graph tensors are
    retained or transferred. Normal CPU validation and mutation rules apply.
    """

    execution_fields: ClassVar[frozenset[str]] = frozenset(("history", "labels"))

    history: KeyValueTimeBatch
    anchors: tuple[GraphAnchor, ...]
    labels: Tensor
    label_available_us: Tensor
    label_cutoff_us: int
    target_level: Literal["sequence", "entity"] = "sequence"

    def __post_init__(self) -> None:
        require_cpu(self)
        if self.target_level not in {"sequence", "entity"}:
            raise ValueError("Binary target_level must explicitly name sequence or entity")
        if not self.anchors or len({a.sample_id for a in self.anchors}) != len(self.anchors):
            raise ValueError("Sequence controls need unique, nonempty anchors")
        if (
            self.history.sequence_ids != tuple(a.sample_id for a in self.anchors)
            or self.history.entity_ids != tuple(a.entity_id for a in self.anchors)
            or self.history.batch_size != len(self.anchors)
        ):
            raise ValueError("Sequence control history sample order differs from anchors")
        if (
            self.labels.shape != (len(self.anchors),)
            or self.labels.requires_grad
            or not bool(((self.labels == 0) | (self.labels == 1)).all())
        ):
            raise ValueError("Expected detached binary task labels [B]")
        if (
            self.label_available_us.shape != self.labels.shape
            or self.label_available_us.dtype != torch.long
            or bool((self.label_available_us > self.label_cutoff_us).any())
        ):
            raise ValueError("Task labels are not available at the declared fitting cutoff")
        self._seal()

    @classmethod
    def from_graph_history(cls, batch: GraphHistoryBatch) -> SequenceSupervisedBatch:
        """Keep already aligned histories/labels, excluding all graph tensors."""
        batch.assert_validated()
        if (
            batch.labels is None
            or batch.label_available_us is None
            or batch.label_cutoff_us is None
        ):
            raise ValueError("Sequence control requires available binary task labels")
        cached = batch._history_projection
        if cached is not None:
            cached.assert_validated()
            return cached
        result = cls(
            batch.history,
            batch.history_anchors,
            batch.labels,
            batch.label_available_us,
            batch.label_cutoff_us,
            target_level=batch.target_level,
        )
        object.__setattr__(batch, "_history_projection", result)
        return result
