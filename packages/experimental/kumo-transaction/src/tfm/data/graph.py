# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic relational oracle and graph sampler (experimental).

This in-memory implementation prioritizes inspectable temporal semantics. Use a
native adapter for large stores; architecture modules never transform tables.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

import torch

from tfm.contracts.graph import (
    STATIC_TIME,
    EdgeType,
    FutureSubgraphTargets,
    GraphAnchor,
    GraphBatch,
    utc_us,
)

NodeKey = tuple[str, str]
STATISTICS = ("finite_count", "sum", "mean", "min", "max", "std")


@dataclass(frozen=True)
class GraphTable:
    """Explicit numeric feature/target allowlists; IDs and clocks are metadata.

    Untimed tables must be explicitly immutable and cannot be forecast terminals.
    Feature values must already be fitted on historical training data, or be
    data-independent numeric representations. Targets contain raw measurements.
    """

    name: str
    features: tuple[str, ...]
    targets: tuple[str, ...] = ()
    immutable: bool = False

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.features
            or len(set(self.features)) != len(self.features)
            or len(set(self.targets)) != len(self.targets)
        ):
            raise ValueError("Graph table requires unique, nonempty feature names")
        if self.immutable and self.targets:
            raise ValueError("Immutable untimed tables cannot be future terminals")


@dataclass(frozen=True)
class GraphRow:
    """One synthetic/de-identified keyed row. Times separate event from availability."""

    table: str
    row_id: str
    values: Mapping[str, float]
    event_time: datetime | None = None
    available_at: datetime | None = None


@dataclass(frozen=True)
class GraphLink:
    """A declared FK role and keyed endpoints; availability must be explicit."""

    relation: EdgeType
    source_id: str
    target_id: str
    available_at: datetime | None = None


@dataclass(frozen=True)
class RelationalGraph:
    """Schema plus rows/links; directions are declared and reversed mechanically.

    Forbidden fields are mandatory schema metadata, not a name-guessing filter.
    Each table allowlist is checked against them. Do not include task outcomes.
    """

    tables: tuple[GraphTable, ...]
    relations: tuple[EdgeType, ...]
    rows: tuple[GraphRow, ...]
    links: tuple[GraphLink, ...]
    forbidden_fields: frozenset[str]
    observed_through: datetime | None = None

    def __post_init__(self) -> None:
        if self.observed_through is not None:
            utc_us(self.observed_through)
        tables = {t.name: t for t in self.tables}
        if len(tables) != len(self.tables) or not tables:
            raise ValueError("Graph table names must be unique")
        all_roles = self.directed_relations
        if len(set(all_roles)) != len(all_roles):
            raise ValueError("Relation roles, including reverse roles, must be unique")
        for table in self.tables:
            if self.forbidden_fields & (set(table.features) | set(table.targets)):
                raise ValueError(
                    "Forbidden label/key/clock field in graph feature/target allowlist"
                )
        for source, role, target in self.relations:
            if source not in tables or target not in tables or not role:
                raise ValueError("Unknown table or empty relation role")
        rows: dict[NodeKey, GraphRow] = {}
        for row in self.rows:
            if row.table not in tables or not row.row_id or (row.table, row.row_id) in rows:
                raise ValueError("Unknown table, missing key, or duplicate graph primary key")
            table = tables[row.table]
            if table.immutable:
                if row.event_time is not None or row.available_at is not None:
                    raise ValueError("Immutable rows must be untimed")
            elif (
                row.event_time is None
                or row.available_at is None
                or utc_us(row.available_at) < utc_us(row.event_time)
            ):
                raise ValueError("Temporal rows need event and availability clocks")
            if any(k not in row.values or not math.isfinite(row.values[k]) for k in table.features):
                raise ValueError(
                    "Input features must be finite, explicitly supplied numeric values"
                )
            rows[row.table, row.row_id] = row
        seen = set()
        for link in self.links:
            source, _, target = link.relation
            key = (link.relation, link.source_id, link.target_id)
            if (
                link.relation not in self.relations
                or (source, link.source_id) not in rows
                or (target, link.target_id) not in rows
                or key in seen
            ):
                raise ValueError("Unknown/dangling/duplicate graph link")
            if link.available_at is None:
                if not (tables[source].immutable and tables[target].immutable):
                    raise ValueError("Temporal links require an availability clock")
            else:
                utc_us(link.available_at)
            seen.add(key)

    @property
    def directed_relations(self) -> tuple[EdgeType, ...]:
        return tuple(
            r for s, name, t in self.relations for r in ((s, name, t), (t, f"rev_{name}", s))
        )


class GraphSampler:
    """Recent per-role sampling with stable PK tie breaks and minimum-distance shells."""

    def __init__(self, graph: RelationalGraph, *, depth: int, fanout: int) -> None:
        if depth < 1 or fanout < 1:
            raise ValueError("Graph depth and fanout must be positive")
        self.graph, self.depth, self.fanout = graph, depth, fanout
        self.tables = {t.name: t for t in graph.tables}
        self.rows = {(r.table, r.row_id): r for r in graph.rows}
        self.adjacency: dict[NodeKey, list[tuple[EdgeType, NodeKey, int]]] = {}
        for link in graph.links:
            s, role, t = link.relation
            time = STATIC_TIME if link.available_at is None else utc_us(link.available_at)
            self.adjacency.setdefault((s, link.source_id), []).append(
                (link.relation, (t, link.target_id), time)
            )
            self.adjacency.setdefault((t, link.target_id), []).append(
                ((t, f"rev_{role}", s), (s, link.source_id), time)
            )

    @property
    def loader_manifest(self) -> dict[str, str]:
        """Identify this loader independently of its callers."""
        return {"name": "python"}

    @property
    def tie_policy(self) -> str:
        """Describe equal-time neighbor ordering for run provenance."""
        return "canonical PK ascending"

    def row_time(self, key: NodeKey) -> int:
        at = self.rows[key].available_at
        return STATIC_TIME if at is None else utc_us(at)

    def event_time_us(self, key: NodeKey) -> int:
        at = self.rows[key].event_time
        return STATIC_TIME if at is None else utc_us(at)

    def neighbors(
        self,
        key: NodeKey,
        relation: EdgeType,
        cutoff: int,
        start: int = STATIC_TIME,
        *,
        capped: bool = True,
    ) -> list[tuple[NodeKey, int]]:
        found = [
            (other, time)
            for role, other, time in self.adjacency.get(key, ())
            if role == relation
            and time <= cutoff
            and self.row_time(other) <= cutoff
            and (self.event_time_us(other) == STATIC_TIME or self.event_time_us(other) >= start)
        ]
        found.sort(key=lambda item: (-self.event_time_us(item[0]), item[0][1]))
        return found[: self.fanout] if capped else found

    def membership(
        self, root: NodeKey, cutoff: int, depth: int, start: int = STATIC_TIME
    ) -> tuple[dict[NodeKey, int], dict[tuple[EdgeType, NodeKey, NodeKey], int]]:
        if root not in self.rows or self.row_time(root) > cutoff:
            raise ValueError("Root is missing or not available at anchor cutoff")
        if self.event_time_us(root) != STATIC_TIME and self.event_time_us(root) < start:
            raise ValueError("Root predates observation window; declare immutable identity roots")
        distances = {root: 0}
        edges: dict[tuple[EdgeType, NodeKey, NodeKey], int] = {}
        queue = deque([root])
        while queue:
            parent = queue.popleft()
            if distances[parent] >= depth:
                continue
            for role in self.graph.directed_relations:
                if role[0] != parent[0]:
                    continue
                for child, time in self.neighbors(parent, role, cutoff, start):
                    edges[role, parent, child] = time
                    if child not in distances:
                        distances[child] = distances[parent] + 1
                        queue.append(child)
        return distances, edges

    @torch.inference_mode(False)
    @torch.no_grad()
    def sample(
        self,
        anchors: tuple[GraphAnchor, ...],
        root_type: str,
        *,
        sharing: Literal["disjoint", "shared"] = "disjoint",
        hop_trim: bool = True,
    ) -> GraphBatch:
        """Sample per root, then optionally union canonical nodes/edges at one cutoff.

        Sharing changes node multiplicities and BatchNorm statistics; it is an
        explicit training policy, not a promise of identical model outputs.
        Different cutoffs/windows cannot share context. Roots/labels remain in
        anchor order and repeated roots remain repeated prediction examples.
        """
        if sharing not in {"shared", "disjoint"} or not anchors:
            raise ValueError("Choose shared/disjoint sampling with nonempty anchors")
        if (
            sharing == "shared"
            and len(
                {
                    (
                        utc_us(a.cutoff),
                        None if a.observation_start is None else utc_us(a.observation_start),
                    )
                    for a in anchors
                }
            )
            != 1
        ):
            raise ValueError("Shared graphs require identical cutoffs and observation windows")
        if root_type not in self.tables:
            raise ValueError("Unknown root type")
        features: dict[str, list[list[float]]] = {t: [] for t in self.tables}
        owners: dict[str, list[int]] = {t: [] for t in self.tables}
        times: dict[str, list[int]] = {t: [] for t in self.tables}
        edges: dict[EdgeType, list[tuple[int, int]]] = {
            r: [] for r in self.graph.directed_relations
        }
        edge_times: dict[EdgeType, list[int]] = {r: [] for r in edges}
        roots = []
        local: dict[tuple[int, NodeKey], int] = {}
        emitted: set[tuple[int, EdgeType, NodeKey, NodeKey]] = set()
        for i, anchor in enumerate(anchors):
            root = (root_type, anchor.entity_id)
            nodes, links = self.membership(
                root,
                utc_us(anchor.cutoff),
                self.depth,
                STATIC_TIME
                if anchor.observation_start is None
                else utc_us(anchor.observation_start),
            )
            component = 0 if sharing == "shared" else i
            for key in sorted(nodes):
                name, _ = key
                if (component, key) in local:
                    continue
                local[component, key] = len(features[name])
                features[name].append(
                    [self.rows[key].values[f] for f in self.tables[name].features]
                )
                owners[name].append(component)
                times[name].append(self.row_time(key))
            roots.append(local[component, root])
            # Every sampled physical link supplies both declared message
            # directions, including the outermost shell's messages inward.
            opposite = {
                r: reverse
                for s, name, t in self.graph.relations
                for r, reverse in (
                    ((s, name, t), (t, f"rev_{name}", s)),
                    ((t, f"rev_{name}", s), (s, name, t)),
                )
            }
            links.update(
                {
                    (opposite[role], child, parent): time
                    for (role, parent, child), time in list(links.items())
                }
            )
            for (role, parent, child), time in sorted(links.items()):
                identity = (component, role, parent, child)
                if identity not in emitted:
                    emitted.add(identity)
                    edges[role].append((local[component, parent], local[component, child]))
                    edge_times[role].append(time)
        batch = GraphBatch(
            anchors,
            root_type,
            {
                t: torch.tensor(x, dtype=torch.float32).reshape(-1, len(self.tables[t].features))
                for t, x in features.items()
            },
            {t: torch.tensor(x, dtype=torch.long) for t, x in owners.items()},
            {t: torch.tensor(x, dtype=torch.long) for t, x in times.items()},
            {
                r: torch.tensor(x, dtype=torch.long).reshape(-1, 2).t().contiguous()
                for r, x in edges.items()
            },
            {r: torch.tensor(x, dtype=torch.long) for r, x in edge_times.items()},
            torch.tensor(roots, dtype=torch.long),
            sharing=sharing,
        )
        return order_graph_by_hop(batch, self.depth) if hop_trim else batch


def _statistics(rows: list[GraphRow], column: str) -> list[tuple[float, bool]]:
    finite = [
        r.values.get(column, math.nan)
        for r in rows
        if math.isfinite(r.values.get(column, math.nan))
    ]
    n = len(finite)
    total = math.fsum(finite)
    mean = total / n if n else 0.0
    return [
        (float(n), True),
        (total, not rows or n > 0),
        (mean, n > 0),
        (min(finite) if n else 0.0, n > 0),
        (max(finite) if n else 0.0, n > 0),
        (math.sqrt(math.fsum((x - mean) ** 2 for x in finite) / (n - 1)) if n > 1 else 0.0, n > 1),
    ]


@torch.inference_mode(False)
@torch.no_grad()
def future_subgraph_targets(
    sampler: GraphSampler,
    anchors: tuple[GraphAnchor, ...],
    root_type: str,
    *,
    horizon_us: int,
    depth: int,
    partition_end_us: int,
    label_delay_us: int = 0,
) -> FutureSubgraphTargets:
    """Window-end shells, uncapped future terminal rows, per-role PK deduplication.

    Input graph sampling is never called with a future cutoff. This separate
    label builder uses membership at t+H and includes only terminal event times
    in (t,t+H]. Rows/links arriving late extend the label's availability.
    """
    if depth < 1 or horizon_us <= 0 or label_delay_us < 0:
        raise ValueError("Target depth/horizon must be positive; delay nonnegative")
    if sampler.graph.observed_through is None:
        raise ValueError("Future labels require an explicit observed-through boundary")
    if any(utc_us(a.cutoff) + horizon_us > utc_us(sampler.graph.observed_through) for a in anchors):
        raise ValueError("Incomplete future window beyond source observation coverage")
    groups = [
        (h, r)
        for h in range(1, depth + 1)
        for r in sampler.graph.directed_relations
        if not sampler.tables[r[2]].immutable
    ]
    columns = tuple(
        f"{h}/{s}/{role}/{t}/{field}"
        for h, (s, role, t) in groups
        for field in (
            "count",
            *(f"{c}/{stat}" for c in sampler.tables[t].targets for stat in STATISTICS),
        )
    )
    values, masks, availability = [], [], []
    for anchor in anchors:
        cutoff, end = utc_us(anchor.cutoff), utc_us(anchor.cutoff) + horizon_us
        distances, _ = sampler.membership((root_type, anchor.entity_id), end, depth - 1)
        out: list[float] = []
        valid: list[bool] = []
        available = end + label_delay_us
        for h, role in groups:
            terminals: dict[NodeKey, GraphRow] = {}
            for parent, distance in distances.items():
                if distance != h - 1 or parent[0] != role[0]:
                    continue
                # Do not discard late-arriving terminal events: reflect their
                # availability in the label instead of calling a partial count complete.
                for relation, child, link_time in sampler.adjacency.get(parent, ()):
                    row = sampler.rows[child]
                    if (
                        relation == role
                        and link_time <= end
                        and row.event_time is not None
                        and cutoff < utc_us(row.event_time) <= end
                    ):
                        terminals[child] = row
                        available = max(
                            available,
                            sampler.row_time(child) + label_delay_us,
                            link_time + label_delay_us,
                        )
            terminal_rows = [terminals[k] for k in sorted(terminals)]
            result = [(float(len(terminal_rows)), True)]
            for column in sampler.tables[role[2]].targets:
                result.extend(_statistics(terminal_rows, column))
            out.extend(x for x, _ in result)
            valid.extend(m for _, m in result)
        if available > utc_us(sampler.graph.observed_through):
            raise ValueError("Target availability must fit inside declared source coverage")
        values.append(out)
        masks.append(valid)
        availability.append(available)
    return FutureSubgraphTargets(
        anchors,
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.bool),
        columns,
        horizon_us,
        label_delay_us,
        torch.tensor(availability, dtype=torch.long),
        partition_end_us,
    )


@dataclass(frozen=True)
class FutureTargetTransform:
    """Train-only signed-log target selection/normalization, persisted in checkpoints."""

    input_columns: tuple[str, ...]
    selected: tuple[int, ...]
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def fit(cls, targets: FutureSubgraphTargets) -> FutureTargetTransform:
        targets.assert_validated()
        raw = torch.where(targets.valid, targets.values.double(), 0.0)
        x = raw.sign() * raw.abs().log1p()
        n = targets.valid.sum(0)
        mean = x.sum(0) / n.clamp_min(1)
        variance = torch.where(targets.valid, (x - mean).square(), 0.0).sum(0) / n.clamp_min(1)
        selected = ((n >= 2) & (variance.sqrt() > 1e-6)).nonzero().flatten()
        if not selected.numel():
            raise ValueError("FutureSubgraph has no nonconstant training targets")
        return cls(
            targets.columns,
            tuple(selected.tolist()),
            mean[selected].float(),
            variance[selected].sqrt().float(),
        )

    @torch.inference_mode(False)
    @torch.no_grad()
    def apply(self, targets: FutureSubgraphTargets) -> FutureSubgraphTargets:
        targets.assert_validated()
        if targets.columns != self.input_columns:
            raise ValueError("Future target schema/order differs from fitted transform")
        index = list(self.selected)
        mask = targets.valid[:, index]
        raw = torch.where(mask, targets.values[:, index], 0.0)
        x = raw.sign() * raw.abs().log1p()
        x = (x - self.mean.to(x.device)) / self.scale.to(x.device)
        return replace(
            targets,
            values=torch.where(mask, x, 0.0),
            valid=mask,
            columns=tuple(targets.columns[i] for i in index),
        )


@torch.inference_mode(False)
@torch.no_grad()
def order_graph_by_hop(batch: GraphBatch, depth: int) -> GraphBatch:
    """Build Engine's node/receiver-edge prefixes from a retrieved CPU graph.

    Distances are minimum distances from this batch's root set. Keep incoming
    edges only for receiver shells 0..depth-1: shell depth is removed from the
    first layer's destination features. Reindex every tensor and preserve root
    sample order, including repeated roots in a shared graph.
    """
    batch.assert_validated()
    if depth < 1 or any(x.device.type != "cpu" for x in batch.node_features.values()):
        raise ValueError("Build hop ordering on CPU with positive depth before device transfer")
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for (source, _, target), edges in batch.edge_index.items():
        for send, receive in edges.t().tolist():
            a, b = (source, send), (target, receive)
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
    distances = {(batch.root_type, i): 0 for i in batch.root_index.tolist()}
    queue = deque(distances)
    while queue:
        parent = queue.popleft()
        for child in sorted(adjacency.get(parent, ())):
            if child not in distances:
                distances[child] = distances[parent] + 1
                queue.append(child)
    if (
        len(distances) != sum(len(x) for x in batch.node_features.values())
        or max(distances.values(), default=0) > depth
    ):
        raise ValueError("Retrieved graph has unreachable nodes or exceeds requested hop depth")
    order, inverse, hops, node_counts = {}, {}, {}, {}
    for name, features in batch.node_features.items():
        indices = sorted(range(len(features)), key=lambda i: (distances[name, i], i))
        order[name] = torch.tensor(indices, dtype=torch.long)
        inverse[name] = torch.empty(len(features), dtype=torch.long)
        inverse[name][order[name]] = torch.arange(len(features))
        hops[name] = torch.tensor([distances[name, i] for i in indices], dtype=torch.long)
        node_counts[name] = tuple(int((hops[name] == h).sum()) for h in range(depth + 1))
    edges, clocks, edge_counts = {}, {}, {}
    for role, original in batch.edge_index.items():
        source, _, target = role
        indices = [
            i
            for i, receive in enumerate(original[1].tolist())
            if distances[target, receive] < depth
        ]
        indices.sort(key=lambda i: (distances[target, int(original[1, i])], i))
        selected = torch.tensor(indices, dtype=torch.long)
        edges[role] = torch.stack(
            (inverse[source][original[0, selected]], inverse[target][original[1, selected]])
        )
        clocks[role] = batch.edge_available_us[role][selected]
        receiver_hop = hops[target][edges[role][1]]
        edge_counts[role] = tuple(int((receiver_hop == h).sum()) for h in range(depth))
    return replace(
        batch,
        node_features={name: x[order[name]] for name, x in batch.node_features.items()},
        node_sample={name: x[order[name]] for name, x in batch.node_sample.items()},
        node_available_us={name: x[order[name]] for name, x in batch.node_available_us.items()},
        edge_index=edges,
        edge_available_us=clocks,
        root_index=inverse[batch.root_type][batch.root_index],
        node_hop=hops,
        num_sampled_nodes=node_counts,
        num_sampled_edges=edge_counts,
    )
