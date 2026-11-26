# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded native candidate retrieval for arbitrary fixed-FK relational graphs.

Cutoff snapshots filter row and link availability before retrieval. One-hop
plans avoid the backend's repeated-relation layer restriction. All candidates
are retrieved for deterministic boundary ties, then recent fanout is applied.
This reference integration makes no out-of-core or performance claim.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tfm.contracts.graph import STATIC_TIME, EdgeType, GraphAnchor, GraphBatch, utc_us
from tfm.data.adapters.kumo_graph import from_kumo_graph
from tfm.data.graph import GraphSampler, NodeKey, RelationalGraph
from tfm.extras.kumo_pytorch import SOURCE_REVISION, native_api, temporary_engine

Membership = tuple[dict[NodeKey, int], dict[tuple[EdgeType, NodeKey, NodeKey], int]]


@dataclass
class _BoundGraph:
    inputs: Mapping[str, Any]
    metadata: Mapping[str, Any]


class _MembershipPacker(GraphSampler):
    """Use the ordinary validated packer without altering the future-target oracle."""

    def __init__(self, sampler: GraphSampler, memberships: Mapping[tuple[Any, ...], Membership]):
        self.graph, self.depth, self.fanout = sampler.graph, sampler.depth, sampler.fanout
        self.tables, self.rows, self.adjacency = sampler.tables, sampler.rows, sampler.adjacency
        self.memberships = memberships

    def membership(
        self, root: NodeKey, cutoff: int, depth: int, start: int = STATIC_TIME
    ) -> Membership:
        nodes, edges = self.memberships[root, cutoff, depth, start]
        return dict(nodes), dict(edges)


class _Snapshot:
    def __init__(self, sampler: NativeRelationalGraphSampler, cutoff: int, start: int):
        self.sampler, self.cutoff, self.start = sampler, cutoff, start
        self.stack = ExitStack()
        self.plans: dict[EdgeType, Any] = {}
        self.index: str | None = None
        self.calls = 0
        try:
            self.root = Path(self.stack.enter_context(TemporaryDirectory(prefix="tfm-relational-")))
            self.api = native_api()
            self.keys = {
                name: tuple(
                    sorted(
                        key
                        for table, key in sampler.rows
                        if table == name
                        and sampler.row_time((table, key)) <= cutoff
                        and (
                            not sampler.tables[table].windowed
                            or sampler.event_time_us((table, key)) == STATIC_TIME
                            or sampler.event_time_us((table, key)) >= start
                        )
                    )
                )
                for name in sampler.tables
            }
            self.codes = {
                name: {key: i for i, key in enumerate(keys)} for name, keys in self.keys.items()
            }
            self.clocks = {}
            parents: dict[EdgeType, dict[int, int]] = {}
            for relation in sampler.graph.relations:
                source, _, target = relation
                mapping = {}
                for link in sampler.graph.links:
                    if (
                        link.relation != relation
                        or link.source_id not in self.codes[source]
                        or link.target_id not in self.codes[target]
                    ):
                        continue
                    known = STATIC_TIME if link.available_at is None else utc_us(link.available_at)
                    if known > cutoff:
                        continue
                    parent, child = (
                        self.codes[source][link.source_id],
                        self.codes[target][link.target_id],
                    )
                    if child in mapping:
                        raise ValueError(
                            f"Native graph path {relation[1]}: multiple parents for fixed FK child"
                        )
                    mapping[child] = parent
                    self.clocks[relation, parent, child] = known
                parents[relation] = mapping
            self.relations = tuple(relation for relation, mapping in parents.items() if mapping)
            active = {name for name, keys in self.keys.items() if keys}
            tables, roles = [], []
            for no, (source, role, target) in enumerate(self.relations):
                roles.append(self.api.RelationSpec(role, source, "id", target, f"fk{no}"))
            for name in sorted(active):
                columns = {
                    "id": pa.array(range(len(self.keys[name])), type=pa.int64()),
                    "bias": pa.array([1.0] * len(self.keys[name]), type=pa.float32()),
                }
                types = {"id": self.api.DType("int64"), "bias": self.api.DType("float32")}
                for no, relation in enumerate(self.relations):
                    if relation[2] == name:
                        columns[f"fk{no}"] = pa.array(
                            [parents[relation].get(i) for i in range(len(self.keys[name]))],
                            type=pa.int64(),
                        )
                        types[f"fk{no}"] = self.api.DType("int64")
                pq.write_table(pa.table(columns), self.root / f"{name}.parquet")
                tables.append(
                    self.api.TableSpec(name, "id", types, time_column=self.api.NON_TEMPORAL)
                )
            self.schema = self.api.DataSchema(tuple(tables), tuple(roles))
            self.dataset = self.api.Dataset(
                {name: self.api.Parquet(str(self.root / f"{name}.parquet")) for name in active}
            )
        except BaseException:
            self.stack.close()
            raise

    def close(self) -> None:
        self.stack.close()

    def _plan(self, directed: EdgeType) -> Any:
        if directed in self.plans:
            return self.plans[directed]
        relation = next(
            r for r in self.relations if r == directed or (r[2], f"rev_{r[1]}", r[0]) == directed
        )
        forward = relation == directed
        # Full candidate retrieval preserves exact canonical ties. This physical
        # fanout is intentionally recorded separately from the logical fanout.
        bound = max(1, len(self.keys[directed[2]]))
        contract = self.api.DataLoaderSpec(
            self.schema,
            self.api.SampleSpec(
                self.api.EntitySpec(directed[0]),
                neighborhoods={
                    "one": self.api.NeighborhoodSpec(
                        (
                            self.api.GraphLayer(
                                (
                                    self.api.GraphSample(
                                        relation[1],
                                        bound,
                                        direction="forward" if forward else "reverse",
                                        strategy="uniform",
                                    ),
                                )
                            ),
                        )
                    )
                },
            ),
            self.api.BatchSpec(
                {
                    "graph": self.api.Input(
                        self.api.NeighborhoodSource("one"),
                        self.api.Graph(
                            {
                                name: self.api.Fields((self.api.Encode("bias", self.api.Raw()),))
                                for name, keys in self.keys.items()
                                if keys and name in {directed[0], directed[2]}
                            },
                            topology="relations",
                        ),
                        None,
                    )
                }
            ),
        )
        if self.index is None:
            engine = temporary_engine(self.stack)
            plan = self.api.prepare(contract, self.dataset, engine)
            self.index = str(engine.index)
        else:
            engine = self.api.DiskGraphEngine.from_snapshot(
                SimpleNamespace(index_path=self.index, id=None), num_threads=1
            )
            plan = self.api.prepare(contract, engine=engine)
        self.plans[directed] = self.stack.enter_context(plan)
        return plan

    @torch.inference_mode(False)
    @torch.no_grad()
    def neighbors(
        self, directed: EdgeType, roots: tuple[str, ...]
    ) -> dict[str, list[tuple[NodeKey, int]]]:
        result: dict[str, list[tuple[NodeKey, int]]] = {root: [] for root in roots}
        physical = next(
            (r for r in self.relations if r == directed or (r[2], f"rev_{r[1]}", r[0]) == directed),
            None,
        )
        if physical is None:
            return result
        api, plan = self.api, self._plan(directed)
        at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=self.cutoff)
        ids = tuple(self.codes[directed[0]][root] for root in roots)
        anchors = tuple(GraphAnchor(f"candidate-{i}", root, at) for i, root in enumerate(roots))
        path = self.root / f"anchors{self.calls}.parquet"
        self.calls += 1
        try:
            pq.write_table(
                pa.table(
                    {
                        "sample_id": [a.sample_id for a in anchors],
                        "owner": pa.array(ids, type=pa.int64()),
                        "at": pa.array([at] * len(roots), type=pa.timestamp("us", tz="UTC")),
                        "split": ["train"] * len(roots),
                    }
                ),
                path,
            )
            binding = api.AnchorTable(
                api.Parquet(str(path)),
                entity_id="owner",
                sample_id="sample_id",
                split="split",
            )
            with plan.materialize(binding, fit_split="train") as task:
                batches = iter(
                    task.batches(
                        split="train", batch_size=len(roots), shuffle=False, output_format="model"
                    )
                )
                native = next(batches)
                if next(batches, None) is not None:
                    raise ValueError("Native relational query returned extra batches")
                if tuple(int(i) for i in native.metadata["entity_ids"]) != ids:
                    raise ValueError("Native relational query reordered root bindings")
                payload = native.inputs["graph"]
                if set(payload["nodes"]) - {physical[0], physical[2]} or set(payload["edges"]) - {
                    physical[1]
                }:
                    raise ValueError("Native one-hop query returned undeclared tables/roles")
                nodes = dict(payload["nodes"])
                for name in (physical[0], physical[2]):
                    nodes.setdefault(
                        name,
                        {
                            "ids": torch.empty(0, dtype=torch.long),
                            "batch": torch.empty(0, dtype=torch.long),
                            "features": {
                                "bias": {
                                    "values": torch.empty(0),
                                    "valid": torch.empty(0, dtype=torch.bool),
                                }
                            },
                        },
                    )
                for name, node in nodes.items():
                    row_ids, owners = node["ids"], node["batch"]
                    if (
                        row_ids.dtype != torch.long
                        or row_ids.ndim != 1
                        or owners.dtype != torch.long
                        or owners.shape != row_ids.shape
                    ):
                        raise ValueError("Native one-hop query has invalid canonical node IDs")
                    if any(not 0 <= key < len(self.keys[name]) for key in row_ids.tolist()):
                        raise ValueError("Native one-hop query has unbound canonical node IDs")
                    if len(set(zip(owners.tolist(), row_ids.tolist(), strict=True))) != len(
                        row_ids
                    ):
                        raise ValueError("Native one-hop query duplicated a node within a sample")
                role = physical[1]
                edge = payload["edges"].get(
                    role,
                    {
                        "source_type": physical[0],
                        "destination_type": physical[2],
                        "index": torch.empty((2, 0), dtype=torch.long),
                    },
                )
                edge_index = edge["index"]
                if (
                    edge_index.dtype != torch.long
                    or edge_index.ndim != 2
                    or edge_index.shape[0] != 2
                ):
                    raise ValueError("Native one-hop query has invalid edge indices")
                for axis, name in enumerate((physical[0], physical[2])):
                    if edge_index.shape[1] and (
                        int(edge_index[axis].min()) < 0
                        or int(edge_index[axis].max()) >= len(nodes[name]["ids"])
                    ):
                        raise ValueError("Native one-hop query has out-of-range edge indices")
                bound = _BoundGraph(
                    inputs={"graph": {"nodes": nodes, "edges": {role: edge}}},
                    metadata=native.metadata,
                )
                pairs = edge_index.T.tolist()
                known = []
                for left, right in pairs:
                    parent = int(nodes[physical[0]]["ids"][left])
                    child = int(nodes[physical[2]]["ids"][right])
                    known.append(self.clocks[physical, parent, child])
                node_clocks = {
                    name: torch.tensor(
                        [
                            self.sampler.row_time((name, self.keys[name][i]))
                            for i in node["ids"].tolist()
                        ],
                        dtype=torch.long,
                    )
                    for name, node in nodes.items()
                }
                # The ordinary adapter checks temporal bounds, duplicate edges,
                # crossed components and exact sample order before IDs are used.
                checked = from_kumo_graph(
                    bound,
                    anchors=anchors,
                    root_type=directed[0],
                    root_node_ids=ids,
                    feature_columns={name: ("bias",) for name in nodes},
                    relations=(physical, (physical[2], f"rev_{role}", physical[0])),
                    forbidden_fields=frozenset(),
                    node_available_us=node_clocks,
                    edge_available_us={physical: torch.tensor(known, dtype=torch.long)},
                )
                checked.assert_validated()
                index = checked.edge_index[directed]
                for left, right in index.T.tolist():
                    parent = self.keys[directed[0]][int(nodes[directed[0]]["ids"][left])]
                    child = self.keys[directed[2]][int(nodes[directed[2]]["ids"][right])]
                    owner = int(checked.node_sample[directed[0]][left])
                    if parent != roots[owner]:
                        raise ValueError("Native one-hop edge is not rooted at its bound frontier")
                    pair = (left, right) if directed == physical else (right, left)
                    parent_id = int(nodes[physical[0]]["ids"][pair[0]])
                    child_id = int(nodes[physical[2]]["ids"][pair[1]])
                    result[parent].append(
                        ((directed[2], child), self.clocks[physical, parent_id, child_id])
                    )
                for items in result.values():
                    items.sort(key=lambda item: (-self.sampler.event_time_us(item[0]), item[0][1]))
                    del items[self.sampler.fanout :]
            return result
        finally:
            path.unlink(missing_ok=True)


class NativeRelationalGraphSampler(GraphSampler):
    """Native one-hop retrieval, arbitrary frontier types, canonical logical fanout.

    Input snapshots retain exact original clocks outside the native index. This
    bounded implementation retrieves all eligible candidates to resolve ties;
    it does not claim the physical work of native fanout-limited retrieval.
    FutureSubgraph continues using the inherited separate Python membership.
    """

    def __init__(self, graph: RelationalGraph, *, depth: int, fanout: int) -> None:
        super().__init__(graph, depth=depth, fanout=fanout)
        native_api()
        self.snapshots: OrderedDict[tuple[int, int], _Snapshot] = OrderedDict()
        self.closed = False

    def close(self) -> None:
        """Release cached plans and indexes; safe after a partial sampling failure."""
        self.closed = True
        snapshots = tuple(self.snapshots.values())
        self.snapshots.clear()
        with ExitStack() as cleanup:
            for snapshot in snapshots:
                cleanup.callback(snapshot.close)

    def _snapshot(self, cutoff: int, start: int) -> _Snapshot:
        key = cutoff, start
        if key not in self.snapshots:
            if len(self.snapshots) == 2:
                _, old = self.snapshots.popitem(last=False)
                old.close()
            self.snapshots[key] = _Snapshot(self, cutoff, start)
        self.snapshots.move_to_end(key)
        return self.snapshots[key]

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
        if self.closed:
            raise ValueError("Native relational sampler is closed")
        if not anchors or root_type not in self.tables:
            raise ValueError(
                "Native relational sampling needs nonempty anchors and a declared root"
            )
        memberships = {}
        for anchor in anchors:
            cutoff = utc_us(anchor.cutoff)
            start = (
                STATIC_TIME
                if anchor.observation_start is None
                else utc_us(anchor.observation_start)
            )
            root = root_type, anchor.entity_id
            if (
                root not in self.rows
                or self.row_time(root) > cutoff
                or (
                    self.tables[root[0]].windowed
                    and self.event_time_us(root) != STATIC_TIME
                    and self.event_time_us(root) < start
                )
            ):
                raise ValueError("Root is absent or outside the observable input window")
            members: Membership = ({root: 0}, {})
            memberships[root, cutoff, self.depth, start] = members
        # Coalesce repeated roots and same-cutoff frontier rows, retaining one
        # logical neighborhood per original anchor for disjoint/shared packing.
        groups = sorted(
            {
                (
                    utc_us(a.cutoff),
                    STATIC_TIME if a.observation_start is None else utc_us(a.observation_start),
                )
                for a in anchors
            }
        )
        for cutoff, start in groups:
            snapshot = self._snapshot(cutoff, start)
            active = [
                value for key, value in memberships.items() if key[1] == cutoff and key[3] == start
            ]
            for hop in range(self.depth):
                for role in self.graph.directed_relations:
                    roots = tuple(
                        sorted(
                            {
                                node[1]
                                for nodes, _ in active
                                for node, distance in nodes.items()
                                if distance == hop and node[0] == role[0]
                            }
                        )
                    )
                    if not roots:
                        continue
                    neighbors = snapshot.neighbors(role, roots)
                    for nodes, edges in active:
                        for parent in tuple(
                            node
                            for node, distance in nodes.items()
                            if distance == hop and node[0] == role[0]
                        ):
                            for child, clock in neighbors[parent[1]]:
                                edges[role, parent, child] = clock
                                nodes.setdefault(child, hop + 1)
        return _MembershipPacker(self, memberships).sample(
            anchors, root_type, sharing=sharing, hop_trim=hop_trim
        )

    @property
    def tie_policy(self) -> str:
        return "canonical PK ascending"

    @property
    def loader_manifest(self) -> dict[str, str]:
        return {
            "name": "kumo_pytorch",
            "revision": SOURCE_REVISION,
            "graph_sampling": "cutoff_snapshot_native_candidates_canonical_fanout",
            "physical_fanout": "all eligible candidates per frontier and role",
            "target_builder": "python_reference",
        }
