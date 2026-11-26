# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiskGraph input sampling for the transaction FK example; targets stay separate."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from weakref import finalize

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.graph import (
    STATIC_TIME,
    GraphAnchor,
    GraphBatch,
    _has_duplicate_edges,
    utc_us,
)
from tfm.data.adapters.kumo_graph import from_kumo_graph
from tfm.data.graph import GraphSampler, RelationalGraph, order_graph_by_hop
from tfm.extras.kumo_pytorch import SOURCE_REVISION, native_api, temporary_engine


def _clock(value: datetime | None) -> int:
    return STATIC_TIME if value is None else utc_us(value)


def _first_occurrences(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable first-row selection and inverse, with an integer-pair overflow fallback."""
    _unique, inverse = (
        torch.unique(values, return_inverse=True)
        if values.ndim == 1
        else torch.unique(values, dim=0, return_inverse=True)
    )
    first = torch.full((len(_unique),), len(values), dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(len(values)), reduce="amin")
    order = first.argsort()
    remap = torch.empty_like(order)
    remap[order] = torch.arange(len(order))
    return first[order], remap[inverse]


@dataclass
class _BoundGraph:
    inputs: Mapping[str, Any]
    metadata: Mapping[str, Any]


class NativeGraphSampler(GraphSampler):
    """Sample historical input graphs with DiskGraph; inherit the separate target oracle.

    The worked schema is a static users table with one-parent event/transfer FK
    roles, second-aligned clocks, and row availability equal to event time.
    Unsupported schemas/windows fail explicitly. Recent fanout ties follow the
    native index's policy; Python-oracle neighborhood parity is not promised.
    """

    def __init__(self, graph: RelationalGraph, *, depth: int, fanout: int) -> None:
        super().__init__(graph, depth=depth, fanout=fanout)
        api = native_api()
        if (
            "users" not in self.tables
            or not self.tables["users"].immutable
            or any(s != "users" for s, _, _ in graph.relations)
        ):
            raise UnsupportedCombinationError(
                "Native graph recipes require static users and user-to-event FK roles"
            )
        for row in graph.rows:
            if _clock(row.available_at) != _clock(row.event_time) or (
                row.available_at is not None and utc_us(row.available_at) % 1_000_000
            ):
                raise UnsupportedCombinationError(
                    "Native graph examples require second-aligned event/availability clocks; select data.loader: python for other clocks"
                )
        self.keys = {
            name: tuple(sorted(r.row_id for r in graph.rows if r.table == name))
            for name in self.tables
        }
        self.codes = {
            name: {key: i for i, key in enumerate(keys)} for name, keys in self.keys.items()
        }
        self.node_clocks = {
            name: torch.tensor([self.row_time((name, key)) for key in keys], dtype=torch.long)
            for name, keys in self.keys.items()
        }
        # DiskGraph rejects physically empty tables during ingest. Omit them
        # from its schema, then restore their declared empty tensors at adaptation.
        active = {name for name, keys in self.keys.items() if keys}
        active_relations = tuple(r for r in graph.relations if r[2] in active)
        if not active_relations:
            raise UnsupportedCombinationError(
                "Native graph examples require a nonempty referenced event table; select data.loader: python for root-only sources"
            )
        self.native_stack = ExitStack()
        try:
            self.native_root = Path(
                self.native_stack.enter_context(TemporaryDirectory(prefix="tfm-native-graph-"))
            )
            self.api = api
            schema_tables, schema_relations = [], []
            parents: dict[str, dict[int, int]] = {}
            self.edge_clocks = {}
            self.edge_parents = {}
            for no, relation in enumerate(graph.relations):
                source, role, target = relation
                if relation not in active_relations:
                    continue
                mapping = {}
                clocks = {}
                for link in graph.links:
                    if link.relation != relation:
                        continue
                    dest = self.codes[target][link.target_id]
                    if dest in mapping:
                        raise UnsupportedCombinationError(
                            "Native graph examples require one parent per FK role"
                        )
                    if _clock(link.available_at) != _clock(
                        self.rows[target, link.target_id].available_at
                    ):
                        raise UnsupportedCombinationError(
                            "Native FK clocks must equal the event availability clock"
                        )
                    mapping[dest] = self.codes[source][link.source_id]
                    clocks[mapping[dest], dest] = _clock(link.available_at)
                parents[role] = mapping
                self.edge_parents[relation] = torch.full(
                    (len(self.keys[target]),), -1, dtype=torch.long
                )
                self.edge_clocks[relation] = torch.full(
                    (len(self.keys[target]),), STATIC_TIME, dtype=torch.long
                )
                if mapping:
                    destinations = torch.tensor(list(mapping), dtype=torch.long)
                    self.edge_parents[relation][destinations] = torch.tensor(
                        list(mapping.values()), dtype=torch.long
                    )
                    self.edge_clocks[relation][destinations] = torch.tensor(
                        [clocks[parent, child] for child, parent in mapping.items()],
                        dtype=torch.long,
                    )
                schema_relations.append(api.RelationSpec(role, source, "id", target, f"fk_{no}"))
            for name, table in self.tables.items():
                if name not in active:
                    continue
                columns: dict[str, Any] = {
                    "id": pa.array(range(len(self.keys[name])), type=pa.int64())
                }
                types = {"id": api.DType("int64")}
                for field in table.features:
                    columns[field] = pa.array(
                        [self.rows[name, key].values[field] for key in self.keys[name]],
                        type=pa.float32(),
                    )
                    types[field] = api.DType("float32")
                clock = api.NON_TEMPORAL
                if not table.immutable:
                    clock = "available"
                    columns[clock] = pa.array(
                        [self.rows[name, key].available_at for key in self.keys[name]],
                        type=pa.timestamp("s", tz="UTC"),
                    )
                    types[clock] = api.DType("timestamp", unit="s", timezone="UTC")
                for no, (_, role, target) in enumerate(graph.relations):
                    if target == name:
                        column = f"fk_{no}"
                        columns[column] = pa.array(
                            [parents[role].get(i) for i in range(len(self.keys[name]))],
                            type=pa.int64(),
                        )
                        types[column] = api.DType("int64")
                pq.write_table(pa.table(columns), self.native_root / f"{name}.parquet")
                schema_tables.append(api.TableSpec(name, "id", types, time_column=clock))
            # Bipartite expansion alternates directions, so no physical FK role
            # appears twice in one native layer. Both message directions are
            # declared explicitly to the model-facing adapter below.
            neighborhood = api.NeighborhoodSpec(
                tuple(
                    api.GraphLayer(
                        tuple(
                            api.GraphSample(
                                role,
                                fanout,
                                direction="forward" if hop % 2 == 0 else "reverse",
                                strategy="recent" if hop % 2 == 0 else "uniform",
                            )
                            for _, role, _ in active_relations
                        )
                    )
                    for hop in range(depth)
                )
            )
            contract = api.DataLoaderSpec(
                api.DataSchema(tuple(schema_tables), tuple(schema_relations)),
                api.SampleSpec(api.EntitySpec("users"), neighborhoods={"neighbors": neighborhood}),
                api.BatchSpec(
                    {
                        "graph": api.Input(
                            api.NeighborhoodSource("neighbors"),
                            api.Graph(
                                {
                                    name: api.Fields(
                                        tuple(
                                            api.Encode(field, api.Raw()) for field in table.features
                                        )
                                    )
                                    for name, table in self.tables.items()
                                    if name in active
                                },
                                topology="relations",
                            ),
                            None,
                        )
                    }
                ),
            )
            dataset = api.Dataset(
                {
                    name: api.Parquet(str(self.native_root / f"{name}.parquet"))
                    for name in self.tables
                    if name in active
                }
            )
            self.native_plan = self.native_stack.enter_context(
                api.prepare(
                    contract,
                    dataset,
                    temporary_engine(self.native_stack),
                )
            )
            self.native_calls = 0
            self._finalizer = finalize(self, self.native_stack.close)
        except BaseException:
            self.native_stack.close()
            raise

    def close(self) -> None:
        """Close the native plan before removing its sources/index."""
        self._finalizer()

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
        if not anchors or root_type != "users" or sharing not in {"shared", "disjoint"}:
            raise ValueError("Native graph sampling requires user anchors and shared/disjoint mode")
        if any(a.observation_start is not None or utc_us(a.cutoff) % 1_000_000 for a in anchors):
            raise UnsupportedCombinationError(
                "Native graph example requires second-aligned cutoffs without observation_start; select data.loader: python otherwise"
            )
        if sharing == "shared" and len({utc_us(a.cutoff) for a in anchors}) != 1:
            raise ValueError("Shared graphs require identical cutoffs")
        ids = tuple(self.codes[root_type][a.entity_id] for a in anchors)
        path = self.native_root / f"anchors-{self.native_calls}.parquet"
        self.native_calls += 1
        pq.write_table(
            pa.table(
                {
                    "sample_id": [a.sample_id for a in anchors],
                    "owner": pa.array(ids, type=pa.int64()),
                    "at": pa.array([a.cutoff for a in anchors], type=pa.timestamp("s", tz="UTC")),
                    "cutoff_us": pa.array([utc_us(a.cutoff) for a in anchors], type=pa.int64()),
                    "split": ["train"] * len(anchors),
                }
            ),
            path,
        )
        binding = self.api.AnchorTable(
            self.api.Parquet(str(path)),
            entity_id="owner",
            as_of_time="at",
            sample_id="sample_id",
            split="split",
            metadata=("cutoff_us",),
        )
        try:
            with self.native_plan.materialize(binding, fit_split="train") as task:
                batches = iter(
                    task.batches(
                        split="train", batch_size=len(anchors), shuffle=False, output_format="model"
                    )
                )
                native = next(batches)
                if next(batches, None) is not None:
                    raise ValueError("Native graph split produced unexpected extra batches")
                self._assert_bindings(native, anchors, ids)
                payload = self._owned_graph(native.inputs["graph"], len(anchors))
                view = _BoundGraph({"graph": payload}, native.metadata)
                node_clocks = {
                    name: self.node_clocks[name][node["ids"]]
                    for name, node in payload["nodes"].items()
                }
                edge_clocks = {}
                for source, role, target in self.graph.relations:
                    edge = payload["edges"][role]["index"]
                    source_ids = payload["nodes"][source]["ids"][edge[0]]
                    target_ids = payload["nodes"][target]["ids"][edge[1]]
                    parents = self.edge_parents.get((source, role, target))
                    if parents is None:
                        if edge.shape[1]:
                            raise ValueError("Native graph returned an unbound edge")
                        edge_clocks[source, role, target] = torch.empty(0, dtype=torch.long)
                    else:
                        if not torch.equal(parents[target_ids], source_ids):
                            raise ValueError("Native graph returned an unbound edge")
                        edge_clocks[source, role, target] = self.edge_clocks[source, role, target][
                            target_ids
                        ]
                result = from_kumo_graph(
                    view,
                    anchors=anchors,
                    root_type=root_type,
                    root_node_ids=ids,
                    feature_columns={name: table.features for name, table in self.tables.items()},
                    relations=self.graph.directed_relations,
                    forbidden_fields=self.graph.forbidden_fields,
                    node_available_us=node_clocks,
                    edge_available_us=edge_clocks,
                )
                # Validate the disjoint result before any intentional union, so
                # shared mode cannot conceal invalid duplicate native edges.
                if sharing == "shared":
                    result = self._union(result, payload["nodes"])
                return order_graph_by_hop(result, self.depth) if hop_trim else result
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _assert_bindings(
        native: Any, anchors: tuple[GraphAnchor, ...], ids: tuple[int, ...]
    ) -> None:
        if tuple(int(i) for i in native.metadata["entity_ids"]) != ids:
            raise ValueError("Native root/entity alignment changed")
        if tuple(int(x["cutoff_us"]) for x in native.metadata["request_metadata"]) != tuple(
            utc_us(a.cutoff) for a in anchors
        ):
            raise ValueError("Native graph cutoff bindings changed")
        scale = {
            "seconds": 1,
            "milliseconds": 1000,
            "microseconds": 1_000_000,
            "nanoseconds": 1_000_000_000,
        }
        unit = native.metadata.get("timestamp_unit", "seconds")
        if unit not in scale or tuple(int(t) for t in native.metadata["as_of_time"]) != tuple(
            utc_us(a.cutoff) // 1_000_000 * scale[unit] for a in anchors
        ):
            raise ValueError("Native graph sampling clock differs from its bound anchor")

    def _owned_graph(self, graph: Mapping[str, Any], sample_count: int) -> dict[str, Any]:
        if set(graph["nodes"]) - set(self.tables) or set(graph["edges"]) - {
            r for _, r, _ in self.graph.relations
        }:
            raise ValueError("Native graph returned undeclared tables or roles")
        nodes = {}
        for name, table in self.tables.items():
            if name in graph["nodes"]:
                node = graph["nodes"][name]
                nodes[name] = {
                    "ids": node["ids"].clone(),
                    "batch": node["batch"].clone(),
                    "features": {
                        field: {k: v.clone() for k, v in node["features"][field].items()}
                        for field in table.features
                    },
                }
            else:
                nodes[name] = {
                    "ids": torch.empty(0, dtype=torch.long),
                    "batch": torch.empty(0, dtype=torch.long),
                    "features": {
                        field: {"values": torch.empty(0), "valid": torch.empty(0, dtype=torch.bool)}
                        for field in table.features
                    },
                }
            ids, owners = nodes[name]["ids"], nodes[name]["batch"]
            if ids.dtype != torch.long or ids.ndim != 1 or ids.shape != owners.shape:
                raise ValueError("Invalid native canonical row IDs")
            if owners.dtype != torch.long or bool(((owners < 0) | (owners >= sample_count)).any()):
                raise ValueError("Invalid native node sample IDs")
            if bool(((ids < 0) | (ids >= len(self.keys[name]))).any()):
                raise ValueError("Native graph returned an unbound canonical row ID")
            if _has_duplicate_edges(torch.stack((owners, ids)), sample_count, len(self.keys[name])):
                raise ValueError("Duplicate native node identity within one sample")
        edges = {}
        for source, role, target in self.graph.relations:
            edge = graph["edges"].get(role)
            edges[role] = (
                {
                    "source_type": source,
                    "destination_type": target,
                    "index": torch.empty((2, 0), dtype=torch.long),
                }
                if edge is None
                else {**edge, "index": edge["index"].clone()}
            )
            index = edges[role]["index"]
            if index.dtype != torch.long or index.ndim != 2 or index.shape[0] != 2:
                raise ValueError("Invalid native edge indices")
            if bool((index < 0).any()) or (
                index.shape[1]
                and (
                    int(index[0].max()) >= len(nodes[source]["ids"])
                    or int(index[1].max()) >= len(nodes[target]["ids"])
                )
            ):
                raise ValueError("Native edge index is outside its bound nodes")
        return {"nodes": nodes, "edges": edges}

    def _union(self, batch: GraphBatch, nodes: dict[str, Any]) -> GraphBatch:
        batch.assert_validated()
        first, inverse = {}, {}
        for name, node in nodes.items():
            first[name], inverse[name] = _first_occurrences(node["ids"])
        edges, clocks = {}, {}
        for role, index in batch.edge_index.items():
            source, _, target = role
            remapped = torch.stack((inverse[source][index[0]], inverse[target][index[1]]))
            source_size, target_size = len(first[source]), len(first[target])
            keys = (
                remapped[0] * target_size + remapped[1]
                if source_size * target_size <= torch.iinfo(torch.int64).max
                else remapped.T
            )
            chosen, _inverse = _first_occurrences(keys)
            edges[role], clocks[role] = remapped[:, chosen], batch.edge_available_us[role][chosen]
        return GraphBatch(
            batch.anchors,
            batch.root_type,
            {n: batch.node_features[n][indices] for n, indices in first.items()},
            {n: torch.zeros(len(indices), dtype=torch.long) for n, indices in first.items()},
            {n: batch.node_available_us[n][indices] for n, indices in first.items()},
            edges,
            clocks,
            inverse[batch.root_type][batch.root_index],
            sharing="shared",
        )

    @property
    def tie_policy(self) -> str:
        """Native recent-fanout ties follow index-local ordering."""
        return "native index-local recent ties"

    @property
    def loader_manifest(self) -> dict[str, str]:
        """Native provenance; future targets are deliberately a separate reference pass."""
        return {
            "name": "kumo_pytorch",
            "revision": SOURCE_REVISION,
            "graph_sampling": "native_recent",
            "target_builder": "python_reference",
        }
