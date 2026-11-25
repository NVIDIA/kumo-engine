# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt delivered native tensors, without event lookup or Python tokenization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import torch
from torch import Tensor

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import SchemaError
from tfm.contracts.graph import (
    STATIC_TIME,
    GraphAnchor,
    GraphBatch,
    GraphHistoryBatch,
    GraphSupervisedBatch,
    SequencePredictionBatch,
    SequenceSupervisedBatch,
)
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.table_source import TableSource
from tfm.data.adapters.kumo_graph import from_kumo_graph
from tfm.data.graph import order_graph_by_hop
from tfm.data.native_source import AVAILABLE_US


def _pad_events(value: Tensor) -> Tensor:
    return value if value.shape[1] else value.new_zeros((value.shape[0], 1, *value.shape[2:]))


def _owned(value: Tensor) -> Tensor:
    if value.device.type != "cpu":
        raise SchemaError("Native tensor adaptation requires CPU delivery")
    return value.detach().clone()


def event_grid(history: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gather native retained events into owned [batch,event,field] tensors."""
    event_mask = _pad_events(history["sequence"]["valid"].bool()).clone()
    values = history["value_ids"]
    if not values.numel():
        zeros = torch.zeros((*event_mask.shape, 1), dtype=torch.long)
        return zeros, zeros.clone(), zeros.clone(), zeros.bool(), event_mask
    indices = _pad_events(history["sequence"]["event_indices"]).masked_fill(~event_mask, 0)
    valid = history["valid"][indices].bool() & event_mask.unsqueeze(-1)
    arrays = tuple(
        _owned(history[name][indices].long().masked_fill(~valid, 0))
        for name in ("field_ids", "value_ids", "position_ids")
    )
    return arrays[0], arrays[1], arrays[2], valid.clone(), event_mask


def _bindings(metadata: Mapping[str, Any]) -> tuple[tuple[GraphAnchor, ...], tuple[int, ...]]:
    unit = metadata["timestamp_unit"]
    factors = {"seconds": 1_000_000, "milliseconds": 1000, "microseconds": 1}
    if unit not in factors:
        raise SchemaError(f"Unsupported native anchor timestamp unit: {unit}")
    times = metadata["as_of_time"].tolist()
    requests = metadata["request_metadata"]
    ids = metadata["entity_ids"]
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    samples = metadata["sample_ids"]
    if not len(times) == len(requests) == len(ids) == len(samples):
        raise SchemaError("Native anchor metadata is misaligned")
    anchors = tuple(
        GraphAnchor(
            str(sid),
            str(extra.get("entity_id", owner)),
            datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=int(at) * factors[unit]),
        )
        for sid, owner, at, extra in zip(samples, ids, times, requests, strict=True)
    )
    return anchors, tuple(int(owner) for owner in ids)


@dataclass
class _GraphDelivery:
    inputs: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _share_graph(batch: GraphBatch, nodes: Mapping[str, Any]) -> GraphBatch:
    batch.assert_validated()
    if len({anchor.cutoff for anchor in batch.anchors}) != 1:
        raise SchemaError("Shared native graphs require identical anchor cutoffs")
    first, inverse = {}, {}
    for name, node in nodes.items():
        seen, positions, remap = {}, [], []
        for i, key in enumerate(node["ids"].tolist()):
            if key not in seen:
                seen[key] = len(positions)
                positions.append(i)
            remap.append(seen[key])
        first[name] = torch.tensor(positions, dtype=torch.long)
        inverse[name] = torch.tensor(remap, dtype=torch.long)
    edges, clocks = {}, {}
    for role, index in batch.edge_index.items():
        source, _, target = role
        remapped = torch.stack((inverse[source][index[0]], inverse[target][index[1]]))
        selected = {}
        for i, (send, receive) in enumerate(remapped.T.tolist()):
            selected.setdefault((send, receive), i)
        chosen = torch.tensor(list(selected.values()), dtype=torch.long)
        edges[role], clocks[role] = remapped[:, chosen], batch.edge_available_us[role][chosen]
    return GraphBatch(
        batch.anchors,
        batch.root_type,
        {name: batch.node_features[name][indices] for name, indices in first.items()},
        {name: torch.zeros(len(indices), dtype=torch.long) for name, indices in first.items()},
        {name: batch.node_available_us[name][indices] for name, indices in first.items()},
        edges,
        clocks,
        inverse[batch.root_type][batch.root_index],
        sharing="shared",
    )


class NativeTensorAdapter:
    """Copy one native delivery into validated TFM inputs and supervision.

    Encoded IDs already belong to the fitted native model namespace. Availability
    columns travel as metadata and never join numeric model features. Event
    aggregates use only the retained history and treat non-finite amounts as
    missing. Caller-supplied entity labels travel separately on anchor metadata.
    """

    def __init__(
        self,
        source: TableSource,
        *,
        root: str,
        mode: Literal["history", "sequence", "graph", "hybrid", "flat"] = "history",
        depth: int = 1,
        max_tokens: int = 128,
        require_labels: bool = False,
        sharing: Literal["shared", "disjoint"] = "disjoint",
        hop_trim: bool = True,
        target_level: Literal["entity", "sequence"] = "entity",
        history_limit: int | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("Native history needs a positive per-event token cap")
        self.source, self.root, self.mode = source, root, mode
        self.depth, self.max_tokens, self.require_labels = depth, max_tokens, require_labels
        self.sharing, self.hop_trim = sharing, hop_trim
        self.target_level: Literal["entity", "sequence"] = target_level
        if history_limit is not None and history_limit < 1:
            raise ValueError("Native history limit must be positive")
        self.history_limit = history_limit

    def history(
        self, inputs: Mapping[str, Any], anchors: tuple[GraphAnchor, ...]
    ) -> KeyValueTimeBatch:
        """Build KVT arrays directly; no event objects or tokenizer are retained."""
        keys, values, positions, valid, event_mask = event_grid(inputs["history"])
        auxiliary = inputs["history_aux"]
        times = {
            name: _owned(_pad_events(auxiliary[name]["values"]))
            for name in ("__hour", "__dow", "__dom", "__age", "__amount")
        }
        if self.history_limit is not None and event_mask.shape[1] > self.history_limit:
            count = event_mask.sum(1)
            indices = (count - self.history_limit).clamp_min(0).unsqueeze(1) + torch.arange(
                self.history_limit
            )
            mask = indices < count.unsqueeze(1)
            indices = indices.clamp_max(event_mask.shape[1] - 1)

            def fields(value: Tensor) -> Tensor:
                return value.gather(1, indices.unsqueeze(-1).expand(-1, -1, value.shape[2])).clone()

            keys, values, positions, valid = (
                fields(value) for value in (keys, values, positions, valid)
            )
            event_mask = mask & event_mask.gather(1, indices)
            valid &= event_mask.unsqueeze(-1)
            keys, values, positions = (
                value.masked_fill(~valid, 0) for value in (keys, values, positions)
            )
            times = {
                name: value.gather(1, indices).masked_fill(~event_mask, 0)
                for name, value in times.items()
            }
        ages = times["__age"].double()
        if bool((ages[event_mask] < 0).any()):
            raise SchemaError("Native history contains future events")
        last = (event_mask.sum(1) - 1).clamp_min(0).unsqueeze(1)
        elapsed = (ages[:, :1] - ages).masked_fill(~event_mask, 0)
        recency = (ages - ages.gather(1, last)).masked_fill(~event_mask, 0)
        if bool((elapsed < 0).any()) or bool((recency < 0).any()):
            raise SchemaError("Native history must be ordered chronologically")
        amounts = times["__amount"].double()
        amounts = amounts.masked_fill(~event_mask | ~torch.isfinite(amounts), 0)
        counts = event_mask.sum(1)
        maximum = amounts.masked_fill(~event_mask, -torch.inf).amax(1)
        maximum = torch.where(counts > 0, maximum, 0)
        raw = torch.stack(
            (
                amounts.sum(1) / counts.clamp_min(1),
                maximum,
                counts,
                times["__hour"].gather(1, last).squeeze(1),
                times["__dow"].gather(1, last).squeeze(1),
            ),
            dim=1,
        ).float()
        raw[counts == 0] = 0
        # An entity marker represents an empty profile. Optional profile
        # deliveries use their own native field/value IDs without a refit.
        profile = inputs.get("profile")
        if profile is None:
            profile_values = torch.full((len(anchors), 1), 3, dtype=torch.long)
            profile_keys = profile_values.clone()
            profile_positions = torch.zeros_like(profile_values)
            profile_mask = torch.ones_like(profile_values, dtype=torch.bool)
        else:
            profile_mask = _owned(profile["valid"].bool())
            profile_keys, profile_values, profile_positions = (
                _owned(profile[name].long().masked_fill(~profile_mask, 0))
                for name in ("field_ids", "value_ids", "position_ids")
            )
        profile_times = torch.zeros_like(profile_values, dtype=torch.float32)
        facts = inputs.get("profile_facts")
        if facts is not None:
            fact_keys, fact_values, fact_positions, fact_mask, fact_events = event_grid(facts)
            ages = _owned(_pad_events(inputs["profile_facts_age"]["__age"]["values"])).double()
            if bool((ages[fact_events] < 0).any()) or not bool(
                torch.isfinite(ages[fact_events]).all()
            ):
                raise SchemaError("Native profile contains future or invalid milestone clocks")
            fact_times = (8 * torch.log1p(ages / 8)).unsqueeze(-1).expand_as(fact_values)
            profile_keys = torch.cat((profile_keys, fact_keys.flatten(1)), dim=1)
            profile_values = torch.cat((profile_values, fact_values.flatten(1)), dim=1)
            profile_positions = torch.cat((profile_positions, fact_positions.flatten(1)), dim=1)
            profile_mask = torch.cat((profile_mask, fact_mask.flatten(1)), dim=1)
            profile_times = torch.cat(
                (
                    profile_times,
                    fact_times.flatten(1).float().masked_fill(~fact_mask.flatten(1), 0),
                ),
                dim=1,
            )
        return KeyValueTimeBatch(
            key_ids=keys[..., : self.max_tokens],
            value_ids=values[..., : self.max_tokens],
            positions=positions[..., : self.max_tokens],
            token_mask=valid[..., : self.max_tokens],
            event_mask=event_mask,
            time_log=(8 * torch.log1p(recency / 8)).float(),
            elapsed_log=(8 * torch.log1p(elapsed / 8)).float(),
            events={
                name: times[f"__{name}"].long().masked_fill(~event_mask, 0)
                for name in ("hour", "dow", "dom")
            },
            prof_key_ids=profile_keys,
            prof_value_ids=profile_values,
            prof_positions=profile_positions,
            prof_mask=profile_mask,
            prof_time_log=profile_times,
            entity_ids=tuple(a.entity_id for a in anchors),
            sequence_ids=tuple(a.sample_id for a in anchors),
            numeric_features=raw,
        )

    def graph(
        self, delivered: Any, anchors: tuple[GraphAnchor, ...], roots: tuple[int, ...]
    ) -> GraphBatch:
        """Bind features and availability to delivered source identities and edges."""
        payload = delivered.inputs["graph"]
        nodes, columns, clocks = {}, {}, {}
        tables = {
            table.name: table
            for table in self.source.tables
            if table.role == "input"
            and (self.source.graph_numeric is None or table.name in self.source.graph_numeric)
        }
        for name, table in tables.items():
            node = payload["nodes"].get(name)
            if node is None:
                numeric = (
                    table.numeric
                    if self.source.graph_numeric is None
                    else self.source.graph_numeric[name]
                )
                node = {
                    "ids": torch.empty(0, dtype=torch.long),
                    "batch": torch.empty(0, dtype=torch.long),
                    "features": {
                        column: {
                            "values": torch.empty(
                                0,
                                dtype=torch.long
                                if column in (*table.key, AVAILABLE_US)
                                else torch.float32,
                            ),
                            "valid": torch.empty(0, dtype=torch.bool),
                        }
                        for column in (*table.key, *numeric.values(), AVAILABLE_US)
                    },
                }
            fields = node["features"]
            key = fields[table.key[0]]
            if not bool(key["valid"].all()):
                raise SchemaError(f"{name}: native graph omitted source identities")
            ids = _owned(key["values"].long())
            features = {}
            numeric = (
                table.numeric
                if self.source.graph_numeric is None
                else self.source.graph_numeric[name]
            )
            for alias, column in numeric.items():
                raw_values = fields[column]["values"]
                values = _owned(raw_values.float())
                observed = _owned(fields[column]["valid"].bool())
                if bool((observed & torch.isfinite(raw_values) & ~torch.isfinite(values)).any()):
                    raise SchemaError(f"{name}.{column}: graph value overflows float32")
                finite = observed & torch.isfinite(values)
                feature = f"value:{alias}" if self.source.graph_numeric is None else alias
                if self.source.graph_numeric is not None and not bool(finite.all()):
                    raise SchemaError(
                        f"{name}.{column}: explicit graph features must be finite and present"
                    )
                features[feature] = {
                    "values": values.masked_fill(~finite, 0),
                    "valid": torch.ones_like(finite),
                }
                if self.source.graph_numeric is None:
                    features[f"present:{alias}"] = {
                        "values": finite.float(),
                        "valid": torch.ones_like(finite),
                    }
            if not features:
                features["bias"] = {
                    "values": torch.ones(len(ids)),
                    "valid": torch.ones(len(ids), dtype=torch.bool),
                }
            columns[name] = tuple(features)
            if table.immutable:
                clocks[name] = torch.full((len(ids),), STATIC_TIME, dtype=torch.long)
            else:
                known = fields[AVAILABLE_US]
                if not bool(known["valid"].all()):
                    raise SchemaError(f"{name}: native graph omitted availability")
                clocks[name] = _owned(known["values"].long())
            nodes[name] = {"ids": ids, "batch": _owned(node["batch"].long()), "features": features}
        edges, edge_clocks, relations = {}, {}, []
        for role in self.source.relationships:
            if role.parent not in tables or role.child not in tables:
                continue
            relation = role.parent, role.name, role.child
            native = payload["edges"].get(
                role.name,
                {
                    "source_type": role.parent,
                    "destination_type": role.child,
                    "index": torch.empty((2, 0), dtype=torch.long),
                },
            )
            index = _owned(native["index"].long())
            # The supported fixed-FK contract declares link observation equal
            # to child-row observation; ingestion rejects other FK clocks.
            observed = clocks[role.child][index[1]]
            edges[role.name] = {**native, "index": index}
            edge_clocks[relation] = observed.clone()
            edge_clocks[role.child, f"rev_{role.name}", role.parent] = observed.clone()
            relations.extend((relation, (role.child, f"rev_{role.name}", role.parent)))
        view = _GraphDelivery(
            inputs={"graph": {"nodes": nodes, "edges": edges}}, metadata=delivered.metadata
        )
        graph = from_kumo_graph(
            view,
            anchors=anchors,
            root_type=self.root,
            root_node_ids=roots,
            feature_columns=columns,
            relations=tuple(relations),
            forbidden_fields=frozenset(),
            node_available_us=clocks,
            edge_available_us=edge_clocks,
        )
        if self.sharing == "shared":
            graph = _share_graph(graph, nodes)
        return order_graph_by_hop(graph, self.depth) if self.hop_trim else graph

    @torch.inference_mode(False)
    @torch.no_grad()
    def __call__(self, delivered: Any) -> Any:
        anchors, roots = _bindings(delivered.metadata)
        # Padded distributed anchors must be excluded/weighted by a dedicated
        # distributed adapter; ordinary losses cannot silently count them.
        padding = delivered.metadata.get("is_padding")
        if padding is not None and bool(padding.any()):
            raise SchemaError("Native padded distributed anchors require a masked-loss adapter")
        labels = [row.get("target") for row in delivered.metadata["request_metadata"]]
        known = all(label is not None for label in labels)
        if self.require_labels and not known:
            raise SchemaError("Native supervised batches require available entity labels")
        target = torch.tensor(labels, dtype=torch.float32) if known else None
        if target is not None and (
            not bool(torch.isfinite(target).all()) or bool(((target != 0) & (target != 1)).any())
        ):
            raise SchemaError("Native entity targets must be finite binary labels")
        available, cutoff = None, None
        if target is not None:
            metadata = delivered.metadata["request_metadata"]
            if any(
                row.get("label_available_us") is None or row.get("label_cutoff_us") is None
                for row in metadata
            ):
                raise SchemaError(
                    "Native entity targets require explicit availability and fitting cutoffs"
                )
            cutoffs = {int(row["label_cutoff_us"]) for row in metadata}
            if len(cutoffs) != 1:
                raise SchemaError("Native supervised batches require one label fitting cutoff")
            cutoff = cutoffs.pop()
            available = torch.tensor(
                [int(row["label_available_us"]) for row in metadata], dtype=torch.long
            )
            if bool((available > cutoff).any()):
                raise SchemaError("Native entity target is unavailable at the fitting cutoff")
        if self.mode == "graph":
            graph = self.graph(delivered, anchors, roots)
            if target is None:
                return graph
            assert available is not None and cutoff is not None
            return GraphSupervisedBatch(
                graph, target, available, cutoff, target_level=self.target_level
            )
        history = self.history(delivered.inputs, anchors)
        if self.mode == "flat":
            # Preserve complete events when enforcing a flat token budget.
            rows = []
            for values, masks, events in zip(
                history.value_ids, history.token_mask, history.event_mask, strict=True
            ):
                retained, used = [], 2
                for value, mask in reversed(list(zip(values[events], masks[events], strict=True))):
                    tokens = value[mask]
                    cost = len(tokens) + bool(retained)
                    if used + cost > self.max_tokens:
                        break
                    retained.append(tokens)
                    used += cost
                row = [torch.tensor([1], dtype=torch.long)]
                for i, tokens in enumerate(reversed(retained)):
                    if i:
                        row.append(torch.tensor([3], dtype=torch.long))
                    row.append(tokens)
                row.append(torch.tensor([2], dtype=torch.long))
                rows.append(torch.cat(row))
            tokens = torch.zeros((len(rows), self.max_tokens), dtype=torch.long)
            mask = torch.zeros_like(tokens)
            for i, row in enumerate(rows):
                tokens[i, : len(row)] = row
                mask[i, : len(row)] = 1
            return TransactionBatch(
                tokens,
                mask,
                labels=tokens.masked_fill(~mask.bool(), -100),
                fraud_labels=target,
                numeric_features=history.numeric_features,
                entity_ids=history.entity_ids,
                sequence_ids=history.sequence_ids,
            )
        if self.mode == "history":
            return replace(history, fraud_labels=target)
        if self.mode == "sequence":
            if target is None:
                return SequencePredictionBatch(history, anchors, target_level=self.target_level)
            assert available is not None and cutoff is not None
            return SequenceSupervisedBatch(
                history, anchors, target, available, cutoff, target_level=self.target_level
            )
        return GraphHistoryBatch(
            self.graph(delivered, anchors, roots),
            history,
            anchors,
            target,
            available,
            cutoff,
            target_level=self.target_level,
        )
