# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Kumo's disjoint graph output to model tensors without resampling.

This module deliberately does not import Kumo. It consumes the documented
``Batch.inputs`` and ``Batch.metadata`` interface, so the same adapter works
with both Kumo engines and does not install another copy of the platform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .config import GraphSAGEConfig

EdgeType = tuple[str, str, str]
FeatureValues = dict[str, dict[str, dict[str, Tensor]]]


@dataclass(frozen=True)
class GraphBatch:
    """A validated disjoint graph with explicit roots in request order.

    Numerical values and validity are ``[N, width]``; categorical values and
    validity are ``[N]``. ``batch_dict`` assigns each node to an example row,
    and ``seed_index`` identifies the local root node for every example.
    ``seed_time`` and ``time_dict`` retain integer epoch seconds.
    """

    features: FeatureValues
    edge_index_dict: dict[EdgeType, Tensor]
    batch_dict: dict[str, Tensor]
    seed_index: Tensor
    seed_time: Tensor
    root_type: str
    time_dict: dict[str, Tensor] = field(default_factory=dict)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> GraphBatch:
        """Move every model tensor together, preserving the original batch."""

        def move(tensor: Tensor) -> Tensor:
            return tensor.to(device=device, non_blocking=non_blocking)

        return GraphBatch(
            features={
                node: {
                    name: {key: move(tensor) for key, tensor in value.items()}
                    for name, value in fields.items()
                }
                for node, fields in self.features.items()
            },
            edge_index_dict={edge: move(index) for edge, index in self.edge_index_dict.items()},
            batch_dict={node: move(owners) for node, owners in self.batch_dict.items()},
            seed_index=move(self.seed_index),
            seed_time=move(self.seed_time),
            root_type=self.root_type,
            time_dict={node: move(time) for node, time in self.time_dict.items()},
        )


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _cpu_tensor(value: Any, name: str, dtype: torch.dtype | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(
            f"{name} must be on CPU; call KumoGraphAdapter before GraphBatch.to(device) "
            "to avoid accelerator synchronization during validation"
        )
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    return value


def _ids(value: Any, name: str) -> list[Any]:
    if isinstance(value, Tensor):
        value = _cpu_tensor(value, name, torch.long)
        if value.ndim != 1:
            raise ValueError(f"{name} must have shape [N]")
        result = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
    else:
        raise TypeError(f"{name} must be a one-dimensional int64 tensor or a sequence of IDs")
    for identifier in result:
        if identifier is None or isinstance(identifier, Tensor):
            raise ValueError(f"{name} must contain non-null scalar IDs")
        try:
            hash(identifier)
        except TypeError as error:
            raise TypeError(f"{name} must contain hashable scalar IDs") from error
    return result


def _field(raw_fields: Mapping, name: str, count: int, context: str) -> Mapping | None:
    if name not in raw_fields:
        if count == 0:
            return None
        raise ValueError(f"{context} is missing configured field {name!r}")
    result = _mapping(raw_fields[name], f"{context}.{name}")
    if "values" not in result or "valid" not in result:
        raise ValueError(f"{context}.{name} must contain values and valid")
    return result


def _numerical(raw: Mapping | None, count: int, width: int, name: str) -> dict[str, Tensor]:
    if raw is None:
        return {
            "values": torch.empty((0, width), dtype=torch.float32),
            "valid": torch.empty((0, width), dtype=torch.bool),
        }
    values = _cpu_tensor(raw["values"], f"{name}.values")
    valid = _cpu_tensor(raw["valid"], f"{name}.valid", torch.bool)
    if values.ndim == 1 and width == 1:
        values = values.unsqueeze(-1)
    if values.shape != (count, width):
        raise ValueError(f"{name}.values must have shape [{count}, {width}]")
    if values.is_complex():
        raise TypeError(f"{name}.values must be real numerical values")
    if valid.shape == (count,):
        valid = valid.unsqueeze(-1)
    if valid.shape == (count, 1):
        valid = valid.expand(count, width)
    if valid.shape != values.shape:
        raise ValueError(f"{name}.valid must have shape [N], [N, 1], or [N, {width}]")
    values = values.to(dtype=torch.float32)
    if not torch.isfinite(values[valid]).all().item():
        raise ValueError(f"{name} contains non-finite valid numerical values")
    return {"values": torch.where(valid, values, 0.0), "valid": valid}


def _categorical(raw: Mapping | None, count: int, size: int, name: str) -> dict[str, Tensor]:
    if raw is None:
        return {
            "values": torch.empty(0, dtype=torch.long),
            "valid": torch.empty(0, dtype=torch.bool),
        }
    values = _cpu_tensor(raw["values"], f"{name}.values", torch.long)
    valid = _cpu_tensor(raw["valid"], f"{name}.valid", torch.bool)
    if values.shape != (count,) or valid.shape != (count,):
        raise ValueError(f"{name}.values and valid must have shape [{count}]")
    selected = values[valid]
    if ((selected < 0) | (selected >= size)).any().item():
        raise ValueError(f"{name} contains valid categorical IDs outside [0, {size})")
    return {"values": torch.where(valid, values, 0), "valid": valid}


class KumoGraphAdapter:
    """Validate and adapt ``Batch.inputs[input_name]`` from Kumo's Graph output.

    Fit encodings, temporal filtering, and fanout sampling remain in Kumo. This
    adapter preserves example ownership, resolves roots by ``(owner, ID)``,
    and adds the configured reverse message edges once. Raw timestamps must
    be requested with ``Raw(dtype="int64")`` in the future dataset contract;
    normalized or missing timestamps cannot supply relative-time encoding.

    Adapt CPU batches before moving them to the model device. Extra feature
    fields are ignored, but unknown node/edge types are refused. Configured
    node or edge types absent from a batch are represented by empty tensors.
    """

    def __init__(self, config: GraphSAGEConfig, input_name: str = "neighborhood") -> None:
        if not isinstance(input_name, str) or not input_name:
            raise ValueError("input_name must be a non-empty string")
        self.config = config
        self.input_name = input_name

    def __call__(self, kumo_batch: Any) -> GraphBatch:
        inputs = _mapping(kumo_batch.inputs, "batch.inputs")
        if self.input_name not in inputs:
            raise ValueError(f"batch.inputs is missing graph input {self.input_name!r}")
        graph = _mapping(inputs[self.input_name], f"batch.inputs[{self.input_name!r}]")
        if "nodes" not in graph or "edges" not in graph:
            raise ValueError("Kumo Graph output must contain nodes and edges")
        nodes = _mapping(graph["nodes"], "graph.nodes")
        edges = _mapping(graph["edges"], "graph.edges")
        metadata = _mapping(kumo_batch.metadata, "batch.metadata")
        if "entity_ids" not in metadata or "as_of_time" not in metadata:
            raise ValueError("batch.metadata must contain entity_ids and as_of_time")
        entity_ids = _ids(metadata["entity_ids"], "metadata.entity_ids")
        batch_size = len(entity_ids)
        if batch_size == 0:
            raise ValueError("an Kumo graph batch must contain at least one request")
        seed_time = _cpu_tensor(metadata["as_of_time"], "metadata.as_of_time", torch.long)
        if seed_time.shape != (batch_size,):
            raise ValueError("metadata.as_of_time must have one timestamp per entity ID")
        from ._kumo import prediction_time
        seed_time = prediction_time(metadata)

        unknown_nodes = set(nodes) - set(self.config.node_features)
        if unknown_nodes:
            raise ValueError(f"graph contains unconfigured node types: {sorted(unknown_nodes)!r}")
        features: FeatureValues = {}
        batch_dict: dict[str, Tensor] = {}
        time_dict: dict[str, Tensor] = {}
        node_lookup: dict[str, dict[tuple[int, Any], int]] = {}
        for node_type, spec in self.config.node_features.items():
            if node_type in nodes:
                node = _mapping(nodes[node_type], f"nodes.{node_type}")
                if "ids" not in node or "batch" not in node:
                    raise ValueError(f"nodes.{node_type} must contain ids and batch")
                identifiers = _ids(node["ids"], f"nodes.{node_type}.ids")
                owners = _cpu_tensor(node["batch"], f"nodes.{node_type}.batch", torch.long)
                raw_fields = _mapping(node.get("features", {}), f"nodes.{node_type}.features")
            else:
                identifiers = []
                owners = torch.empty(0, dtype=torch.long)
                raw_fields = {}
            count = len(identifiers)
            if owners.shape != (count,):
                raise ValueError(f"nodes.{node_type}.batch must have one owner per node ID")
            if ((owners < 0) | (owners >= batch_size)).any().item():
                raise ValueError(f"nodes.{node_type}.batch contains an out-of-range example owner")
            lookup = {}
            for index, (owner, identifier) in enumerate(zip(owners.tolist(), identifiers)):
                key = (owner, identifier)
                if key in lookup:
                    raise ValueError(f"nodes.{node_type} contains duplicate (owner, ID) {key!r}")
                lookup[key] = index
            node_lookup[node_type] = lookup
            batch_dict[node_type] = owners
            context = f"nodes.{node_type}.features"
            features[node_type] = {
                name: _numerical(_field(raw_fields, name, count, context), count, width, f"{context}.{name}")
                for name, width in spec.numerical.items()
            }
            features[node_type].update({
                name: _categorical(_field(raw_fields, name, count, context), count, size, f"{context}.{name}")
                for name, size in spec.categorical.items()
            })
            if spec.timestamp is not None:
                raw_time = _field(raw_fields, spec.timestamp, count, context)
                if raw_time is None:
                    node_time = torch.empty(0, dtype=torch.long)
                else:
                    name = f"{context}.{spec.timestamp}"
                    node_time = _cpu_tensor(raw_time["values"], f"{name}.values", torch.long)
                    valid = _cpu_tensor(raw_time["valid"], f"{name}.valid", torch.bool)
                    if node_time.shape != (count,) or valid.shape != (count,):
                        raise ValueError(f"{name} timestamps and validity must have shape [{count}]")
                    if not valid.all().item():
                        raise ValueError(f"{name} contains missing temporal values")
                    if (node_time >= seed_time[owners]).any().item():
                        raise ValueError(f"{name} timestamps must be strictly before the owning request's as_of_time")
                time_dict[node_type] = node_time

        root_lookup = node_lookup[self.config.root_type]
        seed_indices = []
        for owner, entity_id in enumerate(entity_ids):
            key = (owner, entity_id)
            if key not in root_lookup:
                raise ValueError(f"missing root {self.config.root_type!r} node for request {owner}, entity ID {entity_id!r}")
            seed_indices.append(root_lookup[key])

        edge_index_dict: dict[EdgeType, Tensor] = {}
        configured_edges = set(self.config.edge_types)
        for relation, raw_edge in edges.items():
            raw_edge = _mapping(raw_edge, f"edges.{relation}")
            if not all(key in raw_edge for key in ("source_type", "destination_type", "index")):
                raise ValueError(f"edges.{relation} must contain source_type, destination_type, and index")
            edge_type = (raw_edge["source_type"], relation, raw_edge["destination_type"])
            if edge_type not in configured_edges:
                raise ValueError(f"graph contains unconfigured edge type {edge_type!r}")
            index = _cpu_tensor(raw_edge["index"], f"edges.{relation}.index", torch.long)
            if index.ndim != 2 or index.shape[0] != 2:
                raise ValueError(f"edges.{relation}.index must have shape [2, E]")
            source_type, _, destination_type = edge_type
            source_owners = batch_dict[source_type]
            destination_owners = batch_dict[destination_type]
            if (
                (index < 0).any().item()
                or (index[0] >= source_owners.numel()).any().item()
                or (index[1] >= destination_owners.numel()).any().item()
            ):
                raise ValueError(f"edges.{relation}.index contains out-of-range node indices")
            if (source_owners[index[0]] != destination_owners[index[1]]).any().item():
                raise ValueError(f"edges.{relation} crosses example owners; Kumo graphs must be disjoint")
            edge_index_dict[edge_type] = index.contiguous()
        # A stable complete set also keeps parameter usage consistent when a
        # minibatch has no observations for a particular configured relation.
        edge_index_dict = {
            edge_type: edge_index_dict.get(edge_type, torch.empty((2, 0), dtype=torch.long))
            for edge_type in self.config.edge_types
        }
        if self.config.add_reverse_edges:
            for source_type, relation, destination_type in self.config.edge_types:
                edge_index_dict[(destination_type, f"rev__{relation}", source_type)] = (
                    edge_index_dict[(source_type, relation, destination_type)].flip(0).contiguous()
                )
        return GraphBatch(
            features=features,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            seed_index=torch.tensor(seed_indices, dtype=torch.long),
            seed_time=seed_time,
            root_type=self.config.root_type,
            time_dict=time_dict,
        )
