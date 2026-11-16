"""Private checks and identity operations for native GraphSAGE inputs.

This reads Kumo's public batch fields without importing Kumo or producing an
adapted batch. Feature encoding and message directions belong to the model.
"""

from collections.abc import Mapping, Sequence
from numbers import Complex

import torch
from torch import Tensor


def mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def tensor(value, name, *, dtype=None, device=None, cpu=False):
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if cpu and value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for validate_kumo_batch")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the same device as the model")
    return value


def ids(value, name, *, device=None, cpu=False):
    if isinstance(value, Tensor):
        tensor(value, name, dtype=torch.long, device=device, cpu=cpu)
        if value.ndim != 1:
            raise ValueError(f"{name} must have shape [N]")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for identifier in value:
            if identifier is None or isinstance(identifier, Tensor):
                raise ValueError(f"{name} must contain non-null scalar IDs")
            try:
                hash(identifier)
            except TypeError as error:
                raise TypeError(f"{name} must contain hashable scalar IDs") from error
    else:
        raise TypeError(f"{name} must be a one-dimensional int64 tensor or a sequence of IDs")
    return value


def field(fields, name, count, context):
    if name not in fields:
        if count == 0:
            return None
        raise ValueError(f"{context} is missing configured field {name!r}")
    raw = mapping(fields[name], f"{context}.{name}")
    if "values" not in raw or "valid" not in raw:
        raise ValueError(f"{context}.{name} must contain values and valid")
    return raw


def numerical(raw, width):
    """Return tensor views with explicit component axes; do not cast or mask."""
    values, valid = raw["values"], raw["valid"]
    if values.ndim == 1 and width == 1:
        values = values.unsqueeze(-1)
    if valid.ndim == 1:
        valid = valid.unsqueeze(-1)
    if valid.ndim == 2 and valid.shape[1] == 1:
        valid = valid.expand(-1, width)
    return values, valid


def prediction_time(metadata, *, check_values=True):
    """Keep graph ages at prediction T when sampling used inclusive C = T - 1s.

    Kumo preserves user metadata in request_metadata. Executable counterparts
    have whole-second prediction times. Validate that explicit provenance before
    using the one-second offset; ordinary batches retain their original clock.
    """
    clock = metadata["as_of_time"]
    records = metadata.get("request_metadata", ())
    present = [isinstance(row, Mapping) and "prediction_time_ns" in row for row in records]
    if not any(present):
        return clock
    if len(records) != len(clock) or not all(present):
        raise ValueError("Every graph request must carry its original prediction time")
    stamps = [row["prediction_time_ns"] for row in records]
    if any(type(value) is not int or value % 1_000_000_000 for value in stamps):
        raise ValueError("Graph prediction times must be exact whole-second epoch nanoseconds")
    if check_values:
        expected = clock.new_tensor([value // 1_000_000_000 - 1 for value in stamps])
        if not torch.equal(clock, expected):
            raise ValueError("Graph execution clock differs from its declared prediction-time offset")
    return clock + 1


def inspect(batch, config, input_name, *, check_values, device=None, cpu=False):
    if not isinstance(input_name, str) or not input_name:
        raise ValueError("input_name must be a non-empty string")
    inputs = mapping(batch.inputs, "batch.inputs")
    if input_name not in inputs:
        raise ValueError(f"batch.inputs is missing graph input {input_name!r}")
    graph = mapping(inputs[input_name], "graph")
    if "nodes" not in graph or "edges" not in graph:
        raise ValueError("Kumo Graph output must contain nodes and edges")
    nodes, edges = mapping(graph["nodes"], "graph.nodes"), mapping(graph["edges"], "graph.edges")
    metadata = mapping(batch.metadata, "batch.metadata")
    if "entity_ids" not in metadata or "as_of_time" not in metadata:
        raise ValueError("batch.metadata must contain entity_ids and as_of_time")
    entities = ids(metadata["entity_ids"], "metadata.entity_ids", device=device, cpu=cpu)
    batch_size = len(entities)
    if not batch_size:
        raise ValueError("an Kumo graph batch must contain at least one request")
    seed_time = tensor(metadata["as_of_time"], "metadata.as_of_time", dtype=torch.long, device=device, cpu=cpu)
    if seed_time.shape != (batch_size,):
        raise ValueError("metadata.as_of_time must have one timestamp per entity ID")
    seed_time = prediction_time(metadata, check_values=check_values)
    unknown = set(nodes) - set(config.node_features)
    if unknown:
        raise ValueError(f"graph contains unconfigured node types: {sorted(unknown)!r}")
    counts, owners_by_type, root_lookup = {}, {}, {}
    for name, spec in config.node_features.items():
        context = f"nodes.{name}"
        if name not in nodes:
            counts[name] = 0
            continue
        node = mapping(nodes[name], context)
        if "ids" not in node or "batch" not in node:
            raise ValueError(f"{context} must contain ids and batch")
        identifiers = ids(node["ids"], f"{context}.ids", device=device, cpu=cpu)
        count = counts[name] = len(identifiers)
        owners = tensor(node["batch"], f"{context}.batch", dtype=torch.long, device=device, cpu=cpu)
        owners_by_type[name] = owners
        if owners.shape != (count,):
            raise ValueError(f"{context}.batch must have one owner per node ID")
        if check_values:
            if ((owners < 0) | (owners >= batch_size)).any().item():
                raise ValueError(f"{context}.batch contains an out-of-range example owner")
            physical = identifiers.tolist() if isinstance(identifiers, Tensor) else identifiers
            lookup = {}
            for index, key in enumerate(zip(owners.tolist(), physical)):
                if key in lookup:
                    raise ValueError(f"{context} contains duplicate (owner, ID) {key!r}")
                lookup[key] = index
            if name == config.root_type:
                root_lookup = lookup
        fields = mapping(node.get("features", {}), f"{context}.features")
        selected = list(spec.numerical) + list(spec.categorical)
        if spec.timestamp:
            selected.append(spec.timestamp)
        for key in selected:
            label = f"{context}.features.{key}"
            raw = field(fields, key, count, f"{context}.features")
            if raw is None:
                continue
            values = tensor(raw["values"], f"{label}.values", device=device, cpu=cpu)
            valid = tensor(raw["valid"], f"{label}.valid", dtype=torch.bool, device=device, cpu=cpu)
            if key in spec.numerical:
                width = spec.numerical[key]
                if values.is_complex():
                    raise TypeError(f"{label}.values must be real numerical values")
                values, valid = numerical(raw, width)
                if values.shape != (count, width):
                    raise ValueError(f"{label}.values must have shape [{count}, {width}]")
                if valid.shape != values.shape:
                    raise ValueError(f"{label}.valid must have shape [N], [N, 1], or [N, {width}]")
                if check_values and not torch.isfinite(values.to(torch.float32)[valid]).all().item():
                    raise ValueError(f"{label} contains non-finite valid numerical values")
            else:
                if values.dtype != torch.long:
                    raise TypeError(f"{label}.values must have dtype torch.int64")
                if values.shape != (count,) or valid.shape != (count,):
                    raise ValueError(f"{label}.values and valid must have shape [{count}]")
                if check_values and key == spec.timestamp:
                    if not valid.all().item():
                        raise ValueError(f"{label} contains missing temporal values")
                    if (values >= seed_time[owners]).any().item():
                        raise ValueError(f"{label} timestamps must be strictly before the owning request's as_of_time")
                elif check_values:
                    observed = values[valid]
                    if ((observed < 0) | (observed >= spec.categorical[key])).any().item():
                        raise ValueError(f"{label} contains valid categorical IDs outside [0, {spec.categorical[key]})")
    if check_values:
        physical = entities.tolist() if isinstance(entities, Tensor) else entities
        for owner, entity_id in enumerate(physical):
            if (owner, entity_id) not in root_lookup:
                raise ValueError(f"missing root {config.root_type!r} node for request {owner}, entity ID {entity_id!r}")
    configured = set(config.edge_types)
    for relation, raw in edges.items():
        raw = mapping(raw, f"edges.{relation}")
        if not all(key in raw for key in ("source_type", "destination_type", "index")):
            raise ValueError(f"edges.{relation} must contain source_type, destination_type, and index")
        source, destination = raw["source_type"], raw["destination_type"]
        if (source, relation, destination) not in configured:
            raise ValueError(f"graph contains unconfigured edge type {(source, relation, destination)!r}")
        index = tensor(raw["index"], f"edges.{relation}.index", dtype=torch.long, device=device, cpu=cpu)
        if index.ndim != 2 or index.shape[0] != 2:
            raise ValueError(f"edges.{relation}.index must have shape [2, E]")
        if check_values:
            if ((index < 0).any().item() or (index[0] >= counts[source]).any().item()
                    or (index[1] >= counts[destination]).any().item()):
                raise ValueError(f"edges.{relation}.index contains out-of-range node indices")
            if index.numel() and (owners_by_type[source][index[0]] != owners_by_type[destination][index[1]]).any().item():
                raise ValueError(f"edges.{relation} crosses example owners; Kumo graphs must be disjoint")
    return nodes, edges, metadata


def _integer_sequence(values, device):
    # Mixed tensor/host IDs compare with the same numeric equality as Python's
    # dictionary lookup. Non-integer host IDs cannot match an int64 tensor ID.
    encoded, valid = [], []
    for value in values:
        try:
            numeric = value.real if isinstance(value, Complex) and value.imag == 0 else value
            integer = int(numeric)
            matches = value == integer and -(2**63) <= integer < 2**63
        except (TypeError, ValueError, OverflowError):
            integer, matches = 0, False
        encoded.append(integer if matches else 0)
        valid.append(bool(matches))
    return torch.tensor(encoded, dtype=torch.long, device=device), torch.tensor(valid, dtype=torch.bool, device=device)


def root_indices(node_ids, owners, entity_ids):
    """Select roots in O(nodes + requests), without reading device tensors."""
    device = owners.device
    if isinstance(node_ids, Tensor) and isinstance(entity_ids, Tensor):
        matches = node_ids == entity_ids[owners]
    elif isinstance(node_ids, Tensor):
        encoded, valid = _integer_sequence(entity_ids, device)
        matches = (node_ids == encoded[owners]) & valid[owners]
    elif isinstance(entity_ids, Tensor):
        encoded, valid = _integer_sequence(node_ids, device)
        matches = (encoded == entity_ids[owners]) & valid
    else:
        codes = {}
        entities = [codes.setdefault(identifier, len(codes)) for identifier in entity_ids]
        nodes = [codes.setdefault(identifier, len(codes)) for identifier in node_ids]
        entities = torch.tensor(entities, dtype=torch.long, device=device)
        nodes = torch.tensor(nodes, dtype=torch.long, device=device)
        matches = nodes == entities[owners]
    positions = torch.arange(len(node_ids), dtype=torch.long, device=device) + 1
    selected = torch.where(matches, positions, 0)
    # Validation guarantees exactly one root per owner. A fixed-size reduction
    # avoids the device synchronization required by nonzero's dynamic output.
    roots = torch.zeros(len(entity_ids), dtype=torch.long, device=device)
    return roots.scatter_reduce(0, owners, selected, reduce="amax", include_self=True) - 1
