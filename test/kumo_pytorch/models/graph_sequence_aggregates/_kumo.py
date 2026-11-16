"""Check aligned native fusion inputs without constructing an adapted batch."""

from collections.abc import Mapping, Sequence

import torch


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _tensor(value, name, *, dtype=None, cpu=False, device=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if cpu and value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for validate_kumo_batch")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the same device as the model")
    return value


def _identifiers(value, name, *, cpu=False):
    if isinstance(value, torch.Tensor):
        _tensor(value, name, dtype=torch.long, cpu=cpu)
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    elif not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional int64 tensor or sequence of IDs")
    return value


def _identity_values(value):
    return tuple(value.tolist()) if isinstance(value, torch.Tensor) else tuple(value)


def aggregate_views(raw, width):
    """Return views with a component axis; casting and masking stay in forward."""
    values, valid = raw["values"], raw["valid"]
    if values.ndim == 1 and width == 1:
        values = values.unsqueeze(-1)
    if valid.ndim == 1:
        valid = valid.unsqueeze(-1)
    if valid.ndim == 2 and valid.shape[1] == 1:
        valid = valid.expand(-1, width)
    return values, valid


def inspect(batch, config, graph_input, sequence_input, aggregate_input, *, check_values, cpu=False, device=None):
    names = (graph_input, sequence_input, aggregate_input)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("input names must be nonempty strings")
    if len(set(names)) != 3:
        raise ValueError("graph, sequence, and aggregate inputs must have distinct names")
    inputs, metadata = _mapping(batch.inputs, "batch.inputs"), _mapping(batch.metadata, "batch.metadata")
    for name in names:
        if name not in inputs:
            raise ValueError(f"batch.inputs is missing input {name!r}")
    if "entity_ids" not in metadata or "sample_ids" not in metadata:
        raise ValueError("batch.metadata must contain entity_ids and sample_ids")
    entities = _identifiers(metadata["entity_ids"], "metadata.entity_ids", cpu=cpu)
    samples = _identifiers(metadata["sample_ids"], "metadata.sample_ids", cpu=cpu)
    size = len(entities)
    if not size:
        raise ValueError("a fusion batch must contain at least one request")
    if len(samples) != size:
        raise ValueError("metadata.sample_ids must contain one ID per request")
    if check_values:
        sample_values = _identity_values(samples)
        if any(type(value) not in (str, int) or value == "" for value in sample_values):
            raise ValueError("metadata.sample_ids must contain nonempty strings or integer IDs")
        if len(set(sample_values)) != size:
            raise ValueError("metadata.sample_ids must be unique within the batch")
    if "is_padding" in metadata:
        padding = _tensor(metadata["is_padding"], "metadata.is_padding", dtype=torch.bool, cpu=cpu, device=device)
        if padding.shape != (size,):
            raise ValueError("metadata.is_padding must contain one flag per request")
    history = _mapping(inputs[sequence_input], sequence_input)
    ids = _tensor(history["value_ids"], f"{sequence_input}.ids", dtype=torch.long, cpu=cpu, device=device)
    valid = _tensor(history["valid"], f"{sequence_input}.valid", dtype=torch.bool, cpu=cpu, device=device)
    if ids.ndim != 2 or ids.shape[0] != size:
        raise ValueError(f"{sequence_input}.ids must have shape [{size}, L]")
    if valid.shape != ids.shape:
        raise ValueError(f"{sequence_input}.valid must have the same shape as ids")
    if ids.shape[1] > config.sequence.max_seq_len:
        raise ValueError("history length exceeds sequence.max_seq_len")
    if check_values:
        if ((~valid[:, :-1]) & valid[:, 1:]).any().item():
            raise ValueError("history.valid must use right padding without gaps")
        observed = ids[valid]
        if ((observed < 0) | (observed >= config.sequence.vocab_size)).any().item():
            raise ValueError("history contains valid token IDs outside the configured vocabulary")
    aggregates = _mapping(inputs[aggregate_input], aggregate_input)
    for name, width in config.aggregate_fields.items():
        context = f"{aggregate_input}.{name}"
        if name not in aggregates:
            raise ValueError(f"{aggregate_input} is missing configured field {name!r}")
        raw = _mapping(aggregates[name], context)
        if "values" not in raw or "valid" not in raw:
            raise ValueError(f"{context} must contain values and valid")
        values = _tensor(raw["values"], f"{context}.values", cpu=cpu, device=device)
        _tensor(raw["valid"], f"{context}.valid", dtype=torch.bool, cpu=cpu, device=device)
        if values.is_complex():
            raise TypeError(f"{context}.values must be real numerical values")
        values, valid = aggregate_views(raw, width)
        if values.shape != (size, width):
            raise ValueError(f"{context}.values must have shape [{size}, {width}]")
        if valid.shape != values.shape:
            raise ValueError(f"{context}.valid must have shape [{size}], [{size}, 1], or [{size}, {width}]")
        if check_values and not torch.isfinite(values.to(torch.float32)[valid]).all().item():
            raise ValueError(f"{context} contains non-finite valid numerical values")
    return aggregates
