"""Adapt aligned graph, token history, and aggregate inputs from one Kumo batch.

Kumo owns sampling, chronological ordering, feature fitting, and temporal
cutoffs. This module only validates and arranges its already encoded outputs;
it has no dependency on an Kumo installation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..graphsage.adapter import GraphBatch, KumoGraphAdapter
from .config import FusionConfig


@dataclass(frozen=True)
class FusionBatch:
    """Model tensors and request identity in the emitted Kumo batch order.

    ``request_valid`` excludes distributed sampler padding from supervised
    loss. Such rows remain in all three branches so their alignment is kept.
    Numerical values are zero wherever ``aggregate_valid`` is false, and
    masked token positions contain the configured padding token ID.
    """

    graph: GraphBatch
    input_ids: Tensor
    attention_mask: Tensor
    aggregate_values: Tensor
    aggregate_valid: Tensor
    sample_ids: tuple[str | int, ...]
    request_valid: Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> FusionBatch:
        """Move every model tensor together while preserving request IDs."""

        def move(value: Tensor) -> Tensor:
            return value.to(device=device, non_blocking=non_blocking)

        return FusionBatch(
            graph=self.graph.to(device, non_blocking=non_blocking),
            input_ids=move(self.input_ids),
            attention_mask=move(self.attention_mask),
            aggregate_values=move(self.aggregate_values),
            aggregate_valid=move(self.aggregate_valid),
            sample_ids=self.sample_ids,
            request_valid=move(self.request_valid),
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
            f"{name} must be on CPU; adapt before calling FusionBatch.to(device)"
        )
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    return value


def _identifiers(value: Any, name: str) -> tuple:
    if isinstance(value, Tensor):
        value = _cpu_tensor(value, name, torch.long)
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return tuple(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise TypeError(f"{name} must be a one-dimensional int64 tensor or sequence of IDs")


def _sample_ids(value: Any, batch_size: int) -> tuple[str | int, ...]:
    identifiers = _identifiers(value, "metadata.sample_ids")
    if len(identifiers) != batch_size:
        raise ValueError("metadata.sample_ids must contain one ID per request")
    if any(
        type(identifier) not in (str, int) or identifier == ""
        for identifier in identifiers
    ):
        raise ValueError("metadata.sample_ids must contain nonempty strings or integer IDs")
    if len(set(identifiers)) != batch_size:
        raise ValueError("metadata.sample_ids must be unique within the batch")
    return identifiers


def _aggregate_field(raw: Any, batch_size: int, width: int, name: str) -> tuple[Tensor, Tensor]:
    field = _mapping(raw, name)
    if "values" not in field or "valid" not in field:
        raise ValueError(f"{name} must contain values and valid")
    values = _cpu_tensor(field["values"], f"{name}.values")
    valid = _cpu_tensor(field["valid"], f"{name}.valid", torch.bool)
    if values.ndim == 1 and width == 1:
        values = values.unsqueeze(-1)
    if values.shape != (batch_size, width):
        raise ValueError(f"{name}.values must have shape [{batch_size}, {width}]")
    if values.is_complex():
        raise TypeError(f"{name}.values must be real numerical values")
    if valid.shape == (batch_size,):
        valid = valid.unsqueeze(-1)
    if valid.shape == (batch_size, 1):
        valid = valid.expand(batch_size, width)
    if valid.shape != values.shape:
        raise ValueError(
            f"{name}.valid must have shape [{batch_size}], [{batch_size}, 1], "
            f"or [{batch_size}, {width}]"
        )
    values = values.to(dtype=torch.float32)
    if not torch.isfinite(values[valid]).all().item():
        raise ValueError(f"{name} contains non-finite valid numerical values")
    return torch.where(valid, values, 0.0), valid


class KumoFusionAdapter:
    """Consume three named inputs belonging to a single CPU Kumo ``Batch``.

    Expected inputs are ``Graph``, ``Fields``/``Tokens``/``Flatten`` with
    ``Padded``, and ``Fields`` with ``Separate``. Aggregate columns are concatenated in
    ``config.aggregate_fields`` order. Extra aggregate columns are ignored.

    The shared batch metadata determines request order and prediction times.
    History rows must match the request count; graph roots are resolved by the
    existing graph adapter using both request owner and entity ID. These
    checks cannot detect manually spliced histories for the same entity at
    different times: all three inputs must come from the same Kumo batch.
    Kumo remains responsible for history/aggregate temporal filtering and
    history ordering; neither timestamps nor event order are reconstructed.
    """

    def __init__(
        self,
        config: FusionConfig,
        graph_input: str = "neighborhood",
        sequence_input: str = "history",
        aggregate_input: str = "aggregates",
    ) -> None:
        if not isinstance(config, FusionConfig):
            raise TypeError("config must be a FusionConfig")
        names = (graph_input, sequence_input, aggregate_input)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("input names must be nonempty strings")
        if len(set(names)) != 3:
            raise ValueError("graph, sequence, and aggregate inputs must have distinct names")
        self.config = config
        self.graph_input = graph_input
        self.sequence_input = sequence_input
        self.aggregate_input = aggregate_input
        self.graph_adapter = KumoGraphAdapter(config.graph, graph_input)

    def __call__(self, kumo_batch: Any) -> FusionBatch:
        inputs = _mapping(kumo_batch.inputs, "batch.inputs")
        metadata = _mapping(kumo_batch.metadata, "batch.metadata")
        for name in (self.graph_input, self.sequence_input, self.aggregate_input):
            if name not in inputs:
                raise ValueError(f"batch.inputs is missing input {name!r}")
        if "entity_ids" not in metadata or "sample_ids" not in metadata:
            raise ValueError("batch.metadata must contain entity_ids and sample_ids")
        entity_ids = _identifiers(metadata["entity_ids"], "metadata.entity_ids")
        batch_size = len(entity_ids)
        if batch_size == 0:
            raise ValueError("a fusion batch must contain at least one request")
        sample_ids = _sample_ids(metadata["sample_ids"], batch_size)

        if "is_padding" in metadata:
            is_padding = _cpu_tensor(metadata["is_padding"], "metadata.is_padding", torch.bool)
            if is_padding.shape != (batch_size,):
                raise ValueError("metadata.is_padding must contain one flag per request")
            request_valid = ~is_padding
        else:
            request_valid = torch.ones(batch_size, dtype=torch.bool)

        context = f"batch.inputs[{self.sequence_input!r}]"
        history = _mapping(inputs[self.sequence_input], context)
        if not all(key in history for key in ("value_ids", "valid")):
            raise ValueError(f"{context} must contain value_ids and valid")
        input_ids = _cpu_tensor(history["value_ids"], f"{context}.value_ids", torch.long)
        attention_mask = _cpu_tensor(history["valid"], f"{context}.valid", torch.bool)
        if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
            raise ValueError(f"{context}.value_ids must have shape [{batch_size}, L]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(f"{context}.valid must have the same shape as value_ids")
        sequence = self.config.sequence
        if input_ids.shape[1] > sequence.max_seq_len:
            raise ValueError("history length exceeds sequence.max_seq_len")
        if ((~attention_mask[:, :-1]) & attention_mask[:, 1:]).any().item():
            raise ValueError("history.valid must use right padding without gaps")
        if input_ids.shape[1] == 0:
            # Accept empty histories at the tensor boundary. The decoder
            # requires a nonzero sequence dimension; the new column is masked.
            input_ids = torch.full((batch_size, 1), sequence.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
        else:
            input_ids = torch.where(attention_mask, input_ids, sequence.pad_token_id)
        if ((input_ids < 0) | (input_ids >= sequence.vocab_size)).any().item():
            raise ValueError("history contains valid token IDs outside the configured vocabulary")

        context = f"batch.inputs[{self.aggregate_input!r}]"
        aggregates = _mapping(inputs[self.aggregate_input], context)
        values, validities = [], []
        for name, width in self.config.aggregate_fields.items():
            if name not in aggregates:
                raise ValueError(f"{context} is missing configured field {name!r}")
            value, valid = _aggregate_field(
                aggregates[name], batch_size, width, f"{context}.{name}"
            )
            values.append(value)
            validities.append(valid)

        return FusionBatch(
            graph=self.graph_adapter(kumo_batch),
            input_ids=input_ids,
            attention_mask=attention_mask,
            aggregate_values=torch.cat(values, dim=1),
            aggregate_valid=torch.cat(validities, dim=1),
            sample_ids=sample_ids,
            request_valid=request_valid,
        )
