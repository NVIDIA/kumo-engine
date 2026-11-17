"""Concatenate three request-aligned embeddings and predict the task target."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..graphsage import GraphSAGEModel
from ..nuformer import NuFormerModel
from . import _kumo
from .adapter import FusionBatch
from .config import FusionConfig


@dataclass
class FusionOutput:
    logits: Tensor
    graph_repr: Tensor
    sequence_repr: Tensor
    aggregate_repr: Tensor
    fused_repr: Tensor


class FusionModel(nn.Module):
    """Untrained reference fusion architecture, with reusable branch encoders.

    Validate one CPU Kumo batch, move it with .to(device), and use forward_kumo.
    The legacy forward(FusionBatch) interface remains available.
    The task owns label validity, loss selection and sampling specifications.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.graph_encoder = GraphSAGEModel(config.graph, encoder_only=True)
        self.sequence_encoder = NuFormerModel(config.sequence)
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(2 * config.aggregate_width, config.aggregate_dim),
            nn.GELU(),
            nn.Linear(config.aggregate_dim, config.aggregate_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.graph.channels + config.sequence.dim + config.aggregate_dim, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.fusion_dim, config.out_channels)

    def validate_kumo_batch(self, batch, *, graph_input="graph", sequence_input="history",
                           aggregate_input="aggregates") -> None:
        """Validate aligned CPU inputs without modifying or replacing the batch.

        Targets remain the task's responsibility. Fusion keeps the legacy
        consumer's identity and target rules independently of token pretraining.
        """
        _kumo.inspect(batch, self.config, graph_input, sequence_input, aggregate_input,
                     check_values=True, cpu=True)
        self.graph_encoder.validate_kumo_batch(batch, input_name=graph_input)

    def forward_kumo(self, batch, *, graph_input="graph", sequence_input="history",
                    aggregate_input="aggregates", validate_inputs=True) -> FusionOutput:
        """Consume native graph, history and aggregate fields from one Kumo batch.

        After CPU validation and batch transfer, validate_inputs=False avoids
        value checks and device readbacks. Structure and device checks remain.
        All feature shaping, masking and concatenation happen inside the model.
        """
        aggregates = _kumo.inspect(
            batch, self.config, graph_input, sequence_input, aggregate_input,
            check_values=validate_inputs, device=self.head.weight.device,
        )
        graph = self.graph_encoder.encode_kumo(
            batch, input_name=graph_input, validate_inputs=validate_inputs,
        ).seed_embeddings
        # Fusion validates alignment itself, preserving its original broader
        # entity-ID contract and independence from sequence target aliases.
        sequence = self.sequence_encoder.encode_kumo(
            batch, input_name=sequence_input, validate_inputs=False,
        ).user_repr
        values, validity = [], []
        for name, width in self.config.aggregate_fields.items():
            value, valid = _kumo.aggregate_views(aggregates[name], width)
            values.append(torch.where(valid, value.to(torch.float32), 0.0))
            validity.append(valid)
        value = torch.cat(values, dim=-1).to(self.head.weight.dtype)
        valid = torch.cat(validity, dim=-1).to(value.dtype)
        aggregate = self.aggregate_encoder(torch.cat((value, valid), dim=-1))
        fused = self.fusion(torch.cat((graph, sequence, aggregate), dim=-1))
        return FusionOutput(self.head(fused), graph, sequence, aggregate, fused)

    def forward(self, batch: FusionBatch) -> FusionOutput:
        if not isinstance(batch, FusionBatch):
            raise TypeError("Pass KumoFusionAdapter(config)(kumo_batch), followed by .to(device)")
        size = batch.graph.seed_index.numel()
        if size < 1 or len(batch.sample_ids) != size:
            raise ValueError("Graph roots and sample_ids must identify the same nonempty batch")
        if batch.input_ids.ndim != 2 or batch.input_ids.shape[0] != size:
            raise ValueError("History rows must match graph request rows")
        if batch.aggregate_values.shape != (size, self.config.aggregate_width):
            raise ValueError("Aggregate shape does not match requests and configured fields")
        if batch.aggregate_valid.dtype != torch.bool or batch.aggregate_valid.shape != batch.aggregate_values.shape:
            raise ValueError("Aggregate validity must be boolean and match aggregate values")
        if not batch.aggregate_values.is_floating_point():
            raise TypeError("Aggregate values must be floating point")
        if batch.request_valid.shape != (size,) or batch.request_valid.dtype != torch.bool:
            raise ValueError("request_valid must be boolean with one entry per request")
        device = self.head.weight.device
        if any(value.device != device for value in (
            batch.input_ids, batch.attention_mask, batch.aggregate_values,
            batch.aggregate_valid, batch.request_valid, batch.graph.seed_index,
        )):
            raise ValueError("Move the FusionBatch and model to the same device")

        graph = self.graph_encoder.encode(batch.graph).seed_embeddings
        # The CPU adapter has checked token IDs, padding and sequence length.
        # Skipping value checks here avoids accelerator-host synchronization.
        sequence = self.sequence_encoder.encode(
            batch.input_ids, batch.attention_mask, validate_inputs=False,
        ).user_repr
        values = torch.where(batch.aggregate_valid, batch.aggregate_values, 0).to(self.head.weight.dtype)
        aggregates = self.aggregate_encoder(torch.cat((values, batch.aggregate_valid.to(values.dtype)), dim=-1))
        fused = self.fusion(torch.cat((graph, sequence, aggregates), dim=-1))
        return FusionOutput(self.head(fused), graph, sequence, aggregates, fused)
