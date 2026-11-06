# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training objectives that attach to encoded model state."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from kumo.pytorch.data.typed_vectors import AttributeSchema, TypedVectorBatch
from kumo.pytorch.models.protocols import LabeledBatch, ModelState, ObjectiveOutput, model_inputs


def _global_mean(loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor | None:
    """Return a global mean whose gradient remains correct after DDP averaging.

    DDP averages rank gradients. Scaling each local sum by ``world_size / total_count``
    therefore produces the same gradient as one process averaging every target.
    The detached correction makes the reported value the actual global mean too.
    """

    total_count = count.detach().to(device=loss_sum.device, dtype=loss_sum.dtype)
    total_sum = loss_sum.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
    if not bool(total_count > 0):
        return None
    backward_value = loss_sum * world_size / total_count
    reported_value = total_sum / total_count
    return backward_value + (reported_value - backward_value.detach())


class MaskedValueObjective(nn.Module):
    """Recover field values from the context retained at masked positions."""

    def __init__(self, embedding: nn.Embedding, *, label_smoothing: float = 0.0):
        super().__init__()
        self.embedding = embedding
        self.label_smoothing = label_smoothing

    def forward(self, state: ModelState, batch: object) -> ObjectiveOutput:
        embeddings = state.auxiliary["masked_value_embeddings"]
        targets = state.auxiliary["masked_value_targets"]
        logits = F.linear(embeddings, self.embedding.weight)
        loss = _global_mean(
            F.cross_entropy(
                logits,
                targets,
                label_smoothing=self.label_smoothing,
                reduction="sum",
            ),
            torch.tensor(targets.numel(), device=targets.device),
        )
        if loss is None:
            raise ValueError("masked-value pretraining requires at least one masked field")
        return ObjectiveOutput(
            loss,
            {"masked_value_logits": logits, "masked_value_targets": targets},
            {},
        )


class NextTokenObjective(nn.Module):
    """Predict the next ID in a causal token sequence."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, state: ModelState, batch: object) -> ObjectiveOutput:
        inputs = model_inputs(batch)
        hidden = state.contextual[state.mask]
        targets = inputs.target_ids[state.mask]
        logits = F.linear(hidden, self.embedding.weight)
        loss = _global_mean(
            F.cross_entropy(logits, targets, reduction="sum"),
            torch.tensor(targets.numel(), device=targets.device),
        )
        if loss is None:
            raise ValueError("next-token pretraining requires at least one target")
        return ObjectiveOutput(
            loss,
            {"next_token_logits": logits, "next_token_targets": targets},
            {},
        )


class ClassificationObjective(nn.Module):
    """Classify the named entity representation from a labeled batch."""

    def __init__(self, input_width: int, classes: int = 2):
        super().__init__()
        if classes < 2:
            raise ValueError("classification requires at least two classes")
        self.projection = nn.Linear(input_width, classes)

    def forward(self, state: ModelState, batch: object) -> ObjectiveOutput:
        if not isinstance(batch, LabeledBatch):
            raise TypeError("classification requires LabeledBatch")
        logits = self.projection(state.representations["entity"])
        labels = batch.labels.to(torch.long)
        loss = _global_mean(
            F.cross_entropy(logits, labels, reduction="sum"),
            torch.tensor(labels.numel(), device=labels.device),
        )
        if loss is None:  # pragma: no cover - a labeled batch cannot be empty
            raise ValueError("classification requires at least one label")
        return ObjectiveOutput(loss, {"class_logits": logits}, {})


class ContrastiveObjective(nn.Module):
    """Bring in-batch entities from the same declared group closer together."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, state: ModelState, batch: object) -> ObjectiveOutput:
        inputs = model_inputs(batch)
        groups = inputs.contrastive_groups
        embeddings = state.representations["entity"]
        scores = -torch.cdist(embeddings, embeddings, p=2) / self.temperature
        diagonal = torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
        positives = (groups[:, None] == groups[None, :]) & (groups[:, None] >= 0) & ~diagonal
        eligible = positives.any(dim=1)
        if not bool(eligible.any()):
            loss_sum = scores.sum() * 0.0
        else:
            denominator = scores.masked_fill(diagonal, float("-inf")).logsumexp(dim=1)
            log_probabilities = scores - denominator[:, None]
            loss_sum = -log_probabilities[positives].sum()
        loss = _global_mean(loss_sum, positives.sum())
        if loss is None:
            loss = loss_sum * 0.0
        return ObjectiveOutput(loss, {"contrastive_scores": scores}, {})


class TypedAttributeObjective(nn.Module):
    """Predict current signal fields and every field of the next event.

    Numerical fields use normal negative log-likelihood in log space. Small
    categorical fields use full cross entropy. Large categorical fields reuse
    one negative set across the complete batch and sequence, avoiding a full
    vocabulary-sized logit tensor at every position.
    """

    def __init__(
        self,
        input_width: int,
        dynamic_schema: AttributeSchema,
        signal_schema: AttributeSchema,
        dynamic_embeddings: nn.ModuleList,
        *,
        negative_samples: int = 64,
        anchor_loss: str = "current.synthetic_abnormal",
        signal_embedding_width: int = 32,
    ):
        super().__init__()
        if negative_samples <= 0:
            raise ValueError("negative_samples must be positive")
        if signal_embedding_width <= 0:
            raise ValueError("signal embedding width must be positive")
        self.dynamic_schema = dynamic_schema
        self.signal_schema = signal_schema
        self.dynamic_embeddings = dynamic_embeddings
        self.signal_embeddings = nn.ModuleList(
            nn.Embedding(signal_schema.cardinality(name), signal_embedding_width)
            for name in signal_schema.categorical
        )
        self.next_numerical = nn.ModuleList(
            nn.Linear(input_width, 2) for _ in dynamic_schema.numerical
        )
        self.current_numerical = nn.ModuleList(
            nn.Linear(input_width, 2) for _ in signal_schema.numerical
        )
        self.next_categorical = nn.ModuleList(
            nn.Linear(input_width, embedding.embedding_dim) for embedding in dynamic_embeddings
        )
        self.current_categorical = nn.ModuleList(
            nn.Linear(input_width, embedding.embedding_dim) for embedding in self.signal_embeddings
        )
        self.negative_samples = negative_samples
        self.anchor_loss = anchor_loss

    @staticmethod
    def _numerical_loss(
        hidden: torch.Tensor,
        targets: torch.Tensor,
        head: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = head(hidden)
        mean = parameters[:, 0]
        scale = F.softplus(parameters[:, 1]) + 1e-4
        log_targets = torch.log1p(targets.clamp_min(0))
        loss = (
            (log_targets - mean).square() / (2.0 * scale.square())
            + scale.log()
            + 0.5 * math.log(2.0 * math.pi)
        ).sum()
        return loss, torch.stack((mean, scale), dim=-1)

    def _categorical_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        head: nn.Linear,
        embedding: nn.Embedding,
        *,
        high_cardinality: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = head(hidden)
        if not high_cardinality:
            logits = projected @ embedding.weight.T
            return F.cross_entropy(logits, targets, reduction="sum"), logits

        negative_ids = torch.randint(
            embedding.num_embeddings,
            (min(self.negative_samples, embedding.num_embeddings),),
            device=targets.device,
        )
        positive = (projected * embedding(targets)).sum(dim=-1, keepdim=True)
        negatives = projected @ embedding(negative_ids).T
        logits = torch.cat((positive, negatives), dim=-1)
        return F.cross_entropy(
            logits,
            torch.zeros_like(targets),
            reduction="sum",
        ), logits

    def _field_losses(
        self,
        prefix: str,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        numerical_targets: torch.Tensor,
        numerical_present: torch.Tensor,
        categorical_targets: torch.Tensor,
        categorical_present: torch.Tensor,
        schema: AttributeSchema,
        numerical_heads: nn.ModuleList,
        categorical_heads: nn.ModuleList,
        embeddings: nn.ModuleList,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        predictions: dict[str, torch.Tensor] = {}
        for index, (name, head) in enumerate(zip(schema.numerical, numerical_heads, strict=True)):
            field_mask = mask & numerical_present[..., index]
            loss_sum, parameters = self._numerical_loss(
                hidden[field_mask],
                numerical_targets[..., index][field_mask],
                head,
            )
            loss = _global_mean(loss_sum, field_mask.sum())
            if loss is None:
                continue
            losses[f"{prefix}.{name}"] = loss
            predictions[f"{prefix}.{name}.distribution"] = parameters
        for index, (name, head, embedding) in enumerate(
            zip(schema.categorical, categorical_heads, embeddings, strict=True)
        ):
            field_mask = mask & categorical_present[..., index]
            targets = categorical_targets[..., index][field_mask]
            loss_sum, logits = self._categorical_loss(
                hidden[field_mask],
                targets,
                head,
                embedding,
                high_cardinality=name in schema.high_cardinality,
            )
            loss = _global_mean(loss_sum, field_mask.sum())
            if loss is None:
                continue
            losses[f"{prefix}.{name}"] = loss
            predictions[f"{prefix}.{name}.logits"] = logits
            predictions[f"{prefix}.{name}.targets"] = targets
        return losses, predictions

    def _aggregate(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        if not losses:
            raise ValueError("typed prediction produced no supervised fields")
        anchor_name = self.anchor_loss if self.anchor_loss in losses else next(iter(losses))
        anchor = losses[anchor_name]
        auxiliaries = [loss for name, loss in losses.items() if name != anchor_name]
        if not auxiliaries:
            return anchor
        scaled = [
            torch.minimum(
                loss,
                loss * anchor.detach() / loss.detach().clamp_min(1e-12),
            )
            for loss in auxiliaries
        ]
        return anchor + torch.stack(scaled).mean()

    def forward(self, state: ModelState, batch: object) -> ObjectiveOutput:
        inputs = model_inputs(batch)
        if not isinstance(inputs, TypedVectorBatch):
            raise TypeError("typed attribute prediction requires TypedVectorBatch")
        current_losses, current_predictions = self._field_losses(
            "current",
            state.contextual,
            inputs.event_mask,
            inputs.signal_numerical,
            inputs.signal_numerical_present,
            inputs.signal_categorical,
            inputs.signal_categorical_present,
            self.signal_schema,
            self.current_numerical,
            self.current_categorical,
            self.signal_embeddings,
        )
        next_mask = inputs.event_mask[:, 1:] & inputs.event_mask[:, :-1]
        next_losses, next_predictions = self._field_losses(
            "next",
            state.contextual[:, :-1],
            next_mask,
            inputs.dynamic_numerical[:, 1:],
            inputs.dynamic_numerical_present[:, 1:],
            inputs.dynamic_categorical[:, 1:],
            inputs.dynamic_categorical_present[:, 1:],
            self.dynamic_schema,
            self.next_numerical,
            self.next_categorical,
            self.dynamic_embeddings,
        )
        components = {**current_losses, **next_losses}
        return ObjectiveOutput(
            self._aggregate(components),
            {**current_predictions, **next_predictions},
            components,
        )
