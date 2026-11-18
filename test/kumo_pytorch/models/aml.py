# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML model construction and objectives, adapted from Arena's model_execution.

Kumo owns fitting, sampling, corruption and next-token alignment. These consumers
use the emitted batch unchanged; the caller owns backward and optimizer steps.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

AMOUNTS = ("amount_paid", "amount_received")
CATEGORIES = ("payment_currency", "receiving_currency", "payment_format")
AGGREGATES = tuple(
    f"{window}_{role}_count"
    for window in ("all_history", "24h", "7d", "30d")
    for role in ("total_touching", "sender", "receiver")
)


@dataclass
class StepResult:
    logits: torch.Tensor
    loss: torch.Tensor
    weight: int


class AMLModel(nn.Module):
    """One registered backbone and task head; loss excludes invalid/padding rows."""

    def __init__(self, backbone, family, workload, dim):
        super().__init__()
        self.backbone = backbone
        self.family = family
        self.workload = workload
        self.head = None
        if family == "nuformer" and workload == "account-involvement-next24h":
            self.head = nn.Linear(dim, 1)
        elif family == "pragma":
            from .pragma import MLMHead

            self.head = MLMHead(dim)

    def forward(self, batch, *, validate_inputs=True):
        if self.family in {"graphsage", "combined"}:
            records = batch.metadata.get("request_metadata", ())
            if len(records) != len(batch.metadata["sample_ids"]) or any(
                "prediction_time_ns" not in row for row in records
            ):
                raise ValueError(
                    "AML v2 graphs require original prediction_time_ns metadata"
                )
        if self.family == "pragma":
            output = self.backbone.forward_kumo(batch, validate_inputs=validate_inputs)
            if self.workload == "account-involvement-next24h":
                return self.head(output.user_repr)
            indices, _ = self.backbone.select_kumo_targets(batch)
            return self.head(output, self.backbone.embedding_weight, indices)
        if self.head is not None:
            return self.head(
                self.backbone.encode_kumo(
                    batch, validate_inputs=validate_inputs
                ).user_repr
            )
        return self.backbone.forward_kumo(batch, validate_inputs=validate_inputs).logits

    def validate_batch(self, batch):
        """Validate loader outputs on CPU before an optional device transfer."""
        if self.family in {"pragma", "nuformer"}:
            target = {
                "account-involvement-next24h": None,
                "masked-value-pretrain": "masked_values",
                "next-token-pretrain": "next_token",
            }[self.workload]
            self.backbone.validate_kumo_batch(batch, target_name=target)
        else:
            self.backbone.validate_kumo_batch(batch)

    def training_step(self, batch, *, validate_inputs=True):
        """Validate a CPU batch and return a differentiable mean objective.

        A zero-weight batch returns zero loss; the caller must skip its optimizer
        update (including weight decay).
        For next-token training, Kumo already shifted the targets; do not shift
        again through nuFormer's standalone next_token_loss helper.
        """
        target_name = {
            "account-involvement-next24h": "label",
            "masked-value-pretrain": "masked_values",
            "next-token-pretrain": "next_token",
        }[self.workload]
        if validate_inputs:
            self.validate_batch(batch)
        size = len(batch.metadata["sample_ids"])
        padding = batch.metadata.get("is_padding", torch.zeros(size, dtype=torch.bool))
        if padding.dtype != torch.bool or padding.shape != (size,):
            raise ValueError("is_padding must contain one boolean per request")

        if target_name == "masked_values":
            _, labels = self.backbone.select_kumo_targets(batch)
            selected = None  # The MLM head emits only these positions.
        else:
            target = batch.targets[target_name]
            labels, valid = target["values"], target["valid"]
            if valid.dtype != torch.bool:
                raise ValueError("target validity must be boolean")
            if target_name == "label":
                if labels.shape == (size, 1):
                    labels = labels[:, 0]
                if valid.shape == (size, 1):
                    valid = valid[:, 0]
                if labels.shape != (size,) or valid.shape != (size,):
                    raise ValueError("binary targets require one value per request")
                selected = valid & ~padding
                labels = labels[selected].float()
                if (
                    not torch.isfinite(labels).all()
                    or not ((labels == 0) | (labels == 1)).all()
                ):
                    raise ValueError("valid binary labels must be finite zero or one")
            else:
                selected = (
                    valid
                    & batch.inputs["history"]["valid"]
                    & ~padding[:, None]
                    & (labels != -100)
                )
                labels = labels[selected]

        logits = self(batch, validate_inputs=validate_inputs)
        values = (
            logits.flatten()[selected]
            if target_name == "label"
            else logits
            if selected is None
            else logits[selected]
        )
        weight = labels.numel()
        if not weight:
            loss = values.sum() * 0.0
        elif target_name == "label":
            loss = F.binary_cross_entropy_with_logits(values.float(), labels)
        else:
            loss = F.cross_entropy(
                values.float(),
                labels,
                label_smoothing=0.1 if self.family == "pragma" else 0.0,
            )
        return StepResult(logits, loss, weight)


def build_model(task, selection, family, *, dim=16, max_seq_len=2048, seed=0):
    """Build one of the five selected AML workload/model combinations.

    Vocabulary dimensions come from a fitted Kumo task, never evaluation data.
    These small prototype dimensions are not paper-size configurations.
    """
    from .nuformer import NuFormerConfig, NuFormerModel

    selector = selection["selector"]
    workload = selector["workload_id"]
    allowed = {
        "account-involvement-next24h": {"graphsage", "nuformer", "combined"},
        "masked-value-pretrain": {"pragma"},
        "next-token-pretrain": {"nuformer"},
    }
    if (
        not selector["dataset_id"].startswith("aml-data/")
        or family not in allowed.get(workload, ())
        or family not in selection["models"]
    ):
        raise ValueError("unselected AML workload/model combination")
    version = "v2" if family in {"graphsage", "combined"} else "v1"
    if selector["anchor_version"] != version:
        raise ValueError(f"{family} requires account-daily {version} anchors")
    if type(dim) is not int or dim < 4 or dim % 4:
        raise ValueError("dim must be a positive multiple of four")

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if family in {"pragma", "nuformer", "combined"}:
            space = task.spaces["history"]
            if space.roles.get("pad") != 0:
                raise ValueError("the AML models require padding ID zero")
        if family == "pragma":
            from .pragma import ModelConfig, NativePragmaModel

            vocabulary = [f"kumo-id:{i}" for i in range(space.size)]
            for role, name in (
                ("pad", "[PAD]"),
                ("unknown", "[UNK]"),
                ("mask", "[MASK]"),
                ("entity", "[USR]"),
                ("event", "[EVT]"),
            ):
                vocabulary[space.roles[role]] = name
            config = ModelConfig(
                vocab_size=space.size,
                dim=dim,
                n_heads=2,
                depth_profile=1,
                depth_event=1,
                depth_history=1,
                dropout=0,
            )
            backbone = NativePragmaModel(config, vocabulary)
        if family in {"nuformer", "combined"}:
            sequence = NuFormerConfig(
                vocab_size=space.size,
                dim=dim,
                depth=1,
                n_heads=2,
                max_seq_len=max_seq_len,
                dropout=0,
            )
            if family == "nuformer":
                backbone = NuFormerModel(sequence)
        if family in {"graphsage", "combined"}:
            from .graphsage import (
                GraphSAGEConfig,
                GraphSAGEModel,
                NodeFeatureSpec,
            )

            fields = task.output_schema["inputs"]["graph"]["transactions"]
            graph = GraphSAGEConfig(
                node_features={
                    "accounts": NodeFeatureSpec(),
                    "transactions": NodeFeatureSpec(
                        numerical={name: 1 for name in AMOUNTS},
                        categorical={
                            name: fields[name]["vocabulary_size"] for name in CATEGORIES
                        },
                        timestamp="event_time_seconds",
                    ),
                },
                edge_types=(
                    ("accounts", "sender", "transactions"),
                    ("accounts", "receiver", "transactions"),
                ),
                root_type="accounts",
                channels=dim,
                num_layers=2,
            )
            if family == "graphsage":
                backbone = GraphSAGEModel(graph)
            else:
                from .graph_sequence_aggregates import (
                    FusionConfig,
                    FusionModel,
                )

                backbone = FusionModel(
                    FusionConfig(
                        graph,
                        sequence,
                        {name: 1 for name in AGGREGATES},
                        aggregate_dim=dim // 2,
                        fusion_dim=dim,
                    )
                )
        return AMLModel(backbone, family, workload, dim)
