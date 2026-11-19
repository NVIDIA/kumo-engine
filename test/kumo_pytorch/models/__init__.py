# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model construction from declared input profiles and task objectives.

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


class TaskModel(nn.Module):
    """One registered backbone and task head; loss excludes invalid/padding rows."""

    def __init__(self, backbone, family, kind, dim):
        super().__init__()
        self.backbone = backbone
        self.family = family
        self.kind = kind
        self.head = None
        if family == "nuformer" and kind == "binary":
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
                    "Graph models require original prediction_time_ns metadata"
                )
        if self.family == "pragma":
            output = self.backbone.forward_kumo(batch, validate_inputs=validate_inputs)
            if self.kind == "binary":
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
                "binary": None,
                "masked_values": "masked_values",
                "next_token": "next_token",
            }[self.kind]
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
            "binary": "label",
            "masked_values": "masked_values",
            "next_token": "next_token",
        }[self.kind]
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
    """Build a backbone and task head from the selected binding.

    Vocabulary dimensions come from a fitted Kumo task, never evaluation data.
    These small prototype dimensions are not paper-size configurations.
    """
    from ..bindings import aggregate_fields, binding_for, graph_layout
    from ..bindings import profile as input_profile
    from ..bindings import task_definition
    from ..population_anchors import validate_binding
    from .nuformer import NuFormerConfig, NuFormerModel
    from .stored_targets import StoredTargetModel, output_widths

    if family not in selection["models"]:
        raise ValueError("Unselected model representation")
    binding = binding_for(selection, family)
    validate_binding(binding)
    task_spec = task_definition(binding)
    profile = input_profile(binding, "input")
    kind = task_spec["kind"]
    rules = task_spec["targets"]
    if type(dim) is not int or dim < 4 or dim % 4:
        raise ValueError("dim must be a positive multiple of four")
    # Preserve the established scalar-binary consumer's checkpoint layout.
    scalar_binary = (
        kind == "stored"
        and list(rules) == ["label"]
        and rules["label"]["kind"] == "binary"
        and family != "pragma"
    )
    stored = kind == "stored" and not scalar_binary
    widths = output_widths(task, rules) if stored else {}
    channels = sum(widths.values()) if stored else 1
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if family in {"pragma", "nuformer", "combined"}:
            space = task.spaces["history"]
            if space.roles.get("pad") != 0:
                raise ValueError("Sequence models require padding ID zero")
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

            nodes, relations, _ = graph_layout(profile)
            root = profile["root_table"]
            fields = task.output_schema["inputs"]["graph"]
            graph = GraphSAGEConfig(
                node_features={
                    root: NodeFeatureSpec(),
                    **{
                        node: NodeFeatureSpec(
                            numerical={name: 1 for name in graph_profile["numerical"]},
                            categorical={
                                name: fields[node][name]["vocabulary_size"]
                                for name in graph_profile["categorical"]
                            },
                            timestamp=graph_profile.get("timestamp"),
                        )
                        for node, graph_profile in nodes.items()
                    },
                },
                edge_types=tuple(
                    (root, relation, node) for relation, node in relations.items()
                ),
                root_type=root,
                channels=dim,
                num_layers=2,
                out_channels=channels,
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
                        dict.fromkeys(aggregate_fields(profile), 1),
                        aggregate_dim=dim // 2,
                        fusion_dim=dim,
                        out_channels=channels,
                    )
                )
        if stored:
            return StoredTargetModel(
                backbone,
                family,
                rules,
                widths,
                dim,
                check_targets=(
                    *task_spec.get("generated_checks", ()),
                    *task_spec.get("auxiliary_checks", ()),
                ),
            )
        return TaskModel(backbone, family, "binary" if scalar_binary else kind, dim)
