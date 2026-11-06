# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build model encoders and objectives from an already fitted data task."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from torch import nn

from kumo.pytorch.config import ModelConfig
from kumo.pytorch.contract.batch import Raw, ZScore
from kumo.pytorch.data.typed_vectors import AttributeSchema
from kumo.pytorch.models.causal import CausalEncoderConfig, CausalTokenEncoder
from kumo.pytorch.models.fusion import JointFusionEncoder, add_transformer_lora
from kumo.pytorch.models.hierarchical import HierarchicalFieldEncoder
from kumo.pytorch.models.model import MultiObjectiveModel
from kumo.pytorch.models.patterns import PatternMemoryEncoder
from kumo.pytorch.models.typed import TypedVectorEncoder
from kumo.pytorch.objectives import (
    ClassificationObjective,
    ContrastiveObjective,
    MaskedValueObjective,
    NextTokenObjective,
    TypedAttributeObjective,
)
from kumo.pytorch.task import Task


class EncoderConfig(Protocol):
    def build(self, task: Task, bindings: Mapping[str, str]) -> nn.Module: ...


class ObjectiveConfig(Protocol):
    def build(self, encoder: nn.Module, task: Task, bindings: Mapping[str, str]) -> nn.Module: ...


def _attribute_schema(
    view: Any, *, high_cardinality_threshold: int | None = None
) -> AttributeSchema:
    """Read ordered model attributes from fitted output bindings, without fitting."""
    numerical, categorical, categories = [], [], {}
    for name, binding in view.bindings.items():
        encoder = binding.encoder
        if isinstance(encoder.config, (Raw, ZScore)):
            if not encoder.dtype.numeric:
                raise ValueError(f"typed numerical input {name!r} requires a numeric dtype")
            numerical.append(name)
        else:
            categorical.append(name)
            size = binding.destination.size if binding.destination else encoder.size
            categories[name] = tuple(range(size))
    high = frozenset(
        name
        for name, values in categories.items()
        if high_cardinality_threshold is not None and len(values) > high_cardinality_threshold
    )
    return AttributeSchema(tuple(numerical), tuple(categorical), categories, high)


@dataclass(frozen=True)
class HierarchicalTransformerConfig:
    """Separate profile, event, and history Transformer stacks."""

    width: int
    ffn_width: int
    attention_heads: int
    profile_layers: int
    event_layers: int
    history_layers: int
    dropout: float
    event_layout: str

    def build(self, task: Task, bindings: Mapping[str, str]) -> nn.Module:
        space = task.spaces[bindings["space"]]
        return HierarchicalFieldEncoder(
            ModelConfig(
                variant="field_hierarchy",
                d_model=self.width,
                d_ffn=self.ffn_width,
                attention_heads=self.attention_heads,
                profile_encoder_layers=self.profile_layers,
                event_encoder_layers=self.event_layers,
                history_encoder_layers=self.history_layers,
                dropout=self.dropout,
                event_encoder_layout=self.event_layout,
            ),
            vocabulary_size=space.size,
            pad_id=space.special(role="pad"),
            entity_token_id=space.special(role="entity"),
            event_token_id=space.special(role="event"),
        )


@dataclass(frozen=True)
class CausalTransformerConfig:
    """Causal Transformer over an explicitly bound token input."""

    width: int
    ffn_width: int
    attention_heads: int
    layers: int
    dropout: float
    profile_conditioned: bool
    profile_mix: float

    def build(self, task: Task, bindings: Mapping[str, str]) -> CausalTokenEncoder:
        space = task.spaces[bindings["space"]]
        profile_size = (
            task.spaces[bindings["profile_space"]].size if self.profile_conditioned else 0
        )
        return CausalTokenEncoder(
            CausalEncoderConfig(
                d_model=self.width,
                d_ffn=self.ffn_width,
                attention_heads=self.attention_heads,
                layers=self.layers,
                dropout=self.dropout,
            ),
            vocabulary_size=space.size,
            pad_id=space.special(role="pad"),
            profile_vocabulary_size=profile_size,
            profile_mix=self.profile_mix,
        )


@dataclass(frozen=True)
class PatternMemoryConfig:
    """Causal token Transformer plus local convolutions and prototype memory."""

    transformer: CausalTransformerConfig
    prototypes: int
    dilations: tuple[int, ...]
    convolution_kernel_size: int
    initial_pattern_mix: float

    def build(self, task: Task, bindings: Mapping[str, str]) -> nn.Module:
        if not self.transformer.profile_conditioned:
            raise ValueError("pattern memory requires a profile-conditioned Transformer")
        return PatternMemoryEncoder(
            self.transformer.build(task, bindings),
            prototypes=self.prototypes,
            dilations=self.dilations,
            convolution_kernel_size=self.convolution_kernel_size,
            initial_pattern_mix=self.initial_pattern_mix,
        )


@dataclass(frozen=True)
class TypedTransformerConfig:
    """Separate typed input modules followed by a causal Transformer."""

    width: int
    ffn_width: int
    attention_heads: int
    layers: int
    dropout: float
    categorical_embedding_width: int

    def build(self, task: Task, bindings: Mapping[str, str]) -> nn.Module:
        return TypedVectorEncoder(
            CausalEncoderConfig(
                d_model=self.width,
                d_ffn=self.ffn_width,
                attention_heads=self.attention_heads,
                layers=self.layers,
                dropout=self.dropout,
            ),
            _attribute_schema(task.output(bindings["entity"])),
            _attribute_schema(task.output(bindings["history"])),
            categorical_embedding_width=self.categorical_embedding_width,
        )


def _output_embedding(encoder: nn.Module) -> nn.Embedding:
    embedding = getattr(encoder, "output_embedding", None)
    if not isinstance(embedding, nn.Embedding):
        raise TypeError("this objective requires an encoder with a token output embedding")
    return embedding


@dataclass(frozen=True)
class MaskedValueConfig:
    label_smoothing: float

    def build(self, encoder: nn.Module, task: Task, bindings: Mapping[str, str]) -> nn.Module:
        return MaskedValueObjective(
            _output_embedding(encoder), label_smoothing=self.label_smoothing
        )


@dataclass(frozen=True)
class NextTokenConfig:
    def build(self, encoder: nn.Module, task: Task, bindings: Mapping[str, str]) -> nn.Module:
        return NextTokenObjective(_output_embedding(encoder))


@dataclass(frozen=True)
class ContrastiveConfig:
    temperature: float

    def build(self, encoder: nn.Module, task: Task, bindings: Mapping[str, str]) -> nn.Module:
        return ContrastiveObjective(temperature=self.temperature)


@dataclass(frozen=True)
class TypedAttributePredictionConfig:
    negative_samples: int
    anchor_loss: str
    signal_embedding_width: int
    high_cardinality_threshold: int = 1_024

    def __post_init__(self) -> None:
        if self.high_cardinality_threshold < 2:
            raise ValueError("high-cardinality threshold must be at least two")

    def build(self, encoder: nn.Module, task: Task, bindings: Mapping[str, str]) -> nn.Module:
        if not isinstance(encoder, TypedVectorEncoder):
            raise TypeError("typed attribute prediction requires typed-vector encoding")
        return TypedAttributeObjective(
            encoder.output_width,
            _attribute_schema(
                task.output(bindings["history"]),
                high_cardinality_threshold=self.high_cardinality_threshold,
            ),
            _attribute_schema(
                task.target(bindings["signals"]),
                high_cardinality_threshold=self.high_cardinality_threshold,
            ),
            encoder.dynamic.categorical,
            negative_samples=self.negative_samples,
            anchor_loss=self.anchor_loss,
            signal_embedding_width=self.signal_embedding_width,
        )


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Name one objective and its contribution to total loss."""

    name: str
    config: ObjectiveConfig
    weight: float


@dataclass(frozen=True)
class ArchitectureConfig:
    """Model parameters only; the data task supplies all fitted encodings."""

    encoder: EncoderConfig
    objectives: tuple[ObjectiveDefinition, ...]


@dataclass
class Architecture:
    """A model and the explicit adapter from its task's public batches."""

    config: ArchitectureConfig
    model: MultiObjectiveModel
    adapt: Callable[[Any], Any]


def build_architecture(
    *,
    task: Task,
    config: ArchitectureConfig,
    bindings: Mapping[str, str],
    adapt: Callable[[Any], Any],
) -> Architecture:
    """Read fitted metadata and build model modules; never read or fit training rows."""
    if not config.objectives:
        raise ValueError("at least one objective is required")
    names = [value.name for value in config.objectives]
    if len(names) != len(set(names)):
        raise ValueError("objective names must be unique")
    if any(value.weight < 0 for value in config.objectives):
        raise ValueError("objective weights cannot be negative")
    encoder = config.encoder.build(task, bindings)
    objectives = {
        definition.name: definition.config.build(encoder, task, bindings)
        for definition in config.objectives
    }
    return Architecture(
        config,
        MultiObjectiveModel(
            encoder, objectives, {value.name: value.weight for value in config.objectives}
        ),
        adapt,
    )


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    dropout: float


@dataclass(frozen=True)
class TabularFusionConfig:
    numerical_means: tuple[float, ...]
    numerical_scales: tuple[float, ...]
    periodic_frequencies: int
    categorical_cardinalities: tuple[int, ...]
    categorical_embedding_width: int
    layers: int
    rank: int


@dataclass(frozen=True)
class ClassificationConfig:
    """Configuration for an entity-classification adaptation."""

    classes: int
    lora: LoraConfig | None
    tabular_fusion: TabularFusionConfig | None


def for_classification(
    architecture: Architecture, config: ClassificationConfig
) -> MultiObjectiveModel:
    """Copy an encoder and attach explicitly configured adaptation layers and loss."""
    encoder = copy.deepcopy(architecture.model.encoder)
    if config.lora is not None:
        add_transformer_lora(encoder, rank=config.lora.rank, dropout=config.lora.dropout)
    if config.tabular_fusion is not None:
        tabular = config.tabular_fusion
        encoder = JointFusionEncoder(
            encoder,
            tabular.numerical_means,
            tabular.numerical_scales,
            tabular.periodic_frequencies,
            tabular.categorical_cardinalities,
            tabular.categorical_embedding_width,
            layers=tabular.layers,
            rank=tabular.rank,
        )
    if not hasattr(encoder, "output_width"):
        raise TypeError("classification requires an encoder with output_width")
    return MultiObjectiveModel(
        encoder,
        {"classification": ClassificationObjective(int(encoder.output_width), config.classes)},
    )
