# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete, explicit definitions of the four paper-guided reference models.

These are ordinary configuration values. The model builder does not inspect the
dictionary keys or paper names; changing any component changes what is built.
Where a paper does not publish an exact value, the number below is an explicit
reference choice rather than an implied paper value.
"""

from kumo.pytorch.architectures import (
    ArchitectureConfig,
    CausalTransformerConfig,
    ClassificationConfig,
    ContrastiveConfig,
    HierarchicalTransformerConfig,
    LoraConfig,
    MaskedValueConfig,
    NextTokenConfig,
    ObjectiveDefinition,
    PatternMemoryConfig,
    TabularFusionConfig,
    TypedAttributePredictionConfig,
    TypedTransformerConfig,
)

PRAGMA = ArchitectureConfig(
    encoder=HierarchicalTransformerConfig(
        width=192,
        ffn_width=768,
        attention_heads=3,
        profile_layers=1,
        event_layers=5,
        history_layers=2,
        dropout=0.1,
        event_layout="packed",
    ),
    objectives=(
        ObjectiveDefinition(
            name="masked_value", config=MaskedValueConfig(label_smoothing=0.0), weight=1.0
        ),
    ),
)
NUFORMER = ArchitectureConfig(
    encoder=CausalTransformerConfig(
        width=256,
        ffn_width=1024,
        attention_heads=4,
        layers=4,
        dropout=0.1,
        profile_conditioned=False,
        profile_mix=0.0,
    ),
    objectives=(ObjectiveDefinition(name="next_token", config=NextTokenConfig(), weight=1.0),),
)
PANTHER = ArchitectureConfig(
    encoder=PatternMemoryConfig(
        transformer=CausalTransformerConfig(
            width=256,
            ffn_width=1024,
            attention_heads=4,
            layers=4,
            dropout=0.1,
            profile_conditioned=True,
            profile_mix=0.1,
        ),
        prototypes=64,
        dilations=(1, 2, 4),
        convolution_kernel_size=3,
        initial_pattern_mix=0.1,
    ),
    objectives=(
        ObjectiveDefinition(name="next_behavior", config=NextTokenConfig(), weight=0.9),
        ObjectiveDefinition(
            name="profile_contrastive", config=ContrastiveConfig(temperature=0.1), weight=0.1
        ),
    ),
)
TREASURE = ArchitectureConfig(
    encoder=TypedTransformerConfig(
        width=256,
        ffn_width=1024,
        attention_heads=4,
        layers=4,
        dropout=0.1,
        categorical_embedding_width=32,
    ),
    objectives=(
        ObjectiveDefinition(
            name="typed_attributes",
            config=TypedAttributePredictionConfig(
                negative_samples=64,
                anchor_loss="current.synthetic_abnormal",
                signal_embedding_width=32,
            ),
            weight=1.0,
        ),
    ),
)
ARCHITECTURES = {"pragma": PRAGMA, "nuformer": NUFORMER, "panther": PANTHER, "treasure": TREASURE}
CLASSIFICATION_TASKS = {
    "pragma": ClassificationConfig(
        classes=2, lora=LoraConfig(rank=8, dropout=0.1), tabular_fusion=None
    ),
    "nuformer": ClassificationConfig(
        classes=2,
        lora=LoraConfig(rank=8, dropout=0.1),
        tabular_fusion=TabularFusionConfig(
            numerical_means=(0.0, 0.0, 0.0, 0.0),
            numerical_scales=(1.0, 1.0, 1.0, 1.0),
            periodic_frequencies=8,
            categorical_cardinalities=(32, 32),
            categorical_embedding_width=8,
            layers=2,
            rank=16,
        ),
    ),
    "panther": ClassificationConfig(classes=2, lora=None, tabular_fusion=None),
}

MODEL_BINDINGS = {
    "pragma": {
        "space": "pragma",
        "profile": "entity",
        "history": "history",
        "calendar": "history_calendar",
        "timing": "history_timing",
        "profile_timing": "profile_timing",
        "objective": "masked_values",
    },
    "nuformer": {"space": "nuformer", "history": "history", "objective": "next_token"},
    "panther": {
        "space": "panther",
        "profile_space": "profile",
        "entity": "entity",
        "history": "history",
        "objective": "next_behavior",
    },
    "treasure": {"entity": "entity", "history": "events", "signals": "current_event"},
}
