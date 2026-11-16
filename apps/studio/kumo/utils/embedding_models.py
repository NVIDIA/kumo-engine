# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The embedding models baked into this deployment."""

import os
from functools import lru_cache
from pathlib import Path

from kumo_core.embedding_models import (
    RELEASES,
    EmbeddingModelStore,
    GloVeVectors,
    default_name,
)
from kumo_core.model_plan import (
    ColumnProcessingPlan,
    ModelPlan,
    ModelPlanValueError,
)
from kumoapi.encoder import TextEmbedding


@lru_cache(maxsize=None)
def embedding_model_store() -> EmbeddingModelStore:
    r"""Registers every model in ``KUMO_GLOVE_PATH`` under its default name;
    installed packages supply any other model."""
    store = EmbeddingModelStore()
    store._missing_advice = (
        "Use one of those, or bake the model into the image "
        "(BAKE_GLOVE_LEVEL=all).")
    path = os.environ.get("KUMO_GLOVE_PATH")
    if path and Path(path).is_dir():
        for model in GloVeVectors.in_directory(path):
            store.add(default_name(model), model)
    return store


def hosted_model_names() -> list[str]:
    r"""The models a hosted run can use: the official GloVe releases. The
    trainer images, not this process, decide which of them are baked."""
    return sorted(f"{release.model}-{dim}d" for release in RELEASES.values()
                  for dim in release.members)


def validate_text_embedding_models(plan: ModelPlan) -> None:
    overrides = ColumnProcessingPlan.get_encoder_overrides(
        plan.column_processing.encoder_overrides)
    available = hosted_model_names()
    unknown = [
        f"'{column}' uses {encoder.model!r}"
        for column, encoder in overrides.items()
        if isinstance(encoder, TextEmbedding) and encoder.model not in available
    ]
    if unknown:
        raise ModelPlanValueError([
            ValueError(f"Unknown text embedding model: {'; '.join(unknown)}. "
                       f"Available models: {', '.join(available)}.")
        ])
