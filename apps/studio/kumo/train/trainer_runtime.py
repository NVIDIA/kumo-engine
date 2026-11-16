# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monolith environment capabilities for the portable Trainer."""

from typing import Any

from kumo_core.training.runtime import (
    LocalDiskgraphTrainingRuntime,
    TrainingArtifactRuntime,
)

from kumo.explain.runtime import MonolithExplainRuntime
from kumo.home import get_home_dir, get_local_model_dir
from kumo.utils.embedding_models import embedding_model_store


class MonolithTrainingArtifactRuntime(TrainingArtifactRuntime):
    def default_model_root(self) -> str:
        return get_home_dir()

    def local_model_root(self) -> str:
        return get_local_model_dir()


def monolith_trainer_runtime_kwargs() -> dict[str, Any]:
    return {
        "explain_runtime": MonolithExplainRuntime(),
        "artifact_runtime": MonolithTrainingArtifactRuntime(),
        "diskgraph_runtime": LocalDiskgraphTrainingRuntime(),
        "embedding_models": embedding_model_store(),
    }
