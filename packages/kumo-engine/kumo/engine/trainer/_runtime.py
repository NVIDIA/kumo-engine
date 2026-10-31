# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy adapter to core-owned local training capabilities."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo_core.training.config import SamplerKwargs

__all__ = ["local_training_bindings"]


def local_training_bindings(
    training_input: Any,
    run_name: str | None = None,
    embedding_models: Any = None,
    accelerator: str = "auto",
    sampling: "SamplerKwargs | None" = None,
) -> Any:
    from kumo_core.training.runtime import local_training_bindings as bind

    return bind(
        training_input,
        run_name=run_name,
        embedding_models=embedding_models,
        accelerator=accelerator,
        sampling=sampling,
    )
