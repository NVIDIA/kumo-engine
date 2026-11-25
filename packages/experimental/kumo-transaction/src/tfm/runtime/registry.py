# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open recipe orchestration selected by the shared configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from tfm.config import RecipeConfig
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.registries import Registry


class RecipeRunner(Protocol):
    def __call__(
        self,
        command: str,
        *,
        config: RecipeConfig,
        config_path: Path | None,
        run_dir: Path | None,
        device: str | None,
        reuse_checkpoint: bool = False,
    ) -> Path: ...


_RUNNERS: Registry[RecipeRunner | str] = Registry("recipe family")
_RUNNERS.register("sequence", "tfm.runtime.sequence:run_command")
_RUNNERS.register("graph", "tfm.runtime.graph:run_command")


def register_recipe_runner(name: str, runner: RecipeRunner) -> RecipeRunner:
    """Register a lifecycle runner with the same config and invocation as built-ins."""
    _RUNNERS.register(name, runner)
    return runner


def recipe_runner(name: str) -> RecipeRunner:
    """Resolve a registered runner or an importable ``module:function``."""
    entry = _RUNNERS.get(name)
    if isinstance(entry, str):
        entry = _RUNNERS.get(entry)
    if not callable(entry):
        raise UnsupportedCombinationError(f"Recipe family {name!r} must name a callable runner")
    return cast(RecipeRunner, entry)
