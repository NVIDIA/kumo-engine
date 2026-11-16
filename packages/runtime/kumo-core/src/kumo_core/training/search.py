# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML search controls for portable training workflows."""

from __future__ import annotations

from dataclasses import dataclass

from kumoapi.model_plan import RunMode


@dataclass(frozen=True)
class AutoMLSearchOptions:
    """Explicit controls for the original Kumo AutoML search breadth."""

    run_mode: RunMode | str = RunMode.NORMAL
    num_trials: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        mode = (self.run_mode if isinstance(self.run_mode, RunMode) else
                RunMode(str(self.run_mode).lower()))
        object.__setattr__(self, "run_mode", mode)
        invalid_trials = self.num_trials is not None and (
            isinstance(self.num_trials, bool)
            or not isinstance(self.num_trials, int) or self.num_trials <= 0)
        if invalid_trials:
            raise ValueError("num_trials must be a positive integer")
        invalid_seed = self.seed is not None and (isinstance(
            self.seed, bool) or not isinstance(self.seed, int))
        if invalid_seed:
            raise TypeError("seed must be an integer or None")


__all__ = ["AutoMLSearchOptions"]
