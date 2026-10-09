# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML trial trainer base shared by native and legacy data paths."""

from kumo_core.train.trainer import Trainer
from kumo_core.training.config import NestedTrainerConfig
from kumo_core.utils.registry import register


@register(data_cls=NestedTrainerConfig)
class NestedTrainer(Trainer):
    """Trainer whose logs are nested directly below an AutoTrainer trial."""
    @property
    def name(self) -> str | None:
        return None
