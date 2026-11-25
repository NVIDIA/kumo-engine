# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit batch capabilities consumed by training backends."""

from __future__ import annotations

from typing import Protocol

import torch


class TrainableBatch(Protocol):
    """Training transport and supervision, independent of model architecture.

    Subclassing supplies ordinary full transfer and supervised-step defaults.
    Structural implementations declare both capabilities explicitly. A false
    ``has_supervision`` suppresses optimizer and training-buffer updates, while
    preserving a differentiable forward/backward and the step record.
    """

    def to(self, device: torch.device) -> TrainableBatch: ...

    def for_execution(self, device: torch.device) -> TrainableBatch:
        """Transfer model inputs; override to retain CPU validation metadata."""
        return self.to(device)

    @property
    def has_supervision(self) -> bool:
        """Host metadata, established while building the batch; never a device read."""
        return True
