# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact references and availability, separate from explanation schemas."""

from dataclasses import dataclass, field
from typing import Literal

CohortKey = tuple[str, int, str]
UnavailableReason = Literal['disabled', 'failed', 'model_only', 'unknown']


@dataclass(frozen=True)
class ArtifactAvailability:
    """File presence, without parsing artifacts or implying valid contents.

    ``reason`` applies when unavailable. A failed generation can leave usable
    partial outputs; ``generation_failure`` preserves that fact independently.
    """

    available: bool
    path: str | None
    reason: UnavailableReason | None = None
    generation_failure: str | None = None


@dataclass(frozen=True)
class SampleReference:
    """One collector-local artifact ID, with optional selection metadata.

    ``original_ids`` maps selection categories to original entity IDs, when
    recorded. These are separate from the collector-local ``artifact_id``.
    A mapping may refer to a missing graph; check ``available`` before loading.
    """

    artifact_id: str
    path: str
    available: bool
    categories: tuple[str, ...] = ()
    original_ids: dict[str, str | int] = field(default_factory=dict)
