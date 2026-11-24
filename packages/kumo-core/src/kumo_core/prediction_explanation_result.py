# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable prediction explanation outcomes and observed coverage."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PredictionExplanationOptions:
    """Independent prediction row budgets.

    Omitted cohort rows use ``max(50 * B, 1000)`` rounded up to whole batches,
    where ``B = max(eval_batch_size // 4, 1)``. Omitted sample rows use 1,000.
    Explicit limits select at most that many task rows independently.
    """

    cohort_rows: int | None = None
    sample_rows: int | None = None

    def __post_init__(self) -> None:
        for name, value in (('cohort_rows', self.cohort_rows),
                            ('sample_rows', self.sample_rows)):
            if value is not None:
                if type(value) is not int:
                    raise TypeError(f'{name} must be a positive integer or None')
                if value <= 0:
                    raise ValueError(f'{name} must be positive')


@dataclass(frozen=True)
class PredictionExplanationSupport:
    cohorts: bool
    samples: bool


@dataclass(frozen=True)
class ProducerOutcome:
    """Independent producer completion or an actionable failure."""

    status: Literal['unsupported', 'complete', 'failed']
    error: str | None = None

    @property
    def supported(self) -> bool:
        return self.status != 'unsupported'


@dataclass(frozen=True)
class PredictionExplanationCoverage:
    """Observed work; ``None`` means unresolved and zero means observed empty.

    Cohort rows include failed feature callbacks. Sampled graphs count retained
    graphs named in the saved selection mappings. ``checkpoint_path`` is
    relative to the published model root.
    """

    cohort_rows_processed: int | None = None
    sampled_graphs: int | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class PredictionExplanationResult:
    cohorts: ProducerOutcome
    samples: ProducerOutcome
    coverage: PredictionExplanationCoverage = field(
        default_factory=PredictionExplanationCoverage)

    @property
    def failure(self) -> str | None:
        errors = [outcome.error for outcome in (self.cohorts, self.samples)
                  if outcome.error is not None]
        return '; '.join(errors) or None
