# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public preparation factory for snapshot-backed FM payloads."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal

from ._snapshot import PreparedRequest, _FMLoader

if TYPE_CHECKING:
    from kumo.pytorch import api

__all__ = ["PreparedRequest", "prepare"]


def prepare(*, engine: api.DiskGraphEngine, sample: api.SampleSpec, target: str,
            task: Literal["regression", "binary_classification"], features: Mapping[str, Iterable[str]],
            classes: dict[object, object] | None = None, batch_size: int = 1024,
            budgets: Mapping[str, int] | None = None, schema: api.DataSchema | None = None) -> _FMLoader:
    """Prepare FM over one adopted engine; budgets use the documented positive integer limits."""
    return _FMLoader(engine=engine, sample=sample, target=target, task=task,
                     features=features, classes=classes, batch_size=batch_size,
                     budgets=budgets, schema=schema)
