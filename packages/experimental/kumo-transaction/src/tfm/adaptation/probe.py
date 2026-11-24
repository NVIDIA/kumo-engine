# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen probe arms: raw features, embeddings, or both (early fusion).

Every arm fits a logistic-regression or XGBoost probe on the same rows and
labels, so the arms differ only in their inputs:

- `raw`: the leakage-safe aggregates in `numeric_features`, the no-model baseline.
- `embedding`: the frozen embedding alone.
- `combined`: both concatenated. This is early fusion, the usual first way a
  fraud team adds embeddings to an existing model.
"""

from __future__ import annotations

from typing import Literal

ProbeArm = Literal["raw", "embedding", "combined"]
ALL_ARMS: tuple[ProbeArm, ...] = ("raw", "embedding", "combined")


def probe_arms(*, use_raw_features: bool, use_tfm_embeddings: bool) -> tuple[ProbeArm, ...]:
    """The arms `fraud_task.use_raw_features` and `use_tfm_embeddings` select."""
    arms: list[ProbeArm] = []
    if use_raw_features:
        arms.append("raw")
    if use_tfm_embeddings:
        arms.append("embedding")
    if use_raw_features and use_tfm_embeddings:
        arms.append("combined")
    if not arms:
        raise ValueError("At least one of raw features or embeddings must be used.")
    return tuple(arms)
