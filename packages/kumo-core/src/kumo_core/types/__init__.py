# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public enum surface for Kumo value types.

Conversion, edge, and schema helpers live in
:mod:`kumo_core.types.conversions`, :mod:`kumo_core.types.edges`, and
:mod:`kumo_core.types.schema`.

``Dtype`` and ``Stype`` are deliberately absent: they are wire contract and are
defined once, in :mod:`kumoapi.typing`. Import them from there. The conversions
that need torch or pyarrow remain in :mod:`kumo_core.types.conversions`, which
is why this package can stay out of a client install.
"""

from typing import TYPE_CHECKING

from .enums import (
    AggregationType,
    ColStatType,
    ConnectorType,
    EdgeStatType,
    ExplainMethod,
    PredictionArtifactType,
    PredTime,
    TableStatType,
    TargetType,
    TaskType,
    TaskTypeWithReason,
    TrainModuleOutput,
    TrainTableType,
    persisted_type_token,
)

if TYPE_CHECKING:
    from .features import FeatureStoreData, TensorData, TimeUnit

__all__ = [
    "AggregationType",
    "ColStatType",
    "ConnectorType",
    "EdgeStatType",
    "ExplainMethod",
    "FeatureStoreData",
    "PredTime",
    "PredictionArtifactType",
    "TableStatType",
    "TargetType",
    "TaskType",
    "TaskTypeWithReason",
    "TensorData",
    "TimeUnit",
    "persisted_type_token",
    "TrainModuleOutput",
    "TrainTableType",
]


def __getattr__(name: str):
    if name in {"FeatureStoreData", "TensorData", "TimeUnit"}:
        from . import features

        return getattr(features, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
