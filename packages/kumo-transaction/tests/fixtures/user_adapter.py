# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An adapter defined outside `tfm`, selected by import path."""

from __future__ import annotations

from tfm.config import DataConfig
from tfm.data.adapters import Adapter, resolve_data_path
from tfm.data.adapters.pragmatiq_columns import PragmatiqColumns
from tfm.data.adapters.registry import load_pragmatiq_registered, load_synthetic_adapter


def columns(config: DataConfig) -> PragmatiqColumns:
    return PragmatiqColumns(resolve_data_path(config))


BOOK = Adapter(load=load_synthetic_adapter, columnar_source=columns)

PRAGMATIQ_BOOK = Adapter(load=load_pragmatiq_registered, columnar_source=columns)
"""pragmatiq's loader under an import path: a user adapter to every check the SDK makes."""
