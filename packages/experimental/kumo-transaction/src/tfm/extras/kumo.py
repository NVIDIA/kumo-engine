# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct Kumo Engine neural components, imported only when graph models are built."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from torch import nn

from tfm.contracts.errors import ExtraUnavailableError


def hetero_gnn(**kwargs: Any) -> nn.Module:
    """Instantiate the installed Engine class. Never substitute a local GNN."""
    try:
        module = importlib.import_module("kumo_core.modeling.gnn.hetero")
    except ImportError as error:
        raise ExtraUnavailableError(
            "GNNEncoder requires Kumo Engine's HeteroGNN and its PyG dependencies. "
            "Run `uv sync --extra dev --extra kumo` (pinned kumo-core source). "
            f"Import failed: {error}"
        ) from error
    return module.HeteroGNN(**kwargs)


def engine_version() -> str:
    """Installed distribution identity for checkpoint/benchmark provenance."""
    try:
        return "kumo-core:" + version("kumo-core")
    except PackageNotFoundError:
        return version("kumo-engine")
