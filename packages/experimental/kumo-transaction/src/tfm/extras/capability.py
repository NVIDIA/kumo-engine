# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability checks for optional extras (never imported by CPU CI paths)."""

from __future__ import annotations

import importlib

from tfm.contracts.errors import ExtraUnavailableError

_INSTALL_HINTS = {
    "huggingface": "uv sync --extra huggingface",
    "rapids": (
        "install cuDF for your CUDA version outside the lockfile, e.g. "
        "`uv pip install cudf-cu13 'numpy<2.5' --extra-index-url=https://pypi.nvidia.com`"
    ),
    "xgboost": "uv sync --extra xgboost",
    "baselines": "uv sync --extra baselines",
    "umap": "uv sync --extra umap",
}


def extra_available(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def require_extra(name: str, module: str) -> None:
    if extra_available(module):
        return
    hint = _INSTALL_HINTS.get(name, f"install the {name!r} extra")
    raise ExtraUnavailableError(
        f"Optional extra {name!r} is not available (could not import {module!r}). {hint}."
    )
