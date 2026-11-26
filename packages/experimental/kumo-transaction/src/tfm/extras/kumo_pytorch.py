# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy access to the pinned experimental native loader, separate from model code."""

from __future__ import annotations

import importlib
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from tfm.contracts.errors import ExtraUnavailableError

SOURCE_REVISION = "5c3a22cacc43e8de99531693e3b6612f0b34d637"


def native_api() -> Any:
    """Require the matching Python API and native runtime; never fall back silently."""
    try:
        api = importlib.import_module("kumo.pytorch.api")
        runtime = importlib.import_module("diskgraph")
    except ImportError as error:
        raise ExtraUnavailableError(
            "The native loader requires kumo-pytorch and DiskGraph from the pinned "
            "Kumo Engine main revision. Run `uv sync --frozen --extra dev --extra native`; see "
            "examples/kumo-engine/README.md for build prerequisites. "
            "Choose data.loader: python explicitly for the reference loader."
        ) from error
    if not callable(getattr(runtime, "pack_token_stream", None)):
        raise ExtraUnavailableError(
            "Incompatible DiskGraph runtime: install the Python API and native wheel "
            "from the same pinned revision. The published kumo-engine 1.0.1 bundle "
            "must not overwrite the native loader's diskgraph package."
        )
    return api


def temporary_engine(stack: ExitStack, *, num_threads: int = 1) -> Any:
    """Own one fresh managed build; close consumers before removing its files.

    Honor the application's process-wide kumo.settings scratch root without
    changing it. Cleanup is restricted to the fresh build ID allocated here.
    """
    engine = native_api().DiskGraphEngine(num_threads=num_threads, num_shards=1)
    index = Path(engine.index)
    build_id = engine.index_build_id
    root = index.parent.parent
    if index != root / "indexes" / build_id or Path(build_id).name != build_id:
        raise ValueError("Unexpected managed native index layout")

    def cleanup() -> None:
        for section in ("indexes", "cache", "tasks"):
            path = root / section / build_id
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("Refusing redirected native cleanup path")
            if path.exists():
                shutil.rmtree(path)

    stack.callback(cleanup)
    return engine
