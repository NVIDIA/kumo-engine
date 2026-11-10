# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide settings shared by the Kumo packages.

The scratch directory is where DiskGraph ingest, training and prediction read
and write their working data. It resolves, first match wins, from a
`scoped(scratch_dir=...)` block, then `configure(scratch_dir=...)`, then the
`KUMO_SCRATCH_DIR` environment variable, then `~/.cache/kumo-engine`.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional, Union

SCRATCH_DIR_ENV = "KUMO_SCRATCH_DIR"

_DEFAULT_SCRATCH_DIR = Path.home() / ".cache" / "kumo-engine"
_configured: Optional[Path] = None
_scoped: ContextVar[Optional[Path]] = ContextVar("kumo_scratch_dir", default=None)


def _local_path(path: Union[str, "os.PathLike[str]"], source: str = "scratch_dir") -> Path:
    text = os.fspath(path)
    if text.startswith("@") or "://" in text:
        raise ValueError(
            f"{source} must be a local filesystem path, not {text!r} -- "
            "DiskGraph ingest and training read and write it directly.")
    return Path(text).expanduser()


def configure(*, scratch_dir: Optional[Union[str, "os.PathLike[str]"]] = None) -> None:
    """Sets process-wide settings. `None` leaves a setting unchanged.

    Call it once, early: paths already derived from the scratch directory are
    not recomputed.

    Raises:
        ValueError: if `scratch_dir` is not a local filesystem path.
    """
    global _configured
    if scratch_dir is not None:
        _configured = _local_path(scratch_dir)


def scratch_dir() -> Path:
    """The scratch directory."""
    path = _scoped.get() or _configured
    if path is not None:
        return path
    env = os.environ.get(SCRATCH_DIR_ENV)
    return _local_path(env, SCRATCH_DIR_ENV) if env else _DEFAULT_SCRATCH_DIR


@contextmanager
def scoped(*, scratch_dir: Union[str, "os.PathLike[str]"]) -> Iterator[Path]:
    """Uses `scratch_dir` as the scratch directory within this context."""
    path = _local_path(scratch_dir)
    token = _scoped.set(path)
    try:
        yield path
    finally:
        _scoped.reset(token)
