# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide settings shared by the Kumo packages, set with `configure`.

The scratch directory is where DiskGraph ingest, training and prediction read
and write their working data. It defaults to `~/.cache/kumo-engine`.
"""

import os
from pathlib import Path
from typing import Optional, Union

_DEFAULT_SCRATCH_DIR = Path.home() / ".cache" / "kumo-engine"
_configured: Optional[Path] = None


def configure(*, scratch_dir: Optional[Union[str, "os.PathLike[str]"]] = None) -> None:
    """Sets process-wide settings. `None` leaves a setting unchanged.

    Call it once, early: paths already derived from the scratch directory are
    not recomputed.

    Raises:
        ValueError: if `scratch_dir` is not a local filesystem path.
    """
    global _configured
    if scratch_dir is None:
        return
    text = os.fspath(scratch_dir)
    if text.startswith("@") or "://" in text:
        raise ValueError(
            f"scratch_dir must be a local filesystem path, not {text!r} -- "
            "DiskGraph ingest and training read and write it directly.")
    _configured = Path(text).expanduser()


def scratch_dir() -> Path:
    """The scratch directory."""
    return _configured or _DEFAULT_SCRATCH_DIR
