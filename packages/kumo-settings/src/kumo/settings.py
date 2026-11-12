# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide settings shared by the Kumo packages."""

import os
from pathlib import Path
from typing import Optional, Union

_DEFAULT_SCRATCH_DIR = Path.home() / ".cache" / "kumo-engine"
_configured: Optional[Path] = None


def configure(*, scratch_dir: Optional[Union[str, "os.PathLike[str]"]] = None) -> None:
    """Sets process-wide settings; call it once, early. `None` leaves a setting unchanged."""
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
    """The local directory DiskGraph ingest, training and prediction work in."""
    return _configured or _DEFAULT_SCRATCH_DIR
