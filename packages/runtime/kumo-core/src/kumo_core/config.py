# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for a service-independent local training runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class NativeRuntimeConfig:
    """Filesystem and recovery policy for one native runtime.

    ``scratch_root`` contains reconstructable working data. ``artifact_root``
    contains durable run manifests and outputs. The roots are kept explicit so
    adapters can map them to local NVMe and durable storage respectively.
    """

    scratch_root: Path
    artifact_root: Path
    resume: bool = True
    manifest_filename: str = "run-manifest.json"

    def __post_init__(self) -> None:
        object.__setattr__(self, "scratch_root", Path(self.scratch_root))
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))
        if not self.manifest_filename:
            raise ValueError("manifest_filename must not be empty")
        manifest_path = Path(self.manifest_filename)
        if (manifest_path.is_absolute() or len(manifest_path.parts) != 1
                or _SAFE_FILENAME.fullmatch(self.manifest_filename) is None):
            raise ValueError("manifest_filename must be a safe single filename")
