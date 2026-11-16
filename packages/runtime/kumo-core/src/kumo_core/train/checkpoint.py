# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch loading support for trusted Kumo training checkpoints."""

from pathlib import Path
from typing import IO, Any

from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning_fabric.utilities.cloud_io import _load as pl_load


def _preserve_storage_location(storage: Any, location: str) -> Any:
    return storage


def load_kumo_checkpoint(
    path: str | Path | IO[bytes],
    *,
    map_location: Any,
) -> dict[str, Any]:
    """Load a trusted Kumo checkpoint across supported PyTorch runtimes."""
    return pl_load(
        path,
        map_location=map_location,
        weights_only=False,
    )


class KumoCheckpointIO(TorchCheckpointIO):
    """Load trusted Kumo checkpoints with stable PyTorch semantics."""
    def load_checkpoint(
        self,
        path: str | Path,
        map_location: Any = _preserve_storage_location,
        weights_only: bool | None = None,
    ) -> dict[str, Any]:
        # TorchCheckpointIO defines this override parameter.
        del weights_only
        return load_kumo_checkpoint(path, map_location=map_location)
