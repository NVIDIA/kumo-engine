# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve recipe runtime devices without silent CUDA fallbacks."""

from __future__ import annotations

from typing import Any

import torch


def resolve_device(spec: str) -> torch.device:
    """Map `runtime.device` to a `torch.device`.

    Accepted values: `cpu`, `cuda`, `cuda:N`, and `auto` (CUDA if available).
    Requesting CUDA when it is not available is an error.
    """

    requested = spec.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"runtime.device={spec!r} requires CUDA, but torch.cuda.is_available() is False."
            )
        device = torch.device(requested)
        index = 0 if device.index is None else device.index
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise RuntimeError(
                f"runtime.device={spec!r} is out of range; torch.cuda.device_count()={count}."
            )
        return device
    raise ValueError(
        f"Unsupported runtime.device {spec!r}; use 'cpu', 'cuda', 'cuda:N', or 'auto'."
    )


def device_environment(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        info["gpu_index"] = int(index)
        info["gpu_name"] = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        info["gpu_capability"] = f"{major}.{minor}"
    return info
