# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-backend check for the installed PyTorch build.

PyTorch publishes CPU, CUDA, ROCm and Intel XPU builds of the same version,
distinguished only by a PEP 440 local version label such as ``2.9.0+cu126``.
A package index that carries every variant hands pip the one whose label
sorts last, and ``xpu`` sorts above ``cu126``, ``cpu`` and ``rocm``. An
organization mirroring that index therefore gets the Intel build by default,
even though Kumo cannot use it.
"""

from __future__ import annotations

import functools
import os
import warnings

# Device backends Kumo does not support. Matched against the leading
# component of the local version label, so "rocm6.2" matches "rocm".
UNSUPPORTED_BACKENDS = {
    "xpu": "an Intel XPU",
    "rocm": "an AMD ROCm",
    "hpu": "an Intel Gaudi",
}

CUDA_INDEX_HINT = (
    "pip install torch --index-url https://download.pytorch.org/whl/cu126")


def local_label(version: str) -> str | None:
    """Return the PEP 440 local version label, or ``None`` when absent.

    ``"2.9.0+cu126"`` gives ``"cu126"``; ``"2.9.0"`` gives ``None``.
    """
    label = version.partition("+")[2]
    return label or None


def unsupported_backend(version: str) -> str | None:
    """Return the unsupported backend key for a torch version, else ``None``.

    A plain version and the ``cpu`` and ``cu*`` builds are all supported, so
    they return ``None``.
    """
    label = local_label(version)
    if label is None:
        return None
    # Local labels are dot-separated; "rocm6.2" -> "rocm6".
    head = label.split(".")[0]
    for backend in UNSUPPORTED_BACKENDS:
        if head.startswith(backend):
            return backend
    return None


def nvidia_driver_present() -> bool:
    """Whether an NVIDIA driver looks present, without running a subprocess.

    Only meaningful on Linux. Returns ``False`` on macOS, where Apple silicon
    uses the MPS backend and no NVIDIA driver is ever expected.
    """
    try:
        if not os.path.exists("/proc/driver/nvidia/version"):
            return os.path.exists("/dev/nvidiactl")
        return True
    except OSError:
        return False


@functools.cache
def check_torch_build() -> None:
    """Warn once per process when the installed PyTorch build cannot serve.

    Reports an unsupported device backend, and separately reports a PyTorch
    without CUDA support on a machine that has an NVIDIA driver. Never raises.
    Set ``KUMO_CHECK_TORCH_BUILD=0`` to skip.
    """
    if os.environ.get("KUMO_CHECK_TORCH_BUILD") == "0":
        return
    try:
        import torch

        version = torch.__version__
        backend = unsupported_backend(version)
        if backend is not None:
            warnings.warn(
                f"PyTorch {version} is {UNSUPPORTED_BACKENDS[backend]} build. "
                f"Kumo needs a CUDA build on Linux, or the default build on "
                f"Apple silicon, so training will fail or fall back to CPU. "
                f"This usually means a package index served every PyTorch "
                f"variant and pip chose this one. Reinstall PyTorch with: "
                f"{CUDA_INDEX_HINT}. Set KUMO_CHECK_TORCH_BUILD=0 to silence "
                f"this check.", UserWarning, stacklevel=3)
            return
        if torch.version.cuda is None and nvidia_driver_present():
            warnings.warn(
                f"PyTorch {version} has no CUDA support, but this machine has "
                f"an NVIDIA driver. Kumo will run on CPU and be much slower. "
                f"Reinstall PyTorch with: {CUDA_INDEX_HINT}. Set "
                f"KUMO_CHECK_TORCH_BUILD=0 to silence this check.",
                UserWarning, stacklevel=3)
    except Exception:  # noqa: BLE001
        # A diagnostic must never break an import.
        pass
