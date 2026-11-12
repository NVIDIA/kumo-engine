# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host checks for the Lightning accelerator choice.

PyTorch MPS kernels in the supported torch range call Metal APIs that exist
only on macOS 15 and later. On macOS 14 the first MPS linear layer aborts the
whole process with an Objective-C exception, which Python cannot catch. This
module rejects that choice before any MPS kernel runs.
"""

from __future__ import annotations

import logging
import platform

__all__ = [
    'MPS_MIN_MACOS',
    'UnsupportedAcceleratorError',
    'mps_supported',
    'resolve_accelerator',
]

MPS_MIN_MACOS = 15

logger = logging.getLogger(__name__)
_warned_auto_cpu = False


class UnsupportedAcceleratorError(ValueError):
    """The requested accelerator cannot run on this host."""


def _macos_release() -> str | None:
    """The macOS product version, or None off macOS or when it is unknown."""
    if platform.system() != 'Darwin':
        return None
    return platform.mac_ver()[0] or None


def mps_supported() -> bool:
    """Whether this host's macOS can run the PyTorch MPS kernels.

    A host off macOS, or with an unknown version, returns True: PyTorch then
    decides on its own whether MPS exists.
    """
    release = _macos_release()
    try:
        return not release or int(release.split('.')[0]) >= MPS_MIN_MACOS
    except ValueError:
        return True


def resolve_accelerator(accelerator: str) -> str:
    """Return the Lightning accelerator to use on this host.

    On macOS earlier than 15, ``mps`` and ``gpu`` (which Lightning maps to
    MPS on a Mac) raise :class:`UnsupportedAcceleratorError`, and ``auto``
    resolves to ``cpu``. Every other host and choice is returned unchanged.
    """
    global _warned_auto_cpu
    if mps_supported() or accelerator not in ('mps', 'gpu', 'auto'):
        return accelerator
    release = _macos_release()
    if accelerator in ('mps', 'gpu'):
        raise UnsupportedAcceleratorError(
            f"accelerator={accelerator!r} uses Apple MPS, which requires "
            f"macOS {MPS_MIN_MACOS} or later, but this host runs macOS "
            f"{release}. Use accelerator='cpu' on this host.")
    if not _warned_auto_cpu:
        _warned_auto_cpu = True
        logger.warning(
            "Apple MPS requires macOS %d or later; this host runs macOS %s, "
            "so accelerator='auto' uses the CPU.", MPS_MIN_MACOS, release)
    return 'cpu'
