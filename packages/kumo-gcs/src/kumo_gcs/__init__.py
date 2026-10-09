# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared GCS downloads and local SDK source/sink plugins.

Plugin exports are lazy so importing the storage namespace does not load
provider clients or the SDK.
"""

from __future__ import annotations

from kumo_core.sources.lazy_exports import module_dir, module_getattr
from kumo_gcs.session import GCS_NAMESPACE

#: Where each lazily exported name is defined.
_LAZY_EXPORTS = {
    'GCSSourceBackend': 'backend',
    'GCSSinkBackend': 'sink',
    'probe': 'backend',
}

__all__ = ['GCS_NAMESPACE', *sorted(_LAZY_EXPORTS)]

__getattr__ = module_getattr(__name__, _LAZY_EXPORTS)
__dir__ = module_dir(__all__)
