# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Azure Data Lake Storage plugins; provider clients load on first use."""

from kumo_core.sources.lazy_exports import module_dir, module_getattr

ADLS_NAMESPACE = 'azure-adls'
_LAZY_EXPORTS = {
    'ADLSSourceBackend': 'backend',
    'ADLSSinkBackend': 'sink',
    'probe': 'session',
}
__all__ = ['ADLS_NAMESPACE', *sorted(_LAZY_EXPORTS)]
__getattr__ = module_getattr(__name__, _LAZY_EXPORTS)
__dir__ = module_dir(__all__)
