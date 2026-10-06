# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from kumo_core.sources.lazy_exports import module_dir, module_getattr

from kumo_s3.errors import translated
from kumo_s3.session import AWS_NAMESPACE, ambient_session, bucket_region, filesystem

#: Where each lazily exported name is defined. The plugin's modules, and only
#: those: everything eager above is the binding the platform uses too, and
#: stays cheap for it.
_LAZY_EXPORTS = {
    "S3SourceBackend": "backend",
    "S3SinkBackend": "sink",
    "probe": "backend",
    "source_access": "access",
}

__all__ = [
    "AWS_NAMESPACE",
    "ambient_session",
    "bucket_region",
    "filesystem",
    "translated",
    *sorted(_LAZY_EXPORTS),
]

__getattr__ = module_getattr(__name__, _LAZY_EXPORTS)
__dir__ = module_dir(__all__)
