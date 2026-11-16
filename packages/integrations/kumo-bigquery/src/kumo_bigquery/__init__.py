# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BigQuery, for the Kumo platform and for the SDK's local backend.

Two readers live here and they are not the same thing.
:class:`BigQuerySourceReader` implements ``kumo_core``'s ``SourceReader`` and
is what the platform binds through. :class:`BigQuerySourceBackend` implements
:class:`kumoapi.plugin.SourceBackend` and is what a ``LocalBackend`` registers
for ``DataSourceType.BIGQUERY``; the identity lifecycle beside it is the rest
of that plugin; it is in ``backend.py``, not ``source.py``, because that name
is already the platform reader's. :class:`BigQuerySinkBackend` writes
prediction output.

Nothing here imports ``kumoai_studio``. That is what lets one distribution serve
both callers, and ``tests/test_package_boundary.py`` enforces it rather than
leaving it to discipline.

The plugin half is exported **lazily**, matching ``kumo_snowflake`` and
``kumo_databricks``, so ``import kumo_bigquery`` stays cheap for a
platform caller that wants :class:`BigQuerySourceReader` alone.
"""

from __future__ import annotations

from kumo_core.sources.lazy_exports import module_dir, module_getattr
from kumo_core.sources.types import Capability

from kumo_bigquery.source import BigQuerySourceReader

#: Where each lazily exported name is defined. The plugin's modules, and only
#: those: everything eager above is reachable on the platform's install.
_LAZY_EXPORTS = {
    "BigQuerySinkBackend": "sink",
    "BigQuerySourceBackend": "backend",
    "close": "session",
    "probe": "session",
}

__all__ = [
    "BigQuerySourceReader",
    "Capability",
    *sorted(_LAZY_EXPORTS),
]

__getattr__ = module_getattr(__name__, _LAZY_EXPORTS)
__dir__ = module_dir(__all__)
