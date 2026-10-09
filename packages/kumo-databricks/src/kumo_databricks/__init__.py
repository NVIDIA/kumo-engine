# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Databricks provider capabilities for Kumo workflow hosts.

Two readers live here and they are not the same thing.
:class:`DatabricksSourceReader` implements ``kumo_core``'s ``SourceReader``
and is what the platform binds through. :class:`DatabricksSourceBackend`
implements :class:`kumoapi.plugin.SourceBackend` and is what ``kumo-engine``
and the hosted SDK use for ``DataSourceType.DATABRICKS``; the sink and the
identity lifecycle beside it are the rest of that plugin. Named ``backend.py``,
not ``source.py``, because that name is already the platform reader's.

Nothing here imports either workflow host. That lets one distribution serve
both callers, and ``tests/test_package_boundary.py`` enforces it rather than
leaving it to discipline.

The plugin half is exported **lazily**, matching ``kumo_snowflake``, so
``import kumo_databricks`` stays cheap for a platform caller that
wants :class:`DatabricksSourceReader` alone.
"""

from __future__ import annotations

from kumo_core.sources.lazy_exports import module_dir, module_getattr

from kumo_databricks.source import DatabricksSourceReader

#: Where each lazily exported name is defined. The plugin's modules, and only
#: those: everything eager above is reachable on the platform's install.
_LAZY_EXPORTS = {
    "DatabricksSession": "session",
    "DatabricksSinkBackend": "sink",
    "DatabricksSourceBackend": "backend",
    "probe": "session",
}

__all__ = [
    "DatabricksSourceReader",
    *sorted(_LAZY_EXPORTS),
]

__getattr__ = module_getattr(__name__, _LAZY_EXPORTS)
__dir__ = module_dir(__all__)
