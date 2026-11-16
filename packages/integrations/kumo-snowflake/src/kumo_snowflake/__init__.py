# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snowflake, for the Kumo platform and for the SDK's local backend.

Two readers live here and they are not the same thing.
:class:`SnowflakeSourceReader` implements ``kumo_core``'s ``SourceReader`` and
is what the platform binds through. :class:`SnowflakeSourceBackend` implements
:class:`kumoapi.plugin.SourceBackend` and is what a ``LocalBackend`` registers
for ``DataSourceType.SNOWFLAKE``; the sink, the stage artifact location and the
identity lifecycle beside it are the rest of that plugin.

Nothing here imports ``kumoai_studio``. That is what lets one distribution serve both
callers, and ``tests/test_package_boundary.py`` enforces it rather than leaving
it to discipline.

The plugin half is exported **lazily**. It needs Snowpark, which arrives with
the ``sdk`` extra; the platform installs this distribution without that extra
and imports :class:`SnowflakeSourceReader` alone, so naming the plugin here
eagerly would make the platform's own import fail on a dependency it declined.
Resolving a name below is what loads it.
"""

from __future__ import annotations

from kumo_core.sources.lazy_exports import module_dir, module_getattr

from kumo_snowflake.source import SnowflakeSourceReader
from kumo_snowflake.spcs import spcs_identity_create_args

#: Where each lazily exported name is defined. The plugin's modules, and only
#: those: everything eager above is reachable on the platform's install.
_LAZY_EXPORTS = {
    "SnowflakeSession": "session",
    "SnowflakeSinkBackend": "sink",
    "SnowflakeSourceBackend": "backend",
    "SourceConnector": "session",
    "StageArtifactLocation": "artifacts",
    "close": "session",
    "probe": "session",
}

__all__ = [
    "SnowflakeSourceReader",
    "spcs_identity_create_args",
    *sorted(_LAZY_EXPORTS),
]

__getattr__ = module_getattr(__name__, _LAZY_EXPORTS)
__dir__ = module_dir(__all__)
