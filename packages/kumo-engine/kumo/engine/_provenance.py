# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime versions and source tables recorded in manifests."""

import hashlib
import platform
import subprocess
import warnings
from functools import cache
from pathlib import Path
from typing import Any, Dict, Optional

from kumo.engine._manifest import Environment, SourceTableEntry

_LOCATED_CONNECTORS = frozenset(
    {'LOCAL_FILE', 'S3', 'GCS', 'SNOWFLAKE', 'DATABRICKS', 'BIGQUERY'})


@cache
def _build_commit(version: str) -> Optional[str]:
    if '+' in version:
        return version.split('+', 1)[1]
    from kumo.engine import _version
    stamped = getattr(_version, 'BUILD_COMMIT', None)
    if stamped:
        return stamped
    root = Path(__file__).resolve().parents[2]
    if not (root / 'pyproject.toml').is_file():
        return None
    command = ['git', '-C', str(root), 'rev-parse', '--short', 'HEAD']
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _diskgraph() -> Any:
    try:
        import diskgraph
        return diskgraph
    except ImportError:
        return None


def environment() -> Environment:
    from kumo_core.version import (
        get_kumo_version,
        installed_distribution_version,
    )

    from kumo.engine._version import __version__

    diskgraph = _diskgraph()
    return Environment(
        kumo_engine=get_kumo_version(),
        build_commit=_build_commit(__version__),
        kumo_core=installed_distribution_version('kumo-core'),
        diskgraph=getattr(diskgraph, '__version__', None),
        diskgraph_build_commit=getattr(diskgraph, 'BUILD_COMMIT', None),
        torch=installed_distribution_version('torch'),
        python=platform.python_version(),
    )


def source_location(connector: Any, table_name: str,
                    sql: Optional[str] = None) -> str:
    if sql is not None:
        return 'sql:' + hashlib.sha256(sql.encode()).hexdigest()
    if _connector_kind(connector) not in _LOCATED_CONNECTORS:
        return table_name
    return connector.source_backend.table_location(connector, table_name)


def _connector_kind(connector: Any) -> str:
    kind = connector.source_type
    return str(getattr(kind, 'value', kind))


def source_table(connector: Any, table_name: str, sql: Optional[str],
                 version: Any, *,
                 record_sources: bool = True) -> SourceTableEntry:
    from kumo_core.sources.identity import version_kind

    entry: Dict[str, Any] = {'connector': None, 'location': None}
    dependencies = getattr(version, 'dependencies', None)
    try:
        entry['connector'] = _connector_kind(connector)
        entry['location'] = (source_location(connector, table_name, sql)
                             if record_sources or sql is not None else
                             table_name)
        if sql is not None:
            _record_sql(entry, connector, sql, record_sources)
    except Exception as error:
        warnings.warn(f"Could not record a source table's location: {error}",
                      UserWarning, stacklevel=3)
    return SourceTableEntry(
        version=None if version is None else str(version),
        version_kind=version_kind(version),
        dependencies=None if dependencies is None else {
            name: str(value)
            for name, value in dependencies.items()
        },
        **entry,
    )


def _record_sql(entry: Dict[str, Any], connector: Any, sql: str,
                record_sources: bool) -> None:
    from kumo_core.custom_sql import (
        CUSTOM_SQL_NAMESPACE_ATTRS,
        custom_sql_synthetic_name,
    )
    from kumo_core.types import ConnectorType

    kind = ConnectorType[entry['connector']]
    entry['normalized_sql_fingerprint'] = custom_sql_synthetic_name(sql, kind)
    if record_sources:
        entry['sql'] = sql
        entry['dialect'] = entry['connector'].lower()
        entry['namespace'] = [
            getattr(connector.config, attr)
            for attr in CUSTOM_SQL_NAMESPACE_ATTRS[kind]
        ]
