# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Local-mode connectors.

One connector class per already-existing `kumo-*` plugin package
(`kumo-snowflake`, `kumo-s3`, `kumo-gcs`, `kumo-bigquery`,
`kumo-databricks`). Each wraps
a `kumoapi.plugin.SourceBackend` implementation (and, where available,
`ArtifactSink`/`ModelPublishingSink`/`ModelFetchingSink`/`ModelListingSink`)
and is threaded directly wherever a connector is needed -- never referenced
by a string id.

Provider argument lists mirror the equivalent cloud SDK classes
(`kumoai/connector/{snowflake,s3,gcs,bigquery,databricks}_connector.py`)
minus the server-registry-only fields (`name`, `description`, `project_id`,
`_response`) and the `_bind_connector` round-trip. See `kumo.engine.connector.base`.
"""

from typing import Any, Optional, Tuple

from kumo.engine.connector.base import Connector
from kumo.engine.connector.bigquery import BigQueryConnector
from kumo.engine.connector.databricks import DatabricksConnector
from kumo.engine.connector.file import FileConnector
from kumo.engine.connector.gcs import GCSConnector
from kumo.engine.connector.identity import (
    BigQueryIdentity,
    DatabricksIdentity,
    GCSIdentity,
    Identity,
    S3Identity,
    SnowflakeIdentity,
)
from kumo.engine.connector.s3 import S3Connector
from kumo.engine.connector.snowflake import SnowflakeConnector
from kumo.engine.connector.source_table import SourceTable, SQLSourceTable


def _file_sink(connector: Any, path: str) -> Any:
    from kumoapi.plugin import FileStoreSink

    try:
        sink = connector.sink_backend
    except Exception:
        return None
    return sink if isinstance(sink, FileStoreSink) and sink.holds_files(
        connector, path) else None


def is_file_destination(connector: Any, path: str) -> bool:
    r"""Whether ``path`` is a file location of ``connector``'s sink
    (`kumoapi.plugin.FileStoreSink`) rather than a table; no I/O."""
    return _file_sink(connector, path) is not None


def file_store(connector: Any, path: str) -> Optional[Tuple[Any, str]]:
    r"""``connector``'s ``(pyarrow filesystem, resolved path)`` for a file
    destination; None for a table."""
    sink = _file_sink(connector, path)
    return None if sink is None else sink.filesystem(connector, path)


def model_sink(connector: Any, path: Optional[str]) -> Any:
    r"""The sink that publishes or fetches a model at ``path``: the file
    writers over ``connector``'s sink (`FilesystemSink`) for a file path of
    a sink without its own, else the sink."""
    from kumo_core.object_store_sink import (
        FilesystemSink,
        ObjectStoreSinkBackend,
    )

    sink = connector.sink_backend
    if (path is None or isinstance(sink, ObjectStoreSinkBackend)
            or _file_sink(connector, path) is None):
        return sink
    return FilesystemSink(sink)


__all__ = [
    "Connector",
    "FileConnector",
    "SnowflakeConnector",
    "S3Connector",
    "GCSConnector",
    "BigQueryConnector",
    "DatabricksConnector",
    "SourceTable",
    "SQLSourceTable",
    "Identity",
    "SnowflakeIdentity",
    "S3Identity",
    "GCSIdentity",
    "BigQueryIdentity",
    "DatabricksIdentity",
]
