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
