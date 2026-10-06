# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`DatabricksConnector`, mirroring the argument list of the cloud SDK's
`kumoai.connector.databricks_connector.DatabricksConnector`, minus everything
that exists only to round-trip a server-side connector registry. See
`kumo.engine.connector.base`.
"""

from functools import cached_property
from typing import Optional

from kumoapi.data_source import (
    DatabricksConnectorResourceConfig,
    DataSourceType,
)
from kumoapi.plugin import ArtifactSink, SourceBackend
from kumoapi.source_table import DatabricksSourceTable
from typing_extensions import Self

from kumo.engine.connector.base import Connector
from kumo.engine.connector.identity import DatabricksIdentity


class DatabricksConnector(Connector):
    r"""Connect directly to a Databricks Unity Catalog catalog and schema.

    Reads and writes through `kumo_databricks.
    DatabricksSourceBackend`/`DatabricksSinkBackend` -- install the
    `databricks` extra of this package to use it.

    Specify exactly one of `warehouse_id` and `cluster_id` (enforced by
    `kumoapi.data_source.DatabricksConnectorResourceConfig` itself).

    Args:
        catalog: Databricks Unity Catalog name.
        schema_name: Schema within `catalog`.
        identity: The `DatabricksIdentity` used to authenticate. See
            `from_notebook` for the credential-free notebook case.
        warehouse_id: Databricks SQL warehouse ID. Mutually exclusive with
            `cluster_id`.
        cluster_id: Databricks all-purpose cluster ID. Mutually exclusive
            with `warehouse_id`.
        staging_volume_path: Optional Unity Catalog volume path used to
            unload Parquet before downloading it locally. Without one,
            tables are streamed through CloudFetch.
        require_staged: Require staged export/download instead of falling
            back to streaming. Requires `staging_volume_path`.
    """
    def __init__(
        self,
        *,
        catalog: str,
        schema_name: str,
        identity: DatabricksIdentity,
        warehouse_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        staging_volume_path: Optional[str] = None,
        require_staged: bool = False,
    ) -> None:
        if require_staged and not staging_volume_path:
            raise ValueError("require_staged needs a staging_volume_path")
        super().__init__(identity=identity)
        self._config = DatabricksConnectorResourceConfig(
            catalog=catalog,
            schema_name=schema_name,
            warehouse_id=warehouse_id,
            cluster_id=cluster_id,
        )
        self.catalog = self._config.catalog
        self.schema_name = self._config.schema_name
        self.warehouse_id = self._config.warehouse_id
        self.cluster_id = self._config.cluster_id
        self.staging_volume_path = staging_volume_path
        self.require_staged = require_staged

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.DATABRICKS

    @property
    def config(self) -> DatabricksConnectorResourceConfig:
        return self._config

    @cached_property
    def source_backend(self) -> SourceBackend:
        from kumo_core.sources.types import ReadMode
        from kumo_databricks.backend import DatabricksSourceBackend
        return DatabricksSourceBackend(
            identity=self.identity,
            staging_volume_path=self.staging_volume_path,
            mode=ReadMode.STAGED if self.require_staged else None,
        )

    @cached_property
    def sink_backend(self) -> ArtifactSink:
        from kumo_databricks.sink import DatabricksSinkBackend
        return DatabricksSinkBackend(identity=self.identity)

    def source_table_for(self, table_name: str) -> DatabricksSourceTable:
        return DatabricksSourceTable(
            databricks_connector_id=self.id,
            table=f'{self.catalog}.{self.schema_name}.{table_name}',
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(catalog={self.catalog!r}, "
                f"schema_name={self.schema_name!r})")

    @classmethod
    def from_notebook(
        cls,
        *,
        catalog: str,
        schema_name: str,
        warehouse_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        staging_volume_path: Optional[str] = None,
        require_staged: bool = False,
    ) -> Self:
        r"""Builds a connector whose identity defers entirely to the
        Databricks SDK's own ambient auth-resolution chain -- inside a
        Databricks notebook, that chain already has everything it needs
        from the notebook's attached execution context, no credentials
        passed in. Similar in spirit to `kumo.engine.connector.snowflake.
        SnowflakeConnector.from_spcs_env`, which reads a Snowpark Container
        Services-mounted OAuth token; here there is no file or environment
        variable this call reads itself; `kumo_databricks.session` builds a
        bare `databricks.sdk.WorkspaceClient()` and lets the SDK resolve the
        notebook context on its own.

        Unlike `from_spcs_env`, a Databricks notebook's environment carries
        no query target to default to, so `catalog`/`schema_name` and
        exactly one of `warehouse_id`/`cluster_id` must still be named
        explicitly.

        Raises:
            ValueError: if neither `warehouse_id` nor `cluster_id` is given,
                or both are (enforced by `DatabricksConnectorResourceConfig`).
        """
        return cls(
            catalog=catalog,
            schema_name=schema_name,
            identity=DatabricksIdentity(),
            warehouse_id=warehouse_id,
            cluster_id=cluster_id,
            staging_volume_path=staging_volume_path,
            require_staged=require_staged,
        )
