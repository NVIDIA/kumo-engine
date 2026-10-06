# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`BigQueryConnector`, mirroring the argument list of the cloud SDK's
`kumoai.connector.bigquery_connector.BigQueryConnector`, minus everything
that exists only to round-trip a server-side connector registry. See
`kumo.engine.connector.base`.
"""

from functools import cached_property
from typing import Optional

from kumoapi.data_source import BigQueryConnectorResourceConfig, DataSourceType
from kumoapi.source_table import BigQuerySourceTable

from kumo.engine.connector.base import Connector
from kumo.engine.connector.identity import BigQueryIdentity


class BigQueryConnector(Connector):
    r"""Connect directly to a Google BigQuery dataset.

    Read-only: reads through `kumo_bigquery.BigQuerySourceBackend` (there is
    no BigQuery sink plugin) -- install the `bigquery` extra of this package
    to use it.

    Args:
        project_id: Google Cloud project containing the dataset.
        dataset_id: BigQuery dataset ID.
        identity: A `BigQueryIdentity`, passed directly. `None` defers to
            `kumo_bigquery`'s own ambient-credential resolution.
        location: BigQuery dataset location, e.g. `"US"` or
            `"europe-west1"`.
        billing_project_id: Optional Google Cloud project to bill for
            queries.
        staging_uri: Optional `gs://` prefix used to export Parquet before it
            is downloaded locally. Without one, tables are streamed through
            the BigQuery Storage Read API.
        require_staged: Require staged export/download instead of falling
            back to streaming when staging is unavailable. Requires
            `staging_uri`.
    """
    def __init__(
        self,
        *,
        project_id: str,
        dataset_id: str,
        identity: Optional[BigQueryIdentity] = None,
        location: str = 'US',
        billing_project_id: Optional[str] = None,
        staging_uri: Optional[str] = None,
        require_staged: bool = False,
    ) -> None:
        if require_staged and not staging_uri:
            raise ValueError("require_staged needs a staging_uri")
        super().__init__(identity=identity)
        self._config = BigQueryConnectorResourceConfig(
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
            billing_project_id=billing_project_id,
        )
        self.project_id = self._config.project_id
        self.dataset_id = self._config.dataset_id
        self.location = self._config.location
        self.billing_project_id = self._config.billing_project_id
        self.staging_uri = staging_uri
        self.require_staged = require_staged

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.BIGQUERY

    @property
    def config(self) -> BigQueryConnectorResourceConfig:
        return self._config

    @cached_property
    def source_backend(self):
        from kumo_bigquery.backend import BigQuerySourceBackend
        from kumo_core.sources.types import ReadMode
        return BigQuerySourceBackend(
            identity=self.identity,
            staging_uri=self.staging_uri,
            mode=ReadMode.STAGED if self.require_staged else None,
        )

    @property
    def sink_backend(self):
        raise NotImplementedError(
            "BigQueryConnector is read-only: there is no BigQuery sink "
            "plugin, so it cannot be used as an output_connector/"
            "export_config connector, or as Trainer.load_from_connector(...).")

    def source_table_for(self, table_name: str) -> BigQuerySourceTable:
        return BigQuerySourceTable(
            bigquery_connector_id=self.id,
            table_name=table_name,
            project_id=self.project_id,
            dataset_id=self.dataset_id,
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(project_id={self.project_id!r}, "
                f"dataset_id={self.dataset_id!r})")
