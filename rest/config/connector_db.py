# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import kumoapi.data_source as api
from pydantic import Field

from io_utils import kumo_dataclass
from rest.config.connector import (
    ADLSConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    ExternalConnectorResourceConfigDB,
    GCSConnectorResourceConfig,
    InternalConnectorResourceConfigDB,
    S3ConnectorResourceConfig,
    SnowflakeConnectorResourceConfig,
)
from rest.config.data_backend import ConnectorSpec
from statedb.rest.connector_db import \
    ExternalConnectorDBEntry as SerializedExternalConnectorDBEntry
from statedb.rest.connector_db import \
    InternalConnectorDBEntry as SerializedInternalConnectorDBEntry


def _api_timestamps(timestamps) -> api.ResourceTimestamps:
    return api.ResourceTimestamps(
        created_at=timestamps.created_at,
        updated_at=timestamps.updated_at,
    )


def _api_external_config(
    config: ExternalConnectorResourceConfigDB, ) -> api.ConnectorResourceConfig:
    if isinstance(config, SnowflakeConnectorResourceConfig):
        return api.SnowflakeConnectorResourceConfig(
            warehouse=config.warehouse,
            database=config.database,
            schema_name=config.schema_name,
            role=config.role,
        )
    if isinstance(config, DatabricksConnectorResourceConfig):
        return api.DatabricksConnectorResourceConfig(
            catalog=config.catalog,
            schema_name=config.schema_name,
            cluster_id=config.cluster_id,
            warehouse_id=config.warehouse_id,
        )
    if isinstance(config, BigQueryConnectorResourceConfig):
        return api.BigQueryConnectorResourceConfig(
            project_id=config.project_id,
            dataset_id=config.dataset_id,
            location=config.location,
            billing_project_id=config.billing_project_id,
        )
    if isinstance(config, S3ConnectorResourceConfig):
        return api.S3ConnectorResourceConfig(target=config.target)
    if isinstance(config, GCSConnectorResourceConfig):
        return api.GCSConnectorResourceConfig(
            target=config.target,
            requester_pays_project=config.requester_pays_project,
        )
    if isinstance(config, ADLSConnectorResourceConfig):
        return api.ADLSConnectorResourceConfig(target=config.target)
    raise TypeError(f'Unsupported external connector config: {type(config)}')


@kumo_dataclass
class ExternalConnectorDBEntry(SerializedExternalConnectorDBEntry):
    config: ExternalConnectorResourceConfigDB = Field(discriminator='type')

    def to_connector_response(self) -> api.ConnectorResponse:
        return api.ConnectorResponse(
            id=str(self.id),
            project_id=str(self.project_id),
            identity_id=str(self.identity_id),
            name=self.name,
            config=_api_external_config(self.config),
            timestamps=_api_timestamps(self.timestamps),
            description=self.description,
            created_by=self.created_by,
        )


@kumo_dataclass
class InternalConnectorDBEntry(SerializedInternalConnectorDBEntry):
    config: InternalConnectorResourceConfigDB = Field(discriminator='type')

    def as_connector_spec(self) -> ConnectorSpec:
        return ConnectorSpec(
            connector_id=self.id,
            connector_config=self.config.as_connector().config(),
        )


ConnectorDBEntry = Union[ExternalConnectorDBEntry, InternalConnectorDBEntry]
