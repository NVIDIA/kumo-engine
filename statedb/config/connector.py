# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Union

import kumoapi.data_source as api
from kumo_core.types import ConnectorType
from pydantic import Field, model_validator

from io_utils import kumo_dataclass

# Reserved, non-customer namespace for the one hidden upload connector in each
# project. Repository validation prevents external connectors from claiming it.
PROJECT_UPLOAD_CONNECTOR_NAME = '__kumo_internal_project_uploads__'


@kumo_dataclass
class SnowflakeConnectorResourceConfig:
    name: str
    warehouse: str
    database: str
    schema_name: str
    role: Optional[str] = None
    type: Literal[api.ConnectorType.SNOWFLAKE] = api.ConnectorType.SNOWFLAKE


@kumo_dataclass
class DatabricksConnectorResourceConfig:
    name: str
    catalog: str
    schema_name: str
    cluster_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    type: Literal[api.ConnectorType.DATABRICKS] = api.ConnectorType.DATABRICKS


@kumo_dataclass
class BigQueryConnectorResourceConfig:
    name: str
    project_id: str
    dataset_id: str
    location: str
    billing_project_id: Optional[str] = None
    type: Literal[api.ConnectorType.BIGQUERY] = api.ConnectorType.BIGQUERY


@kumo_dataclass
class S3ConnectorResourceConfig:
    name: str
    target: api.S3ConnectorTarget = Field(discriminator='mode')
    type: Literal[api.ConnectorType.S3] = api.ConnectorType.S3


@kumo_dataclass
class GCSConnectorResourceConfig:
    name: str
    target: api.GCSConnectorTarget = Field(discriminator='mode')
    requester_pays_project: Optional[str] = None
    type: Literal[api.ConnectorType.GCS] = api.ConnectorType.GCS

    @model_validator(mode='before')
    @classmethod
    def migrate_legacy_rooted_target(cls, values):
        return api.GCSConnectorResourceConfig.migrate_legacy_rooted_target(
            values)


@kumo_dataclass
class ADLSConnectorResourceConfig:
    name: str
    target: api.ADLSConnectorTarget = Field(discriminator='mode')
    type: Literal[api.ConnectorType.ADLS] = api.ConnectorType.ADLS

    @model_validator(mode='before')
    @classmethod
    def migrate_legacy_rooted_target(cls, values):
        return api.ADLSConnectorResourceConfig.migrate_legacy_rooted_target(
            values)


ExternalConnectorResourceConfigDB = Union[
    SnowflakeConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
    S3ConnectorResourceConfig,
    GCSConnectorResourceConfig,
    ADLSConnectorResourceConfig,
]


@kumo_dataclass
class FileConnectorResourceConfig:
    name: str
    root_dir: str
    type: Literal[ConnectorType.FILE] = ConnectorType.FILE


InternalConnectorResourceConfigDB = FileConnectorResourceConfig

ConnectorResourceConfigDB = Union[
    SnowflakeConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    BigQueryConnectorResourceConfig,
    S3ConnectorResourceConfig,
    GCSConnectorResourceConfig,
    ADLSConnectorResourceConfig,
    FileConnectorResourceConfig,
]
