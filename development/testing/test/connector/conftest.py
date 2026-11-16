# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from dataclasses import replace
from typing import Generator

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.testing.databricks import DatabricksTableConnector
from kumo.testing.rest import (
    financial_snowflake_config,
    hm_databricks_config,
    writable_snowflake_config,
)
from kumo.testing.snowflake import SnowflakeTableConnector
from service_framework.deploy_env import DATABRICKS_HOST


@pytest.fixture(scope='session')
def require_kumo_public_datasets_s3() -> None:
    """Skip remote dataset tests when this environment lacks S3 access."""
    import boto3

    try:
        boto3.client('s3').list_objects_v2(
            Bucket='kumo-public-datasets',
            Prefix='financial/',
            MaxKeys=1,
        )
    except NoCredentialsError:
        pytest.skip('AWS credentials are unavailable for public S3 datasets')
    except ClientError as error:
        code = error.response.get('Error', {}).get('Code', '')
        if code in {
                'AccessDenied',
                'ExpiredToken',
                'InvalidAccessKeyId',
                'SignatureDoesNotMatch',
        }:
            pytest.skip(
                f'AWS access is unavailable for public S3 datasets ({code})')
        raise


@pytest.fixture()
def writable_snowflake_table(
        snowflake_credentials
) -> Generator[SnowflakeTableConnector, None, None]:
    name = f'TEST_PREDICTION_{str(uuid.uuid1()).upper().replace("-", "")}'
    connector = writable_snowflake_config('snowflake').as_connector('xva19026')
    try:
        yield SnowflakeTableConnector(name, connector)
    finally:
        connector.drop_table_if_exists(name)


@pytest.fixture()
def snowflake_financial_connector(snowflake_credentials) -> SnowflakeConnector:
    return financial_snowflake_config("name").as_connector('xva19026')


@pytest.fixture()
def writable_databricks_table(
) -> Generator[DatabricksTableConnector, None, None]:
    name = ('predictions__kumo__PREDICTION_'
            f'{str(uuid.uuid1()).upper().replace("-", "")}')
    connector = replace(hm_databricks_config('databricks'),
                        schema_name='predictions').as_connector(
                            DATABRICKS_HOST.parse_value())
    try:
        yield DatabricksTableConnector(name, connector)
    finally:
        # Credential-gated tests can be skipped after fixture setup. Do not
        # turn that skip into a teardown error by opening an otherwise unused
        # live connector solely for cleanup. A connected connector means live
        # setup succeeded far enough that the idempotent cleanup is needed.
        if connector.is_connected:
            connector.drop_table_if_exists(name)


@pytest.fixture
def databricks_hm_connector() -> DatabricksConnector:
    connector = DatabricksConnector('databricks',
                                    host=DATABRICKS_HOST.parse_value(),
                                    cluster_id='0626-191119-75431625',
                                    warehouse_id='82d859e4a6b283a0',
                                    catalog='hive_mapped_uc', schema_name='hm')
    return connector


@pytest.fixture
def databricks_test_connector() -> DatabricksConnector:
    connector = DatabricksConnector('databricks',
                                    host=DATABRICKS_HOST.parse_value(),
                                    cluster_id='0626-191119-75431625',
                                    warehouse_id='82d859e4a6b283a0',
                                    catalog='kumo_test_catalogue',
                                    schema_name='movielens')
    return connector


@pytest.fixture
def databricks_test_schema_connector() -> DatabricksConnector:
    return DatabricksConnector('databricks', host=DATABRICKS_HOST.parse_value(),
                               cluster_id='0626-191119-75431625',
                               warehouse_id='82d859e4a6b283a0',
                               catalog='hive_mapped_uc',
                               schema_name='test_schema')


@pytest.fixture
def databricks_otto_connector() -> DatabricksConnector:
    return DatabricksConnector('databricks', host=DATABRICKS_HOST.parse_value(),
                               cluster_id='0626-191119-75431625',
                               warehouse_id='82d859e4a6b283a0',
                               catalog='kumo_test_catalogue',
                               schema_name='otto_3k')
