# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from kumo_core.types import ColStatType
from kumoapi.data_source import DEFAULT_PROJECT_ID
from kumoapi.json_serde import to_json_dict
from kumoapi.typing import Stype

from kumo.stats.stats import Stats
from kumo.testing.bigquery import withBigQueryCredentials
from kumo.testing.rest import (
    bigquery_test_identity,
    create_identity_backed_connector,
    get_and_expect_success,
    post_and_expect_success,
)
from kumo.validation.validation import ValidationResponse
from rest.config import table
from rest.config.connector import BigQueryConnectorResourceConfig
from rest.config.status import IDResponse
from rest.config.table import ColStatsResponse, ColumnStats, ColumnSummary
from rest.config.table_v2 import TableDefinition
from rest.driver.error_constants import TABLE_SIZE_EXCEEDS_LIMIT_TITLE
from rest.driver.rest_api_driver import RestApiDriver
from rest.testing.config import TestFunctionCleanupOptions
from statedb.config.common import TableName
from statedb.db.identities import ConnectorService


@pytest.fixture(scope="function")
def test_cleanup_options() -> TestFunctionCleanupOptions:
    return TestFunctionCleanupOptions(
        should_reset_data_backend_cache=True,
        should_delete_data_persist_dir=True,
    )


def test_column_stats_config(client: TestClient, ):
    # Basic test
    time_now = datetime.datetime.now()
    time_tomorrow = time_now + datetime.timedelta(days=1)
    valid_stats_dict = {
        ColStatType.NUM_NA:
        Stats(value=50, approx=False, data_version=0),
        ColStatType.NUM_UNIQUE:
        Stats(value=42, approx=False, data_version=0),
        ColStatType.MIN:
        Stats(value=pd.to_datetime(time_now), approx=False, data_version=0),
        ColStatType.MAX:
        Stats(value=pd.to_datetime(time_tomorrow), approx=False,
              data_version=0),
        ColStatType.COUNT:
        Stats(value=100, approx=False, data_version=0),
        ColStatType.INVALID_FRACTION:
        Stats(value=0.0, approx=False, data_version=0)
    }
    col_stats = ColumnStats.from_stats_dict(Stype.timestamp,
                                            stats_dict=valid_stats_dict)
    assert ('Range', '1 days') in col_stats.summary_stats

    # NA time test
    invalid_stats_dict = valid_stats_dict
    invalid_stats_dict[ColStatType.MIN] = Stats(value=pd.NaT, approx=False,
                                                data_version=0)
    invalid_stats_dict[ColStatType.MAX] = Stats(value=pd.NaT, approx=False,
                                                data_version=0)
    col_stats = ColumnStats.from_stats_dict(Stype.timestamp,
                                            stats_dict=invalid_stats_dict)
    assert ('Min', 'NA') in col_stats.summary_stats
    assert ('Max', 'NA') in col_stats.summary_stats
    assert ('Range', 'NA') in col_stats.summary_stats

    # Sequence test
    valid_stats_dict = {
        ColStatType.SEQUENCE_MIN_LENGTH:
        Stats(value=42, approx=False, data_version=0),
        ColStatType.SEQUENCE_MAX_LENGTH:
        Stats(value=50, approx=False, data_version=0),
        ColStatType.COUNT:
        Stats(value=100, approx=False, data_version=0),
        ColStatType.INVALID_FRACTION:
        Stats(value=0.0, approx=False, data_version=0),
        ColStatType.NUM_NA:
        Stats(value=50, approx=False, data_version=0),
    }
    col_stats = ColumnStats.from_stats_dict(Stype.sequence,
                                            stats_dict=valid_stats_dict)
    assert ('Max Length', 50) in col_stats.summary_stats
    assert ('Min Length', 42) in col_stats.summary_stats


def test_missing_percentage_for_categorical_exact():
    valid_stats_dict = {
        ColStatType.NUM_NA:
        Stats(value=35, approx=False, data_version=0),
        ColStatType.COUNT:
        Stats(value=100, approx=False, data_version=0),
        ColStatType.NUM_UNIQUE:
        Stats(value=10, approx=False, data_version=0),
        ColStatType.CATEGORY_COUNTS:
        Stats(value=[('c1', 12), ('c2', 24), ('c3', 64)], approx=False,
              data_version=0),
        ColStatType.INVALID_FRACTION:
        Stats(
            value=0.2,
            approx=False,
            data_version=0,
        ),
    }
    col_stats = ColumnStats.from_stats_dict(
        Stype.categorical,
        stats_dict=valid_stats_dict,
    )
    assert col_stats.counts.missing_percentage == 35


def test_missing_percentage_for_categorical_approx():
    valid_stats_dict = {
        ColStatType.NUM_NA:
        Stats(
            value=25,
            approx=True,
            data_version=0,
            num_entries=100,
        ),
        ColStatType.COUNT:
        Stats(value=100, approx=False, data_version=0),
        ColStatType.NUM_UNIQUE:
        Stats(value=10, approx=False, data_version=0),
        ColStatType.CATEGORY_COUNTS:
        Stats(value=[('c1', 12), ('c2', 24), ('c3', 64)], approx=False,
              data_version=0),
        ColStatType.INVALID_FRACTION:
        Stats(
            value=0.2,
            approx=False,
            data_version=0,
        ),
    }
    col_stats = ColumnStats.from_stats_dict(
        Stype.categorical,
        stats_dict=valid_stats_dict,
    )
    assert col_stats.counts.missing_percentage == 25


def _compare_stats(a: List[ColumnSummary], b: List[ColumnSummary]):
    assert len(a) == len(b)

    def sort_fn(x: ColumnSummary) -> str:
        return x.name

    a.sort(key=sort_fn)
    b.sort(key=sort_fn)
    for i in range(len(a)):
        expected_stat = a[i]
        calculated_stat = b[i]

        # Sample rows are random so we don't want to check them exactly
        assert len(expected_stat.sample_rows) == len(
            calculated_stat.sample_rows)
        expected_stat.sample_rows = []
        calculated_stat.sample_rows = []

        # Make sure any distribution_points roughly sum to 100%
        stats = calculated_stat.statistics
        if stats:
            # Date distributions are still under development
            stats.distribution_date = None
            points = stats.distribution_points
            if points:
                Y = [y for _, y in points]
                assert pytest.approx(sum(Y), 0.1) == 100

        # Do exact check
        assert expected_stat == calculated_stat


@withBigQueryCredentials
def test_validate_size(client: TestClient):
    endpoint = "/v1/tables/validate"

    # Use a deliberately tiny cap so this remains independent of mutable
    # external fixture-table sizes.
    os.environ['MAX_TABLE_SIZE_BYTES_BIGQUERY'] = '1'
    bigquery_args: Dict[str, Any] = {
        'project_id': 'kumo-engineering',
        'dataset_id': 'financial'
    }
    bigquery_resource = BigQueryConnectorResourceConfig(name='bq_financial',
                                                        location='US',
                                                        **bigquery_args)
    identity_config, credentials = bigquery_test_identity()
    setup = create_identity_backed_connector(
        client,
        identity_config=identity_config,
        credentials=credentials,
        connector_config=bigquery_resource,
    )
    connector_id = setup.connector_id
    project_id = setup.project_id
    cols = [
        table_api.Column("ACCOUNT_ID", table_api.Stype.ID, table_api.Dtype.int),
        table_api.Column("TRANS_ID", table_api.Stype.ID, table_api.Dtype.int)
    ]

    def validate(name: str, connector_id: str, should_succeed: bool,
                 error_message: str = '') -> str | None:
        table_args = table_api.TableDefinition(
            cols=cols, source_table=source_table_api.BigQuerySourceTable(
                bigquery_connector_id=connector_id, **bigquery_args,
                table_name=TableName(name)), pkey=None, time_col=None)
        response = post_and_expect_success(
            client,
            endpoint,
            table_args,
            ValidationResponse,
            project_id=project_id,
        )
        if not should_succeed:
            assert response.errors is not None
            assert len(response.errors) > 0
            for err in response.errors:
                if error_message in err.message:
                    return err.message
            assert False, f"Could not find '{error_message}'"
        else:
            assert response.errors is None

    err_msg = TABLE_SIZE_EXCEEDS_LIMIT_TITLE
    assert validate('TRANS', connector_id, False, err_msg) is not None


def _assert_no_leaked_internals(response_text: str) -> None:
    assert 'table_error_message' not in response_text
    assert 'tables_v2.py' not in response_text
    assert 'File "' not in response_text


def test_validate_table_malformed_body_returns_structured_422(
        client: TestClient):
    body = {'cols': 'notalist'}
    response = client.post('/v1/tables/validate',
                           params={'project_id': DEFAULT_PROJECT_ID}, json=body)

    assert response.status_code == 422
    errors = response.json()
    assert isinstance(errors, list) and len(errors) > 0
    assert any(err['type'] == 'list_type' and err['loc'][-1] == 'cols'
               for err in errors)
    _assert_no_leaked_internals(response.text)


def test_validate_name_missing_query_param_returns_422(client: TestClient):
    # The shared session client may carry default query params (e.g. a
    # ``project_id`` bound by ``bind_test_client_project`` in an earlier
    # test's fixtures); clear them so the request really omits the param.
    original_params = client.params
    client.params = {}
    try:
        response = client.get('/v1/tables/some_table/validate_name')
    finally:
        client.params = original_params

    assert response.status_code == 422
    errors = response.json()
    assert isinstance(errors, list) and len(errors) > 0
    assert any(err['type'] == 'missing' and err['loc'][-1] == 'project_id'
               for err in errors)
    _assert_no_leaked_internals(response.text)


def test_format_time():
    jan1 = datetime.datetime(2020, 1, 1)
    jan5 = datetime.datetime(2020, 1, 5)
    assert table._format_time(None) == "NA"
    assert table._format_time(jan1) == "2020-01-01"
    assert (table._format_time(datetime.datetime(2020, 1, 1,
                                                 12)) == "2020-01-01 12:00")

    assert table._format_time_range(1, 2) == "NA"
    assert table._format_time_range(None, jan1) == "NA"
    assert table._format_time_range(jan1, pd.NA) == "NA"
    assert table._format_time_range(jan1, pd.NaT) == "NA"
    assert table._format_time_range(jan1, jan5) == "4 days"

    t1 = pd.Timestamp(year=1900, month=1, day=1)
    t2 = pd.Timestamp(year=2200, month=1, day=1)
    # Case that would overflow if pd.Timestamp used internally -
    # 300*365 + leap days
    assert table._format_time_range(t1, t2) == '109,573 days'


@pytest.fixture
def skip_table_source_validation(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Keep persistence/SDK route tests independent of Temporal workers."""
    validation = AsyncMock()
    monkeypatch.setattr(
        'rest.driver.table_driver.TableDriver.'
        '_validate_new_table_for_state_service', validation)
    monkeypatch.setattr(
        'rest.driver.table_driver.TableDriver.get_col_stats_and_sample_rows',
        AsyncMock(return_value=ColStatsResponse(
            statistics_available=False,
            columns=[],
            approx=False,
            rows_used=0,
        )))
    return validation


def test_create_table_v2(
    client: TestClient,
    table_definition_v2: table_api.TableDefinition,
    skip_table_source_validation: AsyncMock,
):
    table_def = TableDefinition(**dataclasses.asdict(table_definition_v2))

    table_id = post_and_expect_success(
        client=client,
        path='/v1/tables/',
        payload=table_def,
        response_type=IDResponse,
        project_id=DEFAULT_PROJECT_ID,
        name_alias='transactions',
    )

    assert len(table_id.id) > 0
    assert table_id.id.startswith("NA.NA-") is True

    # A nameless table is hidden from the list the UI reads and cannot be
    # deleted, so the route refuses to create one.
    refused = client.post(
        '/v1/tables/',
        json=to_json_dict(table_definition_v2),
        params={'project_id': DEFAULT_PROJECT_ID},
        follow_redirects=True,
    )
    assert refused.status_code == 422, refused.text
    assert 'name_alias' in refused.text


@pytest.mark.asyncio
async def test_get_col_stats_v2(client: TestClient,
                                table_definition_v2: table_api.TableDefinition,
                                rest_api_driver: RestApiDriver,
                                skip_table_source_validation: AsyncMock):
    table_def = TableDefinition(**dataclasses.asdict(table_definition_v2))

    table_id = post_and_expect_success(
        client=client,
        path='/v1/tables/',
        payload=table_def,
        response_type=IDResponse,
        project_id=DEFAULT_PROJECT_ID,
        name_alias='transactions',
    )

    assert len(table_id.id) > 0

    response = get_and_expect_success(
        client=client,
        path=f"/v1/tables/{table_id.id}/col_stats",
        params={'project_id': DEFAULT_PROJECT_ID},
        response_type=ColStatsResponse,
    )
    assert not response.statistics_available
    assert response.approx is False
    assert response.rows_used == 0
    assert all(col.statistics is None for col in response.columns)


def test_table_to_sdk(
    client: TestClient,
    table_definition_v2: table_api.TableDefinition,
    skip_table_source_validation: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
):
    # Test with S3 table and connector
    table_def = TableDefinition(**dataclasses.asdict(table_definition_v2))

    table_id = post_and_expect_success(
        client=client,
        path='/v1/tables/',
        payload=table_def,
        response_type=IDResponse,
        project_id=DEFAULT_PROJECT_ID,
        name_alias='transactions',
    )

    assert len(table_id.id) > 0
    assert table_id.id.startswith("NA.NA-") is True

    # The shared HNM fixture is deliberately an internal local-file connector
    # so ingestion can run without cloud credentials. SDK replay only supports
    # public project connectors, so expose the fixture's safe name at that
    # boundary without exposing its internal root.
    async def visible_test_connector(self, project_id, connector_id, principal):
        connector = await self.get_internal(project_id, connector_id)
        return SimpleNamespace(id=connector.id, name=connector.name)

    monkeypatch.setattr(ConnectorService, 'source_table_connector_for_project',
                        visible_test_connector)

    table_sdk_script = get_and_expect_success(
        client=client,
        path=f"/v1/tables/{table_id.id}/sdk",
        params={'project_id': DEFAULT_PROJECT_ID},
        response_type=str,
    )
    table_sdk_script = json.loads(table_sdk_script)
    connector_id = str(table_definition_v2.source_table.connector_id)
    assert f'{connector_id}_connector' in table_sdk_script
    assert 'kumo.S3Connector.get_by_name' in table_sdk_script
    assert f'project="{DEFAULT_PROJECT_ID}"' in table_sdk_script
    assert 'root_dir=' not in table_sdk_script
    assert "transactions_config" in table_sdk_script
    assert "transactions = kumo.Table" in table_sdk_script


def test_table_to_sdk_missing_table_returns_404(client: TestClient):
    missing_id = 'NA.NA-' + '0' * 32

    response = client.get(f'/v1/tables/{missing_id}/sdk',
                          params={'project_id': DEFAULT_PROJECT_ID})

    assert response.status_code == 404
    assert 'does not exist' in response.json()['detail']
