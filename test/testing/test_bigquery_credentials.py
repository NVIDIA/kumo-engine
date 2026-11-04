# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest import mock

import kumoapi.data_source as data_source_api
import pytest

from kumo.connector import bigquery_connector
from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.testing import bigquery as testing_bigquery
from kumo.testing.rest import (
    bigquery_test_identity,
    invalid_bigquery_key_test_identity_config,
)


def test_bigquery_rest_identity_uses_workload_exchange(monkeypatch):
    service_account = (
        'gha-kumo-gcp-tests@example-project.iam.gserviceaccount.com')
    monkeypatch.setenv('GCP_TEST_SERVICE_ACCOUNT', service_account)

    config, credentials = bigquery_test_identity()

    assert credentials is None
    assert isinstance(config.authentication,
                      data_source_api.GoogleWorkloadIdentityAuthentication)
    assert config.authentication.service_account_email == service_account


def test_invalid_bigquery_identity_remains_key_backed():
    config = invalid_bigquery_key_test_identity_config()

    assert isinstance(config.authentication,
                      data_source_api.GoogleServiceAccountKeyAuthentication)


def test_setup_prefers_application_default_credentials(monkeypatch, tmp_path):
    credentials_path = tmp_path / "gha-creds.json"
    credentials_path.write_text('{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', str(credentials_path))
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'example-project')
    get_secrets = mock.Mock(side_effect=AssertionError(
        "Workload Identity must not read a stored service-account key"))
    monkeypatch.setattr(testing_bigquery, 'get_secrets', get_secrets)

    testing_bigquery.setup_bigquery_credentials()

    assert os.environ[bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON] == str(
        credentials_path)
    assert os.environ['GCP_PROJECT_ID'] == 'example-project'
    assert testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT not in (
        os.environ)
    get_secrets.assert_not_called()


def test_restricted_identity_uses_iam_impersonation(monkeypatch, tmp_path):
    credentials_path = tmp_path / "gha-creds.json"
    credentials_path.write_text('{"type": "external_account"}')
    target = 'gha-kumo-bq-no-table@example-project.iam.gserviceaccount.com'
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', str(credentials_path))
    monkeypatch.setenv(testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
                       target)

    testing_bigquery.setup_bigquery_credentials(
        testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY)

    assert os.environ[
        testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT] == target


def test_restricted_identity_requires_target_variable(monkeypatch, tmp_path):
    credentials_path = tmp_path / "gha-creds.json"
    credentials_path.write_text('{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', str(credentials_path))
    monkeypatch.delenv(
        testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
        raising=False,
    )

    with pytest.raises(RuntimeError, match='GCP_BIGQUERY_NO_TABLE'):
        testing_bigquery.setup_bigquery_credentials(
            testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY)


def test_connector_loads_external_account_and_impersonates(
        monkeypatch, tmp_path):
    credentials_path = tmp_path / "gha-creds.json"
    credentials_path.write_text('{"type": "external_account"}')
    target = 'gha-kumo-bq-no-job@example-project.iam.gserviceaccount.com'
    source_credential = mock.Mock(name='source_credential')
    target_credential = mock.Mock(name='target_credential')
    client = mock.Mock(name='bigquery_client')
    monkeypatch.setenv(bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON,
                       str(credentials_path))
    monkeypatch.setenv(
        bigquery_connector.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT, target)
    load_credentials = mock.Mock(return_value=(source_credential,
                                               'example-project'))
    impersonate = mock.Mock(return_value=target_credential)
    create_client = mock.Mock(return_value=client)
    monkeypatch.setattr(bigquery_connector, 'load_credentials_from_file',
                        load_credentials)
    monkeypatch.setattr(
        bigquery_connector.impersonated_credentials,
        'Credentials',
        impersonate,
    )
    monkeypatch.setattr(bigquery_connector, 'Client', create_client)
    connector = BigQueryConnector(
        name='bigquery',
        project_id='example-project',
        dataset_id='financial',
    )

    result = connector._use_credentials(
        'bigquery',
        mock.Mock(name='from_json'),
        mock.Mock(name='from_account_info'),
        store_credentials=True,
    )

    assert result is client
    load_credentials.assert_called_once_with(
        str(credentials_path),
        scopes=[bigquery_connector.GOOGLE_CLOUD_PLATFORM_SCOPE],
    )
    impersonate.assert_called_once_with(
        source_credentials=source_credential,
        target_principal=target,
        target_scopes=[bigquery_connector.GOOGLE_CLOUD_PLATFORM_SCOPE],
    )
    create_client.assert_called_once_with(
        project='example-project',
        credentials=target_credential,
        location='US',
    )
    assert connector._credentials == {
        'provider': 'bigquery',
        'credential': target_credential,
    }


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('service_account_key', [
    None,
    testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY,
    testing_bigquery.FAIL_PROJECT_PERMISSION_BIGQUERY_KEY,
])
def test_disabled_bigquery_skips(
    monkeypatch, tmp_path, asynchronous, service_account_key,
):
    from io_utils import fsutil

    credentials_path = str(tmp_path / 'adc.json')
    fsutil.write_text(credentials_path, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', credentials_path)
    monkeypatch.setenv('KUMO_RUN_GCP_TESTS', '0')
    monkeypatch.delenv(
        testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT, raising=False)
    monkeypatch.delenv(
        testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT, raising=False)
    setup = mock.Mock(wraps=testing_bigquery.setup_bigquery_credentials)
    monkeypatch.setattr(testing_bigquery, 'setup_bigquery_credentials', setup)
    body = mock.Mock()

    async def async_body():
        body()

    decorated = testing_bigquery.withBigQueryCredentials(
        async_body if asynchronous else body,
        service_account_key=service_account_key,
    )
    with pytest.raises(pytest.skip.Exception, match='KUMO_RUN_GCP_TESTS=0'):
        if asynchronous:
            import asyncio

            asyncio.run(decorated())
        else:
            decorated()
    setup.assert_not_called()
    body.assert_not_called()


@pytest.mark.parametrize('run_gcp', [None, '1'])
@pytest.mark.parametrize('service_account_key, target_env', [
    (testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY,
     testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT),
    (testing_bigquery.FAIL_PROJECT_PERMISSION_BIGQUERY_KEY,
     testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT),
])
def test_enabled_bigquery_requires_target(
    monkeypatch, tmp_path, run_gcp, service_account_key, target_env,
):
    from io_utils import fsutil

    credentials_path = str(tmp_path / 'adc.json')
    fsutil.write_text(credentials_path, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', credentials_path)
    monkeypatch.delenv(target_env, raising=False)
    if run_gcp is None:
        monkeypatch.delenv('KUMO_RUN_GCP_TESTS', raising=False)
    else:
        monkeypatch.setenv('KUMO_RUN_GCP_TESTS', run_gcp)
    body = mock.Mock()
    decorated = testing_bigquery.withBigQueryCredentials(
        body, service_account_key=service_account_key)

    with pytest.raises(RuntimeError, match=target_env):
        decorated()
    body.assert_not_called()


@pytest.mark.parametrize('run_gcp', [None, '1'])
def test_enabled_bigquery_runs(monkeypatch, run_gcp):
    if run_gcp is None:
        monkeypatch.delenv('KUMO_RUN_GCP_TESTS', raising=False)
    else:
        monkeypatch.setenv('KUMO_RUN_GCP_TESTS', run_gcp)
    monkeypatch.setenv(bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON,
                       '/unused/mock-adc.json')
    setup = mock.Mock()
    monkeypatch.setattr(testing_bigquery, 'setup_bigquery_credentials', setup)
    body = mock.Mock(return_value='ran')

    assert testing_bigquery.withBigQueryCredentials(body)() == 'ran'
    setup.assert_called_once_with(None)
    body.assert_called_once_with()
