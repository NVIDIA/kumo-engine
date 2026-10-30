# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from unittest import mock

import kumoapi.data_source as data_source_api
import pytest

from io_utils import fsutil
from kumo.connector import bigquery_connector
from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.testing import bigquery as testing_bigquery
from kumo.testing.rest import (
    bigquery_test_identity,
    invalid_bigquery_key_test_identity_config,
)

_CREDENTIAL_ENV_VARS = (
    'GOOGLE_APPLICATION_CREDENTIALS',
    testing_bigquery.ENV_GOOGLE_CREDENTIALS_JSON,
    testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
    testing_bigquery.ENV_GCS_KUMO_BUCKET.name,
    'GCP_PROJECT_ID',
    'GOOGLE_CLOUD_PROJECT',
    'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED',
)


@pytest.fixture(autouse=True)
def credential_environment(monkeypatch):
    # Register even absent variables: the helper writes them without using
    # monkeypatch, and their original values must survive fixture teardown.
    for name in _CREDENTIAL_ENV_VARS:
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)


@pytest.fixture(params=['true', 'false'])
def wif_credentials(monkeypatch, request):
    monkeypatch.setenv('GITHUB_ACTIONS', request.param)
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    get_secrets = mock.Mock(side_effect=AssertionError(
        'Workload exchange must not read a stored service-account key'))
    monkeypatch.setattr(testing_bigquery, 'get_secrets', get_secrets)
    yield get_secrets
    get_secrets.assert_not_called()


@pytest.mark.parametrize('github_actions', ['true', 'false'])
@pytest.mark.parametrize('exchange_enabled', [None, 'false'])
def test_unconfigured_auth_preserves_fallback(
        monkeypatch, github_actions, exchange_enabled):
    monkeypatch.setenv('GITHUB_ACTIONS', github_actions)
    if exchange_enabled is not None:
        monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', exchange_enabled)

    assert not testing_bigquery._configure_application_default_credentials(
        testing_bigquery.DEFAULT_BIGQUERY_KEY)


@pytest.mark.parametrize('original_value', [None, 'original'])
def test_credential_environment_restores(monkeypatch, tmp_path, original_value):
    for name in _CREDENTIAL_ENV_VARS:
        if original_value is not None:
            monkeypatch.setenv(name, original_value)
    credentials_path = str(tmp_path / 'gha-creds.json')
    fsutil.write_text(credentials_path, '{"type": "external_account"}')

    with pytest.MonkeyPatch.context() as isolated:
        credential_environment.__wrapped__(isolated)
        isolated.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
        isolated.setenv('GOOGLE_APPLICATION_CREDENTIALS', credentials_path)
        isolated.setenv('GOOGLE_CLOUD_PROJECT', 'example-project')
        testing_bigquery.setup_bigquery_credentials()
        assert os.environ[testing_bigquery.ENV_GOOGLE_CREDENTIALS_JSON] == (
            credentials_path)

    for name in _CREDENTIAL_ENV_VARS:
        assert os.environ.get(name) == original_value


@pytest.mark.parametrize('missing_path', [False, True])
def test_wif_requires_adc(monkeypatch, tmp_path, wif_credentials, missing_path):
    if missing_path:
        monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS',
                           str(tmp_path / 'missing.json'))
    else:
        monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)

    with pytest.raises(RuntimeError, match='Federation ADC'):
        testing_bigquery.setup_bigquery_credentials()


@pytest.mark.parametrize('credential_type', [
    'service_account', 'authorized_user', 'impersonated_service_account',
])
def test_wif_rejects_nonfederated_adc(
        monkeypatch, tmp_path, wif_credentials, credential_type):
    credentials_path = str(tmp_path / 'adc.json')
    fsutil.write_text(credentials_path, '{"type": "' + credential_type + '"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', credentials_path)

    with pytest.raises(RuntimeError, match='external_account ADC'):
        testing_bigquery.setup_bigquery_credentials()


def test_wif_switches_impersonation(monkeypatch, tmp_path, wif_credentials):
    credentials_path = str(tmp_path / 'gha-creds.json')
    fsutil.write_text(credentials_path, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', credentials_path)
    targets = {
        testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY: (
            testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
            'no-table@example.test'),
        testing_bigquery.FAIL_PROJECT_PERMISSION_BIGQUERY_KEY: (
            testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
            'no-job@example.test'),
    }
    for env_name, target in targets.values():
        monkeypatch.setenv(env_name, target)

    for key in (None, *targets, None):
        testing_bigquery.setup_bigquery_credentials(key)
        assert os.environ.get(
            testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT) == (
                targets[key][1] if key else None)
        assert os.environ[bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON] == (
            credentials_path)
    assert fsutil.read_text(credentials_path) == '{"type": "external_account"}'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_wif_restores_impersonation(
        monkeypatch, tmp_path, wif_credentials, asynchronous):
    credentials_path = str(tmp_path / 'gha-creds.json')
    fsutil.write_text(credentials_path, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', credentials_path)
    target_env = testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT
    monkeypatch.setenv(target_env, 'previous@example.test')
    monkeypatch.setenv(testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
                       'no-job@example.test')

    def fail():
        assert os.environ[target_env] == 'no-job@example.test'
        raise ValueError('test failed')

    async def async_fail():
        fail()

    wrapped = testing_bigquery.withBigQueryCredentials(
        service_account_key=testing_bigquery.FAIL_PROJECT_PERMISSION_BIGQUERY_KEY
    )(async_fail if asynchronous else fail)
    with pytest.raises(ValueError, match='test failed'):
        if asynchronous:
            asyncio.run(wrapped())
        else:
            wrapped()

    assert os.environ[target_env] == 'previous@example.test'


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


@pytest.mark.parametrize(('key', 'target_env'), [
    (testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY,
     testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT),
    (testing_bigquery.FAIL_PROJECT_PERMISSION_BIGQUERY_KEY,
     testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT),
])
def test_restricted_identity_requires_target_variable(
        monkeypatch, tmp_path, wif_credentials, key, target_env):
    credentials_path = tmp_path / "gha-creds.json"
    credentials_path.write_text('{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', str(credentials_path))
    monkeypatch.delenv(target_env, raising=False)

    with pytest.raises(RuntimeError, match=target_env):
        testing_bigquery.setup_bigquery_credentials(key)


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
