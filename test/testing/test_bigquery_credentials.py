# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from unittest import mock

import kumoapi.data_source as data_source_api
import pytest

from io_utils import fsutil, secrets
from kumo.connector import bigquery_connector
from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.testing import bigquery as testing_bigquery
from kumo.testing.bigquery import BigQueryIdentity
from kumo.testing.rest import (
    bigquery_test_identity,
    invalid_bigquery_key_test_identity_config,
)

_ENV_NAMES = (
    *testing_bigquery._CREDENTIAL_ENV_VARS,
    'GOOGLE_CLOUD_PROJECT', 'KUMO_TEST_PROFILE',
    'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED',
    testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
    testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
    testing_bigquery.ENV_BIGQUERY_NO_TABLE_CREDENTIALS,
    testing_bigquery.ENV_BIGQUERY_NO_JOB_CREDENTIALS,
)
_RESTRICTED_IDENTITIES = (
    (BigQueryIdentity.NO_TABLE,
     testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
     testing_bigquery.ENV_BIGQUERY_NO_TABLE_CREDENTIALS),
    (BigQueryIdentity.NO_JOB,
     testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
     testing_bigquery.ENV_BIGQUERY_NO_JOB_CREDENTIALS),
)


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)
    secret_read = mock.Mock(side_effect=AssertionError('unexpected secret access'))
    monkeypatch.setattr(secrets, 'get_secrets', secret_read)
    monkeypatch.setattr(secrets.AWSSecretProvider, 'get_secret', secret_read)
    yield
    secret_read.assert_not_called()


def write_credentials(tmp_path, kind, filename='input-credentials.json'):
    path = str(tmp_path / filename)
    fsutil.write_text(path, json.dumps({'type': kind}))
    return path


@pytest.mark.parametrize('kind', ['external_account', 'authorized_user',
                                  'impersonated_service_account'])
@pytest.mark.parametrize(('identity', 'target_env', 'file_env'), _RESTRICTED_IDENTITIES)
def test_adc_identity_sequence(monkeypatch, tmp_path, kind, identity,
                               target_env, file_env):
    path = write_credentials(tmp_path, kind)
    original = fsutil.read_text(path)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_CREDENTIALS_JSON, 'original-input')
    monkeypatch.setenv(target_env, 'restricted@example.test')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'example-project')
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                       'original@example.test')
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    for selected in (BigQueryIdentity.DEFAULT, identity, BigQueryIdentity.DEFAULT):
        with testing_bigquery.setup_bigquery_credentials(selected):
            assert os.environ[bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON] == path
            assert os.environ['GCP_PROJECT_ID'] == 'example-project'
            assert os.environ.get(
                testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT) == (
                    None if selected == BigQueryIdentity.DEFAULT
                    else 'restricted@example.test')
        assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert fsutil.read_text(path) == original


@pytest.mark.parametrize(('identity', 'target_env', 'file_env'), _RESTRICTED_IDENTITIES)
def test_key_identity_sequence(monkeypatch, tmp_path, identity, target_env, file_env):
    path = write_credentials(tmp_path, 'service_account')
    restricted = write_credentials(tmp_path, 'service_account', 'restricted.json')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    monkeypatch.setenv(file_env, restricted)
    monkeypatch.setenv(target_env, 'must-not-impersonate@example.test')
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    original_files = {path: fsutil.read_text(path), restricted: fsutil.read_text(restricted)}
    for selected in (BigQueryIdentity.DEFAULT, identity, BigQueryIdentity.DEFAULT):
        with testing_bigquery.setup_bigquery_credentials(selected):
            expected = path if selected == BigQueryIdentity.DEFAULT else restricted
            assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == expected
            assert os.environ[bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON] == expected
            assert testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT not in os.environ
        assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert {path: fsutil.read_text(path) for path in original_files} == original_files


@pytest.mark.parametrize('kind', ['external_account', 'service_account'])
@pytest.mark.parametrize(('identity', 'target_env', 'file_env'), _RESTRICTED_IDENTITIES)
def test_restricted_requires_explicit_input(monkeypatch, tmp_path, kind, identity,
                                            target_env, file_env):
    path = write_credentials(tmp_path, kind)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    required = file_env if kind == 'service_account' else target_env
    with pytest.raises(RuntimeError, match=required):
        with testing_bigquery.setup_bigquery_credentials(identity):
            pytest.fail('missing identity must fail during setup')
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


@pytest.mark.parametrize('restricted_kind', [None, 'external_account'])
def test_restricted_key_requires_valid_file(monkeypatch, tmp_path, restricted_kind):
    path = write_credentials(tmp_path, 'service_account')
    restricted = str(tmp_path / 'restricted.json')
    if restricted_kind:
        fsutil.write_text(restricted, json.dumps({'type': restricted_kind}))
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    monkeypatch.setenv(testing_bigquery.ENV_BIGQUERY_NO_JOB_CREDENTIALS, restricted)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    message = 'service_account credentials' if restricted_kind else 'does not exist'
    with pytest.raises(RuntimeError, match=message):
        with testing_bigquery.setup_bigquery_credentials(BigQueryIdentity.NO_JOB):
            pytest.fail('invalid restricted key must fail during setup')
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


@pytest.mark.parametrize('contents', ['not-json', '[]', '{}',
                                      '{"type": "unsupported"}'])
def test_invalid_adc_fails(monkeypatch, tmp_path, contents):
    path = str(tmp_path / 'bad.json')
    fsutil.write_text(path, contents)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    with pytest.raises(RuntimeError, match='invalid or unsupported'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('invalid input must fail during setup')
    assert fsutil.read_text(path) == contents


@pytest.mark.parametrize('profile', [None, 'gcp', 'gcp-aws'])
def test_requires_explicit_adc(monkeypatch, tmp_path, profile):
    if profile:
        monkeypatch.setenv('KUMO_TEST_PROFILE', profile)
    path = write_credentials(tmp_path, 'external_account')
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_CREDENTIALS_JSON, path)
    with pytest.raises(RuntimeError, match='require explicit GOOGLE_APPLICATION_CREDENTIALS'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('other credential inputs must not activate BigQuery')


def test_missing_adc_file(monkeypatch, tmp_path):
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', str(tmp_path / 'missing'))
    with pytest.raises(RuntimeError, match='does not exist'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('missing file must fail')


@pytest.mark.parametrize('mode', ['gcp', 'gcp-aws', 'workload'])
@pytest.mark.parametrize('kind', ['service_account', 'authorized_user',
                                  'impersonated_service_account'])
def test_wif_rejects_nonfederated(monkeypatch, tmp_path, mode, kind):
    if mode == 'workload':
        monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    else:
        monkeypatch.setenv('KUMO_TEST_PROFILE', mode)
    path = write_credentials(tmp_path, kind)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    with pytest.raises(RuntimeError, match='external_account ADC'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('nonfederated input must fail')
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('outcome', ['pass', 'fail', 'skip'])
def test_explicit_scope_restores(monkeypatch, tmp_path, asynchronous, outcome):
    path = write_credentials(tmp_path, 'external_account')
    original = fsutil.read_text(path)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_CREDENTIALS_JSON, 'previous-input')
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    writes = mock.Mock(side_effect=AssertionError('input files must be read-only'))
    monkeypatch.setattr(fsutil, 'write_text', writes)

    def body():
        for name in testing_bigquery._CREDENTIAL_ENV_VARS:
            os.environ[name] = 'test mutation'
        if outcome == 'fail':
            raise ValueError('test failure')
        if outcome == 'skip':
            pytest.skip('intentional skip')

    async def invoke_async():
        with testing_bigquery.setup_bigquery_credentials():
            await asyncio.sleep(0)
            body()

    def invoke():
        if asynchronous:
            asyncio.run(invoke_async())
        else:
            with testing_bigquery.setup_bigquery_credentials():
                body()

    if outcome == 'pass':
        invoke()
    else:
        error = ValueError if outcome == 'fail' else pytest.skip.Exception
        with pytest.raises(error):
            invoke()
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert fsutil.read_text(path) == original
    writes.assert_not_called()


@pytest.mark.parametrize('profile', [None, 'gcp'])
def test_decorator_only_marks(monkeypatch, profile):
    if profile:
        monkeypatch.setenv('KUMO_TEST_PROFILE', profile)
    read = mock.Mock(side_effect=AssertionError('credential I/O during import'))
    monkeypatch.setattr(fsutil, 'read_text', read)

    def test_body():
        return 'test body'

    decorated = testing_bigquery.withBigQueryCredentials(test_body)
    assert decorated is test_body
    assert {mark.name for mark in decorated.pytestmark} == {
        'requires_credentials', 'bigquery_credentials'}
    read.assert_not_called()


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


def test_connector_loads_external_account_and_impersonates(
        monkeypatch, tmp_path):
    credentials_path = tmp_path / "gha-creds.json"
    fsutil.write_text(str(credentials_path), '{"type": "external_account"}')
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
