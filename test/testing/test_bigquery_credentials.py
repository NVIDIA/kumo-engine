# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
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

_ENV_NAMES = (
    *testing_bigquery._CREDENTIAL_ENV_VARS,
    'GOOGLE_CLOUD_PROJECT',
    testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
    testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
    'KUMO_TEST_PROFILE',
)
_RESTRICTED_KEYS = (
    (testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY,
     testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT),
    (testing_bigquery.FAIL_PROJECT_PERMISSION_BIGQUERY_KEY,
     testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT),
)


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch, tmp_path):
    for name in _ENV_NAMES:
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)
    monkeypatch.setattr(testing_bigquery, 'ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH',
                        str(tmp_path / 'absent-home-credentials.json'))
    # No focused test may reach a real secret provider.
    monkeypatch.setattr(testing_bigquery, 'get_secrets', mock.Mock(
        side_effect=AssertionError('unexpected secret access')))


def write_credentials(tmp_path, kind):
    path = str(tmp_path / 'input-credentials.json')
    fsutil.write_text(path, json.dumps({'type': kind}))
    return path


@pytest.mark.parametrize('kind', ['external_account', 'authorized_user',
                                  'impersonated_service_account'])
@pytest.mark.parametrize(('key', 'target_env'), _RESTRICTED_KEYS)
def test_adc_identity_sequence(monkeypatch, tmp_path, kind, key, target_env):
    path = write_credentials(tmp_path, kind)
    original = fsutil.read_text(path)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_CREDENTIALS_JSON,
                       'original-connector-input')
    monkeypatch.setenv(target_env, 'restricted@example.test')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'example-project')
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                       'original@example.test')
    before = {name: os.environ.get(name) for name in _ENV_NAMES}

    for selected in (None, key, None):
        with testing_bigquery.setup_bigquery_credentials(selected):
            assert os.environ[bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON] == path
            assert os.environ['GCP_PROJECT_ID'] == 'example-project'
            assert os.environ.get(
                testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT) == (
                    'restricted@example.test' if selected else None)
        assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert fsutil.read_text(path) == original
    testing_bigquery.get_secrets.assert_not_called()


@pytest.mark.parametrize(('key', 'target_env'), _RESTRICTED_KEYS)
def test_key_identity_sequence(monkeypatch, tmp_path, key, target_env):
    path = write_credentials(tmp_path, 'service_account')
    original = fsutil.read_text(path)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    # An unrelated WIF mode flag must not relabel an explicit stored key.
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    secrets = mock.Mock(return_value={'type': 'service_account', 'identity': key})
    monkeypatch.setattr(testing_bigquery, 'get_secrets', secrets)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    temporary_paths = []
    for selected in (None, key, None):
        with testing_bigquery.setup_bigquery_credentials(selected):
            configured = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            assert testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT not in os.environ
            if selected:
                assert configured != path
                assert json.loads(fsutil.read_text(configured))['identity'] == key
                temporary_paths.append(configured)
            else:
                assert configured == path
        assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    secrets.assert_called_once_with(key)
    assert all(not fsutil.exists(path) for path in temporary_paths)
    assert fsutil.read_text(path) == original


@pytest.mark.parametrize(('key', 'target_env'), _RESTRICTED_KEYS)
def test_adc_requires_restricted_target(monkeypatch, tmp_path, key, target_env):
    path = write_credentials(tmp_path, 'external_account')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    with pytest.raises(RuntimeError, match=target_env):
        with testing_bigquery.setup_bigquery_credentials(key):
            pytest.fail('missing identity must fail during setup')
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    testing_bigquery.get_secrets.assert_not_called()


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
    testing_bigquery.get_secrets.assert_not_called()


def test_missing_adc_fails(monkeypatch, tmp_path):
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS',
                       str(tmp_path / 'missing.json'))
    with pytest.raises(RuntimeError, match='does not exist'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('missing input must fail during setup')
    testing_bigquery.get_secrets.assert_not_called()


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('outcome', ['pass', 'fail', 'skip'])
@pytest.mark.parametrize('prior_values', [False, True])
def test_fallback_lifetime(monkeypatch, tmp_path, asynchronous, outcome,
                           prior_values):
    # Preserve a preexisting legacy home file without letting it select ADC.
    untouched = str(tmp_path / 'home-key.json')
    fsutil.write_text(untouched, 'leave unchanged')
    if prior_values:
        for name in (testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                     testing_bigquery.ENV_GCS_KUMO_BUCKET.name, 'GCP_PROJECT_ID'):
            monkeypatch.setenv(name, 'original')
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    secrets = mock.Mock(return_value={'type': 'service_account'})
    monkeypatch.setattr(testing_bigquery, 'get_secrets', secrets)
    created = []

    def body():
        path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        created.append(path)
        assert fsutil.exists(path)
        assert os.environ[bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON] == path
        for name in testing_bigquery._CREDENTIAL_ENV_VARS:
            os.environ[name] = 'test mutation'
        if outcome == 'fail':
            raise ValueError('test failed')
        if outcome == 'skip':
            pytest.skip('intentional skip')
        return 'result'

    async def async_body():
        return body()

    wrapped = testing_bigquery.withBigQueryCredentials(
        async_body if asynchronous else body)

    def invoke():
        return asyncio.run(wrapped()) if asynchronous else wrapped()

    if outcome == 'pass':
        assert invoke() == 'result'
    else:
        error = ValueError if outcome == 'fail' else pytest.skip.Exception
        with pytest.raises(error):
            invoke()
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert len(created) == 1
    assert not fsutil.exists(os.path.dirname(created[0]))
    assert fsutil.read_text(untouched) == 'leave unchanged'
    secrets.assert_called_once_with(testing_bigquery.DEFAULT_BIGQUERY_KEY)


def test_fallback_setup_failure_cleans(monkeypatch):
    secrets = mock.Mock(return_value={'type': 'unsupported'})
    monkeypatch.setattr(testing_bigquery, 'get_secrets', secrets)
    original_mkdtemp = fsutil.mkdtemp
    created = []

    def record_tempdir():
        path = original_mkdtemp()
        created.append(path)
        return path

    monkeypatch.setattr(fsutil, 'mkdtemp', record_tempdir)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    with pytest.raises(RuntimeError, match='unsupported'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('bad stored key must fail during setup')
    assert len(created) == 1
    assert not fsutil.exists(created[0])
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


def test_missing_credentials_skip(monkeypatch):
    monkeypatch.setattr(testing_bigquery, 'get_secrets', mock.Mock(
        side_effect=RuntimeError('secret unavailable')))
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    with pytest.raises(pytest.skip.Exception, match='No BigQuery credentials'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('missing credentials must skip')
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


def test_fallback_identity_sequence(monkeypatch):
    secrets = mock.Mock(side_effect=lambda key: {
        'type': 'service_account', 'identity': key})
    monkeypatch.setattr(testing_bigquery, 'get_secrets', secrets)
    keys = [testing_bigquery.DEFAULT_BIGQUERY_KEY,
            testing_bigquery.FAIL_TABLE_PERMISSION_BIGQUERY_KEY,
            testing_bigquery.DEFAULT_BIGQUERY_KEY]
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    paths = []
    for key in keys:
        with testing_bigquery.setup_bigquery_credentials(key):
            path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            paths.append(path)
            assert json.loads(fsutil.read_text(path))['identity'] == key
            assert testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT not in os.environ
        assert not fsutil.exists(path)
        assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert len(set(paths)) == len(keys)
    assert secrets.call_args_list == [mock.call(key) for key in keys]


@pytest.mark.parametrize('profile', ['gcp', 'gcp-aws'])
@pytest.mark.parametrize('kind', ['service_account', 'authorized_user',
                                  'impersonated_service_account'])
def test_profile_rejects_nonfederated(monkeypatch, tmp_path, profile, kind):
    monkeypatch.setenv('KUMO_TEST_PROFILE', profile)
    path = write_credentials(tmp_path, kind)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    with pytest.raises(RuntimeError, match='external_account ADC'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('nonfederated ADC must fail before test setup')
    testing_bigquery.get_secrets.assert_not_called()
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


@pytest.mark.parametrize('profile', ['gcp', 'gcp-aws'])
def test_profile_requires_explicit_adc(monkeypatch, tmp_path, profile):
    monkeypatch.setenv('KUMO_TEST_PROFILE', profile)
    path = write_credentials(tmp_path, 'external_account')
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_CREDENTIALS_JSON, path)
    monkeypatch.setattr(testing_bigquery, 'ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH',
                        path)
    with pytest.raises(RuntimeError, match='GOOGLE_APPLICATION_CREDENTIALS'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('ambient files must not activate profiled tests')
    testing_bigquery.get_secrets.assert_not_called()


def test_profile_decorator_only_marks(monkeypatch):
    monkeypatch.setenv('KUMO_TEST_PROFILE', 'gcp')
    read = mock.Mock(side_effect=AssertionError('credential I/O during import'))
    monkeypatch.setattr(fsutil, 'read_text', read)

    def test_body():
        return 'test body'

    decorated = testing_bigquery.withBigQueryCredentials(test_body)
    assert decorated is test_body
    assert {mark.name for mark in decorated.pytestmark} == {
        'requires_credentials', 'bigquery_credentials'}
    read.assert_not_called()
    testing_bigquery.get_secrets.assert_not_called()


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
