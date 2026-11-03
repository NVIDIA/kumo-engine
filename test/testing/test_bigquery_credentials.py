# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


@pytest.fixture
def explicit_adc(monkeypatch, tmp_path):
    path = str(tmp_path / 'adc.json')
    fsutil.write_text(path, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    monkeypatch.delenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', raising=False)
    return path


@pytest.mark.parametrize('kind', list(testing_bigquery.CredentialKind))
def test_explicit_types(monkeypatch, explicit_adc, kind):
    fsutil.write_text(explicit_adc, json.dumps({'type': kind.value}))
    monkeypatch.setenv(testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                       'previous')
    with testing_bigquery.setup_bigquery_credentials():
        assert os.environ[bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON] == explicit_adc
        assert testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT not in os.environ
    assert os.environ[testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT] == 'previous'


@pytest.mark.parametrize('identity, target_var, key_var', [
    (testing_bigquery.BigQueryIdentity.NO_TABLE,
     testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
     testing_bigquery.ENV_BIGQUERY_NO_TABLE_CREDENTIALS),
    (testing_bigquery.BigQueryIdentity.NO_JOB,
     testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
     testing_bigquery.ENV_BIGQUERY_NO_JOB_CREDENTIALS),
])
@pytest.mark.parametrize('kind', list(testing_bigquery.CredentialKind))
def test_restricted_identity(monkeypatch, tmp_path, explicit_adc, identity,
                             target_var, key_var, kind):
    fsutil.write_text(explicit_adc, json.dumps({'type': kind.value}))
    restricted = str(tmp_path / 'restricted.json')
    fsutil.write_text(restricted, '{"type": "service_account"}')
    monkeypatch.setenv(key_var, restricted)
    monkeypatch.setenv(target_var, 'restricted@example.com')
    with testing_bigquery.setup_bigquery_credentials(identity):
        if kind == testing_bigquery.CredentialKind.KEY:
            assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == restricted
            assert testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT not in os.environ
        else:
            assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == explicit_adc
            assert os.environ[testing_bigquery.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT] == 'restricted@example.com'
    assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == explicit_adc
    assert fsutil.read_text(restricted) == '{"type": "service_account"}'


@pytest.mark.parametrize('document', ['not-json', '[]', '{}', '{"type":"unknown"}'])
def test_invalid_adc(explicit_adc, document):
    fsutil.write_text(explicit_adc, document)
    with pytest.raises(RuntimeError, match='invalid|unsupported'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('Invalid credentials accepted')


@pytest.mark.parametrize('kind', ['service_account', 'authorized_user',
                                  'impersonated_service_account'])
def test_workload_requires_federation(monkeypatch, explicit_adc, kind):
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    fsutil.write_text(explicit_adc, json.dumps({'type': kind}))
    with pytest.raises(RuntimeError, match='external_account'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('Non-federated workload credentials accepted')


def test_missing_adc_fails(monkeypatch):
    monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', 'invalid-base64!')
    with pytest.raises(RuntimeError, match='explicit'):
        with testing_bigquery.setup_bigquery_credentials():
            pytest.fail('Implicit credentials accepted')


@pytest.mark.parametrize('failure', [RuntimeError, pytest.skip.Exception])
def test_environment_restored(monkeypatch, explicit_adc, failure):
    names = testing_bigquery._CREDENTIAL_ENV_VARS
    monkeypatch.setenv('GCP_PROJECT_ID', 'previous-project')
    monkeypatch.delenv(bigquery_connector.ENV_GOOGLE_CREDENTIALS_JSON, raising=False)
    before = {name: os.environ.get(name) for name in names}
    with pytest.raises(failure):
        with testing_bigquery.setup_bigquery_credentials():
            for name in names:
                os.environ[name] = 'changed-by-test'
            raise failure('test failure')
    assert {name: os.environ.get(name) for name in names} == before
    assert fsutil.read_text(explicit_adc) == '{"type": "external_account"}'


@pytest.mark.parametrize('identity', [testing_bigquery.BigQueryIdentity.NO_TABLE,
                                      testing_bigquery.BigQueryIdentity.NO_JOB])
def test_restricted_requires_input(monkeypatch, explicit_adc, identity):
    for variable in (testing_bigquery.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
                     testing_bigquery.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT):
        monkeypatch.delenv(variable, raising=False)
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match='restricted'):
        with testing_bigquery.setup_bigquery_credentials(identity):
            pytest.fail('Missing target accepted')
    assert dict(os.environ) == before


@pytest.mark.parametrize('path', ['s3://bucket/key.json', 'gs://bucket/key.json'])
def test_remote_adc_rejected(monkeypatch, path):
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    with mock.patch.object(fsutil, 'exists', side_effect=AssertionError('Remote IO')):
        with pytest.raises(RuntimeError, match='local ADC'):
            testing_bigquery.resolve_credentials(testing_bigquery.BigQueryIdentity.DEFAULT)


@pytest.mark.parametrize('document', [None, '{"type":"authorized_user"}'])
def test_restricted_key_required(monkeypatch, tmp_path, explicit_adc, document):
    fsutil.write_text(explicit_adc, '{"type":"service_account"}')
    variable = testing_bigquery.ENV_BIGQUERY_NO_TABLE_CREDENTIALS
    monkeypatch.delenv(variable, raising=False)
    if document is not None:
        restricted = str(tmp_path / 'restricted.json')
        fsutil.write_text(restricted, document)
        monkeypatch.setenv(variable, restricted)
    with pytest.raises(RuntimeError, match=variable):
        testing_bigquery.resolve_credentials(testing_bigquery.BigQueryIdentity.NO_TABLE)
