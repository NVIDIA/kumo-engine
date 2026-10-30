# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
from unittest import mock

import pytest

from io_utils import fsutil
from kumo.testing import bigquery as testing_bigquery
from kumo.testing import gcs as testing_gcs

pytest_plugins = ['pytester']

_ENV_NAMES = (
    'KUMO_TEST_PROFILE', 'GOOGLE_APPLICATION_CREDENTIALS',
    'GOOGLE_CREDENTIALS_JSON', 'GOOGLE_CREDENTIALS_STR_JSON', 'GCS_TEST_BUCKET',
    'GCP_PROJECT_ID', 'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT',
    'GCP_BIGQUERY_NO_JOB_SERVICE_ACCOUNT', 'GCP_STORAGE_TRANSFER_BUCKET',
    'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'GCP_BIGQUERY_NO_JOB_CREDENTIALS',
)


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch, tmp_path):
    for name in _ENV_NAMES:
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)
    for module, attribute in (
            (testing_gcs, 'ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH'),
            (testing_gcs, 'ADC_DEFAULT_PATH')):
        monkeypatch.setattr(module, attribute, str(tmp_path / attribute))


def test_decorators_do_not_probe(monkeypatch):
    setup = mock.Mock(side_effect=AssertionError('setup during collection'))
    monkeypatch.setattr(testing_gcs, 'setup_gcs_credentials', setup)

    def test_body():
        return 'test body'

    def fixture_body():
        yield 'fixture body'

    assert testing_gcs.withGCSCredentials(test_body) is test_body
    testing_gcs.withGCSCredentials(fixture_body)
    setup.assert_not_called()


@pytest.mark.parametrize('kind', ['service_account', 'authorized_user'])
def test_profile_rejects_nonfederated(monkeypatch, tmp_path, kind):
    monkeypatch.setenv('KUMO_TEST_PROFILE', 'gcp')
    path = str(tmp_path / 'adc.json')
    fsutil.write_text(path, json.dumps({'type': kind}))
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    with pytest.raises(RuntimeError, match='external_account ADC'):
        with testing_gcs.setup_gcs_credentials():
            pytest.fail('profile must reject nonfederated ADC')
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


def test_profile_requires_explicit_adc(monkeypatch):
    monkeypatch.setenv('KUMO_TEST_PROFILE', 'gcp')
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', base64.b64encode(
        b'{"type":"service_account"}').decode())
    with pytest.raises(RuntimeError, match='GOOGLE_APPLICATION_CREDENTIALS'):
        with testing_gcs.setup_gcs_credentials():
            pytest.fail('profile must ignore ambient encoded keys')


@pytest.mark.parametrize('failure', [False, True])
def test_unprofiled_key_lifetime(monkeypatch, failure):
    encoded = base64.b64encode(json.dumps({
        'type': 'service_account', 'project_id': 'test-project'}).encode()).decode()
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', encoded)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    created = []

    def invoke():
        with testing_gcs.setup_gcs_credentials():
            path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            created.append(path)
            assert fsutil.exists(path)
            assert os.environ['GCP_PROJECT_ID'] == 'test-project'
            if failure:
                raise ValueError('test failure')

    if failure:
        with pytest.raises(ValueError, match='test failure'):
            invoke()
    else:
        invoke()
    assert len(created) == 1
    assert not fsutil.exists(created[0])
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('profile', [None, 'gcp'])
def test_profile_fixture_order(pytester, monkeypatch, tmp_path, failure, profile):
    path = str(tmp_path / 'adc.json')
    fsutil.write_text(path, '{"type":"external_account"}')
    if profile:
        monkeypatch.setenv('KUMO_TEST_PROFILE', profile)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
    monkeypatch.setenv('GOOGLE_CREDENTIALS_JSON', 'before-scope')
    monkeypatch.setenv('GCP_BIGQUERY_NO_JOB_SERVICE_ACCOUNT', 'no-job@example.test')
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    pytester.makeini('''
[pytest]
markers =
    requires_credentials: external capabilities
    bigquery_credentials: BigQuery identity
    gcs_credentials: GCS identity
''')
    pytester.makepyfile('''
import os
import pytest
from kumo.testing.bigquery import withBigQueryCredentials, BigQueryIdentity
from kumo.testing.gcs import withGCSCredentials

@pytest.fixture(scope='session')
def workers(configure_profile_credentials):
    assert os.environ['GOOGLE_CREDENTIALS_JSON'] == os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    assert 'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT' not in os.environ
    yield
    assert 'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT' not in os.environ

@pytest.fixture
def cloud_resource(workers):
    assert os.environ['GOOGLE_IMPERSONATE_SERVICE_ACCOUNT'] == 'no-job@example.test'
    yield
    assert os.environ['GOOGLE_IMPERSONATE_SERVICE_ACCOUNT'] == 'no-job@example.test'

@withBigQueryCredentials(identity=BigQueryIdentity.NO_JOB)
def test_restricted(cloud_resource):
    os.environ['GOOGLE_CREDENTIALS_JSON'] = 'test mutation'
    ''' + ("raise ValueError('expected failure')" if failure else "pass") + '''

@withGCSCredentials
def test_gcs(workers):
    assert os.environ['GOOGLE_CREDENTIALS_JSON'] == os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    assert 'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT' not in os.environ
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', plugins=[testing_bigquery, testing_gcs])
    result.assert_outcomes(passed=1 if failure else 2, failed=1 if failure else 0)
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert fsutil.read_text(path) == '{"type":"external_account"}'


def test_profile_missing_adc_before_resources(pytester, monkeypatch):
    monkeypatch.setenv('KUMO_TEST_PROFILE', 'gcp')
    pytester.makepyfile('''
import pytest
@pytest.fixture(scope='session')
def workers(configure_profile_credentials):
    raise AssertionError('resource fixture must not run')
def test_cloud(workers):
    pass
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', plugins=[testing_bigquery])
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(['*require explicit*GOOGLE_APPLICATION_CREDENTIALS*'])


@pytest.mark.parametrize('source', ['encoded', 'home'])
def test_unprofiled_session_credentials(pytester, monkeypatch, tmp_path, source):
    payload = '{"type":"service_account","project_id":"test-project"}'
    if source == 'encoded':
        monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON',
                           base64.b64encode(payload.encode()).decode())
    else:
        fsutil.write_text(testing_gcs.ADC_DEFAULT_PATH, payload)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    pytester.makeini('''
[pytest]
markers =
    requires_credentials: external capabilities
    gcs_credentials: GCS identity
''')
    record = str(tmp_path / 'worker-input-path')
    monkeypatch.setenv('GCS_TEST_WORKER_INPUT', record)
    pytester.makepyfile('''
import os
import pytest
from io_utils import fsutil
from kumo.testing.gcs import withGCSCredentials

@pytest.fixture(scope='session')
def workers(configure_gcs_session):
    path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    assert fsutil.exists(path)
    fsutil.write_text(os.environ['GCS_TEST_WORKER_INPUT'], path)
    yield path
    # The input inherited when workers started still exists at shutdown.
    assert fsutil.exists(path)

@withGCSCredentials
def test_first(workers):
    assert fsutil.exists(workers)

@withGCSCredentials
def test_second(workers):
    assert fsutil.exists(workers)
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', plugins=[testing_bigquery, testing_gcs])
    result.assert_outcomes(passed=2)
    path = fsutil.read_text(record)
    assert fsutil.exists(path) == (source == 'home')
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before


def test_unprofiled_missing_inputs_skip_workers(pytester):
    pytester.makeini('''
[pytest]
markers =
    requires_credentials: external capabilities
    gcs_credentials: GCS identity
''')
    pytester.makepyfile('''
import pytest
from kumo.testing.gcs import withGCSCredentials

@pytest.fixture(scope='session')
def workers():
    raise AssertionError('worker must not start without credentials')

@withGCSCredentials
def test_cloud(workers):
    pass

def test_local():
    pass
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', plugins=[testing_bigquery, testing_gcs])
    result.assert_outcomes(passed=1, skipped=1)


def test_local_key_fixture_order(pytester, monkeypatch, tmp_path):
    default = str(tmp_path / 'default.json')
    restricted = str(tmp_path / 'restricted.json')
    fsutil.write_text(default, '{"type":"service_account"}')
    fsutil.write_text(restricted, '{"type":"service_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', default)
    monkeypatch.setenv('GCP_BIGQUERY_NO_JOB_CREDENTIALS', restricted)
    before = {name: os.environ.get(name) for name in _ENV_NAMES}
    pytester.makeini('''
[pytest]
markers =
    requires_credentials: external capabilities
    bigquery_credentials: BigQuery identity
''')
    pytester.makepyfile('''
import os
import pytest
from kumo.testing.bigquery import BigQueryIdentity, withBigQueryCredentials

@pytest.fixture(scope='session')
def workers(configure_profile_credentials):
    base = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    assert base != os.environ['GCP_BIGQUERY_NO_JOB_CREDENTIALS']
    yield base
    assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == base

@pytest.fixture
def resource(workers):
    assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == os.environ['GCP_BIGQUERY_NO_JOB_CREDENTIALS']
    assert 'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT' not in os.environ
    yield
    assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == os.environ['GCP_BIGQUERY_NO_JOB_CREDENTIALS']

@withBigQueryCredentials(identity=BigQueryIdentity.NO_JOB)
def test_restricted(resource):
    pass

@withBigQueryCredentials
def test_default(workers):
    assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == workers
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', plugins=[testing_bigquery])
    result.assert_outcomes(passed=2)
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == before
    assert fsutil.read_text(default) == '{"type":"service_account"}'
    assert fsutil.read_text(restricted) == '{"type":"service_account"}'


def test_local_bigquery_missing_input(pytester):
    pytester.makeini('''
[pytest]
markers =
    requires_credentials: external capabilities
    bigquery_credentials: BigQuery identity
''')
    pytester.makepyfile('''
import pytest
from kumo.testing.bigquery import withBigQueryCredentials

@pytest.fixture(scope='session')
def workers(configure_profile_credentials):
    raise AssertionError('worker must not start without explicit credentials')

@withBigQueryCredentials
def test_bigquery(workers):
    pass
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', plugins=[testing_bigquery])
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(['*require explicit*GOOGLE_APPLICATION_CREDENTIALS*'])
