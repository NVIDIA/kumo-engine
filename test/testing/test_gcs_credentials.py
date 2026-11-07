# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest import mock

import pytest

from io_utils import fsutil
from io_utils.fsutil import impl_gcs
from kumo.connector.bigquery_connector import ENV_GOOGLE_CREDENTIALS_JSON
from kumo.testing import gcs

pytest_plugins = ['pytester']


def test_decoration_never_configures(monkeypatch):
    def unexpected():
        pytest.fail('Collection must not access credentials')

    monkeypatch.setattr(gcs, 'setup_bigquery_credentials', unexpected)

    @gcs.withGCSCredentials
    def example():
        pass

    assert example.pytestmark[0].name == 'gcs_credentials'


def test_explicit_adc_and_cache(monkeypatch, tmp_path):
    adc = str(tmp_path / 'adc.json')
    fsutil.write_text(adc, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', adc)
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', 'invalid-base64!')
    monkeypatch.setenv(ENV_GOOGLE_CREDENTIALS_JSON, 'previous')
    monkeypatch.setenv(gcs.ENV_GCS_TEST_BUCKET, 'test-bucket')
    previous = impl_gcs.threadsafe_gcs
    client = mock.Mock()
    monkeypatch.setattr(impl_gcs.storage, 'Client', mock.Mock(return_value=client))

    with gcs.setup_gcs_credentials():
        assert os.environ[ENV_GOOGLE_CREDENTIALS_JSON] == adc
        assert impl_gcs.threadsafe_gcs is not previous
        assert impl_gcs.threadsafe_gcs.client() is client
        assert impl_gcs.threadsafe_gcs.client() is client
        impl_gcs.storage.Client.assert_called_once()
        os.environ[gcs.ENV_GCS_TEST_BUCKET] = 'changed'

    assert impl_gcs.threadsafe_gcs is previous
    assert os.environ[gcs.ENV_GCS_TEST_BUCKET] == 'test-bucket'
    assert os.environ[ENV_GOOGLE_CREDENTIALS_JSON] == 'previous'
    assert fsutil.read_text(adc) == '{"type": "external_account"}'


@pytest.mark.parametrize('failure', [
    'none', 'fixture', 'test', 'teardown', 'missing', 'skip',
])
def test_lifecycle(pytester, monkeypatch, failure):
    adc = str(pytester.path / 'adc.json')
    fsutil.write_text(adc, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', adc)
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', 'invalid-base64!')
    monkeypatch.setenv(ENV_GOOGLE_CREDENTIALS_JSON, 'original')
    monkeypatch.setenv('GCP_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT', 'restricted')
    monkeypatch.setenv('GOOGLE_IMPERSONATE_SERVICE_ACCOUNT', 'original-target')
    if failure == 'missing':
        monkeypatch.delenv('GCP_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT')
    before = dict(os.environ)
    pytester.makeconftest(f'''
import os
import pytest
from test.conftest import pytest_runtest_setup, pytest_runtest_teardown
from kumo.testing.bigquery import GCP_WORKERS_STARTED

def pytest_configure(config):
    config.addinivalue_line('markers', 'bigquery_credentials(identity): identity')

@pytest.fixture(scope='session')
def worker(request):
    assert os.environ['GOOGLE_CREDENTIALS_JSON'] == {adc!r}
    request.config.stash[GCP_WORKERS_STARTED] = True
    yield
    assert os.environ['GOOGLE_CREDENTIALS_JSON'] == {adc!r}

@pytest.fixture
def resource(worker):
    assert os.environ['GOOGLE_IMPERSONATE_SERVICE_ACCOUNT'] == 'restricted'
    if {failure!r} == 'fixture':
        raise RuntimeError('fixture failure')
    yield
    assert os.environ['GOOGLE_IMPERSONATE_SERVICE_ACCOUNT'] == 'restricted'
    if {failure!r} == 'teardown':
        raise RuntimeError('teardown failure')
''')
    pytester.makepyfile(f'''
import os
import pytest
from kumo.testing.bigquery import BigQueryIdentity, withBigQueryCredentials

@pytest.mark.skipif({failure!r} == 'skip', reason='Snowflake mode')
@withBigQueryCredentials(identity=BigQueryIdentity.NO_TABLE)
def test_live(resource):
    if {failure!r} == 'test':
        raise RuntimeError('test failure')


def test_next():
    expected = 'original-target' if {failure!r} == 'skip' else None
    assert os.environ.get('GOOGLE_IMPERSONATE_SERVICE_ACCOUNT') == expected
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', '--confcutdir', str(pytester.path))
    if failure in ('fixture', 'missing'):
        result.assert_outcomes(passed=1, errors=1)
    elif failure == 'teardown':
        result.assert_outcomes(passed=2, errors=1)
    elif failure == 'test':
        result.assert_outcomes(passed=1, failed=1)
    elif failure == 'skip':
        result.assert_outcomes(passed=1, skipped=1)
    else:
        result.assert_outcomes(passed=2)
    for variable in ('GOOGLE_APPLICATION_CREDENTIALS', ENV_GOOGLE_CREDENTIALS_JSON,
                     'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT', 'GCP_PROJECT_ID',
                     gcs.ENV_GCS_TEST_BUCKET):
        assert os.environ.get(variable) == before.get(variable)


@pytest.mark.parametrize('skipped', [False, True])
def test_existing_workers_rejected(pytester, monkeypatch, skipped):
    adc = str(pytester.path / 'adc.json')
    fsutil.write_text(adc, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', adc)
    pytester.makeconftest('''
import pytest
from test.conftest import pytest_runtest_setup, pytest_runtest_teardown
from kumo.testing.bigquery import GCP_WORKERS_STARTED

def pytest_configure(config):
    config.addinivalue_line('markers', 'gcs_credentials: identity')
    config.stash[GCP_WORKERS_STARTED] = True
''')
    pytester.makepyfile(f'''
import pytest
from kumo.testing.gcs import withGCSCredentials

@pytest.mark.skipif({skipped!r}, reason='Snowflake')
@withGCSCredentials
def test_late_gcp():
    pass
''')
    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', '--confcutdir', str(pytester.path))
    if skipped:
        result.assert_outcomes(skipped=1)
    else:
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(['*separate pytest invocation*'])


def test_mixed_filesystem_marker(pytester):
    pytester.makeconftest('''
def pytest_configure(config):
    config.addinivalue_line('markers', 'gcs_credentials: identity')

def pytest_collection_modifyitems(items):
    marked = [item for item in items
              if item.get_closest_marker('gcs_credentials') is not None]
    assert len(marked) == 1
    assert marked[0].name == 'test_mixed[gcs]'
    assert len(items) == 4
''')
    pytester.makepyfile('''
from test.io_utils.fsutil.test_all import root

def test_mixed(root):
    pass
''')
    result = pytester.runpytest_inprocess(
        '--collect-only', '-q', '-o', 'addopts=',
        '--confcutdir', str(pytester.path))
    assert result.ret == 0
