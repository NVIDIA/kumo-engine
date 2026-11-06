# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from io_utils import fsutil
from kumo.connector.bigquery_connector import ENV_GOOGLE_CREDENTIALS_JSON
from kumo.testing import gcs

pytest_plugins = ['pytester']


def test_decoration_never_configures(monkeypatch):
    def unexpected():
        pytest.fail('Collection must not access credentials')

    monkeypatch.setattr(gcs, '_check_gcs_credentials', unexpected)

    @gcs.withGCSCredentials
    def example():
        pass

    assert example.pytestmark[0].name == 'gcs_credentials'


def test_explicit_adc_wins(monkeypatch, tmp_path):
    adc = str(tmp_path / 'adc.json')
    fsutil.write_text(adc, '{"type": "external_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', adc)
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', 'invalid-base64!')
    monkeypatch.setenv(ENV_GOOGLE_CREDENTIALS_JSON, 'previous')
    monkeypatch.setenv(gcs.ENV_GCS_TEST_BUCKET, 'test-bucket')

    gcs.require_gcs_credentials()

    assert os.environ[ENV_GOOGLE_CREDENTIALS_JSON] == adc
    assert fsutil.read_text(adc) == '{"type": "external_account"}'


@pytest.mark.parametrize('skipped', [False, True])
@pytest.mark.parametrize('available', [False, True])
def test_prerequisite_precedes_resources(pytester, monkeypatch, skipped,
                                         available):
    calls = []

    def credentials_available():
        calls.append('credentials')
        if available:
            monkeypatch.setenv('GCS_TEST_READY', '1')
        return available

    monkeypatch.delenv('GCS_TEST_READY', raising=False)
    monkeypatch.setattr(gcs, '_check_gcs_credentials', credentials_available)
    # Exercise the real hook in an isolated pytest session, including pytest's
    # built-in skip handling and session-resource ordering.
    pytester.makeconftest('''
import os
import pytest
from test.conftest import pytest_runtest_setup

def pytest_configure(config):
    config.addinivalue_line('markers', 'gcs_credentials: GCS prerequisite')

@pytest.fixture(scope='session')
def resource():
    assert os.environ.get('GCS_TEST_READY') == '1'
    yield
''')
    pytester.makepyfile(f'''
import pytest
from kumo.testing.gcs import withGCSCredentials

@pytest.mark.skipif({skipped!r}, reason='Snowflake mode')
@withGCSCredentials
def test_live(resource):
    pass
''')

    result = pytester.runpytest_inprocess(
        '-q', '-o', 'addopts=', '--confcutdir', str(pytester.path))

    if skipped:
        result.assert_outcomes(skipped=1)
        assert calls == []
    elif available:
        result.assert_outcomes(passed=1)
        assert calls == ['credentials']
    else:
        result.assert_outcomes(skipped=1)
        assert calls == ['credentials']
