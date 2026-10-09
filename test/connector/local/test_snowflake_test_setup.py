# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from service_framework import deploy_env_config


@pytest.mark.parametrize('selection, credentials_available, expected_calls', [
    ('cloud', True, 1),
    ('cloud', False, 1),
    ('local', False, 0),
    ('collect', False, 0),
])
def test_pytest_loads_credentials_only_for_selected_cloud_tests(
        tmp_path, selection, credentials_available, expected_calls):
    test_file = tmp_path / 'test_cloud_setup.py'
    test_file.write_text('''
import os
import pytest
from kumo.testing import withSnowflakeCredentials

@pytest.fixture
def cloud_resource():
    assert os.environ['SNOWFLAKE_USER'] == 'kumo'
    assert os.environ['SNOWFLAKE_PRIVATE_KEY'] == 'fake-key'
    yield
    assert os.environ['SNOWFLAKE_PRIVATE_KEY'] == 'fake-key'

@withSnowflakeCredentials
def test_cloud(cloud_resource):
    pass

@pytest.mark.asyncio
@withSnowflakeCredentials
async def test_cloud_async(cloud_resource):
    pass

def test_local():
    assert 'SNOWFLAKE_PRIVATE_KEY' not in os.environ

@pytest.mark.skip(reason='An unrelated skip must not load credentials')
@withSnowflakeCredentials
def test_skipped():
    raise AssertionError('skipped test ran')
''')
    options = ['--collect-only'] if selection == 'collect' else [
        '-k', 'local' if selection == 'local' else 'not local'
    ]
    script = f'''
import os
from unittest.mock import patch
import pytest
import io_utils.secrets
for name in ('SNOWFLAKE_USER', 'SNOWFLAKE_PRIVATE_KEY'):
    os.environ.pop(name, None)
with patch.object(io_utils.secrets, 'get_secrets',
                  return_value={{'kumo_user_secretkey': 'fake-key'}},
                  side_effect={"None" if credentials_available else "RuntimeError('no credentials')"}) as load:
    result = pytest.main(['-p', 'test.conftest', '-p', 'asyncio',
                         '-o', 'addopts=', '-q', {str(test_file)!r}] + {options!r})
    assert result == 0
    assert load.call_count == {expected_calls}
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        'collect': '4 tests collected',
        'local': '1 passed',
        'cloud':
        '2 passed, 1 skipped' if credentials_available else '3 skipped',
    }[selection]
    assert expected in result.stdout


def test_environment_credentials_work_without_secret_access(monkeypatch):
    from kumo.testing import snowflake

    monkeypatch.setenv('SNOWFLAKE_USER', 'supplied-user')
    monkeypatch.setenv('SNOWFLAKE_PRIVATE_KEY', 'supplied-key')
    monkeypatch.setattr(snowflake, 'get_secrets',
                        Mock(side_effect=RuntimeError('no secret access')))

    snowflake.setup_snowflake_credentials()

    assert os.environ['SNOWFLAKE_USER'] == 'supplied-user'
    assert os.environ['SNOWFLAKE_PRIVATE_KEY'] == 'supplied-key'


@pytest.mark.parametrize('supply_connector', [False, True])
def test_session_loads_missing_credentials_at_call_time(
        monkeypatch, tmp_path, valid_fake_private_key, supply_connector):
    from kumo.testing import snowflake

    monkeypatch.setattr(deploy_env_config, '_deploy_env', None)
    monkeypatch.setattr(snowflake.Connector, '_DISABLE_INTERACTIVE_INPUT', True)
    for name in ('SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 'SNOWFLAKE_PRIVATE_KEY'):
        monkeypatch.setenv(name, '')
    monkeypatch.setenv('KUMO_HOME', str(tmp_path))
    load = Mock(return_value={'kumo_user_secretkey': valid_fake_private_key})
    monkeypatch.setattr(snowflake, 'get_secrets', load)
    session = Mock()

    def connect(connector):
        credentials = connector.get_user_pass_or_key(
            'test', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD',
            credentials=connector._credentials,
            env_var_private_key='SNOWFLAKE_PRIVATE_KEY')
        assert credentials['user'] == 'kumo'
        assert 'private_key' in credentials
        return session

    monkeypatch.setattr(snowflake, 'get_or_build_session', connect)
    connector = snowflake.SnowflakeConnector(
        'test', account='example', warehouse='test', database='test',
        schema_name='test') if supply_connector else None
    assert snowflake.setup_full_snowflake_session(
        tmp_path, snow_connector=connector, persist_dir=str(tmp_path)) is session
    load.assert_called_once()
    session.sql.assert_called_once_with("ALTER SESSION SET TIMEZONE = 'UTC'")


@pytest.mark.parametrize('use_environment', [False, True])
@pytest.mark.parametrize('auth_method', ['password', 'private_key'])
def test_supplied_session_keeps_credentials(monkeypatch, tmp_path,
                                            use_environment, auth_method):
    from kumo.testing import snowflake

    monkeypatch.setattr(deploy_env_config, '_deploy_env', None)
    for name in ('SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 'SNOWFLAKE_PRIVATE_KEY'):
        monkeypatch.setenv(name, '')
    monkeypatch.setenv('KUMO_HOME', str(tmp_path))
    load = Mock(side_effect=pytest.fail.Exception(
        'Supplied credentials must not require secret access'))
    monkeypatch.setattr(snowflake, 'get_secrets', load)
    connector = snowflake.SnowflakeConnector('test', account='example',
                                             warehouse='test', database='test',
                                             schema_name='test')
    credentials = {'user': 'other-user', auth_method: 'other-secret'}
    if use_environment:
        for key, value in credentials.items():
            monkeypatch.setenv(f'SNOWFLAKE_{key.upper()}', value)
    else:
        connector.set_credentials(credentials)
    session = Mock()
    connect = Mock(return_value=session)
    monkeypatch.setattr(snowflake, 'get_or_build_session', connect)

    assert snowflake.setup_full_snowflake_session(tmp_path,
                                                  snow_connector=connector,
                                                  persist_dir=str(tmp_path),
                                                  timezone='') is session
    connect.assert_called_once_with(connector)
    if use_environment:
        assert os.environ['SNOWFLAKE_USER'] == 'other-user'
        assert os.environ[f'SNOWFLAKE_{auth_method.upper()}'] == 'other-secret'
    else:
        assert connector._credentials == credentials
    load.assert_not_called()
