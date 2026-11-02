# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real pytest collection/fixture ordering without cloud access."""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_pytest(tmp_path, source, group=None, strict=False, collect=False):
    test = tmp_path / 'test_selection.py'
    test.write_text(source)
    manifest = tmp_path / f'{group or "all"}.json'
    command = [
        sys.executable, '-m', 'pytest', '-c', '/dev/null', '--noconftest', '-p',
        'test.ci_resources', '-q', '--resource-manifest',
        str(manifest),
        str(test)
    ]
    if group:
        command += ['--resource-group', group]
    if strict:
        command += ['--require-resource-credentials']
    if collect:
        command += ['--collect-only']
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=45, env={
            **os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'
        })
    return result, json.loads(manifest.read_text())


def test_public_selection_never_resolves_private_fixture(tmp_path):
    source = '''
import pytest
import runpy
requires_resource = runpy.run_path('kumo/testing/_skip.py')['requires_resource']

@pytest.fixture
@requires_resource('gcp')
def private_input():
    raise AssertionError('Private fixture ran on a public runner')

@pytest.fixture
def indirect_input(private_input):
    return private_input

def test_local():
    pass

@pytest.mark.private_resource('aws')
@pytest.mark.ci_group('aws-public-api')
def test_balanced():
    pass

@pytest.mark.parametrize('storage_backend', [
    pytest.param('s3', marks=pytest.mark.private_resource('aws'))])
def test_live(indirect_input, storage_backend):
    pass

@pytest.mark.parametrize('storage_backend', [
    'local', pytest.param('s3', marks=pytest.mark.private_resource('aws'))])
def test_backend(storage_backend):
    assert storage_backend == 'local'
'''
    result, public = run_pytest(tmp_path, source, 'public')
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(public) == 2
    assert all(not value['resources'] for value in public.values())
    _, complete = run_pytest(tmp_path, source, collect=True)
    _, aws = run_pytest(tmp_path, source, 'aws', collect=True)
    _, gcp = run_pytest(tmp_path, source, 'gcp', collect=True)
    _, balanced = run_pytest(tmp_path, source, 'aws-public-api', collect=True)
    assert len(balanced) == 1
    assert set(complete) == set(public) | set(aws) | set(gcp) | set(balanced)
    assert sum(map(len, (public, aws, gcp, balanced))) == len(complete)
    assert not (set(public) & (set(aws) | set(gcp)))
    assert not (set(aws) & set(gcp))
    assert [row['resources'] for row in gcp.values()] == [['aws', 'gcp']]


def test_private_ci_rejects_missing_credentials_but_preserves_intentional_skips(
        tmp_path):
    source = '''
import pytest
@pytest.mark.private_resource('aws')
def test_missing():
    pytest.skip('No AWS credentials found for S3')
'''
    result, _ = run_pytest(tmp_path, source, 'aws', strict=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Private CI skipped tests' in result.stdout
    source = source.replace('No AWS credentials found for S3',
                            'Only run in the full suite')
    result, _ = run_pytest(tmp_path, source, 'aws', strict=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_discovery_runs_declared_profiles_in_fresh_processes(tmp_path):
    component = tmp_path / 'test' / 'new_component'
    component.mkdir(parents=True)
    helper = repr(str(Path('kumo/testing/_skip.py').resolve()))
    source = f'''
import os
import runpy
import pytest
from test.ci_resources import execution_profile
requires_resource = runpy.run_path({helper})['requires_resource']
IMPORTED_AUTH = {{key: os.environ.get(key) for key in (
    'DATABRICKS_TOKEN', 'DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET')}}

@pytest.fixture(scope='session')
@execution_profile('diskgraph')
def process_state():
    assert not os.path.exists(os.environ['PROFILE_PROCESS_STATE'])
    with open(os.environ['PROFILE_PROCESS_STATE'], 'w') as state:
        state.write(str(os.getpid()))
    yield
    os.unlink(os.environ['PROFILE_PROCESS_STATE'])

@requires_resource('databricks', profile='databricks-pat')
def test_pat(process_state):
    assert IMPORTED_AUTH == {{'DATABRICKS_TOKEN': 'fake-pat',
                            'DATABRICKS_CLIENT_ID': '',
                            'DATABRICKS_CLIENT_SECRET': ''}}

@requires_resource('databricks', profile='databricks-oauth')
def test_oauth(process_state):
    assert IMPORTED_AUTH == {{'DATABRICKS_TOKEN': '',
                            'DATABRICKS_CLIENT_ID': 'fake-client',
                            'DATABRICKS_CLIENT_SECRET': 'fake-secret'}}
'''
    for name in ('one', 'two'):
        (component / f'test_{name}.py').write_text(source)
    (component / 'test_gcp.py').write_text('''
import os
import pytest
IMPORTED_GCP = os.environ.get('IS_GCP_DEPLOY')

@pytest.mark.private_resource('gcp')
@pytest.mark.ci_profile('gcp')
@pytest.mark.parametrize('storage_backend', [
    pytest.param('s3', marks=pytest.mark.private_resource('aws'))])
def test_combined_identity(storage_backend):
    assert IMPORTED_GCP == '1'

@pytest.mark.private_resource('gcp')
@pytest.mark.ci_gate('gcp-e2e')
def test_gated():
    raise AssertionError('Disabled provider suite ran')
''')
    # Unsupported release/script entry points must not enter component discovery.
    (tmp_path / 'test' / 'test_script.py').write_text('raise RuntimeError()')
    release = tmp_path / 'test' / 'release_tests'
    release.mkdir()
    (release / 'test_release.py').write_text('raise RuntimeError()')
    env = {
        **os.environ,
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD':
        '1',
        'PYTHONPATH':
        os.pathsep.join([
            str(Path(__file__).resolve().parents[2]),
            os.environ.get('PYTHONPATH', '')
        ]),
        'PYTEST_ADDOPTS':
        '--noconftest -c /dev/null -p test.ci_resources -p pytest_timeout',
        'DATABRICKS_TOKEN':
        'fake-pat',
        'DATABRICKS_CLIENT_ID':
        'fake-client',
        'DATABRICKS_CLIENT_SECRET':
        'fake-secret',
        'IS_GCP_DEPLOY':
        '0',
        'CI_RUN_GCP_E2E':
        'false',
        'PROFILE_PROCESS_STATE':
        str(tmp_path / 'process-state'),
    }
    for group, sessions, cases in [('databricks', 4, 4), ('gcp', 1, 1)]:
        output = tmp_path / group
        result = subprocess.run(
            [sys.executable, '-m', 'tools.run_private_tests', group],
            cwd=tmp_path, env={
                **env, 'CI_TEST_RESULTS': str(output)
            }, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        discovered = json.loads((output / 'discovery.json').read_text())
        selections = [
            json.loads(path.read_text())
            for path in output.glob('*-selection.json')
        ]
        assert len(selections) == sessions
        assert len(discovered) == cases
        assert discovered == {
            key: value
            for part in selections
            for key, value in part.items()
        }
        assert len(list(output.glob('*.xml'))) == sessions
        if group == 'gcp':
            assert [row['resources']
                    for row in discovered.values()] == [['aws', 'gcp']]


def test_phase_timings_keep_only_safe_fields(tmp_path):
    source = """
def test_phase():
    print('CI_PHASE public-api provision elapsed_seconds=1.234https://example.test/data?X-Amz-Signature=sensitive')
"""
    result, _ = run_pytest(tmp_path, source, 'public')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CI_PHASE public-api provision elapsed_seconds=1.234' in result.stdout
    assert 'X-Amz-Signature' not in result.stdout


def test_private_output_redacts_download_queries(monkeypatch):
    from tools.run_private_tests import redact_download_urls
    monkeypatch.setenv('DATABRICKS_CLIENT_SECRET', 'synthetic-private-secret')
    text = 'download https://example.test/data?sig=sensitive&token=value end'
    assert redact_download_urls(
        text) == 'download https://example.test/data?REDACTED end'
    assert redact_download_urls(
        'secret synthetic-private-secret') == 'secret REDACTED'


def test_provider_fixture_changes_keep_e2e_enabled():
    from tools.ci_private_changes import selection
    matrix, gcp, _ = selection(
        ['test/rest/integ/test_cases/test_bigquery_e2e_fixtures.json'])
    assert gcp
    assert {'group': 'gcp-e2e'} in matrix['include']
    matrix, _, _ = selection(['docs/connector-test-matrix.md'])
    assert matrix == {'include': [{'group': 'none'}]}
    matrix, gcp, snowflake = selection(['README.md'], force=True)
    assert len(matrix['include']) == 7
    assert gcp and snowflake
