# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from unittest import mock

import pytest
from coverage import CoverageData

from io_utils import fsutil
from tools import ci_test_profiles
from tools.ci_test_profiles import (
    GCP_AWS_MODULE,
    GCP_MIXED_MODULE,
    GCP_MODULES,
    GCP_NODES,
    MANIFEST_PREFIX,
    collect_manifest,
    explicit_targets,
    integration_required,
    main,
    owns_node,
    profile_for,
    pytest_arguments,
    run_profile,
    runtime_environment,
)


def test_gcp_ownership():
    ordinary = profile_for('gcp')
    cross_provider = profile_for('gcp-aws')
    assert set(ordinary.targets).isdisjoint(cross_provider.targets)
    assert set(ordinary.targets) | set(cross_provider.targets) == (
        GCP_MODULES | GCP_NODES | {GCP_MIXED_MODULE})
    assert cross_provider.targets == (GCP_AWS_MODULE,)
    assert cross_provider.capabilities == {'gcp', 'aws'}
    assert ordinary.capabilities == {'gcp'}
    assert 'test/rest/integ/test_gcs_public_api_e2e.py' in ordinary.targets


def test_registry_import_has_no_application_side_effects():
    result = subprocess.run([
        sys.executable, '-c',
        'import sys; import tools.ci_test_profiles; '
        'assert not any(n == "io_utils" or n.startswith("kumo.") '
        'for n in sys.modules)',
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_environment_is_allowlisted():
    source = {
        'PATH': '/bin', 'GOOGLE_APPLICATION_CREDENTIALS': '/explicit/adc.json',
        'GCP_PROJECT_ID': 'project', 'GCP_STORAGE_TRANSFER_BUCKET': 'bucket',
        'AWS_ACCESS_KEY_ID': 'aws-secret', 'AWS_PROFILE': 'ambient',
        'AWS_CONTAINER_CREDENTIALS_FULL_URI': 'http://credential-service',
        'AZURE_CLIENT_SECRET': 'azure-secret', 'SNOWFLAKE_PASSWORD': 'password',
        'GITHUB_TOKEN': 'token', 'ACTIONS_ID_TOKEN_REQUEST_TOKEN': 'token',
        'GOOGLE_CREDENTIALS_STR_JSON': 'old-key', 'HOME': '/old/home',
        'DEPLOY_ENV_CONFIG_JSON': '{"secrets":{"backend":"aws_sm"}}',
        'PYTEST_ADDOPTS': '-p unsafe', 'NEW_PROVIDER_TOKEN': 'unlisted',
        'COVERAGE_FILE': '/ambient/coverage',
        'COVERAGE_PROCESS_START': '/ambient/config',
    }
    environment = runtime_environment(profile_for('gcp'), source, '/isolated')
    for key in source.keys() - {
            'PATH', 'GOOGLE_APPLICATION_CREDENTIALS', 'GCP_PROJECT_ID',
            'GCP_STORAGE_TRANSFER_BUCKET', 'HOME'}:
        assert key not in environment
    assert environment['HOME'] == '/isolated/home'
    assert environment['TMPDIR'] == '/isolated/tmp'
    assert environment['AWS_EC2_METADATA_DISABLED'] == 'true'
    assert environment['AWS_SHARED_CREDENTIALS_FILE'].startswith('/isolated/')
    assert environment['KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED'] == 'true'
    assert source['HOME'] == '/old/home'


def test_cross_provider_preserves_only_explicit_aws():
    source = {'AWS_ACCESS_KEY_ID': 'access', 'AWS_SECRET_ACCESS_KEY': 'secret',
              'AWS_SESSION_TOKEN': 'session', 'AWS_PROFILE': 'default',
              'AWS_TEST_ROLE_ARN': 'arn:aws:iam::123:role/explicit-test',
              'SNOWFLAKE_PASSWORD': 'snowflake'}
    environment = runtime_environment(profile_for('gcp-aws'), source, '/tmp/run')
    assert environment['AWS_ACCESS_KEY_ID'] == 'access'
    assert environment['AWS_SECRET_ACCESS_KEY'] == 'secret'
    assert environment['AWS_SESSION_TOKEN'] == 'session'
    assert environment['AWS_TEST_ROLE_ARN'] == source['AWS_TEST_ROLE_ARN']
    assert 'AWS_PROFILE' not in environment
    assert 'SNOWFLAKE_PASSWORD' not in environment


def test_general_does_not_receive_snowflake():
    source = {'AWS_ACCESS_KEY_ID': 'access', 'SNOWFLAKE_PASSWORD': 'secret'}
    general = runtime_environment(profile_for('general'), source, '/tmp/general')
    snowflake = runtime_environment(profile_for('snowflake'), source, '/tmp/snow')
    assert 'SNOWFLAKE_PASSWORD' not in general
    assert snowflake['SNOWFLAKE_PASSWORD'] == 'secret'
    assert general['AWS_ACCESS_KEY_ID'] == snowflake['AWS_ACCESS_KEY_ID']


def test_marked_parameters_have_one_owner():
    profiles = [profile_for(name) for name in ('general', 'gcp', 'gcp-aws')]
    for module, nodeid, required, owner in [
            (GCP_MIXED_MODULE, 'test[root_gcs]', frozenset({'gcp'}), 'gcp'),
            (GCP_MIXED_MODULE, 'test[root_local]', frozenset(), 'general'),
            (GCP_AWS_MODULE, 'test', frozenset({'gcp', 'aws'}), 'gcp-aws'),
            ('test/rest/test_tables.py', next(iter(GCP_NODES)),
             frozenset(), 'gcp')]:
        assert [profile.id for profile in profiles
                if owns_node(profile, module, nodeid, required)] == [owner]
    assert owns_node(profile_for('snowflake'), GCP_MIXED_MODULE)
    assert not owns_node(profile_for('snowflake'), GCP_AWS_MODULE)


@pytest.mark.parametrize('paths,required', [
    ([], True), (['docs/setup.md', 'README.md'], False),
    (['docs/example.py'], True), (['examples/README.md'], True),
    (['docs/../tools/runner.py'], True), (['uv.lock'], True),
    (['test/conftest.py'], True), (['.github/workflows/testing.yml'], True),
    (['unrecognized'], True), ([''], True),
])
def test_impact_is_conservative(paths, required):
    assert integration_required(paths) is required


@pytest.mark.parametrize('arguments', [
    ['-p', 'unsafe'], ['--noconftest'], ['--test-profile', 'non-gcp'],
    ['--override-ini', 'addopts=-p unsafe'], ['../other/test.py'],
    ['--unknown'], ['--timeout'],
])
def test_launcher_rejects_bypasses(arguments):
    with pytest.raises(ValueError):
        pytest_arguments(arguments)


def test_launcher_accepts_collection_controls():
    arguments = ['--collect-only', '-q', '-k', 'permission', '--tb=short',
                 '--capture=fd', '--verbosity=0',
                 'test/connector/test_bigquery_connector.py']
    assert pytest_arguments(arguments) == arguments
    with pytest.raises(ValueError, match='Unknown test profile'):
        profile_for('unknown')
    assert not explicit_targets(['-k', 'test/name'])
    assert explicit_targets(arguments)


@pytest.mark.parametrize('identifier', ['gcp', 'gcp-aws'])
def test_launch_scopes_config_and_credentials(monkeypatch, tmp_path, identifier):
    source = str(tmp_path / 'source.json')
    payload = '{"type":"external_account","audience":"test"}'
    fsutil.write_text(source, payload)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', source)
    monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', 'must-not-survive')
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'aws-access')
    created = []

    def execute(command, env):
        assert command[:4] == [sys.executable, '-m', 'pytest', '-p']
        assert '-q' in command
        assert command[command.index('--test-profile') + 1] == identifier
        assert '--test-profile-manifest' in command
        assert 'DEPLOY_ENV_CONFIG_JSON' not in env
        copied = env['GOOGLE_APPLICATION_CREDENTIALS']
        assert copied != source
        assert fsutil.read_text(copied) == payload
        config = json.loads(fsutil.read_text(env['DEPLOY_ENV_CONFIG_FILE']))
        assert config['secrets'] == {'backend': 'env_var', 'fallbacks': []}
        if identifier == 'gcp':
            assert not config['storage']['artifact_root'].startswith('s3:')
            assert 'AWS_ACCESS_KEY_ID' not in env
        else:
            assert config['storage']['artifact_root'].startswith('s3:')
            assert env['AWS_ACCESS_KEY_ID'] == 'aws-access'
        assert fsutil.read_text(env['AWS_CONFIG_FILE']) == ''
        created.append(os.path.dirname(copied))
        return 17, json.dumps({'profile': identifier, 'nodeids': ['test']})

    monkeypatch.setattr(ci_test_profiles, 'collect_manifest', execute)
    assert run_profile(profile_for(identifier), ['-q'],
                       str(tmp_path / 'manifest.json')) == 17
    assert len(created) == 1
    assert not fsutil.exists(created[0])
    assert fsutil.read_text(source) == payload
    assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == source
    assert json.loads(fsutil.read_text(str(tmp_path / 'manifest.json'))) == {
        'profile': identifier, 'nodeids': ['test']}


def test_manifest_stream_preserves_status(capsys):
    payload = json.dumps({'profile': 'gcp', 'nodeids': ['test::case']})
    command = [sys.executable, '-c',
               f'print("ordinary output"); print({MANIFEST_PREFIX + payload!r}); '
               'raise SystemExit(3)']
    result, captured = collect_manifest(command, {'KUMO_TEST_PROFILE': 'gcp'})
    assert result == 3
    assert captured == payload
    assert 'ordinary output' in capsys.readouterr().out


def test_wrong_profile_manifest_fails():
    payload = json.dumps({'profile': 'non-gcp', 'nodeids': []})
    with pytest.raises(ValueError, match='Invalid pytest profile manifest'):
        collect_manifest([sys.executable, '-c',
                          f'print({MANIFEST_PREFIX + payload!r})'],
                         {'KUMO_TEST_PROFILE': 'gcp'})


def test_missing_explicit_adc_never_launches(monkeypatch):
    monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
    execute = mock.Mock()
    monkeypatch.setattr(subprocess, 'run', execute)
    with pytest.raises(ValueError, match='require explicit'):
        run_profile(profile_for('gcp'), [])
    execute.assert_not_called()


def test_collection_needs_no_credentials(monkeypatch):
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', '/unreadable/input')

    def execute(command, *, env, check):
        assert '--collect-only' in command
        assert 'GOOGLE_APPLICATION_CREDENTIALS' not in env
        assert 'GOOGLE_CREDENTIALS_JSON' not in env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', execute)
    assert run_profile(profile_for('gcp'), ['--collect-only']) == 0


def test_remote_adc_is_rejected_before_read(monkeypatch):
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', 's3://private/key.json')
    reader = mock.Mock(side_effect=AssertionError('must not read remote ADC'))
    monkeypatch.setattr(fsutil, 'read_text', reader)
    with pytest.raises(ValueError, match='local ADC file'):
        run_profile(profile_for('gcp'), [])
    reader.assert_not_called()


@pytest.mark.parametrize('identifier', ['general', 'snowflake'],
                         ids=['general', 'warehouse'])
def test_unclassified_runtime_refuses_before_setup(monkeypatch, identifier):
    reader = mock.Mock(side_effect=AssertionError('must not read credentials'))
    execute = mock.Mock(side_effect=AssertionError('must not execute pytest'))
    temporary = mock.Mock(side_effect=AssertionError('must not create runtime'))
    monkeypatch.setattr(fsutil, 'read_text', reader)
    monkeypatch.setattr(fsutil, 'TemporaryDirectory', temporary)
    monkeypatch.setattr(subprocess, 'run', execute)
    with pytest.raises(ValueError, match='isolated collection only'):
        run_profile(profile_for(identifier), ['-q'])
    reader.assert_not_called()
    execute.assert_not_called()
    temporary.assert_not_called()


@pytest.mark.parametrize('identifier', ['general', 'snowflake'],
                         ids=['general', 'warehouse'])
def test_unclassified_collection_is_available(monkeypatch, identifier):
    def execute(command, *, env, check):
        assert '--collect-only' in command
        assert ('--runsnowflake' in command) == (identifier == 'snowflake')
        assert env['KUMO_TEST_PROFILE'] == identifier
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', execute)
    assert run_profile(profile_for(identifier), ['--collect-only']) == 0


def test_unknown_cli_option_fails_before_launch(monkeypatch):
    execute = mock.Mock()
    monkeypatch.setattr(subprocess, 'run', execute)
    with pytest.raises(SystemExit) as error:
        main(['run', 'gcp', '--unknown'])
    assert error.value.code == 2
    execute.assert_not_called()


def test_inner_child_produces_coverage(monkeypatch, tmp_path):
    root = str(tmp_path)
    config = fsutil.read_text('rest/devserver/configs/diskgraph-test.json')
    for directory in ('tools', 'test/connector', 'rest/devserver/configs'):
        fsutil.makedirs(os.path.join(root, directory), exist_ok=True)
    fsutil.write_text(os.path.join(root, 'tools/__init__.py'), '')
    for module in ('ci_test_profiles.py', 'pytest_test_profiles.py'):
        fsutil.write_text(os.path.join(root, 'tools', module),
                          fsutil.read_text('tools/' + module))
    fsutil.write_text(os.path.join(root, 'rest/devserver/configs/diskgraph-test.json'),
                      config)
    fsutil.write_text(os.path.join(root, 'example.py'), 'VALUE = 42\n')
    fsutil.write_text(os.path.join(root, '.coveragerc'),
                      '[run]\nsource = example\nbranch = True\n'
                      'concurrency = thread,multiprocessing\nrelative_files = True\n')
    fsutil.write_text(os.path.join(root, 'test/connector/test_bigquery_connector.py'),
                      'def test_measured():\n'
                      '    import example\n'
                      '    assert example.VALUE == 42\n')
    adc = os.path.join(root, 'adc.json')
    fsutil.write_text(adc, '{"type":"external_account"}')
    monkeypatch.chdir(root)
    monkeypatch.setenv('PYTHONPATH', root)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', adc)
    monkeypatch.setenv('COVERAGE_FILE', '/must-not-write/ambient')
    destination = os.path.join(root, '.coverage-profile')
    assert main(['run', 'gcp', '--coverage-file', destination,
                 'test/connector/test_bigquery_connector.py', '-q']) == 0
    files = fsutil.listdir(root, include=[destination + '*'])
    assert files
    measured = set()
    for path in files:
        data = CoverageData(basename=path)
        data.read()
        measured.update(data.measured_files())
    assert 'example.py' in measured
