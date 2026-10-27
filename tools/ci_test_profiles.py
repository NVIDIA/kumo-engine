# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declare test ownership and launch pytest with scoped runtime credentials.

The registry imports only the standard library so collection can consult it
before importing application modules. Run the launcher from the checkout root
after installing dependencies; package installation is outside this boundary.
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

GCP_AWS_MODULE = 'test/rest/integ/test_bigquery_diskgraph_e2e.py'
GCP_MODULES = frozenset({
    'test/connector/test_bigquery_connector.py',
    'test/io_utils/fsutil/test_gcs.py',
    'test/rest/integ/test_gcs_e2e.py',
    'test/rest/integ/test_gcs_public_api_e2e.py',
    'test/rest/integ/test_bigquery_ingestion_diskgraph_e2e.py',
    GCP_AWS_MODULE,
})
GCP_NODES = frozenset({'test/rest/test_tables.py::test_validate_size'})
GCP_MIXED_MODULE = 'test/io_utils/fsutil/test_all.py'
MANIFEST_PREFIX = 'KUMO_TEST_PROFILE_MANIFEST='


@dataclass(frozen=True)
class TestProfile:
    id: str
    label: str
    targets: tuple[str, ...]
    selection: str
    backend: str
    capabilities: frozenset[str]
    environment: str


PROFILES = {
    # These two profiles describe selection during migration. Isolated runtime
    # execution is disabled until their other provider dependencies are split.
    'general': TestProfile(
        'general', 'non-GCP tests', ('test',), 'non-gcp', 'default',
        frozenset({'aws'}), 'azure-adls-tests'),
    'snowflake': TestProfile(
        'snowflake', 'Snowflake tests', ('test',), 'non-gcp', 'snowflake',
        frozenset({'aws', 'snowflake'}), 'gha-kumo-workflow-tests'),
    'gcp': TestProfile(
        'gcp', 'GCP tests',
        (*sorted(GCP_MODULES - {GCP_AWS_MODULE}), GCP_MIXED_MODULE,
         *sorted(GCP_NODES)),
        'gcp', 'local', frozenset({'gcp'}), 'gha-kumo-workflow-tests'),
    'gcp-aws': TestProfile(
        'gcp-aws', f'gcs-e2e-tests ({GCP_AWS_MODULE})', (GCP_AWS_MODULE,),
        'gcp-aws', 's3', frozenset({'gcp', 'aws'}),
        'gha-kumo-workflow-tests'),
}

# Explicit runtime inputs, not inherited prefixes: newly introduced secrets do
# not silently become available to every test process.
RUNTIME_ENV = frozenset({
    'PATH', 'PYTHONPATH', 'VIRTUAL_ENV', 'LD_LIBRARY_PATH', 'LANG', 'LC_ALL',
    'TZ', 'TERM', 'NO_COLOR', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
    'REQUESTS_CA_BUNDLE', 'CUDA_VISIBLE_DEVICES', 'CUDA', 'FULL_TEST',
    'DEPLOY_STAGE', 'DEPLOY_ENV_CONFIG_FILE', 'KUMO_TESTING',
    'TEMPORAL_SERVER', 'SPARK_LOCAL_IP',
    'SPARK_VERSION', 'DATA_SPARK_MEM', 'MAX_DATA_BACKEND_WORKER_PROCESSES',
    'KUMO_ENABLE_GLOVE', 'KUMO_GLOVE_PATH', 'ARTIFACTS_DIR', 'GLOG_logtostdout',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'PYARROW_IGNORE_TIMEZONE', 'PYTHONUNBUFFERED',
})
PROVIDER_ENV = {
    'gcp': frozenset({
        'GOOGLE_APPLICATION_CREDENTIALS', 'GCP_PROJECT_ID',
        'GOOGLE_CLOUD_PROJECT', 'GCP_TEST_SERVICE_ACCOUNT',
        'GCP_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT',
        'GCP_BIGQUERY_NO_JOB_SERVICE_ACCOUNT', 'GCS_TEST_BUCKET',
        'GCP_STORAGE_TRANSFER_BUCKET',
    }),
    'aws': frozenset({
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN',
        'AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_TEST_ROLE_ARN',
    }),
    'snowflake': frozenset({
        'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 'SNOWFLAKE_ACCOUNT',
        'SNOWFLAKE_PRIVATE_KEY', 'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE',
        'SNOWFLAKE_WAREHOUSE', 'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA',
        'SNOWFLAKE_ROLE',
    }),
    'azure': frozenset({
        'AZURE_CLIENT_ID', 'AZURE_TENANT_ID', 'AZURE_CLIENT_SECRET',
        'AZURE_STORAGE_ACCOUNT', 'AZURE_CONTAINER_NAME',
        'AZURE_STORAGE_KEY', 'AZURE_STORAGE_SAS_TOKEN', 'KUMO_RUN_ADLS_TESTS',
    }),
    'databricks': frozenset({
        'DATABRICKS_HOST', 'DATABRICKS_TOKEN', 'DATABRICKS_CLIENT_ID',
        'DATABRICKS_CLIENT_SECRET', 'DATABRICKS_WAREHOUSE_ID',
    }),
}
PYTEST_VALUE_OPTIONS = frozenset({
    '-k', '-m', '--maxfail', '--tb', '--durations', '--timeout',
    '--timeout-method', '--splits', '--group', '--splitting-algorithm',
    '--junitxml', '--capture', '--verbosity',
})


def profile_for(identifier: str) -> TestProfile:
    try:
        return PROFILES[identifier]
    except KeyError:
        raise ValueError(f'Unknown test profile: {identifier}') from None


def owns_node(profile: TestProfile, module: str, nodeid: str = '',
              capabilities: frozenset[str] = frozenset()) -> bool:
    gcp = (module in GCP_MODULES or nodeid.split('[', 1)[0] in GCP_NODES
           or 'gcp' in capabilities)
    if profile.selection == 'non-gcp':
        return not gcp
    if profile.selection == 'gcp-aws':
        return module == GCP_AWS_MODULE
    return gcp and module != GCP_AWS_MODULE


def permits_target(profile: TestProfile, target: str) -> bool:
    """Reject incompatible files and explicit nodes before importing them."""
    while target.startswith('./'):
        target = target[2:]
    if target in {'', '.'}:
        return True
    module = target.split('::', 1)[0].rstrip('/')
    if module.startswith('/') or '..' in module.split('/'):
        return False
    node = target.split('[', 1)[0]
    if module == GCP_MIXED_MODULE and '[' in target:
        parameters = target.split('[', 1)[1].rstrip(']').split('-')
        providers = set(parameters) & {'gcs', 's3', 'ucv', 'adls'}
        if providers:
            if profile.selection == 'non-gcp':
                return 'gcs' not in providers
            return profile.selection == 'gcp' and providers == {'gcs'}
    if profile.selection == 'non-gcp':
        return module not in GCP_MODULES and node not in GCP_NODES
    for owned in profile.targets:
        owned_module = owned.split('::', 1)[0]
        if owned_module.startswith(module + '/'):
            return True
        if module != owned_module:
            continue
        if '::' not in target or '::' not in owned:
            return True
        if node == owned or owned.startswith(node + '::'):
            return True
    return False


def integration_required(paths: Sequence[str]) -> bool:
    """Only known prose-only changes may avoid live integration execution."""
    if not paths:
        return True
    for path in paths:
        if not path or path.startswith('/') or '..' in path.split('/'):
            return True
        if path in {'README.md', 'CONTRIBUTING.md'}:
            continue
        if path.startswith('docs/') and path.endswith('.md'):
            continue
        return True
    return False


def runtime_environment(profile: TestProfile,
                        source: Mapping[str, str], root: str) -> dict[str, str]:
    allowed = set(RUNTIME_ENV)
    for capability in profile.capabilities:
        allowed.update(PROVIDER_ENV[capability])
    environment = {key: value for key, value in source.items() if key in allowed}
    environment.update({
        'HOME': os.path.join(root, 'home'),
        'TMPDIR': os.path.join(root, 'tmp'),
        'XDG_CONFIG_HOME': os.path.join(root, 'home', '.config'),
        'XDG_CACHE_HOME': os.path.join(root, 'home', '.cache'),
        'AWS_CONFIG_FILE': os.path.join(root, 'empty-aws-config'),
        'AWS_SHARED_CREDENTIALS_FILE': os.path.join(root, 'empty-aws-credentials'),
        'BOTO_CONFIG': os.path.join(root, 'empty-boto-config'),
        'AWS_EC2_METADATA_DISABLED': 'true',
        'KUMO_TEST_PROFILE': profile.id,
        'KUMO_RUN_GCP_TESTS': '1' if 'gcp' in profile.capabilities else '0',
        'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED': (
            'true' if 'gcp' in profile.capabilities else 'false'),
    })
    return environment


def pytest_arguments(arguments: Sequence[str]) -> list[str]:
    """Allow test selectors and reporting controls, not plugin/config bypasses."""
    flags = {'-q', '-v', '-vv', '-s', '-x', '--collect-only',
             '--disable-warnings', '--strict-markers'}
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in flags:
            result.append(argument)
        elif argument.split('=', 1)[0] in PYTEST_VALUE_OPTIONS:
            result.append(argument)
            if '=' not in argument:
                index += 1
                if index == len(arguments):
                    raise ValueError(f'Missing value for pytest option {argument}')
                result.append(arguments[index])
        elif argument.startswith('test/') and '..' not in argument.split('/'):
            result.append(argument)
        else:
            raise ValueError(f'Unsupported pytest argument: {argument}')
        index += 1
    return result


def explicit_targets(arguments: Sequence[str]) -> bool:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith('test/'):
            return True
        if argument in PYTEST_VALUE_OPTIONS:
            index += 1
        index += 1
    return False


def run_profile(profile: TestProfile, arguments: Sequence[str],
                manifest: str | None = None,
                coverage_file: str | None = None) -> int:
    extra = pytest_arguments(arguments)
    if profile.selection == 'non-gcp' and '--collect-only' not in extra:
        raise ValueError(
            f'Profile {profile.id} supports isolated collection only until '
            'Azure and Databricks credential dependencies are classified; '
            'run pytest directly with the profile plugin and existing '
            'credential configuration for execution')
    # Registry consumers run before application bootstrap; filesystem adapters
    # are needed only when executing a prepared test environment.
    from io_utils import fsutil

    with fsutil.TemporaryDirectory() as root:
        environment = runtime_environment(profile, os.environ, root)
        fsutil.makedirs(environment['HOME'], exist_ok=True)
        fsutil.makedirs(environment['TMPDIR'], exist_ok=True)
        for key in ('AWS_CONFIG_FILE', 'AWS_SHARED_CREDENTIALS_FILE',
                    'BOTO_CONFIG'):
            fsutil.write_text(environment[key], '')
        if 'gcp' in profile.capabilities and '--collect-only' not in extra:
            source = environment.get('GOOGLE_APPLICATION_CREDENTIALS')
            if not source:
                raise ValueError('GCP profiles require explicit '
                                 'GOOGLE_APPLICATION_CREDENTIALS')
            if '://' in source:
                raise ValueError('GCP profiles require a local ADC file')
            path = os.path.join(root, 'adc.json')
            fsutil.write_text(path, fsutil.read_text(os.path.abspath(source)))
            environment['GOOGLE_APPLICATION_CREDENTIALS'] = path
            environment['GOOGLE_CREDENTIALS_JSON'] = path
        else:
            environment.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
        if 'gcp' in profile.capabilities:
            config = json.loads(fsutil.read_text(
                'rest/devserver/configs/diskgraph-test.json'))
            config['secrets'] = {'backend': 'env_var', 'fallbacks': []}
            if 'aws' not in profile.capabilities:
                config['storage']['artifact_root'] = os.path.join(root, 'artifacts')
                fsutil.makedirs(config['storage']['artifact_root'], exist_ok=True)
            config_path = os.path.join(root, 'deploy.json')
            fsutil.write_text(config_path, json.dumps(config))
            environment['DEPLOY_ENV_CONFIG_FILE'] = config_path
        # Explicit targets replace the profile's default roots. The plugin
        # rejects targets belonging to a different profile before collection.
        has_targets = explicit_targets(extra)
        command = [sys.executable]
        if coverage_file:
            command.extend(['-m', 'coverage', 'run',
                            '--data-file=' + os.path.abspath(coverage_file)])
        command.extend(['-m', 'pytest', '-p',
                        'tools.pytest_test_profiles', '--test-profile', profile.id])
        if manifest:
            command.append('--test-profile-manifest')
        if profile.backend == 'snowflake':
            command.append('--runsnowflake')
        if not has_targets:
            command.extend(profile.targets)
        command.extend(extra)
        if not manifest:
            return subprocess.run(command, env=environment, check=False).returncode
        result, payload = collect_manifest(command, environment)
        if payload is not None:
            fsutil.write_text(os.path.abspath(manifest), payload + '\n')
        elif result == 0:
            raise ValueError('Pytest completed without a profile manifest')
        return result


def collect_manifest(command: list[str],
                     environment: dict[str, str]) -> tuple[int, str | None]:
    payload = None
    with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            if line.startswith(MANIFEST_PREFIX):
                payload = line[len(MANIFEST_PREFIX):].strip()
                document = json.loads(payload)
                if (not isinstance(document, dict)
                        or document.get('profile') != environment['KUMO_TEST_PROFILE']
                        or not isinstance(document.get('nodeids'), list)
                        or not all(isinstance(node, str)
                                   for node in document['nodeids'])):
                    raise ValueError('Invalid pytest profile manifest')
        return process.wait(), payload


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    run = subparsers.add_parser('run')
    run.add_argument('profile', choices=tuple(PROFILES))
    run.add_argument('--manifest')
    run.add_argument('--coverage-file')
    subparsers.add_parser('impact')
    options, extra = parser.parse_known_args(arguments)
    if options.command == 'impact':
        if extra:
            parser.error('impact does not accept arguments')
        print(json.dumps({'run_integrations': integration_required(
            sys.stdin.read().splitlines())}))
        return 0
    try:
        return run_profile(profile_for(options.profile), extra, options.manifest,
                           options.coverage_file)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == '__main__':
    sys.exit(main())
