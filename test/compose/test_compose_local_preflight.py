# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _internal_env(repo_root, tmp_path, variant=None):
    env_file = tmp_path / '.env'
    values = (repo_root / 'docker' / 'compose' /
              'env.internal-testing.example').read_text(encoding='utf-8')
    values = values.replace('POSTGRES_PASSWORD=REPLACE_WITH_RANDOM_PASSWORD',
                            'POSTGRES_PASSWORD=test-password')
    if variant is not None:
        values += f'\nKUMO_BACKEND_VARIANT={variant}\n'
    env_file.write_text(values, encoding='utf-8')
    env_file.chmod(0o600)
    return env_file


@pytest.mark.parametrize(
    ('file_variant', 'shell_variant', 'gpu_available', 'expected_mode',
     'file_slots'),
    [
        (None, None, True, 'gpu', None),
        ('gpu', None, True, 'gpu', None),
        ('gpu', 'cpu', False, 'cpu', None),
        ('gpu', None, False, None, None),
        ('gpu', None, True, 'gpu', '"3"'),
    ],
)
def test_internal_testing_entrypoint_resolves_backend_variant(
    tmp_path,
    file_variant,
    shell_variant,
    gpu_available,
    expected_mode,
    file_slots,
):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / 'docker' / 'compose' / 'kumo-compose'
    env_file = _internal_env(repo_root, tmp_path, file_variant)
    if file_slots is not None:
        with env_file.open('a', encoding='utf-8') as f:
            f.write(f'\nML_MAX_GPU_WORKER_SLOTS={file_slots}\n')

    docker = tmp_path / 'docker'
    docker.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = compose ] && [ "${2:-}" = version ]; then\n'
        '  exit 0\n'
        'fi\n'
        'if [ "$1" = info ]; then\n'
        f'  printf \'{json.dumps({"nvidia": {}} if gpu_available else {})}\\n\'\n'
        '  exit 0\n'
        'fi\n'
        'printf \'%s\\n\' "$@"\n'
        'echo "SLOTS=${ML_MAX_GPU_WORKER_SLOTS:-}"\n',
        encoding='utf-8',
    )
    nvidia_smi = tmp_path / 'nvidia-smi'
    nvidia_smi.write_text(
        '#!/bin/sh\n'
        'if [ "${1:-}" = -L ]; then printf \'GPU 0: A\\nGPU 1: B\\n\'; fi\n'
        f'exit {0 if gpu_available else 1}\n',
        encoding='utf-8')
    docker.chmod(0o755)
    nvidia_smi.chmod(0o755)

    env = os.environ.copy()
    env['PATH'] = f'{tmp_path}:{env["PATH"]}'
    env['KUMO_COMPOSE_ENV_FILE'] = str(env_file)
    env.pop('ML_MAX_GPU_WORKER_SLOTS', None)
    if shell_variant is None:
        env.pop('KUMO_BACKEND_VARIANT', None)
    else:
        env['KUMO_BACKEND_VARIANT'] = shell_variant
    result = subprocess.run(
        [script, 'config', '--quiet'],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    if expected_mode is None:
        assert result.returncode != 0
        assert 'GPU mode was requested' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert f'==> Backend mode: {expected_mode}' in result.stderr
        assert ('docker-compose.singlehost.gpu.yml'
                in result.stdout) is (expected_mode == 'gpu')
        assert 'docker-compose.singlehost.postgres.yml' in result.stdout
        detected = expected_mode == 'gpu' and file_slots is None
        assert f'SLOTS={"2" if detected else ""}\n' in result.stdout


def test_internal_testing_entrypoint_rejects_placeholder_password():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / 'docker' / 'compose' / 'kumo-compose'
    env_file = (repo_root / 'docker' / 'compose' /
                'env.internal-testing.example')

    env = os.environ.copy()
    env['KUMO_COMPOSE_ENV_FILE'] = str(env_file)
    result = subprocess.run(
        [script, 'config', '--quiet'],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert 'still contains one or more REPLACE_* placeholders' in result.stderr


def test_singlehost_entrypoint_works_without_local_override(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    compose_dir = tmp_path / 'compose'
    compose_dir.mkdir()
    for name in ('kumo-compose', 'docker-compose.singlehost.yml',
                 'docker-compose.singlehost.postgres.yml',
                 'docker-compose.singlehost.gpu.yml'):
        source = repo_root / 'docker' / 'compose' / name
        destination = compose_dir / name
        destination.write_bytes(source.read_bytes())
        destination.chmod(source.stat().st_mode)

    env_file = compose_dir / '.env'
    env_file.write_text(
        'POSTGRES_PASSWORD=test-password\nKUMO_POSTGRES_MODE=external\n',
        encoding='utf-8',
    )
    env_file.chmod(0o600)
    docker = tmp_path / 'docker'
    docker.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = compose ] && [ "${2:-}" = version ]; then exit 0; fi\n'
        'printf \'%s\\n\' "$@"\n',
        encoding='utf-8',
    )
    docker.chmod(0o755)

    env = os.environ.copy()
    env['PATH'] = f'{tmp_path}:{env["PATH"]}'
    env['KUMO_BACKEND_VARIANT'] = 'cpu'
    result = subprocess.run(
        [compose_dir / 'kumo-compose', 'config', '--quiet'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert 'docker-compose.singlehost.postgres.yml' not in result.stdout
    assert 'docker-compose.production-local.override.yml' not in result.stdout
    assert 'docker-compose.singlehost.gpu.yml' not in result.stdout


@pytest.mark.parametrize(('variant', 'expects_gpu'),
                         [('cpu', False), ('gpu', True)])
def test_singlehost_backend_variant_selects_image(tmp_path, variant,
                                                  expects_gpu):
    if shutil.which('docker') is None:
        pytest.skip('docker compose is unavailable')
    if subprocess.run(['docker', 'compose', 'version'], capture_output=True,
                      check=False).returncode:
        pytest.skip('docker compose is unavailable')

    repo_root = Path(__file__).resolve().parents[2]
    compose = repo_root / 'docker' / 'compose'
    env_file = _internal_env(repo_root, tmp_path)
    values = dict(
        line.split('=', 1)
        for line in env_file.read_text(encoding='utf-8').splitlines()
        if line and not line.startswith('#'))
    command = [
        'docker', 'compose', '--env-file', env_file,
        '-f', compose / 'docker-compose.singlehost.yml',
        '-f', compose / 'docker-compose.singlehost.postgres.yml',
    ]
    if expects_gpu:
        command += ['-f', compose / 'docker-compose.singlehost.gpu.yml']
    result = subprocess.run([*command, 'config', '--format', 'json'],
                            check=True, capture_output=True, text=True)
    worker = json.loads(result.stdout)['services']['kumo-pool-worker']
    devices = worker.get('deploy', {}).get('resources', {}).get(
        'reservations', {}).get('devices', [])
    expected_image = (f'{values["KUMO_IMAGE_REGISTRY"]}/kumo-backend-'
                      f'{variant}:{values["KUMO_IMAGE_TAG"]}')

    assert worker['image'] == expected_image
    assert bool(devices) is expects_gpu


@pytest.mark.parametrize(('slots', 'expected'), [(None, '1'), ('4', '4')])
def test_singlehost_gpu_slots(tmp_path, monkeypatch, slots, expected):
    if shutil.which('docker') is None or subprocess.run(
        ['docker', 'compose', 'version'], capture_output=True,
            check=False).returncode:
        pytest.skip('docker compose is unavailable')
    monkeypatch.delenv('ML_MAX_GPU_WORKER_SLOTS', raising=False)
    if slots is not None:
        monkeypatch.setenv('ML_MAX_GPU_WORKER_SLOTS', slots)

    repo_root = Path(__file__).resolve().parents[2]
    compose = repo_root / 'docker' / 'compose'
    result = subprocess.run([
        'docker', 'compose', '--env-file',
        _internal_env(repo_root, tmp_path), '-f',
        compose / 'docker-compose.singlehost.yml', '-f',
        compose / 'docker-compose.singlehost.gpu.yml', 'config', '--format',
        'json'
    ], check=True, capture_output=True, text=True)
    services = json.loads(result.stdout)['services']

    assert services['kumo-rest']['environment'][
        'ML_MAX_GPU_WORKER_SLOTS'] == expected
    assert services['kumo-pool-worker']['environment'][
        'POOL_GPU_CONCURRENCY'] == expected


def test_compose_local_defaults_to_prod_identity():
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / 'docker' / 'compose' / '.env.compose-local'
    compose_file = repo_root / 'docker' / 'compose' / (
        'docker-compose.singlehost.yml')
    preflight_file = repo_root / 'scripts' / 'compose-local-preflight.sh'

    env_values = env_file.read_text(encoding='utf-8')
    compose_values = compose_file.read_text(encoding='utf-8')
    preflight_values = preflight_file.read_text(encoding='utf-8')

    assert 'DEPLOY_STAGE=prod' in env_values
    assert 'TEMPORAL_NAMESPACE=prod' in env_values
    assert 'DISABLE_AUTH=true' in env_values
    assert 'KUMO_AUTH_BACKEND' not in env_values
    assert 'COMPOSE_PROFILES' not in env_values
    assert 'AUTH0_CREDENTIALS' not in env_values
    assert 'DEPLOY_STAGE: ${DEPLOY_STAGE:-prod}' in compose_values
    assert 'TEMPORAL_NAMESPACE: ${TEMPORAL_NAMESPACE:-prod}' in compose_values
    assert ('KUMO_BOOTSTRAP_OWNER_UID="${KUMO_BOOTSTRAP_OWNER_UID:-999}"'
            in preflight_values)
    assert ('"$KUMO_BOOTSTRAP_OWNER_UID:$KUMO_BOOTSTRAP_OWNER_GID"'
            in preflight_values)
    assert 'secret_name = "kumo-secret-key"' in preflight_values


@pytest.fixture
def compose_values() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    paths = sorted(
        (repo_root / 'docker' / 'compose').glob('docker-compose*.yml'))
    assert paths, 'no docker-compose*.yml files found'
    return '\n'.join(path.read_text(encoding='utf-8') for path in paths)


def test_singlehost_has_no_shared_encryption_key(compose_values):
    assert 'KUMO_CREDENTIALS_ENCRYPTION_SECRET' not in compose_values


def test_databricks_artifact_staging_has_no_deployment_configuration(
        compose_values):
    assert 'DATABRICKS_ARTIFACT_STAGING_PATH' not in compose_values


def test_compose_local_preflight_generates_singlehost_pool_strategy(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / 'scripts' / 'compose-local-preflight.sh'
    output_path = tmp_path / 'deploy_env_config.production-local.json'
    command = (f'COMPOSE_LOCAL_PREFLIGHT_SOURCE_ONLY=1 source "{script}"; '
               'generate_deploy_config '
               '"abfss://test-container@example.dfs.core.windows.net/'
               '{instance_id}/" '
               '"abfss://artifacts@example.dfs.core.windows.net/" '
               f'"7233" "{output_path}"')

    subprocess.run(['bash', '-euo', 'pipefail', '-c', command], check=True)

    deploy_config = json.loads(output_path.read_text(encoding='utf-8'))
    assert deploy_config['data_compute']['pool_strategy'] == 'singlehost'


def test_compose_local_preflight_generates_nginx_without_idp_proxy(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / 'scripts' / 'compose-local-preflight.sh'
    output_path = tmp_path / 'nginx.conf'
    command = (
        f'COMPOSE_LOCAL_PREFLIGHT_SOURCE_ONLY=1 source "{script}"; '
        f'write_nginx_config "{output_path}" "9388" "3000" "8000" "8001"')

    subprocess.run(['bash', '-euo', 'pipefail', '-c', command], check=True)

    nginx_config = output_path.read_text(encoding='utf-8')
    assert 'listen 9388;' in nginx_config
    assert 'proxy_pass http://kumo-rest:8000/v0/;' in nginx_config
    assert 'proxy_pass http://kumo-rest-public:8001/;' in nginx_config
    assert 'proxy_pass http://kumo-ui:3000;' in nginx_config
    assert 'location /auth/' not in nginx_config


def test_compose_local_auto_defaults_use_provider_test_buckets():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / 'scripts' / 'compose-local-preflight.sh'
    env_file = repo_root / 'docker' / 'compose' / '.env.compose-local'
    command = (f'COMPOSE_LOCAL_PREFLIGHT_SOURCE_ONLY=1 source "{script}"; '
               'cloud_root_is_writable() { return 0; }; '
               'printf "%s\n" '
               '"$DEFAULT_PRODUCTION_LOCAL_DATAPLANE_ROOT" '
               '"$DEFAULT_PRODUCTION_LOCAL_ARTIFACT_ROOT" '
               '"$(resolve_auto_dataplane_root aws)" '
               '"$(cloud_default_artifact_root aws)" '
               '"$(resolve_auto_dataplane_root gcp)" '
               '"$(cloud_default_artifact_root gcp)" '
               '"$(resolve_auto_dataplane_root azure)" '
               '"$(cloud_default_artifact_root azure)"')

    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', command],
        check=True,
        capture_output=True,
        text=True,
    )

    defaults = result.stdout.strip().splitlines()
    env_values = env_file.read_text(encoding='utf-8')
    assert defaults == [
        'auto',
        'auto',
        's3://kumo-test-dataplane-bucket/compose-local/{instance_id}/',
        's3://kumo-public-datasets',
        'gs://kumo-test-bucket/compose-local/{instance_id}/',
        'gs://kumo-public-datasets-bigquery',
        'abfss://test-container@kumotestdataplane.dfs.core.windows.net/devserver2/{instance_id}/',
        'abfss://kumopublic@kumopublicdatasets.dfs.core.windows.net/',
    ]
    assert 'COMPOSE_LOCAL_CLOUD_PROVIDER=auto' in env_values
    assert 'PRODUCTION_LOCAL_DATAPLANE_ROOT=auto' in env_values
    assert 'PRODUCTION_LOCAL_ARTIFACT_ROOT=auto' in env_values


def test_compose_local_auto_dataplane_falls_back_to_local_when_cloud_denied():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / 'scripts' / 'compose-local-preflight.sh'
    command = (f'COMPOSE_LOCAL_PREFLIGHT_SOURCE_ONLY=1 source "{script}"; '
               'cloud_root_is_writable() { return 1; }; '
               'printf "%s\n" "$(resolve_auto_dataplane_root aws)"')

    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', command],
        check=True,
        capture_output=True,
        text=True,
    )

    expected = (str(repo_root / '.cache/kumo/compose-local/dataplane') +
                '/{instance_id}/')
    assert result.stdout.strip() == expected
