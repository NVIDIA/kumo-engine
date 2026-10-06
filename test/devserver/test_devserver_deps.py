# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for devserver dependency-gating logic.

Specifically: which devserver config variants should require the AWS EKS
``kubectl`` context dependency (``_needs_eks_context``) and which should
not. The list is asserted explicitly against every JSON config under
``rest/devserver/configs/`` so accidentally flipping a variant (e.g. by
changing the default ``DiskgraphConfig.pool_strategy``) shows up as a
test failure rather than as a silent runtime regression on operator
machines without ``kubectl``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import kumoapi.data_source as api
import pytest
from kumoapi.data_source import DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID

from rest.devserver.main import (
    _AWS_WORKLOAD_RUNTIME_ENV_NAMES,
    _AZURE_WORKLOAD_RUNTIME_ENV_NAMES,
    _CONNECTOR_IO_RUNTIME_ENV_NAMES,
    _DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
    _DEVSERVER_INHERITED_ENV_SCRUB_NAMES,
    _DEVSERVER_RUNTIME_ENV_NAMES,
    _DEVSERVER_SECRET_ENV_NAMES,
    _aws_workload_runtime_env_assignments,
    _azure_workload_runtime_env_assignments,
    _browser_topology_instructions,
    _connector_io_runtime_env,
    _devserver_runtime_env,
    _eks_context_dep,
    _enable_dev_gcp_user_credentials,
    _gcp_workload_runtime_env_assignments,
    _initialize_devserver_sqlite_state,
    _needs_eks_context,
    _optional_env_assignments,
    generate_base_port,
    ui_panel,
    web_devserver,
)
from service_framework._raw_deploy_config import (
    DiskgraphConfig,
    FileStorageConfig,
    SqliteMetadataStorageConfig,
    _parse_raw_deploy_config,
)
from service_framework.deploy_env_config import (
    _resolve_data_compute_roots,
    _resolve_storage_roots,
)
from service_framework.deploy_stage import DeployStage
from service_framework.developer_credentials import (
    ENV_DEV_GCP_USER_CREDENTIALS_ENABLED,
)
from statedb.db import StateDB
from statedb.db.identities import AMBIENT_S3_IDENTITY_ID
from statedb.db.repositories import JsonRepository
from statedb.rest.identity import IdentityGroupEntry

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / 'rest' / 'devserver' / 'configs'

# Explicit per-variant expectation. Keep this table in sync with the
# JSON configs — adding a new config without updating the table will
# fail ``test_table_covers_every_config`` below.
#
# A variant ``needs_eks_context`` iff its data_compute is a
# ``DiskgraphConfig`` with ``pool_strategy='k8s'`` (the only path that
# provisions worker pods on EKS).
EXPECTED_NEEDS_EKS_CONTEXT: dict[str, bool] = {
    # DiskgraphConfig defaults pool_strategy='k8s'
    'adls.json': True,
    # DiskgraphConfig defaults pool_strategy='k8s'
    'databricks.json': True,
    # DiskgraphConfig pool_strategy='singlehost'
    'default.json': False,
    # DiskgraphConfig pool_strategy='singlehost'
    'diskgraph-test.json': False,
    # DiskgraphConfig defaults pool_strategy='k8s'
    'ec2.json': True,
    # GCP variant — handled by is_gcp_deploy() in the caller (stern
    # only); _needs_eks_context still returns True since the underlying
    # diskgraph pool is k8s. The caller short-circuits via
    # is_gcp_deploy() before this check matters.
    'gcp.json': True,
    # DiskgraphConfig pool_strategy='k8s' (explicit)
    'k8s.json': True,
    # DiskgraphConfig pool_strategy='singlehost'
    'local.json': False,
    # Direct Connect local auth, singlehost diskgraph pool.
    'local-direct-connect.json': False,
    # Local Postgres metadata, singlehost diskgraph pool.
    'local-postgres.json': False,
    # Direct Connect OIDC auth, singlehost diskgraph pool.
    'oidc.json': False,
    # DiskgraphConfig defaults pool_strategy='k8s'
    'remote.json': True,
}


def _all_config_files() -> list[Path]:
    return sorted(CONFIGS_DIR.glob('*.json'))


def test_importing_devserver_main_does_not_apply_tracked_env() -> None:
    tracked_names = {
        line.split('=', 1)[0]
        for line in (REPO_ROOT / 'rest' / 'devserver' /
                     '.env').read_text(encoding='utf-8').splitlines()
        if line and not line.startswith('#') and '=' in line
    }
    env = os.environ.copy()
    for name in tracked_names:
        env.pop(name, None)
    check = ('import os; import rest.devserver.main as main; '
             f'assert not ({tracked_names!r} & set(os.environ)); '
             'main._load_devserver_env(); '
             f'assert {tracked_names!r} <= set(os.environ)')

    subprocess.run(
        [sys.executable, '-c', check],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_ui_panel_passes_selected_browser_and_internal_api_ports() -> None:
    panel = ui_panel(12000, 12001, 'devserver_id=12000',
                     'kumo-ui-node-modules-devserver-train-a-deadbeef')

    assert panel.command == ('./scripts/run_ui_devserver.sh '
                             "12000 12001 devserver_id=12000 '' "
                             'kumo-ui-node-modules-devserver-train-a-deadbeef')
    assert panel.ssh_ports == [12000]
    assert panel.web_urls == [('Web UI', 'http://localhost:12000/')]


def test_ui_devserver_script_has_valid_shell_syntax() -> None:
    """Keep embedded Docker bootstrap snippets parseable by the outer shell."""
    subprocess.run(['bash', '-n', 'scripts/run_ui_devserver.sh'], cwd=REPO_ROOT,
                   check=True)


def test_browser_topology_uses_ui_origin_and_keeps_direct_api_ports() -> None:
    instructions = _browser_topology_instructions(12000, 12001, 12002,
                                                  direct_connect_oidc=True)

    assert 'http://localhost:12000/' in instructions
    assert 'http://localhost:12000/v0/auth/callback' in instructions
    assert 'http://localhost:12001/docs' in instructions
    assert 'http://localhost:12002/docs' in instructions


def test_non_oidc_topology_does_not_print_callback_registration() -> None:
    instructions = _browser_topology_instructions(10000, 10001, 10002,
                                                  direct_connect_oidc=False)

    assert '/v0/auth/callback' not in instructions


def test_explicit_base_port_precedes_environment_and_default(monkeypatch):
    monkeypatch.setenv('PHASE1B_DEVSERVER_BASE_PORT', '12000')

    assert generate_base_port(base_port=11000) == 11000
    assert generate_base_port() == 12000

    monkeypatch.delenv('PHASE1B_DEVSERVER_BASE_PORT')
    assert generate_base_port() == 10000


def test_deploy_config_suffix_isolated_paths_and_legacy_paths(monkeypatch):
    storage = FileStorageConfig(data_plane_storage_root='/mnt/localssd/data',
                                artifact_root='/mnt/localssd/artifacts')
    compute = DiskgraphConfig(data_root='/mnt/localssd/diskgraph')
    args = (DeployStage.DEV, 'developer-host', False)

    monkeypatch.delenv('DEVSERVER_INSTANCE_SUFFIX', raising=False)
    legacy_storage = _resolve_storage_roots(storage, *args)
    legacy_compute = _resolve_data_compute_roots(compute, args[1])
    assert legacy_storage.data_plane_storage_root == '/mnt/localssd/data'
    assert legacy_storage.artifact_root == '/mnt/localssd/artifacts'
    assert legacy_compute.data_root == '/mnt/localssd/diskgraph'

    monkeypatch.setenv('DEVSERVER_INSTANCE_SUFFIX', '/devserver-11000')
    isolated_storage = _resolve_storage_roots(storage, *args)
    isolated_compute = _resolve_data_compute_roots(compute, args[1])
    assert isolated_storage.data_plane_storage_root == (
        '/mnt/localssd/data/devserver-11000')
    assert isolated_storage.artifact_root == (
        '/mnt/localssd/artifacts/devserver-11000')
    assert isolated_compute.data_root == (
        '/mnt/localssd/diskgraph/devserver-11000')


def test_deploy_config_kumo_home_child_suffix_is_applied_once(monkeypatch):
    """A child KUMO_HOME must not receive its own suffix a second time."""
    child_home = '/mnt/localssd/cache/kumo/devserver/devserver-11000'
    suffix = '/devserver-11000'
    monkeypatch.setenv('KUMO_HOME', child_home)
    monkeypatch.setenv('DEVSERVER_INSTANCE_SUFFIX', suffix)

    storage = _resolve_storage_roots(
        FileStorageConfig(data_plane_storage_root='{kumo_home}',
                          artifact_root='{kumo_home}/artifacts'),
        DeployStage.DEV, 'developer-host', False)
    compute = _resolve_data_compute_roots(
        DiskgraphConfig(data_root='{kumo_home}/diskgraph',
                        local_ingest_root='{kumo_home}/local-ingest'),
        'developer-host')

    assert storage.data_plane_storage_root == child_home
    assert storage.control_plane_storage_root == child_home
    assert storage.artifact_root == child_home + '/artifacts' + suffix
    assert compute.data_root == child_home + '/diskgraph' + suffix
    assert compute.local_ingest_root == child_home + '/local-ingest' + suffix


@pytest.mark.asyncio
async def test_devserver_initializes_default_resources_before_panels(
        tmp_path) -> None:
    deploy_config = SimpleNamespace(
        metadata_storage=SqliteMetadataStorageConfig(),
        deploy_mode=DeployStage.DEV_LOCAL,
    )

    await _initialize_devserver_sqlite_state(
        deploy_config,  # type: ignore[arg-type]
        db_storage_root=str(tmp_path),
    )
    # The preflight is safe to repeat when a devserver restarts.
    await _initialize_devserver_sqlite_state(
        deploy_config,  # type: ignore[arg-type]
        db_storage_root=str(tmp_path),
    )

    state_db = await StateDB.open_sqlite(str(tmp_path / 'state.sqlite'))
    try:
        async with state_db.read_tx() as tx:
            group_rows = await tx.fetch('SELECT id FROM groups')
            project_rows = await tx.fetch('SELECT id FROM projects')
    finally:
        await state_db.close()

    assert [row[0] for row in group_rows] == [DEFAULT_GROUP_ID]
    assert [row[0] for row in project_rows] == [DEFAULT_PROJECT_ID]


@pytest.mark.asyncio
async def test_ambient_devserver_bootstrap_binds_only_default_group(
        tmp_path, monkeypatch) -> None:
    """A fresh ambient devserver must authorize its canonical project."""
    import statedb.db.identities as identities_module

    monkeypatch.setattr(identities_module, '_ambient_s3_enabled', lambda: True)
    monkeypatch.setattr(
        identities_module,
        '_ambient_s3_config',
        lambda: api.S3IdentityConfig(authentication=api.S3AmbientAuthentication(
        )),
    )
    deploy_config = SimpleNamespace(
        metadata_storage=SqliteMetadataStorageConfig(),
        deploy_mode=DeployStage.DEV,
    )

    await _initialize_devserver_sqlite_state(
        deploy_config,  # type: ignore[arg-type]
        db_storage_root=str(tmp_path),
    )
    # The cold-start setup is safely idempotent.
    await _initialize_devserver_sqlite_state(
        deploy_config,  # type: ignore[arg-type]
        db_storage_root=str(tmp_path),
    )

    state_db = await StateDB.open_sqlite(str(tmp_path / 'state.sqlite'))
    try:
        async with state_db.read_tx() as tx:
            bindings = await JsonRepository('identity_groups',
                                            IdentityGroupEntry).list_by_field(
                                                tx, 'identity_id',
                                                str(AMBIENT_S3_IDENTITY_ID))
    finally:
        await state_db.close()

    assert [(str(row.payload.identity_id), str(row.payload.group_id))
            for row in bindings] == [(str(AMBIENT_S3_IDENTITY_ID),
                                      DEFAULT_GROUP_ID)]


@pytest.mark.asyncio
async def test_devserver_does_not_preflight_non_sqlite_metadata(
        tmp_path) -> None:
    deploy_config = SimpleNamespace(metadata_storage=object())

    await _initialize_devserver_sqlite_state(
        deploy_config,  # type: ignore[arg-type]
        db_storage_root=str(tmp_path),
    )

    assert not (tmp_path / 'state.sqlite').exists()


def test_table_covers_every_config():
    """Guard: EXPECTED_NEEDS_EKS_CONTEXT must list every JSON config."""
    on_disk = {p.name for p in _all_config_files()}
    expected = set(EXPECTED_NEEDS_EKS_CONTEXT.keys())
    missing = on_disk - expected
    extra = expected - on_disk
    assert not missing, (
        f'New devserver config(s) not covered by '
        f'EXPECTED_NEEDS_EKS_CONTEXT: {sorted(missing)}. Add an explicit '
        f'entry asserting whether the variant needs the AWS EKS context '
        f'dependency.')
    assert not extra, (
        f'EXPECTED_NEEDS_EKS_CONTEXT references config(s) that no '
        f'longer exist on disk: {sorted(extra)}')


@pytest.mark.parametrize('config_file', _all_config_files(),
                         ids=lambda p: p.name)
def test_needs_eks_context_matches_expectation(config_file: Path):
    """Each variant's _needs_eks_context() must match the table above."""
    data = json.loads(config_file.read_text())
    raw = _parse_raw_deploy_config(data)
    expected = EXPECTED_NEEDS_EKS_CONTEXT[config_file.name]
    actual = _needs_eks_context(raw.data_compute)
    assert actual == expected, (
        f'{config_file.name}: expected _needs_eks_context={expected} '
        f'but got {actual}. data_compute={raw.data_compute!r}')


def test_eks_context_dep_targets_testing_uw4_for_pool_k8s():
    """pool_strategy='k8s' must target the testing-uw4 cluster, while
    legacy variants (no pool / non-pool data_compute) stay on dev-uw2."""
    dep_k8s = _eks_context_dep('k8s')
    dep_legacy = _eks_context_dep(None)
    assert 'testing-uw4' in dep_k8s.name
    assert 'dev-uw2' in dep_legacy.name
    assert 'kubectl config use-context' in dep_k8s.command
    assert 'kubectl config use-context' in dep_legacy.command


def test_provider_workload_runtime_env_allowlist_is_safe(monkeypatch, tmp_path):
    expected_base_controls = {
        'AWS_DEFAULT_REGION',
        'DB_STORAGE_ROOT',
        'KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED',
        'KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED',
        'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED',
        ENV_DEV_GCP_USER_CREDENTIALS_ENABLED,
        'KUMO_TTNG_DEBUG_BATCH_TIMING',
        'KUMO_TTNG_EXPERIMENT_FORCE_SAMPLER_MODE',
        'KUMO_TORCH_SHARING_STRATEGY',
        'GOOGLE_CLOUD_PROJECT',
    }
    expected_local_aws_selectors = {
        'AWS_CONTAINER_CREDENTIALS_FULL_URI',
        'AWS_ENDPOINT_URL_S3',
        'AWS_ENDPOINT_URL_STS',
    }
    forbidden_secrets = {
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_SESSION_TOKEN',
        'AWS_SECURITY_TOKEN',
        'AWS_CONTAINER_AUTHORIZATION_TOKEN',
        'AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE',
        'AZURE_CLIENT_CERTIFICATE_PASSWORD',
        'AZURE_CLIENT_SECRET',
        'AZURE_PASSWORD',
        'AZURE_STORAGE_CONNECTION_STRING',
        'DATABRICKS_CLIENT_SECRET',
        'GOOGLE_CREDENTIALS_STR_JSON',
        'GOOGLE_APPLICATION_CREDENTIALS',
        'SNOWFLAKE_PASSWORD',
        'SNOWFLAKE_PRIVATE_KEY',
    }
    assert expected_base_controls <= set(_DEVSERVER_RUNTIME_ENV_NAMES)
    assert expected_local_aws_selectors == set(_AWS_WORKLOAD_RUNTIME_ENV_NAMES)
    assert _AZURE_WORKLOAD_RUNTIME_ENV_NAMES == ('AZURE_FEDERATED_TOKEN_FILE', )
    assert forbidden_secrets.isdisjoint(_DEVSERVER_RUNTIME_ENV_NAMES)
    assert forbidden_secrets.isdisjoint(_AWS_WORKLOAD_RUNTIME_ENV_NAMES)
    assert forbidden_secrets == set(_DEVSERVER_SECRET_ENV_NAMES)
    assert set(_AWS_WORKLOAD_RUNTIME_ENV_NAMES) <= set(
        _DEVSERVER_INHERITED_ENV_SCRUB_NAMES)
    assert 'AWS_ENDPOINT_URL' in _DEVSERVER_INHERITED_ENV_SCRUB_NAMES

    for name in _DEVSERVER_INHERITED_ENV_SCRUB_NAMES:
        monkeypatch.setenv(name, f'inherited-{name.lower()}')
    monkeypatch.delenv(_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME)

    monkeypatch.setenv('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv(ENV_DEV_GCP_USER_CREDENTIALS_ENABLED, 'true')
    monkeypatch.setenv('KUMO_TTNG_DEBUG_BATCH_TIMING', '1')
    monkeypatch.setenv('KUMO_TTNG_EXPERIMENT_FORCE_SAMPLER_MODE', 'fused')
    monkeypatch.setenv('KUMO_TORCH_SHARING_STRATEGY', 'file_descriptor')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'kumo-staging-project')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-west-2')
    monkeypatch.setenv('DB_STORAGE_ROOT', str(tmp_path / 'metadata'))
    monkeypatch.setenv(
        'AWS_CONTAINER_CREDENTIALS_FULL_URI',
        'http://127.0.0.1:18500/credentials;echo-injected',
    )
    monkeypatch.setenv('AWS_ENDPOINT_URL_S3', 'http://localhost:18501')
    monkeypatch.setenv('AWS_ENDPOINT_URL_STS', 'http://[::1]:18502')
    azure_token = tmp_path / 'azure-workload-token'
    azure_token.write_text('azure-secret-token-content')
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', str(azure_token))
    for name in forbidden_secrets:
        monkeypatch.setenv(name, f'secret-{name.lower()}')

    rendered = _devserver_runtime_env()
    for name in expected_base_controls | expected_local_aws_selectors:
        assert f'{name}=' in rendered
    assert ("AWS_CONTAINER_CREDENTIALS_FULL_URI="
            "'http://127.0.0.1:18500/credentials;echo-injected'") in rendered
    for name in forbidden_secrets:
        assert f'-u {name}' in rendered
        assert f'{name}=' not in rendered
        assert f'secret-{name.lower()}' not in rendered
    for name in _DEVSERVER_INHERITED_ENV_SCRUB_NAMES:
        assert f'-u {name}' in rendered
        assert f'inherited-{name.lower()}' not in rendered
    assert '-u AWS_ENDPOINT_URL ' in rendered
    assert 'AWS_ENDPOINT_URL=' not in rendered
    assert f'AZURE_FEDERATED_TOKEN_FILE={azure_token}' in rendered
    assert 'azure-secret-token-content' not in rendered


def test_provider_workload_runtime_env_defaults_are_unchanged(monkeypatch):
    for name in (
            *_DEVSERVER_RUNTIME_ENV_NAMES,
            *_AWS_WORKLOAD_RUNTIME_ENV_NAMES,
            *_AZURE_WORKLOAD_RUNTIME_ENV_NAMES,
            _DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
    ):
        monkeypatch.delenv(name, raising=False)
    assert _optional_env_assignments(list(_DEVSERVER_RUNTIME_ENV_NAMES)) == ''
    assert _aws_workload_runtime_env_assignments() == (
        'AWS_CONFIG_FILE=/dev/null '
        'AWS_SHARED_CREDENTIALS_FILE=/dev/null ')
    assert _azure_workload_runtime_env_assignments() == ''
    assert _gcp_workload_runtime_env_assignments() == ''
    rendered = _devserver_runtime_env()
    assert rendered.startswith('env ')
    assert all(f'-u {name}' in rendered
               for name in _DEVSERVER_INHERITED_ENV_SCRUB_NAMES)
    assert 'AWS_CONFIG_FILE=/dev/null' in rendered
    assert 'AWS_SHARED_CREDENTIALS_FILE=/dev/null' in rendered


def test_devserver_enables_google_user_adc_by_default(monkeypatch):
    monkeypatch.delenv(ENV_DEV_GCP_USER_CREDENTIALS_ENABLED, raising=False)

    _enable_dev_gcp_user_credentials()

    assert os.environ[ENV_DEV_GCP_USER_CREDENTIALS_ENABLED] == 'true'
    assert (f'{ENV_DEV_GCP_USER_CREDENTIALS_ENABLED}=true'
            in _devserver_runtime_env())


def test_devserver_respects_explicit_google_user_adc_disable(monkeypatch):
    monkeypatch.setenv(ENV_DEV_GCP_USER_CREDENTIALS_ENABLED, 'false')

    _enable_dev_gcp_user_credentials()

    assert os.environ[ENV_DEV_GCP_USER_CREDENTIALS_ENABLED] == 'false'


def test_runtime_env_forwards_gcp_project_but_limits_transfer_bucket(
        monkeypatch):
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET',
                       'transfer-bucket;echo-not-executed')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'kumo-staging-project')
    secret = 'private-key-must-not-render'
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', secret)

    ordinary = _devserver_runtime_env()
    connector_io = _connector_io_runtime_env()

    assert set(_CONNECTOR_IO_RUNTIME_ENV_NAMES) == {
        'GCP_STORAGE_TRANSFER_BUCKET',
    }
    assert 'GCP_STORAGE_TRANSFER_BUCKET=' not in ordinary
    assert 'GOOGLE_CLOUD_PROJECT=kumo-staging-project' in ordinary
    assert '-u GCP_STORAGE_TRANSFER_BUCKET' in ordinary
    assert '-u GOOGLE_CLOUD_PROJECT' in ordinary
    assert ("GCP_STORAGE_TRANSFER_BUCKET="
            "'transfer-bucket;echo-not-executed'") in connector_io
    assert 'GOOGLE_CLOUD_PROJECT=kumo-staging-project' in connector_io
    assert secret not in connector_io


def test_devserver_forwards_gcp_project_but_limits_transfer_bucket_to_workers(
        monkeypatch):
    import rest.config.direct_connect as direct_connect
    import rest.devserver.main as devserver_main

    deploy_env = SimpleNamespace(
        data_compute=SimpleNamespace(pool_strategy='singlehost'),
        instance_id='dev-test',
        id_suffix='-dev-test',
        task_queue=lambda queue: f'{queue}-dev-test',
    )
    monkeypatch.setattr(devserver_main, 'deploy_env_v2', lambda: deploy_env)
    monkeypatch.setattr(devserver_main, '_validate_devserver_non_prod_env',
                        lambda: None)
    monkeypatch.setattr(devserver_main, '_build_dependencies',
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        direct_connect,
        'resolve_auth_settings_config',
        lambda _env: SimpleNamespace(
            enabled=False,
            direct_connect=SimpleNamespace(login_method='disabled'),
        ),
    )
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'transfer-bucket')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'staging-project')
    repo_root = Path(devserver_main.__file__).resolve().parents[2]
    monkeypatch.setenv('PYTHONPATH', str(repo_root))
    secret = 'credential-must-not-appear-in-tmux-command'
    monkeypatch.setenv('GOOGLE_CREDENTIALS_STR_JSON', secret)

    driver = web_devserver('rest-api', base_port=24000)
    commands = {panel.title: panel.command for panel in driver.panels}

    assert ('POOL_TASK_QUEUE=diskgraph-daemon-tq-dev-test '
            in commands['pool-worker'])

    connector_io_panels = {
        'pool-controller',
        'pool-worker',
        't-worker-dataworkflow',
        't-worker-mlworkflow',
    }
    for title in connector_io_panels:
        assert 'GCP_STORAGE_TRANSFER_BUCKET=transfer-bucket' in commands[title]
        assert 'GOOGLE_CLOUD_PROJECT=staging-project' in commands[title]
        # These are script entrypoints below the checkout root.  A shared
        # virtualenv can point at another editable worktree, so every worker
        # command must receive the active checkout explicitly.
        assert f'PYTHONPATH={repo_root}' in commands[title]
    for title in {'Rest API', 'Public Rest API'}:
        assert 'GCP_STORAGE_TRANSFER_BUCKET=' not in commands[title]
        assert 'GOOGLE_CLOUD_PROJECT=staging-project' in commands[title]
    assert all(secret not in command for command in commands.values())


def test_devserver_can_disable_ui_without_disabling_job_workers(monkeypatch):
    import rest.config.direct_connect as direct_connect
    import rest.devserver.main as devserver_main

    deploy_env = SimpleNamespace(
        data_compute=SimpleNamespace(pool_strategy='singlehost'),
        instance_id='dev-test',
        id_suffix='-dev-test',
        task_queue=lambda queue: f'{queue}-dev-test',
    )
    monkeypatch.setattr(devserver_main, 'deploy_env_v2', lambda: deploy_env)
    monkeypatch.setattr(devserver_main, '_validate_devserver_non_prod_env',
                        lambda: None)
    docker_requirements = []

    def build_dependencies(*_args, requires_docker):
        docker_requirements.append(requires_docker)
        return []

    monkeypatch.setattr(devserver_main, '_build_dependencies',
                        build_dependencies)
    monkeypatch.setattr(
        direct_connect,
        'resolve_auth_settings_config',
        lambda _env: SimpleNamespace(
            enabled=False,
            direct_connect=SimpleNamespace(login_method='disabled'),
        ),
    )
    monkeypatch.setenv('DEVSERVER_UI', 'false')
    monkeypatch.setenv('DEVSERVER_MONITORING', 'false')

    driver = web_devserver('rest-api', base_port=25000)
    titles = {panel.title for panel in driver.panels}

    assert 'Web UI' not in titles
    assert {
        'Rest API', 'Public Rest API', 't-worker-dataworkflow',
        't-worker-mlworkflow', 'pool-controller', 'pool-worker'
    } <= titles
    assert docker_requirements == [False]


@pytest.mark.parametrize('disabled_value', ['', '0', 'false', 'no', 'off'])
def test_aws_runtime_selectors_are_suppressed_when_exchange_is_off(
        monkeypatch, disabled_value):
    monkeypatch.setenv('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED', disabled_value)
    monkeypatch.setenv(
        'AWS_CONTAINER_CREDENTIALS_FULL_URI',
        'http://attacker.example:1234/credentials',
    )
    monkeypatch.setenv('AWS_ENDPOINT_URL_STS', 'http://attacker.example:1234')
    assert _aws_workload_runtime_env_assignments() == (
        'AWS_CONFIG_FILE=/dev/null '
        'AWS_SHARED_CREDENTIALS_FILE=/dev/null ')


@pytest.mark.parametrize('invalid_value', ['enabled', 'on', 'tru', '2'])
def test_aws_runtime_selectors_reject_ambiguous_exchange_flag(
        monkeypatch, invalid_value):
    monkeypatch.setenv('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED', invalid_value)
    with pytest.raises(ValueError, match='explicit true/false'):
        _aws_workload_runtime_env_assignments()


@pytest.mark.parametrize(
    'unsafe_url',
    [
        'https://127.0.0.1:18501',
        'http://127.0.0.1',
        'http://example.com:18501',
        'http://169.254.170.2:18501',
        'http://user@127.0.0.1:18501',
        'http://127.0.0.1:18501?query=value',
        'http://127.0.0.1:18501#fragment',
    ],
)
def test_aws_runtime_selectors_reject_unsafe_urls(monkeypatch, unsafe_url):
    monkeypatch.setenv('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv('AWS_ENDPOINT_URL_S3', unsafe_url)
    with pytest.raises(ValueError, match='loopback'):
        _aws_workload_runtime_env_assignments()


@pytest.mark.parametrize('disabled_value', ['', '0', 'false', 'no', 'off'])
def test_azure_runtime_selector_is_suppressed_when_exchange_is_off(
        monkeypatch, disabled_value):
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', disabled_value)
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', '/unsafe/inherited-token')
    assert _azure_workload_runtime_env_assignments() == ''


@pytest.mark.parametrize('invalid_value', ['enabled', 'on', 'tru', '2'])
def test_azure_runtime_selector_rejects_ambiguous_exchange_flag(
        monkeypatch, invalid_value):
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', invalid_value)
    with pytest.raises(ValueError, match='explicit true/false'):
        _azure_workload_runtime_env_assignments()


@pytest.mark.parametrize('unsafe_path', ['relative-token', '/missing/token'])
def test_azure_runtime_selector_rejects_unsafe_path(monkeypatch, unsafe_path):
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', unsafe_path)
    with pytest.raises(ValueError, match='absolute existing regular file'):
        _azure_workload_runtime_env_assignments()


def test_gcp_external_account_selector_propagates_only_validated_path(
        monkeypatch, tmp_path):
    secret = 'subject-token-secret-must-not-render'
    credential_file = tmp_path / 'external account; credentials.json'
    credential_file.write_text(
        json.dumps({
            'type': 'external_account',
            'audience': '//iam.googleapis.com/projects/123/locations/global',
            'credential_source': {
                'file': secret,
            },
        }))
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv(_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
                       str(credential_file))
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS',
                       '/unsafe/service-account.json')

    rendered = _devserver_runtime_env()

    assert '-u GOOGLE_APPLICATION_CREDENTIALS' in rendered
    assert f'-u {_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME}' in rendered
    assert ("GOOGLE_APPLICATION_CREDENTIALS='"
            f"{credential_file}'") in rendered
    assert f'{_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME}=' not in rendered
    assert '/unsafe/service-account.json' not in rendered
    assert secret not in rendered


@pytest.mark.parametrize('disabled_value', ['', '0', 'false', 'no', 'off'])
def test_gcp_external_account_selector_is_suppressed_when_exchange_is_off(
        monkeypatch, disabled_value):
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', disabled_value)
    monkeypatch.setenv(_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
                       '/missing/external-account.json')
    assert _gcp_workload_runtime_env_assignments() == ''


@pytest.mark.parametrize('invalid_value', ['enabled', 'on', 'tru', '2'])
def test_gcp_external_account_selector_rejects_ambiguous_exchange_flag(
        monkeypatch, invalid_value):
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', invalid_value)
    with pytest.raises(ValueError, match='explicit true/false'):
        _gcp_workload_runtime_env_assignments()


@pytest.mark.parametrize(
    'document',
    [
        {
            'type': 'service_account',
            'private_key': 'secret-key'
        },
        {
            'type': 'authorized_user',
            'refresh_token': 'secret-token'
        },
        {
            'type': 'external_account '
        },
        {
            'audience': 'missing-type'
        },
    ],
)
def test_gcp_external_account_selector_rejects_other_credential_types(
        monkeypatch, tmp_path, document):
    credential_file = tmp_path / 'credentials.json'
    credential_file.write_text(json.dumps(document))
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv(_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
                       str(credential_file))

    with pytest.raises(ValueError, match='exact credential type') as caught:
        _gcp_workload_runtime_env_assignments()

    assert 'secret-key' not in str(caught.value)
    assert 'secret-token' not in str(caught.value)


@pytest.mark.parametrize('contents', ['not-json', '[]', '"external_account"'])
def test_gcp_external_account_selector_rejects_malformed_or_non_object_json(
        monkeypatch, tmp_path, contents):
    credential_file = tmp_path / 'credentials.json'
    credential_file.write_text(contents)
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv(_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
                       str(credential_file))

    with pytest.raises(ValueError, match='valid JSON|JSON object'):
        _gcp_workload_runtime_env_assignments()


def test_gcp_external_account_selector_rejects_unsafe_paths(
        monkeypatch, tmp_path):
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    valid_file = tmp_path / 'external.json'
    valid_file.write_text('{"type":"external_account"}')
    symlink = tmp_path / 'external-link.json'
    symlink.symlink_to(valid_file)
    unsafe_paths = (
        'relative.json',
        str(tmp_path / 'missing.json'),
        str(tmp_path),
        str(symlink),
    )

    for unsafe_path in unsafe_paths:
        monkeypatch.setenv(_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
                           unsafe_path)
        with pytest.raises(ValueError,
                           match='absolute existing regular non-symlink file'):
            _gcp_workload_runtime_env_assignments()
