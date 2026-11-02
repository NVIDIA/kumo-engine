# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import socket
import sys
import uuid
from pathlib import Path
from test.ci_resources import execution_profile
from typing import Dict, Generator, List

import pytest
from fastapi.testclient import TestClient

from io_utils import fsutil, logging
from io_utils.json import load_json
from monitoring.settings import MonitoringSettings
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.backends.ml_backend.settings import MLBackendSettings
from rest.config.api_command import FixtureDefinition
from rest.config.authentication import AuthenticationSettings
from rest.deps.authentication import Authentication
from rest.driver.groups_driver import GroupsDriver
from rest.driver.projects_driver import ProjectsDriver
from rest.driver.rest_api_driver import RestApiDriver
from rest.public.app import AppBuilder
from rest.settings import RestApiDriverSettings, get_db_storage_root
from rest.testing import integration
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.db import StateDB, StateServices, open_state_db_from_deploy_config
from temporalio_commons.temporal_client import TemporalClient, asyncio

logging.initialize_logging()


@pytest.fixture(scope="session")
def ml_backend_settings(
        deploy_env_config: DeployEnvConfig) -> MLBackendSettings:
    # integ overrides the unit fixture: it uses the REAL ML backend
    # (use_fake_job_backend=False), built the way production builds it — via
    # parse_from_env(). In TEST mode parse_from_env() defaults the ML StateDB
    # StateDB is opened from '<db_root>', so job state the workers persist is
    # visible to the in-process backend.
    return MLBackendSettings.parse_from_env()


@pytest.fixture(scope="session")
def ml_state_db(
    deploy_env_config: DeployEnvConfig,
    ml_backend_settings: MLBackendSettings,
) -> StateDB:
    # integ overrides the unit fixture: open the ML StateDB at the real backend's
    return asyncio.run(
        open_state_db_from_deploy_config(
            deploy_env_config,
            state_root=get_db_storage_root(),
        ))


# For all integ tests, data backend async ingestion will always be enabled.
@pytest.fixture(scope='session', autouse=True)
def enable_async_ingestion(data_backend_ctx: DataBackendContext):
    data_backend_settings = data_backend_ctx._settings
    disable_async_ingestion = data_backend_settings.disable_async_ingestion
    data_backend_settings.disable_async_ingestion = False
    yield
    data_backend_settings.disable_async_ingestion = disable_async_ingestion


async def clean_temporal_state(deploy_env_config: DeployEnvConfig) -> None:
    """Stop only this pytest session's workflows before worker shutdown."""
    await integration.terminate_and_confirm_no_open_workflows(deploy_env_config)


@pytest.fixture(scope="session")
def temporal_workers_ready(
    deploy_env_config: DeployEnvConfig,
    rest_state_db: StateDB,
) -> Generator[bool, None, None]:
    del rest_state_db  # ordering dependency: pins DB_STORAGE_ROOT first
    yield from integration.run_temporal_workers(
        deploy_env_config,
        before_shutdown=lambda: clean_temporal_state(deploy_env_config),
    )


@pytest.fixture(scope="session")
def ml_backend(
    deploy_env_config: DeployEnvConfig,
    temporal_workers_ready: bool,
    temporal_client: TemporalClient,
    ml_backend_settings: MLBackendSettings,
    ml_state_db: StateDB,
    event_loop: asyncio.AbstractEventLoop,
) -> MLJobBackend:
    # In-process MLJobBackend over the same StateDB the e2e workers use (no
    # AutoRPC client). Uses the integ ml_backend_settings/ml_state_db overrides
    # (real backend, use_fake_job_backend=False) — NOT the fake unit fixtures —
    # so start_training/start_batch_prediction launch real Temporal workflows.
    backend = MLJobBackend(ml_backend_settings, temporal_client, ml_state_db)
    event_loop.run_until_complete(backend.start(read_only=True))
    return backend


def _build_public_api_app(
    rest_api_driver_settings: RestApiDriverSettings,
    ml_backend: MLJobBackend,
    data_backend_ctx: DataBackendContext,
    state_services: StateServices,
):
    # Build a dedicated RestApiDriver to source the in-process metadata driver
    # instances production uses (see rest/public/main.py); the public app's
    # startup_event configures each driver's StateServices. The in-process
    # MLJobBackend is shared by the driver and the public job routers.
    auth_settings = AuthenticationSettings(
        disable=True,
        is_unit_test=True,
    )
    auth = Authentication(auth_settings, is_unit_test=True)
    groups_driver = GroupsDriver()
    projects_driver = ProjectsDriver()
    public_driver = RestApiDriver(
        rest_api_driver_settings,
        auth,
        ml_backend,
        data_backend_ctx,
        projects_driver=projects_driver,
    )
    return AppBuilder(
        public_driver.pquery_driver,
        public_driver.connector_driver,
        public_driver.artifact_driver,
        public_driver.data_snapshot_api_driver,
        public_driver.table_driver,
        public_driver.graph_driver,
        ml_backend,
        MonitoringSettings(init_tracing=False),
        is_unit_test=True,
        auth_disabled=True,
        groups_driver=groups_driver,
        projects_driver=projects_driver,
        state_services=state_services,
    ).app


@pytest.fixture
def public_api_client(
    client: TestClient,
    temporal_client: TemporalClient,
    temporal_workers_ready: bool,
    rest_api_driver_settings: RestApiDriverSettings,
    ml_backend: MLJobBackend,
    data_backend_ctx: DataBackendContext,
    state_services: StateServices,
    event_loop: asyncio.AbstractEventLoop,
):
    # Exercise public routes with a real owner-bound user key.  Static test
    # keys bypass the final user/membership model and are intentionally not
    # accepted by the API-key driver.
    key_response = client.post(
        '/v0/api-keys',
        json={'name': f'integ-{uuid.uuid4().hex}'},
    )
    assert key_response.status_code == 201, key_response.text
    api_key = key_response.json()

    # Inject the shared StateServices (backed by the same StateDB the rest of
    # the e2e topology uses) so the public app reads/writes the same metadata.
    app = _build_public_api_app(rest_api_driver_settings, ml_backend,
                                data_backend_ctx, state_services)
    with TestClient(app) as public_client:
        public_client.headers['X-API-KEY'] = api_key['value']
        public_client.state_services = state_services
        try:
            yield public_client
        finally:
            delete_response = client.delete(f"/v0/api-keys/{api_key['id']}")
            assert delete_response.status_code == 204, delete_response.text


# Return a list of test cases where each test case is formatted as
# "<test fixture json file path>:<pquery name>"
def scan_test_cases() -> List[str]:
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'test_cases')
    tests_file = []
    for test_file in fsutil.listdir(dir):
        if not test_file.endswith('.json'):
            continue
        tests_file.append(test_file)
    test_cases = []
    for test_file in tests_file:
        fixture = load_json(test_file, FixtureDefinition)
        for pquery in fixture.queries:
            graph_name = f"graph_{pquery.graph_fpid[:8]}"
            graph_name = re.sub(r'_[0-9a-f]+$', '', graph_name)
            test_cases.append(f'{test_file}:{graph_name}:{pquery.pquery_name}')
    return test_cases


def test_case_id(test_file_param: str):
    _, graph, pquery_id = test_file_param.rsplit(':', 2)
    return f'{graph}-{pquery_id}'


@pytest.fixture(params=scan_test_cases(), ids=test_case_id)
def pquery_db_dump(request) -> FixtureDefinition:
    # Load parametrized fixture cases for the active public-API E2E suites.
    fixture_json_file, _, pquery_id = request.param.rsplit(':', 2)
    fixture = load_json(fixture_json_file, FixtureDefinition)
    fixture.queries = [
        pq for pq in fixture.queries if pq.pquery_name == pquery_id
    ]
    return fixture


# ---------------------------------------------------------------------------
# Diskgraph pool fixtures.
#
# Shared by ``test_diskgraph_e2e.py`` and ``test_diskgraph_public_api_e2e.py``
# via ``pytestmark = pytest.mark.usefixtures("_pool_subprocesses")`` at the
# top of each file. Not autouse — non-diskgraph integ tests opt out and
# avoid the storage_backend parametrize and the subprocess spawn cost.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", params=[
    "local", pytest.param("s3", marks=pytest.mark.private_resource('aws')),
])
def storage_backend(request):
    """Parametrize diskgraph tests over local and S3 storage backends."""
    return request.param


@pytest.fixture(scope="session")
def _diskgraph_deploy_config(tmp_path_factory, storage_backend, request,
                             temporal_workflow_env, test_execution_environment):
    """Configure storage and workers for the selected DiskGraph backend."""
    import json

    config_file = os.environ.get('DEPLOY_ENV_CONFIG_FILE') or str(
        Path(__file__).resolve().parents[3] / 'rest' / 'devserver' /
        'configs' / 'diskgraph-test.json')
    with open(config_file) as f:
        config = json.load(f)

    if temporal_workflow_env is not None:
        target_host = temporal_workflow_env.client.service_client.config.target_host
        config.setdefault('orchestration', {})['server_address'] = target_host

    if storage_backend == 's3':
        import datetime

        # S3-backed DiskGraph E2E tests can spend several minutes between
        # snapshot ingest and later training-table materialization. Keep the
        # test snapshot resident long enough that slow CI does not evict the
        # local/dataplane copy mid-test.
        data_compute = config.setdefault('data_compute', {})
        data_compute['pool_max_ssd_idle_secs'] = max(
            data_compute.get('pool_max_ssd_idle_secs') or 0, 1800)
        data_compute['pool_max_dataplane_retention_secs'] = max(
            data_compute.get('pool_max_dataplane_retention_secs') or 0, 1800)

        # 6-hour age guard avoids racing concurrent CI jobs sharing the bucket.
        # Compare in epoch seconds: fsutil.get_last_modified_time() returns a
        # float (Unix epoch), so it must not be subtracted from a datetime.
        _s3_test_prefix = 's3://kumo-dev-dataplane-bucket/test/'
        _stale_threshold_s = datetime.timedelta(hours=6).total_seconds()
        _now_s = datetime.datetime.now(datetime.timezone.utc).timestamp()

        def _newest_mtime(prefix):
            """Best-effort epoch-seconds mtime of an S3 prefix, or 0 if it
            cannot be dated. get_last_modified_time() only inspects direct
            children, so a prefix whose top-level ``.marker`` was already
            removed by a partially-completed prior cleanup reads back the
            DEFAULT_DATA_VERSION (<= 0) sentinel even though it still holds
            stale nested data. Fall back to a recursive scan in that case so
            those orphaned prefixes still get reaped."""
            mtime = fsutil.get_last_modified_time(prefix)
            if mtime > 0:
                return mtime
            newest = 0.0
            try:
                from io_utils.fsutil.s3_filesystem import listdir_recursive
                bucket = prefix.split('/')[2]
                for key in listdir_recursive(prefix):
                    # listdir_recursive yields bucket-relative keys; restore
                    # the scheme so get_last_modified_time hits S3, not local.
                    full = key if key.startswith('s3://') \
                        else f's3://{bucket}/{key}'
                    m = fsutil.get_last_modified_time(full)
                    if m > newest:
                        newest = m
            except Exception:
                pass  # best-effort dating; leave undatable prefixes alone
            return newest

        try:
            for entry in fsutil.listdir(_s3_test_prefix):
                if 'ci-diskgraph-e2e-' in str(entry):
                    try:
                        mtime = _newest_mtime(entry)
                        # mtime <= 0 means the prefix could not be dated (no
                        # objects / unreadable); skip rather than treat the
                        # epoch (1970) as ancient and delete blindly.
                        if mtime > 0 and (_now_s - mtime) > _stale_threshold_s:
                            fsutil.rm(entry, ignore_errors=True)
                    except Exception:
                        pass  # best-effort cleanup; skip this entry
        except Exception:
            pass  # best-effort

        s3_root = (f'{_s3_test_prefix}'
                   f'ci-diskgraph-e2e-{uuid.uuid4().hex[:12]}/')
        # Marker so set_home_dir()'s fsutil.exists() check passes on S3.
        fsutil.write_text(s3_root + '.marker', '')
        config.setdefault('storage', {})['data_plane_storage_root'] = s3_root

        def cleanup_s3():
            fsutil.rm(s3_root, ignore_errors=True)

        request.addfinalizer(cleanup_s3)
    else:
        tmp_root = str(tmp_path_factory.mktemp("diskgraph-storage"))
        config.setdefault('storage', {})['data_plane_storage_root'] = tmp_root

    # Per-session data_root avoids stale _SUCCESS markers cache-hitting
    # DiskgraphIngestWorkflow across pytest invocations. Also wipes any
    # stale ``.job_slot_holder`` from a previous (crashed) pytest run
    # that would otherwise make ``recover_occupied_slot`` block on an
    # AutoRPC call to a non-existent ml-backend service worker.
    dg_data_root = str(tmp_path_factory.mktemp("diskgraph-data-root"))
    config.setdefault('data_compute', {})['data_root'] = dg_data_root

    # Make ``bin/`` and the active venv first on PATH so worker subprocesses
    # (legacy path; pool=off only) and the pool subprocess (which spawns
    # ``kumo-auto`` / ``kumo-pred``) resolve the in-tree scripts using
    # the same Python interpreter that's running pytest.
    repo_root = Path(__file__).resolve().parents[3]
    venv_bin = Path(sys.executable).parent
    extra_path = [str(venv_bin), str(repo_root / 'bin')]
    os.environ['PATH'] = os.pathsep.join(extra_path +
                                         [os.environ.get('PATH', '')])

    os.environ['DEPLOY_ENV_CONFIG_JSON'] = json.dumps(config)
    os.environ.pop('DEPLOY_ENV_CONFIG_FILE', None)
    from service_framework.deploy_env_config import _reset_deploy_env
    _reset_deploy_env()

    # This fixture just reset deploy_env() above. The data persist dir is
    # derived from deploy_env() (kumo.data.get_persist_dir), so drop the lazy
    # BackendCacheManager to force it to re-root under the new storage root;
    # otherwise source exports land under the old KUMO_HOME while Rust resolves
    # them under this fixture's storage root.
    from kumo.data import reset_active_cache_for_test
    reset_active_cache_for_test()

    # Clear stale slot holder from a prior crashed pytest run.
    for stale_root in ('/tmp/diskgraph-ingest', dg_data_root):
        stale_holder = os.path.join(stale_root, '.job_slot_holder')
        try:
            os.remove(stale_holder)
        except FileNotFoundError:
            pass  # clean run — file may not exist
        except Exception:  # pylint: disable=broad-except
            pass  # cleanup is best-effort; don't fail setup on errors


# Pool subprocesses bake in DEPLOY_ENV_CONFIG_JSON at spawn time.
# pytest does not run session-scoped fixture finalizers between
# parametrize values, so workers spawned during [local] would otherwise
# survive into [s3] with stale env (writing/reading locally while the
# main process queries S3, producing 404 HeadObject). Track procs in a
# module-global and kill any prior pool at fixture entry.
_POOL_SUBPROCESSES_LIVE: List = []


def _free_tcp_port() -> str:
    for family, host in ((socket.AF_INET6, "::"), (socket.AF_INET,
                                                   "127.0.0.1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.bind((host, 0))
                return str(sock.getsockname()[1])
        except OSError:
            continue
    raise RuntimeError("Unable to reserve a free TCP port")


def _free_tcp_ports(count: int) -> List[str]:
    ports = set()
    while len(ports) < count:
        ports.add(_free_tcp_port())
    return list(ports)


def _kill_pool_subprocesses(procs: List, log) -> None:
    import signal
    import subprocess
    for proc in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for proc in procs:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log.warning('pool subprocess %s did not exit on SIGTERM; killing',
                        proc.pid)
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


@pytest.fixture(scope="session")
@execution_profile('diskgraph')
def _pool_subprocesses(_diskgraph_deploy_config, request):
    """Spawn the pool controller AutoRPC server and a singlehost pool
    worker as long-running subprocesses for the duration of the session.

    Each subprocess inherits the full env (DEPLOY_ENV_CONFIG_JSON +
    INSTANCE_ID + Temporal target host) so they connect to the same
    test-local Temporal dev server as the main pytest process.

    The pool worker is configured for singlehost-flavored routing —
    SinglehostPlacementStrategy on the controller side returns
    the deployment-scoped diskgraph daemon queue; the worker subscribes to
    it (one Worker per queue inside a single subprocess via
    pool.worker_main). Pool worker registers every SnapshotActivities
    method via inspect-based auto-discovery; activity bodies delegate
    to existing kumo activity functions.
    """
    import subprocess
    import time as _time

    log = logging.getLogger(__name__)

    # Kill any pool subprocesses left over from a prior
    # storage_backend parametrize before spawning fresh ones with the
    # current DEPLOY_ENV_CONFIG_JSON.
    if _POOL_SUBPROCESSES_LIVE:
        log.info('Tearing down %d stale pool subprocess(es) before respawn',
                 len(_POOL_SUBPROCESSES_LIVE))
        _kill_pool_subprocesses(list(_POOL_SUBPROCESSES_LIVE), log)
        _POOL_SUBPROCESSES_LIVE.clear()

    procs: List[subprocess.Popen] = []

    def _spawn(name: str, args: List[str], env_overrides: Dict[str, str]):
        env = {**os.environ, **env_overrides}
        log.info('Spawning %s: %s', name, ' '.join(args))
        proc = subprocess.Popen(args, env=env)
        procs.append(proc)
        _POOL_SUBPROCESSES_LIVE.append(proc)
        return proc

    def _check_started() -> None:
        exited = [(proc.pid, proc.returncode) for proc in procs
                  if proc.poll() is not None]
        if exited:
            raise RuntimeError(
                f"Pool subprocess startup failed; exited processes: {exited}")

    controller_pool_metrics_port, controller_worker_metrics_port, \
        worker_metrics_port = _free_tcp_ports(3)

    # Controller subprocess. Override POOL_METRICS_PORT (the var its
    # Prometheus exporter actually reads, see ``pool/controller_main.py``)
    # and WORKER_METRICS_PORT (Temporal SDK exporter) so repeated or parallel
    # local runs do not collide with stale subprocesses or other test sessions.
    # ``PYTHONUNBUFFERED=1`` is critical: without it Python block-buffers
    # stdout when piped to a file, so the controller's progress logs are
    # lost on test timeout when SIGTERM doesn't flush. Same fix logic as
    # ``_run_kumo_subprocess`` in ``pool.activities``.
    _spawn(
        'pool-controller',
        [sys.executable, '-m', 'pool.controller_main'],
        env_overrides={
            'POOL_METRICS_PORT': controller_pool_metrics_port,
            'WORKER_METRICS_PORT': controller_worker_metrics_port,
            'PYTHONUNBUFFERED': '1',
        },
    )

    # Pool worker subprocess (singlehost daemon role for now — covers
    # the ingest-chain activities the modified DiskgraphIngestWorkflow
    # routes through. CPU/GPU queue workers can be added the same way
    # once the train/predict workflows are pool-modified).
    from pool.task_queues import SINGLEHOST_DAEMON_TQ
    from service_framework.deploy_env_config import deploy_env

    _spawn(
        'pool-worker-daemon',
        [sys.executable, '-m', 'pool.worker_main'],
        env_overrides={
            'POOL_WORKER_UID': 'singlehost',
            'POOL_TASK_QUEUE': deploy_env().task_queue(SINGLEHOST_DAEMON_TQ),
            'POOL_STORAGE_MODE': 'daemon-on-pvc',
            'WORKER_METRICS_PORT': worker_metrics_port,
            'PYTHONUNBUFFERED': '1',
        },
    )

    # Give the controller a moment to register its AutoRPC workflow on
    # the controller task queue before any test issues a place_snapshot
    # call. ~3s is generous for a local dev server.
    _time.sleep(3)
    _check_started()

    def _shutdown():
        _kill_pool_subprocesses(procs, log)
        for proc in procs:
            try:
                _POOL_SUBPROCESSES_LIVE.remove(proc)
            except ValueError:
                pass

    request.addfinalizer(_shutdown)
    yield
