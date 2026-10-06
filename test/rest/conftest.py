# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import os
import socket
import uuid
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Callable, Generator, List, Optional, Tuple

import kumoapi.pquery as pquery_api
import kumoapi.table as table_api
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from kumo_core.types import ConnectorType, TaskType
from kumoapi.graph import ColumnKey, ColumnKeyGroup, GraphDefinition
from kumoapi.model_plan import RunMode, SuggestModelPlanRequest
from kumoapi.source_table import DataSourceType, FileType, S3SourceTable
from kumoapi.typing import Dtype, Stype
from pydantic import SecretStr
from pytest import MonkeyPatch

from io_utils import fsutil, logging
from io_utils.json import load_json
from io_utils.logging import LoggingMode
from kumo.data import get_active_cache, get_persist_dir
from kumo.data.backend_cache import BackendCacheManager
from kumo.runtime import init
from kumo.testing.rest import (
    TEST_REST_ADMIN_USER_ID,
    configure_test_client_token,
    create_identity_binding,
    financial_snowflake_config,
    get_and_expect_success,
    post_and_expect_success,
    snowflake_test_identity,
)
from kumo.testing.snowflake import disable_snowflake_env_credentials
from monitoring.settings import MonitoringSettings
from rest.app import AppBuilder
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.data_util import get_job_base_dir
from rest.backends.ml_backend.ml_job_backend import (
    MLBackendSettings,
    MLJobBackend,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.config.authentication import PYTEST_TOKEN, AuthenticationSettings
from rest.config.connector import (
    ConnectorResponse,
    CreateConnectorArgs,
    FileConnectorResourceConfig,
)
from rest.config.credentials import KeyPair
from rest.config.graph_v2 import GraphDefinition as RestGraphDefinition
from rest.config.graph_v2 import GraphResourceResponse
from rest.config.predictive_query_v2 import ModelPlanResponse
from rest.config.status import IDResponse
from rest.config.table_v2 import Column, TableDefinition, TableResourceResponse
from rest.config.training_job_v2 import TrainingJobRequest
from rest.deps import authentication as authentication_deps
from rest.deps.authentication import (
    AUTHORIZATION_API_KEY,
    Authentication,
    Permissions,
    RequestContext,
    UserAuthorization,
)
from rest.driver.api_key_driver import APIKeyDriver
from rest.driver.graph_driver import GraphDriver
from rest.driver.groups_driver import GroupsDriver
from rest.driver.projects_driver import ProjectsDriver
from rest.driver.rest_api_driver import RestApiDriver
from rest.public.app import AppBuilder as PublicAppBuilder
from rest.settings import (
    AppSettings,
    DataBackendSettings,
    DataScaleLimits,
    PredictiveQuerySettings,
    RestApiDriverSettings,
    get_db_storage_root,
)
from rest.testing import integration
from rest.testing.config import TestFunctionCleanupOptions
from service_framework.deploy_env import ENV_KUMO_HOME, get_default_home_dir
from service_framework.deploy_env_config import (
    DeployEnvConfig,
    _reset_deploy_env,
)
from service_framework.deploy_env_config import deploy_env as deploy_env_v2
from statedb.config.common import (
    ConnectorID,
    PredictiveQueryID,
    TableIDv2,
    TableName,
    TrainingJobID,
    UserID,
)
from statedb.db import StateDB, StateServices, open_state_db_from_deploy_config
from temporalio_commons import TemporalClient

log = logging.getLogger(__name__)

_HNM_TINY_ROOT = os.path.join(os.path.dirname(__file__), 'csv_data', 'HNM_TINY')
_TEST_REST_ADMIN_API_KEY = 'pytest-rest-admin-fixture'

_REST_METADATA_STATE_TABLES = (
    "connectors",
    "uploaded_files",
    "identities",
    "identity_groups",
    "tables",
    "graphs",
    "predictive_queries",
    "pquery_definitions",
    "pquery_aliases",
    "job_tags",
    "groups",
    "group_members",
    "external_group_mappings",
    "group_membership_rules",
    "projects",
    "batch_prediction_jobs",
    "api_keys",
    "api_key_secrets",
)


@dataclasses.dataclass
class InternalConnectorTestRequest:
    """Test-only shape for directly provisioned internal file connectors."""

    config: FileConnectorResourceConfig


def _register_test_project_upload_connector(
    client: TestClient,
    *,
    root_dir: str,
) -> ConnectorID:
    """Copy local test data behind the canonical project upload connector."""
    from test.rest.integ import fixture_connector_args

    _, connector = asyncio.run(
        fixture_connector_args.provision_project_upload_connector(
            client,
            source_root=root_dir,
            api_prefix='/v0',
        ))
    return connector.id


@pytest.fixture()
def default_project_id(state_services: StateServices) -> str:
    async def load() -> str:
        _, project = await state_services.groups.ensure_default_resources()
        return str(project.id)

    return asyncio.run(load())


_ML_JOB_STATE_TABLES = (
    "training_jobs",
    "batch_prediction_jobs",
    "generate_train_table_jobs",
    "generate_pred_table_jobs",
    "job_tags",
)


class RestStateDBTestStorage:
    def __init__(self, state_services: StateServices) -> None:
        self.state_services = state_services
        self._state_db = state_services.state_db

    async def init_once(self) -> None:
        return

    async def close(self) -> None:
        return

    async def reset_storage(self) -> None:
        async with self._state_db.write_tx() as tx:
            for table_name in _REST_METADATA_STATE_TABLES:
                await tx.execute(f"DELETE FROM {table_name}")

    async def reset_ml_storage(self) -> None:
        async with self._state_db.write_tx() as tx:
            for table_name in _ML_JOB_STATE_TABLES:
                await tx.execute(f"DELETE FROM {table_name}")


# Skip the autouse fixture `run_before_and_after_tests` (/test/conftest.py)
#
# By default, all REST tests use session-scoped test environment setup (with
# REST api app, data and graph backend, test client and dependencies), along
# with per test (function-scoped autouse fixture) cleanup/reset defined in
# `auto_cleanp_per_func()` with configurable cleanup options, see the
# `test_cleanup_options` fixture.


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


# REST tests set ``auto_setup_kumo_backend=False`` (above), which
# short-circuits the root autouse fixture and skips its DG col-stats
# seed-on-miss fallback; re-install it here so REST-side Task/Graph
# materialization tests don't fail with ``UnsupportedInDiskgraphError:
# Requested col stats missing from the DiskGraph-populated cache``.
# Removable once REST tests use real DG ingest for synthetic fixtures.


@pytest.fixture(autouse=True)
def _rest_install_dg_col_stats_fallback() -> Generator[None, None, None]:
    from test.conftest import _install_dg_col_stats_fallback
    restore = _install_dg_col_stats_fallback()
    try:
        yield
    finally:
        restore()


@pytest.fixture(autouse=True, scope='session')
def setup_rest_test_session(
        deploy_env_config: DeployEnvConfig) -> Generator[str, None, None]:
    default_home = get_default_home_dir()
    fsutil.makedirs(default_home, exist_ok=True)
    # Preserve the test temporal target so the re-installed singleton
    # below keeps pointing at the dev server, not Temporal Cloud.
    from service_framework._raw_deploy_config import CustomTemporalConfig
    target_host = (deploy_env_config.orchestration.server_address if isinstance(
        deploy_env_config.orchestration, CustomTemporalConfig) else None)
    with TemporaryDirectory(dir=default_home,
                            ignore_cleanup_errors=True) as test_home_dir:
        os.environ[ENV_KUMO_HOME] = test_home_dir
        # Reset + re-install the deploy_env singleton so it re-reads
        # ENV_KUMO_HOME (resolves data_plane_storage_root to the temp
        # dir) and the worker boot path sees the DG-configured object.
        _reset_deploy_env()
        DeployEnvConfig.for_test(temporal_target_host=target_host)
        os.chmod(test_home_dir, 0o777)
        init(
            'rest-api',
            test_home_dir,
            logging_mode=LoggingMode.DEVELOPMENT,
        )
        persist_dir = get_persist_dir()
        assert persist_dir is not None
        yield persist_dir
    os.environ[ENV_KUMO_HOME] = default_home


@pytest.fixture(scope="session")
def fixture_export_path(
        deploy_env_config: DeployEnvConfig) -> Generator[str, None, None]:
    yield os.path.join(deploy_env_config.data_plane_storage_root, 'fixtures')


@pytest.fixture(scope="session")
def deploy_env_config(temporal_workflow_env) -> DeployEnvConfig:
    os.environ['DEPLOY_STAGE'] = 'test'
    # The shared REST app is explicitly authentication-disabled.  Worker-side
    # capability redemption reads this deployment setting independently of the
    # in-process AppBuilder, so keep both halves in the same native mode.
    os.environ['DISABLE_AUTH'] = 'true'
    unique_suffix = uuid.uuid4().hex[:8]
    # Isolate AutoRPC task queues by unique instance id
    os.environ["INSTANCE_ID"] = f"pytest-{unique_suffix}"
    target_host = (
        temporal_workflow_env.client.service_client.config.target_host
        if temporal_workflow_env is not None else None)
    return DeployEnvConfig.for_test(temporal_target_host=target_host)


@pytest.fixture(scope="session")
def job_base_dir(deploy_env_config: DeployEnvConfig) -> str:
    return get_job_base_dir()


@pytest.fixture(scope="session")
def temporal_client(
    deploy_env_config: DeployEnvConfig,
    event_loop: asyncio.AbstractEventLoop,
) -> Generator[TemporalClient, None, None]:
    client = TemporalClient(deploy_env_config)
    try:
        yield client
    finally:
        event_loop.run_until_complete(client.close())


@pytest.fixture(scope="session")
def ml_backend_settings(
        deploy_env_config: DeployEnvConfig) -> MLBackendSettings:
    return MLBackendSettings(ml_job_settings=MLJobSettings(
        use_fake_job_backend=True))


@pytest.fixture(scope="session")
def ml_state_db(rest_state_db: StateDB, ) -> StateDB:
    """Use the single authoritative StateDB for REST and ML metadata."""
    return rest_state_db


@pytest.fixture(scope="session")
def ml_backend(
    ml_backend_settings: MLBackendSettings,
    temporal_client: TemporalClient,
    ml_state_db: StateDB,
    event_loop: asyncio.AbstractEventLoop,
) -> Generator[MLJobBackend, None, None]:
    """In-memory ML backend with mock jobs
    """
    backend = MLJobBackend(
        ml_backend_settings,
        temporal_client,
        ml_state_db,
    )
    event_loop.run_until_complete(backend.start())
    yield backend


@pytest.fixture(scope="session")
def data_backend_settings() -> DataBackendSettings:
    # TODO(siyang): enable this in unitttest once Temporal testing
    # framework is released
    return DataBackendSettings(disable_async_ingestion=True)


@pytest.fixture(scope="session")
def data_backend_ctx(
    deploy_env_config: DeployEnvConfig,
    data_backend_settings: DataBackendSettings,
):
    return DataBackendContext(deploy_env_config, data_backend_settings)


@pytest.fixture(scope="session")
def data_scale_limits(deploy_env_config: DeployEnvConfig) -> DataScaleLimits:
    return DataScaleLimits.parse_from_env(deploy_env_config)


@pytest.fixture(scope="session")
def pquery_settings(
        deploy_env_config: DeployEnvConfig) -> PredictiveQuerySettings:
    return PredictiveQuerySettings.parse_from_env(deploy_env_config)


@pytest.fixture(scope="session")
def graph_driver(rest_api_driver: RestApiDriver) -> GraphDriver:
    return rest_api_driver.graph_driver


@pytest.fixture(scope="session")
def rest_api_driver_settings(
    fixture_export_path: str,
    data_scale_limits: DataScaleLimits,
    pquery_settings: PredictiveQuerySettings,
) -> RestApiDriverSettings:
    settings = RestApiDriverSettings(
        fixture_export_path=fixture_export_path,
        allow_local_batch_pred_paths=True,
        allow_local_connector_file_paths=True,
    )
    settings.supported_connector_types = \
        list(ConnectorType.__members__.values())
    settings.allow_predict_on_new_data_version = True
    settings.data_scale_limits = data_scale_limits
    settings.pquery_settings = pquery_settings
    return settings


@pytest.fixture(scope="session")
def authentication_settings() -> AuthenticationSettings:
    return AuthenticationSettings(
        disable=True,
        kumo_external_shared_iam_role="arn:aws:iam::123456789012:root",
        is_unit_test=True)


@pytest.fixture
def apply_predictive_query_setting(deploy_env_config: DeployEnvConfig,
                                   client: TestClient,
                                   monkeypatch: MonkeyPatch):
    def _apply(key: str, value: bool):
        app_builder: AppBuilder = getattr(client, 'app_builder')

        monkeypatch.setenv(key, str(value))

        predictive_query_settings = PredictiveQuerySettings.parse_from_env(
            deploy_env_config)
        predictive_query_settings.__dict__.update({key: value})
        app_builder._driver.pquery_driver._pquery_settings = (
            predictive_query_settings)

    return _apply


@pytest_asyncio.fixture
async def ctx(
    authentication_settings: AuthenticationSettings, ) -> RequestContext:
    assert authentication_settings.disable
    return RequestContext(user_authorization=UserAuthorization.disabled(),
                          trace_id="")


@pytest.fixture(scope="session")
def rest_state_db() -> StateDB:
    # REST, public API, ML, and activity workers must share the configured
    # metadata store. A private tmp_path database makes activity-only
    # connector resolution observe a different project universe.
    #
    # DiskGraph integration fixtures later replace DEPLOY_ENV_CONFIG_JSON to
    # swap only the data-plane backend. Pin the already-isolated session DB
    # root before that swap so a lazy activity broker cannot drift to the new
    # data-plane root and open an empty StateDB.
    metadata_root = get_db_storage_root()
    os.environ['DB_STORAGE_ROOT'] = metadata_root
    return asyncio.run(
        open_state_db_from_deploy_config(
            deploy_env_v2(),
            state_root=metadata_root,
        ))


@pytest.fixture(scope="session", autouse=True)
def temporal_workers_ready(
    deploy_env_config: DeployEnvConfig,
    rest_state_db: StateDB,
) -> Generator[bool, None, None]:
    """Run activity workers required by REST routes in unit test sessions."""
    del rest_state_db  # ordering dependency: pin DB_STORAGE_ROOT first
    yield from integration.run_temporal_workers(deploy_env_config)


@pytest.fixture(scope="session")
def state_services(rest_state_db: StateDB) -> StateServices:
    return StateServices(rest_state_db)


@pytest.fixture(scope="session")
def storage_dao(state_services: StateServices) -> RestStateDBTestStorage:
    storage = RestStateDBTestStorage(state_services)
    asyncio.run(storage.init_once())
    try:
        yield storage
    finally:
        asyncio.run(storage.close())


def _make_rest_api_driver(
    rest_api_driver_settings: RestApiDriverSettings,
    auth: Authentication,
    ml_backend: MLJobBackend,
    data_backend_ctx: DataBackendContext,
    state_services: StateServices,
) -> RestApiDriver:
    driver = RestApiDriver(
        rest_api_driver_settings,
        auth,
        ml_backend,
        data_backend_ctx,
    )
    driver._configure_state_services(state_services)
    return driver


@pytest.fixture(scope="session")
def rest_api_driver(
    rest_api_driver_settings: RestApiDriverSettings,
    ml_backend: MLJobBackend,
    data_backend_ctx: DataBackendContext,
    authentication_settings: AuthenticationSettings,
    state_services: StateServices,
) -> RestApiDriver:
    auth = Authentication(authentication_settings)
    return _make_rest_api_driver(
        rest_api_driver_settings,
        auth,
        ml_backend,
        data_backend_ctx,
        state_services,
    )


@pytest.fixture
def random_test_identifier() -> str:
    return str(int(hash(socket.gethostname()) % 1e15))


@pytest.fixture(scope='session')
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Overrides pytest default function scoped event loop"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def _make_app_builder_with_auth_disabled(
    rest_api_driver: RestApiDriver,
    temporal_client: TemporalClient,
    ml_backend_settings: MLBackendSettings,
) -> AppBuilder:
    auth_settings = AuthenticationSettings(
        disable=True,
        kumo_external_shared_iam_role="arn:aws:iam::123456789012:root",
        is_unit_test=True)
    auth = Authentication(auth_settings, is_unit_test=True)
    # Each app owns its MLJobBackend (its startup binds it to the app's StateDB
    # and the app closes that StateDB on shutdown). The shared session
    # ml_backend fixture is reserved for the test harness (cleanup / direct
    # backend calls) and the RestApiDriver, so an app's shutdown can never
    # strand it -- which is what lets the bind be unconditional (no guard).
    return AppBuilder(
        settings=AppSettings(is_unit_test=True),
        auth=auth,
        driver=rest_api_driver,
        temporal_client=temporal_client,
        ml_backend=MLJobBackend(ml_backend_settings, temporal_client),
        monitoring_settings=MonitoringSettings(init_tracing=False),
    )


@pytest.fixture(scope='session')
def _session_app_builder(
    rest_api_driver: RestApiDriver,
    temporal_client: TemporalClient,
    ml_backend_settings: MLBackendSettings,
    deploy_env_config: DeployEnvConfig,
) -> AppBuilder:
    return _make_app_builder_with_auth_disabled(rest_api_driver,
                                                temporal_client,
                                                ml_backend_settings)


@pytest.fixture
def app_builder(
    rest_api_driver_settings: RestApiDriverSettings,
    temporal_client: TemporalClient,
    ml_backend: MLJobBackend,
    ml_backend_settings: MLBackendSettings,
    data_backend_ctx: DataBackendContext,
    authentication_settings: AuthenticationSettings,
    state_services: StateServices,
    deploy_env_config: DeployEnvConfig,
) -> Generator[AppBuilder, None, None]:
    auth = Authentication(authentication_settings)
    rest_api_driver = _make_rest_api_driver(
        rest_api_driver_settings,
        auth,
        ml_backend,
        data_backend_ctx,
        state_services,
    )
    yield _make_app_builder_with_auth_disabled(rest_api_driver, temporal_client,
                                               ml_backend_settings)


@pytest.fixture(scope="session")
def _session_client(
        _session_app_builder: AppBuilder) -> Generator[TestClient, None, None]:
    app = _session_app_builder.app
    with TestClient(app) as client:
        token = {'access_token': PYTEST_TOKEN, 'refresh_token': PYTEST_TOKEN}
        configure_test_client_token(app, token, call_with_access_token=True)

        # Allows us to access client.app_builder so we can edit things like
        # settings, driver, etc
        setattr(client, 'app_builder', _session_app_builder)

        yield client
        app.dependency_overrides = {}


@pytest.fixture
def client(
    _session_client: TestClient,
    monkeypatch: MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Return the shared client with explicit test-admin authority.

    Auth is disabled in tests, so every request already resolves to the
    anonymous customer-admin principal. This fixture instead authenticates as a
    stable, named admin identity so administrative-setup tests have a
    consistent owner and the fixture credential stays out of StateDB (API-key
    repository tests must observe exactly the records they create).
    """
    resolve_api_key = authentication_deps.resolve_user_authorization_from_api_key

    async def resolve_test_api_key(
        api_key: str,
        auth_disabled: bool = False,
        auth_proxied: bool = False,
        api_key_driver: Optional[APIKeyDriver] = None,
    ) -> UserAuthorization:
        if api_key == _TEST_REST_ADMIN_API_KEY:
            return UserAuthorization(
                user_id=UserID(TEST_REST_ADMIN_USER_ID),
                access_token=AUTHORIZATION_API_KEY,
                permissions=[
                    Permissions.READ.compose(),
                    Permissions.WRITE.compose(),
                    Permissions.ADMIN.compose(),
                ],
                user_name='Pytest REST administrator',
            )
        return await resolve_api_key(
            api_key=api_key,
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
            api_key_driver=api_key_driver,
        )

    # Keep the fixture credential out of StateDB: API-key tests need to
    # observe exactly the records they create, and StateDB background tasks
    # must only be started by the TestClient application's event loop.
    monkeypatch.setattr(authentication_deps,
                        'resolve_user_authorization_from_api_key',
                        resolve_test_api_key)
    _session_client.headers['X-API-Key'] = _TEST_REST_ADMIN_API_KEY
    try:
        yield _session_client
    finally:
        if 'X-API-Key' in _session_client.headers:
            del _session_client.headers['X-API-Key']


@pytest.fixture
def unauthenticated_client(
    _session_client: TestClient, ) -> Generator[TestClient, None, None]:
    """Return the shared auth-disabled client without fixture authority."""
    assert 'X-API-Key' not in _session_client.headers
    yield _session_client


# Per test function cleanup options. The default behavior is to reset rest and
# ML job state after each test function, but will not clear data backend cache and
# data persist dir. This best mimics UX behavior, i.e., customers have no way
# to clear in-memory cache or persist dir.
#
# Certain tests may want to customize and change the cleanup options. To do so,
# simply override the `test_cleanup_options` fixture in the test file (or
# conftest.py of sub-package) to toggle the cleanup options.


@pytest.fixture(scope="function")
def test_cleanup_options() -> TestFunctionCleanupOptions:
    return TestFunctionCleanupOptions()


@pytest.fixture(scope="function", autouse=True)
def auto_cleanup_per_func(rest_api_driver: RestApiDriver,
                          storage_dao: RestStateDBTestStorage,
                          state_services: StateServices,
                          deploy_env_config: DeployEnvConfig,
                          test_cleanup_options: TestFunctionCleanupOptions,
                          event_loop: asyncio.AbstractEventLoop):
    yield

    if test_cleanup_options.should_reset_rest_db:
        asyncio.run(storage_dao.reset_storage())
        asyncio.run(state_services.groups.ensure_default_resources())

    if test_cleanup_options.should_reset_ml_db:
        event_loop.run_until_complete(storage_dao.reset_ml_storage())

    if test_cleanup_options.should_reset_data_backend_cache:
        cache: BackendCacheManager = get_active_cache()
        cache.reset()

    if test_cleanup_options.should_delete_data_persist_dir:
        fsutil.rm(get_persist_dir(), ignore_errors=True)


@pytest.fixture
def public_api_client(
    _session_client: TestClient,
    monkeypatch: MonkeyPatch,
    deploy_env_config: DeployEnvConfig,
    rest_api_driver_settings: RestApiDriverSettings,
    ml_backend: MLJobBackend,
    ml_backend_settings: MLBackendSettings,
    temporal_client: TemporalClient,
    data_backend_ctx: DataBackendContext,
    state_services: StateServices,
    event_loop: asyncio.AbstractEventLoop,
):
    # Worker-side capability redemption derives the authentication mode from
    # deploy configuration/environment, not from the in-process app object.
    # Keep both halves of this explicitly auth-disabled test fixture aligned.
    monkeypatch.setenv('DISABLE_AUTH', 'true')
    # Build a dedicated RestApiDriver to source the in-process metadata driver
    # instances production uses (see rest/public/main.py); the public app's
    # startup_event wires each driver to this process's StateDB. The public app
    # gets its OWN in-process MLJobBackend (bound + closed by its lifecycle);
    # the shared ml_backend fixture stays the driver's + test harness's.
    auth_settings = AuthenticationSettings(
        disable=True,
        is_unit_test=True,
    )
    auth = Authentication(auth_settings, is_unit_test=True)
    public_driver = RestApiDriver(
        rest_api_driver_settings,
        auth,
        ml_backend,
        data_backend_ctx,
    )
    groups_driver = GroupsDriver()
    projects_driver = ProjectsDriver()
    groups_driver._configure_state_services(state_services)
    projects_driver._configure_state_services(state_services)
    # Inject the shared StateServices so the public app reads/writes the same
    # StateDB the test fixtures (and per-function reset_storage() cleanup) use.
    app = PublicAppBuilder(
        public_driver.pquery_driver,
        public_driver.connector_driver,
        public_driver.artifact_driver,
        public_driver.data_snapshot_api_driver,
        public_driver.table_driver,
        public_driver.graph_driver,
        MLJobBackend(ml_backend_settings, temporal_client),
        MonitoringSettings(init_tracing=False),
        is_unit_test=True,
        auth_disabled=True,
        groups_driver=groups_driver,
        projects_driver=projects_driver,
        state_services=state_services,
    ).app
    with TestClient(app) as public_client:
        yield public_client


#######################################################
# CSV-based fixtures
#######################################################


@pytest.fixture()
def make_default_table_def_v2() -> Callable[..., table_api.TableDefinition]:
    def _make_default_table_def_v2(
        source_table_name: str = 'source_table', ) -> table_api.TableDefinition:
        return table_api.TableDefinition(
            cols=[
                table_api.Column(name='pkey', stype=Stype.ID, dtype=Dtype.int),
                table_api.Column(name='f1', stype=Stype.categorical,
                                 dtype=Dtype.string),
                table_api.Column(name='f2', stype=Stype.sequence,
                                 dtype=Dtype.string),
                table_api.Column(name='f3', stype=Stype.numerical,
                                 dtype=Dtype.int),
            ],
            source_table=S3SourceTable(
                connector_id=f'connector_{"1" * 32}',
                source={
                    'mode': 'relative',
                    'path': source_table_name
                },
                file_type=FileType.PARQUET,  # pyright: ignore
                data_source_type=DataSourceType.S3,
            ),
            pkey='pkey',
        )

    return _make_default_table_def_v2


@pytest.fixture()
def fake_graph_and_tables_request_v2(
) -> List[Tuple[TableName, TableDefinition]]:
    """Connector-relative financial-shaped tables for graph validators."""
    connector_id = f'connector_{"1" * 32}'
    column_names = {
        'DISTRICT': ('DISTRICT_ID', ),
        'ACCOUNT': ('DISTRICT_ID', 'ACCOUNT_ID'),
        'CLIENT': ('DISTRICT_ID', 'CLIENT_ID'),
        'DISP': ('CLIENT_ID', 'ACCOUNT_ID', 'DISP_ID'),
        'LOAN': ('ACCOUNT_ID', 'DATE'),
        'ORDERS': ('ACCOUNT_ID', ),
        'TRANS': ('ACCOUNT_ID', 'DATE'),
        'CARD': ('DISP_ID', ),
    }
    pairs: List[Tuple[TableName, TableDefinition]] = []
    for source_name, names in column_names.items():
        cols = [
            Column(
                name=name,
                stype=Stype.timestamp if name == 'DATE' else Stype.ID,
                dtype=Dtype.date if name == 'DATE' else Dtype.int,
            ) for name in names
        ]
        table_def = TableDefinition(
            cols=cols,
            source_table=S3SourceTable(
                connector_id=connector_id,
                source={
                    'mode': 'relative',
                    'path': source_name
                },
                file_type=FileType.CSV,
                data_source_type=DataSourceType.S3,
            ),
            pkey=None,
        )
        pairs.append((TableName(f'{source_name}_FIXTURE'), table_def))
    return pairs


@pytest.fixture()
def fake_graph_v2(
    client: TestClient,
    state_services: StateServices,
    default_project_id: str,
) -> GraphResourceResponse:
    """Create the financial graph inside the real default project."""
    connector_id = _register_test_project_upload_connector(
        client,
        root_dir=os.path.join(os.path.dirname(__file__), '..', 'csv_data',
                              'FINANCIAL'),
    )
    resource = load_json(
        os.path.join(os.path.dirname(__file__), 'test_data',
                     'pquery_resource.json'),
        pquery_api.PQueryResource,
    )
    for table_name, table in list(resource.graph.tables.items()):
        resource.graph.tables[table_name] = dataclasses.replace(
            table,
            source_table=dataclasses.replace(
                table.source_table,
                connector_id=str(connector_id),
            ),
        )
    response = post_and_expect_success(
        client,
        f"/v1/graphs?project_id={default_project_id}&skip_validation=true",
        resource.graph,
        IDResponse,
    )
    return get_and_expect_success(
        client,
        f"/v1/graphs/{response.id}",
        {'project_id': default_project_id},
        GraphResourceResponse,
    )


def _v2_table_id(graph: GraphResourceResponse, table_name: str) -> TableIDv2:
    """Return the table's canonical project-owned resource ID."""
    fpid = graph.table_name_to_fpid_mapping[TableName(table_name)]
    return TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=fpid,
    )


@dataclasses.dataclass
class FakePQueryRequest:
    """Request carrier for the fake predictive-query fixtures: just the
    fields `_fake_pquery` consumes to drive the v1 resource endpoints.
    """
    name: str
    project_id: str
    graph_id: str
    query_yaml: str
    run_mode: RunMode


@pytest.fixture()
def fake_predictive_query_request(
    client: TestClient,
    fake_graph_v2: GraphResourceResponse,
    default_project_id: str,
) -> FakePQueryRequest:
    """
    STATIC BINARY CLASSIFICATION
    Make a predictive query that predicts one column.
    """
    trans_id = _v2_table_id(fake_graph_v2, 'TRANS')
    table_resp = get_and_expect_success(client, f'/v1/tables/{trans_id}', {
        'project_id': default_project_id,
    }, TableResourceResponse)
    pkey = table_resp.table.pkey
    assert pkey is not None
    return FakePQueryRequest(
        name="fake_query",
        project_id=default_project_id,
        graph_id=fake_graph_v2.id,
        query_yaml=(f'PREDICT TRANS.K_SYMBOL != "1"\n'
                    f'            FOR EACH TRANS.{pkey}'),
        run_mode=RunMode.FAST,
    )


@pytest.fixture()
def fake_predictive_query_pqlv2_request(
    client: TestClient,
    fake_graph_v2: GraphResourceResponse,
    default_project_id: str,
) -> FakePQueryRequest:
    """
    STATIC BINARY CLASSIFICATION
    Make a predictive query that predicts one column.
    """
    trans_id = _v2_table_id(fake_graph_v2, 'TRANS')
    table_resp = get_and_expect_success(client, f'/v1/tables/{trans_id}', {
        'project_id': default_project_id,
    }, TableResourceResponse)
    pkey = table_resp.table.pkey
    assert pkey is not None
    return FakePQueryRequest(
        name="fake_query",
        project_id=default_project_id,
        graph_id=fake_graph_v2.id,
        query_yaml=f"PREDICT TRANS.K_SYMBOL != '1'\nFOR EACH TRANS.{pkey}",
        run_mode=RunMode.FAST,
    )


@pytest.fixture()
def fake_predictive_query_with_model_plan_request(
    client: TestClient,
    fake_graph_v2: GraphResourceResponse,
    default_project_id: str,
) -> FakePQueryRequest:
    """
    TEMPORAL MULTICLASS CLASSIFICATION
    Make a predictive query that predicts one column.
    """
    account_id = _v2_table_id(fake_graph_v2, 'ACCOUNT')
    account_resp = get_and_expect_success(client, f'/v1/tables/{account_id}', {
        'project_id': default_project_id,
    }, TableResourceResponse)
    pkey = account_resp.table.pkey
    assert pkey is not None
    return FakePQueryRequest(
        name="fake_temporal_query",
        project_id=default_project_id,
        graph_id=fake_graph_v2.id,
        query_yaml=(f'\nPREDICT FIRST(LOAN.STATUS, 0, 7) = "A"\n'
                    f'FOR EACH ACCOUNT.{pkey}'),
        run_mode=RunMode.FAST,
    )


@dataclasses.dataclass
class FakePQuery:
    """Job-centric stand-in carrying just the predictive-query fields
    consumers need (id, graph, query_string, task_type).
    """
    id: PredictiveQueryID
    project_id: str
    graph_id: str
    query_string: str
    graph: object
    task_type: TaskType


@dataclasses.dataclass
class FakeTrainingJob:
    """A predictive query plus the training job created for it (job-centric)."""
    pquery_id: PredictiveQueryID
    project_id: str
    training_job_id: TrainingJobID
    task_type: TaskType
    graph_id: str


def _fake_pquery(client: TestClient, request: FakePQueryRequest) -> FakePQuery:
    # Suggest a model plan via the v1 endpoint (which needs the graph
    # definition, not just an id), then create the (named) predictive query.
    # The model plan makes the query trainable; task_type is read off the
    # suggested plan rather than a v0 GET.
    graph = RestGraphDefinition(**dataclasses.asdict(
        get_and_expect_success(client, f"/v1/graphs/{request.graph_id}", {
            'project_id': request.project_id,
        }, GraphResourceResponse).graph))
    model_plan = post_and_expect_success(
        client,
        f'/v1/predictive_queries/model_plan?project_id={request.project_id}',
        SuggestModelPlanRequest(
            query_string=request.query_yaml,
            graph_id=request.graph_id,
            run_mode=request.run_mode or RunMode.FAST,
            graph_definition=graph,
        ),
        ModelPlanResponse,
    )
    # Create the predictive query via the v1 resource endpoint and use the
    # returned (content/FPID-based) id everywhere.
    pq_id = post_and_expect_success(
        client,
        f"/v1/predictive_queries/?project_id={request.project_id}",
        pquery_api.PQueryResource(
            name=request.name,
            query_string=request.query_yaml,
            graph=graph,
            desc='',
        ),
        IDResponse,
    ).id
    return FakePQuery(
        id=PredictiveQueryID(pq_id),
        project_id=request.project_id,
        graph_id=request.graph_id,
        query_string=request.query_yaml,
        graph=graph,
        task_type=model_plan.suggested_model_plan.task_type,
    )


@pytest.fixture()
def fake_predictive_query(
        client: TestClient,
        fake_predictive_query_request: FakePQueryRequest) -> FakePQuery:
    return _fake_pquery(client, fake_predictive_query_request)


@pytest.fixture()
def fake_predictive_query_pqlv2(
    client: TestClient,
    fake_predictive_query_pqlv2_request: FakePQueryRequest,
    apply_predictive_query_setting: Callable[[str, bool], None],
) -> FakePQuery:
    return _fake_pquery(client, fake_predictive_query_pqlv2_request)


@pytest.fixture()
def fake_predictive_query_with_model_plan(
    client: TestClient,
    fake_predictive_query_with_model_plan_request: FakePQueryRequest
) -> FakePQuery:
    return _fake_pquery(client, fake_predictive_query_with_model_plan_request)


@pytest.fixture()
def fake_predictive_query_2(
        client: TestClient,
        fake_predictive_query_request: FakePQueryRequest) -> FakePQuery:
    request = deepcopy(fake_predictive_query_request)
    request.name = "fake_query2"
    return _fake_pquery(client, request)


def _fake_training_job(client: TestClient,
                       query: FakePQuery) -> FakeTrainingJob:
    # Train the already-created predictive query job-centrically: pass its
    # pquery_id so the training job links to it. No /v0/predictive_queries
    # route is used.
    plan = post_and_expect_success(
        client,
        f'/v1/predictive_queries/model_plan?project_id={query.project_id}',
        SuggestModelPlanRequest(
            query_string=query.query_string,
            graph_id=query.graph_id,
            run_mode=RunMode.FAST,
            graph_definition=query.graph,
        ),
        ModelPlanResponse,
    ).suggested_model_plan.model_plan
    training_job_id = post_and_expect_success(
        client,
        '/v0/training_jobs/train',
        TrainingJobRequest(
            predictive_query=query.query_string,
            graph_definition=query.graph,
            training_table_gen_plan=plan.training_table_generation,
            model_plan=plan,
            project_id=query.project_id,
        ),
        TrainingJobID,
    )
    return FakeTrainingJob(
        pquery_id=query.id,
        project_id=query.project_id,
        training_job_id=TrainingJobID(training_job_id),
        task_type=query.task_type,
        graph_id=query.graph_id,
    )


@pytest.fixture()
def fake_predictive_query_v2_id(client: TestClient, graph_definition_v2,
                                default_project_id: str) -> PredictiveQueryID:
    post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
        graph_definition_v2,
        IDResponse,
    )
    graph_definition_internal = RestGraphDefinition(
        **dataclasses.asdict(graph_definition_v2))
    post_and_expect_success(
        client,
        f'/v1/predictive_queries/?project_id={default_project_id}',
        pquery_api.PQueryResource(
            name="fake_query",
            query_string='PREDICT COUNT(transactions.*, 0, 20) FOR EACH '
            'customers.customer_id',
            graph=graph_definition_internal,
            desc='Fixture export v2 graph test query',
        ),
        IDResponse,
    )
    return PredictiveQueryID("fake_query")


@pytest.fixture()
def fake_training_job_v2(
    client: TestClient,
    graph_definition_v2: GraphDefinition,
    default_project_id: str,
) -> TrainingJobID:
    graph_definition_internal = RestGraphDefinition(
        **dataclasses.asdict(graph_definition_v2))
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
        graph_definition_v2,
        IDResponse,
    ).id
    pquery_str = (
        'PREDICT COUNT(transactions.*, 0, 30) FOR EACH customers.customer_id')
    model_plan = post_and_expect_success(
        client,
        f'/v1/predictive_queries/model_plan?project_id={default_project_id}',
        SuggestModelPlanRequest(
            query_string=pquery_str,
            graph_id=graph_id,
            run_mode=RunMode.FAST,
            graph_definition=graph_definition_internal,
        ),
        ModelPlanResponse,
    )
    return post_and_expect_success(
        client,
        '/v0/training_jobs/train',
        TrainingJobRequest(
            predictive_query=pquery_str,
            graph_definition=graph_definition_internal,
            training_table_gen_plan=(model_plan.suggested_model_plan.model_plan.
                                     training_table_generation),
            model_plan=model_plan.suggested_model_plan.model_plan,
            project_id=default_project_id,
        ),
        TrainingJobID,
    )


@pytest.fixture()
def fake_training_job(
    client: TestClient,
    fake_predictive_query: FakePQuery,
) -> FakeTrainingJob:
    return _fake_training_job(client, fake_predictive_query)


@pytest.fixture()
def fake_training_job_with_model_plan(
    client: TestClient,
    fake_predictive_query_with_model_plan: FakePQuery,
) -> FakeTrainingJob:
    return _fake_training_job(client, fake_predictive_query_with_model_plan)


@pytest.fixture()
def fake_training_job_pqlv2(
    client: TestClient,
    fake_predictive_query_pqlv2: FakePQuery,
) -> FakeTrainingJob:
    return _fake_training_job(client, fake_predictive_query_pqlv2)


@pytest.fixture()
def fake_training_job_2(
    client: TestClient,
    fake_predictive_query_2: FakePQuery,
) -> FakeTrainingJob:
    return _fake_training_job(client, fake_predictive_query_2)


@pytest.fixture()
def table_definition_v2(client: TestClient):
    """
    Creates a table definition from the table transactions
    after creating an s3 connector to public hnm_tiny.
    """
    connector_id = _register_test_project_upload_connector(
        client,
        root_dir=_HNM_TINY_ROOT,
    )
    table = table_api.TableDefinition(
        cols=[
            table_api.Column(
                name="t_dat",
                stype=Stype.timestamp,
                dtype=Dtype.time,
                timestamp_format=None,
            ),
            table_api.Column(
                name="customer_id",
                stype=Stype.ID,
                dtype=Dtype.string,
                timestamp_format=None,
            ),
        ],
        source_table=S3SourceTable(
            connector_id=str(connector_id),
            source={
                'mode': 'relative',
                'path': "transactions"
            },
            file_type=FileType.PARQUET,  # pyright: ignore
            data_source_type=DataSourceType.S3,
        ),
        pkey=None,
        time_col="t_dat",
        end_time_col=None,
    )
    return table


@pytest.fixture()
def graph_definition_v2(client: TestClient):
    """
    Creates a graph definition with two tables, transactions and customers
    from hnm data, after creating an s3 connector to public hnm_tiny.
    """
    connector_id = _register_test_project_upload_connector(
        client,
        root_dir=_HNM_TINY_ROOT,
    )
    return GraphDefinition(
        tables={
            "transactions":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="t_dat",
                        stype=Stype.timestamp,
                        dtype=Dtype.time,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="customer_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="article_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="sales_channel_id",
                        stype=Stype.numerical,
                        dtype=Dtype.int,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="price",
                        stype=Stype.numerical,
                        dtype=Dtype.float,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "transactions"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey=None,
                time_col="t_dat",
                end_time_col=None,
            ),
            "customers":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="customer_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="age",
                        stype=Stype.numerical,
                        dtype=Dtype.int,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="club_member_status",
                        stype=Stype.categorical,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "customers"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey="customer_id",
                time_col=None,
                end_time_col=None,
            ),
            "articles":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="article_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "articles"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey="article_id",
                time_col=None,
                end_time_col=None,
            ),
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(
                    table_name="transactions",
                    col_name="customer_id",
                ),
                ColumnKey(
                    table_name="customers",
                    col_name="customer_id",
                ),
            )),
            ColumnKeyGroup(columns=(
                ColumnKey(
                    table_name="transactions",
                    col_name="article_id",
                ),
                ColumnKey(
                    table_name="articles",
                    col_name="article_id",
                ),
            ))
        ],
    )


@pytest.fixture()
def graph_definition_v2_with_tx_pkey(client: TestClient):
    """Variant of ``graph_definition_v2`` whose ``transactions`` table has a
    string primary key ``transaction_id`` so it can be used as an entity table
    for fact-level predictions.

    Sources from ``s3://kumo-public-datasets/hnm_tiny/transactions_w_pkey/``,
    which is identical to the ``transactions`` parquet plus a synthetic
    ``transaction_id`` column.
    """
    connector_id = _register_test_project_upload_connector(
        client,
        root_dir=_HNM_TINY_ROOT,
    )
    return GraphDefinition(
        tables={
            "transactions":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="transaction_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="t_dat",
                        stype=Stype.timestamp,
                        dtype=Dtype.time,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="customer_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="article_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="sales_channel_id",
                        stype=Stype.numerical,
                        dtype=Dtype.int,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="price",
                        stype=Stype.numerical,
                        dtype=Dtype.float,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "transactions_w_pkey"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey="transaction_id",
                time_col="t_dat",
                end_time_col=None,
            ),
            "customers":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="customer_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="age",
                        stype=Stype.numerical,
                        dtype=Dtype.int,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "customers"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey="customer_id",
                time_col=None,
                end_time_col=None,
            ),
            "articles":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="article_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "articles"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey="article_id",
                time_col=None,
                end_time_col=None,
            ),
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(
                    table_name="transactions",
                    col_name="customer_id",
                ),
                ColumnKey(
                    table_name="customers",
                    col_name="customer_id",
                ),
            )),
            ColumnKeyGroup(columns=(
                ColumnKey(
                    table_name="transactions",
                    col_name="article_id",
                ),
                ColumnKey(
                    table_name="articles",
                    col_name="article_id",
                ),
            ))
        ],
    )


@pytest.fixture()
def graph_definition_v2_no_edges(client: TestClient):
    """
    Creates a graph definition with two tables, transactions and customers
    from hnm data, after creating an s3 connector to public hnm_tiny.
    This graph has no edges and is used to test link suggestion.
    """
    connector_id = _register_test_project_upload_connector(
        client,
        root_dir=_HNM_TINY_ROOT,
    )
    return GraphDefinition(
        tables={
            "transactions":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="t_dat",
                        stype=Stype.timestamp,
                        dtype=Dtype.time,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="customer_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="article_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="sales_channel_id",
                        stype=Stype.numerical,
                        dtype=Dtype.int,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="price",
                        stype=Stype.numerical,
                        dtype=Dtype.float,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "transactions"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey=None,
                time_col="t_dat",
                end_time_col=None,
            ),
            "customers":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="customer_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                    table_api.Column(
                        name="age",
                        stype=Stype.numerical,
                        dtype=Dtype.int,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "customers"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey="customer_id",
                time_col=None,
                end_time_col=None,
            ),
            "articles":
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name="article_id",
                        stype=Stype.ID,
                        dtype=Dtype.string,
                        timestamp_format=None,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=str(connector_id),
                    source={
                        'mode': 'relative',
                        'path': "articles"
                    },
                    file_type=FileType.PARQUET,  # pyright: ignore
                    data_source_type=DataSourceType.S3,
                ),
                pkey="article_id",
                time_col=None,
                end_time_col=None,
            ),
        },
        col_groups=[],
    )


#######################################################
# Snowflake-based fixtures
# TODO: Unify the code with CSV fixture,
# TODO: Also support parquet
#######################################################


@pytest.fixture()
def snowflake_key_pair_credentials() -> Generator[KeyPair, None, None]:
    from io_utils.secrets import get_secrets
    secrets = get_secrets("xva19026.snowflakecomputing.com")
    credentials = KeyPair(
        user=SecretStr('mfatest'),
        private_key=SecretStr(secrets['kumo_secretkey']),
    )
    with disable_snowflake_env_credentials():
        yield credentials


@pytest.fixture()
def snowflake_connector_request(
    client: TestClient, ) -> Callable[[str, bool], CreateConnectorArgs]:
    key_pair = KeyPair(
        user=SecretStr(os.environ['SNOWFLAKE_USER']),
        private_key=SecretStr(os.environ['SNOWFLAKE_PRIVATE_KEY']),
    )
    identity_config, credentials = snowflake_test_identity(key_pair)
    binding = create_identity_binding(
        client,
        identity_config=identity_config,
        credentials=credentials,
    )

    def _snowflake_connector_request(
        connector_name: str = "financial",
        keypair: Optional[bool] = True,
    ) -> CreateConnectorArgs:
        if not keypair:
            raise ValueError('Snowflake identity tests use key-pair auth')
        return CreateConnectorArgs(
            identity_id=binding.identity_id,
            group_id=binding.group_id,
            config=financial_snowflake_config(connector_name),
        )

    return _snowflake_connector_request


@pytest.fixture()
def snowflake_connector(
    client: TestClient,
    snowflake_connector_request: Callable[[str], CreateConnectorArgs]
) -> ConnectorResponse:
    response1 = post_and_expect_success(
        client, "/v0/projects/default/connectors/",
        snowflake_connector_request("financial_2"), IDResponse)
    return get_and_expect_success(
        client, f"/v0/projects/default/connectors/{response1.id}", {},
        ConnectorResponse)
