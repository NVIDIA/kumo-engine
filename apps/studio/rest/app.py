# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect
import os
from functools import partial
from typing import Any, Coroutine, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from kumo_core.errors import PublicError, TrainingValidationError
from kumo_core.version import get_kumo_version
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.types import ASGIApp

import rest.utils.tracing as tracing
from async_utils.instrument import measure_blocking_code
from async_utils.loop_guards import install_loop_guards
from common_utils.error import PublicException
from io_utils import logging
from kumo.connector.connector import Connector
from kumo.platform_errors import as_platform_public_exception
from monitoring.settings import MonitoringSettings
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.job_queue import StateDbJobQueue
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.backends.ml_backend.settings import (
    MLBackendSettings,
    MLJobQueueSettings,
)
from rest.config.direct_connect import resolve_direct_connect_runtime_config
from rest.config.encryption import CredentialsEncryptor
from rest.deps.authentication import Authentication, authorization_middleware
from rest.deps.db_common import OptimisticLockError
from rest.deps.direct_connect_group_resolver import DirectConnectGroupResolver
from rest.deps.group_membership import NativeGroupResolver
from rest.deps.project_resolver import ProjectResolver
from rest.driver.api_key_driver import APIKeyDriver
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from rest.driver.groups_driver import GroupsDriver
from rest.driver.projects_driver import ProjectsDriver
from rest.driver.rest_api_driver import RestApiDriver
from rest.routers.admin import AdminResource
from rest.routers.api_keys_v2 import UserAPIKeyResource
from rest.routers.authentication import AuthenticationResource
from rest.routers.batch_prediction_jobs_v2 import BatchPredictionV1Resource
from rest.routers.connectors import ConnectorResource
from rest.routers.constants import ConstantsResource
from rest.routers.direct_connect_auth import DirectConnectAuthResource
from rest.routers.evaluation_statistics import EvaluationStatisticsResource
from rest.routers.fixtures import FixturesResource
from rest.routers.gen_pred_table import GeneratePredictionTableRouter
from rest.routers.gen_train_table import GenerateTrainTableRouter
from rest.routers.graphs_v2 import GraphRouter
from rest.routers.groups import GroupsRouter
from rest.routers.identities import IdentitiesResource
from rest.routers.jobs import JobsResource
from rest.routers.kumo_tensorboard import TensorboardRouter
from rest.routers.log import LogResource
from rest.routers.predictive_queries_v2 import PredictiveQueryV2Router
from rest.routers.projects import ProjectsRouter
from rest.routers.snapshots import SnapshotResource
from rest.routers.tables_v2 import TableRouter
from rest.routers.training_jobs import TrainingJobsResource
from rest.settings import (
    AppSettings,
    get_db_storage_root,
    parse_cors_allowed_origins,
    parse_from_env_vars,
    validate_cors_allowed_origins,
)
from rest.utils.ambient_identity import ensure_startup_ambient_identities
from rest.utils.json_utils import NonStrictJSONResponse
from rest.utils.profiling_middleware import ProfilingMiddleware
from rest.utils.request_time_middleware import RequestTimeMiddleware
from rest.utils.security_header_middleware import SecurityHeaderMiddleware
from rest.utils.strip_header_middleware import StripHeaderMiddleware
from service_framework.deploy_env import DeployStage
from service_framework.deploy_env_config import deploy_env as deploy_env_v2
from statedb.db import StateDB, StateServices, open_state_db_from_deploy_config
from temporalio_commons import TemporalClient

# Identity-backed connector execution always rechecks its principal inside an
# activity, independent of request authentication. Install
# identity propagation before any Temporal client or worker is created.

log = logging.getLogger(__name__)
instrumentor = Instrumentator()
instrumentor.add(
    metrics.default(latency_lowr_buckets=(0.2, 0.5, 1, 2, 5, 10, 30, 60)))


async def _state_db_health_response(
    state_services: Optional[StateServices], ) -> Response:
    if state_services is None:
        return NonStrictJSONResponse(
            {
                "status": "starting",
                "state_db": None
            },
            status_code=503,
        )
    state_db_status = await state_services.status.get_schema_status()
    if not state_db_status.ok:
        return NonStrictJSONResponse(
            {
                "status": "unhealthy",
                "state_db": dataclasses.asdict(state_db_status),
            },
            status_code=503,
        )
    return NonStrictJSONResponse({"status": "ok"})


class AppBuilder:
    KUMO_REST_API_VERSION = "0.0.1"

    def __init__(
        self,
        settings: AppSettings,
        auth: Authentication,
        driver: RestApiDriver,
        temporal_client: TemporalClient,
        ml_backend: MLJobBackend,
        monitoring_settings: MonitoringSettings,
        projects_driver: Optional[ProjectsDriver] = None,
        direct_connect_auth_store: Optional[DirectConnectAuthStore] = None,
    ):
        self._settings = settings
        self._auth = auth
        if auth.settings.disable:
            log.warning(
                'Authentication is DISABLED (DISABLE_AUTH=true): every request '
                'is treated as a customer administrator. This is insecure and '
                'must never be used on a network-reachable deployment.')
        self._driver = driver
        self._temporal_client = temporal_client
        self._ml_backend = ml_backend
        self._projects_driver = projects_driver
        self._direct_connect_auth_store = direct_connect_auth_store
        self._state_db: Optional[StateDB] = None
        self._state_services: Optional[StateServices] = None
        # DB-backed admission queue, built lazily once the StateDB is opened
        # in `_setup` (used only by the debug-status endpoint here).
        self._job_queue: Optional[StateDbJobQueue] = None

        api_key_driver = APIKeyDriver(
            direct_connect_auth_store=direct_connect_auth_store)
        if direct_connect_auth_store is not None:
            group_resolver = DirectConnectGroupResolver(
                auth_store=direct_connect_auth_store)
        else:
            group_resolver = NativeGroupResolver()

        project_resolver = ProjectResolver()

        self._api_key_driver = api_key_driver
        self._group_resolver = group_resolver
        self._project_resolver = project_resolver

        auth_rsc = AuthenticationResource(auth)
        direct_connect_auth_rsc = (DirectConnectAuthResource(
            direct_connect_auth_store,
            group_resolver=(group_resolver if isinstance(
                group_resolver, DirectConnectGroupResolver) else None),
            login_method=auth.settings.direct_connect_login_method,
            allowed_return_origins=(
                auth.settings.direct_connect_allowed_return_origins),
            password_login_ip_rate_limit_enabled=(
                auth.settings.
                direct_connect_password_login_ip_rate_limit_enabled),
        ) if direct_connect_auth_store is not None else None)
        api_keys_v2_rsc = UserAPIKeyResource(auth, api_key_driver)
        self._groups_driver = GroupsDriver()
        groups_driver = self._groups_driver

        projects_driver = self._projects_driver
        if projects_driver is None:
            projects_driver = ProjectsDriver()
        self._projects_driver = projects_driver

        driver.table_driver._projects_driver = projects_driver
        driver.graph_driver._projects_driver = projects_driver
        conn_rsc = ConnectorResource(auth, driver.connector_driver)
        identities_rsc = IdentitiesResource(auth, driver.connector_driver)
        constants_rsc = ConstantsResource(auth)
        tensorboard_rsc = TensorboardRouter(auth, driver.tensorboard_driver)
        log_rsc = LogResource(auth)
        eval_rsc = EvaluationStatisticsResource(auth, driver,
                                                driver.pquery_driver)
        graph_router_2 = GraphRouter(
            auth,
            driver.graph_driver,
        )
        table_router_2 = TableRouter(
            auth=auth,
            table_driver=driver.table_driver,
        )
        fixtures_rsc = FixturesResource(
            auth,
            driver,
            table_router_v2=table_router_2,
            graph_router_v2=graph_router_2,
        )
        jobs_rsc = JobsResource(auth, driver.pquery_driver, self._ml_backend)
        projects_rsc = ProjectsRouter(auth, projects_driver)
        training_jobs_rsc = TrainingJobsResource(
            auth,
            driver,
            driver.pquery_driver,
            driver._jobs_driver,
            self._ml_backend,
        )
        bp_jobs_v1_resource = BatchPredictionV1Resource(
            auth,
            jobs_driver=driver._jobs_driver,
            data_driver=driver,
            ml_job_backend=ml_backend,
        )

        groups_rsc = GroupsRouter(auth, groups_driver)
        admin_rsc = AdminResource(
            auth,
            direct_connect_auth_store,
            direct_connect_login_method=(
                auth.settings.direct_connect_login_method),
        )
        snapshot_rsc = SnapshotResource(auth, driver.data_snapshot_api_driver,
                                        self._ml_backend)
        gen_train_table_rsc = GenerateTrainTableRouter(auth,
                                                       driver.pquery_driver,
                                                       self._ml_backend)
        gen_pred_table_rsc = GeneratePredictionTableRouter(
            auth, driver.pquery_driver, self._ml_backend)

        predictive_query_v2_router = PredictiveQueryV2Router(
            auth,
            driver.pquery_driver,
            projects_driver=projects_driver,
        )

        # Disable /openapi.json, /docs and /redoc endpoints in production
        # mode.
        if deploy_env_v2().deploy_mode == DeployStage.PROD:
            self._app = FastAPI(default_response_class=NonStrictJSONResponse,
                                debug=self._settings.debug, openapi_url=None)
        else:
            self._app = FastAPI(default_response_class=NonStrictJSONResponse,
                                debug=self._settings.debug)
        instrumentor.instrument(self._app)
        self._app.add_api_route("/", self.root, methods=['GET'])
        self._app.add_api_route("/health", self.health, methods=['GET'],
                                include_in_schema=False)

        async def _version_endpoint() -> JSONResponse:
            return JSONResponse({"version": get_kumo_version()})

        self._app.add_api_route("/version", _version_endpoint, methods=['GET'])
        self._app.include_router(auth_rsc.router)
        if direct_connect_auth_rsc is not None:
            self._app.include_router(direct_connect_auth_rsc.router)
        self._app.include_router(admin_rsc.router)
        self._app.include_router(api_keys_v2_rsc.router)
        self._app.include_router(conn_rsc.router)
        self._app.include_router(identities_rsc.router)
        self._app.include_router(identities_rsc.project_router)
        self._app.include_router(constants_rsc.router)
        self._app.include_router(fixtures_rsc.router)
        self._app.include_router(tensorboard_rsc.router)
        self._app.include_router(eval_rsc.router)
        self._app.include_router(log_rsc.router)
        self._app.include_router(jobs_rsc.router)
        self._app.include_router(training_jobs_rsc.router)
        self._app.include_router(snapshot_rsc.router)
        self._app.include_router(gen_train_table_rsc.router)
        self._app.include_router(gen_pred_table_rsc.router)
        self._app.include_router(graph_router_2.router)
        self._app.include_router(predictive_query_v2_router.router)
        self._app.include_router(table_router_2.router)
        self._app.include_router(bp_jobs_v1_resource.router)
        self._app.include_router(groups_rsc.router)
        self._app.include_router(groups_rsc.v1_router)
        self._app.include_router(projects_rsc.router)

        for route in self._app.routes:
            if isinstance(route, APIRoute):
                func = route.dependant.call
                if func and inspect.iscoroutinefunction(func):
                    log.debug(
                        'Enable asyncio blocking detection on path %s: %s',
                        route.path, route.name)
                    route.dependant.call = measure_blocking_code(func)

        self._app.on_event("startup")(self.startup_event)
        self._app.on_event("shutdown")(self.shutdown_event)

        allow_credentials = not auth.settings.disable
        allow_origins = parse_cors_allowed_origins(settings.middleware_origin)
        validate_cors_allowed_origins(allow_origins,
                                      allow_credentials=allow_credentials)
        self._cors_app = CORSMiddleware(
            self._app,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Expose resolvers on app state for the authorization middleware
        self._app.state.group_resolver = group_resolver
        self._app.state.project_resolver = project_resolver

        authorization_context_middleware: partial[Coroutine[
            Any, Any,
            Any]] = partial(authorization_middleware,
                            auth_disabled=auth.settings.disable,
                            auth_proxied=auth.settings.proxy,
                            is_unit_test=self.settings.is_unit_test,
                            group_resolver=group_resolver,
                            project_resolver=project_resolver,
                            api_key_driver=self._api_key_driver,
                            direct_connect_auth_store=direct_connect_auth_store)
        self._app.middleware("http")(authorization_context_middleware)

        # Print slow requests to the log
        self._app.add_middleware(
            RequestTimeMiddleware,
            exclude_prefixes=[
                "/v0/fixtures/",
                "/metrics",
            ],
            threshold_sec=10,
        )

        # Strip same-subdomain www-authenticate header:
        self._app.add_middleware(
            StripHeaderMiddleware,
            headers=["WWW-Authenticate"],
            exclude_paths=["/openapi.json", "/docs", "/redoc"],
        )

        # Support request profiling (enabled by profile=true query param)
        self._app.add_middleware(ProfilingMiddleware)

        # Add more security recommendations:
        self._app.add_middleware(SecurityHeaderMiddleware)

        async def validation_exception_handler(request: Request,
                                               exc: RequestValidationError):
            # ``exc.errors()`` may embed a raw exception object under
            # ``ctx``; encode it before returning the structured 422.
            return JSONResponse(content=jsonable_encoder(exc.errors()),
                                status_code=422)

        async def public_exception_handler(request: Request,
                                           exc: PublicException):
            return JSONResponse(
                status_code=exc.http_code, content={
                    "error": exc.get_user_message(),
                    "error_type": exc.__class__.__name__,
                })

        async def core_public_exception_handler(request: Request,
                                                exc: Exception):
            platform_error = as_platform_public_exception(exc)
            assert platform_error is not None
            return await public_exception_handler(request, platform_error)

        async def optimistic_lock_exception_handler(request: Request,
                                                    exc: OptimisticLockError):
            return JSONResponse(
                status_code=409,  # Conflict
                content={
                    "error":
                    str(exc),
                    "error_type":
                    "OptimisticLockError",
                    "retry":
                    True,
                    "message":
                    "The resource was modified by another request. "
                    "Please retry your operation.",
                })

        self._app.add_exception_handler(RequestValidationError,
                                        validation_exception_handler)
        self._app.add_exception_handler(PublicException,
                                        public_exception_handler)
        self._app.add_exception_handler(PublicError,
                                        core_public_exception_handler)
        self._app.add_exception_handler(TrainingValidationError,
                                        core_public_exception_handler)
        self._app.add_exception_handler(OptimisticLockError,
                                        optimistic_lock_exception_handler)

        # Initialize OpenTelemetry tracing for requests to this API
        if monitoring_settings.init_tracing:
            tracing_provider = tracing.init_tracing('kumo-rest',
                                                    monitoring_settings)
            tracing.connect_fastapi(self._app, tracing_provider)

    @classmethod
    def build_from_env_vars(cls) -> 'AppBuilder':
        settings = parse_from_env_vars()

        deploy_config = settings.deploy_config
        temporal_client = TemporalClient(deploy_config)

        # Initialize the credential encryptor for the customer.
        CredentialsEncryptor(deploy_mode=deploy_env_v2().deploy_mode)

        # Set up Data Backend
        data_backend_ctx = DataBackendContext(deploy_config,
                                              settings.data_backend_settings)

        # Set up ML Backend in-process (no AutoRPC client). Its StateDB-backed
        # services are late-bound in startup_event once the StateDB is open.
        ml_backend = MLJobBackend(MLBackendSettings.parse_from_env(),
                                  temporal_client)

        auth = Authentication(settings.authentication,
                              is_unit_test=settings.app.is_unit_test)

        # Set up REST API Driver.
        projects_driver = ProjectsDriver()

        driver = RestApiDriver(
            settings.rest_api_driver,
            auth,
            ml_backend,
            data_backend_ctx,
            projects_driver=projects_driver,
        )

        direct_connect_auth_store = DirectConnectAuthStore()

        return cls(settings.app, auth, driver, temporal_client, ml_backend,
                   settings.monitoring_settings, projects_driver,
                   direct_connect_auth_store)

    @property
    def settings(self) -> AppSettings:
        return self._settings

    @property
    def app(self) -> FastAPI:
        return self._app

    @property
    def cors_app(self) -> ASGIApp:
        return self._cors_app

    @property
    def db(self) -> RestApiDriver:
        return self._driver

    async def root(self) -> Response:
        content = {"Kumo Rest API": f"{self.KUMO_REST_API_VERSION}"}
        return NonStrictJSONResponse(content)

    async def health(self) -> Response:
        return await _state_db_health_response(self._state_services)

    async def startup_event(self):
        Connector.disable_interactive_input()
        await self._temporal_client.init()
        # Tests and embedded deployments may inject a fully configured
        # StateServices through RestApiDriver.  Preserve that single source of
        # truth instead of silently opening a second database and rebinding
        # every driver away from the caller's store.  Normal production startup
        # has no injected services and owns the database opened here.
        injected_services = getattr(self._driver, '_state_services', None)
        if injected_services is not None:
            self._state_services = injected_services
        else:
            self._state_db = await open_state_db_from_deploy_config(
                deploy_env_v2(),
                state_root=get_db_storage_root(),
            )
            self._state_services = StateServices(self._state_db)
        state_db = self._state_services.state_db
        # Late-bind this app's in-process MLJobBackend to the active StateDB and
        # start it read-only (the ml-workflow-worker is the single lifecycle
        # owner that runs reconciliation/admission recovery/migration).
        self._ml_backend._configure_state_db(state_db)
        await self._ml_backend.start(read_only=True)
        # DB-backed admission queue over the active StateDB (read-only debug
        # use here; the workflow worker owns admission/release).
        self._job_queue = StateDbJobQueue(state_db,
                                          MLJobQueueSettings.parse_from_env())
        self._api_key_driver.configure_state_services(self._state_services)
        self._driver._configure_state_services(self._state_services)
        self._groups_driver._configure_state_services(self._state_services)
        self._projects_driver._configure_state_services(self._state_services)
        default_group, _ = (
            await self._state_services.groups.ensure_default_resources())
        await ensure_startup_ambient_identities(
            self._state_services.identities,
            default_group.id,
            auth_disabled=self._auth.settings.disable,
        )
        if self._group_resolver is not None and hasattr(
                self._group_resolver, "configure_state_services"):
            self._group_resolver.configure_state_services(self._state_services)
        if self._project_resolver is not None:
            self._project_resolver.configure_state_services(
                self._state_services)
        if (self._direct_connect_auth_store is not None
                and not self._auth.settings.disable):
            auth_store = self._direct_connect_auth_store
            auth_store.configure_auth_service(
                self._state_services.direct_connect_auth)
            await auth_store.init_once()
            direct_connect_config = resolve_direct_connect_runtime_config(
                deploy_env_v2())
            if direct_connect_config.login_method == 'oidc':
                await auth_store.seed_or_reconcile_oidc_config(
                    issuer=direct_connect_config.oidc.issuer,
                    client_id=direct_connect_config.oidc.client_id,
                    client_secret=direct_connect_config.oidc.client_secret,
                    groups_claim=direct_connect_config.oidc.groups_claim,
                    admin_emails=list(direct_connect_config.admin_emails),
                    google_sa_key=direct_connect_config.oidc.google_sa_key,
                    google_admin_email=(
                        direct_connect_config.oidc.google_admin_email),
                )
                # Dev-only escape hatch (devserver `oidc` variant): IdPs like
                # Entra ID omit `email_verified`, which blocks first-admin
                # bootstrap. When KUMO_DEV_TRUST_ADMIN_EMAILS=1, force the
                # configured admin_emails to role=admin. Gated on a dev
                # deploy-stage so a stray env var can't grant admin in prod.
                if os.getenv('KUMO_DEV_TRUST_ADMIN_EMAILS') == '1':
                    if deploy_env_v2().deploy_mode.is_dev:
                        await auth_store.seed_oidc_admins(admin_emails=list(
                            direct_connect_config.admin_emails))
                    else:
                        log.warning(
                            'Ignoring KUMO_DEV_TRUST_ADMIN_EMAILS=1 outside a '
                            'dev deploy-stage (deploy_mode=%s).',
                            deploy_env_v2().deploy_mode.value)
            if direct_connect_config.login_method == 'password':
                await auth_store.warm_password_reset_token_lookup_key()
                await auth_store.seed_local_admins(
                    admin_emails=list(direct_connect_config.admin_emails),
                    temporary_password=(
                        direct_connect_config.local_admin_temp_password),
                )
            await auth_store.cleanup_expired_sessions()
            await auth_store.cleanup_expired_password_reset_tokens()
            await auth_store.cleanup_expired_password_login_attempts()

        # StateDB capability retention has a bounded periodic owner in every
        # REST replica. PostgreSQL replicas coordinate with an advisory lock;
        # SQLite writes are serialized by StateDB.
        await self._state_services.connector_authorizations.start_maintenance()

        # Startup init is once-only; start the event-loop stall watchdog now
        # so steady-state stalls past this point are what we flag.
        install_loop_guards()

        log.info('REST App startup is completed.')

    async def shutdown_event(self):
        log.info('Starting REST App shutdown.')

        if self._state_services is not None:
            await self._state_services.connector_authorizations.stop_maintenance(
            )

        if self._direct_connect_auth_store is not None:
            await self._direct_connect_auth_store.close()

        if self._state_db is not None:
            await self._state_db.close()
            self._state_db = None
            log.info('Closed StateDB connections.')

        # Release Temporal gRPC connections so the event loop can shut down
        # cleanly.  Must be last because earlier steps may still use gRPC.
        await self._temporal_client.close()
        log.info('Closed Temporal client connections.')
