# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import partial
from typing import Any, Coroutine, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from kumo_core.errors import PublicError, TrainingValidationError
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.types import ASGIApp

from async_utils.instrument import measure_blocking_code
from async_utils.loop_guards import install_loop_guards
from common_utils.error import PublicException
from io_utils import logging
from kumo.connector.connector import Connector
from kumo.platform_errors import as_platform_public_exception
from monitoring.settings import MonitoringSettings
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config import desc
from rest.config.authentication import AuthenticationSettings
from rest.config.status import EmptyResponse
from rest.deps.authentication import (
    Authentication,
    authorization_middleware_public,
)
from rest.deps.db_common import OptimisticLockError
from rest.deps.direct_connect_group_resolver import DirectConnectGroupResolver
from rest.deps.project_resolver import ProjectResolver
from rest.driver.api_key_driver import APIKeyDriver
from rest.driver.artifact_api_driver import ArtifactApiDriver
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.driver.data_snapshot_api_driver import DataSnapshotApiDriver
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from rest.driver.graph_driver import GraphDriver
from rest.driver.groups_driver import GroupsDriver
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.driver.projects_driver import ProjectsDriver
from rest.driver.table_driver import TableDriver
from rest.public.routers.artifact import ArtifactRouterV1
from rest.public.routers.authentication import AuthenticationRouter
from rest.public.routers.common import APPLIED_TAG_FILTERS_HEADER
from rest.public.routers.connector import ConnectorRouter
from rest.public.routers.data_snapshot import (
    GraphSnapshotRouter,
    TableSnapshotRouter,
)
from rest.public.routers.gen_pred_table import GeneratePredictionTableRouter
from rest.public.routers.gen_train_table import GenerateTrainTableRouter
from rest.public.routers.graph import GraphRouter
from rest.public.routers.groups import GroupsRouter
from rest.public.routers.identity import IdentityRouter
from rest.public.routers.pquery import PQueryRouter
from rest.public.routers.prediction import PredictionRouterV1
from rest.public.routers.projects import ProjectsRouter
from rest.public.routers.source_table import SourceTableRouter
from rest.public.routers.table import TableRouter
from rest.public.routers.training import TrainingRouterV1
from rest.routers.authentication import api_key_auth, get_api_key
from rest.settings import get_db_storage_root
from rest.utils import tracing
from rest.utils.ambient_identity import ensure_startup_ambient_identities
from rest.utils.json_utils import NonStrictJSONResponse
from rest.utils.request_time_middleware import RequestTimeMiddleware
from rest.utils.security_header_middleware import SecurityHeaderMiddleware
from service_framework.deploy_env_config import deploy_env as deploy_env_v2
from statedb.db import StateDB, StateServices, open_state_db_from_deploy_config

log = logging.getLogger(__name__)
instrumentor = Instrumentator()
instrumentor.add(metrics.default(latency_lowr_buckets=(1, 10, 30, 60), ))


def _build_cors_app(app: ASGIApp) -> ASGIApp:
    return CORSMiddleware(
        app,
        allow_origins=['*'],
        allow_credentials=False,
        allow_methods=['GET', 'POST', 'DELETE'],
        allow_headers=["*"],
        expose_headers=[APPLIED_TAG_FILTERS_HEADER],
    )


class AppBuilder:
    def __init__(
        self,
        driver: PQueryApiDriver,
        connector_driver: ConnectorApiDriver,
        artifact_api_driver: ArtifactApiDriver,
        data_snapshot_driver: DataSnapshotApiDriver,
        table_driver: TableDriver,
        graph_driver: GraphDriver,
        ml_job_backend: MLJobBackend,
        monitoring_settings: MonitoringSettings,
        groups_driver: GroupsDriver,
        projects_driver: ProjectsDriver,
        auth_disabled: bool = False,
        auth_proxied: bool = False,
        is_unit_test: bool = False,
        state_services: Optional[StateServices] = None,
    ):
        self._driver = driver
        self._connector_driver = connector_driver
        self._data_snapshot_driver = data_snapshot_driver
        self._table_driver = table_driver
        # Retain references to every product-metadata driver so startup_event
        # can configure each one's StateServices (these are direct in-process
        # driver instances built in rest/public/main.py).
        self._artifact_api_driver = artifact_api_driver
        self._graph_driver = graph_driver
        self._groups_driver = groups_driver
        self._projects_driver = projects_driver
        # In-process MLJobBackend (shared with the RestApiDriver); its StateDB
        # is late-bound in startup_event.
        self._ml_backend = ml_job_backend

        self._app = FastAPI(
            title='Kumo REST API',
            description=desc.overview,
            openapi_url=None,
            terms_of_service='https://kumo.ai/terms',
            contact={'email': 'hello@kumo.ai'},
            license_info=None,
            default_response_class=NonStrictJSONResponse,
            # A trailing-slash miss (e.g. GET /api/v1/projects/) must return a
            # clean 404 rather than a 307 redirect. Behind the reverse proxy the
            # redirect Location is reconstructed from this app's internal ASGI
            # scope (http://<host>:8080/...), which downgrades https->http, leaks
            # the internal uvicorn port, and is unreachable from the public
            # origin. redirect_slashes is applied on the app router because
            # include_router flattens sub-router routes into it, so a
            # per-sub-router flag would not take effect. See issue #29812.
            redirect_slashes=False,
        )
        instrumentor.instrument(self._app)
        self._auth_disabled = auth_disabled
        auth_settings = AuthenticationSettings(
            disable=auth_disabled,
            is_unit_test=is_unit_test,
            proxy=auth_proxied,
        )
        # API-key verification, group resolution, and project resolution run
        # in-process against this app's StateDB (configured in startup_event).
        # The API-key driver keeps no process-local success cache so key
        # deletion/revocation takes effect immediately.
        #
        # state_services may be injected (e.g. tests that share one StateDB
        # across the main and public apps); when None, startup_event opens this
        # process's own StateDB.
        self._state_db: Optional[StateDB] = None
        self._state_services: Optional[StateServices] = state_services
        self._direct_connect_auth_store = DirectConnectAuthStore()
        self._api_key_driver = APIKeyDriver(
            direct_connect_auth_store=self._direct_connect_auth_store)
        group_resolver = DirectConnectGroupResolver(
            auth_store=self._direct_connect_auth_store)
        project_resolver = ProjectResolver()
        self._group_resolver = group_resolver
        self._project_resolver = project_resolver

        self._app.add_api_route(
            "/",
            self.root,
            methods=['GET'],
            response_class=RedirectResponse,
            include_in_schema=False,
        )
        self._app.add_api_route(
            '/health',
            self.health,
            methods=['GET'],
            response_class=JSONResponse,
            include_in_schema=False,
        )

        self._api_key_auth = api_key_auth(
            api_key_driver=self._api_key_driver,
            group_resolver=group_resolver,
            project_resolver=project_resolver,
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
        )

        self._app.add_api_route(
            '/validate_api_key',
            self.validate_api_key,
            methods=['POST'],
            dependencies=[Depends(self._api_key_auth)],
            response_model=EmptyResponse,
            include_in_schema=False,
        )

        self._app.add_api_route(
            '/openapi.json',
            self.get_open_api_schema,
            methods=['GET'],
            dependencies=[Depends(self._api_key_auth)],
            response_class=JSONResponse,
            include_in_schema=False,
        )
        self._app.add_api_route(
            '/docs',
            self.get_documentation,
            methods=['GET'],
            dependencies=[Depends(self._api_key_auth)],
            response_class=HTMLResponse,
            include_in_schema=False,
        )
        self._app.add_api_route(
            '/redoc',
            self.get_redoc,
            methods=['GET'],
            dependencies=[Depends(self._api_key_auth)],
            response_class=HTMLResponse,
            include_in_schema=False,
        )

        auth = Authentication(settings=auth_settings, is_unit_test=is_unit_test)

        authorization_context_middleware: partial[Coroutine[
            Any, Any, Any]] = partial(
                authorization_middleware_public,
                auth_disabled=auth.settings.disable,
                auth_proxied=auth.settings.proxy,
                is_unit_test=is_unit_test,
                group_resolver=group_resolver,
                project_resolver=project_resolver,
                api_key_driver=self._api_key_driver,
            )
        self._app.middleware("http")(authorization_context_middleware)

        # Print slow requests to the log
        self._app.add_middleware(
            RequestTimeMiddleware,
            exclude_prefixes=["/metrics"],
            threshold_sec=10,
        )

        # Add security headers to all responses
        self._app.add_middleware(SecurityHeaderMiddleware)

        # Initialize OpenTelemetry tracing for requests to this API
        if monitoring_settings.init_tracing:
            tracing_provider = tracing.init_tracing('kumo-rest-public',
                                                    monitoring_settings)
            tracing.connect_fastapi(self._app, tracing_provider)

        # Job routers share the single in-process MLJobBackend (no AutoRPC).
        authentication_router = AuthenticationRouter(self._api_key_auth)
        artifact_router_v1 = ArtifactRouterV1(self._api_key_auth,
                                              artifact_api_driver, auth)
        training_router_v1 = TrainingRouterV1(self._api_key_auth, driver, auth,
                                              self._ml_backend)
        prediction_router_v1 = PredictionRouterV1(self._api_key_auth, driver,
                                                  auth, self._ml_backend)
        pquery_router = PQueryRouter(self._api_key_auth, driver, auth)
        connector_router = ConnectorRouter(self._api_key_auth, connector_driver)
        identity_router = IdentityRouter(self._api_key_auth, connector_driver,
                                         auth)
        source_table_router = SourceTableRouter(self._api_key_auth,
                                                connector_driver, auth)
        table_router = TableRouter(
            self._api_key_auth,
            table_driver,
            auth,
        )
        graph_router = GraphRouter(self._api_key_auth, graph_driver, auth)
        table_snapshot_router = TableSnapshotRouter(self._api_key_auth,
                                                    data_snapshot_driver, auth)
        graph_snapshot_router = GraphSnapshotRouter(self._api_key_auth,
                                                    data_snapshot_driver, auth)
        gen_train_table_router = GenerateTrainTableRouter(
            self._api_key_auth, driver, auth, self._ml_backend)
        gen_pred_table_router = GeneratePredictionTableRouter(
            self._api_key_auth, driver, auth, self._ml_backend)
        # V1 APIs: designed for SDK functionalities.
        self._app.include_router(authentication_router.router, prefix='/v1')
        self._app.include_router(artifact_router_v1.router, prefix='/v1')
        self._app.include_router(training_router_v1.router, prefix='/v1')
        self._app.include_router(prediction_router_v1.router, prefix='/v1')
        self._app.include_router(pquery_router.router, prefix='/v1')
        self._app.include_router(connector_router.router, prefix='/v1')
        self._app.include_router(identity_router.router, prefix='/v1')
        self._app.include_router(identity_router.admin_router, prefix='/v1')
        self._app.include_router(source_table_router.router, prefix='/v1')
        self._app.include_router(table_router.router, prefix='/v1')
        self._app.include_router(graph_router.router, prefix='/v1')
        self._app.include_router(table_snapshot_router.router, prefix='/v1')
        self._app.include_router(graph_snapshot_router.router, prefix='/v1')
        self._app.include_router(gen_train_table_router.router_v1, prefix='/v1')
        self._app.include_router(gen_pred_table_router.router, prefix='/v1')
        groups_router = GroupsRouter(self._api_key_auth, groups_driver, auth)
        self._app.include_router(groups_router.router, prefix='/v1')
        projects_router = ProjectsRouter(self._api_key_auth, projects_driver,
                                         auth)
        self._app.include_router(projects_router.router, prefix='/v1')

        for route in self._app.routes:
            if isinstance(route, APIRoute):
                func = route.dependant.call
                if func and inspect.iscoroutinefunction(func):
                    log.debug(
                        'Enable asyncio blocking detection on path %s: %s',
                        route.path, route.name)
                    route.dependant.call = measure_blocking_code(func)

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
            # OptimisticLockError is not a PublicException, so it needs its own
            # handler to return 409 + a retry signal on a concurrent-write
            # conflict (otherwise it would surface as a generic 500).
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

        self._app.add_exception_handler(PublicException,
                                        public_exception_handler)
        self._app.add_exception_handler(PublicError,
                                        core_public_exception_handler)
        self._app.add_exception_handler(TrainingValidationError,
                                        core_public_exception_handler)
        self._app.add_exception_handler(OptimisticLockError,
                                        optimistic_lock_exception_handler)

        self._app.on_event("startup")(self.startup_event)

        self._cors_app = _build_cors_app(self._app)

    @property
    def app(self) -> FastAPI:
        return self._app

    @property
    def cors_app(self) -> ASGIApp:
        return self._cors_app

    async def root(self) -> RedirectResponse:
        return RedirectResponse(url='redoc')

    async def health(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def validate_api_key(
        self, api_key: str = Depends(get_api_key)) -> EmptyResponse:
        # The dependency has already verified the key. Never log any portion
        # of the bearer credential, including structured prefixes or suffixes.
        log.info('Validated API key')
        return EmptyResponse()

    async def get_open_api_schema(self) -> JSONResponse:
        return JSONResponse(self._app.openapi())

    async def get_documentation(self) -> HTMLResponse:
        return get_swagger_ui_html(openapi_url="./openapi.json", title="docs")

    async def get_redoc(self) -> HTMLResponse:
        return get_redoc_html(openapi_url="./openapi.json", title="redoc")

    async def startup_event(self):
        Connector.disable_interactive_input()

        # Open this process's own StateDB for in-process metadata access.
        # Skipped when StateServices was injected (e.g. shared-DB tests).
        if self._state_services is None:
            self._state_db = await open_state_db_from_deploy_config(
                deploy_env_v2(),
                state_root=get_db_storage_root(),
            )
            self._state_services = StateServices(self._state_db)
        # Late-bind this app's in-process MLJobBackend to this process's StateDB
        # and start it read-only (the ml-workflow-worker is the single
        # lifecycle owner that runs reconciliation/admission/migration).
        self._ml_backend._configure_state_db(self._state_services.state_db)
        await self._ml_backend.start(read_only=True)
        default_group, _ = (
            await self._state_services.groups.ensure_default_resources())
        await ensure_startup_ambient_identities(
            self._state_services.identities,
            default_group.id,
            auth_disabled=self._auth_disabled,
        )
        self._api_key_driver.configure_state_services(self._state_services)
        if self._direct_connect_auth_store is not None:
            self._direct_connect_auth_store.configure_auth_service(
                self._state_services.direct_connect_auth)
        if self._group_resolver is not None and hasattr(
                self._group_resolver, "configure_state_services"):
            self._group_resolver.configure_state_services(self._state_services)
        if self._project_resolver is not None:
            self._project_resolver.configure_state_services(
                self._state_services)

        # Wire every product-metadata driver to this process's StateDB so its
        # route handlers read/write StateDB locally.
        for metadata_driver in (
                self._driver,
                self._connector_driver,
                self._data_snapshot_driver,
                self._table_driver,
                self._artifact_api_driver,
                self._graph_driver,
                self._groups_driver,
                self._projects_driver,
        ):
            if metadata_driver is not None:
                metadata_driver._configure_state_services(self._state_services)
        log.info('Public REST in-process auth + metadata StateDB initialized')

        # Startup init is once-only; start the event-loop stall watchdog now
        # so steady-state stalls past this point are what we flag.
        install_loop_guards()
