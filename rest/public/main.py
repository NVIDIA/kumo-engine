# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import faulthandler
import logging.config
import os
from logging import LogRecord

import uvicorn
from prometheus_client import start_http_server

from io_utils.logging import LoggingMode
from kumo.runtime import init
from monitoring.settings import MonitoringSettings
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.backends.ml_backend.settings import MLBackendSettings
from rest.config.encryption import CredentialsEncryptor
from rest.deps.authentication import Authentication
from rest.driver.groups_driver import GroupsDriver
from rest.driver.projects_driver import ProjectsDriver
from rest.driver.rest_api_driver import RestApiDriver
from rest.public.app import AppBuilder
from rest.settings import _DISABLE_AUTH, _PROXY_AUTH, parse_from_env_vars
from rest.utils.request_time_middleware import SensitiveQueryLogFilter
from service_framework.deploy_env_config import deploy_env
from service_framework.env_var import EnvVar
from temporalio_commons.runtime_utils import init_temporal_runtime
from temporalio_commons.temporal_client import TemporalClient

# Env variable names:
_REST_API_HOST = EnvVar('REST_API_HOST', str, required=False,
                        default_value='0.0.0.0')
_REST_API_PORT = EnvVar('REST_API_PORT', int, default_value=8001)
_METRICS_PORT = EnvVar('METRICS_PORT', int, required=False, default_value=9100)
_METRICS_HOST = EnvVar('METRICS_HOST', str, required=False, default_value='::')
_DRIVER_RPC_TIMEOUT_SEC = EnvVar('DRIVER_RPC_TIMEOUT_SEC', int,
                                 default_value=180)
_ACCESS_LOG = EnvVar('UVICORN_ACCESS_LOG', bool, required=False,
                     default_value=True)
_APP_LOG_LEVEL = EnvVar('KUMO_REST_MAIN_LOG_LEVEL', str, required=False,
                        default_value='DEBUG')

# Dump the stacktrace when segfault occurs
faulthandler.enable()


# The public REST API is a thin, single-tenant service with one process-wide
# pquery driver.
def create_app_builder() -> AppBuilder:
    deploy_env_config = deploy_env()

    if 'RPC_TIMEOUT_DEFAULT' not in os.environ:
        os.environ['RPC_TIMEOUT_DEFAULT'] = str(
            _DRIVER_RPC_TIMEOUT_SEC.parse_value())

    # Public REST hosts product-metadata drivers in-process, backed by its own
    # StateDB (configured in AppBuilder.startup_event). The metadata driver
    # dependency graph is assembled via a single RestApiDriver, the same way as
    # rest/app.py:build_from_env_vars, so
    # connector/table/graph/pquery/data-snapshot/artifact
    # share the same wiring and scale limits.
    #
    # MLJobBackend is also constructed in-process (no AutoRPC) and shared by
    # both the RestApiDriver (which builds the pquery/artifact
    # drivers) and the public AppBuilder's job routers.
    settings = parse_from_env_vars()

    # Initialize the credential encryptor for the customer (mirrors the main
    # REST app); connector drivers decrypt stored credentials in-process.
    CredentialsEncryptor(deploy_mode=deploy_env_config.deploy_mode)

    data_backend_ctx = DataBackendContext(deploy_env_config,
                                          settings.data_backend_settings)

    # In-process MLJobBackend (no AutoRPC client), shared by the RestApiDriver
    # and the public job routers. Its StateDB is late-bound in
    # AppBuilder.startup_event.
    temporal_client = TemporalClient(deploy_env_config)
    ml_backend = MLJobBackend(MLBackendSettings.parse_from_env(),
                              temporal_client)

    auth = Authentication(settings.authentication, is_unit_test=False)

    # The public AppBuilder owns request-time membership resolution. Project
    # CRUD operates directly on StateServices configured during startup.
    projects_driver = ProjectsDriver()

    driver = RestApiDriver(
        settings.rest_api_driver,
        auth,
        ml_backend,
        data_backend_ctx,
        projects_driver=projects_driver,
    )

    groups_driver = GroupsDriver()

    auth_disabled = (_DISABLE_AUTH.parse_value()
                     or deploy_env_config.deploy_mode.is_preprod)
    auth_proxied = _PROXY_AUTH.parse_value()
    return AppBuilder(
        driver.pquery_driver,
        driver.connector_driver,
        driver.artifact_driver,
        driver.data_snapshot_api_driver,
        driver.table_driver,
        driver.graph_driver,
        ml_backend,
        MonitoringSettings.parse_from_env(),
        auth_disabled=auth_disabled,
        auth_proxied=auth_proxied,
        groups_driver=groups_driver,
        projects_driver=projects_driver,
    )


app_builder = create_app_builder()
app = app_builder.cors_app


class IgnoreMetrics:
    KEYWORDS = ["/metrics", "200"]

    def filter(self, record: LogRecord):
        message = record.getMessage()
        return any(k not in message for k in self.KEYWORDS)


_app_log_level = _APP_LOG_LEVEL.parse_value().upper()
LOGGING_CONFIG = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)  # type: ignore
LOGGING_CONFIG["loggers"]["kumo"] = {
    "handlers": ["default"],
    "level": _app_log_level,
}
LOGGING_CONFIG["loggers"]["rest"] = {
    "handlers": ["default"],
    "level": _app_log_level,
}
if _app_log_level != 'DEBUG':
    LOGGING_CONFIG["root"] = {
        "handlers": ["default"],
        "level": _app_log_level,
    }
    for _logger_name in (
            "async_utils",
            "common_utils",
            "io_utils",
            "service_framework",
            "temporalio_commons",
    ):
        LOGGING_CONFIG["loggers"][_logger_name] = {
            "handlers": ["default"],
            "level": _app_log_level,
        }
    LOGGING_CONFIG["loggers"]["opentelemetry"] = {
        "handlers": ["default"],
        "level": "CRITICAL",
    }
LOGGING_CONFIG["filters"] = {
    "ignore_metrics": {
        "()": IgnoreMetrics
    },
    "redact_sensitive_query": {
        "()": SensitiveQueryLogFilter
    },
}
LOGGING_CONFIG["handlers"]["access"]["filters"] = ["redact_sensitive_query"]
LOGGING_CONFIG["loggers"]["uvicorn.access"]["filters"] = ["ignore_metrics"]

if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    init(
        'public-rest-api',
        logging_mode=(LoggingMode.DEBUG
                      if _app_log_level == 'DEBUG' else LoggingMode.PRODUCTION),
    )
    start_http_server(_METRICS_PORT.parse_value(),
                      addr=_METRICS_HOST.parse_value())

    # Init Temporal SDK runtime (client & worker telemetry).
    init_temporal_runtime(app_name='public-rest-api')

    uvicorn.run(
        "main:app",
        host=_REST_API_HOST.parse_value(),
        port=_REST_API_PORT.parse_value(),
        timeout_keep_alive=15,
        log_config=LOGGING_CONFIG,
        workers=1,
        access_log=_ACCESS_LOG.parse_value(),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
