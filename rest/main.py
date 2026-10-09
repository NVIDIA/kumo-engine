# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import faulthandler
import logging.config
import os
from copy import deepcopy
from logging import LogRecord

import uvicorn
from prometheus_client import start_http_server

from common_utils.startup import log_app_startup
from io_utils.logging import LoggingMode
from kumo.runtime import init
from rest.app import AppBuilder
from rest.utils.frontend_proxy_metadata import ExternalHostMiddleware
from rest.utils.request_time_middleware import SensitiveQueryLogFilter
from service_framework.deploy_env_config import deploy_env
from service_framework.env_var import EnvVar
from temporalio_commons.runtime_utils import init_temporal_runtime

_METRICS_PORT = EnvVar('METRICS_PORT', int, required=False, default_value=9100)
_METRICS_HOST = EnvVar('METRICS_HOST', str, required=False, default_value='::')
_ACCESS_LOG = EnvVar('UVICORN_ACCESS_LOG', bool, required=False,
                     default_value=True)
_APP_LOG_LEVEL = EnvVar('KUMO_REST_MAIN_LOG_LEVEL', str, required=False,
                        default_value='DEBUG')
_app_log_level = _APP_LOG_LEVEL.parse_value().upper()
LOGGING_CONFIG = deepcopy(uvicorn.config.LOGGING_CONFIG)  # type: ignore
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


class IgnoreMetrics:
    KEYWORDS = ["/metrics", "200"]

    def filter(self, record: LogRecord):
        message = record.getMessage()
        return any(k not in message for k in self.KEYWORDS)


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

# Dump the stacktrace when segfault occurs
faulthandler.enable()

#######################################

app_builder = AppBuilder.build_from_env_vars()
app = app_builder.cors_app
if external_host := deploy_env().external_host:
    app = ExternalHostMiddleware(app, external_host)

if __name__ == "__main__":
    init(
        'rest-api',
        logging_mode=(LoggingMode.DEBUG
                      if _app_log_level == 'DEBUG' else LoggingMode.PRODUCTION),
    )
    log_app_startup()

    # Init Temporal SDK runtime (client & worker telemetry).
    init_temporal_runtime(app_name='rest-api')

    start_http_server(_METRICS_PORT.parse_value(),
                      addr=_METRICS_HOST.parse_value())
    # Start the REST server
    # For V0.1, we limit # of workers to 1, because we need to ensure that the
    # Compute Engine is a singleton, as it maintains in-memory state that must
    # be shared across all requests. This restriction can be removed once the
    # Compute Engine is moved to an external process.
    uvicorn.run(
        "main:app",
        host=app_builder.settings.server_host,
        port=app_builder.settings.port,
        log_config=LOGGING_CONFIG,
        timeout_keep_alive=15,
        workers=1,
        access_log=_ACCESS_LOG.parse_value(),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
