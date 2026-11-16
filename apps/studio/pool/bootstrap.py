# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common boot sequence for pool entrypoints.

``controller_main`` and ``worker_main`` both initialize logging and
tracing in the same way before any other code touches deploy_env().
Centralized here so future entrypoints (e.g. singlehost) don't drift.
"""
import logging as stdlib_logging
from typing import Dict, Mapping, Optional

from io_utils import logging
from monitoring.settings import MonitoringSettings
from monitoring.tracing import (
    SpanExporterType,
    enable_tracing,
    initialize_tracing,
)
from service_framework.deploy_stage import DeployStage, deploy_stage

# Per-activity heartbeat loggers fire every ~10s for every in-flight
# RPC and dominate the controller pane in DEBUG mode. Pin to INFO so
# real signal stays visible. Devserver only; prod runs at INFO already.
_DEV_HEARTBEAT_LOG_OVERRIDES: Dict[str, int] = {
    'temporalio_commons.heartbeat_utils': stdlib_logging.INFO,
    'temporalio.worker._activity': stdlib_logging.INFO,
}


def init_logging_and_tracing(
    service_name: str,
    extra_log_overrides: Optional[Mapping[str, int]] = None,
) -> None:
    """Initialize logging + Jaeger tracing for a pool entrypoint.

    Logging is initialized BEFORE the first ``deploy_env()`` call so
    its singleton-init log lands in the configured handlers instead
    of the default root handler. DEPLOY_STAGE is read directly to
    pick the log mode so we don't trigger ``deploy_env()`` early.

    ``service_name`` is used as the OpenTelemetry resource service
    name for tracing — distinguishes spans from different pool
    entrypoints (controller vs worker).
    """
    is_prod = deploy_stage() == DeployStage.PROD
    overrides: Dict[str, int] = {}
    if not is_prod:
        overrides.update(_DEV_HEARTBEAT_LOG_OVERRIDES)
    if extra_log_overrides:
        overrides.update(extra_log_overrides)
    mode = (logging.LoggingMode.PRODUCTION
            if is_prod else logging.LoggingMode.DEBUG)
    logging.initialize_logging(
        mode=mode,
        log_to_file=False,
        logging_level_overrides=overrides,
    )

    # ``@trace``-decorated functions no-op (with a debug log per call)
    # unless tracing is initialized. Devserver's Jaeger pane and
    # prod's tracing stack both listen on ``MonitoringSettings``.
    monitoring_settings = MonitoringSettings.parse_from_env()
    enable_tracing()
    initialize_tracing(
        service_name,
        [SpanExporterType.JAEGER],
        jaeger_agent_host_name=monitoring_settings.jaeger_hostname,
        jaeger_port=monitoring_settings.jaeger_port,
    )
