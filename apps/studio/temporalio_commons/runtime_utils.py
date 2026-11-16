# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ipaddress import ip_address
from typing import Optional

from kumo_core.version import get_kumo_version
from temporalio.runtime import PrometheusConfig, Runtime, TelemetryConfig

from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from service_framework.env_var import EnvVar

_WORKER_METRICS_PORT = EnvVar(
    'WORKER_METRICS_PORT',
    int,
    required=False,
    default_value=9000,
    description='Port to allow metrics scraping from Promethus.',
).parse_value()
_WORKER_METRICS_HOST = EnvVar(
    'WORKER_METRICS_HOST',
    str,
    required=False,
    default_value='[::]',
    description=(
        'Host to allow metrics scraping from Promethus. Default is "[::]"'),
).parse_value()


def init_temporal_runtime(
    app_name: str,
    deploy_config: Optional[DeployEnvConfig] = None,
    *,
    host: str = _WORKER_METRICS_HOST,
    port: int = _WORKER_METRICS_PORT,
):
    """
    Utility function to initialize Temporal SDK runtime with telemetry/metrics
    meter. This function should be called at the very beginning of any
    application that uses Temporal SDK.
    """
    deploy_config = deploy_config or deploy_env()
    assert deploy_config
    global_tags = {
        'stage': deploy_config.deploy_mode.value,  # 'prod', 'dev', 'preprod'
        'instance_id': deploy_config.instance_id,
        'app': deploy_config.client_id,
        'kumo_version': get_kumo_version(),
    }

    host = host or _WORKER_METRICS_HOST
    port = port or _WORKER_METRICS_PORT

    # Ensure IPv6 address is wrapped in brackets
    raw_host = host.strip('[]')
    addr = ip_address(raw_host)
    if addr.version == 6:
        host = f'[{raw_host}]'

    Runtime.set_default(
        Runtime(telemetry=TelemetryConfig(
            metrics=PrometheusConfig(bind_address=f"{host}:{port}"),
            global_tags=global_tags,
            # Not useful, service name is always "temporal-core-sdk"
            attach_service_name=False,
        )))
