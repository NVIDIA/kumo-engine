# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils import kumo_dataclass
from service_framework.env_var import EnvVar

_JAEGER_HOSTNAME = EnvVar('JAEGER_HOSTNAME', str, required=False,
                          default_value='localhost')
# Jaeger supports OTLP directly. The default port for OTLP/gRPC is 4317
_JAEGER_PORT = EnvVar('JAEGER_PORT', int, required=False, default_value=4317)


@kumo_dataclass
class MonitoringSettings:
    jaeger_hostname: str = 'localhost'
    # Jaeger supports OTLP directly. The default port for OTLP/gRPC is 4317
    jaeger_port: int = 4317
    init_tracing: bool = True

    @classmethod
    def parse_from_env(cls) -> 'MonitoringSettings':
        return MonitoringSettings(
            jaeger_hostname=_JAEGER_HOSTNAME.parse_value(),
            jaeger_port=_JAEGER_PORT.parse_value(),
        )
