# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils import kumo_dataclass
from rest.settings import DataBackendSettings
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import TemporalClient


@kumo_dataclass
class DataBackendContext:
    _config: DeployEnvConfig
    _settings: DataBackendSettings

    def config(self) -> DeployEnvConfig:
        return self._config

    def temporal_client(self) -> TemporalClient:
        return TemporalClient(self._config)
