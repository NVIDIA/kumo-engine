# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from service_framework.deploy_env_config import DeployEnvConfig


@pytest.fixture
def temporal_config(temporal_workflow_env) -> DeployEnvConfig:
    target_host = (temporal_workflow_env.client.service_client.config.target_host
                   if temporal_workflow_env is not None else None)
    return DeployEnvConfig.for_test(temporal_target_host=target_host)
