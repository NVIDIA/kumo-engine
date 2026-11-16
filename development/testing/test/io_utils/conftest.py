# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test fixtures for io_utils tests."""

import os
import uuid

import pytest

from service_framework.deploy_env_config import DeployEnvConfig


@pytest.fixture(scope="session")
def deploy_env_config() -> DeployEnvConfig:
    os.environ['DEPLOY_STAGE'] = 'test'
    os.environ['INSTANCE_ID'] = f"pytest-{uuid.uuid4().hex[:8]}"
    return DeployEnvConfig.for_test()
