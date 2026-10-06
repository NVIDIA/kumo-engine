# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used by "make devserver" in order to initialize temporal.
# See rest/devserver/main.py for more info.
#
# Confirm that we are able to connect to temporal cloud. This requires
# downloading the proper TLS certificate from KMS.
import argparse
import asyncio
import os

from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from temporalio_commons import TemporalClient


async def init_temporal(deploy_mode: DeployStage):
    os.environ['DEPLOY_STAGE'] = deploy_mode.value
    deploy_env_config = DeployEnvConfig.parse_from_env(app_name='devserver',
                                                       deploy_mode=deploy_mode)
    await TemporalClient(deploy_env_config).init()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_mode")
    args = parser.parse_args()

    deploy_mode = DeployStage(args.deploy_mode)

    asyncio.run(init_temporal(deploy_mode))
