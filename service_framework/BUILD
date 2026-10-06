# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="core",
    sources=[
        "deploy_stage.py",
        "env_var.py",
    ],
    dependencies=[
        "//3rdparty:pydantic",
    ],
)

python_sources(
    name="deploy_env_config",
    sources=[
        "_raw_deploy_config.py",
    ],
    dependencies=[
        "//3rdparty:pydantic",
    ],
)

python_sources(
    name="config_migration",
    sources=[
        "_config_migration.py",
    ],
    dependencies=[
        ":deploy_env_config",
    ],
)

python_sources(
    name="deploy_env_config_v2",
    sources=[
        "deploy_env_config.py",
    ],
    dependencies=[
        ":core",
        ":deploy_env_config",
        ":config_migration",
        ":service_framework",
    ],
)

python_sources(
    name="service_framework",
    sources=[
        "deploy_env.py",
        "aws_region.py",
    ],
    dependencies=[
        ":core",
        ":deploy_env_config",
        ":config_migration",
        "//async_utils",
        "//3rdparty:ec2_metadata",
    ],
)
