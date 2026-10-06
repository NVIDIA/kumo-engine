# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="fsutil",
    sources=[
        "__init__.py",
        "api.py",
        "common.py",
        "credentials.py",
        "generic.py",
        "logging_policy.py",
        "s3_connection.py",
        "uri.py",
    ],
    dependencies=[
        "//async_utils",
        "//io_utils:io_utils",
        "//io_utils:logging",
        "//io_utils:secrets",
        "//io_utils/snowflake:lite",
        "//service_framework:deploy_env_config",
        "//service_framework:deploy_env_config_v2",
        "//3rdparty:asgiref",
        "//3rdparty:botocore",
        "//3rdparty:pyarrow",
        "//3rdparty:typing_extensions",
    ],
)

python_sources(
    name="impl_local",
    sources=["impl_local.py"],
    dependencies=[
        ":fsutil",
        "//io_utils:retry",
        "//monitoring",
        "//3rdparty:smart_open",
        "//3rdparty:pandas",
    ],
)

python_sources(
    name="impl_s3",
    sources=["impl_s3.py", "s3_filesystem.py"],
    dependencies=[
        ":fsutil",
        "//3rdparty:boto_s3",
        "//async_utils",
        "//monitoring",
        "//packages/kumo-s3/src/kumo_s3:download",
    ],
)

python_sources(
    name="impl_gcs",
    sources=["impl_gcs.py"],
    dependencies=[
        ":fsutil",
        "//3rdparty:gcp_storage",
        "//3rdparty:google_auth",
        "//monitoring",
        "//packages/kumo-gcs/src/kumo_gcs:download",
    ],
)


python_sources(
    name="impl_ucv",
    sources=["impl_ucv.py"],
    dependencies=[
        ":fsutil",
        "//io_utils:databricks",
        "//service_framework",
        "//3rdparty:botocore",
        "//3rdparty:http",
        "//3rdparty:databricks",
    ],
)
