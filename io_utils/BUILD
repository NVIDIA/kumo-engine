# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal, lightweight target that only includes errors and dataclasses
python_sources(
    name="io_utils",
    sources=[
        "__init__.py",
        "dataclasses.py",
        "errors.py"
    ],
    dependencies=["//3rdparty:pydantic"],
)

python_sources(
    name="secrets",
    sources=["secrets.py"],
    dependencies=[
        ":io_utils",
        "//3rdparty:boto_s3",
        "//3rdparty:google_auth",
        "//3rdparty:gcp_secretmanager",
        "//3rdparty:google_api_core",
        "//async_utils",
        "//monitoring",
        "//service_framework",
        "//service_framework:deploy_env_config_v2",
    ],
)

python_sources(
    name="databricks",
    sources=["databricks.py"],
    dependencies=[
        ":secrets",
        "//service_framework",
        "//3rdparty:boto_s3",
        "//3rdparty:http",
    ],
)

python_sources(
    name="logging",
    sources=[
        "logging.py",
        "logging_util.py"
    ],
    dependencies=[
        ":io_utils",
        "//service_framework",
        "//3rdparty:pydantic",
        "//3rdparty:logging_loki",
    ],
)

python_sources(
    name="retry",
    sources=["retry.py"],
    dependencies=[":logging", "//3rdparty:decorator"],
)

python_sources(
    name="exceptions",
    sources=["exceptions.py"],
    dependencies=[":logging", "//async_utils"],
)

python_sources(
    name="encryption",
    sources=["encryption.py"],
    dependencies=[
        ":logging",
        ":secrets",
        "//3rdparty:cryptography",
        "//3rdparty:pydantic",
        "//service_framework",
    ],
)

python_sources(
    name="serde",
    sources=["json.py", "locate.py", "yaml.py"],
    dependencies=[
        ":encryption",
        "//async_utils",
        "//monitoring",
        "//io_utils/fsutil",
        "//3rdparty:pydantic",
        "//3rdparty:yaml",
        "//3rdparty:asgiref",
    ],
)

python_sources(
    name="parquet_util",
    sources=["parquet_util.py"],
    dependencies=[
        ":logging",
        "//async_utils",
        "//io_utils/fsutil",
        "//io_utils/snowflake",
        "//3rdparty:pandas",
        "//3rdparty:pyarrow",
    ],
)

python_sources(
    name="partition",
    sources=["partition.py"],
    dependencies=[
        ":serde",
        ":logging",
        "//io_utils/fsutil",
    ],
)

python_sources(
    name="pyarrow",
    sources=["pyarrow.py"],
    dependencies=[
        ":io_utils",
        "//3rdparty:pandas",
        "//3rdparty:pyarrow",
    ],
)
