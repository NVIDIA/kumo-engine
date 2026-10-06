# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="temporalio_commons",
    sources=[
        "*.py",
        "!debug_utils.py",  # non-production code
        "!health_check.py",  # non-library script, separate build target
        "!worker_main.py",  # [TODO] separate pex_binary targets for workers
        "!replay.py",
    ],
    dependencies=[
        "//common_utils",
        "//async_utils",
        "//io_utils:logging",
        "//io_utils:secrets",
        "//io_utils:serde",
        "//io_utils/fsutil",
        "//monitoring",
        "//service_framework",
        "//3rdparty:asgiref",
        "//3rdparty:temporalio",
        "//3rdparty:pydantic",
        "//3rdparty:protobuf",
        "//3rdparty:typing_extensions",
        "//3rdparty:zstd",
    ]
)

python_sources(
    name="health_check",
    sources=["health_check.py"],
    dependencies=[
        ":temporalio_commons",
        "//io_utils:logging",
        "//service_framework",
        "//3rdparty:http",
    ],
)

pex_binary(
    name="health_check_bin",
    entry_point="temporalio_commons.health_check",
    dependencies=[":health_check"]
)
