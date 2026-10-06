# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="lite",
    sources=["__init__.py", "env.py", "path.py",  "volume_stage_map.py"],
    dependencies=[
        "//io_utils:logging",
        "//service_framework:core",
    ],
)

python_sources(
    name="snowflake",
    sources=[
        "*.py", "!__init__.py", "!env.py", "!path.py", "!volume_stage_map.py",
    ],
    dependencies=[
        ":lite",
        "//io_utils:logging",
        "//io_utils/fsutil:fsutil",
        "//service_framework",
        "//3rdparty:snowflake",
        "//3rdparty:botocore",
        "//3rdparty:pyarrow",
        "//3rdparty:wrapt",
    ],
)
