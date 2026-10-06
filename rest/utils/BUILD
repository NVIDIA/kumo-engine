# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="tracing",
    sources=["tracing.py"],
    dependencies=[
        "//3rdparty:fastapi",
        "//3rdparty:setuptools",  # opentelemetry-instrumentation imports pkg_resources
        "//3rdparty:cachetools",
        "//monitoring",
        "//common_utils",
        "//io_utils:logging",
        "//io_utils:serde",
        "//io_utils/fsutil",
        "//service_framework",
    ],
)
