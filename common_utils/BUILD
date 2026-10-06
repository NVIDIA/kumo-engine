# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="common_utils",
    sources=["*.py", "!prediction_formatting.py"],
    dependencies=[
        "//io_utils:io_utils",
        "//io_utils:logging",
    ]
)

python_sources(
    name="prediction_formatting",
    sources=["prediction_formatting.py"],
    dependencies=[
        "//3rdparty:pandas",
    ],
)
