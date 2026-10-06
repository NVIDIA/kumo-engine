# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="worker_manager",
    sources=["*.py"],
    dependencies=[
        "//io_utils",                # kumo_dataclass
        "//io_utils:logging",
        "//3rdparty:pydantic",       # transitively via io_utils.dataclasses
        "//3rdparty:kubernetes",     # k8s client + models
    ],
)
