# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_tests(
    name="worker_manager_tests",
    sources=["test_*.py"],
    dependencies=[
        "//pool/worker_manager",
        "//3rdparty:kubernetes",
        "//3rdparty:pytest_asyncio",
    ],
)

python_test_utils(
    name="test_utils",
)
