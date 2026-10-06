# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_tests(
    name="test_connector_identity_contract",
    sources=["test_connector_identity_contract.py"],
    dependencies=["//packages/kumo-api/kumoapi"],
)

python_tests(
    name="test_sdk_api_source_contract",
    sources=["test_sdk_api_source_contract.py"],
    dependencies=[
        "//packages/kumo-api/kumoapi",
        "//packages/kumo-sdk:kumo_sdk",
    ],
)
