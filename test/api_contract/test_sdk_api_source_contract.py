# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the SDK and API artifacts selected by the Pants resolve."""

import kumoai_studio
import kumoapi.data_source as api
from pydantic import TypeAdapter


def test_first_party_sdk_and_api_import_together() -> None:
    # The SDK now comes from packages/kumo-sdk while kumo-api remains selected
    # by the shared Pants resolve. Keep the public version assertion explicit
    # so SDK version changes remain deliberate contract updates.
    assert kumoai_studio.__version__ == '3.1.0.dev0'
    assert api.CreateConnectorArgs is not None
    assert api.S3RootedConnectorTarget is not None
    assert TypeAdapter(
        api.CreateConnectorArgs).json_schema()['additionalProperties'] is False
    assert TypeAdapter(api.S3RootedConnectorTarget).json_schema(
    )['additionalProperties'] is False
