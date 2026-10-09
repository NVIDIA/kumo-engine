# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the SDK and API artifacts from the uv workspace."""

from pathlib import Path

import kumoai_studio
import kumoapi.data_source as api
from pydantic import TypeAdapter

REPO_ROOT = Path(__file__).parents[2]


def test_first_party_sdk_and_api_import_together() -> None:
    # The SDK now comes from packages/kumo-sdk while kumo-api is a
    # workspace member too. The SDK shares the root VERSION file (stamped by
    # scripts/stamp_version.py), so a version change is one deliberate edit.
    assert kumoai_studio.__version__ == (REPO_ROOT /
                                         'VERSION').read_text().strip()
    assert api.CreateConnectorArgs is not None
    assert api.S3RootedConnectorTarget is not None
    assert TypeAdapter(
        api.CreateConnectorArgs).json_schema()['additionalProperties'] is False
    assert TypeAdapter(api.S3RootedConnectorTarget).json_schema(
    )['additionalProperties'] is False
