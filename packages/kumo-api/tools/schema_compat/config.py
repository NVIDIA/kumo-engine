# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
"""Configuration for schema compatibility checker."""

import pydantic

# S3 base path for storing schema files
# Format: s3://<bucket>/<prefix>/
# Schema files will be stored as:
# {S3_SCHEMA_BASE_PATH}v{version}/pydantic-v{1|2}.json
S3_SCHEMA_BASE_PATH = "s3://kumocloud-test-bucket/kumo-api-schemas/"

# The first version where schema compatibility enforcement starts
# All versions >= this will be checked for backward compatibility
ENFORCEMENT_BASELINE_VERSION = "0.52.0"

# Pydantic version detection
PYDANTIC_VERSION = int(pydantic.__version__.split('.')[0])
IS_PYDANTIC_V2 = PYDANTIC_VERSION >= 2


def get_pydantic_schema_key() -> str:
    """Get the schema filename suffix for current pydantic version."""
    return f"pydantic-v{PYDANTIC_VERSION}.json"
