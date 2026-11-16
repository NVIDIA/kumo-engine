# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Schema compatibility checker for kumo-api
#
# This tool enforces backward compatibility of pydantic dataclass schemas
# across minor versions within a major version.
#
# Usage:
#   python -m tools.schema_compat dump [--output schema.json]
#   python -m tools.schema_compat compare --old old.json --new new.json
#   python -m tools.schema_compat check-release --version 0.52.3

from tools.schema_compat.extractor import extract_schema
from tools.schema_compat.comparator import compare_schemas
from tools.schema_compat.storage import fetch_schema, upload_schema

__all__ = [
    'extract_schema',
    'compare_schemas',
    'fetch_schema',
    'upload_schema',
]
