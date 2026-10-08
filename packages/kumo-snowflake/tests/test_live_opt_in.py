# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snowflake live collection needs opt-in before credentials or SDK imports."""

from pathlib import Path

import pytest
from sources.live_opt_in import (
    COLLECTION_CASES,
    SECRET,
    assert_live_collection,
)

CREDENTIALS = {
    "account": "synthetic-account",
    "user": "synthetic-user",
    "password": SECRET,
}


@pytest.mark.parametrize("case", [
    *COLLECTION_CASES,
    *(f"missing-field-{key}" for key in CREDENTIALS),
])
def test_live_collection_requires_explicit_opt_in_and_credentials(
        tmp_path, case):
    assert_live_collection(
        tmp_path,
        Path(__file__).parent / "live" / "test_live_conformance.py",
        prefix="KUMO_SNOWFLAKE",
        credentials=CREDENTIALS,
        case=case,
    )
