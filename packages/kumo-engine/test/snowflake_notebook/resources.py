# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require explicit authorization of the dedicated Snowflake test resources."""
import hashlib
import json
import os
from pathlib import Path

TARGET = dict(
    account='lfwgwbp-kumo_us_west_test_5', account_locator='SVB14499',
    user='KUMO_SNOWFLAKE_CI', browser_user='KUMO_SNOWFLAKE_UI_TEST',
    role='KUMO_SNOWFLAKE_TEST_ROLE', database='KUMO_TEST_DB',
    schema='KUMO_TEST', warehouse='KUMO_TEST_WH', pool='KUMO_TEST_POOL',
    stage='@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE',
    denied_stage='@KUMO_TEST_DB.KUMO_TEST.KUMO_TEST_DENIED_STAGE',
    notebook_service='EGU_GRAPH_UI_SERVICE',
    account_url='https://app.snowflake.com/lfwgwbp/kumo_us_west_test_5/')


def expected():
    root = Path(__file__).parent
    return dict(
        TARGET, manifests={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('candidate.json', 'graph-assets.json')
        })


def require():
    raw = os.environ.get('SNOWFLAKE_TEST_CONFIG')
    if not raw:
        raise ValueError('SNOWFLAKE_TEST_CONFIG is required for live execution')
    try:
        value = json.loads(raw)
    except ValueError:
        raise ValueError('SNOWFLAKE_TEST_CONFIG must be valid JSON') from None
    if value != expected():
        raise ValueError(
            'SNOWFLAKE_TEST_CONFIG must authorize the dedicated test resources '
            'and the exact candidate/asset manifests')
    return value
