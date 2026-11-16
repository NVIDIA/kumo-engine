# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require explicit authorization of the dedicated Snowflake test resources."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

TARGET = dict(account='account-test_account', account_locator='TEST_LOCATOR',
              user='KUMO_SNOWFLAKE_CI', browser_user='KUMO_SNOWFLAKE_UI_TEST',
              role='KUMO_SNOWFLAKE_TEST_ROLE', database='KUMO_TEST_DB',
              schema='KUMO_TEST', warehouse='KUMO_TEST_WH',
              pool='KUMO_TEST_POOL', stage='@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE',
              denied_stage='@KUMO_TEST_DB.KUMO_TEST.KUMO_TEST_DENIED_STAGE',
              notebook_service='KUMO_TEST_NOTEBOOK_SERVICE',
              account_url='https://app.snowflake.com/account/test_account/')


def expected():
    root = Path(__file__).parent
    return dict(
        TARGET, manifests={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('graph-assets.json', )
        })


def require():
    raw = os.environ.get('SNOWFLAKE_TEST_CONFIG')
    if not raw:
        raise ValueError('SNOWFLAKE_TEST_CONFIG is required for live execution')
    try:
        value = json.loads(raw)
    except ValueError:
        raise ValueError('SNOWFLAKE_TEST_CONFIG must be valid JSON') from None
    template = expected()
    # Older configurations pinned dependency snapshots; resolution now follows the wheel.
    if isinstance(value, dict) and isinstance(value.get('manifests'), dict):
        value['manifests'] = {
            key: item
            for key, item in value['manifests'].items()
            if key not in {'dependencies.json', 'dependency-asset.json'}
        }
    identity = {'account', 'account_locator', 'account_url', 'notebook_service'}
    valid = isinstance(value, dict) and set(value) == set(template)
    valid = valid and all(isinstance(value[k], str) for k in identity)
    valid = valid and all(value[k] == v
                          for k, v in template.items() if k not in identity)
    if valid:
        url = urlsplit(value['account_url'])
        parts = url.path.strip('/').split('/')
        valid = (url.scheme == 'https' and url.netloc == 'app.snowflake.com'
                 and not url.query and not url.fragment and len(parts) == 2
                 and url.path == '/' + '/'.join(parts) + '/'
                 and all(re.fullmatch(r'[a-z0-9_]+', part)
                         for part in parts) and 'test' in parts[1].split('_')
                 and value['account'] == '-'.join(parts)
                 and isinstance(value['account_locator'], str)
                 and re.fullmatch(r'[A-Z0-9_]+', value['account_locator'])
                 and re.fullmatch(r'[A-Z][A-Z0-9_]{0,254}',
                                  value['notebook_service']))
    if not valid:
        raise ValueError(
            'SNOWFLAKE_TEST_CONFIG must authorize the dedicated test resources '
            'and the exact graph asset manifest')
    return value


def candidate_path():
    """Require the runtime manifest produced by the staging command."""
    value = os.environ.get('SNOWFLAKE_TEST_CANDIDATE')
    if not value:
        raise ValueError(
            'SNOWFLAKE_TEST_CANDIDATE must name the staged candidate manifest')
    return Path(value)


def admit_candidate(revision='HEAD', path=None):
    selected = Path(path) if path is not None else candidate_path()
    command = [
        sys.executable,
        str(Path(__file__).parent.parent / 'platform_candidate.py'),
        '--candidate',
        str(selected), '--source-ref', revision
    ]
    result = subprocess.run(command, capture_output=True, text=True,
                            check=False)
    if result.returncode:
        raise ValueError(
            'Candidate source admission failed; stage a wheel built '
            'from the selected product revision')
    return selected
