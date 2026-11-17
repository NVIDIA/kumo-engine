# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/snowflake_notebook/launch.py'
spec = importlib.util.spec_from_file_location('cleanup_launch', SOURCE)
launch = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', [str(SOURCE.parent), *sys.path]):
    spec.loader.exec_module(launch)


class Connection:
    def __init__(self, fail_drop=False):
        self.commands = []
        self.fail_drop = fail_drop

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql):
        self.commands.append(sql)
        if self.fail_drop and sql.startswith('DROP'):
            raise RuntimeError('drop failed')


@pytest.fixture
def manifest():
    attempt = 'a' * 12
    return dict(attempt=attempt,
                bundle='KUMO_TEST_DB.KUMO_TEST.MLOPS_PROBE_' + attempt.upper(),
                stage_prefix=launch.STAGE + '/notebook-output-probe/' + attempt)


@pytest.mark.parametrize('drop_fails', [False, True])
def test_submission_cleanup_attempts_both_owned_resources(manifest, drop_fails):
    connection = Connection(drop_fails)
    result = launch.cleanup_submission(connection, manifest,
                                       query_unknown=False, bundle_created=True,
                                       bundle_pending=False)
    assert connection.commands == [
        'DROP CODE BUNDLE ' + manifest['bundle'],
        'REMOVE ' + manifest['stage_prefix'] + '/'
    ]
    assert result['status'] == ('failed' if drop_fails else 'passed')


def test_unknown_query_never_deletes_remote_resources(manifest):
    connection = Connection()
    result = launch.cleanup_submission(connection, manifest, query_unknown=True,
                                       bundle_created=True,
                                       bundle_pending=False)
    assert result['status'] == 'failed'
    assert not connection.commands


def test_unknown_create_cannot_claim_success(manifest):
    connection = Connection()
    result = launch.cleanup_submission(connection, manifest,
                                       query_unknown=False,
                                       bundle_created=False,
                                       bundle_pending=True)
    assert result['status'] == 'failed'
    assert connection.commands == ['REMOVE ' + manifest['stage_prefix'] + '/']


@pytest.mark.parametrize('field', ['attempt', 'bundle', 'stage_prefix'])
def test_cleanup_rejects_non_attempt_resources(manifest, field):
    manifest[field] = 'shared'
    connection = Connection()
    with pytest.raises(ValueError):
        launch.cleanup_submission(connection, manifest, query_unknown=False,
                                  bundle_created=True, bundle_pending=False)
    assert not connection.commands
