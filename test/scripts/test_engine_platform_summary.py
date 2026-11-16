# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/platform_summary.py'
spec = importlib.util.spec_from_file_location('platform_summary', SOURCE)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_public_summary_never_copies_raw_evidence(tmp_path):
    raw, public = tmp_path / 'raw', tmp_path / 'public'
    raw.mkdir()
    value = dict(
        status='passed', user='private@example.com', account='PRIVATE',
        url='https://private.example/?token=PRIVATE', error='PRIVATE',
        elapsed_seconds=12.5, phases={
            'fit': {
                'status': 'passed',
                'error': 'PRIVATE'
            },
            'score': {
                'status': 'https://PRIVATE'
            },
            'PRIVATE': {
                'status': 'passed'
            }
        })
    (raw / 'summary.json').write_text(json.dumps(value))
    for name in ('notebook.ipynb', 'run.json', 'screenshot.png', 'config.json'):
        (raw / name).write_text('PRIVATE')
    result = summary.publish(raw, public, 'snowflake', 'engine')
    assert list(public.iterdir()) == [public / 'summary.json']
    assert 'PRIVATE' not in (public / 'summary.json').read_text()
    assert result['results']['summary'] == dict(
        status='passed', elapsed_seconds=12.5,
        phases=dict(fit='passed', score='not_run'))


def test_public_summary_does_not_follow_symlinks_or_copy_into_raw(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    (tmp_path / 'private').write_text('{"status":"passed"}')
    (raw / 'summary.json').symlink_to(tmp_path / 'private')
    result = summary.publish(raw, tmp_path / 'public', 'databricks', 'graph')
    assert result['results']['summary']['status'] == 'not_run'
    with pytest.raises(ValueError):
        summary.publish(raw, raw / 'public', 'databricks', 'graph')


def test_parent_symlink_is_excluded_and_fixed_verdicts_survive(tmp_path):
    raw, other = tmp_path / 'raw', tmp_path / 'other'
    raw.mkdir()
    other.mkdir()
    (other / 'result.json').write_text('{"status":"passed"}')
    (raw / 'browser').symlink_to(other, target_is_directory=True)
    (raw / 'ci-verdict.json').write_text(
        json.dumps(
            dict(verdict=dict(engine_checks='passed',
                              positive_lifecycle='passed',
                              full_qualification='incomplete'))))
    result = summary.publish(raw, tmp_path / 'public', 'sagemaker', 'all')
    assert 'browser' not in result['results']
    assert result['results']['engine']['verdict'][
        'full_qualification'] == 'incomplete'


def test_public_diagnostics_accept_only_fixed_enum_values():
    value = dict(status='failed', failure_code='assertion_failed',
                 failure_phase='notebook_visualization',
                 failure_step='dedicated_notebook_service_selected',
                 reason='PRIVATE token=SECRET')
    assert summary.aggregate(value) == {
        key: item
        for key, item in value.items() if key != 'reason'
    }
    for key in summary.DIAGNOSTICS:
        malicious = dict(value, **{key: 'PRIVATE token=SECRET'})
        result = summary.aggregate(malicious)
        assert key not in result
        assert 'PRIVATE' not in json.dumps(result)
        malicious[key] = ['PRIVATE']
        assert key not in summary.aggregate(malicious)


def test_cleanup_recovery_publishes_statuses_without_original_error_or_paths():
    value = dict(
        status='passed', verification_status='passed',
        browser_cleanup=dict(status='passed'),
        initial_cleanup_failure=dict(reason='PRIVATE auth URL token=SECRET'),
        cleanup_retry=dict(status='passed', evidence='/PRIVATE/summary.json'))
    result = summary.aggregate(value)
    assert result == dict(status='passed', verification_status='passed',
                          browser_cleanup=dict(status='passed'),
                          cleanup_retry=dict(status='passed'))
    assert 'PRIVATE' not in json.dumps(result)
