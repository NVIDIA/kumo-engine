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


def exported_graph_result():
    return dict(
        scope='PRIVATE notebook path',
        interactive=dict(status='passed',
                         checks=[dict(receipt='PRIVATE', samples=['SECRET'])]),
        saved_export=dict(status='passed',
                          checks=[dict(url='https://PRIVATE/?token=SECRET')]),
        browser_errors=[])


def test_databricks_export_dialect_preserves_actual_ui_checks_without_raw_values(
        tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'ui-result.json').write_text(json.dumps(exported_graph_result()))
    result = summary.publish(raw, tmp_path / 'public', 'databricks', 'graph')
    assert result['results']['graph'] == {
        'status': 'passed',
        'interactive': {
            'status': 'passed'
        },
        'saved_export': {
            'status': 'passed'
        },
        'browser_error_count': 0
    }
    rendered = (tmp_path / 'public/summary.json').read_text()
    assert 'PRIVATE' not in rendered and 'SECRET' not in rendered


@pytest.mark.parametrize('fault', [
    'interactive-failed', 'saved-failed', 'browser-error',
    'contradictory-error', 'top-level-failed'
])
def test_exported_graph_reports_failures_without_error_contents(fault):
    value = exported_graph_result()
    if fault == 'interactive-failed':
        value['interactive'].update(status='failed', error='PRIVATE assertion')
    elif fault == 'saved-failed':
        value['saved_export'].update(status='failed', error='PRIVATE assertion')
    elif fault == 'browser-error':
        value['browser_errors'] = ['PRIVATE token=SECRET']
    elif fault == 'contradictory-error':
        value['interactive']['error'] = 'PRIVATE assertion'
    else:
        value['status'] = 'failed'
    result = summary.aggregate(value)
    assert result['status'] == 'failed'
    assert result['browser_error_count'] == (1
                                             if fault == 'browser-error' else 0)
    assert 'PRIVATE' not in json.dumps(result) and 'SECRET' not in json.dumps(
        result)


@pytest.mark.parametrize('fault', [
    'missing-interactive', 'missing-saved', 'missing-both', 'missing-errors',
    'invalid-errors', 'invalid-check', 'unknown-status', 'pending', 'scalar'
])
def test_incomplete_databricks_graph_receipt_cannot_publish_a_pass(
        tmp_path, fault):
    value = exported_graph_result()
    if fault == 'missing-interactive':
        del value['interactive']
    elif fault == 'missing-saved':
        del value['saved_export']
    elif fault == 'missing-both':
        value = dict(status='passed', browser_errors=[])
    elif fault == 'missing-errors':
        del value['browser_errors']
    elif fault == 'invalid-errors':
        value['browser_errors'] = 'PRIVATE'
    elif fault == 'invalid-check':
        value['saved_export'] = 'passed'
    elif fault == 'unknown-status':
        value['saved_export']['status'] = 'PRIVATE'
    elif fault == 'pending':
        value['saved_export']['status'] = 'pending'
    else:
        value = 'PRIVATE'
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'ui-result.json').write_text(json.dumps(value))
    result = summary.publish(raw, tmp_path / 'public', 'databricks', 'graph')
    assert result['results']['graph']['status'] != 'passed'
    assert 'PRIVATE' not in json.dumps(result)
