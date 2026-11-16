# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish only fixed-field aggregate results, never raw platform evidence."""
import argparse
import json
import math
from pathlib import Path

STATUSES = {'passed', 'failed', 'blocked', 'incomplete', 'pending', 'running'}
PHASES = ('fit', 'score', 'evaluate', 'negative')
FILES = {
    'summary': 'summary.json',
    'engine': 'ci-verdict.json',
    'graph': 'ui-result.json',
    'browser': 'browser/result.json',
    'studio': 'studio/result.json',
    'snowsight': 'ui/summary.json'
}

DIAGNOSTICS = {
    'failure_code':
    {'assertion_failed', 'timeout', 'capacity_blocked', 'browser_error'},
    'failure_phase': {
        'browser_start', 'password_login', 'totp_login',
        'notebook_visualization', 'run_status', 'run_details', 'epoch_history',
        'model_versions', 'notebook_cleanup', 'browser_cleanup'
    },
    'failure_step': {
        'authenticated', 'workspace_opened',
        'dedicated_notebook_service_selected', 'waiting_for_notebook_compute',
        'waiting_for_dedicated_pool_after_service_timeout',
        'notebook_connection_retried_once', 'test_resources_verified',
        'notebook_install_run_requested', 'notebook_install_started',
        'kernel_reset_completed', 'notebook_graph_run_requested',
        'notebook_graph_started', 'live_graph_verified',
        'test_notebook_kernel_stopped', 'saved_output_verified',
        'saved_output_rechecked', 'attempt_notebook_deleted', 'checking_run',
        'run_verified', 'ui_verified'
    },
}


def status(value):
    return value if isinstance(value, str) and value in STATUSES else 'not_run'


def aggregate(value, depth=0):
    if depth > 4 or not isinstance(value, dict):
        return {'status': 'not_run'}
    result = {'status': status(value.get('status'))}
    for key, allowed in DIAGNOSTICS.items():
        diagnostic = value.get(key)
        if isinstance(diagnostic, str) and diagnostic in allowed:
            result[key] = diagnostic
    elapsed = value.get('elapsed_seconds')
    if type(elapsed) in (int,
                         float) and math.isfinite(elapsed) and elapsed >= 0:
        result['elapsed_seconds'] = round(elapsed, 3)
    for key in ('phases', 'stages'):
        phases = value.get(key, {})
        if isinstance(phases, dict):
            selected = {
                name:
                status(phases[name].get('status')) if isinstance(
                    phases[name], dict) else status(phases[name])
                for name in PHASES if name in phases
            }
            if selected:
                result['phases'] = selected
    for key in ('engine_checks', 'verdict', 'positive_lifecycle',
                'full_qualification', 'cleanup', 'notebook_cleanup',
                'stage_cleanup', 'cleanup_retry', 'browser_cleanup',
                'verification_status'):
        nested = value.get(key)
        if isinstance(nested, dict):
            selected = aggregate(nested, depth + 1)
            if selected != {'status': 'not_run'}:
                result[key] = selected
        elif isinstance(nested, str):
            result[key] = status(nested)
    return result


def publish(source, destination, platform, scenario):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Public summaries need a separate output directory')
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(platform=platform, scenario=scenario, results={})
    for label, relative in FILES.items():
        path = source / relative
        if (not path.is_file() or path.is_symlink()
                or source not in path.resolve().parents
                or path.stat().st_size > 8 * 1024**2):
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            report['results'][label] = {'status': 'failed'}
        else:
            report['results'][label] = aggregate(value)
    if not report['results']:
        report['results']['summary'] = {'status': 'not_run'}
    (destination /
     'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--platform',
                        choices=('snowflake', 'databricks', 'sagemaker'),
                        required=True)
    parser.add_argument('--scenario', choices=('engine', 'graph', 'all', 'ui'),
                        required=True)
    args = parser.parse_args()
    publish(args.input, args.output, args.platform, args.scenario)


if __name__ == '__main__':
    main()
