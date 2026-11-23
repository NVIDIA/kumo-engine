# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge the selected engine checks without waiving full qualification gaps."""
import argparse
import json
import os
from pathlib import Path

PHASES = {'fit', 'score', 'evaluate', 'negative'}
KNOWN_GAPS = {'exported_epoch_history', 'ui_review'}


def verdict(report):
    if report.get('status') not in ('passed', 'blocked'):
        raise ValueError('Notebook runner did not complete cleanly')
    checks = report.get('engine_checks') or {}
    if checks.get('status') != 'passed' or checks.get(
            'phases') != dict.fromkeys(PHASES, 'passed'):
        raise ValueError('All four engine phases must pass')
    stages = report.get('stages', {})
    if stages.get('preflight', {}).get('status') != 'passed':
        raise ValueError('Installed-wheel notebook preflight must pass')
    if stages.get('lifecycle', {}).get('status') not in ('passed', 'blocked'):
        raise ValueError('Lifecycle execution did not complete')
    for item in [report, *stages.values()]:
        if any(
                item.get(name)
                for name in ('error', 'close_error', 'collection_error')):
            raise ValueError('Runner, collection, or cleanup error remains')
    gaps = checks.get('remaining_requirements', {})
    if set(gaps) != KNOWN_GAPS:
        raise ValueError('Both unverified requirements must remain explicit')
    if report['status'] == 'blocked' and not gaps:
        raise ValueError('Unexplained blocked result')
    return dict(engine_checks='passed', full_qualification='incomplete',
                remaining_requirements=gaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    result = verdict(json.loads(args.summary.read_text()))
    result_path = args.summary.with_name('ci-verdict.json')
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    text = ('Selected Snowflake notebook engine checks: PASSED\n\n'
            'Full qualification: ' + result['full_qualification'].upper() +
            '\n')
    for name, gap in result['remaining_requirements'].items():
        text += '\n- ' + name + ': ' + gap['status']
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(text + '\n')


if __name__ == '__main__':
    main()
