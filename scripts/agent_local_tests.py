#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Vedaant Jain's marked local feedback tests with one suite deadline."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'test/scripts/agent-local.ini'
FILES = [
    ROOT / 'test/scripts' / name
    for name in ('test_engine_pr_check.py', 'test_engine_pr_owner_source.py',
                 'test_engine_pr_timing.py', 'test_engine_pr_workflow.py',
                 'test_engine_pr_data.py', 'test_engine_pr_graph.py',
                 'test_engine_pr_runtime.py',
                 'test_engine_wheel_task_checks.py')
]
LIMIT_SECONDS = 300


def accepted(report, limit):
    return (report['returncode'] == 0 and report['selected_nodeids'] and len(
        set(report['selected_nodeids'])) == len(report['selected_nodeids'])
            and sorted(report['selected_nodeids']) == sorted(
                report['completed_nodeids']) and not report['unmarked_nodeids']
            and report['tests'] > 0 and not report['failures']
            and not report['skipped'] and not report['timed_out']
            and report['elapsed_seconds'] < limit)


def pytest_worker(selection, arguments):
    import pytest

    class Selection:
        def __init__(self):
            self.report = dict(selected_nodeids=[], unmarked_nodeids=[],
                               completed_nodeids=[])

        def save(self):
            selection.write_text(json.dumps(self.report, indent=2) + '\n')

        def pytest_collection_finish(self, session):
            self.report.update(
                selected_nodeids=[item.nodeid for item in session.items],
                unmarked_nodeids=[
                    item.nodeid for item in session.items
                    if not item.get_closest_marker('agent_local')
                ])
            self.save()

        def pytest_runtest_logfinish(self, nodeid, location):
            self.report['completed_nodeids'].append(nodeid)
            self.save()

    return pytest.main(arguments, plugins=[Selection()])


def run_tests(files, output, limit=LIMIT_SECONDS):
    started = time.monotonic()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection, junit = output / 'selection.json', output / 'tests.xml'
    for path in (selection, junit):
        path.unlink(missing_ok=True)
    command = [
        sys.executable, '-I',
        str(Path(__file__).resolve()), '--worker',
        str(selection), '-q', '-c',
        str(CONFIG), '--strict-markers', '--confcutdir',
        str(ROOT / 'test/scripts'), '-p', 'no:cacheprovider', '-m',
        'agent_local', '--junitxml',
        str(junit), *map(str, files)
    ]
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                       PYTHONDONTWRITEBYTECODE='1')
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        environment.pop(name, None)
    report = dict(
        status='FAIL', command=command, limit_seconds=limit,
        selected_files=list(map(str, files)), selected_nodeids=[],
        completed_nodeids=[], unmarked_nodeids=[], tests=0, failures=0,
        skipped=0, timed_out=False, returncode=None, python=sys.version,
        platform=platform.platform(), source_sha256={
            str(path.relative_to(ROOT)):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*files, CONFIG,
                         Path(__file__).resolve()] if path.is_relative_to(ROOT)
        }, environment={
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
            'PYTEST_ADDOPTS': None,
            'PYTEST_PLUGINS': None
        }, packages={
            d.metadata['Name']: d.version
            for d in importlib.metadata.distributions()
        })
    with (output / 'pytest.log').open('w') as stream:
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                              stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                process.wait(timeout=max(0, limit -
                                         (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                report['timed_out'] = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            report['returncode'] = process.returncode
    try:
        report.update(json.loads(selection.read_text()))
        suites = ET.parse(junit).iter('testsuite')
        for suite in suites:
            report['tests'] += int(suite.get('tests', 0))
            report['failures'] += int(suite.get('failures', 0)) + int(
                suite.get('errors', 0))
            report['skipped'] += int(suite.get('skipped', 0))
    except (OSError, ValueError, ET.ParseError) as error:
        report['report_error'] = str(error)
    report['elapsed_seconds'] = time.monotonic() - started
    if not report.get('report_error') and accepted(report, limit):
        report['status'] = 'PASS'
    receipt = output / 'result.json'
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(
        f"{report['status']}: {len(report['selected_nodeids'])} selected tests, "
        f"{report['skipped']} skipped; evidence: {output}", flush=True)
    report['elapsed_seconds'] = time.monotonic() - started
    if report['elapsed_seconds'] >= limit:
        report['status'] = 'FAIL'
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    if time.monotonic() - started >= limit:
        report.update(status='FAIL', elapsed_seconds=time.monotonic() - started)
        receipt.write_text(json.dumps(report, indent=2) + '\n')
        print('FAIL: local suite deadline reached during reporting', flush=True)
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        return pytest_worker(Path(sys.argv[2]), sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='kumo-agent-local-'))
    report = run_tests(FILES, output)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
