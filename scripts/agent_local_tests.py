#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the marked GPU smoke guards within Vedaant Jain's local deadline."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_SECONDS = 300


def worker(output):
    import pytest
    counts = dict(selected=0, completed=0, skipped=0)

    class Proof:
        def pytest_collection_finish(self, session):
            counts['selected'] = len(session.items)

        def pytest_runtest_logfinish(self, nodeid, location):
            counts['completed'] += 1

        def pytest_runtest_logreport(self, report):
            counts['skipped'] += int(report.skipped)

    code = pytest.main([
        '-q', '-c',
        str(ROOT / 'test/scripts/agent-local.ini'), '--confcutdir',
        str(ROOT / 'test/scripts'), '--strict-markers', '-p',
        'no:cacheprovider', '-m', 'agent_local',
        str(ROOT / 'test/scripts/test_engine_gpu_smoke.py')
    ], plugins=[Proof()])
    output.write_text(json.dumps(dict(counts, returncode=int(code))) + '\n')
    return int(code)


def main():
    started = time.monotonic()
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        return worker(Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    output = parser.parse_args().output or Path(
        tempfile.mkdtemp(prefix='kumo-agent-local-'))
    output.mkdir(parents=True, exist_ok=True)
    receipt = output.resolve() / 'result.json'
    receipt.unlink(missing_ok=True)
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                       PYTHONDONTWRITEBYTECODE='1')
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        environment.pop(name, None)
    command = [
        sys.executable, '-I',
        str(Path(__file__).resolve()), '--worker',
        str(receipt)
    ]
    timed_out = False
    with (output / 'pytest.log').open('w') as log:
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                              stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                process.wait(
                    timeout=max(0, LIMIT_SECONDS -
                                (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    try:
        result = json.loads(receipt.read_text())
    except (OSError, ValueError):
        result = dict(selected=0, completed=0, skipped=0, returncode=1)
    passed = (process.returncode == result['returncode'] == 0
              and result['selected'] > 0
              and result['selected'] == result['completed']
              and result['skipped'] == 0 and not timed_out)
    print(
        f"{result['selected']} tests, {result['skipped']} skipped; log: {output / 'pytest.log'}",
        flush=True)
    result.update(status='PASS' if passed else 'FAIL', timed_out=timed_out,
                  elapsed_seconds=time.monotonic() - started)
    if result['elapsed_seconds'] >= LIMIT_SECONDS:
        result['status'] = 'FAIL'
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] == 'PASS' and time.monotonic(
    ) - started < LIMIT_SECONDS else 1


if __name__ == '__main__':
    sys.exit(main())
