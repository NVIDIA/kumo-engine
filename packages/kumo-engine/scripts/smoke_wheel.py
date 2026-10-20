#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install and exercise one wheel in a fresh environment outside the checkout."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from check_pypi_requirements import read_metadata


def wheel_identity(wheel):
    metadata = read_metadata(wheel)
    if metadata['Name'].lower().replace('_', '-') != 'kumo-engine':
        raise ValueError('Expected a kumo-engine wheel')
    if not metadata['Version']:
        raise ValueError('The wheel has no version')
    with wheel.open('rb') as stream:
        fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {
        'filename': wheel.name,
        'version': metadata['Version'],
        'sha256': fingerprint
    }


def verify_install(report, identity):
    engine, = [
        item for item in report['install']
        if item['metadata']['name'].lower().replace('_', '-') == 'kumo-engine'
    ]
    if engine['metadata']['version'] != identity['version']:
        raise ValueError('Installed version differs from the selected wheel')
    hashes = engine['download_info']['archive_info']['hashes']
    if hashes.get('sha256') != identity['sha256']:
        raise ValueError('Installed SHA-256 differs from the selected wheel')


def run_command(command, cwd, log, timeout=60):
    with log.open('w') as stream:
        subprocess.run(command, cwd=cwd, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for logs and JSON results')
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {
        'status': 'RUNNING',
        'stage': 'wheel_identity',
        'python': sys.version,
        'platform': platform.platform(),
        'stage_seconds': {}
    }

    def run(stage, command, cwd, timeout=60):
        summary['stage'] = stage
        print(stage, flush=True)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        begin = time.monotonic()
        try:
            run_command(command, cwd, output / f'{stage}.log', timeout)
        finally:
            summary['stage_seconds'][stage] = round(time.monotonic() - begin, 3)

    try:
        identity = wheel_identity(wheel)
        summary['wheel'] = identity
        summary['stage'] = 'configuration'
        if (sys.platform != 'linux' or platform.machine() != 'x86_64'
                or sys.version_info[:2] != (3, 12)):
            raise ValueError('Use Linux x64 with Python 3.12')
        with tempfile.TemporaryDirectory(prefix='kumo-wheel-smoke-') as tmp:
            root = Path(tmp)
            venv = root / 'venv'
            python = str(venv / 'bin/python')
            run('venv', [sys.executable, '-I', '-m', 'venv', str(venv)], root)
            run('install', [
                python, '-I', '-m', 'pip', '--isolated', 'install',
                '--no-cache-dir', '--index-url', 'https://pypi.org/simple',
                '--report',
                str(output / 'install-report.json'),
                str(wheel)
            ], root, timeout=600)
            summary['stage'] = 'installed_identity'
            verify_install(
                json.loads((output / 'install-report.json').read_text()),
                identity)
            run('pip-check', [python, '-I', '-m', 'pip', '--isolated', 'check'],
                root)
            run('packages',
                [python, '-I', '-m', 'pip', '--isolated', 'freeze', '--all'],
                root)
            source = Path(__file__).resolve(
            ).parents[3] / 'tools/ci/kumo_engine_artifact_smoke.py'
            exercise = root / source.name
            shutil.copyfile(source, exercise)
            run('workflow', [
                python, '-I',
                str(exercise), '--version', identity['version'], '--output-dir',
                str(output)
            ], root, timeout=300)
            result = json.loads((output / 'workflow-result.json').read_text())
            if result['status'] != 'PASS':
                raise ValueError('The workflow did not report PASS')
        summary.update(status='PASS', stage='complete')
    except Exception as exc:
        summary.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        summary['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
