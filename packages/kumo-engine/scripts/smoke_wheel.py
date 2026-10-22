#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install engine wheels in fresh environments outside the checkout."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from check_pypi_requirements import read_metadata

REQUIREMENTS = Path(__file__).resolve().parents[1] / 'test/wheel_workflow.json'


def load_requirements(path):
    requirements = json.loads(path.read_text())
    try:
        configuration = requirements['configuration']
        if tuple(configuration[key]
                 for key in ('os', 'machine', 'python',
                             'device')) != ('linux', 'x86_64', '3.12', 'cpu'):
            raise ValueError(
                'Only Linux x64, Python 3.12, CPU is supported by this test')
        if type(configuration['threads']
                ) is not int or configuration['threads'] < 1:
            raise ValueError('Thread count must be a positive integer')
        fixture = requirements['fixture']
        for key in ('entities', 'events', 'days', 'epochs', 'steps_per_epoch'):
            if type(fixture[key]) is not int or fixture[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(fixture['seed']) is not int or fixture['seed'] < 0:
            raise ValueError('Seed must be a nonnegative integer')
        for key in ('query', 'start_time'):
            if not isinstance(fixture[key], str) or not fixture[key].strip():
                raise ValueError(f'{key} is required')
        for key in ('reload_atol', 'reload_rtol', 'threshold'):
            if not math.isfinite(fixture[key]) or fixture[key] < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if fixture['threshold'] > 1:
            raise ValueError('Threshold must be between zero and one')
        checks = requirements['required_checks']
        if not isinstance(checks, list) or not checks or not all(
                isinstance(check, str) and check for check in checks):
            raise ValueError('Required checks must be a nonempty list of names')
        reference = requirements['reference']
        if not reference['approved_by'] or not reference['approved_on']:
            raise ValueError('Reference approval is required')
        wheel = reference['wheel']
        if not wheel['filename'] or not wheel['version'] or not re.fullmatch(
                '[0-9a-f]{64}', wheel['sha256']):
            raise ValueError(
                'Reference filename, version, and SHA-256 are required')
        evidence = reference['evidence']
        if evidence['status'] != 'PASS' or not all(
                evidence[key]
                for key in ('url', 'harness_commit', 'python', 'platform')):
            raise ValueError('Passing reference evidence is required')
        dependencies = reference['dependencies']
        if not dependencies or not all(
                re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', name)
                and isinstance(version, str)
                and re.fullmatch(r'[A-Za-z0-9.!+_-]+', version)
                for name, version in dependencies.items()):
            raise ValueError('Exact reference dependency versions are required')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Missing or invalid wheel test requirements') from exc
    return requirements


def verify_reference_dependencies(report, expected):
    installed = {
        re.sub(r'[-_.]+', '-', item['metadata']['name']).lower():
        item['metadata']['version']
        for item in report['install']
        if item['metadata']['name'].lower().replace('_', '-') != 'kumo-engine'
    }
    if installed != expected:
        raise ValueError(
            'Installed dependencies differ from the approved reference')


def verify_workflow_result(result, required_checks, *, status='PASS'):
    if result.get('status') != status or any(
            result.get('checks', {}).get(check) != 'PASS'
            for check in required_checks):
        raise ValueError('A required workflow check is missing or failed')


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
    # Build-dependency pip installers do not inherit --isolated.
    environment = {
        key: value
        for key, value in os.environ.items() if not key.startswith('PIP_')
    }
    environment['PIP_CONFIG_FILE'] = os.devnull
    with log.open('w') as stream:
        subprocess.run(command, cwd=cwd, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout,
                       env=environment)


def check_wheel(wheel, output, *, reference=False, learning=False,
                fixed_dependencies=False, measure_learning=False,
                compare_with=None, report_learning=False):
    wheel = wheel.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    measurement = measure_learning or report_learning
    summary = {
        'status': 'RUNNING',
        'stage': 'wheel_identity',
        'python': sys.version,
        'platform': platform.platform()
    }

    def run(stage, command, cwd, timeout=60):
        summary['stage'] = stage
        print(stage, flush=True)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        run_command(command, cwd, output / f'{stage}.log', timeout)

    try:
        identity = wheel_identity(wheel)
        summary['wheel'] = identity
        summary['stage'] = 'requirements'
        requirements = load_requirements(REQUIREMENTS)
        shutil.copyfile(REQUIREMENTS, output / 'requirements.json')
        summary['requirements_sha256'] = hashlib.sha256(
            REQUIREMENTS.read_bytes()).hexdigest()
        summary['configuration'] = requirements['configuration']
        summary['fixture'] = requirements['fixture']
        summary['reference_mode'] = reference
        if reference:
            summary['stage'] = 'reference'
            if identity != requirements['reference']['wheel']:
                raise ValueError(
                    'Selected wheel differs from the approved reference')
        summary['stage'] = 'configuration'
        configuration = requirements['configuration']
        if (sys.platform != configuration['os']
                or platform.machine() != configuration['machine']
                or f'{sys.version_info.major}.{sys.version_info.minor}'
                != configuration['python']):
            raise ValueError('Use Linux x64 with Python 3.12')
        with tempfile.TemporaryDirectory(prefix='kumo-wheel-smoke-') as tmp:
            root = Path(tmp)
            venv = root / 'venv'
            python = str(venv / 'bin/python')
            run('venv', [sys.executable, '-I', '-m', 'venv', str(venv)], root)
            constraints = []
            if reference or fixed_dependencies:
                pins = output / 'reference-constraints.txt'
                pins.write_text(
                    ''.join(f'{name}=={version}\n' for name, version in
                            requirements['reference']['dependencies'].items()))
                constraints = ['--constraint', str(pins)]
            run('install', [
                python, '-I', '-m', 'pip', '--isolated', 'install',
                '--no-cache-dir', '--index-url', 'https://pypi.org/simple',
                '--report',
                str(output / 'install-report.json'), *constraints,
                str(wheel)
            ], root, timeout=600)
            summary['stage'] = 'installed_identity'
            report = json.loads((output / 'install-report.json').read_text())
            verify_install(report, identity)
            if reference or fixed_dependencies:
                verify_reference_dependencies(
                    report, requirements['reference']['dependencies'])
            run('pip-check', [python, '-I', '-m', 'pip', '--isolated', 'check'],
                root)
            run('packages',
                [python, '-I', '-m', 'pip', '--isolated', 'freeze', '--all'],
                root)
            source = REQUIREMENTS.with_suffix('.py')
            exercise = root / source.name
            shutil.copyfile(source, exercise)
            shutil.copyfile(REQUIREMENTS, exercise.with_suffix('.json'))
            run('workflow', [
                python, '-I',
                str(exercise), '--version', identity['version'], '--output-dir',
                str(output)
            ], root, timeout=300)
            result = json.loads((output / 'workflow-result.json').read_text())
            verify_workflow_result(result, requirements['required_checks'])
            if learning or measurement:
                learning = root / 'wheel_learning.py'
                shutil.copyfile(REQUIREMENTS.with_name('wheel_learning.py'),
                                learning)
                try:
                    run('learning', [
                        python, '-I',
                        str(learning), '--version', identity['version'],
                        '--output-dir',
                        str(output / 'learning'),
                        *(['--measure'] if measurement else [])
                    ], root, timeout=1200)
                except subprocess.CalledProcessError as exc:
                    failed_report = output / 'learning/learning-result.json'
                    if failed_report.is_file():
                        summary['learning'] = json.loads(
                            failed_report.read_text())
                        raise ValueError(summary['learning'].get('error')
                                         or 'Learning worker failed') from exc
                    raise
                learning_result = json.loads(
                    (output / 'learning/learning-result.json').read_text())
                verify_workflow_result(
                    learning_result, [],
                    status='MEASURED' if measurement else 'PASS')
                summary['learning'] = learning_result
                if compare_with is not None:
                    inputs = root / 'comparison-input.json'
                    inputs.write_text(
                        json.dumps([
                            compare_with,
                            dict(
                                summary, status='MEASURED' if report_learning
                                else 'PASS'), requirements['learning']
                        ]))
                    comparison = output / 'comparison-result.json'
                    command = [
                        python, '-I', '-c', 'import json, runpy, sys\n'
                        'reference, candidate, baseline = json.load(open(sys.argv[1]))\n'
                        'result = runpy.run_path(sys.argv[2])["compare_results"]('
                        'reference, candidate, baseline)\n'
                        'json.dump(result, open(sys.argv[3], "w"))\n',
                        str(inputs),
                        str(learning),
                        str(comparison)
                    ]
                    if report_learning:
                        reporter = root / 'wheel_learning_report.py'
                        shutil.copyfile(REQUIREMENTS.with_name(reporter.name),
                                        reporter)
                        command = [
                            python, '-I',
                            str(reporter),
                            str(inputs),
                            str(comparison)
                        ]
                    run('comparison', command, root)
                    summary['comparison'] = json.loads(comparison.read_text())
                    verify_workflow_result(
                        summary['comparison'], [],
                        status='MEASURED' if report_learning else 'PASS')
        summary.update(status='MEASURED' if measurement else 'PASS',
                       stage='complete')
    except Exception as exc:
        summary.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        summary['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['status'] in ('PASS', 'MEASURED') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--learning', action='store_true',
                        help='Also run paired learning checks')
    parser.add_argument(
        '--measure-learning', action='store_true',
        help='Collect reference variation without quality limits')
    parser.add_argument(
        '--report-learning', action='store_true',
        help='Report scores and missed limits; do not approve release')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        '--reference', action='store_true',
        help='Require the approved reference wheel and dependencies')
    modes.add_argument(
        '--compare-with', type=Path,
        help='Compare learning with this approved reference wheel')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for logs and JSON results')
    args = parser.parse_args()
    if args.measure_learning and not args.reference:
        parser.error('--measure-learning requires --reference')
    if args.report_learning and args.compare_with is None:
        parser.error('--report-learning requires --compare-with')
    if args.compare_with is None:
        return check_wheel(args.wheel, args.output_dir,
                           reference=args.reference, learning=args.learning,
                           measure_learning=args.measure_learning)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='RUNNING', stage='reference',
                   report_only=args.report_learning)
    try:
        requirements = load_requirements(REQUIREMENTS)
        if wheel_identity(
                args.compare_with) != requirements['reference']['wheel']:
            raise ValueError(
                'Selected comparison wheel differs from the approved reference')
        for name, wheel in [('reference', args.compare_with),
                            ('candidate', args.wheel)]:
            summary['stage'] = name
            code = check_wheel(wheel, output / name,
                               reference=name == 'reference', learning=True,
                               fixed_dependencies=True,
                               report_learning=args.report_learning,
                               compare_with=summary.get('reference'))
            summary[name] = json.loads(
                (output / name / 'summary.json').read_text())
            if code:
                if summary[name].get('stage') == 'comparison':
                    summary['stage'] = 'comparison'
                raise ValueError(
                    f'{name} wheel check failed: '
                    f'{summary[name].get("error", "see saved evidence")}')
        summary['comparison'] = summary['candidate']['comparison']
        summary.update(status='MEASURED' if args.report_learning else 'PASS',
                       stage='complete')
    except Exception as exc:
        summary.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['status'] in ('PASS', 'MEASURED') else 1


if __name__ == '__main__':
    sys.exit(main())
