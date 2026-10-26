#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install engine wheels in fresh environments outside the checkout."""

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from check_pypi_requirements import read_metadata

REQUIREMENTS = Path(__file__).resolve().parents[1] / 'test/wheel_workflow.json'
QUALITY_PROFILE = REQUIREMENTS.with_name('wheel_task_quality.json')


def task_source_identity(device='cpu'):
    worker = runpy.run_path(
        str(
            REQUIREMENTS.with_name('wheel_gpu_learning.py' if device ==
                                   'cuda' else 'wheel_task_learning.py')))
    return worker['source_identity']()


def verify_task_comparison(result, *, calibration_budget=None,
                           require_quality=True):
    tasks = runpy.run_path(str(
        REQUIREMENTS.with_name('wheel_task_learning.py')))['TASKS']
    if calibration_budget is None:
        if ('calibration_fault' in result
                or 'calibration_plan_checks' in result):
            raise ValueError('A calibration comparison cannot approve release')
    else:
        expected = {(task, seed, case)
                    for task in tasks
                    for seed in range(5)
                    for case in ('learnable', 'independent')}
        proofs = result.get('calibration_plan_checks', [])
        if (result.get('calibration_fault') != dict(
                kind='reduced_training_budget', budget=calibration_budget)
                or result.get('quality_gate_approved') is not False
                or len(proofs) != len(expected)
                or {(row['task'], row['seed'], row['case'])
                    for row in proofs} != expected
                or any(not re.fullmatch('[0-9a-f]{64}', row.get(key, ''))
                       for row in proofs
                       for key in ('reference_plan_sha256',
                                   'candidate_plan_sha256')) or
                any(row['reference_plan_sha256'] == row['candidate_plan_sha256']
                    for row in proofs)):
            raise ValueError('Complete audited calibration plans are required')
    if (result.get('status') != 'MEASURED'
            or result.get('execution_passed') is not True
            or set(result.get('tasks', {})) != set(tasks)):
        raise ValueError('All seven additional task results are required')
    controls = result.get('independent_controls')
    if not isinstance(controls, dict) or set(controls) != set(tasks):
        raise ValueError('All seven independent-label controls are required')
    for name, task in result['tasks'].items():
        control = controls[name]
        if (not isinstance(control, dict) or control.get('method')
                != 'joint_monte_carlo_label_permutation'
                or control.get('draws') != 999 or control.get('seed') != 0
                or control.get('alpha') != .01 / 7 or control.get('null_method')
                != ('uniform_eligible_items'
                    if name == 'static_link' else 'shared_label_rows') or any(
                        type(control.get(key)) not in
                        (int, float) or not math.isfinite(control[key])
                        for key in ('statistic', 'p_value'))
                or not control['alpha'] < control['p_value'] <= 1):
            raise ValueError('Independent-label control evidence did not pass')
        rows = task.get('per_seed', [])
        if (task.get('metric') != tasks[name][0] or len(rows) != 5
                or {row['seed']
                    for row in rows} != set(range(5)) or any(
                        type(row['seed']) is not int or any(
                            not math.isfinite(row[key])
                            for key in ('reference', 'candidate', 'baseline',
                                        'quality_drop')) for row in rows)):
            raise ValueError(
                'Additional task comparison evidence is incomplete')
    if require_quality:
        quality = runpy.run_path(
            str(REQUIREMENTS.with_name('wheel_task_quality.py')))
        expected = quality['assess_quality'](result,
                                             json.loads(
                                                 QUALITY_PROFILE.read_text()))
        if (expected['quality_gate_approved'] is not True or any(
                result.get(key) != expected[key]
                for key in ('quality_gate_approved', 'quality_status',
                            'quality_limits', 'quality_profile_sha256',
                            'qualification', 'tasks'))):
            raise ValueError(
                'All seven quality limits need validated passing evidence')


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


def compare_predictions(left, right, threshold=0.5):
    """Measure differences only after validating the same prediction rows."""
    for prediction in (left, right):
        ids = prediction['entity_ids']
        probabilities = prediction['probabilities']
        labels = prediction['labels']
        if (not ids or len(set(ids)) != len(ids)
                or len(probabilities) != len(ids) or len(labels) != len(ids)):
            raise ValueError('Invalid prediction rows')
        for values, label in zip(probabilities, labels):
            if (len(values) != 2 or not all(
                    math.isfinite(value) and 0 <= value <= 1
                    for value in values) or
                    not math.isclose(sum(values), 1, abs_tol=1e-6, rel_tol=1e-5)
                    or label != (values[1] > threshold)):
                raise ValueError('Invalid prediction probabilities or labels')
    if left['entity_ids'] != right['entity_ids']:
        raise ValueError('Prediction entity IDs differ')
    differences = [
        abs(a - b)
        for row_a, row_b in zip(left['probabilities'], right['probabilities'])
        for a, b in zip(row_a, row_b)
    ]
    return {
        'max_abs_difference':
        max(differences),
        'mean_abs_difference':
        sum(differences) / len(differences),
        'probability_values':
        len(differences),
        'label_changes':
        sum(a != b for a, b in zip(left['labels'], right['labels'])),
    }


def summarize_variation(runs, device, expected_runs):
    """Keep reload variation separate from variation between fresh models."""
    if len(runs) != expected_runs or expected_runs < 2:
        raise ValueError('The measurement run count is incomplete')
    phases = {
        'training', 'prediction', 'repeat_prediction', 'reload_and_predict'
    }
    for run in runs:
        observed = run.get('observed_devices', [])
        if (run.get('status') != 'MEASURED' or run.get('device') != device
                or not phases.issubset(item['stage'] for item in observed)
                or any(item['device'].split(':')[0] != device
                       for item in observed)):
            raise ValueError('A measurement is missing the requested device')
        if run.get('seed') != runs[0].get('seed'):
            raise ValueError('Measurement seeds differ')
        if run.get('threshold', 0.5) != runs[0].get('threshold', 0.5):
            raise ValueError('Measurement label thresholds differ')

    def aggregate(pairs):
        measurements = [
            compare_predictions(a, b, runs[0].get('threshold', 0.5))
            for a, b in pairs
        ]
        count = sum(item['probability_values'] for item in measurements)
        return {
            'max_abs_difference':
            max(item['max_abs_difference'] for item in measurements),
            'mean_abs_difference':
            sum(item['mean_abs_difference'] * item['probability_values']
                for item in measurements) / count,
            'label_changes':
            sum(item['label_changes'] for item in measurements),
        }

    predictions = [run['predictions'] for run in runs]
    between = aggregate((a['before'], b['before'])
                        for a, b in itertools.combinations(predictions, 2))
    between['model_pairs'] = len(runs) * (len(runs) - 1) // 2
    return {
        'status':
        'MEASURED',
        'device':
        device,
        'training_runs':
        len(runs),
        'seed':
        runs[0]['seed'],
        'entities_per_run':
        len(predictions[0]['before']['entity_ids']),
        'repeat_prediction':
        aggregate((p['before'], p['repeat']) for p in predictions),
        'reload':
        aggregate((p['before'], p['after']) for p in predictions),
        'between_training_runs':
        between,
        'approved_gpu_tolerance':
        None,
    }


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
                compare_with=None, report_learning=False, device='cpu', runs=1,
                measure_variation=False, task_coverage=False,
                calibration_budget=None):
    wheel = wheel.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    measurement = (measure_learning or report_learning
                   or compare_with is not None)
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
        if device == 'cuda' and (calibration_budget is not None or (
            (learning or measure_learning) and not report_learning)):
            raise ValueError(
                'GPU learning requires a reporting-only comparison')
        if calibration_budget is not None and (not report_learning
                                               or compare_with is None):
            raise ValueError('Calibration requires a reporting-only wheel pair')
        identity = wheel_identity(wheel)
        summary['wheel'] = identity
        summary['stage'] = 'requirements'
        requirements = load_requirements(REQUIREMENTS)
        shutil.copyfile(REQUIREMENTS, output / 'requirements.json')
        summary['requirements_sha256'] = hashlib.sha256(
            REQUIREMENTS.read_bytes()).hexdigest()
        summary['configuration'] = dict(requirements['configuration'],
                                        device=device)
        summary['fixture'] = requirements['fixture']
        summary['reference_mode'] = reference
        summary['requested_device'] = device
        summary['training_runs'] = runs
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
            summary['workflow_sha256'] = hashlib.sha256(
                source.read_bytes()).hexdigest()
            exercise = root / source.name
            shutil.copyfile(source, exercise)
            shutil.copyfile(REQUIREMENTS, exercise.with_suffix('.json'))
            results = []
            for index in range(runs):
                run_root = root
                evidence = output
                stage = 'workflow'
                if measure_variation:
                    run_root = root / f'run-{index + 1:03d}'
                    run_root.mkdir()
                    evidence = output / run_root.name
                    evidence.mkdir()
                    stage = run_root.name
                command = [
                    python, '-I',
                    str(exercise), '--version', identity['version'],
                    '--output-dir',
                    str(evidence), '--device', device
                ]
                if measure_variation or device == 'cuda':
                    command.append('--measure-variation')
                run(stage, command, run_root, timeout=300)
                result = json.loads(
                    (evidence / 'workflow-result.json').read_text())
                if measure_variation or device == 'cuda':
                    # All functional checks remain required. Numerical agreement
                    # is measured until a device-specific limit is approved.
                    checks = [
                        check for check in requirements['required_checks']
                        if check != 'reload_predictions_match'
                    ]
                    verify_workflow_result(result, checks, status='MEASURED')
                    results.append(result)
                    if device == 'cuda':
                        gpu = runpy.run_path(
                            str(REQUIREMENTS.with_name(
                                'wheel_gpu_learning.py')))
                        gpu['verify_devices'](
                            result,
                            ('training', 'prediction', 'reload_and_predict'))
                else:
                    verify_workflow_result(result,
                                           requirements['required_checks'])
            if measure_variation:
                summary['variation'] = summarize_variation(
                    results, device, runs)
                (output / 'variation.json').write_text(
                    json.dumps(summary['variation'], indent=2))
            if learning or measurement:
                learning = root / 'wheel_learning.py'
                shutil.copyfile(REQUIREMENTS.with_name('wheel_learning.py'),
                                learning)
                if device == 'cuda':
                    task_worker = runpy.run_path(
                        str(REQUIREMENTS.with_name('wheel_task_learning.py')))
                    for name in (*task_worker['SOURCES'],
                                 'wheel_gpu_learning.py'):
                        shutil.copyfile(REQUIREMENTS.with_name(name),
                                        root / name)
                try:
                    run('learning', [
                        python, '-I',
                        str(root / 'wheel_gpu_learning.py' if device ==
                            'cuda' else learning), '--version',
                        identity['version'], '--output-dir',
                        str(output / 'learning'),
                        *(['--group', 'binary', '--device', 'cuda'] if device
                          == 'cuda' else ['--measure'] if measurement else [])
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
                if device == 'cuda':
                    gpu['verify_report'](learning_result, additional=False)
                if task_coverage:
                    summary['task_evidence_dir'] = str(output / 'tasks')
                    task_worker = runpy.run_path(
                        str(REQUIREMENTS.with_name('wheel_task_learning.py')))
                    for name in task_worker['SOURCES']:
                        shutil.copyfile(REQUIREMENTS.with_name(name),
                                        root / name)
                    task_result = output / 'tasks/task-result.json'
                    try:
                        run('task_learning', [
                            python, '-I',
                            str(root /
                                ('wheel_gpu_learning.py' if device == 'cuda'
                                 else 'wheel_task_learning.py')), '--version',
                            identity['version'], '--output-dir',
                            str(output / 'tasks'),
                            *(['--group', 'additional', '--device', 'cuda']
                              if device == 'cuda' else []), *([
                                  '--calibration-budget',
                                  *map(str, calibration_budget)
                              ] if calibration_budget is not None else [])
                        ], root, timeout=2400)
                    finally:
                        if task_result.is_file():
                            summary['task_learning'] = json.loads(
                                task_result.read_text())
                    task_report = summary['task_learning']
                    verify_workflow_result(task_report, [], status='MEASURED')
                    if (task_report.get('source_sha256')
                            != task_source_identity(device)
                            or task_report.get('seeds') != list(range(5))):
                        raise ValueError(
                            'Additional task evidence differs from the harness')
                    if device == 'cuda':
                        gpu['verify_report'](task_report, additional=True)
                if compare_with is not None:
                    inputs = root / 'comparison-input.json'
                    inputs.write_text(
                        json.dumps([
                            compare_with,
                            dict(summary, status='MEASURED'),
                            requirements['learning']
                        ]))
                    comparison = output / 'comparison-result.json'
                    reporter = root / 'wheel_learning_report.py'
                    shutil.copyfile(REQUIREMENTS.with_name(reporter.name),
                                    reporter)
                    if task_coverage:
                        for name in ('wheel_task_quality.py',
                                     'wheel_task_quality.json'):
                            shutil.copyfile(REQUIREMENTS.with_name(name),
                                            root / name)
                    try:
                        run('comparison', [
                            python, '-I',
                            str(reporter),
                            str(inputs),
                            str(comparison),
                            *(['--tasks'] if task_coverage else []), *([
                                '--calibration-task-dirs',
                                compare_with['task_evidence_dir'],
                                summary['task_evidence_dir']
                            ] if calibration_budget is not None else []),
                            *([] if report_learning else ['--release'])
                        ], root, timeout=180 if task_coverage else 60)
                    finally:
                        if comparison.is_file():
                            summary['comparison'] = json.loads(
                                comparison.read_text())
                    verify_workflow_result(
                        summary['comparison'], [],
                        status='MEASURED' if report_learning else 'PASS')
                    if task_coverage:
                        verify_task_comparison(
                            summary['comparison'].get('additional_tasks', {}),
                            calibration_budget=calibration_budget,
                            require_quality=not report_learning)
                    if (not report_learning
                            and summary['comparison'].get('release_approved')
                            is not True):
                        raise ValueError(
                            'Release comparison did not approve release')
        summary.update(
            status='MEASURED' if measurement or measure_variation
            or device == 'cuda' else 'PASS', stage='complete')
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
    parser.add_argument(
        '--calibration-budget', type=int, nargs=2, metavar=('EPOCHS', 'STEPS'),
        help='Measure weaker candidate training in reporting mode')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument(
        '--measure-variation', action='store_true',
        help='Record variation without approving numerical limits')
    args = parser.parse_args()
    if args.calibration_budget is not None:
        if not args.report_learning or args.compare_with is None:
            parser.error(
                'Calibration requires --report-learning and --compare-with')
        settings = runpy.run_path(
            str(REQUIREMENTS.with_name('wheel_learning.py')))['SETTINGS']
        ceiling = [settings['epochs'], settings['steps_per_epoch']]
        if (any(not 1 <= new <= old
                for new, old in zip(args.calibration_budget, ceiling))
                or args.calibration_budget == ceiling):
            parser.error('Calibration must reduce the normal training budget')
    if args.measure_variation and (args.learning or args.measure_learning
                                   or args.report_learning
                                   or args.compare_with is not None):
        parser.error(
            'Variation measurements are separate from learning comparisons')
    if args.measure_variation and args.runs < 2:
        parser.error(
            'Variation measurements require at least two training runs')
    if not args.measure_variation and args.runs != 1:
        parser.error('Repeated runs require --measure-variation')
    if args.device == 'cuda' and (args.calibration_budget is not None
                                  or args.learning or args.measure_learning or
                                  (args.compare_with is not None
                                   and not args.report_learning)):
        parser.error(
            'GPU learning requires --report-learning and --compare-with')
    if args.measure_learning and not args.reference:
        parser.error('--measure-learning requires --reference')
    if args.report_learning and args.compare_with is None:
        parser.error('--report-learning requires --compare-with')
    if args.compare_with is None:
        return check_wheel(args.wheel, args.output_dir,
                           reference=args.reference, learning=args.learning,
                           measure_learning=args.measure_learning,
                           device=args.device, runs=args.runs,
                           measure_variation=args.measure_variation)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='RUNNING', stage='reference',
                   report_only=args.report_learning)
    if args.calibration_budget is not None:
        summary['calibration_fault'] = dict(kind='reduced_training_budget',
                                            budget=args.calibration_budget)
    try:
        requirements = load_requirements(REQUIREMENTS)
        if wheel_identity(
                args.compare_with) != requirements['reference']['wheel']:
            raise ValueError(
                'Selected comparison wheel differs from the approved reference')
        for name, wheel in [('reference', args.compare_with),
                            ('candidate', args.wheel)]:
            summary['stage'] = name
            code = check_wheel(
                wheel, output / name, reference=name == 'reference',
                learning=True, fixed_dependencies=True, measure_learning=True,
                task_coverage=True, device=args.device,
                calibration_budget=(args.calibration_budget
                                    if name == 'candidate' else None),
                report_learning=args.report_learning,
                compare_with=summary.get('reference'))
            summary[name] = json.loads(
                (output / name / 'summary.json').read_text())
            if 'comparison' in summary[name]:
                summary['comparison'] = summary[name]['comparison']
            if code:
                if summary[name].get('stage') == 'comparison':
                    summary['stage'] = 'comparison'
                raise ValueError(
                    f'{name} wheel check failed: '
                    f'{summary[name].get("error", "see saved evidence")}')
            verify_workflow_result(summary[name], [], status='MEASURED')
        summary['comparison'] = summary['candidate']['comparison']
        verify_workflow_result(
            summary['comparison'], [],
            status='MEASURED' if args.report_learning else 'PASS')
        verify_task_comparison(
            summary['comparison'].get('additional_tasks', {}),
            calibration_budget=args.calibration_budget,
            require_quality=not args.report_learning)
        if (not args.report_learning
                and summary['comparison'].get('release_approved') is not True):
            raise ValueError('Release comparison did not approve release')
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
