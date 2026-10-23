# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

SCRIPTS = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/scripts'
RUNNER = SCRIPTS / 'smoke_wheel.py'


class WheelSmokeTest(unittest.TestCase):
    def task_comparison(self):
        metrics = dict(multiclass='accuracy', multilabel='ap_macro',
                       multilabel_ranking='recall@5', regression='mae',
                       forecasting='mae', static_link='recall@5',
                       temporal_link='recall@5')
        return dict(
            status='MEASURED', execution_passed=True, tasks={
                name:
                dict(
                    metric=metric, per_seed=[
                        dict(seed=i, reference=.5, candidate=.5, baseline=.5,
                             quality_drop=0.) for i in range(5)
                    ])
                for name, metric in metrics.items()
            }, independent_controls={
                name:
                dict(
                    method='joint_monte_carlo_label_permutation', draws=999,
                    seed=0, alpha=.01 / 7, statistic=0., p_value=1.,
                    null_method=('uniform_eligible_items' if name
                                 == 'static_link' else 'shared_label_rows'))
                for name in metrics
            })

    def test_task_comparison_requires_complete_passing_control_results(self):
        self.runner.verify_task_comparison(self.task_comparison())
        for mutation in ('missing', 'missing_task', 'missing_value', 'nan',
                         'equal_alpha', 'failed', 'widened_alpha', 'range',
                         'calibration'):
            result = self.task_comparison()
            control = result['independent_controls']['forecasting']
            if mutation == 'missing':
                del result['independent_controls']
            elif mutation == 'missing_task':
                del result['independent_controls']['forecasting']
            elif mutation == 'missing_value':
                del control['p_value']
            elif mutation == 'nan':
                control['p_value'] = float('nan')
            elif mutation == 'equal_alpha':
                control['p_value'] = .01 / 7
            elif mutation == 'failed':
                control['p_value'] = .001
            elif mutation == 'widened_alpha':
                control['alpha'] = .5
            elif mutation == 'calibration':
                result['calibration_fault'] = dict(
                    kind='reduced_training_budget', budget=[1, 2])
            else:
                control['p_value'] = 1.1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.runner.verify_task_comparison(result)

    def calibration_comparison(self):
        result = self.task_comparison()
        result.update(
            quality_gate_approved=False,
            calibration_fault=dict(kind='reduced_training_budget',
                                   budget=[1, 2]), calibration_plan_checks=[
                                       dict(task=task, seed=seed, case=case,
                                            reference_plan_sha256='a' * 64,
                                            candidate_plan_sha256='b' * 64)
                                       for task in result['tasks']
                                       for seed in range(5)
                                       for case in ('learnable', 'independent')
                                   ])
        return result

    def test_calibration_reports_loss_but_never_approves_release(self):
        wheel, reference = self.wheel(), self.wheel(version='1.0.0')
        requirements = self.runner.load_requirements(self.runner.REQUIREMENTS)
        requirements['reference']['wheel'] = self.runner.wheel_identity(
            reference)
        output = self.root / 'calibration'
        observed = []

        def check_wheel(selected, evidence, **options):
            evidence.mkdir()
            observed.append(options.get('calibration_budget'))
            result = dict(status='MEASURED', stage='complete',
                          wheel=self.runner.wheel_identity(selected))
            if options.get('compare_with') is not None:
                result['comparison'] = dict(
                    status='MEASURED', release_approved=False,
                    additional_tasks=self.calibration_comparison())
            (evidence / 'summary.json').write_text(json.dumps(result))
            return 0

        with (patch.object(self.runner, 'check_wheel', check_wheel),
              patch.object(self.runner, 'load_requirements',
                           return_value=requirements),
              patch.object(sys, 'argv', [
                  str(RUNNER),
                  str(wheel), '--compare-with',
                  str(reference), '--report-learning', '--calibration-budget',
                  '1', '2', '--output-dir',
                  str(output)
              ]), patch('sys.stdout', new_callable=io.StringIO),
              patch('sys.stderr', new_callable=io.StringIO)):
            try:
                code = self.runner.main()
            except SystemExit as error:
                code = error.code
        self.assertEqual(code, 0)
        self.assertEqual(observed, [None, [1, 2]])
        result = json.loads((output / 'summary.json').read_text())
        self.assertEqual(result['status'], 'MEASURED')
        self.assertTrue(result['report_only'])
        self.assertFalse(result['comparison']['release_approved'])
        self.assertEqual(result['calibration_fault'],
                         dict(kind='reduced_training_budget', budget=[1, 2]))

    def test_calibration_options_cannot_enter_a_release_run(self):
        for extra in ([], ['--report-learning'],
                      ['--compare-with', 'unused.whl']):
            with self.subTest(extra=extra):
                output = self.root / ('invalid-calibration-' + str(len(extra)))
                completed = subprocess.run([
                    sys.executable,
                    str(RUNNER),
                    str(self.wheel()), '--calibration-budget', '1', '2',
                    '--output-dir',
                    str(output), *extra
                ], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(output.exists())

    def setUp(self):
        self.assertTrue(RUNNER.is_file(), 'The wheel smoke runner is missing')
        sys.path.insert(0, str(SCRIPTS))
        self.runner = importlib.import_module('smoke_wheel')
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def wheel(self, name='kumo-engine', version='1.2.3', requires=''):
        distribution = f'{name.replace("-", "_")}-{version}'
        wheel = self.root / f'{distribution}-py3-none-any.whl'
        metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'
        if requires:
            metadata += f'Requires-Dist: {requires}\n'
        with ZipFile(wheel, 'w') as archive:
            archive.writestr(f'{distribution}.dist-info/METADATA', metadata)
            archive.writestr(
                f'{distribution}.dist-info/WHEEL',
                'Wheel-Version: 1.0\nRoot-Is-Purelib: true\n'
                'Tag: py3-none-any\n')
            archive.writestr(f'{distribution}.dist-info/RECORD', '')
        return wheel

    def test_reads_selected_version_and_rejects_another_distribution(self):
        wheel = self.wheel()
        identity = self.runner.wheel_identity(wheel)
        self.assertEqual(identity['version'], '1.2.3')
        self.assertEqual(identity['sha256'],
                         hashlib.sha256(wheel.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'kumo-engine'):
            self.runner.wheel_identity(self.wheel(name='another-package'))

    def test_install_report_must_match_both_version_and_fingerprint(self):
        identity = {'version': '1.2.3', 'sha256': 'a' * 64}
        engine = {
            'metadata': {
                'name': 'kumo-engine',
                'version': '1.2.3'
            },
            'download_info': {
                'archive_info': {
                    'hashes': {
                        'sha256': 'a' * 64
                    }
                }
            }
        }
        report = {'install': [engine]}
        self.runner.verify_install(report, identity)
        engine['download_info']['archive_info']['hashes']['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.runner.verify_install(report, identity)
        engine['download_info']['archive_info']['hashes']['sha256'] = 'a' * 64
        engine['metadata']['version'] = '9.9.9'
        with self.assertRaisesRegex(ValueError, 'version'):
            self.runner.verify_install(report, identity)

    def test_subprocess_failure_preserves_output_and_exit_code(self):
        log = self.root / 'failed.log'
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.runner.run_command([
                sys.executable, '-c',
                'print("deliberate failure"); raise SystemExit(7)'
            ], self.root, log)
        self.assertEqual(error.exception.returncode, 7)
        self.assertIn('deliberate failure', log.read_text())

    def test_pip_cannot_resolve_dependencies_from_host_configuration(self):
        self.wheel(name='smoke-local-only-dependency')
        wheel = self.wheel(requires='smoke-local-only-dependency==1.2.3')
        config = self.root / 'pip.conf'
        config.write_text(f'[global]\nfind-links = {self.root}\n')
        venv = self.root / 'venv'
        subprocess.run([sys.executable, '-I', '-m', 'venv',
                        str(venv)], check=True, capture_output=True)
        log = self.root / 'install.log'
        # Stay offline. Only the host configuration can supply this dependency.
        with patch.dict(os.environ, {'PIP_CONFIG_FILE': str(config)}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.runner.run_command([
                    str(venv / 'bin/python'), '-I', '-m', 'pip', '--isolated',
                    'install', '--no-cache-dir', '--no-index',
                    str(wheel)
                ], self.root, log)
        self.assertIn(
            'No matching distribution found for '
            'smoke-local-only-dependency', log.read_text())

        # Build requirements use another pip process. Keep them isolated too.
        source = self.root / 'source'
        source.mkdir()
        shutil.copyfile(self.wheel(),
                        source / 'kumo_engine-1.2.3-py3-none-any.whl')
        (source / 'pyproject.toml').write_text(
            '[build-system]\nrequires = ["smoke-local-only-dependency==1.2.3"]\n'
            'build-backend = "backend"\nbackend-path = ["."]\n')
        (source / 'backend.py').write_text(
            'import importlib.metadata, pathlib, shutil\n'
            'assert importlib.metadata.version("smoke-local-only-dependency") == "1.2.3"\n'
            'def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n'
            '    wheel = next(pathlib.Path(__file__).parent.glob("*.whl"))\n'
            '    shutil.copyfile(wheel, pathlib.Path(wheel_directory) / wheel.name)\n'
            '    return wheel.name\n')
        log = self.root / 'source-build.log'
        with patch.dict(os.environ, {'PIP_FIND_LINKS': str(self.root)}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.runner.run_command([
                    str(venv / 'bin/python'), '-I', '-m', 'pip', '--isolated',
                    'install', '--no-cache-dir', '--no-index',
                    str(source)
                ], self.root, log)
        self.assertIn(
            'No matching distribution found for '
            'smoke-local-only-dependency', log.read_text())

    def test_bad_wheel_fails_the_command_and_saves_a_failure_report(self):
        output = self.root / 'evidence'
        completed = subprocess.run([
            sys.executable,
            str(RUNNER),
            str(self.wheel(name='wrong')), '--output-dir',
            str(output)
        ], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        report = json.loads((output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['stage'], 'wheel_identity')
        self.assertIn('kumo-engine', report['error'])

    def test_learning_keeps_the_wheel_identity_guard(self):
        output = self.root / 'learning-evidence'
        completed = subprocess.run([
            sys.executable,
            str(RUNNER),
            str(self.wheel(name='wrong')), '--learning', '--output-dir',
            str(output)
        ], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue((output / 'summary.json').is_file(), completed.stderr)
        result = json.loads((output / 'summary.json').read_text())
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['stage'], 'wheel_identity')

    def test_learning_status_and_worker_errors_are_preserved(self):
        wheel = self.wheel()
        requirements = self.runner.load_requirements(self.runner.REQUIREMENTS)
        requirements['configuration'].update(
            os=sys.platform, machine=self.runner.platform.machine(),
            python=f'{sys.version_info.major}.{sys.version_info.minor}')
        requirements['reference']['wheel'] = self.runner.wheel_identity(wheel)
        requirements['reference']['dependencies'] = {}
        install = {
            'install': [
                dict(
                    metadata=dict(name='kumo-engine', version='1.2.3'),
                    download_info=dict(archive_info=dict(hashes=dict(
                        sha256=hashlib.sha256(
                            wheel.read_bytes()).hexdigest()))))
            ]
        }
        workflow = dict(
            status='PASS',
            checks={name: 'PASS'
                    for name in requirements['required_checks']})
        requirements['learning'] = {}
        cases = [(None, 'learning', 1), ('RUNNING', 'learning', 1),
                 ('FAIL', 'learning', 1), ('PASS', 'learning', 0),
                 ('MEASURED', 'learning', 1), ('MEASURED', 'measure', 0),
                 ('PASS', 'measure', 1), ('FAIL', 'failed-process', 1)]
        cases += [(status, 'comparison', 0 if status == 'PASS' else 1)
                  for status in (None, 'RUNNING', 'FAIL', 'MEASURED', 'PASS')]
        cases += [(status, 'report-comparison',
                   0 if status == 'MEASURED' else 1)
                  for status in (None, 'RUNNING', 'FAIL', 'MEASURED', 'PASS')]
        cases += [('PASS', mode, 1)
                  for mode in ('comparison-nonzero', 'comparison-timeout',
                               'comparison-malformed', 'comparison-unapproved')]
        cases += [('MEASURED', mode, code)
                  for mode, code in (('task-pass', 0), ('task-missing',
                                                        1), ('task-failed', 1),
                                     ('task-process-failed',
                                      1), ('task-source-mismatch', 1))]
        for status, mode, expected in cases:
            with self.subTest(status=status, mode=mode):
                output = self.root / f'{mode}-{status}'
                interpreters = {}
                compares = mode.startswith(
                    'comparison') or mode == 'report-comparison'

                def commands(command, cwd, log, timeout=60):
                    # Skip installation and training. Keep result handling real.
                    log.write_text('')
                    interpreters[log.stem] = command[0]
                    if log.stem == 'install':
                        (output / 'install-report.json').write_text(
                            json.dumps(install))
                    elif log.stem == 'workflow':
                        self.assertEqual(command[command.index('--device') + 1],
                                         'cpu')
                        self.assertNotIn('--measure-variation', command)
                        (output / 'workflow-result.json').write_text(
                            json.dumps(workflow))
                    elif log.stem == 'learning' and (status is not None
                                                     or compares):
                        result = output / 'learning/learning-result.json'
                        result.parent.mkdir()
                        learning_status = status
                        if compares:
                            self.assertIn('--measure', command)
                            learning_status = 'MEASURED' if '--measure' in command else 'PASS'
                        result.write_text(
                            json.dumps(dict(status=learning_status)))
                        if mode == 'failed-process':
                            result.write_text(
                                json.dumps(
                                    dict(status='FAIL',
                                         error='learnable AUROC is too low',
                                         mean_auroc={'learnable': 0.891132})))
                            subprocess.run(
                                [sys.executable, '-c', 'raise SystemExit(1)'],
                                check=True)
                    elif log.stem == 'comparison' and status is not None:
                        self.assertEqual('--release' in command, mode
                                         != 'report-comparison')
                        self.assertEqual(
                            Path(command[2]).name, 'wheel_learning_report.py')
                        result = dict(status=status,
                                      release_approved=status == 'PASS',
                                      quality_status='FAIL')
                        if mode == 'comparison-unapproved':
                            result['release_approved'] = False
                        (output / 'comparison-result.json').write_text(
                            'not json' if mode ==
                            'comparison-malformed' else json.dumps(result))
                        if mode == 'comparison-nonzero':
                            raise subprocess.CalledProcessError(7, command)
                        if mode == 'comparison-timeout':
                            raise subprocess.TimeoutExpired(command, timeout)
                    elif log.stem == 'task_learning' and mode != 'task-missing':
                        result = dict(
                            status='MEASURED', seeds=list(range(5)),
                            source_sha256=self.runner.task_source_identity())
                        if mode == 'task-failed':
                            result['status'] = 'FAIL'
                        if mode == 'task-source-mismatch':
                            result['source_sha256'] = 'a' * 64
                        result_path = output / 'tasks/task-result.json'
                        result_path.parent.mkdir()
                        result_path.write_text(json.dumps(result))
                        if mode == 'task-process-failed':
                            raise subprocess.CalledProcessError(7, command)

                with (patch.object(self.runner, 'run_command', commands),
                      patch.object(self.runner, 'load_requirements',
                                   return_value=requirements),
                      patch('sys.stdout', new_callable=io.StringIO),
                      patch('sys.stderr', new_callable=io.StringIO)):
                    code = self.runner.check_wheel(
                        wheel, output, learning=True,
                        reference=mode == 'measure',
                        measure_learning=mode == 'measure'
                        or mode.startswith('task-'),
                        task_coverage=mode.startswith('task-'),
                        report_learning=mode == 'report-comparison',
                        compare_with={'status': 'MEASURED'}
                        if compares else None)
                    self.assertEqual(code, expected)
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(report['status'],
                                 ('MEASURED' if mode == 'measure' or compares
                                  or mode.startswith('task-') else 'PASS')
                                 if expected == 0 else 'FAIL')
                self.assertEqual(
                    report['stage'], 'complete'
                    if expected == 0 else 'comparison' if compares else
                    'task_learning' if mode.startswith('task-') else 'learning')
                if mode == 'failed-process':
                    self.assertIn('learnable AUROC is too low', report['error'])
                    self.assertEqual(report['learning']['status'], 'FAIL')
                    self.assertEqual(report['learning']['mean_auroc'],
                                     {'learnable': 0.891132})
                if mode in ('comparison-nonzero', 'comparison-timeout'):
                    self.assertEqual(report['comparison']['status'], 'PASS')
                    self.assertTrue(report['comparison']['release_approved'])
                if compares:
                    self.assertEqual(interpreters['comparison'],
                                     interpreters['install'])
                    self.assertEqual(interpreters['comparison'],
                                     interpreters['learning'])
                if mode == 'task-pass':
                    self.assertEqual(interpreters['task_learning'],
                                     interpreters['install'])
                    self.assertEqual(report['task_learning']['status'],
                                     'MEASURED')

    def test_comparison_preserves_each_result(self):
        wheel = self.wheel()
        reference = self.wheel(version='1.0.0')
        requirements = self.runner.load_requirements(self.runner.REQUIREMENTS)
        requirements['reference']['wheel'] = self.runner.wheel_identity(
            reference)
        for failed_stage in ('reference', 'candidate', 'comparison', None):
            output = self.root / f'comparison-{failed_stage}'

            def check_wheel(wheel, evidence, **kwargs):
                evidence.mkdir()
                self.assertTrue(kwargs['learning'])
                self.assertTrue(kwargs['fixed_dependencies'])
                self.assertEqual(kwargs['reference'],
                                 evidence.name == 'reference')
                previous = kwargs.get('compare_with')
                if evidence.name == 'candidate':
                    self.assertEqual(previous['wheel'],
                                     requirements['reference']['wheel'])
                else:
                    self.assertIsNone(previous)
                failed = evidence.name == failed_stage or (
                    failed_stage == 'comparison'
                    and evidence.name == 'candidate')
                stage = 'comparison' if failed_stage == 'comparison' and failed else 'learning'
                report = dict(status='FAIL' if failed else 'MEASURED',
                              stage=stage,
                              wheel=self.runner.wheel_identity(wheel))
                if previous:
                    report['comparison'] = dict(
                        status='PASS', release_approved=True,
                        additional_tasks=self.task_comparison(),
                        mean_auroc_drop=0)
                if failed:
                    report[
                        'error'] = 'AUROC drop' if stage == 'comparison' else 'learning failed'
                (evidence / 'summary.json').write_text(json.dumps(report))
                return 1 if failed else 0

            with (patch.object(self.runner, 'check_wheel', check_wheel),
                  patch.object(self.runner, 'load_requirements',
                               return_value=requirements),
                  patch.object(sys, 'argv', [
                      str(RUNNER),
                      str(wheel), '--compare-with',
                      str(reference), '--output-dir',
                      str(output)
                  ]), patch('sys.stdout', new_callable=io.StringIO),
                  patch('sys.stderr', new_callable=io.StringIO)):
                code = self.runner.main()
            self.assertEqual(code, 1 if failed_stage else 0)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['status'],
                             'FAIL' if failed_stage else 'PASS')
            self.assertEqual(report['stage'], failed_stage or 'complete')
            if failed_stage == 'comparison':
                self.assertIn('AUROC drop', report['error'])
            elif failed_stage is None:
                self.assertEqual(report['comparison']['status'], 'PASS')
                self.assertEqual(report['comparison']['mean_auroc_drop'], 0)

    def test_release_comparison_requires_measured_runs_and_explicit_approval(
            self):
        wheel = self.wheel()
        reference = self.wheel(version='1.0.0')
        requirements = self.runner.load_requirements(self.runner.REQUIREMENTS)
        requirements['reference']['wheel'] = self.runner.wheel_identity(
            reference)
        passed = dict(status='PASS', release_approved=True,
                      additional_tasks=self.task_comparison(),
                      quality_status='FAIL',
                      failed_limits=[dict(check='max_seed_drop')],
                      mean_auroc_drop=.001)
        cases = [
            ('pass', passed, 0, 0),
            ('missing', None, 0, 1),
            ('unapproved', dict(passed, release_approved=False), 0, 1),
            ('missing-approval', dict(status='PASS'), 0, 1),
            ('measured', dict(passed, status='MEASURED'), 0, 1),
            ('failed', dict(passed, status='FAIL'), 0, 1),
            ('execution-failed', passed, 7, 1),
            ('tasks-missing', dict(passed, additional_tasks={}), 0, 1),
            ('tasks-failed',
             dict(
                 passed, additional_tasks=dict(self.task_comparison(),
                                               execution_passed=False)), 0, 1),
        ]
        for name, comparison, execution_code, expected in cases:
            with self.subTest(case=name):
                output = self.root / f'release-comparison-{name}'
                observed = []

                def check_wheel(selected, evidence, **options):
                    evidence.mkdir()
                    observed.append(options)
                    result = dict(status='MEASURED', stage='complete',
                                  wheel=self.runner.wheel_identity(selected),
                                  learning=dict(status='MEASURED'))
                    if options.get('compare_with'
                                   ) is not None and comparison is not None:
                        result['comparison'] = comparison
                    (evidence / 'summary.json').write_text(json.dumps(result))
                    return execution_code if options.get(
                        'compare_with') is not None else 0

                with (patch.object(self.runner, 'check_wheel', check_wheel),
                      patch.object(self.runner, 'load_requirements',
                                   return_value=requirements),
                      patch.object(sys, 'argv', [
                          str(RUNNER),
                          str(wheel), '--compare-with',
                          str(reference), '--output-dir',
                          str(output)
                      ]), patch('sys.stdout', new_callable=io.StringIO),
                      patch('sys.stderr', new_callable=io.StringIO)):
                    code = self.runner.main()
                result = json.loads((output / 'summary.json').read_text())
                self.assertEqual(code, expected)
                self.assertEqual(result['status'],
                                 'FAIL' if expected else 'PASS')
                self.assertFalse(result['report_only'])
                if name == 'execution-failed':
                    self.assertEqual(result['comparison'], passed)
                self.assertEqual(len(observed), 2)
                for options in observed:
                    self.assertTrue(options.get('measure_learning'))
                    self.assertTrue(options.get('task_coverage'))
                    self.assertFalse(options['report_learning'])

    def test_reference_mode_rejects_a_different_wheel_before_installation(self):
        reference_name = json.loads(self.runner.REQUIREMENTS.read_text()
                                    )['reference']['wheel']['filename']
        for version in ('1.2.3', '1.0.0'):
            with self.subTest(version=version):
                output = self.root / f'reference-evidence-{version}'
                wheel = self.wheel(version=version).rename(self.root /
                                                           reference_name)
                completed = subprocess.run([
                    sys.executable,
                    str(RUNNER),
                    str(wheel), '--reference', '--output-dir',
                    str(output)
                ], capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertTrue((output / 'summary.json').is_file(),
                                completed.stderr)
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['stage'], 'reference')
                self.assertIn('approved reference', report['error'])
                self.assertFalse((output / 'install.log').exists())

    def test_reporting_collects_both_wheels_but_preserves_execution_failures(
            self):
        wheel = self.wheel()
        reference = self.wheel(version='1.0.0')
        requirements = self.runner.load_requirements(self.runner.REQUIREMENTS)
        requirements['reference']['wheel'] = self.runner.wheel_identity(
            reference)
        for failed_stage in (None, 'install', 'workflow', 'learning'):
            output = self.root / f'reporting-{failed_stage}'

            def check_wheel(selected, evidence, **options):
                evidence.mkdir()
                measured = options.get('report_learning', False)
                failed = failed_stage is not None
                report = dict(
                    status='FAIL' if failed else
                    'MEASURED' if measured else 'PASS', stage=failed_stage
                    or 'complete', wheel=self.runner.wheel_identity(selected))
                if measured:
                    report['learning'] = dict(status='MEASURED',
                                              mean_auroc={'learnable': .1})
                if options.get('compare_with'):
                    report['comparison'] = dict(
                        status='MEASURED', quality_status='FAIL',
                        additional_tasks=self.task_comparison(),
                        failed_limits=[dict(check='learnable_min')],
                        mean_auroc_drop=.2, release_approved=False)
                if failed:
                    report['error'] = 'deliberate execution failure'
                (evidence / 'summary.json').write_text(json.dumps(report))
                return 1 if failed else 0

            with (patch.object(self.runner, 'check_wheel', check_wheel),
                  patch.object(self.runner, 'load_requirements',
                               return_value=requirements),
                  patch.object(sys, 'argv', [
                      str(RUNNER),
                      str(wheel), '--compare-with',
                      str(reference), '--report-learning', '--output-dir',
                      str(output)
                  ]), patch('sys.stdout', new_callable=io.StringIO),
                  patch('sys.stderr', new_callable=io.StringIO)):
                try:
                    code = self.runner.main()
                except SystemExit as exc:
                    code = exc.code
            self.assertEqual(code, 1 if failed_stage else 0)
            result = json.loads((output / 'summary.json').read_text())
            self.assertEqual(result['status'],
                             'FAIL' if failed_stage else 'MEASURED')
            if failed_stage:
                self.assertNotIn('candidate', result)
            else:
                self.assertIn('reference', result)
                self.assertIn('candidate', result)
                self.assertEqual(result['comparison']['quality_status'], 'FAIL')
                self.assertFalse(result['comparison']['release_approved'])

    def test_reporting_requires_a_comparison_wheel(self):
        output = self.root / 'invalid-reporting'
        completed = subprocess.run([
            sys.executable,
            str(RUNNER),
            str(self.wheel()), '--report-learning', '--output-dir',
            str(output)
        ], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('requires --compare-with', completed.stderr)
        self.assertFalse(output.exists())

    def test_missing_or_invalid_requirements_are_rejected(self):
        original = json.loads(self.runner.REQUIREMENTS.read_text())
        invalid = [
            ('fixture', 'seed', None),
            ('fixture', 'entities', 0),
            ('configuration', 'device', 'cuda'),
            ('required_checks', None, []),
            ('reference', 'dependencies', {}),
        ]
        for section, key, value in invalid:
            with self.subTest(section=section, key=key):
                requirements = json.loads(json.dumps(original))
                if key is None:
                    requirements[section] = value
                elif value is None:
                    del requirements[section][key]
                else:
                    requirements[section][key] = value
                path = self.root / 'requirements.json'
                path.write_text(json.dumps(requirements))
                with self.assertRaises(ValueError):
                    self.runner.load_requirements(path)

    def test_reference_evidence_must_pass_and_have_an_exact_fingerprint(self):
        original = json.loads(self.runner.REQUIREMENTS.read_text())
        for section, field, value in [
            ('evidence', 'status', 'FAIL'),
            ('wheel', 'sha256', 'not-a-fingerprint'),
        ]:
            with self.subTest(field=field):
                requirements = json.loads(json.dumps(original))
                requirements['reference'][section][field] = value
                path = self.root / 'requirements.json'
                path.write_text(json.dumps(requirements))
                with self.assertRaises(ValueError):
                    self.runner.load_requirements(path)

    def test_reference_dependency_versions_must_match_the_complete_report(self):
        report = {
            'install': [
                dict(metadata=dict(name='kumo-engine', version='1.0.0')),
                dict(metadata=dict(name='Some_Package', version='1.2.3')),
            ]
        }
        self.runner.verify_reference_dependencies(report,
                                                  {'some-package': '1.2.3'})
        for expected in ({}, {
                'some-package': '9.9.9'
        }, {
                'some-package': '1.2.3',
                'missing': '1.0'
        }):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, 'dependencies'):
                    self.runner.verify_reference_dependencies(report, expected)

    def test_missing_or_failed_workflow_checks_cannot_report_success(self):
        self.runner.verify_workflow_result(
            dict(status='PASS', checks=dict(training='PASS')), ['training'])
        for status, checks in [
            ('FAIL', dict(training='PASS')),
            ('PASS', {}),
            ('PASS', dict(training='FAIL')),
        ]:
            with self.subTest(status=status, checks=checks):
                with self.assertRaisesRegex(ValueError, 'missing or failed'):
                    self.runner.verify_workflow_result(
                        dict(status=status, checks=checks), ['training'])

    def measured_run(self, probabilities, repeat=None, reload=None,
                     threshold=0.5):
        def predictions(values):
            return {
                'entity_ids': [0, 1],
                'probabilities': [[1 - value, value] for value in values],
                'labels': [value > threshold for value in values],
            }

        return {
            'status':
            'MEASURED',
            'device':
            'cuda',
            'seed':
            0,
            'threshold':
            threshold,
            'observed_devices': [{
                'stage': stage,
                'device': 'cuda:0'
            } for stage in ('training', 'prediction', 'repeat_prediction',
                            'reload_and_predict')],
            'predictions': {
                'before': predictions(probabilities),
                'repeat': predictions(repeat or probabilities),
                'after': predictions(reload or probabilities),
            },
        }

    def test_reload_and_retraining_differences_are_reported_separately(self):
        runs = [
            self.measured_run([0.25, 0.75], repeat=[0.25, 0.751],
                              reload=[0.25, 0.752]),
            self.measured_run([0.5, 0.5], reload=[0.501, 0.5]),
        ]
        report = self.runner.summarize_variation(runs, 'cuda', 2)
        self.assertEqual(report['status'], 'MEASURED')
        self.assertAlmostEqual(
            report['repeat_prediction']['max_abs_difference'], 0.001)
        self.assertAlmostEqual(report['reload']['max_abs_difference'], 0.002)
        self.assertAlmostEqual(
            report['between_training_runs']['max_abs_difference'], 0.25)
        self.assertEqual(report['between_training_runs']['model_pairs'], 1)
        self.assertEqual(report['reload']['label_changes'], 1)
        self.assertIsNone(report['approved_gpu_tolerance'])

    def test_measurements_reject_missing_runs_and_cpu_fallback(self):
        runs = [self.measured_run([0.25, 0.75]) for _ in range(2)]
        with self.assertRaisesRegex(ValueError, 'run count'):
            self.runner.summarize_variation(runs[:1], 'cuda', 2)
        runs[1]['observed_devices'][-1]['device'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'device'):
            self.runner.summarize_variation(runs, 'cuda', 2)
        runs[1]['observed_devices'] = []
        with self.assertRaisesRegex(ValueError, 'device'):
            self.runner.summarize_variation(runs, 'cuda', 2)

    def test_measurement_status_is_not_an_approved_smoke_result(self):
        result = dict(status='MEASURED', checks=dict(training='PASS'))
        with self.assertRaises(ValueError):
            self.runner.verify_workflow_result(result, ['training'])
        self.runner.verify_workflow_result(result, ['training'],
                                           status='MEASURED')

    def test_measurements_use_the_configured_label_threshold(self):
        runs = [
            self.measured_run([0.6, 0.8], threshold=0.7),
            self.measured_run([0.65, 0.8], threshold=0.7),
        ]
        report = self.runner.summarize_variation(runs, 'cuda', 2)
        self.assertAlmostEqual(
            report['between_training_runs']['max_abs_difference'], 0.05)
        self.assertEqual(report['between_training_runs']['label_changes'], 0)

    def test_measurements_reject_different_training_controls(self):
        for control, value in [('seed', 1), ('threshold', 0.6)]:
            with self.subTest(control=control):
                runs = [self.measured_run([0.25, 0.75]) for _ in range(2)]
                runs[1][control] = value
                with self.assertRaisesRegex(ValueError, 'differ'):
                    self.runner.summarize_variation(runs, 'cuda', 2)

    def test_measurements_reject_invalid_or_unmatched_predictions(self):
        for mutation in ('ids', 'nan', 'normalization', 'labels'):
            with self.subTest(mutation=mutation):
                runs = [self.measured_run([0.25, 0.75]) for _ in range(2)]
                prediction = runs[1]['predictions']['after']
                if mutation == 'ids':
                    prediction['entity_ids'] = [0, 2]
                elif mutation == 'nan':
                    prediction['probabilities'][0][0] = math.nan
                elif mutation == 'normalization':
                    prediction['probabilities'][0] = [0.9, 0.9]
                else:
                    prediction['labels'][0] = True
                with self.assertRaises(ValueError):
                    self.runner.summarize_variation(runs, 'cuda', 2)

    def test_measurement_options_require_multiple_runs_before_installation(
            self):
        completed = subprocess.run([
            sys.executable,
            str(RUNNER),
            str(self.wheel()),
            '--measure-variation',
            '--device',
            'cuda',
            '--runs',
            '1',
            '--output-dir',
            str(self.root / 'evidence'),
        ], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('at least two', completed.stderr)
        self.assertFalse((self.root / 'evidence/install.log').exists())

    def test_variation_cannot_silently_skip_learning_or_comparison(self):
        for options in (['--learning'], ['--measure-learning', '--reference'], [
                '--compare-with', 'reference.whl'
        ], ['--report-learning', '--compare-with', 'reference.whl']):
            with self.subTest(options=options):
                output = self.root / 'incompatible'
                completed = subprocess.run([
                    sys.executable,
                    str(RUNNER),
                    str(self.wheel()), '--measure-variation', '--runs', '2',
                    '--output-dir',
                    str(output), *options
                ], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 2)
                self.assertIn('separate from learning', completed.stderr)
                self.assertFalse(output.exists())

    def test_variation_records_effective_device_and_requires_functional_checks(
            self):
        wheel = self.wheel()
        requirements = self.runner.load_requirements(self.runner.REQUIREMENTS)
        requirements['configuration'].update(
            os=sys.platform, machine=self.runner.platform.machine(),
            python=f'{sys.version_info.major}.{sys.version_info.minor}')
        identity = self.runner.wheel_identity(wheel)
        requirements['reference']['wheel'] = identity
        requirements['reference']['dependencies'] = {}
        install = dict(install=[
            dict(
                metadata=dict(name='kumo-engine', version='1.2.3'),
                download_info=dict(archive_info=dict(hashes=dict(
                    sha256=identity['sha256']))))
        ])
        for device, failed_check in (('cpu', None), ('cuda', None),
                                     ('cuda', 'training'),
                                     ('cuda', 'prediction_rows_and_values')):
            with self.subTest(device=device, failed_check=failed_check):
                output = self.root / f'variation-{device}-{failed_check}'
                stages = []

                def commands(command, cwd, log, timeout=60):
                    log.write_text('')
                    stages.append(log.stem)
                    if log.stem == 'install':
                        (output / 'install-report.json').write_text(
                            json.dumps(install))
                    elif log.stem.startswith('run-'):
                        self.assertEqual(command[command.index('--device') + 1],
                                         device)
                        self.assertIn('--measure-variation', command)
                        evidence = Path(command[command.index('--output-dir') +
                                                1])
                        self.assertEqual(cwd.name, evidence.name)
                        result = self.measured_run([0.25, 0.75])
                        result['device'] = device
                        for observed in result['observed_devices']:
                            observed['device'] = device
                        result['checks'] = {
                            name: 'PASS'
                            for name in requirements['required_checks']
                        }
                        result['checks'][
                            'reload_predictions_match'] = 'NOT_EVALUATED'
                        if failed_check:
                            del result['checks'][failed_check]
                        (evidence / 'workflow-result.json').write_text(
                            json.dumps(result))

                with (patch.object(self.runner, 'run_command', commands),
                      patch.object(self.runner, 'load_requirements',
                                   return_value=requirements),
                      patch.object(sys, 'argv', [
                          str(RUNNER),
                          str(wheel), '--reference', '--measure-variation',
                          '--device', device, '--runs', '2', '--output-dir',
                          str(output)
                      ]), patch('sys.stdout', new_callable=io.StringIO),
                      patch('sys.stderr', new_callable=io.StringIO)):
                    code = self.runner.main()
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(code, 1 if failed_check else 0)
                self.assertEqual(report['configuration']['device'], device)
                self.assertEqual(requirements['configuration']['device'], 'cpu')
                self.assertEqual(
                    json.loads((output / 'requirements.json'
                                ).read_text())['configuration']['device'],
                    'cpu')
                self.assertEqual(report['status'],
                                 'FAIL' if failed_check else 'MEASURED')
                self.assertEqual((output / 'variation.json').exists(),
                                 failed_check is None)
                self.assertEqual(stages, [
                    'venv', 'install', 'pip-check', 'packages', 'run-001',
                    *([] if failed_check else ['run-002'])
                ])

    def test_release_workflow_rejects_missing_or_unapproved_learning_results(
            self):
        workflow = RUNNER.parents[
            3] / '.github/workflows/kumo-engine-wheel-smoke.yml'
        steps = workflow.read_text().split('\n      - ')
        guards = [
            step for step in steps if '\n        id: release_gate\n' in step
        ]
        self.assertEqual(len(guards), 1,
                         'A verified result must authorize release')
        script = textwrap.dedent(guards[0].split('        run: |\n', 1)[1])
        passed = dict(
            status='PASS', stage='complete', report_only=False,
            comparison=dict(status='PASS', release_approved=True,
                            quality_status='FAIL',
                            additional_tasks=self.task_comparison()))
        missing_tasks = dict(passed, comparison=dict(passed['comparison']))
        del missing_tasks['comparison']['additional_tasks']
        cases = [
            None, {},
            dict(passed, report_only=True),
            dict(passed, status='RUNNING'),
            dict(passed, status='MEASURED'),
            dict(passed, status='FAIL'),
            dict(passed, stage='comparison'),
            dict(passed, comparison={}),
            dict(passed, comparison=dict(status='FAIL', release_approved=True)),
            dict(passed, comparison=dict(status='PASS',
                                         release_approved=False)),
            missing_tasks,
            dict(passed, comparison=dict(passed['comparison'],
                                         additional_tasks={})),
            dict(
                passed, comparison=dict(
                    passed['comparison'],
                    additional_tasks=self.calibration_comparison())),
            dict(
                passed, comparison=dict(
                    passed['comparison'],
                    additional_tasks=dict(self.task_comparison(),
                                          independent_controls={}))), passed
        ]
        for index, report in enumerate(cases):
            with self.subTest(report=report):
                folder = self.root / f'release-result-{index}'
                evidence = folder / 'wheel-learning-comparison'
                evidence.mkdir(parents=True)
                if report is not None:
                    (evidence / 'summary.json').write_text(json.dumps(report))
                output = folder / 'github-output'
                output.write_text('')
                completed = subprocess.run(
                    ['bash', '-e', '-o', 'pipefail', '-c', script], env={
                        **os.environ, 'RUNNER_TEMP':
                        str(folder),
                        'GITHUB_OUTPUT':
                        str(output),
                        'PATH':
                        str(Path(sys.executable).parent) + os.pathsep +
                        os.environ['PATH']
                    }, capture_output=True, text=True)
                valid = index == len(cases) - 1
                self.assertEqual(completed.returncode == 0, valid,
                                 completed.stderr)
                self.assertEqual(output.read_text(),
                                 'release_ready=true\n' if valid else '')


if __name__ == '__main__':
    unittest.main()
