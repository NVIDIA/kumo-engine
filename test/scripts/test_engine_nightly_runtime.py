# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate nightly contracts. Metadata fixtures do not prove GPU execution."""

import copy
import fcntl
import hashlib
import inspect
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MODULES = ROOT / 'packages/kumo-engine/test'
sys.path.insert(0, str(MODULES))
SCRIPT = ROOT / 'packages/kumo-engine/scripts/nightly_check.py'


def worker():
    return runpy.run_path(str(MODULES / 'wheel_gpu_learning.py'))


def metadata_row(task, case):
    import wheel_scale_data as data
    import wheel_scale_learning as learning
    targets = ([False, True] if task == 'binary' else [[0.] * 12, [1.] *
                                                       12] if task
               == 'forecasting' else [[1, 0], [0, 1]] if task == 'multiclass'
               else [[0.], [1.]] if task == 'regression' else [[1, 0] *
                                                               6, [0, 1] * 6])
    predictions = [[.1], [.9]] if task == 'binary' else targets
    return dict(
        task=task, seed=0, training_seed=0, case=case,
        fit_id=f'{task}-0-{case}', status='MEASURED',
        selected_checkpoint_global_step=1, batch_retries_disabled=True,
        expected_reload_checkpoint_sha256='d' * 64, reload_identity_checks=[
            dict(status='PASS', checkpoint_sha256='d' * 64,
                 checkpoint_path='/metadata-contract-only/checkpoint',
                 tensor_count=2,
                 restore_path='lightning_strategy.load_model_state_dict')
        ], training_work=[
            dict(status='COMPLETE', completed_batches=1, completed_examples=64,
                 optimizer_steps=1, early_stopped=False)
        ], training_window=dict(completed_batches=1, completed_examples=64,
                                optimizer_steps=1, batch_examples=[64]),
        checks=dict.fromkeys(
            ('fit', 'prediction', 'repeat_prediction', 'partial_prediction',
             'reload', 'hidden_answers', 'chronology'), 'PASS'), device='cuda',
        observed_devices=[
            dict(stage=s, device='cuda:0')
            for s in ('training', 'prediction', 'reload_and_predict')
        ], consistency_measurements=[
            dict(stage=s, scope=c, status='MEASURED', qualified=False,
                 gpu_tolerance=None, max_abs_difference=0.)
            for s, c in [('prediction', 'repeat_full'), (
                'prediction', 'partial'), ('reload_and_predict', 'reload')]
        ], repeat_max_abs_difference=0., metric=data.CASES[task]['metric'],
        task_type=data.CASES[task]['task_type'], entity_ids=[10, 20
                                                             ], targets=targets,
        predictions=predictions, baseline_predictions=predictions,
        score=learning.task_score(task, targets, predictions),
        baseline_score=learning.task_score(task, targets, predictions),
        data_sha256='a' * 64, plan_sha256='b' * 64, stage_sha256=dict.fromkeys(
            ('train', 'val', 'test'), 'c' * 64))


class NightlyRuntimeTests(unittest.TestCase):
    def function(self, name):
        result = runpy.run_path(str(SCRIPT)).get(name)
        self.assertTrue(callable(result), 'Missing nightly contract: ' + name)
        return result

    def test_default_and_explicit_resume_rosters_are_exact(self):
        select = self.function('case_roster')
        tasks = tuple(runpy.run_path(str(SCRIPT))['TASKS'])
        full = select(tasks, ['normal', 'scrambled'])
        self.assertEqual(len(full), 16)
        self.assertEqual({seed for _, seed, _ in full}, {0})
        self.assertEqual(select(['multiclass'], ['scrambled']),
                         [('multiclass', 0, 'independent')])
        for t, c in [([], ['normal']), (['unknown'], ['normal']),
                     (['binary', 'binary'], ['normal']), (['binary'], []),
                     (['binary'], ['normal', 'normal']),
                     (['binary'], ['weak'])]:
            with self.subTest(tasks=t,
                              conditions=c), self.assertRaises(ValueError):
                select(t, c)

    def test_declared_wheel_fingerprint_is_checked_before_install_validation(
            self):
        verify = self.function('verify_archive')
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wheel.whl'
            path.write_bytes(b'exact archive')
            correct = hashlib.sha256(path.read_bytes()).hexdigest()
            verify(path, correct)
            for expected in ('wrong', '0' * 64):
                with self.assertRaises(ValueError):
                    verify(path, expected)
            path.write_bytes(b'substituted archive')
            with self.assertRaises(ValueError):
                verify(path, correct)

    def test_explicit_nightly_roster_keeps_full_row_validation(self):
        checked = worker()
        verify = checked['verify_prepared_report']
        self.assertIn('expected_runs', inspect.signature(verify).parameters)
        expected = [('multiclass', 0, 'learnable'),
                    ('multiclass', 0, 'independent')]
        report = dict(
            status='MEASURED', group='all', seeds=[0],
            configuration=dict(device='cuda', seed_profile='nightly'),
            gpu_source_sha256=checked['source_identity'](prepared=True),
            runs=[metadata_row(task, case) for task, _, case in expected])
        verify(report, expected_runs=expected)
        for change in ('missing', 'duplicate', 'extra', 'cpu', 'reload',
                       'score', 'fault', 'training_seed'):
            r = copy.deepcopy(report)
            if change == 'missing':
                r['runs'].pop()
            elif change == 'duplicate':
                r['runs'][-1] = r['runs'][0]
            elif change == 'extra':
                r['runs'].append(metadata_row('binary', 'learnable'))
            elif change == 'cpu':
                r['runs'][0]['observed_devices'][0]['device'] = 'cpu'
            elif change == 'reload':
                r['runs'][0].pop('reload_identity_checks')
            elif change == 'score':
                r['runs'][0]['score'] += .1
            elif change == 'training_seed':
                r['runs'][0]['training_seed'] = False
            else:
                r['runs'][0]['calibration_fault'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify(r, expected_runs=expected)
        for bad in ([], [expected[0], expected[0]], [
            ('unknown', 0, 'learnable')
        ], [('binary', 1, 'learnable')], [('binary', False, 'learnable')],
                    [('binary', 0, 'weak')]):
            with self.subTest(expected=bad), self.assertRaises(ValueError):
                verify(report, expected_runs=bad)
        with self.assertRaises(ValueError):
            verify(report)

    def test_all_sixteen_nightly_cases_pass_without_a_quality_floor(self):
        checked = worker()
        verify = checked['verify_prepared_report']
        self.assertIn('expected_runs', inspect.signature(verify).parameters)
        tasks = runpy.run_path(str(MODULES / 'wheel_scale_data.py'))['CASES']
        expected = [(task, 0, case) for task in tasks
                    for case in ('learnable', 'independent')]
        rows = [metadata_row(task, case) for task, _, case in expected]
        report = dict(
            status='MEASURED', group='all', seeds=[0],
            configuration=dict(device='cuda', seed_profile='nightly'),
            gpu_source_sha256=checked['source_identity'](prepared=True),
            runs=rows)
        # Equality to the baseline is accepted. This validates metadata/scoring only.
        verify(report, expected_runs=expected)
        import wheel_scale_learning as learning
        row = next(row for row in rows if row['task'] == 'multiclass')
        row['predictions'] = list(reversed(row['predictions']))
        row['score'] = learning.task_score('multiclass', row['targets'],
                                           row['predictions'])
        self.assertLess(row['score'], row['baseline_score'])
        verify(report, expected_runs=expected)

    def test_fixture_preflight_rejects_wrong_or_missing_scrambled_arm(self):
        inspect_sources = self.function('fixture_sources')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, seed, kind in [('normal', None, 'normal'),
                                    ('seed-0', 0, 'control')]:
                path = root / 'multiclass' / arm
                path.mkdir(parents=True)
                (path / 'descriptor.json').write_text(
                    json.dumps(
                        dict(
                            case='multiclass', seed=seed, arm=kind,
                            identity=dict(case='multiclass',
                                          control_seed=seed))))
            sources, ranked = inspect_sources(root, ['multiclass'],
                                              ['scrambled'])
            self.assertFalse(ranked)
            self.assertEqual(set(sources['multiclass']),
                             {'normal', 'scrambled'})
            path = root / 'multiclass/seed-0/descriptor.json'
            original = json.loads(path.read_text())
            for field, value in [('seed', False), ('seed', 1),
                                 ('case', 'binary'), ('arm', 'normal')]:
                path.write_text(json.dumps(dict(original, **{field: value})))
                with self.subTest(field=field,
                                  value=value), self.assertRaises(ValueError):
                    inspect_sources(root, ['multiclass'], ['scrambled'])
            wrong = json.loads(json.dumps(original))
            wrong['identity']['control_seed'] = False
            path.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):
                inspect_sources(root, ['multiclass'], ['scrambled'])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                inspect_sources(root, ['multiclass'], ['scrambled'])

    def test_setup_failure_preserves_requested_roster_without_fitting(self):
        run = self.function('run')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'candidate.whl'
            wheel.write_bytes(b'wrong archive')
            args = SimpleNamespace(tasks=['multiclass'
                                          ], conditions=['scrambled'],
                                   output=root / 'result', version='1.0.0',
                                   wheel=wheel, wheel_sha256='0' * 64,
                                   expected_prefix=root / 'environment',
                                   host_lock=root / 'gpu-study.lock')
            with self.assertRaises(ValueError):
                run(args)
            report = json.loads(
                (args.output / 'nightly-result.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['physical_fits'], 0)
            self.assertEqual(report['completed_runs'], [])
            self.assertEqual(report['expected_runs'],
                             [['multiclass', 0, 'independent']])
            self.assertEqual(
                json.loads((args.output / 'fits.json').read_text()), [])
            with args.host_lock.open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_busy_shared_host_lock_blocks_before_any_gpu_or_fit_work(self):
        run = self.function('run')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'candidate.whl'
            wheel.write_bytes(b'archive identity checked after the host lock')
            args = SimpleNamespace(
                tasks=['multiclass'], conditions=['scrambled'],
                output=root / 'result', version='1.0.0', wheel=wheel,
                wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                expected_prefix=root / 'environment',
                host_lock=root / 'gpu-study.lock')
            with args.host_lock.open('a') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'busy'):
                    run(args)
                report = json.loads(
                    (args.output / 'nightly-result.json').read_text())
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['physical_fits'], 0)
                self.assertEqual(
                    json.loads((args.output / 'fits.json').read_text()), [])
                # A failed contender must not release the other owner's lock.
                with args.host_lock.open('a') as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_fixed_rtx_properties_require_actual_identity_and_memory_observations(
            self):
        inspect_gpu = self.function('gpu_properties')
        values = dict(name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                      uuid='GPU-9079f8f0-7852-f15a-d914-58072aeff113',
                      total_memory=96 * 1024**3)
        self.assertEqual(
            inspect_gpu(SimpleNamespace(**values)),
            dict(gpu=values['name'], gpu_uuid=values['uuid'],
                 gpu_memory_bytes=values['total_memory']))
        for key, value in [('name', 'NVIDIA T4'), ('uuid', ''), ('uuid', None),
                           ('uuid', False), ('uuid', object()),
                           ('total_memory', 0)]:
            with self.subTest(field=key,
                              value=value), self.assertRaises(ValueError):
                inspect_gpu(SimpleNamespace(**dict(values, **{key: value})))

    def test_torch_uuid_object_uses_its_canonical_string(self):
        class CudaUuid:
            # torch._C._CUuuid exposes its value through str(), not a str field.
            def __str__(self):
                return '9079f8f0-7852-f15a-d914-58072aeff113'

        properties = SimpleNamespace(
            name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
            uuid=CudaUuid(), total_memory=101975851008)
        self.assertEqual(
            self.function('gpu_properties')(properties)['gpu_uuid'],
            str(properties.uuid))


if __name__ == '__main__':
    unittest.main()
