# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import importlib.util
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from functools import wraps
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_gpu_learning.py')
sys.path.insert(0, str(WORKER.parent))


class WheelGpuTest(unittest.TestCase):
    def worker(self):
        if not WORKER.is_file():
            self.fail('The GPU execution entry point is missing')
        return runpy.run_path(str(WORKER))

    def test_static_staging_hashes_require_native_v4_sidecar_contract(self):
        worker = runpy.run_path(str(WORKER.with_name('wheel_scale_data.py')))
        verify = worker['_control_staging_hashes']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = b'unchanged native sidecar bytes'
            sidecar = root / 'fact_pseudo_time.bin'
            sidecar.write_bytes(payload)
            for name in ('tt.parquet', 'stats.json', 'split_stats.json',
                         'train_table_col_stats.json'):
                (root / name).write_text('{}')
            manifest = dict(
                version=4, train_table=dict(path='tt.parquet'),
                fact_pseudo_time=dict(path='fact_pseudo_time.bin',
                                      format='tt_fact_pseudo_time_v1',
                                      node_type='purchases'),
                checksums=dict(fact_pseudo_time_sha256=hashlib.sha256(
                    payload).hexdigest()))
            spec = dict(
                schema_version='diskgraph.tt_ingest_spec.v3', is_temporal=False,
                fact_pseudo_time=dict(table='purchases',
                                      entity_col='customer_id',
                                      target_col='merchant_id'),
                split_stages=[
                    dict(key=stage) for stage in ('train', 'val', 'test')
                ])
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'tt_ingest_spec.json').write_text(json.dumps(spec))
            original = verify(root, link=True)
            self.assertEqual(original['fact_pseudo_time.bin'],
                             hashlib.sha256(payload).hexdigest())
            with self.assertRaises(ValueError):
                verify(root)
            for fault in ('missing', 'changed', 'format', 'node_type',
                          'temporal', 'escape'):
                with self.subTest(fault=fault):
                    changed, changed_spec = copy.deepcopy(
                        manifest), copy.deepcopy(spec)
                    if fault == 'missing':
                        sidecar.unlink()
                    elif fault == 'changed':
                        sidecar.write_bytes(payload + b'changed')
                    elif fault == 'temporal':
                        changed_spec['is_temporal'] = True
                    elif fault == 'escape':
                        changed['fact_pseudo_time'][
                            'path'] = '../fact_pseudo_time.bin'
                    else:
                        changed['fact_pseudo_time'][fault] = 'wrong'
                    (root / 'manifest.json').write_text(json.dumps(changed))
                    (root / 'tt_ingest_spec.json').write_text(
                        json.dumps(changed_spec))
                    with self.assertRaises((ValueError, OSError)):
                        verify(root, link=True)
                    sidecar.write_bytes(payload)
            manifest = dict(version=3, train_table=dict(path='tt.parquet'))
            spec.pop('fact_pseudo_time')
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'tt_ingest_spec.json').write_text(json.dumps(spec))
            self.assertIn('train_table_col_stats.json', verify(root))

    @unittest.skipUnless(importlib.util.find_spec('torch'),
                         'Requires the checkpoint runtime')
    def test_reloaded_weights_and_checkpoint_match_exactly(self):
        import pandas as pd
        import torch

        worker = self.worker()
        self.assertIn('verify_loaded_checkpoint', worker)

        class Model(torch.nn.Linear):
            def get_extra_state(self):
                return dict(index_map=self.index_map, is_fitted=True, seq_len=2)

        model = Model(2, 1)
        model.index_map = pd.DataFrame({'arange': [0, 1]}, index=[10, 20])
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 2.]]))
            model.bias.fill_(.5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.ckpt'
            torch.save({'state_dict': model.state_dict()}, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            verify = lambda: worker['verify_loaded_checkpoint'](
                model, str(checkpoint), digest)
            receipt = verify()
            self.assertEqual(
                receipt,
                dict(status='PASS', checkpoint_sha256=digest,
                     checkpoint_path=str(checkpoint), tensor_count=2))
            with torch.no_grad():
                model.weight.add_(1)
            with self.assertRaisesRegex(ValueError, 'tensor'):
                verify()
            with torch.no_grad():
                model.weight.sub_(1)
            model.double()
            with self.assertRaisesRegex(ValueError, 'tensor'):
                verify()
            model.float()
            original_weight = model.weight
            model.weight = torch.nn.Parameter(torch.ones(1, 3))
            with self.assertRaisesRegex(ValueError, 'tensor'):
                verify()
            model.weight = original_weight
            model.register_buffer('unexpected', torch.ones(1))
            with self.assertRaisesRegex(ValueError, 'keys'):
                verify()
            del model.unexpected
            model.index_map['arange'] = [1, 0]
            with self.assertRaisesRegex(ValueError, 'extra state'):
                verify()
            model.index_map['arange'] = [0, 1]
            checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                verify()

    def test_reload_identity_requires_matching_positive_receipts(self):
        worker = self.worker()
        self.assertIn('verify_reload_identity', worker)
        row = dict(
            expected_reload_checkpoint_sha256='a' * 64, reload_identity_checks=[
                dict(status='PASS', checkpoint_sha256='a' * 64,
                     checkpoint_path='/actual/checkpoint',
                     restore_path='lightning_strategy.load_model_state_dict',
                     tensor_count=2)
            ])
        worker['verify_reload_identity'](row)
        for mutation in ('missing', 'changed', 'empty', 'failed', 'not_hash',
                         'restore_path', 'checkpoint_path'):
            altered = copy.deepcopy(row)
            if mutation == 'missing':
                del altered['reload_identity_checks']
            elif mutation == 'changed':
                altered['reload_identity_checks'][0][
                    'checkpoint_sha256'] = 'b' * 64
            elif mutation == 'empty':
                altered['reload_identity_checks'][0]['tensor_count'] = 0
            elif mutation == 'failed':
                altered['reload_identity_checks'][0]['status'] = 'FAIL'
            elif mutation in ('restore_path', 'checkpoint_path'):
                del altered['reload_identity_checks'][0][mutation]
            else:
                altered['expected_reload_checkpoint_sha256'] = 'missing'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['verify_reload_identity'](altered)

    @unittest.skipUnless(importlib.util.find_spec('torch'),
                         'Requires the checkpoint runtime')
    def test_reload_observer_checks_actual_lightning_restore(self):
        import torch
        from kumo_core.train.trainer import Trainer as CoreTrainer
        from pytorch_lightning import LightningModule
        from pytorch_lightning.strategies import SingleDeviceStrategy

        import kumo.engine as kumo

        worker = self.worker()
        with tempfile.TemporaryDirectory() as directory:
            model = LightningModule()
            model.layer = torch.nn.Linear(2, 1)
            checkpoint = Path(directory) / 'model.ckpt'
            state = dict(state_dict=model.state_dict())
            torch.save(state, checkpoint)
            strategy = SingleDeviceStrategy(device=torch.device('cuda:0'))
            strategy.connect(model)
            core = SimpleNamespace(pl_trainer=SimpleNamespace(
                strategy=strategy, ckpt_path=str(checkpoint)))

            class PublicTrainer:
                def __init__(self):
                    pass

                def fit(self):
                    raise AssertionError('A reload check must not fit')

                def predict(self):
                    CoreTrainer.create_pl_trainer(core)
                    if fault != 'missing_restore':
                        strategy.load_model_state_dict(state)

                @staticmethod
                def load():
                    return PublicTrainer()

            sys.path.insert(0, str(WORKER.parent))
            try:
                for fault in ('none', 'changed', 'missing_restore',
                              'missing_path'):
                    core.pl_trainer.ckpt_path = None if fault == 'missing_path' else str(
                        checkpoint)
                    restore = strategy.load_model_state_dict

                    def load_state(*args, **kwargs):
                        restore(*args, **kwargs)
                        if fault == 'changed':
                            with torch.no_grad():
                                model.layer.weight.add_(1)

                    evidence = dict(expected_reload_checkpoint_sha256=hashlib.
                                    sha256(checkpoint.read_bytes()).hexdigest())
                    with (patch.object(kumo, 'Trainer', PublicTrainer),
                          patch.object(CoreTrainer, 'create_pl_trainer',
                                       lambda _: None),
                          patch.object(strategy, 'load_model_state_dict',
                                       load_state),
                          patch.object(torch.cuda, 'is_available',
                                       return_value=True),
                          patch.object(torch.cuda, 'device_count',
                                       return_value=1),
                          worker['observe_gpu'](evidence)):
                        if fault == 'changed':
                            with self.assertRaisesRegex(ValueError, 'tensor'):
                                kumo.Trainer.load().predict()
                        elif fault == 'missing_path':
                            with self.assertRaisesRegex(ValueError,
                                                        'checkpoint path'):
                                kumo.Trainer.load().predict()
                        elif fault == 'missing_restore':
                            kumo.Trainer.load().predict()
                            with self.assertRaisesRegex(ValueError, 'identity'):
                                worker['verify_reload_identity'](evidence)
                        else:
                            kumo.Trainer.load().predict()
                            worker['verify_reload_identity'](evidence)
            finally:
                sys.path.remove(str(WORKER.parent))

    def test_prepared_profiles_require_data_and_source_together(self):
        for option in ('--scale-bench-data', '--scale-bench-source'):
            with self.subTest(
                    option=option), tempfile.TemporaryDirectory() as output:
                result = subprocess.run([
                    sys.executable,
                    str(WORKER), '--version', '1.0.0', '--group', 'binary',
                    '--device', 'cuda', '--output-dir', output, option, output
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('data and source must be provided together',
                              result.stderr)

    def test_scale_bench_rung_requires_prepared_profiles(self):
        for rung in ('pr', '100M'):
            with self.subTest(
                    rung=rung), tempfile.TemporaryDirectory() as output:
                result = subprocess.run([
                    sys.executable,
                    str(WORKER), '--version', '1.0.0', '--group', 'binary',
                    '--device', 'cuda', '--output-dir', output,
                    '--scale-bench-rung', rung
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('rung requires prepared profiles', result.stderr)

    def test_prepared_profiles_default_to_pr_rung_before_runtime_setup(self):
        worker = self.worker()
        parser = worker['argparse'].ArgumentParser
        original = parser.parse_args
        captured = []

        def parse(instance, *args, **kwargs):
            result = original(instance, *args, **kwargs)
            captured.append(result)
            return result

        with tempfile.TemporaryDirectory() as output, patch.object(
                parser, 'parse_args',
                parse), patch.object(sys, 'argv', [
                    str(WORKER), '--version', '1.0.0', '--group', 'all',
                    '--device', 'cuda', '--output-dir', output,
                    '--scale-bench-data', output, '--scale-bench-source', output
                ]):
            with self.assertRaises(SystemExit) as result:
                worker['main']()
        self.assertEqual(result.exception.code, 2)
        self.assertEqual(captured[0].scale_bench_rung, 'pr')

    def test_prepared_profiles_select_the_rung_and_keep_all_seed_pairs(self):
        worker = self.worker()
        cases = runpy.run_path(str(
            WORKER.with_name('wheel_scale_data.py')))['CASES']
        for rung, seed_profile, ranked in product(
            ('pr', '100M'), ('pr', 'full'), (False, True)):
            with self.subTest(
                    rung=rung, seed_profile=seed_profile,
                    ranked=ranked), tempfile.TemporaryDirectory() as output:
                prepare = Mock(
                    side_effect=lambda profile, source, case, out, **kwargs:
                    dict(case=case, identity=dict(case=case)))
                run = Mock(side_effect=lambda prepared, case, seed, independent,
                           out, **kwargs: dict(
                               task=case, seed=seed, case='independent'
                               if independent else 'learnable'))
                modules = {
                    'torch':
                    SimpleNamespace(cuda=SimpleNamespace(
                        empty_cache=lambda: None)),
                    'wheel_learning':
                    SimpleNamespace(chance_sanity=lambda *args: {}),
                    'wheel_scale_data':
                    SimpleNamespace(
                        CASES=cases, prepare_profile=prepare,
                        control_table=lambda prepared, *args: prepared),
                    'wheel_scale_learning':
                    SimpleNamespace(
                        run_case=run,
                        validate_profile_learning=lambda *args, **kwargs: {}),
                    'wheel_task_checks':
                    SimpleNamespace(independent_controls=lambda *args: {})
                }
                function = worker['run_prepared_group']
                args = SimpleNamespace(
                    group='all', scale_bench_data=Path('/data'),
                    scale_bench_source=Path('/source'), scale_bench_rung=rung,
                    calibration_budget=None, seed_profile=seed_profile,
                    control_fixtures=None, ranked_links=ranked)
                report = dict(runs=[], configuration={})
                with patch.dict(sys.modules, modules), patch.dict(
                        function.__globals__,
                        verify_prepared_report=lambda *args, **kwargs: None,
                        gc=SimpleNamespace(collect=lambda: None)):
                    function(args, Path(output), report)
                self.assertEqual(report['configuration']['scale_bench_rung'],
                                 rung)
                seeds = [0] if seed_profile == 'pr' else list(range(5))
                self.assertEqual(report['seeds'], seeds)
                self.assertEqual(report['configuration']['seed_profile'],
                                 seed_profile)
                for call in prepare.call_args_list:
                    self.assertEqual(
                        call.args[0],
                        Path('/data') / cases[call.args[2]]['graph'] / rung)
                    self.assertEqual(call.kwargs.get('ranked_links', False),
                                     ranked and call.args[2].endswith('link'))
                conditions = ('learnable', ) if seed_profile == 'pr' else (
                    'learnable', 'independent')
                self.assertEqual(run.call_count,
                                 8 * len(seeds) * len(conditions))
                self.assertEqual(
                    {(r['task'], r['seed'], r['case'])
                     for r in report['runs']},
                    set(product(cases, seeds, conditions)))
                self.assertEqual('profile_learning' in report,
                                 seed_profile == 'full')
                self.assertEqual('binary_control' in report,
                                 seed_profile == 'full')
                self.assertEqual('task_controls' in report,
                                 seed_profile == 'full')

    def test_all_tasks_require_prepared_profiles(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run([
                sys.executable,
                str(WORKER), '--version', '1.0.0', '--group', 'all', '--device',
                'cuda', '--output-dir', output
            ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('all requires prepared profiles', result.stderr)

    def test_prepared_profiles_require_frozen_dependencies(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run([
                sys.executable,
                str(WORKER), '--version', '1.0.0', '--group', 'all', '--device',
                'cuda', '--output-dir', output, '--scale-bench-data', output,
                '--scale-bench-source', output
            ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('prepared profiles require frozen dependencies',
                      result.stderr)

    def test_dependency_profile_rejects_unexpected_versions_or_packages(self):
        worker = self.worker()
        self.assertIn('verify_dependencies', worker)
        packages = {'torch': '2.14.1', 'pytorch-lightning': '2.6.5'}
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'dependencies.json'
            profile.write_text(json.dumps(packages))
            worker['verify_dependencies'](packages, profile)
            for changed in ({
                    **packages, 'pytorch-lightning': '2.6.6'
            }, {
                    'torch': '2.14.1'
            }, {
                    **packages, 'unexpected': '1.0'
            }):
                with self.subTest(
                        packages=changed), self.assertRaises(ValueError):
                    worker['verify_dependencies'](changed, profile)
            profile.unlink()
            with self.assertRaises(FileNotFoundError):
                worker['verify_dependencies'](packages, profile)

    def test_control_fixtures_require_prepared_profiles(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run([
                sys.executable,
                str(WORKER), '--version', '1.0.0', '--group', 'binary',
                '--device', 'cuda', '--output-dir', output,
                '--control-fixtures', output
            ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('control fixtures require prepared profiles',
                      result.stderr)

    def test_calibration_budget_requires_prepared_profiles(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run([
                sys.executable,
                str(WORKER), '--version', '1.0.0', '--group', 'binary',
                '--device', 'cuda', '--output-dir', output,
                '--calibration-budget', '1', '1'
            ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('calibration requires prepared profiles', result.stderr)

    def test_fixed_batches_propagate_oom_without_a_smaller_retry(self):
        worker = self.worker()
        self.assertIn('without_batch_retries', worker)
        attempts = []

        def retry(function):
            @wraps(function)
            def wrapped(self):
                try:
                    return function(self)
                except RuntimeError:
                    return 'smaller retry'

            return wrapped

        class Core:
            @retry
            def test(self):
                attempts.append('test')
                raise RuntimeError('out of memory')

            predict_v2 = test

        class Auto:
            def _handle_oom_retry(self):
                return 'smaller retry'

        with worker['without_batch_retries'](Core, Auto):
            for name in ('test', 'predict_v2'):
                with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                    getattr(Core(), name)()
            with self.assertRaisesRegex(ValueError, 'batch retry'):
                Auto()._handle_oom_retry()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(Core().test(), 'smaller retry')
        self.assertEqual(Auto()._handle_oom_retry(), 'smaller retry')

    def test_prepared_report_requires_all_tasks_seeds_controls_and_fresh_models(
            self):
        worker = self.worker()
        self.assertIn('verify_prepared_report', worker)
        tasks = runpy.run_path(str(
            WORKER.with_name('wheel_scale_data.py')))['CASES']
        rows = [
            dict(
                task=task, seed=seed, case=case, fit_id=f'{task}-{seed}-{case}',
                status='MEASURED', training_seed=seed,
                selected_checkpoint_global_step=2, batch_retries_disabled=True,
                expected_reload_checkpoint_sha256='d' * 64,
                reload_identity_checks=[
                    dict(
                        status='PASS', checkpoint_sha256='d' * 64,
                        checkpoint_path='/actual/checkpoint',
                        restore_path='lightning_strategy.load_model_state_dict',
                        tensor_count=2)
                ], training_work=[
                    dict(status='COMPLETE', completed_batches=1,
                         completed_examples=64, optimizer_steps=2,
                         early_stopped=False)
                ], training_window=dict(completed_batches=1,
                                        completed_examples=64,
                                        optimizer_steps=2, batch_examples=[64]),
                checks={
                    name: 'PASS'
                    for name in ('fit', 'prediction', 'repeat_prediction',
                                 'partial_prediction', 'reload',
                                 'hidden_answers', 'chronology')
                }, device='cuda', observed_devices=[
                    dict(stage=stage, device='cuda:0')
                    for stage in ('training', 'prediction',
                                  'reload_and_predict')
                ], consistency_measurements=[
                    dict(stage=stage, scope=scope, status='MEASURED',
                         qualified=False, gpu_tolerance=None,
                         max_abs_difference=0.0)
                    for stage, scope in (('prediction', 'repeat_full'),
                                         ('prediction', 'partial'),
                                         ('reload_and_predict', 'reload'))
                ], repeat_max_abs_difference=0.0)
            for task, seed, case in product(tasks, range(5), ('learnable',
                                                              'independent'))
        ]
        import wheel_scale_learning as learning

        for row in rows:
            task = row['task']
            targets = ([False, True] if task == 'binary' else
                       [[0.0] * 12, [1.0] * 12] if task == 'forecasting' else
                       [[1, 0], [0, 1]] if task == 'multiclass' else
                       [[0.0], [1.0]] if task == 'regression' else [[1, 0] *
                                                                    6, [0, 1] *
                                                                    6])
            predictions = [[.1], [.9]] if task == 'binary' else targets
            row.update(
                metric=tasks[task]['metric'],
                task_type=tasks[task]['task_type'], entity_ids=[10, 20],
                targets=targets, predictions=predictions,
                baseline_predictions=predictions,
                score=learning.task_score(task, targets, predictions),
                baseline_score=learning.task_score(task, targets, predictions),
                data_sha256='a' * 64, plan_sha256='b' * 64, stage_sha256={
                    stage: 'c' * 64
                    for stage in ('train', 'val', 'test')
                })
        report = dict(
            status='MEASURED', group='all', seeds=list(range(5)),
            configuration={
                'device': 'cuda',
                'seed_profile': 'full'
            }, gpu_source_sha256=worker['source_identity'](prepared=True),
            runs=rows)
        worker['verify_prepared_report'](report)
        pr = copy.deepcopy(report)
        pr['configuration']['seed_profile'] = 'pr'
        pr['seeds'] = [0]
        pr['runs'] = [
            row for row in pr['runs']
            if row['seed'] == 0 and row['case'] == 'learnable'
        ]
        worker['verify_prepared_report'](pr)
        import numpy as np

        ranked = copy.deepcopy(pr)
        ranked['configuration']['ranked_links'] = True
        for row in ranked['runs']:
            if not row['task'].endswith('link'):
                continue
            targets = row.pop('targets')
            k = int(row['metric'].split('@')[1])
            row.update(
                target_format='ranked_link_v1',
                catalog=list(range(len(targets[0]))), target_ids=[
                    np.flatnonzero(values).tolist() for values in targets
                ])
            if row['task'] == 'static_link':
                row['excluded_ids'] = [
                    np.flatnonzero(~np.asarray(mask)).tolist()
                    for mask in row.pop('control_eligible', [[True] *
                                                             len(targets[0])
                                                             for _ in targets])
                ]
            for old, prefix in (('predictions', 'ranked'),
                                ('baseline_predictions', 'baseline_ranked')):
                values = np.asarray(row.pop(old), dtype=float)
                order = np.argsort(-values, axis=1, kind='stable')[:, :k]
                row[f'{prefix}_ids'] = order.tolist()
                row[f'{prefix}_scores'] = np.take_along_axis(
                    values, order, axis=1).tolist()
            for item in row['consistency_measurements']:
                population = (max(1,
                                  len(row['entity_ids']) // 2) if item['scope']
                              == 'partial' else len(row['entity_ids']))
                item.update(score_coordinates='entity_class',
                            common_recommendations=k * population,
                            added_recommendations=0, removed_recommendations=0,
                            changed_entities=0)
        worker['verify_prepared_report'](ranked)
        for mutation in ('flag', 'score', 'ids', 'coordinates'):
            changed = copy.deepcopy(ranked)
            link = next(row for row in changed['runs']
                        if row['task'] == 'static_link')
            if mutation == 'flag':
                del changed['configuration']['ranked_links']
            elif mutation == 'score':
                link['score'] += .1
            elif mutation == 'ids':
                link['ranked_ids'][0][0] = -1
            else:
                del link['consistency_measurements'][0][
                    'common_recommendations']
            with self.subTest(ranked=mutation), self.assertRaises(ValueError):
                worker['verify_prepared_report'](changed)
        pr['configuration']['seed_profile'] = 'full'
        with self.assertRaises(ValueError):
            worker['verify_prepared_report'](pr)
        for mutation in ('missing', 'duplicate', 'reused', 'failed',
                         'old_source', 'score', 'baseline', 'metric', 'hash',
                         'retry', 'untrained', 'fault', 'no_repeat',
                         'repeat_scope', 'repeat_check', 'repeat_delta',
                         'reload_identity'):
            changed = copy.deepcopy(report)
            if mutation == 'missing':
                changed['runs'].pop()
            elif mutation == 'duplicate':
                changed['runs'][-1] = changed['runs'][0]
            elif mutation == 'reused':
                changed['runs'][-1]['fit_id'] = changed['runs'][0]['fit_id']
            elif mutation == 'failed':
                changed['runs'][-1]['checks']['reload'] = 'FAIL'
            elif mutation == 'old_source':
                changed['gpu_source_sha256'] = worker['source_identity']()
            elif mutation == 'score':
                changed['runs'][-1]['score'] += .1
            elif mutation == 'baseline':
                changed['runs'][-1]['baseline_score'] += .1
            elif mutation == 'metric':
                changed['runs'][-1]['metric'] = 'mae'
            elif mutation == 'retry':
                changed['runs'][-1]['batch_retries_disabled'] = False
            elif mutation == 'untrained':
                changed['runs'][-1]['selected_checkpoint_global_step'] = 0
            elif mutation == 'fault':
                changed['runs'][-1]['calibration_fault'] = {}
            elif mutation == 'no_repeat':
                changed['runs'][-1]['consistency_measurements'].pop(0)
            elif mutation == 'repeat_scope':
                changed['runs'][-1]['consistency_measurements'][0][
                    'scope'] = 'partial'
            elif mutation == 'repeat_check':
                del changed['runs'][-1]['checks']['repeat_prediction']
            elif mutation == 'reload_identity':
                del changed['runs'][-1]['reload_identity_checks']
            elif mutation == 'repeat_delta':
                changed['runs'][-1]['repeat_max_abs_difference'] = .1
            else:
                changed['runs'][-1]['stage_sha256']['train'] = ''
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['verify_prepared_report'](changed)

        fault = dict(kind='reduced_training_budget', budget=[1, 1])
        calibrated = copy.deepcopy(report)
        calibrated.update(calibration_fault=fault, release_approved=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in calibrated['runs']:
                original = learning.make_plan(row['task'],
                                              row['seed']).as_json()
                selected = copy.deepcopy(original)
                selected['optimization'].update(max_epochs=1,
                                                min_steps_per_epoch=1,
                                                max_steps_per_epoch=1)
                row['calibration_fault'] = dict(fault, original_plan=original)
                row['plan_sha256'] = learning.fingerprint(selected)
                evidence = root / f'{row["task"]}-{row["seed"]}-{row["case"]}'
                evidence.mkdir()
                for name in ('requested-plan', 'selected-model-plan'):
                    (evidence / f'{name}.json').write_text(json.dumps(selected))
            worker['verify_prepared_report'](calibrated, calibration_dir=root)
            with self.assertRaises(ValueError):
                worker['verify_prepared_report'](calibrated)
            for mutation in ('missing', 'budget', 'unchanged', 'increasing',
                             'original', 'unrelated', 'selected', 'hash',
                             'approved', 'control', 'reload', 'cpu', 'step'):
                changed = copy.deepcopy(calibrated)
                row = changed['runs'][0]
                if mutation == 'missing':
                    del row['calibration_fault']
                elif mutation == 'budget':
                    row['calibration_fault']['budget'] = [0, 1]
                elif mutation in ('unchanged', 'increasing'):
                    budget = [10 if mutation == 'unchanged' else 11, 20]
                    changed['calibration_fault']['budget'] = budget
                    for item in changed['runs']:
                        item['calibration_fault']['budget'] = budget
                elif mutation == 'original':
                    row['calibration_fault']['original_plan']['optimization'][
                        'base_lr'] = [.123]
                elif mutation == 'unrelated':
                    path = root / 'binary-0-learnable/requested-plan.json'
                    value = json.loads(path.read_text())
                    value['optimization']['base_lr'] = [.123]
                    path.write_text(json.dumps(value))
                elif mutation == 'hash':
                    row['plan_sha256'] = 'e' * 64
                elif mutation == 'selected':
                    path = root / 'binary-0-learnable/selected-model-plan.json'
                    value = json.loads(path.read_text())
                    value['optimization']['base_lr'] = [.123]
                    path.write_text(json.dumps(value))
                    row['plan_sha256'] = learning.fingerprint(value)
                elif mutation == 'approved':
                    changed['release_approved'] = True
                elif mutation == 'control':
                    changed['runs'].pop()
                elif mutation == 'reload':
                    row['checks']['reload'] = 'FAIL'
                elif mutation == 'cpu':
                    row['observed_devices'][0]['device'] = 'cpu'
                else:
                    row['selected_checkpoint_global_step'] = 0
                with self.subTest(
                        mutation=mutation), self.assertRaises(ValueError):
                    worker['verify_prepared_report'](changed,
                                                     calibration_dir=root)
                if mutation in ('unrelated', 'selected'):
                    value['optimization']['base_lr'] = [.01]
                    path.write_text(json.dumps(value))

    def test_gpu_evidence_rejects_fallback_and_missing_execution_phases(self):
        verify = self.worker()['verify_devices']
        result = dict(
            device='cuda', observed_devices=[
                dict(stage=stage, device='cuda:0')
                for stage in ('training', 'prediction', 'reload_and_predict')
            ])
        verify(result, ('training', 'prediction', 'reload_and_predict'))
        for mutation in ('cpu', 'missing', 'empty', 'wrong_gpu', 'requested'):
            changed = copy.deepcopy(result)
            if mutation == 'cpu':
                changed['observed_devices'][1]['device'] = 'cpu'
            elif mutation == 'missing':
                changed['observed_devices'].pop()
            elif mutation == 'empty':
                changed['observed_devices'] = []
            elif mutation == 'wrong_gpu':
                changed['observed_devices'][1]['device'] = 'cuda:1'
            else:
                changed['device'] = 'cpu'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(changed,
                       ('training', 'prediction', 'reload_and_predict'))

    def test_gpu_consistency_measures_variation_without_inventing_a_limit(self):
        worker = self.worker()
        self.assertIn('measure_consistency', worker)
        result = worker['measure_consistency']([[0.25, 0.75]],
                                               [[0.250005, 0.749995]])
        self.assertEqual(result['status'], 'MEASURED')
        self.assertAlmostEqual(result['max_abs_difference'], 0.000005)
        self.assertTrue(result['cpu_tolerance_exceeded'])
        self.assertIsNone(result['gpu_tolerance'])
        self.assertFalse(result['qualified'])
        for changed in ([[0.25]], [[float('nan'), 0.75]], [], [0.25, 0.75]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                worker['measure_consistency']([[0.25, 0.75]], changed)

    def test_gpu_reports_require_matched_runtime_and_fresh_guard_source(self):
        worker = self.worker()
        report = dict(
            configuration=dict(device='cuda', gpu='T4', cuda='13.0'),
            gpu_source_sha256=worker['source_identity'](), runs=[
                dict(device='cuda',
                     observed_devices=[dict(stage='training', device='cuda:0')])
            ])
        worker['verify_report'](report, additional=False)
        for mutation in ('source', 'missing', 'cpu'):
            changed = copy.deepcopy(report)
            if mutation == 'source':
                changed['gpu_source_sha256'] = 'a' * 64
            elif mutation == 'missing':
                changed['runs'][0]['observed_devices'] = []
            else:
                changed['configuration']['device'] = 'cpu'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['verify_report'](changed, additional=False)

    def test_additional_gpu_reports_reject_unqualified_or_nonfinite_consistency(
            self):
        worker = self.worker()
        row = dict(
            device='cuda', observed_devices=[
                dict(stage=stage, device='cuda:0')
                for stage in ('training', 'prediction', 'reload_and_predict')
            ], consistency_measurements=[
                dict(stage=stage, status='MEASURED', qualified=False,
                     gpu_tolerance=None, max_abs_difference=0.0)
                for stage in ('prediction', 'reload_and_predict')
            ])
        report = dict(configuration={'device': 'cuda'},
                      gpu_source_sha256=worker['source_identity'](), runs=[row])
        worker['verify_report'](report, additional=True)
        for mutation in ('missing', 'nan', 'approved', 'extra'):
            changed = copy.deepcopy(report)
            measured = changed['runs'][0]['consistency_measurements']
            if mutation == 'missing':
                measured.pop()
            elif mutation == 'nan':
                measured[0]['max_abs_difference'] = float('nan')
            elif mutation == 'extra':
                measured.append(copy.deepcopy(measured[0]))
            else:
                measured[0]['qualified'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['verify_report'](changed, additional=True)


if __name__ == '__main__':
    unittest.main()
