# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_measurements.py')


class WheelMeasurementsTest(unittest.TestCase):
    def test_training_counts_reject_missing_failed_or_excess_fault_work(self):
        verify = runpy.run_path(str(WORKER)).get('verify_training_counts')
        self.assertTrue(callable(verify),
                        'Completed-work validation is missing')
        row = dict(
            training_work=[
                dict(status='COMPLETE', completed_batches=2,
                     completed_examples=96, optimizer_steps=4,
                     early_stopped=False)
            ], training_window=dict(completed_batches=2, completed_examples=96,
                                    optimizer_steps=4, batch_examples=[64, 32]))
        verify(row, budget=[1, 2])
        for mutation in ('missing', 'failed', 'duplicate', 'examples', 'steps',
                         'partial', 'budget'):
            changed = json.loads(json.dumps(row))
            if mutation == 'missing':
                del changed['training_work']
            elif mutation == 'failed':
                changed['training_work'][0]['status'] = 'FAIL'
            elif mutation == 'duplicate':
                changed['training_work'] *= 2
            elif mutation == 'examples':
                changed['training_window']['batch_examples'][-1] = 31
            elif mutation == 'steps':
                changed['training_work'][0]['optimizer_steps'] = 0
            elif mutation == 'partial':
                changed['training_window']['completed_batches'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(changed,
                       budget=[1, 1] if mutation == 'budget' else [1, 2])

    def test_shared_training_observer_counts_completed_work_and_failures(self):
        observe = runpy.run_path(str(WORKER)).get('observe_training_work')
        self.assertTrue(callable(observe),
                        'Shared training observer is missing')

        class CoreTrainer:
            def create_pl_trainer(self):
                self.pl_trainer = SimpleNamespace(callbacks=[])
                return 'created'

        original = CoreTrainer.create_pl_trainer
        modules = {
            'pytorch_lightning': SimpleNamespace(Callback=object),
            'torch':
            SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
            'kumo_core.train.trainer': SimpleNamespace(Trainer=CoreTrainer),
        }
        for failure in (False, True):
            report = {}
            with self.subTest(failure=failure), patch.dict(
                    sys.modules, modules):
                with observe(report):
                    core = CoreTrainer()
                    self.assertEqual(core.create_pl_trainer(), 'created')
                    [callback] = core.pl_trainer.callbacks
                    trainer = SimpleNamespace(global_step=0, current_epoch=0,
                                              early_stopping_callback=None)
                    module = SimpleNamespace(lhs_name='seed')
                    callback.on_train_start(trainer, module)
                    for i, count in enumerate((64, 32)):
                        trainer.global_step += 2
                        callback.on_train_batch_end(
                            trainer, module, None,
                            {'seed': SimpleNamespace(batch_size=count)}, i)
                    if failure:
                        callback.on_exception(trainer, module,
                                              RuntimeError('test failure'))
                    else:
                        callback.on_train_end(trainer, module)
                self.assertIs(CoreTrainer.create_pl_trainer, original)
            [work] = report['training_work']
            self.assertEqual(work['status'], 'FAIL' if failure else 'COMPLETE')
            self.assertEqual(work['completed_batches'], 2)
            self.assertEqual(work['completed_examples'], 96)
            self.assertEqual(work['optimizer_steps'], 4)
            self.assertEqual(report['training_window']['timed_batches'], 0)
            self.assertEqual(report['training_window']['warmup_batches'], 0)
            self.assertEqual(report['training_window']['timed_examples'], 0)

    def test_timing_control_spends_real_time_inside_the_measured_window(self):
        worker = runpy.run_path(str(WORKER))
        self.assertIn('control_delay_seconds',
                      worker['TrainingWindow'](lambda: None).summary())
        window = worker['TrainingWindow'](lambda: None,
                                          control_delay_seconds=0.02)
        for batch in range(220):
            window.batch_start()
            window.batch_end(optimizer_steps=batch + 1, examples=64)
        result = window.summary()
        self.assertGreaterEqual(result['seconds'], 0.02)
        self.assertGreaterEqual(result['observed_control_delay_seconds'], 0.02)
        self.assertEqual(result['completed_examples'], 14080)
        for delay in (-1, float('nan'), float('inf')):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                worker['TrainingWindow'](lambda: None,
                                         control_delay_seconds=delay)

    def test_memory_control_retains_and_touches_real_host_pages(self):
        worker = runpy.run_path(str(WORKER))
        self.assertIn('allocate_memory_control', worker)
        buffers, record = worker['allocate_memory_control'](cpu_bytes=16385)
        self.assertEqual(len(buffers[0]), 16385)
        self.assertEqual(
            [buffers[0][offset] for offset in range(0, 16385, 4096)],
            [1, 1, 1, 1, 1])
        self.assertEqual(record['cpu_bytes'], 16385)
        self.assertEqual(record['gpu_bytes'], 0)
        for invalid in (-1, 1.5, True):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                worker['allocate_memory_control'](cpu_bytes=invalid)

    def test_training_window_records_real_examples_and_rejects_partial_work(
            self):
        worker = runpy.run_path(str(WORKER))
        window = worker['TrainingWindow'](lambda: None)
        self.assertIn('timed_examples', window.summary())
        for batch in range(220):
            window.batch_start()
            window.batch_end(optimizer_steps=(batch + 1) * 2,
                             examples=14 if batch == 219 else 64)
        measured = window.summary()
        self.assertEqual(measured['warmup_examples'], 1280)
        self.assertEqual(measured['timed_examples'], 12750)
        self.assertEqual(measured['completed_examples'], 14030)
        work = [
            dict(completed_batches=220, optimizer_steps=440,
                 early_stopped=False)
        ]
        with self.assertRaisesRegex(ValueError, 'examples|batch'):
            worker['verify_training_work'](work, measured, expected_batches=220,
                                           expected_batch_size=64)
        complete = worker['TrainingWindow'](lambda: None)
        for batch in range(220):
            complete.batch_start()
            complete.batch_end(optimizer_steps=(batch + 1) * 2, examples=64)
        self.assertEqual(complete.summary()['timed_examples'], 12800)
        worker['verify_training_work'](work, complete.summary(),
                                       expected_batches=220,
                                       expected_batch_size=64)

    def test_training_example_count_uses_native_seed_rows(self):
        worker = runpy.run_path(str(WORKER))
        self.assertTrue(callable(worker.get('training_batch_examples')),
                        'Actual seed-row accounting is missing')
        actual = worker['training_batch_examples']
        batch = {'users': SimpleNamespace(batch_size=14)}
        module = SimpleNamespace(lhs_name='users', batch_size=256)
        self.assertEqual(actual(batch, module), 14)
        self.assertEqual(
            actual(batch, SimpleNamespace(entity_table_name='users')), 14)
        for size in (None, 0, -1, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                actual({'users': SimpleNamespace(batch_size=size)}, module)

    def test_gpu_ranking_measurement_keeps_cutoff_changes_unqualified(self):
        import pandas as pd

        worker = runpy.run_path(str(WORKER))
        self.assertIn('measure_recommendations', worker)
        before = pd.DataFrame({
            'ENTITY': [1, 1],
            'CLASS': ['a', 'b'],
            'SCORE': [0.2, 0.1]
        })
        after = pd.DataFrame({
            'ENTITY': [1, 1],
            'CLASS': ['a', 'c'],
            'SCORE': [0.200001, 0.100001]
        })
        measured = worker['measure_recommendations'](before, after)
        self.assertFalse(measured['qualified'])
        self.assertEqual(measured['common_recommendations'], 1)
        self.assertEqual(measured['removed_recommendations'], 1)
        self.assertEqual(measured['added_recommendations'], 1)
        self.assertEqual(measured['changed_entities'], 1)
        self.assertAlmostEqual(measured['max_abs_difference'], 0.000001)
        with self.assertRaises(ValueError):
            worker['measure_recommendations'](before, after.assign(ENTITY=2))

    def test_fixed_training_can_continue_after_the_timed_window(self):
        worker = runpy.run_path(str(WORKER))
        self.assertIn('verify_training_work', worker)
        window = worker['TrainingWindow'](lambda: None)
        for batch in range(880):
            window.batch_start()
            window.batch_end(optimizer_steps=(batch + 1) * 2)
        work = [
            dict(completed_batches=880, optimizer_steps=1760,
                 early_stopped=False)
        ]
        verify = worker['verify_training_work']
        verify(work, window.summary(), expected_batches=880)
        for broken in ([dict(work[0], completed_batches=220)],
                       [dict(work[0], early_stopped=True)], work * 2):
            with self.assertRaises(ValueError):
                verify(broken, window.summary(), expected_batches=880)

    def test_loader_profile_rejects_automatic_batch_or_worker_changes(self):
        worker = runpy.run_path(str(WORKER))
        self.assertIn('verify_loader_profile', worker)
        verify = worker['verify_loader_profile']
        requested = {'batch_size': 256, 'workers': 8}
        observed = {
            'train_batch_size': 256,
            'eval_batch_size': 256,
            'workers': 8
        }
        verify(observed, **requested)
        for broken in (dict(observed, train_batch_size=128),
                       dict(observed,
                            eval_batch_size=128), dict(observed,
                                                       workers=16), {}):
            with self.assertRaises(ValueError):
                verify(broken, **requested)

    def test_training_window_counts_batches_with_two_optimizer_updates(self):
        worker = runpy.run_path(str(WORKER))
        self.assertTrue(callable(worker.get('TrainingWindow')),
                        'Batch-based timing window is missing')
        window = worker['TrainingWindow'](lambda: None)
        for batch in range(220):
            window.batch_start()
            window.batch_end(optimizer_steps=(batch + 1) * 2)
        result = window.summary()
        self.assertEqual(result['completed_batches'], 220)
        self.assertEqual(result['optimizer_steps'], 440)
        self.assertEqual(result['warmup_batches'], 20)
        self.assertEqual(result['timed_batches'], 200)
        self.assertGreater(result['seconds'], 0)

    def test_stage_records_elapsed_work_and_preserves_failed_stage(self):
        self.assertTrue(WORKER.is_file(), 'Stage measurement helper is missing')
        worker = runpy.run_path(str(WORKER))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            measurements = worker['Measurements'](output)
            with measurements.stage('table'):
                time.sleep(0.01)
            with self.assertRaisesRegex(ValueError, 'broken'):
                with measurements.stage('fit'):
                    raise ValueError('broken')
            saved = json.loads((output / 'measurements.json').read_text())
            self.assertEqual([row['stage'] for row in saved], ['table', 'fit'])
            self.assertGreaterEqual(saved[0]['seconds'], 0.01)
            self.assertEqual(saved[0]['status'], 'COMPLETE')
            self.assertEqual(saved[1]['status'], 'FAIL')

    def test_binary_split_rejects_single_class_before_training(self):
        self.assertTrue(WORKER.is_file(), 'Stage measurement helper is missing')
        worker = runpy.run_path(str(WORKER))
        verify = worker['binary_split_counts']
        self.assertEqual(verify([False, True, False]), {
            'rows': 3,
            'positive': 1,
            'negative': 2
        })
        for values in ([], [False, False], [True], [False, None,
                                                    True], [False, 2, True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                verify(values)


if __name__ == '__main__':
    unittest.main()
