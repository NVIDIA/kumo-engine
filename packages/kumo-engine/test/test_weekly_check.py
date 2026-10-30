# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for learning loss, reference replay, and failed-case retention."""

import copy
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'scripts/weekly_check.py'
SPEC = importlib.util.spec_from_file_location('weekly_check', SOURCE)
weekly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weekly)
WORKER_SPEC = importlib.util.spec_from_file_location(
    'weekly_worker', SOURCE.with_name('weekly_worker.py'))


def scores():
    return {
        f'{task}-{arm}': value
        for task, normal, control in [('Q13', .6, .01), (
            'Q14', 2., 10.), ('Q15', .533, .531), ('Q16', .9,
                                                   .5), ('Q17', .742, .004)]
        for arm, value in [('normal', normal), ('control', control)]
    }


def input_fixture(root):
    payload = root / 'data.json'
    payload.write_text('{}')
    records = [('Q13', 'streaming/100M', 'recall@50'),
               ('Q14', 'community/100M', 'mae'),
               ('Q15', 'payments/100M core', 'auroc'),
               ('Q16', 'fraud/100M', 'auroc'),
               ('Q17', 'commerce/1B', 'recall@10'),
               ('P2', 'commerce/25B', 'recall@10')]
    cases = []
    for task, dataset, metric in records:
        for arm in ('fixed', ) if task == 'P2' else ('normal', 'control'):
            cases.append(
                dict(
                    id=f'{task}-{arm}', arm=arm, dataset=dataset, metric=metric,
                    seed=0, query='frozen query', files=['data.json'],
                    snapshot='data.json', training_table='data.json',
                    model_plan='data.json', prediction_table='data.json',
                    population='data.json',
                    answers=None if task == 'P2' else 'data.json',
                    tolerance=dict(atol=1e-5, rtol=1e-5),
                    capacity=dict(memory_bytes=100, scratch_bytes=200),
                    gpu_class=dict(compute_capability=[12, 0],
                                   minimum_memory_bytes=90)))
    value = dict(format='kumo-engine-weekly-input-v2',
                 reference=dict(version='2.0.0', wheel_sha256='a' * 64),
                 files={'data.json': weekly.digest(payload)}, cases=cases)
    weekly.save(root / 'manifest.json', value)
    return value


class WeeklyChecks(unittest.TestCase):
    def test_build_receipt_binds_source_version_and_wheel(self):
        self.assertTrue(callable(getattr(weekly, 'check_build', None)))
        receipt = dict(revision='a' * 40, tree='b' * 40, version='2.0+abc',
                       wheel_sha256='c' * 64)
        weekly.check_build(receipt, '2.0+abc', 'c' * 64)
        for key in receipt:
            changed = dict(receipt)
            changed[key] = 'wrong'
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    weekly.check_build(changed, '2.0+abc', 'c' * 64)

    def test_manifest_has_all_eleven_cases_and_allows_a_new_reference_version(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = input_fixture(root)
            admitted = weekly.load_manifest(
                root, weekly.digest(root / 'manifest.json'))
            self.assertEqual(admitted['reference']['version'], '2.0.0')
            self.assertEqual(len(admitted['cases']), 11)
            value['cases'].pop()
            weekly.save(root / 'manifest.json', value)
            with self.assertRaises(ValueError):
                weekly.load_manifest(root,
                                     weekly.digest(root / 'manifest.json'))

    def test_manifest_rejects_changed_payload_and_invalid_tolerances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = input_fixture(root)
            (root / 'data.json').write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                weekly.load_manifest(root,
                                     weekly.digest(root / 'manifest.json'))
            value = input_fixture(root)
            value['cases'][0]['tolerance']['atol'] = -1
            weekly.save(root / 'manifest.json', value)
            with self.assertRaises(ValueError):
                weekly.load_manifest(root,
                                     weekly.digest(root / 'manifest.json'))

    def test_candidate_requires_cuda_fit_and_cross_version_replay(self):
        self.assertTrue(callable(getattr(weekly, 'check_case_result', None)))
        wheel = dict(version='2.0+abc', wheel_sha256='a' * 64)
        row = dict(
            status='PASS', case_id='Q17-normal', physical_fits=1,
            fit_id='fresh-fit', runtime=wheel,
            reference_replay=dict(status='PASS'), reload=dict(status='PASS'),
            work=dict(
                completed_fits=1, optimizer_steps=10, training_batches=10,
                prediction_batches=dict(candidate=2, reload=2,
                                        reference_replay=2)))
        weekly.check_case_result(row, 'Q17-normal', wheel, candidate=True)
        for key in ('physical_fits', 'reference_replay', 'runtime'):
            changed = copy.deepcopy(row)
            changed[key] = 0 if key == 'physical_fits' else {}
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    weekly.check_case_result(changed, 'Q17-normal', wheel,
                                             candidate=True)
        row['work']['prediction_batches']['reference_replay'] = 0
        with self.assertRaises(ValueError):
            weekly.check_case_result(row, 'Q17-normal', wheel, candidate=True)

    def test_capacity_shortage_fails_before_case_execution(self):
        self.assertTrue(callable(getattr(weekly, 'check_capacity', None)))
        case = dict(capacity=dict(memory_bytes=100, scratch_bytes=200))
        weekly.check_capacity([case], dict(memory_bytes=100, scratch_bytes=200))
        for available in (dict(memory_bytes=99, scratch_bytes=200),
                          dict(memory_bytes=100, scratch_bytes=199)):
            with self.subTest(available=available):
                with self.assertRaises(ValueError):
                    weekly.check_capacity([case], available)

    def test_only_default_branch_history_can_supply_the_candidate(self):
        self.assertTrue(callable(getattr(weekly, 'trusted_revision', None)))
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ['git', '-C', str(repo), *args], text=True).strip()

            git('init', '-b', 'main')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.com')
            git('commit', '--allow-empty', '-m', 'Default branch')
            approved = git('rev-parse', 'HEAD')
            git('checkout', '-b', 'untrusted')
            git('commit', '--allow-empty', '-m', 'Unmerged product code')
            self.assertEqual(weekly.trusted_revision(repo, approved, 'main'),
                             approved)
            for ref in ('untrusted', '--help', 'missing'):
                with self.subTest(ref=ref):
                    with self.assertRaises(
                        (ValueError, subprocess.CalledProcessError)):
                        weekly.trusted_revision(repo, ref, 'main')

    def test_scalar_reader_uses_complete_original_population_and_anchor(self):
        self.assertTrue(SOURCE.with_name('weekly_worker.py').is_file())
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = pd.DataFrame(
                dict(ENTITY1=[2, 1],
                     TIMESTAMP=pd.to_datetime(['2024-01-01'] * 2)))
            raw = pd.DataFrame(
                dict(ENTITY=[1, 2], TARGET_PRED=[True, False],
                     True_PROB=[.8, .2], False_PROB=[.2, .8],
                     ANCHOR_TIMESTAMP=pd.to_datetime(['2024-01-01'] * 2)))
            path = root / 'raw.parquet'
            raw.to_parquet(path)
            result = worker.prediction_values([path], population,
                                              dict(metric='auroc'),
                                              root / 'sort', root)
            self.assertEqual(
                result,
                dict(entity_ids=[2, 1], values=[[.2], [.8]],
                     decisions=[False, True]))
            self.assertFalse((root / 'sort').exists())
            raw.iloc[:1].to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'omitted'):
                worker.prediction_values([path], population,
                                         dict(metric='auroc'), root / 'sort',
                                         root)
            raw['ANCHOR_TIMESTAMP'] = pd.Timestamp('2024-02-01')
            raw.to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'anchor'):
                worker.prediction_values([path], population,
                                         dict(metric='auroc'), root / 'sort',
                                         root)

    def test_ranked_reader_sorts_raw_rows_and_rejects_duplicate_coordinates(
            self):
        self.assertTrue(SOURCE.with_name('weekly_worker.py').is_file())
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = pd.DataFrame(dict(ENTITY1=[2, 1]))
            pd.DataFrame(dict(CLASS=[10, 11, 12])).to_parquet(root /
                                                              'catalog.parquet')
            raw = pd.DataFrame(
                dict(ENTITY=[1, 2, 1, 2, 1, 2], CLASS=[10, 10, 11, 11, 12, 12],
                     SCORE=[.1, .6, .3, .1, .2, .3]))
            path = root / 'raw.parquet'
            raw.to_parquet(path)
            case = dict(metric='recall@2', raw_limit=3,
                        catalog='catalog.parquet')
            result = worker.prediction_values([path], population, case,
                                              root / 'sort', root)
            self.assertEqual(
                result,
                dict(entity_ids=[2, 1], ranked_ids=[[10, 12], [11, 12]],
                     values=[[.6, .3], [.3, .2]]))
            raw.loc[0, 'CLASS'] = 11
            raw.to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'ranked'):
                worker.prediction_values([path], population, case,
                                         root / 'sort', root)

    def test_gpu_class_accepts_a_host_or_driver_replacement(self):
        self.assertTrue(SOURCE.with_name('weekly_worker.py').is_file())
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        expected = dict(compute_capability=[12, 0], minimum_memory_bytes=90)
        actual = dict(compute_capability=[12, 0], memory_bytes=100, gpu_count=1,
                      driver='600.1', uuid='replacement-host-device')
        worker.check_gpu_class(expected, actual)
        actual['compute_capability'] = [7, 5]
        with self.assertRaises(ValueError):
            worker.check_gpu_class(expected, actual)

    def test_learning_loss_fails_for_each_separating_task(self):
        self.assertTrue(callable(getattr(weekly, 'check_learning', None)))
        reference = scores()
        self.assertEqual(
            weekly.check_learning(reference, reference)['status'], 'PASS')
        for task in ('Q13', 'Q14', 'Q16', 'Q17'):
            with self.subTest(task=task):
                candidate = scores()
                candidate[task + '-normal'] = candidate[task + '-control']
                self.assertEqual(
                    weekly.check_learning(candidate, reference)['status'],
                    'FAIL')

    def test_q15_does_not_claim_learning(self):
        self.assertTrue(callable(getattr(weekly, 'check_learning', None)))
        candidate = scores()
        candidate['Q15-normal'] = .5
        result = weekly.check_learning(candidate, scores())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['tasks']['Q15']['policy'], 'report_only')

    def test_learning_requires_finite_complete_reference_and_positive_gap(self):
        self.assertTrue(callable(getattr(weekly, 'check_learning', None)))
        for score in (None, float('nan'), float('inf'), .004):
            with self.subTest(score=score):
                reference = scores()
                reference['Q17-normal'] = score
                self.assertEqual(
                    weekly.check_learning(scores(), reference)['status'],
                    'FAIL')
        candidate = scores()
        del candidate['Q17-control']
        self.assertEqual(
            weekly.check_learning(candidate, scores())['status'], 'FAIL')

    def test_replay_rejects_missing_population_rank_changes_and_numeric_drift(
            self):
        self.assertTrue(callable(getattr(weekly, 'compare_predictions', None)))
        original = dict(entity_ids=[1, 2], ranked_ids=[[3, 4], [4, 3]],
                        values=[[.8, .2], [.7, .3]])
        self.assertEqual(
            weekly.compare_predictions(original, original, atol=1e-5,
                                       rtol=1e-5)['status'], 'PASS')
        for key, changed in [('entity_ids', [1]),
                             ('ranked_ids', [[4, 3], [4, 3]]),
                             ('values', [[.6, .4], [.7, .3]]),
                             ('values', [[float('nan'), .2], [.7, .3]])]:
            other = copy.deepcopy(original)
            other[key] = changed
            with self.subTest(key=key, changed=changed):
                with self.assertRaises(ValueError):
                    weekly.compare_predictions(original, other, atol=1e-5,
                                               rtol=1e-5)

    def test_case_failure_does_not_skip_remaining_cases(self):
        self.assertTrue(callable(getattr(weekly, 'execute_cases', None)))
        visited = []

        def execute(case, directory):
            visited.append(case['id'])
            if case['id'] == 'first':
                raise ValueError('checkpoint format broke')
            return dict(status='PASS', case_id=case['id'])

        with tempfile.TemporaryDirectory() as directory:
            report = weekly.execute_cases(
                [dict(id='first'), dict(id='second')], Path(directory), execute)
            self.assertEqual(visited, ['first', 'second'])
            self.assertEqual(report['status'], 'FAIL')
            self.assertIn('checkpoint format broke',
                          report['cases'][0]['error'])
            self.assertEqual(
                weekly.read(Path(directory) / 'second/result.json')['status'],
                'PASS')

    def test_manifest_paths_cannot_escape_or_select_code(self):
        self.assertTrue(callable(getattr(weekly, 'data_file', None)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data.json').write_text('{}')
            checksum = weekly.digest(root / 'data.json')
            self.assertEqual(
                weekly.data_file(root, 'data.json', {'data.json': checksum}),
                root / 'data.json')
            for name in ('../data.json', '/data.json', './data.json',
                         'worker.py'):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        weekly.data_file(root, name, {name: checksum})

    def test_replay_uses_the_saved_model_and_has_no_fit(self):
        self.assertTrue(callable(getattr(weekly, 'replay_reference', None)))
        saved = dict(entity_ids=[1, 2], values=[[.2], [.8]],
                     decisions=[False, True])
        events = []

        class LoadedModel:
            def fit(self, *args, **kwargs):
                raise AssertionError('Reference replay must never fit')

        def load(path):
            events.append(('load', path))
            return LoadedModel()

        def predict(model):
            self.assertIsInstance(model, LoadedModel)
            events.append(('predict', ))
            return copy.deepcopy(saved)

        report = weekly.replay_reference(Path('approved/model'), saved, load,
                                         predict, atol=1e-5, rtol=1e-5)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(events, [('load', Path('approved/model')),
                                  ('predict', )])
        saved['values'][0] = [.9]
        with self.assertRaises(ValueError):
            weekly.replay_reference(
                Path('approved/model'), saved, load,
                lambda model: dict(entity_ids=[1, 2], values=[[.2], [.8]],
                                   decisions=[False, True]), atol=1e-5,
                rtol=1e-5)

    def test_replay_propagates_checkpoint_format_failure(self):
        self.assertTrue(callable(getattr(weekly, 'replay_reference', None)))

        def load(path):
            raise ValueError('unsupported checkpoint format')

        with self.assertRaisesRegex(ValueError,
                                    'unsupported checkpoint format'):
            weekly.replay_reference(
                Path('approved/model'), {}, load,
                lambda model: self.fail('Prediction after failed load'),
                atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
