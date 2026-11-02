# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for learning loss, reference replay, and failed-case retention."""

import copy
import fcntl
import importlib.util
import inspect
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

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
                    population='data.json', dataset_manifest='data.json',
                    answers=None if task == 'P2' else 'data.json',
                    catalog='data.json' if metric.startswith('recall@') else
                    None, raw_limit=51 if task == 'Q13' else 11, tolerance=dict(
                        atol=1e-5,
                        rtol=1e-5), capacity=dict(memory_bytes=100,
                                                  scratch_bytes=200),
                    gpu_class=dict(compute_capability=[12, 0],
                                   minimum_memory_bytes=90)))
    value = dict(format='kumo-engine-weekly-input-v2',
                 reference=dict(version='2.0.0', wheel_sha256='a' * 64),
                 files={'data.json': weekly.digest(payload)}, cases=cases)
    weekly.save(root / 'manifest.json', value)
    return value


class WeeklyChecks(unittest.TestCase):
    def test_b9_input_scope_cannot_replace_the_eleven_case_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = input_fixture(root)
            value.update(format='kumo-engine-weekly-p2-input-v1',
                         cases=[value['cases'][-1]])
            weekly.save(root / 'manifest.json', value)
            checksum = weekly.digest(root / 'manifest.json')
            self.assertEqual(
                len(
                    weekly.load_manifest(root, checksum,
                                         retained_p2=True)['cases']), 1)
            with self.assertRaises(ValueError):
                weekly.load_manifest(root, checksum)
            (root / 'data.json').write_text('changed P2 bytes')
            with self.assertRaises(ValueError):
                weekly.load_manifest(root, checksum, retained_p2=True)

    def test_shared_helper_checks_installed_wheel_bytes(self):
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        helper = worker.installation_helper()
        self.assertTrue(callable(getattr(helper, 'verify_checkpoint', None)),
                        'The runtime helper must check restored weights')
        inspect.signature(helper.verify_checkpoint).bind(None, None, 'a' * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed = root / 'installed'
            installed.mkdir()
            (installed / 'module.py').write_bytes(b'approved wheel bytes')
            wheel = root / 'candidate.whl'
            with ZipFile(wheel, 'w') as archive:
                archive.writestr('module.py', b'approved wheel bytes')
            distribution = SimpleNamespace(
                locate_file=lambda name: installed / name)
            self.assertEqual(helper.verify_installed_files(wheel, distribution),
                             1)
            (installed / 'module.py').write_bytes(b'changed wheel bytes')
            with self.assertRaises(ValueError):
                helper.verify_installed_files(wheel, distribution)

    def test_admission_rejects_missing_prediction_inputs_before_a_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field in ('catalog', 'raw_limit', 'population',
                          'dataset_manifest'):
                value = input_fixture(root)
                del value['cases'][0][field]
                weekly.save(root / 'manifest.json', value)
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        weekly.load_manifest(
                            root, weekly.digest(root / 'manifest.json'))

    def test_busy_host_fails_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'host.lock'
            with path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    weekly.run(
                        SimpleNamespace(host_lock=path,
                                        input_root=root / 'missing-inputs',
                                        manifest_sha256='a' * 64, profile=None))
                except BlockingIOError:
                    pass
                except Exception as error:
                    self.fail(f'Input read before host reservation: {error}')
                else:
                    self.fail('Busy host was accepted')

    def test_dataset_admission_rejects_a_smaller_graph_with_a_scale_label(self):
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        self.assertTrue(
            callable(getattr(worker, 'check_graph_population', None)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weekly.save(root / 'ingest_stats.json',
                        dict(stats=dict(node_counts=dict(events=100000000))))
            weekly.save(root / 'index_metadata.json',
                        dict(node_types=[dict(name='events', max_id=99999999)]))
            snapshot = SimpleNamespace(
                index_path=root,
                graph_config=dict(tables=[dict(table_name='events')]))
            source = dict(graph='streaming', rung='100M', total_rows=100000000,
                          tables=dict(events=dict(rows=100000000)))
            worker.check_graph_population(snapshot,
                                          dict(dataset='streaming/100M'),
                                          source)
            source['total_rows'] = 10
            source['tables']['events']['rows'] = 10
            with self.assertRaises(ValueError):
                worker.check_graph_population(snapshot,
                                              dict(dataset='streaming/100M'),
                                              source)
            source.update(graph='commerce', rung='25B', total_rows=100000000)
            with self.assertRaises(ValueError):
                worker.check_graph_population(snapshot,
                                              dict(dataset='commerce/25B'),
                                              source)

    def test_native_contract_rejects_future_features_and_changed_cutoffs(self):
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        self.assertTrue(callable(getattr(worker, 'check_native_contract',
                                         None)))
        graph = dict(tables=[
            dict(table_name='events', cols=[dict(name='amount')], time_col=dict(
                name='TS'))
        ])
        runtime = dict(
            task=dict(task_type='binary_classification', is_temporal=True),
            table=dict(columns=dict(timestamp='TIMESTAMP')),
            graph=dict(node_types=['events']), stages={
                s: dict(row_count=n)
                for s, n in [('train', 100), ('val', 20), ('test', 20)]
            })
        features = dict(node_types=dict(
            events=dict(feature_columns=['amount', 'TS'], time_column='TS')))
        config = dict(sampler_kwargs=dict(temporal_strategy='last'))
        expected = dict(train=100, val=20, test=20)
        worker.check_native_contract(graph, runtime, features, config, expected,
                                     'binary_classification')
        for kind in ('future', 'timestamp', 'rows', 'sampler'):
            r, f, c = copy.deepcopy(runtime), copy.deepcopy(
                features), copy.deepcopy(config)
            if kind == 'future':
                f['node_types']['events']['feature_columns'].append('TARGET1')
            elif kind == 'timestamp':
                f['node_types']['events']['time_column'] = None
            elif kind == 'rows':
                r['stages']['test']['row_count'] = 19
            else:
                c['sampler_kwargs']['temporal_strategy'] = 'uniform'
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    worker.check_native_contract(graph, r, f, c, expected,
                                                 'binary_classification')

    def test_native_extensions_stay_outside_the_frozen_graph(self):
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        self.assertTrue(callable(getattr(worker, 'stage_snapshot', None)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'index').mkdir()
            (root / 'index/graph.bin').write_bytes(b'frozen native graph')
            weekly.save(root / 'index/train_tables/old/manifest.json',
                        dict(old_cache=True))
            files = {'index/graph.bin': weekly.digest(root / 'index/graph.bin')}
            staged = worker.stage_snapshot(root, 'index', files,
                                           root / 'scratch/index')
            self.assertEqual((staged / 'graph.bin').read_bytes(),
                             b'frozen native graph')
            self.assertFalse((staged / 'train_tables').exists())
            weekly.save(staged / 'train_tables/new/manifest.json',
                        dict(new_cache=True))
            self.assertFalse((root / 'index/train_tables/new').exists())
            self.assertTrue(
                (root / 'index/train_tables/old/manifest.json').exists())
            (root / 'index/foreign.bin').write_bytes(b'unlisted graph data')
            with self.assertRaises(ValueError):
                worker.stage_snapshot(root, 'index', files,
                                      root / 'scratch/changed')

    def test_collection_requires_both_profiles_and_enforces_learning(self):
        self.assertTrue(callable(getattr(weekly, 'collect_shards', None)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = input_fixture(root)
            manifest_sha256 = weekly.digest(root / 'manifest.json')
            wheel = dict(version='2.0.0', wheel_sha256='a' * 64)
            paths = []
            for profile in ('rtx', 'scale'):
                output = root / profile
                output.mkdir()
                shutil.copyfile(root / 'manifest.json',
                                output / 'manifest.json')
                rows = []
                for case in weekly.select_cases(manifest, profile):
                    name = case['id']
                    row = dict(
                        status='PASS', case_id=name, physical_fits=1,
                        fit_id='fit-' + name,
                        native_contract=dict(status='PASS'), runtime=wheel,
                        checkpoint_sha256='b' * 64, score=scores().get(name),
                        reload=dict(status='PASS'), work=dict(
                            completed_fits=1, optimizer_steps=10,
                            training_batches=10,
                            restored_tensors=dict(reload=100,
                                                  reference_replay=100),
                            prediction_batches=dict(candidate=2, reload=2)))
                    destination = output / 'cases' / name
                    weekly.save(destination / 'model/model.json',
                                dict(model='fixture'))
                    weekly.save(destination / 'predictions.json',
                                dict(entity_ids=[1], values=[[.2]]))
                    weekly.save(destination / 'worker-result.json', row)
                    weekly.save(destination / 'result.json', row)
                    rows.append(row)
                weekly.save(
                    output / 'weekly-result.json',
                    dict(format='kumo-engine-weekly-run-v2', status='PASS',
                         mode='reference', profile=profile, wheel=wheel,
                         manifest_sha256=manifest_sha256,
                         case_ids=[v['case_id'] for v in rows], cases=rows))
                paths.append(output)
            with self.assertRaises(ValueError):
                weekly.collect_shards(paths[:1], manifest, manifest_sha256,
                                      root / 'missing')
            result = weekly.collect_shards(paths, manifest, manifest_sha256,
                                           root / 'complete')
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(len(result['cases']), 11)
            self.assertTrue((root / 'complete/reference.json').is_file())
            self.assertFalse((root / 'complete/cases').exists())
            reference_sha256 = weekly.digest(root / 'complete/reference.json')
            candidate_wheel = dict(version='2.1.0+candidate',
                                   wheel_sha256='c' * 64)
            for path in paths:
                shutil.copyfile(root / 'complete/reference.json',
                                path / 'reference.json')
                report = weekly.read(path / 'weekly-result.json')
                report.update(mode='candidate',
                              reference_sha256=reference_sha256,
                              wheel=candidate_wheel)
                weekly.save(
                    path / 'build.json',
                    dict(revision='e' * 40, tree='f' * 40, **candidate_wheel))
                for row in report['cases']:
                    row.update(runtime=candidate_wheel,
                               fit_id='candidate-' + row['case_id'],
                               reference_replay=dict(status='PASS'))
                    row['work']['prediction_batches']['reference_replay'] = 2
                    weekly.save(
                        path / 'cases' / row['case_id'] / 'worker-result.json',
                        row)
                    weekly.save(path / 'cases' / row['case_id'] / 'result.json',
                                row)
                weekly.save(path / 'weekly-result.json', report)
            candidate = weekly.collect_shards(paths, manifest, manifest_sha256,
                                              root / 'candidate',
                                              reference_sha256=reference_sha256)
            self.assertEqual(candidate['status'], 'PASS')
            self.assertEqual(candidate['physical_fits'], 11)
            self.assertFalse((root / 'candidate/cases').exists())
            (paths[1] / 'manifest.json').write_text('{}')
            with self.assertRaises(ValueError):
                weekly.collect_shards(paths, manifest, manifest_sha256,
                                      root / 'changed-header',
                                      reference_sha256=reference_sha256)
            shutil.copyfile(root / 'manifest.json', paths[1] / 'manifest.json')
            report = weekly.read(paths[0] / 'weekly-result.json')
            for row in report['cases']:
                if row['case_id'] == 'Q17-normal':
                    row['score'] = .004
                    weekly.save(
                        paths[0] / 'cases/Q17-normal/worker-result.json', row)
                    weekly.save(paths[0] / 'cases/Q17-normal/result.json', row)
            weekly.save(paths[0] / 'weekly-result.json', report)
            result = weekly.collect_shards(paths, manifest, manifest_sha256,
                                           root / 'no-learning',
                                           reference_sha256=reference_sha256)
            self.assertEqual(result['status'], 'FAIL')
            self.assertFalse((root / 'no-learning/reference.json').exists())

    def test_profiles_keep_the_ten_pairs_separate_from_p2(self):
        self.assertTrue(callable(getattr(weekly, 'select_cases', None)))
        with tempfile.TemporaryDirectory() as directory:
            value = input_fixture(Path(directory))
            pairs = weekly.select_cases(value, 'rtx')
            scale = weekly.select_cases(value, 'scale')
            self.assertEqual([v['id'] for v in scale], ['P2-fixed'])
            self.assertEqual(len(pairs), 10)
            self.assertEqual({v['id']
                              for v in pairs} | {v['id']
                                                 for v in scale}, weekly.ROSTER)
            with self.assertRaises(ValueError):
                weekly.select_cases(value, 'unknown')

    def test_admission_does_not_read_data_from_the_other_host(self):
        self.assertTrue(callable(getattr(weekly, 'select_cases', None)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = input_fixture(root)
            value['cases'][-1]['files'] = ['scale-only.json']
            for key in ('snapshot', 'training_table', 'model_plan',
                        'prediction_table', 'population'):
                value['cases'][-1][key] = 'scale-only.json'
            value['files']['scale-only.json'] = 'e' * 64
            weekly.save(root / 'manifest.json', value)
            checksum = weekly.digest(root / 'manifest.json')
            weekly.load_manifest(root, checksum, profile='rtx')
            with self.assertRaises(ValueError):
                weekly.load_manifest(root, checksum, profile='scale')

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
            native_contract=dict(status='PASS'),
            reference_replay=dict(status='PASS'), reload=dict(status='PASS'),
            work=dict(
                completed_fits=1, optimizer_steps=10, training_batches=10,
                restored_tensors=dict(reload=100, reference_replay=100),
                prediction_batches=dict(candidate=2, reload=2,
                                        reference_replay=2)))
        weekly.check_case_result(row, 'Q17-normal', wheel, candidate=True)
        missing_state = copy.deepcopy(row)
        missing_state['work']['restored_tensors']['reference_replay'] = 0
        with self.assertRaises(ValueError):
            weekly.check_case_result(missing_state, 'Q17-normal', wheel,
                                     candidate=True)
        for key in ('physical_fits', 'reference_replay', 'runtime',
                    'native_contract'):
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
            raw['CLASS'] = raw.CLASS.astype(float)
            raw.to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'class key type'):
                worker.prediction_values([path], population, case,
                                         root / 'sort', root)
            raw['CLASS'] = raw.CLASS.astype(int)
            raw.loc[0, 'CLASS'] = 11
            raw.to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'ranked'):
                worker.prediction_values([path], population, case,
                                         root / 'sort', root)

    def test_ranked_reader_preserves_approved_public_topk_ties(self):
        worker = importlib.util.module_from_spec(WORKER_SPEC)
        WORKER_SPEC.loader.exec_module(worker)
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(dict(CLASS=[10, 11, 12, 13])).to_parquet(
                root / 'catalog.parquet')
            path = root / 'raw.parquet'
            pd.DataFrame(
                dict(ENTITY=[1, 1, 1], CLASS=[12, 10, 11],
                     SCORE=[.1, .3, .1])).to_parquet(path)
            case = dict(metric='recall@2', raw_limit=3,
                        catalog='catalog.parquet')
            result = worker.prediction_values([path],
                                              pd.DataFrame(dict(ENTITY1=[1])),
                                              case, root / 'sort', root)
            self.assertEqual(
                result,
                dict(entity_ids=[1], ranked_ids=[[10, 11]], values=[[.3, .1]]))

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

    def test_rank_budget_only_allows_close_score_inversions_with_exact_membership(
            self):
        original = dict(entity_ids=[1], ranked_ids=[[3, 4]],
                        values=[[.5002, .5]])
        swapped = dict(entity_ids=[1], ranked_ids=[[4, 3]], values=[[.5001,
                                                                     .5]])
        with self.assertRaises(ValueError):
            weekly.compare_predictions(original, swapped, atol=.002, rtol=0.)
        result = weekly.replay_reference('saved', original, lambda _: object(),
                                         lambda _: swapped, atol=.002, rtol=0.,
                                         rank_atol=.0004)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['rank_changed_entities'], 1)
        self.assertEqual(result['rank_inversions'], 1)
        self.assertAlmostEqual(result['max_abs_difference'], .0002)
        self.assertAlmostEqual(result['max_reordered_score_gap'], .0002)
        for before, after in [(dict(original, values=[[.5005, .5]]), swapped),
                              (original, dict(swapped, values=[[.5005, .5]])),
                              (original, dict(swapped, ranked_ids=[[5, 3]])),
                              (original, dict(swapped, values=[[.4999, .5]])),
                              (original,
                               dict(swapped, values=[[float('nan'), .5]])),
                              (dict(entity_ids=[1], ranked_ids=[[1, 2, 3]],
                                    values=[[.5006, .5003, .5]]),
                               dict(entity_ids=[1], ranked_ids=[[3, 2, 1]],
                                    values=[[.5006, .5003, .5]]))]:
            with self.subTest(before=before,
                              after=after), self.assertRaises(ValueError):
                weekly.compare_predictions(before, after, atol=.002, rtol=0.,
                                           rank_atol=.0004)
        for invalid in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                weekly.compare_predictions(original, original, atol=.002,
                                           rtol=0., rank_atol=invalid)

    def test_manifest_binds_and_validates_the_optional_rank_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = input_fixture(root)
            value['cases'][-1]['tolerance']['rank_atol'] = .0004
            weekly.save(root / 'manifest.json', value)
            weekly.load_manifest(root, weekly.digest(root / 'manifest.json'))
            for policy in (dict(atol=.002, rtol=0., rank_atol=-1.),
                           dict(atol=.002, rtol=0., rank_atol='wide'),
                           dict(atol=.002, rtol=0.,
                                unreviewed=True), dict(rtol=0.,
                                                       rank_atol=.0004)):
                value['cases'][-1]['tolerance'] = policy
                weekly.save(root / 'manifest.json', value)
                with self.subTest(policy=policy), self.assertRaises(ValueError):
                    weekly.load_manifest(root,
                                         weekly.digest(root / 'manifest.json'))

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
