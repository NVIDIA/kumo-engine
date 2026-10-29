# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test cached evidence contracts. Metadata fixtures do not prove training."""

import copy
import hashlib
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = ROOT / 'packages/kumo-engine/test'
sys.path.insert(0, str(MODULES))
SCRIPT = ROOT / 'packages/kumo-engine/scripts/nightly_compare.py'
APPROVED_SHA = '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NightlyCompareTests(unittest.TestCase):
    def function(self, name):
        self.assertTrue(SCRIPT.is_file(), 'The nightly comparator is missing')
        result = runpy.run_path(str(SCRIPT)).get(name)
        self.assertTrue(callable(result), name)
        return result

    def test_cache_hash_and_every_file_are_checked(self):
        load = self.function('load_manifest')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save(root / 'report.json', {'status': 'MEASURED'})
            manifest = dict(
                schema=1, reference=dict(version='1.0.0',
                                         wheel_sha256=APPROVED_SHA), reports=[],
                files={'report.json': digest(root / 'report.json')})
            save(root / 'manifest.json', manifest)
            pinned = digest(root / 'manifest.json')
            self.assertEqual(load(root, pinned), manifest)
            with self.assertRaises(ValueError):
                load(root, '0' * 64)
            (root / 'report.json').write_text('changed')
            with self.assertRaises(ValueError):
                load(root, pinned)
            (root / 'report.json').unlink()
            with self.assertRaises(ValueError):
                load(root, pinned)

    def test_cache_rejects_escape_and_symlink_paths(self):
        confined = self.function('confined')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'real').write_text('real')
            (root / 'link').symlink_to(root / 'real')
            (root / 'directory').symlink_to(root, target_is_directory=True)
            for name in ('../real', '/tmp/real', 'directory/real', 'link',
                         'a/../real', './real'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    confined(root, name)
            self.assertEqual(confined(root, 'real'), root / 'real')

    def test_the_complete_roster_has_exactly_sixteen_unique_cases(self):
        require = self.function('require_roster')
        tasks = runpy.run_path(
            str(ROOT /
                'packages/kumo-engine/scripts/nightly_check.py'))['TASKS']
        rows = [
            dict(task=task, seed=0, case=arm, fit_id=f'{task}/{arm}')
            for task in tasks for arm in ('learnable', 'independent')
        ]
        self.assertEqual(len(require(rows)), 16)
        for altered in (rows[:-1], rows + [rows[0]],
                        [dict(rows[0], seed=False),
                         *rows[1:]], [dict(rows[0], task='other'), *rows[1:]]):
            with self.assertRaises(ValueError):
                require(altered)
        repeated_fit = copy.deepcopy(rows)
        repeated_fit[-1]['fit_id'] = repeated_fit[0]['fit_id']
        with self.assertRaises(ValueError):
            require(repeated_fit)
        require(rows, reference_fit_ids=['old/reference-fit'])
        with self.assertRaises(ValueError):
            require(rows, reference_fit_ids=[rows[0]['fit_id']])

    def test_dependency_changes_must_be_declared_exactly(self):
        verify = self.function('dependency_changes')
        before = {'torch': '2', 'pytorch-lightning': '2.6.5'}
        after = dict(before, **{'pytorch-lightning': '2.6.6'})
        expected = {
            'pytorch-lightning': dict(reference='2.6.5', candidate='2.6.6')
        }
        self.assertEqual(verify(before, after, expected), expected)
        for declaration in ({},
                            dict(expected, torch=dict(reference='1',
                                                      candidate='2'))):
            with self.assertRaises(ValueError):
                verify(before, after, declaration)
        with self.assertRaises(ValueError):
            verify({}, {}, {})

    def test_quality_drops_report_without_a_floor_or_gate(self):
        compare = self.function('quality_difference')
        self.assertEqual(
            compare(dict(metric='auroc', score=.01),
                    dict(metric='auroc', score=.9)),
            dict(candidate=.01, reference=.9, candidate_minus_reference=-.89,
                 quality_drop=.89, policy='report_only'))
        self.assertEqual(
            compare(dict(metric='mae', score=3.),
                    dict(metric='mae', score=1.))['quality_drop'], 2.)

    def test_semantic_signature_covers_population_and_every_plan(self):
        signature = self.function('input_signature')
        row = dict(task='multiclass', seed=0, training_seed=0, case='learnable',
                   metric='accuracy', task_type='multiclass_classification',
                   data_sha256='a' * 64, stage_sha256={'train': 'b' * 64},
                   query='actual query', entity_ids=[10, 20], targets=[[1, 0],
                                                                       [0, 1]],
                   baseline_predictions=[[1, 0], [1, 0]], classes=['a', 'b'],
                   split_counts=[100, 10, 2], entity_table='users',
                   entity_time_column='signup_time')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('requested-plan', 'selected-model-plan',
                         'resolved_model_plan', 'generation-plan'):
                save(root / f'{name}.json', {'field': name})
            original = signature(row, root, {'source': 'fixed'})
            changed = copy.deepcopy(row)
            changed['entity_ids'] = [10, 30]
            self.assertNotEqual(signature(changed, root, {'source': 'fixed'}),
                                original)
            self.assertNotEqual(signature(row, root, {'source': 'other'}),
                                original)
            for name in ('requested-plan', 'selected-model-plan',
                         'resolved_model_plan', 'generation-plan'):
                path = root / f'{name}.json'
                old = path.read_bytes()
                save(path, {'field': 'changed'})
                self.assertNotEqual(signature(row, root, {'source': 'fixed'}),
                                    original)
                path.write_bytes(old)

    def test_current_report_rejects_failed_execution_and_wrong_source(self):
        verify = self.function('validate_report')
        fixtures = runpy.run_path(
            str(Path(__file__).with_name('test_engine_nightly_runtime.py')))
        import wheel_gpu_learning as gpu
        rows = [
            fixtures['metadata_row']('multiclass', arm)
            for arm in ('learnable', 'independent')
        ]
        report = dict(
            status='MEASURED', group='all', seeds=[0], runs=rows,
            physical_fits=2, expected_runs=[['multiclass', 0, arm]
                                            for arm in ('learnable',
                                                        'independent')],
            completed_runs=[['multiclass', 0, arm]
                            for arm in ('learnable', 'independent')],
            configuration=dict(device='cuda', seed_profile='nightly'),
            gpu_source_sha256=gpu.source_identity(prepared=True))
        verify(report)
        for mutate in (lambda r: r.update(status='FAIL'),
                       lambda r: r.update(gpu_source_sha256='0' * 64),
                       lambda r: r.update(physical_fits=1),
                       lambda r: r['runs'][0]['checks'].update(reload='FAIL')):
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                verify(changed)

    def multiclass_case(self, root, arm='learnable'):
        import pandas as pd
        import torch
        import wheel_gpu_learning as gpu
        import wheel_scale_learning as learning
        from wheel_task_checks import fingerprint

        fixtures = runpy.run_path(
            str(Path(__file__).with_name('test_engine_nightly_runtime.py')))
        row = fixtures['metadata_row']('multiclass', arm)
        row.update(
            fit_id=f'trainer/{arm}', query='fixed query', classes=['a', 'b'],
            entity_table='users', entity_time_column='signup_time',
            split_counts=[64, 2, 2], partial_max_abs_difference=0.,
            reload_max_abs_difference=0., operation_measurements=[
                dict(stage='training', status='COMPLETE', seconds=1.)
            ])
        directory = root / f'multiclass-0-{arm}'
        directory.mkdir(parents=True)
        plan = learning.make_plan('multiclass', 0).as_json()
        row['plan_sha256'] = fingerprint(plan)
        for name in ('requested-plan', 'selected-model-plan',
                     'resolved_model_plan'):
            save(directory / f'{name}.json', plan)
        save(directory / 'generation-plan.json', {'fixed': 'generation'})
        save(directory / 'metadata.json', {'task_type': row['task_type']})
        save(
            directory / 'runtime_manifest.json',
            dict(task=dict(task_type=row['task_type'], is_temporal=True),
                 table=dict(columns=dict(timestamp='TIMESTAMP'))))
        save(
            directory / 'feature_schema.json',
            dict(node_types=dict(
                orders=dict(time_column='order_time', feature_columns=[]))))
        save(directory / 'trainer_config.json',
             dict(sampler_kwargs=dict(temporal_strategy='last')))
        for name in ('encoders', 'target_encoder'):
            save(directory / f'{name}.json', {})
        anchor = pd.Timestamp('2026-01-01')
        for label in ('predictions', 'repeat-predictions',
                      'partial-predictions', 'reloaded-predictions'):
            ids = [10] if label == 'partial-predictions' else [10, 20]
            inputs = directory / 'prediction-inputs' / f'{label}-inputs.parquet'
            inputs.parent.mkdir(exist_ok=True)
            pd.DataFrame(dict(ENTITY1=ids,
                              TIMESTAMP=[anchor] * len(ids))).to_parquet(
                                  inputs, index=False)
            records = [
                dict(ENTITY=entity, CLASS=category,
                     SCORE=float(category == ('a' if entity == 10 else 'b')),
                     PREDICTED=category == ('a' if entity == 10 else 'b'),
                     TIMESTAMP=anchor, ANCHOR_TIMESTAMP=anchor)
                for entity in ids for category in ('a', 'b')
            ]
            pd.DataFrame(records).to_parquet(directory / f'{label}.parquet',
                                             index=False)
        holdout = pd.read_parquet(directory / 'predictions.parquet')
        holdout['TARGET'] = ['a', 'a', 'b', 'b']
        holdout.to_parquet(directory / 'holdout.parquet', index=False)
        bundle = directory / f'scratch/trainers/trainer/runs/{arm}/published-model'
        checkpoint = bundle / 'model/checkpoint'
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            dict(global_step=1, state_dict={'weight': torch.tensor([1.])}),
            checkpoint)
        save(bundle / 'training-spec.json', {'spec_version': 1})
        save(
            bundle / 'publication-manifest.json',
            dict(artifacts=[
                dict(role=role, relative_path=name, sha256=digest(
                    bundle / name), size_bytes=(bundle / name).stat().st_size)
                for role, name in [(
                    'selected_checkpoint',
                    'model/checkpoint'), ('training_spec',
                                          'training-spec.json')]
            ]))
        row['expected_reload_checkpoint_sha256'] = digest(checkpoint)
        row['reload_identity_checks'][0]['checkpoint_sha256'] = digest(
            checkpoint)
        save(
            directory / 'device.json', {
                name: row[name]
                for name in ('device', 'observed_devices',
                             'consistency_measurements',
                             'batch_retries_disabled', 'training_work',
                             'training_window',
                             'expected_reload_checkpoint_sha256',
                             'reload_identity_checks')
            })
        save(directory / 'measurements.json', row['operation_measurements'])
        save(directory / 'result.json', row)
        expected = [['multiclass', 0, arm]]
        report = dict(status='MEASURED', group='all', seeds=[0], runs=[row],
                      physical_fits=1, expected_runs=expected,
                      completed_runs=expected,
                      configuration=dict(device='cuda', seed_profile='nightly'),
                      gpu_source_sha256=gpu.source_identity(prepared=True))
        save(root / 'nightly-result.json', report)
        save(root / 'fits.json', [
            dict(status='PASS', device='cuda:0', steps=1, cuda_batches=1,
                 metadata_test_only=True)
        ])
        save(root / 'cuda-batches.json', dict(training=1, prediction=4))
        return row, directory

    def test_pack_preserves_real_frames_and_bundle_but_omits_native_cache(self):
        pack = self.function('pack')
        verify = self.function('validate_artifacts')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            row, directory = self.multiclass_case(source)
            (directory / 'scratch/native-featurecache').mkdir()
            (directory / 'scratch/native-featurecache/large'
             ).write_bytes(b'not portable evidence')
            output = root / 'packed'
            packed = pack(source / 'nightly-result.json', output)
            artifact = packed['reports'][0]['artifacts'][
                'multiclass-0-learnable']
            verify(output, row, artifact)
            self.assertFalse(
                (output / 'multiclass-0-learnable/scratch').exists())
            self.assertTrue(
                (output / artifact['bundle'] / 'model/checkpoint').is_file())
            self.assertEqual(
                read_json := json.loads(
                    (output / 'nightly-result.json').read_text()),
                json.loads((source / 'nightly-result.json').read_text()))
            self.assertEqual(read_json['runs'][0], row)

    def test_raw_frame_anchor_coordinate_and_checkpoint_corruption_fail(self):
        import pandas as pd

        verify = self.function('validate_artifacts')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, directory = self.multiclass_case(root)
            artifact = dict(
                directory=directory.name, bundle=(
                    directory /
                    'scratch/trainers/trainer/runs/learnable/published-model'
                ).relative_to(root).as_posix())
            verify(root, row, artifact)
            path = directory / 'predictions.parquet'
            original = path.read_bytes()
            for alter in (lambda frame: frame.iloc[:-1], lambda frame: frame.
                          assign(ANCHOR_TIMESTAMP=pd.Timestamp('2027-01-01'))):
                alter(pd.read_parquet(path)).to_parquet(path, index=False)
                with self.assertRaises(ValueError):
                    verify(root, row, artifact)
                path.write_bytes(original)
            checkpoint = root / artifact['bundle'] / 'model/checkpoint'
            checkpoint.write_bytes(b'wrong checkpoint')
            with self.assertRaises(ValueError):
                verify(root, row, artifact)

    def test_pack_rejects_a_failed_physical_fit_even_when_the_report_passed(
            self):
        pack = self.function('pack')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.multiclass_case(root / 'source')
            save(root / 'source/fits.json',
                 [dict(status='FAIL', cuda_batches=0)])
            with self.assertRaises(ValueError):
                pack(root / 'source/nightly-result.json', root / 'packed')

    def test_runtime_comparison_does_not_depend_on_selected_link_shard(self):
        profile = self.function('runtime_profile')
        config = dict(device='cuda',
                      gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                      gpu_memory_bytes=96 * 1024**3, cuda='13.0',
                      torch='2.14.1+cu130', threads=4)
        self.assertEqual(
            profile(dict(configuration=dict(config, ranked_links=True))),
            profile(dict(configuration=dict(config, ranked_links=False))))
        with self.assertRaises(ValueError):
            profile(dict(configuration=dict(config, gpu_memory_bytes=None)))

    def test_legacy_summary_binds_the_selected_row_without_forging_full_rows(
            self):
        verify = self.function('verify_legacy_summary')
        row = dict(task='multiclass', seed=0, case='learnable',
                   fit_id='trainer/run', score=.4, baseline_score=.3)
        summary = dict(seed=0, fit_id='trainer/run', score=.4,
                       baseline_score=.3, result_sha256='a' * 64,
                       frozen_control=dict(descriptor_sha256='b' * 64))
        state = dict(status='PART_MEASURED', trial='R1', task='multiclass',
                     condition='learnable', results=[summary])
        verify(state, row, 'a' * 64, 'b' * 64)
        for field, value in [('seed', False), ('fit_id', 'other/run'),
                             ('score', .9), ('result_sha256', 'c' * 64)]:
            changed = copy.deepcopy(state)
            changed['results'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify(changed, row, 'a' * 64, 'b' * 64)

    def test_native_file_uri_is_resolved_before_path_confinement(self):
        local = self.function('local_artifact')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'name with space.parquet'
            path.write_bytes(b'preserved bytes')
            self.assertEqual(local(path.as_uri()), path)
            self.assertEqual(local(str(path)), path)
            with self.assertRaises(ValueError):
                local('s3://different-storage/file.parquet')

    def test_partial_pack_keeps_completed_case_without_claiming_success(self):
        pack = self.function('pack')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            row, _ = self.multiclass_case(source)
            report = json.loads((source / 'nightly-result.json').read_text())
            report.update(status='FAIL', stage='multiclass-0-independent',
                          error='preserved failure')
            report['expected_runs'].append(['multiclass', 0, 'independent'])
            save(source / 'nightly-result.json', report)
            save(source / 'multiclass-0-independent/requested-plan.json',
                 {'prepared': True})
            result = pack(source / 'nightly-result.json', root / 'packed',
                          allow_incomplete=True)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertIn('multiclass-0-independent', result['missing_cases'])
            self.assertTrue((
                root /
                'packed/multiclass-0-learnable/published-model/model/checkpoint'
            ).is_file())
            self.assertTrue(
                (root / 'packed/multiclass-0-independent/requested-plan.json'
                 ).is_file())
            saved = json.loads(
                (root / 'packed/nightly-result.json').read_text())
            self.assertEqual(saved['status'], 'FAIL')
            self.assertEqual(saved['runs'][0], row)

    def test_static_input_times_can_vary_per_entity_but_cannot_change_on_reload(
            self):
        import pandas as pd

        read_inputs = self.function('read_prediction_inputs')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / 'case/prediction-inputs'
            inputs.mkdir(parents=True)
            full = pd.DataFrame(
                dict(ENTITY1=[10, 20],
                     TIMESTAMP=pd.to_datetime(['2026-01-01', '2026-02-01'])))
            for label in ('predictions', 'repeat-predictions',
                          'partial-predictions', 'reloaded-predictions'):
                (full.iloc[:1] if label == 'partial-predictions' else
                 full).to_parquet(inputs / f'{label}-inputs.parquet',
                                  index=False)
            row = dict(task='static_link', entity_ids=[10, 20])
            self.assertIsNone(read_inputs(root, row, dict(directory='case'))[1])
            full.assign(TIMESTAMP=pd.Timestamp('2026-03-01')).to_parquet(
                inputs / 'reloaded-predictions-inputs.parquet', index=False)
            with self.assertRaises(ValueError):
                read_inputs(root, row, dict(directory='case'))

    def test_ranked_raw_reload_delta_and_incomplete_coordinates_are_rejected(
            self):
        import pandas as pd
        from wheel_measurements import measure_recommendations

        verify = self.function('validate_ranked_files')
        labels = ('predictions', 'repeat-predictions', 'partial-predictions',
                  'reloaded-predictions')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = list(range(100, 115))
            items = catalog[:10]
            scores = [1 - index * .01 for index in range(10)]
            row = dict(task='static_link', metric='recall@5',
                       entity_ids=[10, 20], catalog=catalog, target_ids=[[100],
                                                                         [101]],
                       ranked_ids=[items[:5]] * 2,
                       ranked_scores=[scores[:5]] * 2,
                       ranked_prediction_calls=[], consistency_measurements=[],
                       entity_time_column=None, repeat_max_abs_difference=0.,
                       partial_max_abs_difference=0.,
                       reload_max_abs_difference=0.)
            artifact = dict(ranked_files={})
            frames = {}
            for label in labels:
                ids = [10] if label == 'partial-predictions' else [10, 20]
                frame = pd.DataFrame([
                    dict(ENTITY=entity, CLASS=item, SCORE=1 - index * .01)
                    for entity in ids for index, item in enumerate(catalog[:12])
                ])
                frame.to_parquet(root / f'{label}.parquet', index=False)
                original = 'file:///original/' + label + '.parquet'
                artifact['ranked_files'][original] = label + '.parquet'
                row['ranked_prediction_calls'].append(
                    dict(label=label, entities=len(ids), raw_limit=12,
                         raw_rows=len(frame), files=[original],
                         full_catalog=False, fallback_ids=[]))
                frames[label] = frame[frame['CLASS'].isin(
                    items[:5])].reset_index(drop=True)
            for index, label in enumerate(labels[1:]):
                before = frames[
                    'predictions'] if label != 'partial-predictions' else frames[
                        'predictions'].iloc[:5]
                row['consistency_measurements'].append(
                    measure_recommendations(before, frames[label]))
            verify(root, row, artifact, dict(entity_ids=[10, 20], anchor=None))
            path = root / 'reloaded-predictions.parquet'
            original = path.read_bytes()
            changed = pd.read_parquet(path)
            changed.loc[0, 'SCORE'] += .001
            changed.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, 'agreement'):
                verify(root, row, artifact,
                       dict(entity_ids=[10, 20], anchor=None))
            path.write_bytes(original)
            changed.iloc[:-1].to_parquet(path, index=False)
            with self.assertRaises(ValueError):
                verify(root, row, artifact,
                       dict(entity_ids=[10, 20], anchor=None))

    def test_raw_report_collection_also_checks_the_physical_fit_ledger(self):
        collect = self.function('collect_reports')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, directory = self.multiclass_case(root)
            report_path = root / 'nightly-result.json'
            report = json.loads(report_path.read_text())
            report['configuration'].update(
                gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                gpu_memory_bytes=96 * 1024**3, cuda='13.0',
                torch='2.14.1+cu130', threads=4)
            report['profiles'] = {'multiclass': {'fixed': 'profile'}}
            save(report_path, report)
            save(
                root / 'fits.json',
                [dict(status='FAIL', device='cuda:0', steps=0, cuda_batches=0)])
            bundle = (directory /
                      'scratch/trainers/trainer/runs/learnable/published-model'
                      ).relative_to(root)
            artifact = dict(directory=directory.name, bundle=bundle.as_posix())
            descriptor = dict(path='nightly-result.json',
                              artifacts={'multiclass-0-learnable': artifact})
            with self.assertRaisesRegex(ValueError, 'Every fit requires'):
                collect(root, [descriptor])


if __name__ == '__main__':
    unittest.main()
