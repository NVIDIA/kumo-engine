# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test cached evidence contracts. Metadata fixtures do not prove training."""

import copy
import hashlib
import json
import runpy
import shutil
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

    def test_fixed_recipe_requires_completed_epochs_or_recorded_early_stop(
            self):
        verify = self.function('verify_recipe_work')
        plan = dict(optimization=dict(max_epochs=10, min_steps_per_epoch=20,
                                      max_steps_per_epoch=20))
        for batches, epoch, stopped in [(200, 10, False), (200, 10, True),
                                        (80, 4, True)]:
            verify(
                dict(training_work=[
                    dict(completed_batches=batches, current_epoch=epoch,
                         early_stopped=stopped)
                ]), plan)
        for batches, epoch, stopped in [(199, 10, False), (79, 4, True),
                                        (80, 4, False), (220, 11, True),
                                        (20, True, True)]:
            with self.subTest(batches=batches,
                              epoch=epoch), self.assertRaises(ValueError):
                verify(
                    dict(training_work=[
                        dict(completed_batches=batches, current_epoch=epoch,
                             early_stopped=stopped)
                    ]), plan)

    def test_independent_cuda_ledger_must_match_reported_training_work(self):
        verify = self.function('verify_fit_ledger')
        report = dict(
            physical_fits=1, runs=[
                dict(training_work=[
                    dict(optimizer_steps=200, completed_batches=200)
                ])
            ])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save(root / 'fits.json', [
                dict(status='PASS', device='cuda:0', steps=199,
                     cuda_batches=199)
            ])
            save(root / 'cuda-batches.json', dict(training=199, prediction=1))
            with self.assertRaises(ValueError):
                verify(root, report)

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

    def source_bridge_case(self, root):
        """Metadata-only source review; this fixture does not prove a repair."""
        import wheel_gpu_learning as gpu

        row, directory = self.multiclass_case(root)
        helpers = root / 'producer/helpers'
        helpers.mkdir(parents=True)
        names = (*runpy.run_path(str(
            MODULES / 'wheel_task_learning.py'))['SOURCES'],
                 'wheel_gpu_learning.py', 'wheel_scale_data.py',
                 'wheel_scale_learning.py', 'wheel_measurements.py')
        for name in names:
            shutil.copyfile(MODULES / name, helpers / name)
        learning = helpers / 'wheel_scale_learning.py'
        learning.write_text(learning.read_text(
        ).replace('limit, 500,', 'limit,', 1).replace(
            "                            if len(truth['catalog']) > 500:\n"
            '                                raise ValueError(\n'
            "                                    'Eligible topK remains ambiguous at the '\n"
            "                                    'public 500-result limit')\n",
            '', 1))
        self.assertEqual(
            digest(learning),
            'c4a358ac40da70f7762498baaaaa1da910fc83694d9abce77df47ba712814099')
        before = helpers / 'wheel_scale_data.py'
        before.write_bytes(before.read_bytes() + b'\n# Frozen test producer.\n')
        files = {'helpers/' + name: digest(helpers / name) for name in names}
        save(root / 'producer/manifest.json', {'files': files})
        producer = runpy.run_path(str(helpers / 'wheel_gpu_learning.py'))
        report = json.loads((root / 'nightly-result.json').read_text())
        report.update(
            version='1.0.0', wheel_sha256=APPROVED_SHA,
            gpu_source_sha256=producer['source_identity'](prepared=True),
            profiles={'multiclass': {
                'fixed': 1
            }})
        report['configuration'].update(
            gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition',
            gpu_memory_bytes=96 * 1024**3, cuda='13.0', torch='2.14.1+cu130',
            threads=4)
        save(root / 'nightly-result.json', report)
        old, new = digest(before), digest(MODULES / 'wheel_scale_data.py')
        stages = {
            stage: dict(
                seeds=1, retained_positive_count=1, masked_retained_positives=0,
                native_exclusion_count=0, visible_fact_count=1,
                native_positives_sha256='1' * 64,
                native_exclusions_sha256='2' * 64,
                visible_fact_pk_sha256='3' * 64,
                pseudo_options=dict(pseudo_seed_time=cutoff,
                                    disable_pseudo_time=False))
            for stage, cutoff in [('train', 1), ('val', 1), ('test', 2)]
        }
        detailed = dict(
            status='PASS_TINY_PRODUCER_FREEZE_LOAD_NATIVE', helper_sha256=new,
            producer_freeze_load_consumption=True,
            old_v3_frozen_and_seals_unchanged=True,
            new_v4_frozen_and_seals_unchanged=True, engine_fit_calls=0,
            core_fit_calls=0, lightning_fit_calls=0, prediction_calls=0,
            final_seal_unchanged=True, owner_source_unchanged=True,
            producer_helpers_unchanged=True, arms={
                arm: dict(consumption=stages,
                          stage_sha256={stage: '4' * 64
                                        for stage in stages})
                for arm in ('normal', 'control0')
            })
        detailed['bridges'] = {
            arm: dict(old_stage_sha256={stage: '4' * 64
                                        for stage in stages},
                      new_stage_sha256={stage: '4' * 64
                                        for stage in stages},
                      final_rows_sha256='5' * 64, final_rows_bytes_equal=True,
                      identity_unchanged=True, requested_plan_unchanged=True)
            for arm in ('normal', 'control0')
        }
        detailed['frozen_descriptors'] = {
            arm: dict(files={'fact_pseudo_time.bin': '6' * 64})
            for arm in ('normal', 'control0')
        }
        save(root / 'review/native-result.json', detailed)
        source = {
            'wheel_scale_data.py': dict(sha256=new, bytes=1),
            'full_producer_probe.py': dict(sha256='e' * 64, bytes=1)
        }
        save(root / 'review/native-source.json', {'files': source})
        files = {'source/' + name: {**value} for name, value in source.items()}
        files['output/result.json'] = {
            'sha256': digest(root / 'review/native-result.json'),
            'bytes': (root / 'review/native-result.json').stat().st_size
        }
        save(root / 'review/native-files.json', {'files': files})
        proof = dict(
            format='nightly-static-v4-full-roundtrip-v1', status='PASS',
            old_data_helper_sha256=old, new_data_helper_sha256=new,
            full_roundtrip=True, old_v3_frozen_and_seals_unchanged=True,
            model_fits=0, predictions=0,
            detailed_result_sha256=digest(root / 'review/native-result.json'),
            source_manifest_sha256=digest(root / 'review/native-source.json'),
            file_manifest_sha256=digest(root / 'review/native-files.json'))
        save(root / 'review/native.json', proof)
        required = ('manifest.json', 'tt_ingest_spec.json', 'stats.json',
                    'split_stats.json', 'train_table_col_stats.json',
                    'tt.parquet/part-00000.parquet',
                    'train_seeds/part-00000.parquet',
                    'val_seeds/part-00000.parquet',
                    'test_seeds/part-00000.parquet')
        file_map = {name: 'a' * 64 for name in required}
        tasks = ('binary', 'multiclass', 'multilabel', 'multilabel_ranking',
                 'regression', 'forecasting')
        proof = dict(
            status='PASS_NONLINK_V3_EQUIVALENCE', before_helper_sha256=old,
            after_helper_sha256=new, source_archive_sha256='c' * 64, cases=[
                dict(task=task, condition=arm, descriptor_sha256='d' * 64,
                     manifest_version=3,
                     spec_schema='diskgraph.tt_ingest_spec.v3',
                     before_checker_passed=True, after_checker_passed=True,
                     before_files=file_map, after_files=file_map)
                for task in tasks for arm in ('normal', 'seed-0')
            ])
        save(root / 'review/v3.json', proof)
        review = dict(
            kind='reviewed_static_carrier_reference_source_v1',
            status='APPROVED_REFERENCE_ONLY', before_helper_sha256=old,
            after_helper_sha256=new, proofs={
                name: dict(path='review/' + name + '.json',
                           sha256=digest(root / 'review' / (name + '.json')))
                for name in ('native', 'v3')
            })
        review['proofs']['native'].update({
            name:
            dict(path='review/native-' + name + '.json',
                 sha256=digest(root / 'review' / ('native-' + name + '.json')))
            for name in ('result', 'source', 'files')
        })
        save(root / 'review/review.json', review)
        bundle = (directory /
                  'scratch/trainers/trainer/runs/learnable/published-model')
        descriptor = dict(
            path='nightly-result.json', reference_source_bridge=dict(
                review='review/review.json',
                producer_packet='producer/manifest.json',
                producer_helpers='producer/helpers',
                report_sha256=digest(root / 'nightly-result.json')), artifacts={
                    'multiclass-0-learnable':
                    dict(directory=directory.name,
                         bundle=bundle.relative_to(root).as_posix())
                })
        self.assertNotEqual(report['gpu_source_sha256'],
                            gpu.source_identity(prepared=True))
        return report, descriptor

    def test_learning_cap_review_rejects_other_source_changes(self):
        verify = self.function('verify_learning_cap_repair')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_bridge_case(root)
            before = root / 'producer/helpers/wheel_scale_learning.py'
            after = root / 'current.py'
            original = (MODULES / 'wheel_scale_learning.py').read_text()
            verify(before, MODULES / 'wheel_scale_learning.py')
            for old, new in (('limit, 500,', 'limit, 501,'),
                             ("if len(truth['catalog']) > 500:",
                              "if len(truth['catalog']) >= 500:"),
                             ("if values['fallback_ids']:",
                              'if False:'), ("truth['metric'].split('@')[1]",
                                             "truth['metric'].split('@')[0]"),
                             ("len(order) != wanted", "len(order) < wanted")):
                with self.subTest(change=new):
                    self.assertIn(old, original)
                    after.write_text(original.replace(old, new, 1))
                    with self.assertRaisesRegex(ValueError, 'cap repair'):
                        verify(before, after)

    def test_cap_only_link_bridge_requires_the_exact_sole_source_repair(self):
        verify = self.function('validate_reference_report')
        for change in (None, 'data', 'no_cap_repair'):
            with self.subTest(
                    change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report, descriptor = self.source_bridge_case(root)
                helpers = root / 'producer/helpers'
                shutil.copyfile(MODULES / 'wheel_scale_data.py',
                                helpers / 'wheel_scale_data.py')
                if change == 'data':
                    with (helpers / 'wheel_scale_data.py').open('a') as stream:
                        stream.write('\n# A separate unreviewed change.\n')
                if change == 'no_cap_repair':
                    shutil.copyfile(MODULES / 'wheel_scale_learning.py',
                                    helpers / 'wheel_scale_learning.py')
                packet = root / 'producer/manifest.json'
                save(
                    packet,
                    dict(
                        files={
                            'helpers/' + path.name: digest(path)
                            for path in helpers.glob('*.py')
                        }))
                producer = runpy.run_path(str(helpers /
                                              'wheel_gpu_learning.py'))
                fixtures = runpy.run_path(
                    str(
                        Path(__file__).with_name(
                            'test_engine_nightly_runtime.py')))
                row = fixtures['metadata_row']('static_link', 'learnable')
                expected = [['static_link', 0, 'learnable']]
                report.update(
                    runs=[row], expected_runs=expected, completed_runs=expected,
                    profiles={'static_link': {
                        'fixed': 1
                    }}, gpu_source_sha256=producer['source_identity'](
                        prepared=True))
                save(root / descriptor['path'], report)
                bridge = descriptor['reference_source_bridge']
                del bridge['review']
                bridge['report_sha256'] = digest(root / descriptor['path'])
                original = (root / descriptor['path']).read_bytes()
                if change is not None:
                    with self.assertRaisesRegex(ValueError, 'source review'):
                        verify(root, descriptor, report)
                    continue
                diagnostic = verify(root, descriptor, report)
                self.assertIn('learning_cap_repair', diagnostic)
                self.assertEqual((root / descriptor['path']).read_bytes(),
                                 original)
                with self.assertRaisesRegex(ValueError,
                                            'candidate.*source bridge'):
                    self.function('collect_reports')(root, [descriptor])

    def test_reviewed_reference_keeps_actual_producer_source_and_raw_bytes(
            self):
        verify = self.function('validate_reference_report')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report, descriptor = self.source_bridge_case(root)
            original = (root / descriptor['path']).read_bytes()
            with self.assertRaisesRegex(ValueError, 'source differs'):
                self.function('validate_report')(report)
            diagnostic = verify(root, descriptor, report)
            self.assertEqual(diagnostic['producer_gpu_source_sha256'],
                             report['gpu_source_sha256'])
            self.assertNotEqual(diagnostic['validator_gpu_source_sha256'],
                                report['gpu_source_sha256'])
            self.assertEqual(diagnostic['new_fits'], 0)
            self.assertEqual((root / descriptor['path']).read_bytes(), original)
            self.assertEqual(json.loads(original), report)

    def test_manual_canonical_builder_proof_cannot_approve_source_bridge(self):
        verify = self.function('validate_reference_report')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report, descriptor = self.source_bridge_case(root)
            manual = dict(status='PASS_NATIVE_SEMANTIC_DIAGNOSTIC',
                          engine_fit_calls=0, core_fit_calls=0,
                          lightning_fit_calls=0, prediction_calls=0,
                          final_seal_unchanged=True,
                          owner_source_unchanged=True)
            save(root / 'review/native.json', manual)
            review = json.loads((root / 'review/review.json').read_text())
            review['proofs']['native']['sha256'] = digest(root /
                                                          'review/native.json')
            save(root / 'review/review.json', review)
            with self.assertRaises(ValueError):
                verify(root, descriptor, report)

    def test_full_helper_proof_binds_source_stages_and_zero_model_work(self):
        verify = self.function('validate_reference_report')
        for change in ('helper', 'fit', 'freeze', 'stage', 'cutoff', 'mask',
                       'rows', 'plan', 'frozen_descriptor', 'result_binding',
                       'source_binding', 'file_map'):
            with self.subTest(
                    change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report, descriptor = self.source_bridge_case(root)
                review_path = root / 'review/review.json'
                review = json.loads(review_path.read_text())
                summary = json.loads((root / 'review/native.json').read_text())
                entry = review['proofs']['native']
                name = 'source' if change == 'source_binding' else (
                    'files' if change == 'file_map' else 'result')
                path = root / entry[name]['path']
                proof = json.loads(path.read_text())
                if change == 'source_binding':
                    proof['files']['wheel_scale_data.py']['sha256'] = '0' * 64
                elif change == 'file_map':
                    proof['files']['output/result.json']['sha256'] = '0' * 64
                elif change == 'helper':
                    proof['helper_sha256'] = '0' * 64
                elif change == 'fit':
                    proof['core_fit_calls'] = 1
                elif change == 'freeze':
                    proof['new_v4_frozen_and_seals_unchanged'] = False
                elif change == 'stage':
                    del proof['arms']['control0']['consumption']['val']
                elif change == 'cutoff':
                    proof['arms']['normal']['consumption']['test'][
                        'pseudo_options']['pseudo_seed_time'] = 3
                elif change == 'mask':
                    proof['arms']['normal']['consumption']['val'][
                        'masked_retained_positives'] = 1
                elif change == 'rows':
                    proof['bridges']['normal']['new_stage_sha256'][
                        'train'] = '7' * 64
                elif change == 'plan':
                    proof['bridges']['control0'][
                        'requested_plan_unchanged'] = False
                elif change == 'frozen_descriptor':
                    del proof['frozen_descriptors']['control0']
                else:
                    summary['detailed_result_sha256'] = '0' * 64
                save(path, proof)
                if change != 'result_binding':
                    entry[name]['sha256'] = digest(path)
                    field = {
                        'result': 'detailed_result_sha256',
                        'source': 'source_manifest_sha256',
                        'files': 'file_manifest_sha256'
                    }[name]
                    summary[field] = digest(path)
                    if name == 'result':
                        files_path = root / 'review/native-files.json'
                        files = json.loads(files_path.read_text())
                        files['files']['output/result.json']['sha256'] = digest(
                            path)
                        files['files']['output/result.json'][
                            'bytes'] = path.stat().st_size
                        save(files_path, files)
                        entry['files']['sha256'] = digest(files_path)
                        summary['file_manifest_sha256'] = digest(files_path)
                save(root / 'review/native.json', summary)
                entry['sha256'] = digest(root / 'review/native.json')
                save(review_path, review)
                with self.assertRaises(ValueError):
                    verify(root, descriptor, report)

    def test_reference_bridge_rejects_unreviewed_or_changed_evidence(self):
        verify = self.function('validate_reference_report')
        for change in ('after', 'before', 'missing_proof', 'failed_proof',
                       'v3_roster', 'v3_files', 'v3_archive', 'v3_descriptor',
                       'v3_missing_stage', 'producer', 'metric', 'report',
                       'link', 'failed', 'seed'):
            with self.subTest(
                    change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report, descriptor = self.source_bridge_case(root)
                review = json.loads((root / 'review/review.json').read_text())
                if change in ('after', 'before'):
                    review[change + '_helper_sha256'] = '0' * 64
                elif change == 'missing_proof':
                    (root / 'review/native.json').unlink()
                elif change.startswith('v3_') or change == 'failed_proof':
                    name = 'native' if change == 'failed_proof' else 'v3'
                    proof = json.loads(
                        (root / 'review' / (name + '.json')).read_text())
                    if change == 'failed_proof':
                        proof['status'] = 'FAIL'
                    elif change == 'v3_roster':
                        proof['cases'].pop()
                    elif change == 'v3_archive':
                        del proof['source_archive_sha256']
                    elif change == 'v3_descriptor':
                        del proof['cases'][0]['descriptor_sha256']
                    elif change == 'v3_missing_stage':
                        for field in ('before_files', 'after_files'):
                            del proof['cases'][0][field][
                                'test_seeds/part-00000.parquet']
                    else:
                        proof['cases'][0]['after_files'][
                            'stats.json'] = 'b' * 64
                    save(root / 'review' / (name + '.json'), proof)
                    review['proofs'][name]['sha256'] = digest(root / 'review' /
                                                              (name + '.json'))
                elif change in ('producer', 'metric'):
                    name = ('wheel_scale_data.py' if change == 'producer' else
                            'wheel_measurements.py')
                    path = root / 'producer/helpers' / name
                    path.write_bytes(path.read_bytes() + b'\n# changed\n')
                    if change == 'metric':
                        packet = json.loads(
                            (root / 'producer/manifest.json').read_text())
                        packet['files']['helpers/' + name] = digest(path)
                        save(root / 'producer/manifest.json', packet)
                else:
                    if change == 'report':
                        report['gpu_source_sha256'] = '0' * 64
                    elif change == 'link':
                        report['runs'][0]['task'] = 'static_link'
                    elif change == 'failed':
                        report['status'] = 'FAIL'
                    else:
                        report['runs'][0]['seed'] = False
                    save(root / descriptor['path'], report)
                    descriptor['reference_source_bridge'][
                        'report_sha256'] = digest(root / descriptor['path'])
                save(root / 'review/review.json', review)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    verify(root, descriptor, report)

    def test_candidate_rejects_reference_source_bridge_before_validation(self):
        collect = self.function('collect_reports')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, descriptor = self.source_bridge_case(root)
            with self.assertRaisesRegex(ValueError, 'candidate.*source bridge'):
                collect(root, [descriptor])

    def test_bridged_reference_still_checks_independent_ledger(self):
        collect = self.function('collect_reports')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, descriptor = self.source_bridge_case(root)
            save(
                root / 'fits.json',
                [dict(status='PASS', device='cuda:0', steps=1, cuda_batches=1)])
            with self.assertRaisesRegex(ValueError, 'Independent CUDA work'):
                collect(root, [descriptor], legacy_allowed=True)

    def test_legacy_source_review_keeps_original_row_and_validation_envelope(
            self):
        legacy_report = self.function('legacy_report')
        from wheel_task_checks import fingerprint

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report, descriptor = self.source_bridge_case(root)
            row = report['runs'][0]
            save(root / 'original-row.json', row)
            dependencies = {'torch': '2.14.1+cu130'}
            save(root / 'dependencies.json', dependencies)
            environment = dict(version='1.0.0', wheel_sha256=APPROVED_SHA,
                               gpu=report['configuration']['gpu'], cuda='13.0',
                               torch='2.14.1+cu130',
                               dependencies_sha256=fingerprint(dependencies))
            state = dict(
                status='PART_MEASURED', trial='R1', task='multiclass',
                condition='learnable', environment=environment,
                source_packet_sha256=digest(root / 'producer/manifest.json'),
                results=[
                    dict(seed=0, fit_id=row['fit_id'], score=row['score'],
                         baseline_score=row['baseline_score'],
                         result_sha256=digest(root / 'original-row.json'),
                         frozen_control=dict(descriptor_sha256='d' * 64))
                ])
            save(root / 'state.json', state)
            packet = json.loads((root / 'producer/manifest.json').read_text())
            model = dict(
                task='multiclass', environment=environment,
                frozen_normal_descriptor_sha256='d' * 64,
                selected_result={'sha256': digest(root / 'original-row.json')},
                source_state={'sha256': digest(root / 'state.json')},
                actual_launch_worker_packet={
                    'sha256': digest(root / 'producer/manifest.json')
                }, source_profile={'fixed': 1}, helper_comparison=[
                    dict(file=Path(name).name, source={'sha256': value},
                         current={'sha256': value})
                    for name, value in packet['files'].items()
                ])
            save(
                root / 'compatibility.json',
                dict(status='PASS_SELECTED_SOURCE_AND_PLAN_COMPATIBILITY',
                     models=[model]))
            descriptor.update(
                path='original-row.json', legacy=dict(
                    compatibility='compatibility.json',
                    producer_state='state.json',
                    producer_packet='producer/manifest.json',
                    dependencies='dependencies.json',
                    configuration=report['configuration'],
                    validator_source_sha256=report['gpu_source_sha256']))
            descriptor['reference_source_bridge']['report_sha256'] = digest(
                root / 'original-row.json')
            original = (root / 'original-row.json').read_bytes()
            envelope, diagnostic = legacy_report(root, descriptor)
            self.assertEqual(envelope['envelope_kind'],
                             'legacy_validation_only')
            self.assertEqual(envelope['runs'], [row])
            self.assertEqual(diagnostic['new_fits'], 0)
            self.assertEqual(
                diagnostic['producer_gpu_source_sha256_from_frozen_files'],
                report['gpu_source_sha256'])
            self.assertEqual((root / 'original-row.json').read_bytes(),
                             original)
            del descriptor['reference_source_bridge']
            with self.assertRaises(ValueError):
                legacy_report(root, descriptor)

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
        epochs = plan['optimization']['max_epochs']
        batches = epochs * plan['optimization']['max_steps_per_epoch']
        row['training_work'][0].update(completed_batches=batches,
                                       completed_examples=batches * 64,
                                       optimizer_steps=batches,
                                       current_epoch=epochs)
        row['training_window'].update(completed_batches=batches,
                                      completed_examples=batches * 64,
                                      optimizer_steps=batches,
                                      batch_examples=[64] * batches)
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
            dict(status='PASS', device='cuda:0', steps=batches,
                 cuda_batches=batches, metadata_test_only=True)
        ])
        save(root / 'cuda-batches.json', dict(training=batches, prediction=4))
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
