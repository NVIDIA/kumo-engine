# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CPU checks for failures that would invalidate a nightly result."""

import functools
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

SOURCE = Path(
    __file__).parents[2] / 'packages/kumo-engine/scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_check', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)
APPROVED = '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'


class NightlyRuntimeTest(unittest.TestCase):
    def test_resource_sampler_separates_pss_rss_and_device_usage(self):
        child = SimpleNamespace(
            memory_full_info=lambda: SimpleNamespace(rss=100, pss=40))
        parent = SimpleNamespace(
            children=lambda **_: [child],
            memory_full_info=lambda: SimpleNamespace(rss=100, pss=60))
        torch = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda: (80, 100), memory_allocated=lambda: 7,
            memory_reserved=lambda: 9))
        with patch('psutil.Process',
                   return_value=parent), patch.dict(sys.modules,
                                                    {'torch': torch}):
            measurements, finish = nightly.start_measurement('test-case')
            finish()
        self.assertEqual(measurements['cpu_tree_pss_sampled_peak_bytes'], 100)
        self.assertEqual(measurements['cpu_tree_rss_sampled_peak_bytes'], 200)
        self.assertEqual(measurements['cuda_device_used_sampled_peak_bytes'],
                         20)
        self.assertGreaterEqual(measurements['samples'], 2)
        self.assertEqual(measurements['torch_allocated_sampled_peak_bytes'], 7)
        self.assertEqual(measurements['torch_reserved_sampled_peak_bytes'], 9)
        self.assertIsNone(measurements['torch_peak_allocated_bytes'])

    def test_operation_measurement_finishes_after_failure_without_hiding_error(
            self):
        row, measurements, finished = {}, {
            'scope': 'operation',
            'elapsed_seconds': 1
        }, []
        with patch.object(
                nightly, 'start_measurement',
                return_value=(measurements, lambda: finished.append(True))):
            with self.assertRaisesRegex(ValueError, 'execution failed'):
                with nightly.measure_operation(row, 'training', 'fit'):
                    raise ValueError('execution failed')
        self.assertEqual(finished, [True])
        self.assertIs(row['operation_measurements']['training'], measurements)

    def test_resource_deltas_are_unavailable_without_matching_measurements(
            self):
        current = dict(format='kumo-nightly-resources-v1', scope='case',
                       sample_interval_seconds=.2, elapsed_seconds=12,
                       cpu_tree_pss_sampled_peak_bytes=150,
                       timings={'training': 8})
        previous = dict(current, elapsed_seconds=10,
                        cpu_tree_pss_sampled_peak_bytes=100,
                        timings={'training': 7})
        result = nightly.measurement_differences(current, previous)
        self.assertEqual(result['metrics']['elapsed_seconds']['difference'], 2)
        self.assertEqual(
            result['metrics']['cpu_tree_pss_sampled_peak_bytes']['difference'],
            50)
        self.assertEqual(result['metrics']['timings.training']['difference'], 1)
        for baseline in ({}, dict(previous, scope='whole-block'),
                         dict(previous, sample_interval_seconds=2)):
            result = nightly.measurement_differences(current, baseline)
            self.assertTrue(result['unavailable'])
            self.assertTrue(
                all(v['difference'] is None
                    for v in result['metrics'].values()))
            self.assertNotEqual(result['status'], 'PASS')

    def test_resource_read_failures_remain_unavailable(self):
        import psutil
        parent = SimpleNamespace(
            children=lambda **_: [],
            memory_full_info=lambda: SimpleNamespace(rss=10))

        def unavailable():
            raise RuntimeError('unavailable')

        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=unavailable))
        with patch.object(psutil, 'Process', return_value=parent), patch.dict(
                sys.modules, {'torch': torch}):
            values, finish = nightly.start_measurement('failed-read')
            finish()
        self.assertGreater(values['cpu_read_errors'], 0)
        self.assertGreater(values['cuda_read_errors'], 0)
        self.assertTrue(
            all(values[k] is None for k in nightly.RESOURCE_FIELDS[2:]))
        self.assertGreaterEqual(values['elapsed_seconds'], 0)

    def test_failed_hash_or_peak_read_preserves_cleanup_report_and_original_error(
            self):
        import torch
        digest = nightly.digest
        for fault in ('hash', 'allocated', 'reserved'):
            with self.subTest(
                    fault=fault), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'case'
                stopped = []

                def measurement(scope):
                    return dict.fromkeys(nightly.RESOURCE_FIELDS, None) | {
                        'elapsed_seconds': 1
                    }, lambda: stopped.append(scope)

                def fail_preparation(*_):
                    (output / 'raw.txt').write_text('retained failure')
                    raise ValueError('original preparation failure')
                with patch.object(nightly, 'start_measurement', side_effect=measurement), \
                        patch.object(nightly, 'load_table', side_effect=fail_preparation), \
                        patch.object(nightly, 'digest', side_effect=OSError('artifact hash failed') if fault == 'hash' else digest), \
                        patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                        patch.object(torch.cuda, 'max_memory_allocated', return_value=0, side_effect=RuntimeError('peak failed') if fault == 'allocated' else None), \
                        patch.object(torch.cuda, 'max_memory_reserved', return_value=0, side_effect=RuntimeError('peak failed') if fault == 'reserved' else None):
                    with self.assertRaisesRegex(ValueError,
                                                'original preparation failure'):
                        nightly.run_case(Path(directory), {'files': {}}, {
                            'task': 'binary',
                            'arm': 'normal'
                        }, None, None, output)
                self.assertEqual(len(stopped), 2)
                self.assertEqual(len(set(stopped)), 2)
                result = nightly.read(output / 'result.json')
                self.assertEqual(result['status'], 'FAIL')
                self.assertIn('original preparation failure', result['error'])
                if fault == 'hash':
                    self.assertIn('artifact hash failed',
                                  result['artifact_preservation_error'])
                else:
                    self.assertIn(
                        'peak failed',
                        result['measurements']['torch_peak_read_error'])
                    self.assertIsNone(
                        result['measurements']['torch_peak_allocated_bytes'])
                    self.assertIsNone(
                        result['measurements']['torch_peak_reserved_bytes'])

    def hardware(self):
        return dict(gpu='Tesla T4', gpu_count=1, gpu_memory_bytes=16 * 1024**3,
                    driver='595.91.07', architecture='x86_64',
                    compute_capability=[7,
                                        5], torch='2.14.1+cu130', cuda='13.0',
                    threads=4, gpu_uuid='11111111-2222-3333-4444-555555555555')

    def predecessor(self):
        cases, references, files = [], [], {}
        for task in nightly.smoke.TASKS:
            for arm in ('normal', 'control'):
                name = task + '-' + arm
                case = dict(task=task, arm=arm, seed=0,
                            descriptor=name + '/descriptor.json',
                            prediction_input=name + '/full.parquet',
                            partial_input=name + '/partial.parquet',
                            requested_plan_sha256='1' * 64,
                            resolved_plan_sha256='2' * 64,
                            selected_plan_sha256='2' * 64,
                            trainer_config_sha256='3' * 64)
                files.update({
                    case[k]: '4' * 64
                    for k in ('descriptor', 'prediction_input', 'partial_input')
                })
                cases.append(case)
                references.append(
                    dict(
                        task=task, arm=arm, seed=0,
                        metric=nightly.smoke.TASKS[task]['metric'], score=.25,
                        baseline=.5, fit_id='old-' + name,
                        checkpoint_sha256='5' * 64,
                        input_signature=nightly.input_signature(
                            dict(files=files), case), **{
                                k: case[k]
                                for k in ('resolved_plan_sha256',
                                          'selected_plan_sha256',
                                          'trainer_config_sha256')
                            }))
        reference = dict(
            format='kumo-engine-nightly-reference-v1', version='1.0.0',
            wheel_sha256=APPROVED, cases=references, reference_runtime=dict(
                reference_hardware={'gpu': 'NVIDIA RTX PRO 6000'}))
        return dict(cases=cases, files=files, gpu_model='T4'), reference

    def test_reference_preflight_rejects_missing_cases_inputs_and_recipes(self):
        manifest, reference = self.predecessor()
        nightly.reference_cases(reference, manifest)
        for changed in [
                dict(reference, cases=reference['cases'][:-1]),
                dict(reference,
                     cases=reference['cases'][:-1] + reference['cases'][:1]),
                dict(
                    reference, cases=[
                        dict(v, fit_id='reused') for v in reference['cases']
                    ])
        ]:
            with self.assertRaises(ValueError):
                nightly.reference_cases(changed, manifest)
        for field, value in [('input_signature', {}),
                             ('baseline', float('nan')),
                             ('resolved_plan_sha256', '0' * 64)]:
            changed = dict(
                reference, cases=[
                    dict(reference['cases'][0], **{field: value}),
                    *reference['cases'][1:]
                ])
            with self.subTest(field=field), self.assertRaises(ValueError):
                nightly.reference_cases(changed, manifest)

    def test_input_manifest_requires_t4_and_both_frozen_profiles(self):
        manifest, _ = self.predecessor()
        manifest.update(format='kumo-engine-nightly-inputs-v1', graphs={},
                        candidate_dependencies='candidate.json',
                        reference_dependencies='reference.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in [
                    *manifest['files'], 'candidate.json', 'reference.json',
                    'graphs.json'
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
                manifest['files'][name] = nightly.digest(path)
            for case in manifest['cases']:
                profile = nightly.smoke.TASKS[case['task']]['graph']
                static = case['task'] == 'static_link'
                case['graph'] = profile + ('-static' if static else '')
                case['native_expectations'] = dict(
                    key_failures={key: 0
                                  for key in nightly.KEY_FAILURES})
                manifest['graphs'][case['graph']] = dict(
                    profile=profile, static=static, schema='graphs.json',
                    sources={})
            nightly.save(root / 'manifest.json', manifest)
            nightly.load_inputs(root, nightly.digest(root / 'manifest.json'))
            nightly.save(root / 'manifest.json', dict(manifest,
                                                      gpu_model='RTX'))
            with self.assertRaises(ValueError):
                nightly.load_inputs(root,
                                    nightly.digest(root / 'manifest.json'))
            nightly.save(root / 'manifest.json', manifest)
            (root / 'reference.json').unlink()
            with self.assertRaises(ValueError):
                nightly.load_inputs(root,
                                    nightly.digest(root / 'manifest.json'))

    def test_candidate_rejects_rtx_reference_before_any_graph_or_fit(self):
        manifest, reference = self.predecessor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nightly.save(root / 'inputs/manifest.json', manifest)
            nightly.save(root / 'inputs/profile.json', {})
            manifest['candidate_dependencies'] = 'profile.json'
            manifest['reference_dependencies'] = 'profile.json'
            args = SimpleNamespace(input_root=root / 'inputs',
                                   output_dir=root / 'candidate',
                                   input_sha256='a' * 64,
                                   expected_version='candidate',
                                   reference_only=False,
                                   reference_sha256='b' * 64)
            with patch.object(nightly, 'load_inputs', return_value=manifest), \
                    patch.object(nightly, 'load_reference', return_value=reference), \
                    patch.object(nightly, 'verify_runtime', return_value=self.hardware()), \
                    patch.object(nightly, 'check_install'), \
                    patch.object(nightly.smoke, 'build_graph') as graph:
                with self.assertRaisesRegex(ValueError, 'hardware'):
                    nightly.run(args)
                graph.assert_not_called()

    def test_reference_only_compare_keeps_strict_checks_and_exports_fresh_fits(
            self):
        manifest, reference = self.predecessor()
        profile = {f'package-{i}': '1.0' for i in range(98)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {'GITHUB_SHA': 'b' * 40}):
            root = Path(directory)
            inputs, results, output = root / 'inputs', root / 'results', root / 'comparison'
            nightly.save(inputs / 'profile.json', profile)
            manifest.update(reference_dependencies='profile.json',
                            candidate_dependencies='profile.json')
            manifest['files']['profile.json'] = nightly.digest(inputs /
                                                               'profile.json')
            nightly.save(inputs / 'manifest.json', manifest)
            input_sha = nightly.digest(inputs / 'manifest.json')
            reference['input_manifest_sha256'] = input_sha
            nightly.save(results / 'reference.json', reference)
            runtime = dict(self.hardware(), dependencies=profile,
                           version='1.0.0',
                           wheel_sha256=reference['wheel_sha256'])
            nightly.save(
                results / 'setup/wheel.json',
                dict(version='1.0.0', sha256=runtime['wheel_sha256'],
                     installed_files=390))
            nightly.save(
                results / 'setup/build.json',
                dict(format='kumo-engine-published-reference-v1',
                     version='1.0.0', wheel_sha256=runtime['wheel_sha256'],
                     harness='b' * 40))
            nightly.save(results / 'input-metadata/manifest.json', manifest)
            rows = self.rows()
            for row, previous in zip(rows, reference['cases'], strict=True):
                row.update(
                    fit_id='fresh-' + row['fit_id'], score=.125,
                    metric=previous['metric'],
                    input_signature=previous['input_signature'],
                    checkpoint_artifact='checkpoint',
                    artifacts={'checkpoint': row['checkpoint_sha256']},
                    selected_checkpoint_steps=200, predictions={
                        stage: {
                            'files': [stage + '-raw.parquet']
                        }
                        for stage in nightly.PREDICTIONS
                    }, work=dict(
                        restored_tensors=3, reload_checks=1, core_fit_ledger=[
                            dict(status='PASS', steps=200, cuda_batches=200)
                        ], fit_ledger=[
                            dict(status='PASS', steps=200, cuda_batches=200)
                        ]), seconds=1, differences={})
                row['measurements'] = dict(format='kumo-nightly-resources-v1',
                                           scope='case',
                                           sample_interval_seconds=.2,
                                           elapsed_seconds=1,
                                           cpu_tree_pss_sampled_peak_bytes=100)
                row['timings'] = dict(preparation=.1, training=.5, predict=.1,
                                      repeat=.1, partial=.05, model_load=.05,
                                      reloaded_prediction=.1)
                row['measurements']['timings'] = row['timings']
                row['operation_measurements'] = {
                    'training': row['measurements']
                }
                nightly.save(
                    results / (row['task'] + '-' + row['arm']) / 'result.json',
                    row)
            engine_measurements = dict(rows[0]['measurements'], scope='engine')
            reference_sha = nightly.digest(results / 'reference.json')
            nightly.save(
                results / 'nightly-result.json',
                dict(format='kumo-engine-nightly-result-v1', status='MEASURED',
                     input_manifest_sha256=input_sha,
                     reference_sha256=reference_sha, expected_runs=16,
                     completed_runs=16, runs=rows, reference_only=True,
                     measurements=engine_measurements, runtime=runtime,
                     runner_sha256=nightly.digest(SOURCE),
                     helper_sha256=nightly.digest(nightly.HELPER)))
            args = SimpleNamespace(input_root=inputs, input_sha256=input_sha,
                                   results_dir=results,
                                   reference=results / 'reference.json',
                                   reference_sha256=reference_sha,
                                   output_dir=output, reference_only=True)
            with patch.object(nightly, 'verify_artifacts') as artifacts, \
                    patch.object(nightly, 'check_training') as training, \
                    patch.object(nightly, 'check_recipe') as recipes, \
                    patch.object(nightly, 'check_behavior', return_value=dict(
                        physical_fits=1, coverage={})) as behavior:
                nightly.compare(args)
                self.assertEqual(artifacts.call_count, 16)
                self.assertEqual(training.call_count, 16)
                self.assertEqual(recipes.call_count, 16)
                behavior.assert_called_once_with(results)
            exported = nightly.read(output / 'reference.json')
            self.assertEqual(exported['input_manifest_sha256'], input_sha)
            self.assertEqual(
                exported['reference_runtime']['reference_hardware'],
                nightly.hardware_profile(runtime))
            for fresh, original in zip(exported['cases'], reference['cases'],
                                       strict=True):
                self.assertEqual(fresh['baseline'], original['baseline'])
                self.assertEqual(fresh['score'], .125)
                self.assertEqual(
                    fresh['measurements']['cpu_tree_pss_sampled_peak_bytes'],
                    100)
                self.assertEqual(fresh['timings'], rows[0]['timings'])
                self.assertEqual(fresh['operation_measurements'],
                                 rows[0]['operation_measurements'])
                self.assertTrue(fresh['fit_id'].startswith('fresh-'))
                self.assertNotEqual(fresh['checkpoint_sha256'],
                                    original['checkpoint_sha256'])
            self.assertEqual(exported['measurements'], engine_measurements)
            measured = nightly.read(output / 'comparison.json')
            self.assertEqual(measured['performance']['status'], 'UNAVAILABLE')
            self.assertTrue(
                all(c['performance']['status'] == 'UNAVAILABLE'
                    for c in measured['cases']))
            self.assertEqual(nightly.read(results / 'reference.json'),
                             reference)
            self.assertTrue(
                nightly.read(output / 'comparison.json')['reference_only'])
            self.assertTrue(
                all(row['difference'] is None
                    for row in nightly.read(output /
                                            'comparison.json')['cases']))
            with patch.object(nightly, 'verify_artifacts'), \
                    patch.object(nightly, 'check_training'), \
                    patch.object(nightly, 'check_recipe'), \
                    patch.object(nightly, 'check_behavior', side_effect=ValueError('missing behavior')):
                with self.assertRaisesRegex(ValueError, 'missing behavior'):
                    nightly.compare(args)
            row = rows[0]
            row['work']['core_fit_ledger'][0]['status'] = 'FAIL'
            nightly.save(
                results / (row['task'] + '-' + row['arm']) / 'result.json', row)
            report = nightly.read(results / 'nightly-result.json')
            report['runs'] = rows
            nightly.save(results / 'nightly-result.json', report)
            with patch.object(nightly, 'verify_artifacts'), \
                    patch.object(nightly, 'check_training'), \
                    patch.object(nightly, 'check_recipe'):
                with self.assertRaisesRegex(ValueError, 'physical-fit ledger'):
                    nightly.compare(args)

            row['work']['core_fit_ledger'][0]['status'] = 'PASS'
            args.reference_only = False
            nightly.save(results / 'reference.json', exported)
            args.reference_sha256 = nightly.digest(results / 'reference.json')
            nightly.save(
                results / 'setup/build.json',
                dict(version='1.0.0', wheel_sha256=APPROVED, revision='a' * 40,
                     tree='b' * 40))
            report.update(reference_only=False,
                          reference_sha256=args.reference_sha256)
            report['measurements']['elapsed_seconds'] = 2
            for row in rows:
                row['measurements']['elapsed_seconds'] = 1.5
                row['measurements']['timings']['training'] = .6
                nightly.save(
                    results / (row['task'] + '-' + row['arm']) / 'result.json',
                    row)
            nightly.save(results / 'nightly-result.json', report)
            with patch.object(nightly, 'verify_artifacts'), \
                    patch.object(nightly, 'check_training'), \
                    patch.object(nightly, 'check_recipe'), \
                    patch.object(nightly, 'check_behavior', return_value=dict(physical_fits=1, coverage={})):
                nightly.compare(args)
            comparison = nightly.read(output / 'comparison.json')
            self.assertEqual(
                comparison['performance']['metrics']['elapsed_seconds']
                ['difference'], 1)
            for case in comparison['cases']:
                self.assertEqual(case['difference'], 0)
                self.assertEqual(
                    case['performance']['metrics']['elapsed_seconds']
                    ['difference'], .5)
                self.assertAlmostEqual(
                    case['performance']['metrics']['timings.training']
                    ['difference'], .1)
                self.assertEqual(
                    case['operations']['training']['metrics']['elapsed_seconds']
                    ['difference'], .5)
                self.assertIsNone(case['performance']['metrics']
                                  ['torch_peak_reserved_bytes']['difference'])

    def test_ephemeral_t4_runtime_records_actual_hardware_without_rtx_allowlist(
            self):
        profile = {f'package-{i}': '1.0' for i in range(98)}
        properties = SimpleNamespace(name='Tesla T4', total_memory=16 * 1024**3,
                                     uuid=self.hardware()['gpu_uuid'], major=7,
                                     minor=5)
        cuda = SimpleNamespace(is_available=lambda: True,
                               device_count=lambda: 1,
                               get_device_properties=lambda _: properties)
        torch = SimpleNamespace(cuda=cuda, __version__='2.14.1+cu130',
                                version=SimpleNamespace(cuda='13.0'),
                                set_num_threads=lambda _: None)
        engine = SimpleNamespace(__version__='candidate',
                                 __file__=str(Path(sys.prefix) / 'engine.py'))
        distribution = SimpleNamespace(read_text=lambda _: json.dumps(
            dict(archive_info=dict(hashes=dict(sha256='a' * 64)))))
        packages = [
            SimpleNamespace(metadata={'Name': k}, version=v)
            for k, v in profile.items()
        ]
        with patch.dict(sys.modules, {'torch': torch, 'kumo': SimpleNamespace(engine=engine), 'kumo.engine': engine}), \
                patch.object(nightly.sys, 'platform', 'linux'), \
                patch.object(nightly.sys, 'version_info', (3, 12)), \
                patch('platform.machine', return_value='x86_64'), \
                patch('subprocess.check_output', return_value='595.91.07\n') as driver, \
                patch.object(nightly.importlib.metadata, 'distributions', return_value=packages), \
                patch.object(nightly.importlib.metadata, 'distribution', return_value=distribution), \
                patch.object(nightly.importlib.metadata, 'version', return_value='candidate'):
            runtime = nightly.verify_runtime('candidate', profile)
            self.assertIn('--id=GPU-' + properties.uuid,
                          driver.call_args.args[0])
            for key, value in self.hardware().items():
                self.assertEqual(runtime[key], value)
            with tempfile.TemporaryDirectory() as directory:
                prefix = Path(directory) / 'environment'
                prefix.mkdir()
                alias = Path(directory) / 'environment-alias'
                alias.symlink_to(prefix, target_is_directory=True)
                original = engine.__file__
                with patch.object(nightly.sys, 'prefix', str(alias)):
                    engine.__file__ = str(prefix / 'engine.py')
                    nightly.verify_runtime('candidate', profile)
                    engine.__file__ = str(Path(directory) / 'outside.py')
                    with self.assertRaises(ValueError):
                        nightly.verify_runtime('candidate', profile)
                engine.__file__ = original
            properties.uuid = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
            nightly.verify_runtime('candidate', profile)
            for key, value in [('name', 'NVIDIA RTX PRO 6000'), ('uuid', 'bad'),
                               ('major', 8)]:
                before = getattr(properties, key)
                setattr(properties, key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    nightly.verify_runtime('candidate', profile)
                setattr(properties, key, before)
            with patch.object(cuda, 'device_count',
                              return_value=2), self.assertRaises(ValueError):
                nightly.verify_runtime('candidate', profile)
            with patch.object(
                    cuda, 'is_available',
                    return_value=False), self.assertRaises(ValueError):
                nightly.verify_runtime('candidate', profile)
            with patch('platform.machine',
                       return_value='aarch64'), self.assertRaises(ValueError):
                nightly.verify_runtime('candidate', profile)
            with patch('subprocess.check_output',
                       return_value=''), self.assertRaises(ValueError):
                nightly.verify_runtime('candidate', profile)

    def test_reference_hardware_matches_profile_but_not_physical_uuid(self):
        runtime = self.hardware()
        reference = dict(reference_runtime=dict(reference_hardware={
            k: v
            for k, v in runtime.items() if k != 'gpu_uuid'
        }))
        nightly.check_reference_runtime(reference, runtime, False)
        nightly.check_reference_runtime(
            reference,
            dict(runtime, gpu_uuid='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'),
            False)
        for key, value in [('gpu', 'NVIDIA RTX PRO 6000'), ('driver', '596.1'),
                           ('gpu_memory_bytes', 1), ('architecture', 'aarch64'),
                           ('compute_capability', [8, 0]), ('threads', 8)]:
            changed = dict(
                reference, reference_runtime=dict(reference_hardware=dict(
                    reference['reference_runtime']['reference_hardware'], **
                    {key: value})))
            with self.subTest(key=key), self.assertRaises(ValueError):
                nightly.check_reference_runtime(changed, runtime, False)
        nightly.check_reference_runtime({}, runtime, True)
        with self.assertRaises(ValueError):
            nightly.check_reference_runtime({}, dict(runtime, gpu='RTX'), True)

    def test_published_receipt_is_reference_only_and_bound_to_harness(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {'GITHUB_SHA': 'b' * 40}):
            root = Path(directory)
            runtime = dict(version='1.0.0', wheel_sha256=APPROVED)
            wheel = dict(runtime, sha256=runtime['wheel_sha256'],
                         installed_files=390)
            build = dict(format='kumo-engine-published-reference-v1', **runtime,
                         harness='b' * 40)
            nightly.save(root / 'setup/wheel.json', wheel)
            nightly.save(root / 'setup/build.json', build)
            nightly.check_install(root, runtime, reference_only=True)
            with self.assertRaises(ValueError):
                nightly.check_install(root, runtime)
            nightly.save(root / 'setup/build.json',
                         dict(build, harness='c' * 40))
            with self.assertRaises(ValueError):
                nightly.check_install(root, runtime, reference_only=True)
            nightly.save(root / 'setup/build.json', build)
            with self.assertRaises(ValueError):
                nightly.check_install(root, dict(runtime, version='candidate'),
                                      reference_only=True)

    def rows(self):
        return [
            dict(task=task, arm=arm, seed=0, status='MEASURED', physical_fits=1,
                 fit_id=f'{task}-{arm}', checkpoint_sha256='a' * 64,
                 prediction_batches={stage: 1
                                     for stage in nightly.PREDICTIONS})
            for task in nightly.smoke.TASKS for arm in ('normal', 'control')
        ]

    def test_one_report_requires_all_sixteen_cases_and_rejects_two_reports(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = dict(format='kumo-engine-nightly-result-v1',
                          status='MEASURED', input_manifest_sha256='f' * 64,
                          expected_runs=16, completed_runs=16, runs=self.rows())
            path = root / 'nightly-result.json'
            nightly.save(path, report)
            self.assertEqual(nightly.load_report(root, 'f' * 64),
                             (path, report))
            for bad in [
                    dict(report, completed_runs=15),
                    dict(report, runs=report['runs'][:-1]),
                    dict(report, shard='node'),
                    dict(report, status='FAIL'),
                    dict(
                        report, runs=[
                            dict(row, fit_id='reused') for row in report['runs']
                        ]),
                    dict(
                        report, runs=[
                            dict(row, status='FAIL') if i == 3 else row
                            for i, row in enumerate(report['runs'])
                        ])
            ]:
                nightly.save(path, bad)
                with self.assertRaises(ValueError):
                    nightly.load_report(root, 'f' * 64)
            nightly.save(path, report)
            nightly.save(root / 'old/nightly-result.json', report)
            with self.assertRaises(ValueError):
                nightly.load_report(root, 'f' * 64)
            path.unlink()
            (root / 'old/nightly-result.json').unlink()
            with self.assertRaises(ValueError):
                nightly.load_report(root, 'f' * 64)

    def test_run_cli_has_no_shard_selection(self):
        arguments = [
            'run', '--input-root', 'inputs', '--input-sha256', 'f' * 64,
            '--output-dir', 'result', '--expected-version', 'candidate',
            '--reference', 'reference.json', '--reference-sha256', 'e' * 64
        ]
        with patch.object(nightly, 'run') as run:
            nightly.main(arguments)
            self.assertFalse(hasattr(run.call_args.args[0], 'shard'))
        with patch.object(
                sys, 'stderr',
                io.StringIO()), self.assertRaises(SystemExit) as error:
            nightly.main(arguments + ['--shard', 'node'])
        self.assertEqual(error.exception.code, 2)

    def test_run_executes_all_manifest_cases_once_in_order_without_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'inputs'
            nightly.save(inputs / 'manifest.json', {})
            nightly.save(inputs / 'profile.json', {})
            nightly.save(inputs / 'graphs.json', {'profile': {}})
            cases = [
                dict(task=row['task'], arm=row['arm'], graph='graph')
                for row in self.rows()
            ]
            manifest = dict(
                cases=cases, candidate_dependencies='profile.json',
                reference_dependencies='profile.json', allowed_gpu_uuids=[],
                graphs={
                    'graph':
                    dict(schema='graphs.json', profile='profile', static=False,
                         sources={})
                })
            rows = {(row['task'], row['arm']): row for row in self.rows()}
            observed = []

            def case_run(_root, _manifest, case, *_args):
                observed.append((case['task'], case['arm']))
                return rows[observed[-1]]

            engine = SimpleNamespace(configure=lambda **_: None)
            args = SimpleNamespace(input_root=inputs,
                                   output_dir=root / 'result',
                                   input_sha256='f' * 64,
                                   expected_version='candidate',
                                   reference_only=False,
                                   reference_sha256='e' * 64)
            with patch.dict(sys.modules, {'kumo': SimpleNamespace(engine=engine), 'kumo.engine': engine}), \
                    patch.object(nightly, 'load_inputs', return_value=manifest), \
                    patch.object(nightly, 'load_reference', return_value={}), \
                    patch.object(nightly, 'reference_cases'), \
                    patch.object(nightly, 'check_reference_runtime'), \
                    patch.object(nightly, 'verify_runtime', return_value={}), \
                    patch.object(nightly, 'check_install'), \
                    patch.object(nightly, 'start_measurement', side_effect=lambda scope: (
                        dict(scope=scope, elapsed_seconds=0), lambda: None)), \
                    patch.object(nightly, 'check_snapshot', return_value={}), \
                    patch.object(nightly.smoke, 'build_graph', return_value=('graph', 'snapshot')), \
                    patch.object(nightly, 'run_case', side_effect=case_run):
                nightly.run(args)
            result = nightly.read(args.output_dir / 'nightly-result.json')
            self.assertEqual(observed, [(c['task'], c['arm']) for c in cases])
            self.assertEqual(result['expected_runs'], 16)
            self.assertEqual(result['completed_runs'], 16)
            self.assertNotIn('shard', result)
            nightly.check_results(result['runs'])

    def test_declared_runtime_profile_does_not_require_pytest(self):
        profile = {f'package-{i}': '1.0' for i in range(98)}
        nightly.check_dependencies(profile, profile)
        with self.assertRaises(ValueError):
            nightly.check_dependencies(dict(profile, pytest='8.0'), profile)
        with self.assertRaises(ValueError):
            nightly.check_dependencies({'torch': '2.14.1'}, {'torch': '2.14.1'})

    def test_frozen_files_reject_escape_symlink_and_modified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rows').write_bytes(b'original')
            files = {'rows': nightly.digest(root / 'rows')}
            self.assertEqual(nightly.safe_file(root, 'rows', files),
                             root / 'rows')
            for name in ('../rows', '/rows', './rows'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    nightly.safe_file(root, name, files)
            (root / 'link').symlink_to(root / 'rows')
            with self.assertRaises(ValueError):
                nightly.safe_file(root, 'link', {'link': files['rows']})
            (root / 'rows').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                nightly.safe_file(root, 'rows', files)

    def test_ranked_boundary_and_exclusions_across_batches(self):
        frame = pd.DataFrame(
            dict(ENTITY=[1] * 4 + [2] * 4, CLASS=[10, 11, 12, 13] * 2,
                 SCORE=[.9, .8, .7, .1, .9, .8, .7, .1]))
        result = nightly.ranked_rows(
            [frame.iloc[:3], frame.iloc[3:6], frame.iloc[6:]], [1, 2],
            [10, 11, 12, 13, 14], [[10], []], 2, 4)
        self.assertEqual(result['ranked_ids'], [[11, 12], [10, 11]])
        self.assertEqual(result['raw_rows'], 8)
        for bad in (frame.iloc[:-1], frame.assign(SCORE=.5),
                    frame.assign(CLASS=[10] * 8),
                    frame.assign(ENTITY=frame.ENTITY.astype(float))):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                nightly.ranked_rows([bad], [1, 2], [10, 11, 12, 13, 14],
                                    [[10], []], 2, 4)

    def test_no_full_catalog_retry_when_five_hundred_is_insufficient(self):
        frame = pd.DataFrame(
            dict(ENTITY=[1] * 500, CLASS=list(range(500)),
                 SCORE=list(range(500, 0, -1))))
        with self.assertRaises(ValueError):
            nightly.ranked_rows([frame], [1], list(range(600)),
                                [list(range(499))], 5, 500)

    def test_reported_recall_uses_all_positives_not_only_k(self):
        truth = pd.DataFrame({'TARGET1': [[1, 2, 3, 4, 5, 6]]})
        score = nightly.quality('static_link', truth,
                                {'ranked_ids': [[1, 2, 3, 4, 5]]}, [])
        self.assertAlmostEqual(score, 5 / 6)

    def test_training_requires_real_fixed_work_and_valid_early_stop(self):
        state = dict(
            fits=1, completed_fits=1, core_fit_attempts=1, training_batches=80,
            optimizer_steps=80, training_epochs=[
                dict(epoch=i, batches=20, batch_sizes=[64] * 20)
                for i in range(4)
            ], early_stopping=dict(stopped_epoch=3, wait_count=3, patience=3))
        nightly.check_training(state, 40)
        nightly.check_training(dict(state, optimizer_steps=160), 120)
        nightly.check_training(
            dict(
                state, training_batches=200, optimizer_steps=400,
                training_epochs=[
                    dict(epoch=i, batches=20, batch_sizes=[64] * 20)
                    for i in range(10)
                ]), 400)
        for key, value in [('core_fit_attempts', 2), ('training_batches', 79),
                           ('early_stopping', {})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                nightly.check_training(dict(state, **{key: value}), 40)
        state['training_epochs'][0]['batch_sizes'][0] = 32
        with self.assertRaises(ValueError):
            nightly.check_training(state, 40)

    def test_effective_recipe_changes_are_not_hidden_by_requested_plan(self):
        config = dict(
            root_dir='/old', version=0, kumo_version='1.0.0',
            refit_cfg=dict(best_val_epoch=3, refitted_full=False,
                           refitted_trainval=False),
            loader_kwargs={'batch_size':
                           64}, sampler_kwargs={'adaptive_sampling':
                                                False}, loss='mae')
        self.assertEqual(
            nightly.stable_trainer(config),
            dict(loader_kwargs={'batch_size': 64},
                 sampler_kwargs={'adaptive_sampling': False}, loss='mae'))
        changed = dict(config, loss='mse')
        self.assertNotEqual(nightly.signature(nightly.stable_trainer(config)),
                            nightly.signature(nightly.stable_trainer(changed)))

    def test_core_retry_is_blocked_before_a_second_fit_and_eval_does_not_retry(
            self):
        calls = []

        def retry(function):
            @functools.wraps(function)
            def wrapped(*args):
                try:
                    return function(*args)
                except RuntimeError:
                    return function(*args)

            return wrapped

        class Core:
            def fit(self):
                calls.append('fit')

            @retry
            def predict_v2(self):
                calls.append('predict')
                raise RuntimeError('out of memory')

            test = predict_v2

        class Data:
            def _pseudo_time_options(self, stage):
                return {}

        modules = {
            'diskgraph':
            SimpleNamespace(execute_tt_ingest=lambda: None),
            'kumo_core.train.trainer':
            SimpleNamespace(Trainer=Core),
            'kumo_core.diskgraph.lightning_data':
            SimpleNamespace(DiskgraphLightningDataModule=Data)
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, modules):
            state = dict(core_fit_attempts=0, training_batches=20,
                         optimizer_steps=40,
                         ledger_path=str(Path(directory) / 'core.json'))
            with nightly.native_calls(state):
                core = Core()
                core.fit()
                with self.assertRaises(ValueError):
                    core.fit()
                with self.assertRaises(RuntimeError):
                    core.predict_v2()
            self.assertEqual(calls, ['fit', 'predict'])
            self.assertEqual(len(state['core_fit_ledger']), 2)

    def test_partial_inputs_are_the_same_rows_and_a_proper_subset(self):
        truth = pd.DataFrame(
            dict(ENTITY1=[2, 1],
                 TIMESTAMP=pd.to_datetime(['2024-01-01',
                                           '2024-01-01']), TARGET1=[1, 0]))
        inputs = truth[['ENTITY1', 'TIMESTAMP']]
        self.assertEqual(
            nightly.check_inputs(inputs.iloc[:1], truth, partial=True), [2])
        for bad in [
                inputs,
                inputs.assign(ENTITY1=[3, 4]),
                inputs.assign(TIMESTAMP=pd.Timestamp('2024-01-02'))
        ]:
            with self.assertRaises(ValueError):
                nightly.check_inputs(bad, truth, partial=True)
        with self.assertRaises(ValueError):
            nightly.check_inputs(inputs[['ENTITY1']], truth)

    def test_missing_or_changed_retained_artifact_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'raw.parquet').write_bytes(b'raw')
            artifacts = {'raw.parquet': nightly.digest(root / 'raw.parquet')}
            nightly.verify_artifacts(root, artifacts)
            with self.assertRaises(ValueError):
                nightly.verify_artifacts(root, artifacts,
                                         required=['missing-output.parquet'])
            (root / 'raw.parquet').unlink()
            with self.assertRaises(ValueError):
                nightly.verify_artifacts(root, artifacts)

    def test_behavior_needs_actual_child_work_positive_fits_and_retained_logs(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            behavior = root / 'behavior'
            child_path = behavior / 'pytest-temp/case/child-result.json'
            nightly.save(
                child_path,
                dict(fits=0, fit_attempts=0, state_checks=2, restored_tensors=3,
                     prediction_batches={'reuse': 1}, rows=1071))
            for name in ('child.stdout', 'child.stderr', 'request.json'):
                child_path.with_name(name).write_text('{}')
            (behavior / 'junit.xml').write_text(
                '<testsuites><testsuite><testcase name="a"/></testsuite></testsuites>'
            )
            artifacts = {
                str(p.relative_to(behavior)): nightly.digest(p)
                for p in behavior.rglob('*') if p.is_file()
            }
            result = dict(
                status='PASS', coverage={
                    k: True
                    for k in ('I2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8')
                }, pytest_exit_code=0, bad=[], deselected=0,
                collection_errors=[], collected=['test.py::a'],
                calls={'test.py::a': 'passed'}, physical_fits=1,
                fit_ledger=[dict(status='PASS', steps=2,
                                 cuda_batches=1)], artifacts=artifacts,
                b7_receipt=dict(path=str(child_path.relative_to(behavior)),
                                sha256=nightly.digest(child_path)))
            nightly.save(behavior / 'result.json', result)
            nightly.check_behavior(root)
            nightly.save(behavior / 'result.json',
                         dict(result, physical_fits=0, fit_ledger=[]))
            with self.assertRaises(ValueError):
                nightly.check_behavior(root)
            nightly.save(behavior / 'result.json', result)
            child = json.loads(child_path.read_text())
            nightly.save(child_path, dict(child, restored_tensors=0))
            result['artifacts'][str(
                child_path.relative_to(behavior))] = nightly.digest(child_path)
            result['b7_receipt']['sha256'] = nightly.digest(child_path)
            nightly.save(behavior / 'result.json', result)
            with self.assertRaises(ValueError):
                nightly.check_behavior(root)
            child_path.with_name('child.stdout').unlink()
            with self.assertRaises(ValueError):
                nightly.check_behavior(root)

    def test_complete_roster_rejects_reused_fit_and_partial_result(self):
        rows = self.rows()
        nightly.check_results(rows)
        invalid = [
            rows[:-1], rows + [rows[0]],
            [dict(row, seed=False) for row in rows],
            [dict(row, fit_id=rows[0]['fit_id']) for row in rows],
            [
                dict(row, status='FAIL') if i == 3 else row
                for i, row in enumerate(rows)
            ]
        ]
        for bad in invalid:
            with self.assertRaises(ValueError):
                nightly.check_results(bad)


if __name__ == '__main__':
    unittest.main()
