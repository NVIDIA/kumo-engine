# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete or altered weekly scope before model execution."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/scripts/weekly_check.py'


def hardware():
    return dict(instance_id='i-0123456789abcdef0',
                gpu_uuid='GPU-00000000-0000-4000-8000-000000000001',
                driver='595.91.07', vcpus=16, memory_total_bytes=133971701760,
                gpu_memory_bytes=96 * 2**30)


def manifest():
    profiles = [
        ('Q13', 'streaming/100M', 'next_titles', 'recall@50'),
        ('Q14', 'community/100M', 'messages_30d', 'mae'),
        ('Q15', 'payments/100M core', 'flagged', 'auroc'),
        ('Q16', 'fraud/100M', 'account_flagged', 'auroc'),
        ('Q17', 'commerce/1B', 'next_merchants', 'recall@10'),
    ]
    rows = [
        dict(id=f'{name}-{arm}', workload=name, dataset=data, query=query,
             metric=metric, seed=0, condition=condition,
             hardware_profile='RTX_PRO_6000_BLACKWELL_96GB')
        for name, data, query, metric in profiles for arm, condition in [(
            'normal', 'normal'), ('control', 'scrambled_answer_control')]
    ]
    rows.append(
        dict(id='P2-fixed', workload='P2', dataset='commerce/25B',
             query='next_merchants', metric='recall@10', seed=0,
             condition='fixed_scale_workload', hardware_profile='Scale GPU'))
    return dict(
        schema=1, reference=dict(version='1.0.0', wheel_sha256='1' * 64),
        performance_reference=dict(version='1.0.0+group5.20261023.1',
                                   status='missing_or_incompatible'),
        cases=rows)


def invoke(tmp_path, value, *args, pin=None):
    source = tmp_path / 'manifest.json'
    source.write_text(json.dumps(value))
    digest = hashlib.sha256(
        source.read_bytes()).hexdigest() if pin is None else pin
    return subprocess.run([
        sys.executable, '-I',
        str(SCRIPT), 'inspect', '--manifest',
        str(source), '--manifest-sha256', digest, *args
    ], capture_output=True, text=True, check=False)


def test_inspect_preserves_all_eleven_cases_without_starting_fits(tmp_path):
    result = invoke(tmp_path, manifest())
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['case_ids'] == [
        'Q13-normal', 'Q13-control', 'Q14-normal', 'Q14-control', 'Q15-normal',
        'Q15-control', 'Q16-normal', 'Q16-control', 'Q17-normal', 'Q17-control',
        'P2-fixed'
    ]
    assert report['physical_fits_started'] == 0
    assert list(tmp_path.iterdir()) == [tmp_path / 'manifest.json']


@pytest.mark.parametrize(
    'change',
    ['missing', 'extra', 'full_payments', 'candidate_reference', 'duplicate'])
def test_inspect_refuses_changed_or_incomplete_approved_roster(
        tmp_path, change):
    value = manifest()
    if change == 'missing':
        value['cases'].pop()
    elif change == 'extra':
        value['cases'].append(dict(value['cases'][0], id='Q13-repeat'))
    elif change == 'full_payments':
        value['cases'][4]['dataset'] = 'payments/100M full'
    elif change == 'candidate_reference':
        value['reference']['version'] = '1.0.1'
    else:
        value['cases'][1] = value['cases'][0]
    result = invoke(tmp_path, value)
    assert result.returncode != 0


def test_inspect_refuses_unpinned_manifest_bytes(tmp_path):
    result = invoke(tmp_path, manifest(), pin='2' * 64)
    assert result.returncode != 0
    assert 'fingerprint' in result.stderr.lower()


def weekly():
    import runpy
    return runpy.run_path(str(SCRIPT))


def scalar_result(spec):
    receipt = dict(status='PASS', checkpoint_sha256='a' * 64,
                   checkpoint_path='model/checkpoint',
                   restore_path='lightning_strategy.load_model_state_dict',
                   tensor_count=1)
    work = dict(status='COMPLETE', early_stopped=False, completed_batches=2,
                completed_examples=4, optimizer_steps=2)
    return dict(
        status='MEASURED', identity=spec['identity'], seed=0, training_seed=0,
        case='learnable', fit_id='fit-unique', metric='mae', score=1.0,
        baseline_score=1.0, entity_ids=[1, 2], targets=[0, 2],
        predictions=[[1], [3]], baseline_predictions=[[1], [1]], checks={
            key: 'PASS'
            for key in ('fit', 'prediction', 'repeat_prediction',
                        'partial_prediction', 'reload', 'hidden_answers',
                        'chronology')
        }, device='cuda', observed_devices=[
            dict(device='cuda:0', stage=stage)
            for stage in ('training', 'prediction', 'reload_and_predict')
        ], batch_retries_disabled=True, training_work=[work],
        training_window=dict(work, batch_examples=[2, 2]),
        selected_checkpoint_global_step=2,
        expected_reload_checkpoint_sha256='a' * 64, reload_identity_checks=[
            receipt
        ], stage_sha256=dict(train='b' * 64, val='c' * 64, test='d' * 64),
        data_sha256='e' * 64, plan_sha256='f' * 64, operation_measurements=[
            dict(
                stage='training', seconds=3.0,
                resources=dict(peak_cpu_rss_bytes=100,
                               peak_gpu_allocated_bytes=100))
        ])


def ready_case(tmp_path):
    spec = manifest()['cases'][2]
    spec['identity'] = {
        name: str(index % 10) * 64
        for index, name in enumerate(('source_sha256', 'data_sha256',
                                      'query_sha256', 'split_sha256',
                                      'model_sha256', 'control_sha256',
                                      'dependency_sha256',
                                      'device_profile_sha256'))
    }
    worker = tmp_path / 'worker.py'
    worker.write_text(
        'def run_case(spec, output):\n    raise RuntimeError("No model in this test")\n'
    )
    input_path = tmp_path / 'frozen.json'
    input_path.write_text('{}')
    capacity = tmp_path / 'capacity.json'
    capacity.write_text(
        json.dumps(
            dict(status='ADMITTED', case_id=spec['id'],
                 hardware_profile=spec['hardware_profile'],
                 native_capacity_checked=True,
                 gpu_name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                 observed_hardware=hardware())))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    spec.update(worker=dict(path=str(worker), sha256=sha(worker)),
                inputs={str(input_path): sha(input_path)},
                capacity=dict(path=str(capacity), sha256=sha(capacity)))
    return spec


def test_ready_case_requires_unchanged_inputs_and_native_capacity(tmp_path):
    spec = ready_case(tmp_path)
    api = weekly()
    assert api['verify_case_inputs'](spec)['case_id'] == 'Q14-normal'
    Path(next(iter(spec['inputs']))).write_text('{"changed":true}')
    with pytest.raises(ValueError, match='fingerprint'):
        api['verify_case_inputs'](spec)


def test_ready_case_refuses_software_only_capacity_receipt(tmp_path):
    spec = ready_case(tmp_path)
    capacity = Path(spec['capacity']['path'])
    value = json.loads(capacity.read_text())
    value['native_capacity_checked'] = False
    capacity.write_text(json.dumps(value))
    spec['capacity']['sha256'] = hashlib.sha256(
        capacity.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='capacity'):
        weekly()['verify_case_inputs'](spec)


def test_result_accepts_finite_complete_work_without_quality_threshold(
        tmp_path):
    spec = ready_case(tmp_path)
    row = scalar_result(spec)
    row.update(score=100.0, predictions=[[100], [102]])
    weekly()['verify_result'](spec, row)


@pytest.mark.parametrize('change', [
    'identity', 'nonfinite', 'population', 'cuda', 'reload', 'work', 'score',
    'control', 'batch_retry', 'missing_check'
])
def test_result_rejects_incomplete_or_mismatched_evidence(tmp_path, change):
    spec = ready_case(tmp_path)
    row = scalar_result(spec)
    if change == 'identity':
        row['identity'] = dict(row['identity'], data_sha256='9' * 64)
    elif change == 'nonfinite':
        row['predictions'][0][0] = float('nan')
    elif change == 'population':
        row['entity_ids'] = [1, 1]
    elif change == 'cuda':
        row['observed_devices'][0]['device'] = 'cpu'
    elif change == 'reload':
        row['reload_identity_checks'][0]['checkpoint_sha256'] = '9' * 64
    elif change == 'work':
        row['training_window']['completed_examples'] = 3
    elif change == 'score':
        row['score'] = 0.0
    elif change == 'control':
        spec['condition'] = 'scrambled_answer_control'
    elif change == 'batch_retry':
        row['batch_retries_disabled'] = False
    else:
        del row['checks']['hidden_answers']
    with pytest.raises(ValueError):
        weekly()['verify_result'](spec, row)


def test_busy_host_fails_without_starting_or_hiding_a_physical_attempt(
        tmp_path):
    import fcntl
    value = manifest()
    value['cases'][2] = ready_case(tmp_path)
    source = tmp_path / 'manifest.json'
    source.write_text(json.dumps(value))
    lock_path = tmp_path / 'host.lock'
    output = tmp_path / 'attempt'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([
            sys.executable, '-I',
            str(SCRIPT), 'run', '--manifest',
            str(source), '--manifest-sha256',
            hashlib.sha256(source.read_bytes()).hexdigest(), '--case',
            'Q14-normal', '--mode', 'reference', '--wheel',
            str(tmp_path / 'unused.whl'), '--version', '1.0.0',
            '--wheel-sha256', '1' * 64, '--expected-prefix',
            str(tmp_path), '--dependency-profile',
            str(tmp_path / 'dependencies.json'), '--host-lock',
            str(lock_path), '--output',
            str(output)
        ], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    report = json.loads((output / 'weekly-case.json').read_text())
    assert report['status'] == 'FAIL'
    assert report['physical_fits'] == 0
    assert 'busy' in report['error'].lower()
    assert json.loads((output / 'fits.json').read_text()) == []


def evidence_tree(root, spec, *, mode='reference', wheel='1' * 64,
                  fit_suffix='', frozen_manifest=None):
    root.mkdir(parents=True)
    row = scalar_result(spec)
    row['metric'] = spec['metric']
    row['fit_id'] = spec['id'] + fit_suffix
    row['case'] = 'independent' if spec[
        'condition'] == 'scrambled_answer_control' else 'learnable'
    if spec['metric'] == 'auroc':
        row.update(targets=[0, 1], predictions=[[0.2], [0.8]],
                   baseline_predictions=[[0.5],
                                         [0.5]], score=1.0, baseline_score=0.5)
    elif spec['metric'].startswith('recall@'):
        row.update(target_format='ranked_link_v1', catalog=[1, 2],
                   target_ids=[[1], [2]], ranked_ids=[[1, 2], [1, 2]],
                   ranked_scores=[[0.9, 0.8],
                                  [0.9, 0.8]], baseline_ranked_ids=[[1, 2],
                                                                    [1, 2]],
                   baseline_ranked_scores=[[0.9, 0.8], [0.9, 0.8]], score=1.0,
                   baseline_score=1.0)
    if spec['workload'] == 'P2':
        row.update(
            score=None, baseline_score=None, quality_missing_reason=(
                'P2 fixed performance workload has no frozen quality answers'))
        for key in ('target_ids', 'targets', 'predictions',
                    'baseline_predictions', 'baseline_ranked_ids',
                    'baseline_ranked_scores'):
            row.pop(key, None)
    (root / 'result.json').write_text(json.dumps(row))
    (root / 'fits.json').write_text(
        json.dumps(
            [dict(status='PASS', device='cuda:0', steps=2, cuda_batches=2)]))
    (root / 'cuda-batches.json').write_text(
        json.dumps(dict(training=2, prediction=4)))
    for name in ('requested-plan', 'selected-model-plan', 'generation-plan',
                 'resolved_model_plan', 'trainer_config', 'runtime_manifest',
                 'feature_schema', 'device', 'measurements'):
        (root / f'{name}.json').write_text('{}')
    (root / 'memory.json').write_text('{"overall":{"samples":1}}')
    (root / 'model').mkdir()
    (root / 'model/checkpoint').write_bytes(b'fixture checkpoint')
    for label in ('predictions', 'repeat-predictions', 'partial-predictions',
                  'reloaded-predictions'):
        (root / f'{label}.parquet').write_bytes(b'fixture export')
    capacity = dict(status='ADMITTED', case_id=spec['id'],
                    hardware_profile=spec['hardware_profile'],
                    native_capacity_checked=True,
                    gpu_name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                    observed_hardware=hardware())
    (root / 'native-capacity.json').write_text(json.dumps(capacity))
    spec['capacity'] = dict(
        path='/native/admitted/' + spec['id'] + '.json', sha256=hashlib.sha256(
            (root / 'native-capacity.json').read_bytes()).hexdigest())
    original = manifest() if frozen_manifest is None else frozen_manifest
    original = json.loads(json.dumps(original))
    original['cases'] = [
        spec if item['id'] == spec['id'] else item for item in original['cases']
    ]
    (root / 'input-manifest.json').write_text(json.dumps(original))
    original_sha = hashlib.sha256(
        (root / 'input-manifest.json').read_bytes()).hexdigest()
    report = dict(
        schema=1, status='MEASURED', case_id=spec['id'], mode=mode,
        manifest_sha256=original_sha, observed_hardware=hardware(),
        version='1.0.0' if mode == 'reference' else '1.0.1', wheel_sha256=wheel,
        identity=spec['identity'], hardware_profile=spec['hardware_profile'],
        case_spec=spec, physical_fits=1, result=row,
        device=dict(gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                    gpu_uuid='GPU-00000000-0000-4000-8000-000000000001',
                    gpu_memory_bytes=96 * 2**30))
    if mode == 'candidate':
        report['build'] = dict(revision='a' * 40, tree='b' * 40,
                               bundle_version=report['version'],
                               version=report['version'], wheel='candidate.whl',
                               wheel_sha256=wheel)
        (root / 'build.json').write_text(json.dumps(report['build']))
    report['files'] = {
        path.relative_to(root).as_posix():
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()
    }
    (root / 'weekly-case.json').write_text(json.dumps(report))
    return root / 'weekly-case.json'


def complete_evidence(tmp_path, mode='reference', wheel='1' * 64,
                      fit_suffix=''):
    value = manifest()
    for spec in value['cases']:
        spec['identity'] = {
            key: '0' * 64
            for key in ('source_sha256', 'data_sha256', 'query_sha256',
                        'split_sha256', 'model_sha256', 'control_sha256',
                        'dependency_sha256', 'device_profile_sha256')
        }
    paths = [
        evidence_tree(tmp_path / 'cases' / spec['id'], spec, mode=mode,
                      wheel=wheel, fit_suffix=fit_suffix, frozen_manifest=value)
        for spec in value['cases']
    ]
    return value, paths


def test_collect_refuses_ten_cases_as_a_complete_reference(tmp_path):
    value, paths = complete_evidence(tmp_path)
    with pytest.raises(ValueError, match='eleven'):
        weekly()['collect'](value, paths[:-1], tmp_path / 'bundle', 'reference')


def test_collect_preserves_complete_models_and_physical_fit_receipts(tmp_path):
    value, paths = complete_evidence(tmp_path)
    api = weekly()
    output = tmp_path / 'bundle'
    api['collect'](value, paths, output, 'reference')
    bundle = api['load_bundle'](output,
                                hashlib.sha256(
                                    (output /
                                     'manifest.json').read_bytes()).hexdigest())
    assert bundle['physical_fits'] == 11
    assert bundle['mode'] == 'reference'
    assert (
        output /
        'cases/P2-fixed/model/checkpoint').read_bytes() == b'fixture checkpoint'
    assert len(bundle['cases']) == 11


@pytest.mark.parametrize('change',
                         ['file', 'report', 'symlink', 'missing_checkpoint'])
def test_collect_rejects_changed_or_missing_raw_artifacts(tmp_path, change):
    value, paths = complete_evidence(tmp_path)
    first = paths[0].parent
    if change == 'file':
        (first / 'predictions.parquet').write_bytes(b'changed')
    elif change == 'report':
        report = json.loads(paths[0].read_text())
        report['result']['score'] = 0.0
        paths[0].write_text(json.dumps(report))
    elif change == 'symlink':
        (first / 'unexpected').symlink_to(first / 'result.json')
    else:
        (first / 'model/checkpoint').unlink()
    with pytest.raises(ValueError):
        weekly()['collect'](value, paths, tmp_path / 'bundle', 'reference')


def test_compare_reports_drops_without_adding_quality_or_performance_gates(
        tmp_path):
    value, reference_paths = complete_evidence(tmp_path / 'reference')
    _, candidate_paths = complete_evidence(tmp_path / 'candidate', 'candidate',
                                           '2' * 64, '-candidate')
    api = weekly()
    reference, candidate = tmp_path / 'reference-bundle', tmp_path / 'candidate-bundle'
    api['collect'](value, reference_paths, reference, 'reference')
    api['collect'](value, candidate_paths, candidate, 'candidate')
    pin = lambda root: hashlib.sha256(
        (root / 'manifest.json').read_bytes()).hexdigest()
    result = api['compare'](value, candidate, pin(candidate), reference,
                            pin(reference), tmp_path / 'comparison')
    assert result['status'] == 'MEASURED'
    assert result['candidate_physical_fits'] == 11
    assert result['quality_policy'] == 'report_only'
    assert result['performance_reference'][
        'version'] == '1.0.0+group5.20261023.1'
    assert result['performance_reference'][
        'status'] == 'missing_or_incompatible'
    assert result['performance_comparison_valid'] is False


def test_compare_refuses_reference_model_reuse_as_fresh_candidate_training(
        tmp_path):
    value, reference_paths = complete_evidence(tmp_path / 'reference')
    _, candidate_paths = complete_evidence(tmp_path / 'candidate', 'candidate',
                                           '2' * 64)
    api = weekly()
    reference, candidate = tmp_path / 'reference-bundle', tmp_path / 'candidate-bundle'
    api['collect'](value, reference_paths, reference, 'reference')
    api['collect'](value, candidate_paths, candidate, 'candidate')
    pin = lambda root: hashlib.sha256(
        (root / 'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='fresh'):
        api['compare'](value, candidate, pin(candidate), reference,
                       pin(reference), tmp_path / 'comparison')


def test_auroc_retains_native_boolean_targets(tmp_path):
    spec = ready_case(tmp_path)
    spec['metric'] = 'auroc'
    row = scalar_result(spec)
    row.update(metric='auroc', targets=[False, True],
               predictions=[[0.2], [0.8]], baseline_predictions=[[0.5], [0.5]],
               score=1.0, baseline_score=0.5)
    weekly()['verify_result'](spec, row)


def test_collection_keeps_empty_success_logs(tmp_path):
    value, paths = complete_evidence(tmp_path)
    first = paths[0].parent
    (first / 'worker.stderr').write_bytes(b'')
    report = json.loads(paths[0].read_text())
    report['files']['worker.stderr'] = hashlib.sha256(b'').hexdigest()
    paths[0].write_text(json.dumps(report))
    weekly()['collect'](value, paths, tmp_path / 'bundle', 'reference')


def test_reference_validation_refuses_different_input_identity_before_candidate_fits(
        tmp_path):
    value, paths = complete_evidence(tmp_path)
    api = weekly()
    output = tmp_path / 'bundle'
    api['collect'](value, paths, output, 'reference')
    value['cases'][0]['identity']['split_sha256'] = '9' * 64
    with pytest.raises(ValueError, match='identity'):
        api['verify_reference'](
            value, output,
            hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest())


def test_native_empty_link_answers_remain_in_frozen_population(tmp_path):
    spec = ready_case(tmp_path)
    spec['metric'] = 'recall@50'
    path = evidence_tree(tmp_path / 'ranked', spec)
    row = json.loads((path.parent / 'result.json').read_text())
    row.update(
        target_ids=[[], [2]], score=None, baseline_score=None,
        quality_missing_reason=('Native recall is undefined because the frozen '
                                'population contains an empty target set'))
    weekly()['verify_result'](spec, row)
    assert row['entity_ids'] == [1, 2]


def test_collect_accepts_other_case_preparation_after_a_case_manifest_was_frozen(
        tmp_path):
    value, paths = complete_evidence(tmp_path)
    original = json.loads((paths[0].parent / 'input-manifest.json').read_text())
    del original['cases'][2]['identity']
    source = paths[0].parent / 'input-manifest.json'
    source.write_text(json.dumps(original))
    report = json.loads(paths[0].read_text())
    report['manifest_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    report['files']['input-manifest.json'] = report['manifest_sha256']
    paths[0].write_text(json.dumps(report))
    weekly()['collect'](value, paths, tmp_path / 'bundle', 'reference')


@pytest.mark.parametrize('change', ['case_payload', 'policy'])
def test_collect_refuses_case_payload_or_global_policy_drift_between_manifest_revisions(
        tmp_path, change):
    value, paths = complete_evidence(tmp_path)
    if change == 'case_payload':
        value['cases'][0]['worker'] = dict(path='/changed/worker.py',
                                           sha256='9' * 64)
    else:
        value['policies'] = dict(quality='enforce_new_limit')
    with pytest.raises(ValueError, match='manifest|case'):
        weekly()['collect'](value, paths, tmp_path / 'bundle', 'reference')


def test_p2_has_complete_predictions_without_invented_quality_answers(tmp_path):
    spec = ready_case(tmp_path)
    spec.update(workload='P2', metric='recall@10')
    path = evidence_tree(tmp_path / 'p2', spec)
    row = json.loads((path.parent / 'result.json').read_text())
    for key in ('target_ids', 'targets', 'predictions', 'baseline_predictions',
                'baseline_ranked_ids', 'baseline_ranked_scores'):
        row.pop(key, None)
    row.update(
        score=None, baseline_score=None, quality_missing_reason=(
            'P2 fixed performance workload has no frozen quality answers'))
    weekly()['verify_result'](spec, row)
    row['ranked_ids'].pop()
    with pytest.raises(ValueError):
        weekly()['verify_result'](spec, row)


def test_candidate_fits_cannot_reuse_another_reference_case_model(tmp_path):
    value, reference_paths = complete_evidence(tmp_path / 'reference')
    _, candidate_paths = complete_evidence(tmp_path / 'candidate', 'candidate',
                                           '2' * 64, '-candidate')
    path = candidate_paths[0]
    report = json.loads(path.read_text())
    report['result']['fit_id'] = 'Q13-control'
    (path.parent / 'result.json').write_text(json.dumps(report['result']))
    report['files']['result.json'] = hashlib.sha256(
        (path.parent / 'result.json').read_bytes()).hexdigest()
    path.write_text(json.dumps(report))
    api = weekly()
    reference, candidate = tmp_path / 'reference-bundle', tmp_path / 'candidate-bundle'
    api['collect'](value, reference_paths, reference, 'reference')
    api['collect'](value, candidate_paths, candidate, 'candidate')
    pin = lambda root: hashlib.sha256(
        (root / 'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='fresh'):
        api['compare'](value, candidate, pin(candidate), reference,
                       pin(reference), tmp_path / 'comparison')


@pytest.mark.parametrize('field', [
    'instance_id', 'gpu_uuid', 'driver', 'vcpus', 'memory_total_bytes',
    'gpu_memory_bytes'
])
def test_hardware_admission_cannot_move_to_a_different_host_or_resource_profile(
        field):
    expected = dict(instance_id='i-0123456789abcdef0',
                    gpu_uuid='GPU-00000000-0000-4000-8000-000000000001',
                    driver='595.91.07', vcpus=16,
                    memory_total_bytes=133971701760,
                    gpu_memory_bytes=96 * 2**30)
    actual = dict(expected)
    actual[field] = expected[field] + 1 if type(
        expected[field]) is int else 'different'
    with pytest.raises(ValueError, match='hardware'):
        weekly()['verify_hardware'](expected, actual)


def test_candidate_build_receipt_preserves_exact_source_and_wheel(tmp_path):
    receipt = tmp_path / 'build.json'
    receipt.write_text(
        json.dumps(
            dict(revision='a' * 40, tree='b' * 40,
                 bundle_version='1.0.1+aaaaaaa', version='1.0.1+aaaaaaa',
                 wheel='kumo_engine-1.0.1+aaaaaaa.whl', wheel_sha256='2' * 64)))
    api = weekly()
    assert api['verify_candidate_build'](receipt, '1.0.1+aaaaaaa',
                                         '2' * 64)['revision'] == 'a' * 40
    with pytest.raises(ValueError, match='build'):
        api['verify_candidate_build'](receipt, '1.0.1+aaaaaaa', '3' * 64)


def test_training_critical_path_uses_only_actual_fit_timestamps():
    rows = [
        dict(training_started_utc='2026-10-29T20:00:00+00:00',
             training_completed_utc='2026-10-29T20:10:00+00:00'),
        dict(training_started_utc='2026-10-29T20:05:00+00:00',
             training_completed_utc='2026-10-29T20:15:00+00:00')
    ]
    api = weekly()
    assert api['training_critical_path'](rows) == 900.0
    assert api['training_critical_path']([rows[0], {}]) is None
