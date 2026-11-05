# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the collector against independent case files without training a model."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_gates.py'
TASKS = ('binary', 'multiclass', 'multilabel', 'multilabel_ranking',
         'regression', 'forecasting', 'static_link', 'temporal_link')


@pytest.fixture
def gates():
    assert SOURCE.is_file(), 'The independent collector is not implemented'
    spec = importlib.util.spec_from_file_location('nightly_gates', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def campaign(tmp_path):
    identity = dict(wheel_sha256='a' * 64, reference_sha256='b' * 64,
                    policy_sha256='c' * 64, runner_sha256='d' * 64,
                    recipe_sha256='e' * 64, requirements_sha256='3' * 64,
                    input_manifest_sha256={
                        '100M': '1' * 64,
                        '1B': '2' * 64
                    })
    expected, references, rows, learning = [], [], [], {}
    for scale in ('100M', '1B'):
        learning[scale] = {}
        for task in TASKS:
            lower = task in ('regression', 'forecasting')
            baseline = 1.0 if lower else .5
            learning[scale][task] = dict(min_normal_gain=.1)
            learning[scale][task]['max_control_advantage' if task ==
                                  'forecasting' else 'max_control_drift'] = .05
            for arm in ('normal', 'control'):
                label = f'{scale}/{task}-{arm}'
                case = dict(scale=scale, task=task, arm=arm, seed=0,
                            plan_sha256='f' * 64)
                expected.append(case)
                score = baseline if arm == 'control' else .7 if lower else .8
                references.append(dict(case, score=score, baseline=baseline))
                bindings = {
                    k: v
                    for k, v in identity.items() if k != 'input_manifest_sha256'
                }
                bindings.update(
                    input_manifest_sha256=identity['input_manifest_sha256']
                    [scale], plan_sha256='f' * 64)
                resources = dict(
                    format='kumo-engine-nightly-resources-v1',
                    status='MEASURED', scale=scale, task=task, arm=arm,
                    errors=[], seconds=3.5, gpu_class='NVIDIA A10G', gpus=1,
                    peaks=dict(cpu_cores=2.5, ram_bytes=4096,
                               scratch_bytes=8192, gpu_memory_bytes=1024),
                    memory_accounting=dict(method='cgroup-v2-memory.peak',
                                           dedicated=True,
                                           path='/sys/fs/cgroup/nightly/case'),
                    wheel_sha256=identity['wheel_sha256'],
                    input_manifest_sha256=identity['input_manifest_sha256']
                    [scale], harness_sha256=identity['runner_sha256'])
                rows.append(
                    dict(
                        case, status='MEASURED', score=score, baseline=baseline,
                        execution_id=label + '/canonical', fit_id=label,
                        physical_fits=2, checkpoint_sha256='0' * 64,
                        bindings=bindings,
                        prediction_checks=dict(status='PASS'),
                        resources=resources,
                        reference_replay=dict(status='PASS'),
                        training_coverage=dict(status='PASS', distinct_trials=2,
                                               max_completed_epochs=2)))
    result = SimpleNamespace(
        expected=expected, reference=dict(status='APPROVED', cases=references),
        policy=dict(status='APPROVED', learning=learning,
                    reference_bands={}), identity=identity, rows=rows,
        output=tmp_path / 'combined.json', root=tmp_path)
    return result


def collect(gates, campaign, **options):
    paths = []
    for i, row in enumerate(campaign.rows):
        path = campaign.root / f'case-{i}.json'
        path.write_text(json.dumps(row))
        paths.append(path)
    return gates.combine(paths, expected_cases=campaign.expected,
                         reference=campaign.reference, policy=campaign.policy,
                         identity=campaign.identity, output=campaign.output,
                         **options)


def add_workload_variants(campaign):
    variants = dict(streaming='temporal_link', community='regression',
                    payments='binary', fraud='binary')
    originals = list(campaign.rows)
    for scale in ('100M', '1B'):
        for variant, task in variants.items():
            campaign.policy['learning'][scale][variant] = copy.deepcopy(
                campaign.policy['learning'][scale][task])
            for arm in ('normal', 'control'):
                row = copy.deepcopy(
                    next(value for value in originals if value['scale'] == scale
                         and value['task'] == task and value['arm'] == arm))
                row.update(variant=variant,
                           execution_id=f'{scale}/{variant}-{arm}/canonical',
                           fit_id=f'{scale}/{variant}-{arm}')
                row['resources']['variant'] = variant
                case = {
                    name: row[name]
                    for name in ('scale', 'variant', 'task', 'arm', 'seed',
                                 'plan_sha256')
                }
                campaign.expected.append(case)
                campaign.reference['cases'].append(
                    dict(case, score=row['score'], baseline=row['baseline']))
                campaign.rows.append(row)
    return campaign


def test_variant_identity_separates_workloads_with_the_same_task(gates):
    core = dict(scale='100M', task='binary', arm='normal', seed=0)
    assert gates.case_key(core) == ('100M', 'binary', 'normal', 0)
    assert gates.case_key(dict(core, variant='payments')) != gates.case_key(
        dict(core, variant='fraud'))
    assert gates.case_label(dict(core, variant='fraud')) == '100M/fraud-normal'


def test_collects_all_seven_families_at_both_scales(gates, campaign):
    add_workload_variants(campaign)
    result = collect(gates, campaign)
    assert result['status'] == 'PASS'
    assert len(result['cases']) == 48
    assert len(result['learning']['tasks']) == 24
    assert result['learning']['tasks']['100M/community']['status'] == 'PASS'
    assert result['learning']['tasks']['1B/fraud']['status'] == 'PASS'


@pytest.mark.parametrize(
    'damage',
    ['missing', 'semantic_task', 'resource_variant', 'scale', 'variant'])
def test_variant_collections_reject_missing_or_mixed_workloads(
        gates, campaign, damage):
    add_workload_variants(campaign)
    row = next(value for value in campaign.rows
               if value.get('variant') == 'fraud' and value['scale'] == '100M')
    if damage == 'missing':
        campaign.rows.remove(row)
    elif damage == 'semantic_task':
        row['task'] = row['resources']['task'] = 'regression'
    elif damage == 'resource_variant':
        row['resources']['variant'] = 'payments'
    elif damage == 'scale':
        row['scale'] = '1B'
    else:
        row['variant'] = 'payments'
    result = collect(gates, campaign)
    assert result['status'] == 'FAIL'
    assert json.loads(campaign.output.read_text())['errors']


def test_variants_keep_independent_baselines_and_learning_limits(
        gates, campaign):
    add_workload_variants(campaign)
    for row in campaign.rows + campaign.reference['cases']:
        if row.get('variant') == 'community':
            row.update(baseline=2., score=1.6 if row['arm'] == 'normal' else 2.)
    assert collect(gates, campaign)['status'] == 'PASS'
    campaign.policy['learning']['100M']['fraud']['min_normal_gain'] = .4
    result = collect(gates, campaign)
    assert result['status'] == 'FAIL'
    assert result['learning']['tasks']['100M/fraud']['status'] == 'FAIL'
    assert result['learning']['tasks']['100M/payments']['status'] == 'PASS'
    assert result['learning']['tasks']['100M/binary']['status'] == 'PASS'


def test_reference_proposals_retain_each_workload_variant(gates, campaign):
    add_workload_variants(campaign)
    for tasks in campaign.policy['learning'].values():
        for limits in tasks.values():
            limits.update({name: None for name in limits})
    campaign.reference = dict(status='PENDING_MEASUREMENT', cases=[])
    result = collect(gates, campaign, measurement=True)
    assert result['status'] == 'MEASURED'
    assert len(result['reference_proposal']['cases']) == 48
    assert sum(
        row.get('variant') == 'streaming'
        for row in result['reference_proposal']['cases']) == 4


def test_extra_binary_variants_cannot_replace_the_core_binary_pair(
        gates, campaign):
    add_workload_variants(campaign)

    def keep(row):
        return row['scale'] != '100M' or row[
            'task'] != 'binary' or 'variant' in row

    campaign.expected = list(filter(keep, campaign.expected))
    campaign.rows = list(filter(keep, campaign.rows))
    campaign.reference['cases'] = list(filter(keep,
                                              campaign.reference['cases']))
    campaign.policy['learning']['100M'].pop('binary')
    assert collect(gates, campaign)['status'] == 'FAIL'


def test_reference_semantic_task_must_match_the_declared_variant(
        gates, campaign):
    add_workload_variants(campaign)
    row = next(value for value in campaign.reference['cases']
               if value.get('variant') == 'fraud')
    row['task'] = 'regression'
    assert collect(gates, campaign)['status'] == 'FAIL'


def test_archive_supports_the_declared_forty_eight_case_roster(gates, campaign):
    add_workload_variants(campaign)
    rows = archive_rows(campaign)
    archive = gates.archive_proposal(
        rows, reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'],
        expected_cases=campaign.expected)
    assert len(archive['cases']) == 48
    assert archive['files'][
        '100M/fraud-normal/full-predictions.parquet'] == '5' * 64
    assert sum(row.get('variant') == 'payments'
               for row in archive['cases']) == 4
    with pytest.raises(ValueError):
        gates.archive_proposal(
            rows[:-1], reference_identity=dict(version='1.0.0',
                                               wheel_sha256='a' * 64),
            input_manifest_sha256=campaign.identity['input_manifest_sha256'],
            recipe_sha256=campaign.identity['recipe_sha256'],
            expected_cases=campaign.expected)


def test_archived_repeat_keeps_variant_and_semantic_task(gates, campaign):
    add_workload_variants(campaign)
    rows = archive_rows(campaign)
    repeated = copy.deepcopy(
        next(row for row in rows if row.get('variant') == 'fraud'))
    repeated['execution_id'] = 'fraud-repeat'
    archive = gates.archive_proposal(
        rows, reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'],
        expected_cases=campaign.expected, campaign=dict(repeats=[repeated]))
    index = archive['repeat_payloads'][0]
    assert index['variant'] == 'fraud'
    assert index['task'] == 'binary'
    repeated['task'] = 'regression'
    with pytest.raises(ValueError):
        gates.archive_proposal(
            rows, reference_identity=dict(version='1.0.0',
                                          wheel_sha256='a' * 64),
            input_manifest_sha256=campaign.identity['input_manifest_sha256'],
            recipe_sha256=campaign.identity['recipe_sha256'],
            expected_cases=campaign.expected, campaign=dict(repeats=[repeated]))


def test_collects_both_scales_without_pairing_between_them(gates, campaign):
    result = collect(gates, campaign)
    assert result['status'] == 'PASS'
    assert len(result['cases']) == 32
    assert len(result['learning']['tasks']) == 16
    assert json.loads(campaign.output.read_text())['status'] == 'PASS'


def test_individual_passes_do_not_hide_paired_leakage(gates, campaign):
    for row in campaign.rows:
        if row['scale'] == '100M' and row['task'] == 'binary':
            row['score'] = .66 if row['arm'] == 'normal' else .64
    campaign.policy['learning']['100M']['binary']['max_control_drift'] = .15
    normal, control = campaign.rows[:2]
    limits = campaign.policy['learning']['100M']['binary']
    for row, reference in zip((normal, control),
                              campaign.reference['cases'][:2]):
        assert gates.case_learning(row, reference, limits)['status'] == 'PASS'
    result = collect(gates, campaign)
    assert result['status'] == 'FAIL'
    assert result['learning']['tasks']['100M/binary']['status'] == 'FAIL'


@pytest.mark.parametrize('damage', [
    'missing', 'duplicate', 'extra', 'failed', 'prediction', 'training',
    'unapproved'
])
def test_bad_collections_write_failed_evidence(gates, campaign, damage):
    if damage == 'missing':
        campaign.rows.pop()
    elif damage == 'duplicate':
        campaign.rows.append(copy.deepcopy(campaign.rows[0]))
    elif damage == 'extra':
        row = copy.deepcopy(campaign.rows[0])
        row.update(scale='10B', execution_id='extra')
        campaign.rows.append(row)
    elif damage == 'failed':
        campaign.rows[0]['status'] = 'FAIL'
    elif damage == 'prediction':
        campaign.rows[0]['prediction_checks']['status'] = 'FAIL'
    elif damage == 'training':
        campaign.rows[0]['training_coverage']['max_completed_epochs'] = 1
    else:
        campaign.rows[0]['measurement_only'] = True
    result = collect(gates, campaign)
    assert result['status'] == 'FAIL'
    saved = json.loads(campaign.output.read_text())
    assert saved['status'] == 'FAIL'
    assert saved['errors']


@pytest.mark.parametrize('field', [
    'wheel_sha256', 'reference_sha256', 'policy_sha256', 'runner_sha256',
    'recipe_sha256', 'requirements_sha256', 'input_manifest_sha256',
    'plan_sha256'
])
def test_mixed_identities_cannot_qualify(gates, campaign, field):
    campaign.rows[0]['bindings'][field] = '9' * 64
    assert collect(gates, campaign)['status'] == 'FAIL'


@pytest.mark.parametrize('measurement', [False, True])
@pytest.mark.parametrize('field,value', [
    ('resources', None),
    ('status', 'INCOMPLETE'),
    ('errors', ['counter failed']),
    ('format', 'unrecognized'),
    ('memory_accounting.method', 'summed-rss'),
    ('memory_accounting.dedicated', False),
    ('peaks.cpu_cores', 0),
    ('peaks.ram_bytes', 0),
    ('peaks.scratch_bytes', 0),
    ('peaks.gpu_memory_bytes', 0),
    ('seconds', 0),
    ('gpus', 0),
    ('gpus', True),
    ('gpu_class', ''),
    ('wheel_sha256', '9' * 64),
    ('input_manifest_sha256', '9' * 64),
    ('harness_sha256', '9' * 64),
    ('scale', '10B'),
    ('task', 'forecasting'),
    ('arm', 'control'),
])
def test_incomplete_or_mixed_resource_evidence_cannot_qualify(
        gates, campaign, field, value, measurement):
    if field == 'resources':
        campaign.rows[0]['resources'] = value
    else:
        record = campaign.rows[0]['resources']
        *parents, name = field.split('.')
        for parent in parents:
            record = record[parent]
        record[name] = value
    result = collect(gates, campaign, measurement=measurement)
    assert result['status'] == 'FAIL'
    assert json.loads(campaign.output.read_text())['status'] == 'FAIL'
    assert any('resource' in error.lower() for error in result['errors'])


@pytest.mark.parametrize('score,expected', [(1., 'PASS'), (1.5, 'PASS'),
                                            (.96, 'PASS'), (.8, 'FAIL')])
def test_forecast_control_checks_excess_gain_one_sided(gates, score, expected):
    case = dict(scale='100M', task='forecasting', arm='control', seed=0,
                score=score)
    reference = dict(case, score=1.1, baseline=1.)
    result = gates.case_learning(
        case, reference, dict(min_normal_gain=.1, max_control_advantage=.04))
    assert result['status'] == expected


def test_forecast_never_reuses_the_old_closeness_limit(gates):
    row = dict(scale='100M', task='forecasting', arm='control', seed=0,
               score=1.)
    with pytest.raises(ValueError):
        gates.case_learning(row, dict(row, baseline=1.),
                            dict(min_normal_gain=.1, max_control_drift=.0544))


def test_pending_calibration_is_a_proposal_not_a_pass(gates, campaign):
    for scale in campaign.policy['learning'].values():
        for limits in scale.values():
            for key in limits:
                limits[key] = None
    campaign.reference = dict(status='PROPOSAL_NOT_APPROVED', cases=[])
    result = collect(gates, campaign, measurement=True)
    assert result['status'] == 'MEASURED'
    assert result['qualification'] == 'NOT_QUALIFIED'
    assert len(result['reference_proposal']['cases']) == 32
    assert result['reference_proposal']['status'] == 'PROPOSAL_NOT_APPROVED'
    assert collect(gates, campaign)['status'] == 'FAIL'


def test_bootstrap_uses_measured_baselines_instead_of_null_placeholders(
        gates, campaign):
    campaign.reference['status'] = 'PENDING_MEASUREMENT'
    for row in campaign.reference['cases']:
        row.update(score=None, baseline=None)
    for scale in campaign.policy['learning'].values():
        for limits in scale.values():
            for key in limits:
                limits[key] = None
    result = collect(gates, campaign, measurement=True)
    assert result['status'] == 'MEASURED'
    assert result['reference_proposal']['cases'][0]['baseline'] == .5


def test_a_configured_scale_cannot_disappear_from_the_expected_roster(
        gates, campaign):
    campaign.expected = [r for r in campaign.expected if r['scale'] == '100M']
    campaign.rows = campaign.rows[:16]
    campaign.reference['cases'] = campaign.reference['cases'][:16]
    campaign.policy['learning'].pop('1B')
    assert collect(gates, campaign)['status'] == 'FAIL'


def test_bootstrap_does_not_need_a_checksum_for_a_nonexistent_reference(
        gates, campaign):
    campaign.identity['reference_sha256'] = None
    campaign.reference = dict(status='PENDING_MEASUREMENT', cases=[])
    for row in campaign.rows:
        row['bindings']['reference_sha256'] = None
    result = collect(gates, campaign, measurement=True)
    assert result['status'] == 'MEASURED'
    assert result['qualification'] == 'NOT_QUALIFIED'


@pytest.mark.parametrize('name', ['reference', 'policy'])
def test_checksum_bindings_do_not_approve_a_record(gates, campaign, name):
    getattr(campaign, name).pop('status')
    result = collect(gates, campaign)
    assert result['status'] == 'FAIL'
    assert any('approved' in value.lower() for value in result['errors'])


@pytest.mark.parametrize('field,value', [('training_coverage', []),
                                         ('prediction_checks', None),
                                         ('bindings', 'invalid')])
def test_malformed_case_evidence_still_writes_a_failed_report(
        gates, campaign, field, value):
    campaign.rows[0][field] = value
    assert collect(gates, campaign)['status'] == 'FAIL'
    assert json.loads(campaign.output.read_text())['status'] == 'FAIL'


def test_repeats_require_a_preselected_canonical_execution(gates, campaign):
    repeated = copy.deepcopy(campaign.rows[0])
    repeated.update(execution_id='repeat-1', fit_id='repeat-fit', score=.99)
    campaign.rows.append(repeated)
    assert collect(gates, campaign)['status'] == 'FAIL'
    result = collect(
        gates, campaign, canonical_executions={
            '100M/binary-normal': '100M/binary-normal/canonical'
        })
    assert result['status'] == 'PASS'
    assert result['cases'][0]['score'] == .8
    assert result['repeats'][0]['score'] == .99


def test_nightly_reports_quality_loss_and_weekly_enforces_retention(
        gates, campaign):
    row = campaign.rows[0]
    row['score'] = .65
    campaign.policy['reference_bands'] = {
        '100M': {
            'binary': dict(max_relative_drop=.05)
        }
    }
    result = collect(gates, campaign)
    assert result['status'] == 'PASS'
    assert result['quality_reports']['100M/binary-normal']['status'] == 'WARN'
    assert collect(gates, campaign, tier='weekly')['status'] == 'FAIL'


def test_weekly_replay_failure_stays_visible(gates, campaign):
    campaign.rows[0]['reference_replay']['status'] = 'FAIL'
    result = collect(gates, campaign, tier='weekly')
    assert result['status'] == 'FAIL'
    assert any('replay' in error.lower() for error in result['errors'])


def archive_rows(campaign):
    rows = copy.deepcopy(campaign.rows)
    for row in rows:
        model_files = {
            'checkpoint.ckpt': '0' * 64,
            'model.json': '7' * 64,
            'encoders/target.bin': '8' * 64
        }
        row['model'] = dict(path='published-model', files=model_files)
        row['population'] = dict(path='prediction-inputs/predict.parquet',
                                 sha256='4' * 64)
        row['canonical_predictions'] = dict(path='full-predictions.parquet',
                                            sha256='5' * 64)
        stages = {
            name: dict(rows=number, sha256=checksum * 64)
            for name, number, checksum in (('train', 3, '6'), ('val', 2, '7'),
                                           ('test', 2, '8'))
        }
        row['training_table'] = dict(stages=stages,
                                     normal_stages=copy.deepcopy(stages),
                                     generation_plan=dict(seed=0))
        row['artifacts'] = {
            'published-model/' + name: checksum
            for name, checksum in model_files.items()
        }
        row['artifacts'].update({
            'prediction-inputs/predict.parquet': '4' * 64,
            'full-predictions.parquet': '5' * 64,
            'candidate-validation.json': '9' * 64,
            'requested-plan.json': 'f' * 64
        })
    return rows


def make_archive(gates, campaign, rows):
    return gates.archive_proposal(
        rows, reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'])


def test_collection_keeps_46_original_parents_with_two_explicit_case_bindings(
        gates, campaign):
    add_workload_variants(campaign)
    for scale, checksum in (('100M', '7' * 64), ('1B', '8' * 64)):
        case = next(row for row in campaign.expected
                    if row['scale'] == scale and row['task'] == 'static_link'
                    and row['arm'] == 'control')
        case['input_manifest_sha256'] = checksum
        row = next(row for row in campaign.rows if row['scale'] == scale
                   and row['task'] == 'static_link' and row['arm'] == 'control')
        row['bindings']['input_manifest_sha256'] = checksum
        row['resources']['input_manifest_sha256'] = checksum
    report = collect(gates, campaign, measurement=True)
    assert report['status'] == 'MEASURED'
    assert len(report['cases']) == 48
    assert sum(row['bindings']['input_manifest_sha256'] ==
               campaign.identity['input_manifest_sha256'][row['scale']]
               for row in report['cases']) == 46


def test_archive_records_explicit_case_parents_without_relabeling_old_results(
        gates, campaign):
    add_workload_variants(campaign)
    rows, expected = archive_rows(campaign), copy.deepcopy(campaign.expected)
    overrides = {
        '100M/static_link-control': '7' * 64,
        '1B/static_link-control': '8' * 64
    }
    for case in expected:
        if gates.case_label(case) in overrides:
            case['input_manifest_sha256'] = overrides[gates.case_label(case)]
    for row in rows:
        if gates.case_label(row) in overrides:
            row['bindings']['input_manifest_sha256'] = overrides[
                gates.case_label(row)]
    before = copy.deepcopy(rows)
    archive = gates.archive_proposal(
        rows, reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'],
        expected_cases=expected, campaign=dict(cases=rows, repeats=[]))
    assert archive['input_manifest_sha256'] == campaign.identity[
        'input_manifest_sha256']
    assert archive['case_input_manifest_sha256'] == overrides
    assert len(archive['cases']) == 48
    for case in archive['cases']:
        assert case['input_manifest_sha256'] == overrides.get(
            gates.case_label(case),
            campaign.identity['input_manifest_sha256'][case['scale']])
    assert archive['collector_sha256'] == hashlib.sha256(
        SOURCE.read_bytes()).hexdigest()
    assert all(row['bindings']['runner_sha256'] == 'd' * 64
               for row in archive['campaign']['cases'])
    assert rows == before
    with pytest.raises(ValueError, match='input'):
        gates.archive_proposal(
            rows, reference_identity=dict(version='1.0.0',
                                          wheel_sha256='a' * 64),
            input_manifest_sha256=campaign.identity['input_manifest_sha256'],
            recipe_sha256=campaign.identity['recipe_sha256'],
            expected_cases=campaign.expected)


@pytest.mark.parametrize('status,checksum,accepted', [
    ('FAIL', '1' * 64, True),
    ('FAIL', '7' * 64, True),
    ('MEASURED', '1' * 64, False),
    ('FAIL', '9' * 64, False),
])
def test_repaired_case_retains_only_bound_historical_parent_failures(
        gates, campaign, status, checksum, accepted):
    rows, expected = archive_rows(campaign), copy.deepcopy(campaign.expected)
    case = next(row for row in expected if row['scale'] == '100M'
                and row['task'] == 'static_link' and row['arm'] == 'control')
    case['input_manifest_sha256'] = '7' * 64
    canonical = next(row for row in rows
                     if row['scale'] == '100M' and row['task'] == 'static_link'
                     and row['arm'] == 'control')
    canonical['bindings']['input_manifest_sha256'] = '7' * 64
    repeated = copy.deepcopy(canonical)
    repeated.update(status=status,
                    execution_id='100M/static_link-control/original-attempt')
    repeated['bindings']['input_manifest_sha256'] = checksum
    arguments = dict(
        reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'],
        expected_cases=expected, campaign=dict(cases=rows, repeats=[repeated]))
    if not accepted:
        with pytest.raises(ValueError, match='input'):
            gates.archive_proposal(rows, **arguments)
        return
    archive = gates.archive_proposal(rows, **arguments)
    assert archive['repeat_payloads'][0]['input_manifest_sha256'] == checksum
    assert archive['campaign']['repeats'][0] == repeated


def test_archive_proposal_binds_portable_case_payloads(gates, campaign):
    rows = archive_rows(campaign)
    rows[-1]['raw_limit'] = 21
    archive = make_archive(gates, campaign, rows)
    assert archive['format'] == 'kumo-engine-nightly-archive-v1'
    assert archive['status'] == 'PROPOSAL_NOT_APPROVED'
    assert len(archive['cases']) == 32
    assert len(archive['files']) == 224
    first = archive['cases'][0]
    assert first['execution_id'] == '100M/binary-normal/canonical'
    assert first['replay']['model'] == dict(
        path='100M/binary-normal/published-model', files={
            'checkpoint.ckpt': '0' * 64,
            'model.json': '7' * 64,
            'encoders/target.bin': '8' * 64
        })
    assert first['replay']['population'] == dict(
        path='100M/binary-normal/prediction-inputs/predict.parquet',
        sha256='4' * 64)
    assert first['replay']['predictions'] == dict(
        path='100M/binary-normal/full-predictions.parquet', sha256='5' * 64)
    assert archive['files'][
        '1B/temporal_link-control/published-model/checkpoint.ckpt'] == '0' * 64
    assert first['training_table']['stages']['train']['rows'] == 3
    assert first['training_table']['generation_plan'] == dict(seed=0)
    assert archive['cases'][-1]['replay']['raw_limit'] == 21
    assert 'raw_limit' not in first['replay']
    assert rows[0]['model']['path'] == 'published-model'


@pytest.mark.parametrize('expanded', [False, True])
def test_root_reference_collection_emits_consumable_proposals(
        campaign, tmp_path, expanded):
    if expanded:
        add_workload_variants(campaign)
    spec = importlib.util.spec_from_file_location(
        'nightly_proposal_cli', SOURCE.with_name('nightly_check.py'))
    root = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root)
    configuration = dict(
        format='kumo-engine-nightly-job-v2', min_free_bytes=1, archive=None,
        scales={
            scale:
            dict(
                input_manifest_sha256=checksum,
                inputs=dict(bucket='shared', region='us-west-2',
                            key=scale + '/inputs.tar', sha256='c' * 64))
            for scale, checksum in
            campaign.identity['input_manifest_sha256'].items()
        })
    if expanded:
        configuration['variants'] = root.read(
            SOURCE.parents[1] / 'test/nightly/inputs.json')['variants']
    wheel = tmp_path / 'reference.whl'
    wheel.write_bytes(b'published reference wheel')
    configuration['reference_identity'] = dict(version='1.0.0',
                                               wheel_sha256=root.digest(wheel))
    for name, value in (('reference',
                         dict(status='PENDING_MEASUREMENT',
                              cases=[])), ('policy', campaign.policy),
                        ('requirements',
                         dict(status='PENDING_MEASUREMENT', cases={}))):
        path = tmp_path / (name + '.json')
        root.save(path, value)
        configuration[name] = dict(path=path.name, sha256=root.digest(path))
    path = tmp_path / 'inputs.json'
    root.save(path, configuration)
    identity = root.execution_identity(configuration, wheel)
    paths = []
    for index, row in enumerate(archive_rows(campaign)):
        row['bindings'].update(
            identity, input_manifest_sha256=identity['input_manifest_sha256'][
                row['scale']])
        row['resources'].update(wheel_sha256=identity['wheel_sha256'],
                                harness_sha256=identity['runner_sha256'])
        result = tmp_path / f'execution-{index}.json'
        root.save(result, row)
        paths.append(str(result))
    output = tmp_path / 'proposals'
    assert root.main([
        'combine', '--configuration',
        str(path), '--mode', 'reference', '--wheel',
        str(wheel), '--results', *paths, '--output-dir',
        str(output)
    ]) == 0
    proposal = root.read(output / 'reference-proposal.json')
    assert proposal['status'] == 'PROPOSAL_NOT_APPROVED'
    assert proposal['reference_identity'] == configuration['reference_identity']
    assert proposal['input_manifest_sha256'] == identity[
        'input_manifest_sha256']
    assert proposal['recipe_sha256'] == identity['recipe_sha256']
    count = 48 if expanded else 32
    assert len(proposal['cases']) == count
    requirements = root.read(output / 'requirements-proposal.json')
    assert len(requirements['cases']) == count
    assert requirements['cases']['100M/binary-normal']['cpus'] == 4
    archive = root.read(output / 'archive-proposal.json')
    assert len(archive['cases']) == count
    evidence = archive['campaign']['cases'][0]
    assert evidence['resources']['memory_accounting']['dedicated'] is True
    assert evidence['bindings']['runner_sha256'] == identity['runner_sha256']
    assert archive['campaign']['status'] == 'MEASURED'
    configuration['archive'] = dict(
        path='proposals/archive-proposal.json',
        sha256=root.digest(output / 'archive-proposal.json'))
    assert root.load_archive(configuration, path,
                             tmp_path)['files'] == archive['files']


@pytest.mark.parametrize('damage', [
    'missing', 'duplicate', 'failed', 'no_rows', 'missing_model',
    'missing_training', 'missing_payload', 'changed_payload',
    'extra_model_file', 'checkpoint', 'wrong_wheel', 'wrong_inputs',
    'wrong_recipe'
])
def test_archive_proposal_rejects_missing_or_mismatched_payloads(
        gates, campaign, damage):
    rows = archive_rows(campaign)
    if damage == 'missing':
        rows.pop()
    elif damage == 'no_rows':
        rows = None
    elif damage == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif damage == 'failed':
        rows[0]['status'] = 'FAIL'
    elif damage == 'missing_model':
        rows[0].pop('model')
    elif damage == 'missing_training':
        rows[0].pop('training_table')
    elif damage == 'missing_payload':
        rows[0]['artifacts'].pop('full-predictions.parquet')
    elif damage == 'changed_payload':
        rows[0]['population']['sha256'] = '9' * 64
    elif damage == 'extra_model_file':
        rows[0]['artifacts']['published-model/unrecorded.bin'] = '9' * 64
    elif damage == 'checkpoint':
        rows[0]['checkpoint_sha256'] = 'a' * 64
    else:
        key = {
            'wrong_wheel': 'wheel_sha256',
            'wrong_inputs': 'input_manifest_sha256',
            'wrong_recipe': 'recipe_sha256'
        }[damage]
        rows[0]['bindings'][key] = '9' * 64
    with pytest.raises(ValueError):
        make_archive(gates, campaign, rows)


@pytest.mark.parametrize(
    'name', ['', '../escape', '/absolute', 'a//b', 'a/./b', 'a/../b', 'a\\b'])
def test_archive_proposal_rejects_unsafe_artifact_paths(gates, campaign, name):
    rows = archive_rows(campaign)
    rows[0]['artifacts'][name] = '9' * 64
    with pytest.raises(ValueError):
        make_archive(gates, campaign, rows)


def test_archive_keeps_full_campaign_provenance_and_repeat_payloads(
        gates, campaign):
    rows = archive_rows(campaign)
    rows[0].update(runtime=dict(version='1.0.0', gpu='NVIDIA A10G'),
                   baseline_evidence=dict(method='training-only'),
                   timings=dict(training=2.5))
    repeated = copy.deepcopy(rows[0])
    repeated.update(execution_id='repeat/one', score=.99)
    report = dict(status='MEASURED', cases=rows, repeats=[repeated], errors=[],
                  build=dict(wheel_sha256='a' * 64))
    archive = gates.archive_proposal(
        rows, reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'], campaign=report)
    stored = archive['campaign']['cases'][0]
    assert stored['resources'] == rows[0]['resources']
    assert stored['runtime']['version'] == '1.0.0'
    assert stored['baseline_evidence']['method'] == 'training-only'
    assert stored['timings']['training'] == 2.5
    assert archive['campaign']['build']['wheel_sha256'] == 'a' * 64
    prefix = 'repeats/100M/binary-normal/' + hashlib.sha256(
        b'repeat/one').hexdigest()
    assert archive['repeat_payloads'][0]['path'] == prefix
    assert archive['files'][prefix +
                            '/published-model/checkpoint.ckpt'] == '0' * 64
    report['cases'][0]['score'] = 0.
    assert archive['campaign']['cases'][0]['score'] == .8


def test_failed_partial_repeat_keeps_metadata_and_available_payloads(
        gates, campaign):
    rows = archive_rows(campaign)
    failed = dict(scale='100M', task='binary', arm='normal', seed=0,
                  execution_id='failed-one', status='FAIL',
                  error='Snapshot failed before fitting',
                  artifacts={'failure.log': '9' * 64})
    report = dict(status='FAIL', cases=rows, repeats=[failed],
                  errors=['failed'])
    archive = gates.archive_proposal(
        rows, reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'], campaign=report)
    assert archive['campaign']['repeats'][0]['error'] == failed['error']
    prefix = archive['repeat_payloads'][0]['path']
    assert archive['files'][prefix + '/failure.log'] == '9' * 64


def test_repeat_payload_cannot_escape_the_archive_or_change_reference(
        gates, campaign):
    rows = archive_rows(campaign)
    repeated = copy.deepcopy(rows[0])
    repeated['execution_id'] = 'repeat-one'
    report = dict(status='MEASURED', cases=rows, repeats=[repeated])
    arguments = dict(
        reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        input_manifest_sha256=campaign.identity['input_manifest_sha256'],
        recipe_sha256=campaign.identity['recipe_sha256'], campaign=report)
    repeated['artifacts']['../escape'] = '9' * 64
    with pytest.raises(ValueError):
        gates.archive_proposal(rows, **arguments)
    repeated['artifacts'].pop('../escape')
    repeated['bindings']['wheel_sha256'] = '9' * 64
    with pytest.raises(ValueError):
        gates.archive_proposal(rows, **arguments)


def test_collector_cli_runs_on_saved_files_without_a_gpu(gates, campaign):
    collect(gates, campaign)
    specification = campaign.root / 'specification.json'
    specification.write_text(
        json.dumps(
            dict(cases=campaign.expected, reference=campaign.reference,
                 policy=campaign.policy, identity=campaign.identity)))
    output = campaign.root / 'cli-summary.json'
    command = [
        sys.executable, '-B',
        str(SOURCE), 'combine', '--specification',
        str(specification), '--output',
        str(output), *map(str, sorted(campaign.root.glob('case-*.json')))
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())['status'] == 'PASS'
    command.pop()
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 1
    assert json.loads(output.read_text())['status'] == 'FAIL'
