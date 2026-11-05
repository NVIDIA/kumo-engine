# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate independent scale cases without importing the Engine or CUDA."""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

TASKS = ('binary', 'multiclass', 'multilabel', 'multilabel_ranking',
         'regression', 'forecasting', 'static_link', 'temporal_link')
LOWER_IS_BETTER = ('regression', 'forecasting')
BINDINGS = ('wheel_sha256', 'reference_sha256', 'policy_sha256',
            'runner_sha256', 'recipe_sha256', 'requirements_sha256')


def variant_name(row):
    return row.get('variant', row['task'])


def case_identity(row):
    result = {name: row[name] for name in ('scale', 'task', 'arm', 'seed')}
    if 'variant' in row:
        result['variant'] = row['variant']
    return result


def case_key(row):
    """Keep scales and seeds separate when finding normal/control pairs."""
    if not isinstance(row, dict):
        raise ValueError('A case must be a JSON object')
    variant = variant_name(row) if 'task' in row else None
    key = (row.get('scale'), variant, row.get('arm'), row.get('seed'))
    if (not isinstance(key[0], str)
            or not re.fullmatch(r'[1-9][0-9]*[MB]', key[0])
            or row.get('task') not in TASKS or not isinstance(variant, str)
            or not re.fullmatch('[a-z][a-z0-9_]*', variant)
            or key[2] not in ('normal', 'control') or type(key[3]) is not int
            or key[3] < 0):
        raise ValueError('Invalid scale, task, arm, or seed')
    return key


def case_label(row):
    scale, task, arm, seed = case_key(row)
    return f'{scale}/{task}-{arm}' + (f'/seed-{seed}' if seed else '')


def validate_score(task, value):
    if (task not in TASKS or type(value) not in (int, float)
            or not math.isfinite(value) or value < 0
            or task not in LOWER_IS_BETTER and value > 1):
        raise ValueError('Quality score or baseline is missing or invalid')


def _limit(value, *, positive=False):
    if value is not None and (type(value) not in (int, float)
                              or not math.isfinite(value) or value < 0
                              or positive and value == 0):
        raise ValueError(
            'Policy bounds must be finite, nonnegative, or pending')
    return value


def _limits(task, limits):
    control = ('max_control_advantage'
               if task == 'forecasting' else 'max_control_drift')
    if not isinstance(limits, dict) or set(limits) != {
            'min_normal_gain', control
    }:
        raise ValueError('Learning policy needs the explicit control rule')
    return (_limit(limits['min_normal_gain'],
                   positive=True), _limit(limits[control]))


def validate_policy(policy, expected_cases=None):
    """Reject invalid bounds while retaining explicit pending proposals."""
    if not isinstance(policy, dict) or not isinstance(policy.get('learning'),
                                                      dict):
        raise ValueError('Learning policy is missing')
    semantic = {}
    for row in expected_cases or []:
        key = (case_key(row)[0], variant_name(row))
        if key in semantic and semantic[key] != row['task']:
            raise ValueError('A workload variant cannot change its task type')
        semantic[key] = row['task']
    wanted = set(semantic)
    found = set()
    for scale, tasks in policy['learning'].items():
        if not isinstance(tasks, dict):
            raise ValueError('Scale learning policy is invalid')
        for variant, limits in tasks.items():
            task = semantic.get((scale, variant), variant)
            if task not in TASKS or wanted and (scale, variant) not in wanted:
                raise ValueError('Unknown task in learning policy')
            _limits(task, limits)
            found.add((scale, variant))
    if wanted and found != wanted:
        raise ValueError('Learning policy does not match the configured scales')
    bands = policy.get('reference_bands', {})
    if not isinstance(bands, dict):
        raise ValueError('Reference bands must be scale objects')
    for scale, tasks in bands.items():
        if not isinstance(tasks, dict):
            raise ValueError('Reference bands must contain task objects')
        for task, band in tasks.items():
            if (not isinstance(band, dict) or (scale, task) not in found
                    or set(band) != {'max_relative_drop'}):
                raise ValueError(
                    'Reference band does not match the task policy')
            _limit(band['max_relative_drop'])
    return policy


def quality_report(row, reference, band=None):
    """Report reference loss independently of the enforced learning verdict."""
    task, score, baseline = row['task'], row['score'], reference['baseline']
    validate_score(task, score)
    validate_score(task, baseline)
    previous = reference.get('score')
    if previous is None:
        return dict(status='PENDING', reference_score=None)
    validate_score(task, previous)
    direction = -1 if task in LOWER_IS_BETTER else 1
    change = direction * (score - previous)
    gain = direction * (previous - baseline)
    relative_drop = max(0., -change) / abs(previous) if previous else None
    retained = direction * (score - baseline) / gain if gain > 0 else None
    maximum = (band or {}).get('max_relative_drop')
    _limit(maximum)
    status = ('PENDING' if maximum is None or relative_drop is None else
              'WARN' if relative_drop > maximum else 'PASS')
    return dict(status=status, reference_score=previous,
                signed_quality_change=change, relative_drop=relative_drop,
                retained_reference_gain=retained, max_relative_drop=maximum,
                policy='report-only')


def case_learning(row, reference, limits, *, tier='nightly'):
    """Use one control rule in both standalone and combined execution."""
    if tier not in ('nightly', 'weekly'):
        raise ValueError('Unknown quality tier')
    task, arm, score = row['task'], row['arm'], row['score']
    baseline = reference['baseline']
    validate_score(task, score)
    validate_score(task, baseline)
    margin, tolerance = _limits(task, limits)
    direction = -1 if task in LOWER_IS_BETTER else 1
    gain = direction * (score - baseline)
    passed, pending = True, False
    if arm == 'normal':
        pending = margin is None
        if not pending:
            passed = (score <= baseline -
                      margin if direction < 0 else score >= baseline + margin)
        if tier == 'weekly':
            previous = reference.get('score')
            if previous is None:
                pending = True
            else:
                validate_score(task, previous)
                reference_gain = direction * (previous - baseline)
                boundary = baseline + direction * .75 * reference_gain
                passed = passed and reference_gain > 0 and (
                    (score <= boundary if direction < 0 else score >= boundary)
                    or math.isclose(score, boundary, rel_tol=0.,
                                    abs_tol=math.ulp(boundary)))
    elif arm == 'control':
        pending = tolerance is None
        if not pending:
            passed = (score >= baseline -
                      tolerance if task == 'forecasting' else
                      baseline - tolerance <= score <= baseline + tolerance)
    else:
        raise ValueError('Unknown answer arm')
    return dict(
        status='FAIL' if not passed else 'MEASURED' if pending else 'PASS',
        score=score, baseline=baseline, gain=gain, **limits)


def _index(rows):
    result = {}
    for row in rows:
        key = case_key(row)
        if key in result:
            raise ValueError('Duplicate case identity')
        result[key] = row
    return result


def _roster(rows):
    indexed = _index(rows)
    groups = defaultdict(list)
    for key, row in indexed.items():
        groups[key[:2]].append(row)
    for pair in groups.values():
        if (len(pair) != 2 or {row['arm']
                               for row in pair} != {'normal', 'control'}
                or len({row['seed']
                        for row in pair}) != 1
                or len({row['task']
                        for row in pair}) != 1):
            raise ValueError(
                'Each declared workload variant needs one matching task pair')
    for scale in {key[0] for key in indexed}:
        core = {(row['task'], row['arm'])
                for row in indexed.values()
                if row['scale'] == scale and variant_name(row) == row['task']}
        if not {(task, arm)
                for task in TASKS
                for arm in ('normal', 'control')} <= core:
            raise ValueError(
                'Each configured scale needs all eight core task pairs')
    return indexed


def check_learning(rows, reference_rows, learning_by_scale, *, tier='nightly'):
    references, pairs = _index(reference_rows), defaultdict(list)
    for row in rows:
        scale, variant, _, seed = case_key(row)
        pairs[(scale, variant, seed)].append(row)
    tasks = {}
    for (scale, variant, seed), pair in pairs.items():
        if (len(pair) != 2 or {r['arm']
                               for r in pair} != {'normal', 'control'}
                or len({r['task']
                        for r in pair}) != 1):
            raise ValueError(
                'Each scale/task/seed needs one normal and control')
        task = pair[0]['task']
        normal, control = [
            next(r for r in pair if r['arm'] == arm)
            for arm in ('normal', 'control')
        ]
        limits = learning_by_scale[scale][variant]
        margin, _ = _limits(task, limits)
        checks = [
            case_learning(row, references[case_key(row)], limits, tier=tier)
            for row in pair
        ]
        direction = -1 if task in LOWER_IS_BETTER else 1
        separation = direction * (normal['score'] - control['score'])
        separated = margin is None or (
            normal['score'] <= control['score'] - margin
            if direction < 0 else normal['score'] >= control['score'] + margin)
        statuses = {value['status'] for value in checks}
        status = ('FAIL' if not separated or 'FAIL' in statuses else
                  'MEASURED' if 'MEASURED' in statuses else 'PASS')
        label = f'{scale}/{variant}' + (f'/seed-{seed}' if seed else '')
        tasks[label] = dict(status=status, normal_control_gain=separation,
                            min_normal_gain=margin, cases=checks)
    if not tasks:
        raise ValueError('No learning pairs were collected')
    statuses = {row['status'] for row in tasks.values()}
    return dict(
        status='FAIL' if 'FAIL' in statuses else
        'MEASURED' if 'MEASURED' in statuses else 'PASS', tasks=tasks)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def _validate_resources(row, bindings):
    """Require complete case measurements without recomputing resource minima."""
    resources = row.get('resources')
    label = case_label(row)
    if not isinstance(resources, dict):
        raise ValueError(f'{label}: resource measurements are missing')
    peaks = resources.get('peaks')
    accounting = resources.get('memory_accounting')
    if (resources.get('format') != 'kumo-engine-nightly-resources-v1'
            or resources.get('status') != 'MEASURED'
            or resources.get('errors') != []
            or not isinstance(accounting, dict)
            or accounting.get('method') != 'cgroup-v2-memory.peak'
            or accounting.get('dedicated') is not True
            or not isinstance(peaks, dict) or set(peaks)
            != {'cpu_cores', 'ram_bytes', 'scratch_bytes', 'gpu_memory_bytes'}
            or any(
                type(value) not in
                (int, float) or not math.isfinite(value) or value <= 0
                for value in (*peaks.values(), resources.get('seconds')))
            or type(resources.get('gpus')) is not int or resources['gpus'] < 1
            or not isinstance(resources.get('gpu_class'), str)
            or not resources['gpu_class'].strip()):
        raise ValueError(f'{label}: resource measurements are incomplete')
    if any(
            resources.get(name) != row[name]
            for name in ('scale', 'task', 'arm'
                         )) or variant_name(resources) != variant_name(row):
        raise ValueError(f'{label}: resource case identity differs')
    for observed, binding in (('wheel_sha256', 'wheel_sha256'),
                              ('input_manifest_sha256',
                               'input_manifest_sha256'), ('harness_sha256',
                                                          'runner_sha256')):
        if resources.get(observed) != bindings[binding]:
            raise ValueError(f'{label}: resource {observed} binding differs')


def _validate_result(row, expected, identity, *, measurement, tier):
    label = case_label(row)
    if row['task'] != expected['task']:
        raise ValueError(
            f'{label}: semantic task differs from the declared variant')
    if row.get('status') not in ('PASS', 'MEASURED'):
        raise ValueError(f'{label}: case did not complete')
    if not measurement and (row.get('measurement_only')
                            or row.get('mode') == 'reference'):
        raise ValueError(f'{label}: measurement results cannot qualify')
    allowed = ('PASS', 'MEASURED') if measurement else ('PASS', )
    predictions = row.get('prediction_checks')
    if not isinstance(predictions,
                      dict) or predictions.get('status') not in allowed:
        raise ValueError(f'{label}: prediction checks failed or are missing')
    coverage = row.get('training_coverage', {})
    if (not isinstance(coverage, dict) or coverage.get('status') != 'PASS'
            or any(
                type(coverage.get(key)) is not int or coverage[key] < 2
                for key in ('distinct_trials', 'max_completed_epochs'))):
        raise ValueError(f'{label}: actual training coverage is insufficient')
    if (type(row.get('physical_fits')) is not int
            or row['physical_fits'] < coverage['distinct_trials']
            or not isinstance(row.get('fit_id'), str) or not row['fit_id']
            or not _sha(row.get('checkpoint_sha256'))):
        raise ValueError(f'{label}: completed fit evidence is missing')
    replay = row.get('reference_replay')
    if tier == 'weekly' and not measurement and (
            not isinstance(replay, dict) or replay.get('status') != 'PASS'):
        raise ValueError(f'{label}: reference replay failed or is missing')
    bindings = row.get('bindings', {})
    if not isinstance(bindings, dict):
        raise ValueError(f'{label}: result identity bindings are invalid')
    for name in BINDINGS:
        wanted = identity.get(name)
        if (measurement and name == 'reference_sha256' and wanted is None
                and name in bindings and bindings[name] is None):
            continue
        if not _sha(wanted) or bindings.get(name) != wanted:
            raise ValueError(f'{label}: {name} binding differs')
    inputs = identity.get('input_manifest_sha256')
    wanted = expected.get('input_manifest_sha256') or (inputs.get(
        row['scale']) if isinstance(inputs, dict) else inputs)
    if not _sha(wanted) or bindings.get('input_manifest_sha256') != wanted:
        raise ValueError(f'{label}: input_manifest_sha256 binding differs')
    plan = bindings.get('plan_sha256')
    if not _sha(plan) or (expected.get('plan_sha256') is not None
                          and plan != expected['plan_sha256']):
        raise ValueError(f'{label}: effective plan binding differs')
    _validate_resources(row, bindings)
    validate_score(row['task'], row.get('score'))


def _read(path):
    def invalid(value):
        raise ValueError(f'Non-finite JSON value: {value}')

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def combine(result_paths, *, expected_cases, reference, policy, identity,
            output, measurement=False, tier='nightly',
            canonical_executions=None):
    """Write a verdict even when independent case evidence is incomplete."""
    report = dict(format='kumo-engine-nightly-combined-v2', status='FAIL',
                  qualification='NOT_QUALIFIED', cases=[], repeats=[],
                  errors=[], learning=dict(status='FAIL',
                                           tasks={}), quality_reports={})
    errors = report['errors']
    try:
        if (not isinstance(reference, dict) or not isinstance(identity, dict)
                or canonical_executions is not None
                and not isinstance(canonical_executions, dict)):
            raise ValueError(
                'Collection configuration must contain JSON objects')
        expected = _roster(expected_cases)
        if not expected:
            raise ValueError('The expected case roster is empty')
        scales = {key[0] for key in expected}
        inputs = identity.get('input_manifest_sha256')
        if isinstance(inputs, dict) and set(inputs) != scales:
            raise ValueError('Expected roster omits a configured input scale')
        validate_policy(policy, expected_cases)
        references = _index(reference.get('cases', []))
        if canonical_executions and set(canonical_executions) - {
                case_label(case)
                for case in expected_cases
        }:
            raise ValueError('Canonical selection contains an unexpected case')
        if not measurement:
            if reference.get('status') != 'APPROVED' or policy.get(
                    'status') != 'APPROVED':
                errors.append(
                    'Reference and policy must be explicitly approved')
            if set(references) != set(expected):
                errors.append(
                    'Reference roster differs from the configured cases')
            if any(references[key]['task'] != expected[key]['task']
                   for key in references.keys() & expected.keys()):
                errors.append(
                    'Reference semantic task differs from the declared variant')
        groups, executions, fits = defaultdict(list), set(), set()
        for path in result_paths:
            try:
                row = _read(path)
                key = case_key(row)
                if key not in expected:
                    raise ValueError('Unexpected case or scale')
                execution = row.get('execution_id')
                if not isinstance(
                        execution,
                        str) or not execution or execution in executions:
                    raise ValueError('Missing or duplicate execution ID')
                executions.add(execution)
                groups[key].append(row)
            except (OSError, TypeError, KeyError, ValueError) as error:
                errors.append(f'{Path(path).name}: {error}')
        for key, case in expected.items():
            candidates = groups.get(key, [])
            label = case_label(case)
            chosen_id = (canonical_executions
                         or {}).get(label, case.get('execution_id'))
            if chosen_id is None and len(candidates) == 1:
                canonical = candidates[0]
            else:
                selected = [
                    row for row in candidates
                    if row['execution_id'] == chosen_id
                ]
                if len(selected) != 1:
                    errors.append(
                        f'{label}: missing or ambiguous preselected execution')
                    continue
                canonical = selected[0]
            report['cases'].append(canonical)
            for row in candidates:
                try:
                    _validate_result(row, case, identity,
                                     measurement=measurement, tier=tier)
                    if row['fit_id'] in fits:
                        raise ValueError(f'{label}: fit evidence is reused')
                    fits.add(row['fit_id'])
                    if row['bindings']['plan_sha256'] != canonical['bindings'][
                            'plan_sha256']:
                        raise ValueError(
                            f'{label}: repeat plan differs from canonical')
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(str(error))
                if row is not canonical:
                    report['repeats'].append(row)
        if len(report['cases']) != len(expected):
            errors.append('The canonical case roster is incomplete')
        effective_reference = []
        proposal = []
        for row in report['cases']:
            key = case_key(row)
            previous = references.get(key)
            if measurement:
                previous = dict(case_identity(row),
                                baseline=row.get('baseline'), score=None)
            if previous is None:
                continue
            validate_score(row['task'], previous.get('baseline'))
            effective_reference.append(previous)
            proposal.append(
                dict(case_identity(row), score=row['score'],
                     baseline=previous['baseline']))
            if row['arm'] == 'normal':
                band = policy.get('reference_bands',
                                  {}).get(row['scale'],
                                          {}).get(variant_name(row))
                report['quality_reports'][case_label(row)] = quality_report(
                    row, previous, band)
        if len(effective_reference) == len(expected) and len(
                report['cases']) == len(expected):
            report['learning'] = check_learning(report['cases'],
                                                effective_reference,
                                                policy['learning'], tier=tier)
            if report['learning']['status'] == 'FAIL':
                errors.append('Individual or paired learning checks failed')
            elif report['learning']['status'] == 'MEASURED' and not measurement:
                errors.append(
                    'Learning calibration remains unapproved or pending')
        if measurement:
            report['reference_proposal'] = dict(status='PROPOSAL_NOT_APPROVED',
                                                cases=proposal)
        if not errors:
            report['status'] = 'MEASURED' if measurement else 'PASS'
            report[
                'qualification'] = 'NOT_QUALIFIED' if measurement else 'QUALIFIED'
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    _save(output, report)
    return report


def _relative_path(value):
    if (not isinstance(value, str) or not value or '\\' in value
            or PurePosixPath(value).is_absolute()
            or PurePosixPath(value).as_posix() != value
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError(
            'Archive payload paths must be canonical relative paths')
    return value


def _training_proof(value):
    if not isinstance(value, dict):
        raise ValueError(
            'The archive needs generated training-table identities')
    result = {}
    for name in ('stages', 'normal_stages'):
        stages = value.get(name)
        if not isinstance(stages, dict) or set(stages) != {
                'train', 'val', 'test'
        }:
            raise ValueError('Training proof needs all three nonempty stages')
        result[name] = {}
        for stage, record in stages.items():
            if (not isinstance(record, dict)
                    or set(record) != {'rows', 'sha256'}
                    or type(record['rows']) is not int or record['rows'] <= 0
                    or not _sha(record['sha256'])):
                raise ValueError(
                    'Training proof has invalid counts or label hashes')
            result[name][stage] = dict(record)
    if not isinstance(value.get('generation_plan'), dict):
        raise ValueError('Training proof needs its generation plan')
    try:
        result['generation_plan'] = json.loads(
            json.dumps(value['generation_plan'], allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            'Training generation plan must be finite JSON') from error
    return result


def archive_proposal(rows, *, reference_identity, input_manifest_sha256,
                     recipe_sha256, campaign=None, expected_cases=None):
    """Describe portable payloads; packaging separately checks regular files."""
    if (not isinstance(rows, (list, tuple)) or not rows
            or not isinstance(reference_identity, dict)
            or set(reference_identity) != {'version', 'wheel_sha256'}
            or not isinstance(reference_identity.get('version'), str)
            or not reference_identity['version'].strip()
            or not _sha(reference_identity.get('wheel_sha256'))
            or not isinstance(input_manifest_sha256, dict)
            or not input_manifest_sha256
            or not all(_sha(value) for value in input_manifest_sha256.values())
            or not _sha(recipe_sha256)):
        raise ValueError(
            'Archive proposal needs reference, input, and recipe identities')
    indexed = _index(rows)
    if expected_cases is None:
        expected_cases = [
            dict(scale=scale, task=task, arm=arm, seed=0)
            for scale in input_manifest_sha256 for task in TASKS
            for arm in ('normal', 'control')
        ]
    expected = _roster(expected_cases)
    if (set(indexed) != set(expected)
            or {key[0]
                for key in expected} != set(input_manifest_sha256)
            or any(row['task'] != expected[key]['task']
                   for key, row in indexed.items())):
        raise ValueError(
            'Archive proposal needs the complete canonical case roster')
    for scale, task in {(key[0], key[1]) for key in indexed}:
        if len({key[3] for key in indexed if key[:2] == (scale, task)}) != 1:
            raise ValueError(
                'Archived normal/control pairs must share their seed')
    case_inputs = {}
    for key, case in expected.items():
        checksum = case.get('input_manifest_sha256',
                            input_manifest_sha256[case['scale']])
        if not _sha(checksum):
            raise ValueError('Archive case input binding must be a SHA256')
        case_inputs[key] = checksum
    manifest = dict(
        format='kumo-engine-nightly-archive-v1', status='PROPOSAL_NOT_APPROVED',
        reference_identity=dict(reference_identity),
        input_manifest_sha256=dict(input_manifest_sha256),
        case_input_manifest_sha256={
            case_label(expected[key]): checksum
            for key, checksum in case_inputs.items()
            if checksum != input_manifest_sha256[key[0]]
        }, collector_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        recipe_sha256=recipe_sha256, files={}, cases=[])
    executions = set()
    for row in indexed.values():
        prefix = case_label(row)
        input_checksum = case_inputs[case_key(row)]
        execution = row.get('execution_id')
        if (row.get('status') not in ('MEASURED', 'PASS')
                or not isinstance(execution, str) or not execution
                or execution in executions):
            raise ValueError(
                'Archive needs complete, independent case executions')
        executions.add(execution)
        bindings = row.get('bindings', {})
        if (not isinstance(bindings, dict) or bindings.get('wheel_sha256')
                != reference_identity['wheel_sha256']
                or bindings.get('input_manifest_sha256') != input_checksum
                or bindings.get('recipe_sha256') != recipe_sha256):
            raise ValueError(
                'Archived case wheel, input, or recipe identity differs')
        artifacts = row.get('artifacts')
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(
                'Archive needs a nonempty case artifact checksum map')
        for name, checksum in artifacts.items():
            _relative_path(name)
            if not _sha(checksum):
                raise ValueError('Archive artifact checksum is invalid')
            manifest['files'][prefix + '/' + name] = checksum
        model = row.get('model')
        if not isinstance(model, dict):
            raise ValueError('Archive needs its selected model payload')
        model_path = _relative_path(model.get('path'))
        files = model.get('files')
        if not isinstance(files, dict) or not files:
            raise ValueError(
                'Archive model payload needs its file checksum map')
        for name, checksum in files.items():
            _relative_path(name)
            if not _sha(checksum) or artifacts.get(model_path + '/' +
                                                   name) != checksum:
                raise ValueError(
                    'Archive model file differs from the case artifacts')
        if {
                name
                for name in artifacts
                if name == model_path or name.startswith(model_path + '/')
        } != {model_path + '/' + name
              for name in files}:
            raise ValueError('Archive model payload contains unrecorded files')
        checkpoint = row.get('checkpoint_sha256')
        if not _sha(checkpoint) or checkpoint not in files.values():
            raise ValueError(
                'The selected checkpoint is missing from the model payload')
        replay = dict(
            model=dict(path=prefix + '/' + model_path, files=dict(files)),
            checkpoint_sha256=checkpoint)
        for name, source in (('population', 'population'),
                             ('predictions', 'canonical_predictions')):
            record = row.get(source)
            if not isinstance(record, dict):
                raise ValueError(
                    'Archive needs population and prediction payloads')
            name_in_case = _relative_path(record.get('path'))
            checksum = record.get('sha256')
            if not _sha(checksum) or artifacts.get(name_in_case) != checksum:
                raise ValueError(
                    'Archive prediction payload differs from the case artifacts'
                )
            replay[name] = dict(path=prefix + '/' + name_in_case,
                                sha256=checksum)
        if 'raw_limit' in row:
            if type(row['raw_limit']) is not int or row['raw_limit'] < 1:
                raise ValueError(
                    'Archive ranking candidate count must be positive')
            replay['raw_limit'] = row['raw_limit']
        manifest['cases'].append(
            dict(case_identity(row), execution_id=execution,
                 input_manifest_sha256=input_checksum,
                 training_table=_training_proof(row.get('training_table')),
                 replay=replay))
    if campaign is not None:
        if not isinstance(campaign, dict):
            raise ValueError('Campaign provenance must be a report object')
        try:
            manifest['campaign'] = json.loads(
                json.dumps(campaign, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(
                'Campaign provenance must be finite, acyclic JSON') from error
        repeats = manifest['campaign'].get('repeats', [])
        if not isinstance(repeats, list):
            raise ValueError('Campaign repeats must be a list of case records')
        manifest['repeat_payloads'] = []
        prefixes = set()
        for row in repeats:
            key = case_key(row)
            execution = row.get('execution_id')
            if key not in indexed or not isinstance(execution,
                                                    str) or not execution:
                raise ValueError(
                    'Repeat payload needs a configured case and execution ID')
            if row['task'] != indexed[key]['task']:
                raise ValueError(
                    'Repeat semantic task differs from its workload variant')
            try:
                suffix = hashlib.sha256(execution.encode()).hexdigest()
            except UnicodeError as error:
                raise ValueError(
                    'Repeat execution ID must be valid text') from error
            prefix = 'repeats/' + case_label(row) + '/' + suffix
            if prefix in prefixes:
                raise ValueError('Repeat payload execution is duplicated')
            prefixes.add(prefix)
            bindings = row.get('bindings', {})
            if not isinstance(bindings, dict):
                raise ValueError('Repeat identity bindings must be an object')
            for name, expected in (('wheel_sha256',
                                    reference_identity['wheel_sha256']),
                                   ('input_manifest_sha256', case_inputs[key]),
                                   ('recipe_sha256', recipe_sha256)):
                accepted = (expected, )
                if (name == 'input_manifest_sha256'
                        and row.get('status') not in ('MEASURED', 'PASS')):
                    accepted += (input_manifest_sha256[row['scale']], )
                if ((row.get('status') in ('MEASURED', 'PASS')
                     or name in bindings)
                        and bindings.get(name) not in accepted):
                    raise ValueError(
                        'Repeat wheel, input, or recipe identity differs')
            artifacts = row.get('artifacts', {})
            if not isinstance(artifacts, dict):
                raise ValueError('Repeat artifacts must be a checksum map')
            for name, checksum in artifacts.items():
                _relative_path(name)
                if not _sha(checksum):
                    raise ValueError('Repeat artifact checksum is invalid')
                manifest['files'][prefix + '/' + name] = checksum
            payload = dict(case_identity(row), execution_id=execution,
                           path=prefix)
            if 'input_manifest_sha256' in bindings:
                payload['input_manifest_sha256'] = bindings[
                    'input_manifest_sha256']
            manifest['repeat_payloads'].append(payload)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('combine', ))
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measurement', action='store_true')
    parser.add_argument('--tier', choices=('nightly', 'weekly'),
                        default='nightly')
    parser.add_argument('--canonical-executions', type=Path)
    parser.add_argument('results', type=Path, nargs='+')
    args = parser.parse_args(argv)
    specification = _read(args.specification)
    result = combine(
        args.results, expected_cases=specification['cases'],
        reference=specification['reference'], policy=specification['policy'],
        identity=specification['identity'], output=args.output,
        measurement=args.measurement, tier=args.tier,
        canonical_executions=_read(args.canonical_executions)
        if args.canonical_executions else None)
    return 0 if result['status'] in ('PASS', 'MEASURED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
