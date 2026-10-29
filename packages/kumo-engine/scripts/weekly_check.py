#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the eleven frozen weekly cases and compare matching saved evidence."""

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROFILES = {
    'Q13': ('streaming/100M', 'next_titles', 'recall@50'),
    'Q14': ('community/100M', 'messages_30d', 'mae'),
    'Q15': ('payments/100M core', 'flagged', 'auroc'),
    'Q16': ('fraud/100M', 'account_flagged', 'auroc'),
    'Q17': ('commerce/1B', 'next_merchants', 'recall@10'),
    'P2': ('commerce/25B', 'next_merchants', 'recall@10'),
}
IDENTITY = ('source_sha256', 'data_sha256', 'query_sha256', 'split_sha256',
            'model_sha256', 'control_sha256', 'dependency_sha256',
            'device_profile_sha256')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def pinned(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(
            '[0-9a-f]{64}', expected) or digest(path) != expected:
        raise ValueError(f'File fingerprint differs: {path}')
    return Path(path)


def load_manifest(path, expected):
    value = read(pinned(path, expected))
    rows = value.get('cases', [])
    reference = value.get('reference', {})
    performance = value.get('performance_reference', {})
    if (value.get('schema') != 1 or reference.get('version') != '1.0.0' or
            not re.fullmatch('[0-9a-f]{64}', str(reference.get('wheel_sha256')))
            or performance.get('version') != '1.0.0+group5.20261023.1'
            or not isinstance(rows, list) or len(rows) != 11):
        raise ValueError('Use the approved eleven-case reference manifest')
    expected_roster = {
        (task, condition)
        for task in PROFILES
        for condition in (('fixed_scale_workload', ) if task == 'P2' else (
            'normal', 'scrambled_answer_control'))
    }
    observed = []
    ids = []
    for row in rows:
        task = row.get('workload')
        if (task not in PROFILES or type(row.get('seed')) is not int
                or row['seed'] != 0 or tuple(
                    row.get(key)
                    for key in ('dataset', 'query', 'metric')) != PROFILES[task]
                or not isinstance(row.get('id'), str)
                or not re.fullmatch('[A-Za-z0-9_-]+', row['id'])
                or not isinstance(row.get('hardware_profile'), str)
                or not row['hardware_profile']):
            raise ValueError('The weekly case, population, or metric differs')
        ids.append(row['id'])
        observed.append((task, row.get('condition')))
    if set(observed) != expected_roster or len(set(ids)) != 11:
        raise ValueError('The weekly roster is incomplete or repeated')
    return value


def verify_case_inputs(spec):
    identity = spec.get('identity', {})
    if (set(identity) != set(IDENTITY)
            or any(not isinstance(value, str)
                   or not re.fullmatch('[0-9a-f]{64}', value)
                   for value in identity.values())
            or not isinstance(spec.get('inputs'), dict) or not spec['inputs']):
        raise ValueError(
            'Freeze every case identity and input before execution')
    for name, expected in spec['inputs'].items():
        pinned(name, expected)
    worker = spec.get('worker', {})
    pinned(worker['path'], worker['sha256'])
    capacity_spec = spec.get('capacity', {})
    capacity = read(pinned(capacity_spec['path'], capacity_spec['sha256']))
    if (capacity.get('status') != 'ADMITTED'
            or capacity.get('case_id') != spec['id']
            or capacity.get('hardware_profile') != spec['hardware_profile']
            or capacity.get('native_capacity_checked') is not True
            or not capacity.get('gpu_name')
            or not isinstance(capacity.get('observed_hardware'), dict)):
        raise ValueError(
            'Native case capacity is not admitted on this hardware')
    verify_hardware(capacity['observed_hardware'],
                    capacity['observed_hardware'])
    return capacity


def verify_hardware(expected, actual):
    fields = ('instance_id', 'gpu_uuid', 'driver', 'vcpus',
              'memory_total_bytes', 'gpu_memory_bytes')
    try:
        valid_uuid = uuid.UUID(str(
            actual['gpu_uuid']).removeprefix('GPU-')) == uuid.UUID(
                str(expected['gpu_uuid']).removeprefix('GPU-'))
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(
            'Actual hardware identity differs from native admission') from error
    if (not valid_uuid or any(name not in expected or name not in actual
                              for name in fields)
            or any(actual[name] != expected[name]
                   for name in fields if name != 'gpu_uuid')
            or any(
                type(expected[name]) is not int or expected[name] <= 0
                for name in ('vcpus', 'memory_total_bytes', 'gpu_memory_bytes'))
            or
            not re.fullmatch('i-[0-9a-f]{8,17}', str(expected['instance_id']))
            or not isinstance(expected['driver'], str)
            or not expected['driver']):
        raise ValueError(
            'Actual hardware profile differs from native admission')


def current_hardware(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            'The admitted hardware requires exactly one visible CUDA GPU')
    properties = torch.cuda.get_device_properties(0)
    identifier = str(properties.uuid)
    drivers = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,driver_version',
        '--format=csv,noheader,nounits'
    ], text=True)
    [driver] = [
        row.split(',')[1].strip() for row in drivers.splitlines()
        if uuid.UUID(row.split(',')[0].strip().removeprefix('GPU-')) ==
        uuid.UUID(identifier.removeprefix('GPU-'))
    ]
    instances = [
        path.read_text().strip()
        for path in (Path('/sys/devices/virtual/dmi/id/board_asset_tag'),
                     Path('/var/lib/cloud/data/instance-id'))
        if path.is_file() and os.access(path, os.R_OK)
    ]
    instance = next(
        (value
         for value in instances if re.fullmatch('i-[0-9a-f]{8,17}', value)),
        None)
    memory = next(
        int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemTotal:'))
    return dict(instance_id=instance, gpu_uuid=identifier, driver=driver,
                vcpus=os.cpu_count(), memory_total_bytes=memory,
                gpu_memory_bytes=properties.total_memory)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric_score(spec, row, *, baseline=False):
    ids = row.get('entity_ids')
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError('Prediction entity IDs are missing or repeated')
    if spec['metric'].startswith('recall@'):
        performance_only = spec['workload'] == 'P2'
        catalog = row['catalog']
        targets = None if performance_only else row['target_ids']
        prefix = 'baseline_' if baseline and not performance_only else ''
        predictions, scores = row[prefix + 'ranked_ids'], row[prefix +
                                                              'ranked_scores']
        excluded = row.get('excluded_ids', [[] for _ in ids])
        if (row.get('target_format') != 'ranked_link_v1' or not catalog
                or len(set(catalog)) != len(catalog)
                or catalog != sorted(catalog)
                or targets is not None and len(targets) != len(ids) or any(
                    len(values) != len(ids)
                    for values in (predictions, scores, excluded))):
            raise ValueError('Missing ranked prediction coordinates')
        candidates = set(catalog)
        k = int(spec['metric'].split('@')[1])
        recalls = []
        for index, (items, values,
                    blocked) in enumerate(zip(predictions, scores, excluded)):
            answers = targets[index] if targets is not None else []
            if (len(set(answers)) != len(answers)
                    or len(set(blocked)) != len(blocked)
                    or len(set(items)) != len(items)
                    or not set(answers + items + blocked) <= candidates
                    or set(answers) & set(blocked) or set(items) & set(blocked)
                    or len(items) != min(k,
                                         len(catalog) - len(blocked))
                    or len(items) != len(values) or not all(map(finite, values))
                    or any(
                        values[i] < values[i + 1] or values[i] == values[i + 1]
                        and items[i] > items[i + 1]
                        for i in range(len(items) - 1))):
                raise ValueError(
                    'Ranked predictions are incomplete or nonfinite')
            recalls.append(
                len(set(answers) & set(items)) /
                len(answers) if answers else None)
        return None if performance_only or None in recalls else sum(
            recalls) / len(recalls)
    predictions = row['baseline_predictions' if baseline else 'predictions']
    targets = row['targets']
    if not isinstance(targets, list) or len(targets) != len(ids) or len(
            predictions) != len(ids):
        raise ValueError('Prediction population differs from reserved targets')
    values = []
    for target, prediction in zip(targets, predictions):
        if isinstance(prediction, list) and len(prediction) == 1:
            prediction = prediction[0]
        if not finite(prediction) or not (finite(target) or spec['metric']
                                          == 'auroc' and type(target) is bool):
            raise ValueError('Predictions or targets are nonfinite')
        values.append(prediction)
    if spec['metric'] == 'mae':
        return sum(
            abs(target - value)
            for target, value in zip(targets, values)) / len(ids)
    if any(target not in (0, 1) for target in targets):
        raise ValueError('AUROC requires binary target values')
    if not 0 < sum(targets) < len(targets):
        return None
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError('Binary probabilities are outside [0, 1]')
    ordered = sorted(zip(values, targets))
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum(
            target for _, target in ordered[start:end])
        start = end
    positives = sum(targets)
    return (rank_sum - positives *
            (positives + 1) / 2) / (positives * (len(ids) - positives))


def verify_result(spec, row):
    expected_arm = 'independent' if spec[
        'condition'] == 'scrambled_answer_control' else 'learnable'
    checks = ('fit', 'prediction', 'repeat_prediction', 'partial_prediction',
              'reload', 'hidden_answers', 'chronology')
    if (row.get('status') != 'MEASURED'
            or row.get('identity') != spec['identity'] or row.get('seed') != 0
            or type(row.get('seed')) is not int or row.get('training_seed') != 0
            or type(row.get('training_seed')) is not int
            or row.get('case') != expected_arm or not row.get('fit_id')
            or row.get('metric') != spec['metric']
            or row.get('batch_retries_disabled') is not True
            or 'calibration_fault' in row
            or any(row.get('checks', {}).get(key) != 'PASS' for key in checks)):
        raise ValueError('Missing, mismatched, or failed weekly model evidence')
    stages = row.get('stage_sha256', {})
    fingerprints = [
        row.get('data_sha256'),
        row.get('plan_sha256'), *stages.values()
    ]
    if (set(stages) != {'train', 'val', 'test'}
            or any(not re.fullmatch('[0-9a-f]{64}', str(value))
                   for value in fingerprints)):
        raise ValueError('Missing data, split, or actual plan fingerprints')
    modules = Path(__file__).resolve().parents[1] / 'test'
    gpu = runpy.run_path(str(modules / 'wheel_gpu_learning.py'))
    measurements = runpy.run_path(str(modules / 'wheel_measurements.py'))
    gpu['verify_devices'](row, ('training', 'prediction', 'reload_and_predict'))
    gpu['verify_reload_identity'](row)
    measurements['verify_training_counts'](row)
    step = row.get('selected_checkpoint_global_step')
    if type(
            step
    ) is not int or not 0 < step <= row['training_work'][0]['optimizer_steps']:
        raise ValueError(
            'The saved checkpoint has no matching completed training step')
    for baseline, field in ((False, 'score'), (True, 'baseline_score')):
        score = metric_score(spec, row, baseline=baseline)
        if score is None:
            if (row.get(field) is not None
                    or not row.get('quality_missing_reason')
                    or spec['workload'] == 'P2'
                    and row['quality_missing_reason'] !=
                    'P2 fixed performance workload has no frozen quality answers'
                ):
                raise ValueError(
                    'Report the undefined native metric and its cause')
        elif not finite(row.get(field)) or not math.isclose(
                score, row[field], abs_tol=1e-12, rel_tol=0):
            raise ValueError(
                'The reported score differs from complete saved predictions')
    operations = row.get('operation_measurements')
    if not isinstance(operations, list) or not operations or any(
            not finite(item.get('seconds')) or item['seconds'] < 0
            for item in operations):
        raise ValueError('Operation measurements are missing or nonfinite')


def inventory(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Evidence cannot contain symbolic links')
        if path.is_file() and path != root / 'weekly-case.json':
            files[path.relative_to(root).as_posix()] = digest(path)
    return files


def verify_candidate_build(path, version, wheel_sha256):
    build = read(path)
    if (build.get('version') != version
            or build.get('bundle_version') != version
            or build.get('wheel_sha256') != wheel_sha256
            or not build.get('wheel')
            or any(not re.fullmatch('[0-9a-f]{40}', str(build.get(key)))
                   for key in ('revision', 'tree'))):
        raise ValueError(
            'The candidate build receipt does not bind its exact source and wheel'
        )
    return build


def training_critical_path(rows):
    starts, ends = [], []
    for row in rows:
        if not row.get('training_started_utc') or not row.get(
                'training_completed_utc'):
            return None
        start = datetime.fromisoformat(row['training_started_utc'])
        end = datetime.fromisoformat(row['training_completed_utc'])
        if start.utcoffset() is None or end.utcoffset() is None or end <= start:
            raise ValueError('Actual fit timestamps are invalid')
        starts.append(start)
        ends.append(end)
    return (max(ends) - min(starts)).total_seconds() if rows else None


def run(args, manifest):
    [spec] = [row for row in manifest['cases'] if row['id'] == args.case]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema=1, status='RUNNING', case_id=spec['id'], mode=args.mode,
        version=args.version, wheel_sha256=args.wheel_sha256,
        manifest_sha256=args.manifest_sha256, identity=spec.get('identity'),
        hardware_profile=spec['hardware_profile'], physical_fits=0,
        case_spec=spec, started_utc=datetime.now(timezone.utc).isoformat(),
        quality_policy='report_only', resource_policy='report_only',
        performance_reference=manifest['performance_reference'])
    fits = output / 'fits.json'
    save(fits, [])
    save(output / 'weekly-case.json', report)
    lock = None
    started = time.monotonic()
    try:
        shutil.copyfile(args.manifest, output / 'input-manifest.json')
        pinned(output / 'input-manifest.json', args.manifest_sha256)
        lock = args.host_lock.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('The requested GPU host is busy') from error
        capacity = verify_case_inputs(spec)
        shutil.copyfile(spec['capacity']['path'],
                        output / 'native-capacity.json')
        pinned(args.wheel, args.wheel_sha256)
        if args.mode == 'candidate':
            if args.build_receipt is None:
                raise ValueError(
                    'Candidate training requires an exact source build receipt')
            report['build'] = verify_candidate_build(args.build_receipt,
                                                     args.version,
                                                     args.wheel_sha256)
            shutil.copyfile(args.build_receipt, output / 'build.json')
        if (args.mode == 'reference'
                and dict(version=args.version, wheel_sha256=args.wheel_sha256)
                != manifest['reference']):
            raise ValueError(
                'Reference creation requires the approved 1.0.0 wheel')
        if (not sys.flags.isolated or sys.prefix == sys.base_prefix or Path(
                sys.prefix).resolve() != args.expected_prefix.resolve()):
            raise ValueError('Use the declared isolated wheel environment')
        helpers = runpy.run_path(str(Path(__file__).with_name('pr_check.py')))
        report['installation'] = helpers['verify_install'](dict(
            version=args.version, wheel_sha256=args.wheel_sha256), args.wheel)
        with (output / 'pip-check.log').open('w') as log:
            subprocess.run(
                [sys.executable, '-I', '-m', 'pip', '--isolated', 'check'],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        pinned(args.dependency_profile, spec['identity']['dependency_sha256'])
        packages = {
            re.sub('[-_.]+', '-', item.metadata['Name']).lower(): item.version
            for item in importlib.metadata.distributions() if re.sub(
                '[-_.]+', '-', item.metadata['Name']).lower() != 'kumo-engine'
        }
        if packages != read(args.dependency_profile):
            raise ValueError(
                'Installed dependencies differ from the frozen profile')
        save(output / 'dependencies.json', packages)
        import torch
        actual_hardware = current_hardware(torch)
        verify_hardware(capacity['observed_hardware'], actual_hardware)
        report['observed_hardware'] = actual_hardware
        properties = torch.cuda.get_device_properties(0)
        identifier = str(properties.uuid)
        uuid.UUID(identifier.removeprefix('GPU-'))
        if (properties.name != capacity['gpu_name']
                or properties.total_memory <= 0
                or spec['workload'] != 'P2' and properties.name
                != 'NVIDIA RTX PRO 6000 Blackwell Server Edition'):
            raise ValueError(
                'Actual GPU differs from the admitted case hardware')
        report['device'] = dict(gpu=properties.name, gpu_uuid=identifier,
                                gpu_memory_bytes=properties.total_memory,
                                cuda=torch.version.cuda,
                                torch=torch.__version__)
        save(output / 'weekly-case.json', report)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test'))
        with helpers['count_fits'](fits) as physical:
            worker = runpy.run_path(spec['worker']['path'])
            row = worker['run_case'](copy.deepcopy(spec), output / 'worker')
            save(output / 'result.json', row)
        if len(physical) != 1:
            raise ValueError(
                'A weekly case requires exactly one physical model fit')
        helpers['verify_fits'](physical)
        verify_result(spec, row)
        verify_case_inputs(spec)
        helpers['verify_install'](dict(version=args.version,
                                       wheel_sha256=args.wheel_sha256),
                                  args.wheel)
        verify_artifacts(output, inventory(output))
        report.update(status='MEASURED', result=row)
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        if lock is not None:
            lock.close()
        report.update(seconds=time.monotonic() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      physical_fits=len(read(fits)))
        try:
            report['files'] = inventory(output)
        except BaseException as error:
            report.update(status='FAIL',
                          error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save(output / 'weekly-case.json', report)
    return report


def verify_artifacts(root, files):
    actual = inventory(root)
    if actual != files:
        raise ValueError('Missing, changed, or unlisted case evidence files')
    names = {Path(name).name for name in files}
    required = {
        name + '.json'
        for name in ('requested-plan', 'selected-model-plan', 'generation-plan',
                     'resolved_model_plan', 'trainer_config',
                     'runtime_manifest', 'feature_schema', 'device',
                     'measurements', 'memory')
    }
    if not required <= names or not any(
            name.endswith('model/checkpoint') for name in files):
        raise ValueError(
            'Complete actual plans, resources, or saved model are missing')
    for label in ('predictions', 'repeat-predictions', 'partial-predictions',
                  'reloaded-predictions'):
        if not any(label in name.split('/') or Path(name).stem == label
                   for name in files if name.endswith('.parquet')):
            raise ValueError('A required raw prediction export is missing')
    for name in files:
        path = root / name
        if not path.stat().st_size and (path.name in required
                                        or name.endswith('model/checkpoint')
                                        or path.suffix == '.parquet'):
            raise ValueError('A required evidence file is empty')
        if Path(name).name == 'memory.json' and not read(path).get(
                'overall', {}).get('samples'):
            raise ValueError('CPU and GPU resource sampling did not complete')


def manifest_scope(value):
    fields = ('id', 'workload', 'dataset', 'query', 'seed', 'condition',
              'metric', 'hardware_profile')
    return dict(
        schema=value['schema'], reference=value['reference'],
        performance_reference=value['performance_reference'],
        roster=sorted([{
            name: spec[name]
            for name in fields
        } for spec in value['cases']], key=lambda row: row['id']), policies={
            name: item
            for name, item in value.items()
            if name.endswith('_policy') or name in ('policies', 'policy',
                                                    'strict', 'deferred')
        })


def load_case(path, spec, mode, manifest):
    report = read(path)
    root = path.parent
    if (report.get('schema') != 1 or report.get('status') != 'MEASURED'
            or report.get('mode') != mode or report.get('case_id') != spec['id']
            or report.get('identity') != spec['identity']
            or report.get('hardware_profile') != spec['hardware_profile']
            or report.get('physical_fits') != 1):
        raise ValueError(
            'A weekly case is incomplete or differs from its frozen identity')
    verify_artifacts(root, report.get('files', {}))
    original = load_manifest(root / 'input-manifest.json',
                             report['manifest_sha256'])
    recorded = next(row for row in original['cases'] if row['id'] == spec['id'])
    if (recorded != spec or report.get('case_spec') != spec
            or manifest_scope(original) != manifest_scope(manifest)):
        raise ValueError(
            'The selected case or global manifest scope changed after it was frozen'
        )
    row = read(root / 'result.json')
    if row != report.get('result'):
        raise ValueError('The case report differs from its saved result')
    verify_result(spec, row)
    if mode == 'candidate':
        build = verify_candidate_build(root / 'build.json', report['version'],
                                       report['wheel_sha256'])
        if build != report.get('build'):
            raise ValueError(
                'The stored candidate build differs from its case report')
    capacity = read(
        pinned(root / 'native-capacity.json', spec['capacity']['sha256']))
    verify_hardware(capacity['observed_hardware'],
                    report.get('observed_hardware', {}))
    fits = read(root / 'fits.json')
    helpers = runpy.run_path(str(Path(__file__).with_name('pr_check.py')))
    if not isinstance(fits, list) or len(fits) != 1:
        raise ValueError(
            'The physical fit receipt must contain exactly one model')
    helpers['verify_fits'](fits)
    batches = read(root / 'cuda-batches.json')
    if any(
            type(batches.get(stage)) is not int or batches[stage] <= 0
            for stage in ('training', 'prediction')):
        raise ValueError(
            'Missing completed training or prediction CUDA batches')
    device = report.get('device', {})
    uuid.UUID(str(device.get('gpu_uuid')).removeprefix('GPU-'))
    if (not device.get('gpu_memory_bytes')
            or spec['workload'] != 'P2' and device.get('gpu')
            != 'NVIDIA RTX PRO 6000 Blackwell Server Edition'):
        raise ValueError('The case ran on a different GPU profile')
    return report


def milestone_coverage():
    return dict(primary_block_complete=True, weekly_milestone_complete=False,
                ancillary_checks=dict(
                    B6=dict(status='MEASURED_FROM_PRIMARY_MODELS', physical_fits=0, models=11),
                    P1=dict(status='MISSING', physical_fits=0, required_cpu_operations=5),
                    B9=dict(status='MISSING', physical_fits=0, required_matching_gpus=2)),
                remaining_coverage='Bind and verify P1 and B9 workers before claiming the complete weekly milestone')


def collect(manifest, paths, output, mode):
    if len(paths) != 11:
        raise ValueError('All eleven completed weekly cases are required')
    specs = {spec['id']: spec for spec in manifest['cases']}
    reports = {}
    for path in paths:
        report = read(path)
        name = report.get('case_id')
        if name not in specs or name in reports:
            raise ValueError('Missing, unknown, or repeated weekly case')
        reports[name] = load_case(path, specs[name], mode, manifest)
    wheels = {(row.get('version'), row.get('wheel_sha256'))
              for row in reports.values()}
    if len(wheels) != 1 or mode == 'reference' and wheels != {
        (manifest['reference']['version'],
         manifest['reference']['wheel_sha256'])
    }:
        raise ValueError(
            'The weekly block mixes wheels or uses the wrong reference')
    if len({row['result']['fit_id'] for row in reports.values()}) != 11:
        raise ValueError('The weekly block repeats fitted model identities')
    output.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copytree(path.parent, output / 'cases' / read(path)['case_id'])
    version, wheel = next(iter(wheels))
    bundle = dict(schema=1, mode=mode, status='MEASURED', version=version,
                  wheel_sha256=wheel, reference=manifest['reference'],
                  performance_reference=manifest['performance_reference'],
                  physical_fits=11, **milestone_coverage(), cases=list(reports),
                  case_specs=manifest['cases'], input_manifest=manifest,
                  files=inventory(output))
    save(output / 'manifest.json', bundle)
    return bundle


def load_bundle(root, expected):
    value = read(pinned(root / 'manifest.json', expected))
    if (value.get('schema') != 1 or value.get('status') != 'MEASURED'
            or value.get('physical_fits') != 11
            or len(value.get('cases', [])) != 11
            or len(set(value['cases'])) != 11):
        raise ValueError('The weekly bundle is incomplete')
    actual = inventory(root)
    actual.pop('manifest.json', None)
    if actual != value.get('files'):
        raise ValueError('Missing, changed, or unlisted weekly bundle evidence')
    specs = {spec['id']: spec for spec in value['case_specs']}
    if set(specs) != set(value['cases']):
        raise ValueError('The weekly bundle has mismatched case identities')
    for name in value['cases']:
        load_case(root / 'cases' / name / 'weekly-case.json', specs[name],
                  value['mode'], value['input_manifest'])
    return value


def verify_reference(manifest, root, expected):
    value = load_bundle(root, expected)
    if (value['mode'] != 'reference' or dict(version=value['version'],
                                             wheel_sha256=value['wheel_sha256'])
            != manifest['reference']):
        raise ValueError('Use the complete approved 1.0.0 reference block')
    expected_cases = {
        spec['id']: (spec['identity'], spec['hardware_profile'])
        for spec in manifest['cases']
    }
    actual = {
        spec['id']: (spec['identity'], spec['hardware_profile'])
        for spec in value['case_specs']
    }
    if actual != expected_cases:
        raise ValueError(
            'The reference case identity differs from the frozen weekly manifest'
        )
    return value


def compare(manifest, candidate, candidate_sha256, reference, reference_sha256,
            output):
    baseline = verify_reference(manifest, reference, reference_sha256)
    current = load_bundle(candidate, candidate_sha256)
    if (baseline.get('mode') != 'reference'
            or current.get('mode') != 'candidate' or dict(
                version=baseline['version'],
                wheel_sha256=baseline['wheel_sha256']) != manifest['reference']
            or set(baseline['cases']) != set(current['cases']) or set(
                current['cases']) != {spec['id']
                                      for spec in manifest['cases']}):
        raise ValueError(
            'Use complete matching reference and fresh candidate weekly blocks')
    reference_fits = {
        read(reference / 'cases' / name /
             'weekly-case.json')['result']['fit_id']
        for name in baseline['cases']
    }
    candidate_fits = {
        read(candidate / 'cases' / name /
             'weekly-case.json')['result']['fit_id']
        for name in current['cases']
    }
    if reference_fits & candidate_fits:
        raise ValueError(
            'A routine candidate comparison requires eleven fresh model fits')
    rows = []
    for spec in manifest['cases']:
        name = spec['id']
        old = read(reference / 'cases' / name / 'weekly-case.json')
        new = read(candidate / 'cases' / name / 'weekly-case.json')
        if (old['identity'] != new['identity']
                or new['identity'] != spec['identity']
                or old['hardware_profile'] != new['hardware_profile']
                or new['hardware_profile'] != spec['hardware_profile']):
            raise ValueError(
                'Reference and candidate case inputs or hardware differ')
        if old['result']['fit_id'] == new['result']['fit_id']:
            raise ValueError(
                'A routine candidate comparison requires fresh model fits')
        before, after = old['result'], new['result']
        rows.append(
            dict(
                case_id=name, metric=spec['metric'],
                reference_score=before['score'], candidate_score=after['score'],
                score_difference=(after['score'] -
                                  before['score'] if after['score'] is not None
                                  and before['score'] is not None else None),
                reference_quality_missing_reason=before.get(
                    'quality_missing_reason'),
                candidate_quality_missing_reason=after.get(
                    'quality_missing_reason'),
                reference_operations=before['operation_measurements'],
                candidate_operations=after['operation_measurements']))
    candidate_rows = [
        read(candidate / 'cases' / name / 'weekly-case.json')['result']
        for name in current['cases']
    ]
    training_seconds = training_critical_path(candidate_rows)
    performance_reason = ('The approved 1.0.0 block does not qualify '
                          'the separate performance reference')
    timing_scope = ('Elapsed time from first actual fit start to last '
                    'actual fit end, including gaps')
    report = dict(
        schema=1, status='MEASURED', cases=rows, **milestone_coverage(), quality_policy='report_only',
        resource_policy='report_only', reference_physical_fits=11,
        candidate_physical_fits=11,
        performance_reference=manifest['performance_reference'],
        performance_comparison_valid=False,
        performance_reason=performance_reason,
        candidate_comparison_executed=True,
        training_critical_path_seconds=training_seconds,
        training_critical_path_scope=timing_scope,
        training_target_seconds=3600,
        training_target_met=(training_seconds < 3600
                             if training_seconds is not None else None))
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'run', 'collect', 'compare', 'verify-reference'):
        command_parser = commands.add_parser(command)
        command_parser.add_argument('--manifest', type=Path, required=True)
        command_parser.add_argument('--manifest-sha256', required=True)
        if command in ('run', 'collect'):
            command_parser.add_argument('--mode',
                                        choices=('reference', 'candidate'),
                                        required=True)
        if command not in ('inspect', 'verify-reference'):
            command_parser.add_argument('--output', type=Path, required=True)
        if command == 'run':
            command_parser.add_argument('--case', required=True)
            command_parser.add_argument('--build-receipt', type=Path)
            command_parser.add_argument('--version', required=True)
            command_parser.add_argument('--wheel-sha256', required=True)
            for name in ('wheel', 'expected-prefix', 'dependency-profile',
                         'host-lock'):
                command_parser.add_argument('--' + name, type=Path,
                                            required=True)
        if command == 'collect':
            command_parser.add_argument('--case-reports', nargs='+', type=Path,
                                        required=True)
        if command == 'verify-reference':
            command_parser.add_argument('--reference', type=Path, required=True)
            command_parser.add_argument('--reference-sha256', required=True)
        if command == 'compare':
            for name in ('candidate', 'reference'):
                command_parser.add_argument('--' + name, type=Path,
                                            required=True)
                command_parser.add_argument('--' + name + '-sha256',
                                            required=True)
    args = parser.parse_args()
    value = load_manifest(args.manifest, args.manifest_sha256)
    if args.command == 'inspect':
        print(
            json.dumps(
                dict(case_ids=[row['id'] for row in value['cases']],
                     physical_fits_started=0)))
    elif args.command == 'verify-reference':
        verify_reference(value, args.reference, args.reference_sha256)
    elif args.command == 'run':
        if args.case not in [row['id'] for row in value['cases']]:
            raise ValueError('Unknown weekly case')
        run(args, value)
    elif args.command == 'collect':
        collect(value, args.case_reports, args.output, args.mode)
    else:
        compare(value, args.candidate, args.candidate_sha256, args.reference,
                args.reference_sha256, args.output)


if __name__ == '__main__':
    main()
