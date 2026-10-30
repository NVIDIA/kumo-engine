#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run eleven weekly cases with learning gates and reference-model replay."""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

PROFILES = {
    'Q13': ('streaming/100M', 'recall@50'),
    'Q14': ('community/100M', 'mae'),
    'Q15': ('payments/100M core', 'auroc'),
    'Q16': ('fraud/100M', 'auroc'),
    'Q17': ('commerce/1B', 'recall@10'),
    'P2': ('commerce/25B', 'recall@10'),
}
ARMS = ('normal', 'control')
ROSTER = {f'{task}-{arm}'
          for task in PROFILES if task != 'P2' for arm in ARMS} | {'P2-fixed'}


def trusted_revision(repo, requested, default_ref):
    def resolve(ref):
        return subprocess.check_output([
            'git', '-C',
            str(repo), 'rev-parse', '--verify', '--end-of-options',
            ref + '^{commit}'
        ], text=True).strip()

    default = resolve(default_ref)
    revision = resolve(requested or default_ref)
    result = subprocess.run([
        'git', '-C',
        str(repo), 'merge-base', '--is-ancestor', revision, default
    ])
    if result.returncode != 0:
        raise ValueError(
            'Candidate revision must belong to the default branch history')
    return revision


def check_build(receipt, version, checksum):
    if (receipt.get('version') != version
            or receipt.get('wheel_sha256') != checksum
            or any(not re.fullmatch('[0-9a-f]{40}', receipt.get(k, ''))
                   for k in ('revision', 'tree'))):
        raise ValueError(
            'Source build receipt does not identify the selected wheel')


def check_capacity(cases, available):
    for case in cases:
        if any(
                type(case['capacity'][key]) is not int or case['capacity'][key]
                <= 0 or available[key] < case['capacity'][key]
                for key in ('memory_bytes', 'scratch_bytes')):
            raise ValueError(
                'Available capacity does not admit every fixed workload')


def capacity(path):
    memory = int(
        next(line.split()[1]
             for line in Path('/proc/meminfo').read_text().splitlines()
             if line.startswith('MemAvailable:'))) * 1024
    limit = Path('/sys/fs/cgroup/memory.max')
    current = Path('/sys/fs/cgroup/memory.current')
    if limit.is_file() and current.is_file() and limit.read_text().strip(
    ).isdigit():
        memory = min(memory,
                     max(0,
                         int(limit.read_text()) - int(current.read_text())))
    return dict(memory_bytes=memory, scratch_bytes=shutil.disk_usage(path).free)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
        return checksum.hexdigest()


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def data_file(root, name, files):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '\\' in name
            or any(part in ('', '.', '..') for part in name.split('/'))
            or relative.suffix.lower() in ('.py', '.pyc', '.sh', '.so')
            or not re.fullmatch('[0-9a-f]{64}', str(files.get(name)))):
        raise ValueError(f'Invalid data path or checksum: {name}')
    root = Path(root).resolve()
    path = root / name
    if (not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or digest(path) != files[name]):
        raise ValueError(f'Missing, linked, or changed data file: {name}')
    return path


def check_learning(candidate, reference):
    """Keep at least half the approved normal-versus-control separation."""
    tasks = {}
    for task in PROFILES:
        if task == 'P2':
            continue
        values = [
            source.get(f'{task}-{arm}') for source in (reference, candidate)
            for arm in ARMS
        ]
        if not all(map(finite, values)):
            tasks[task] = dict(status='FAIL',
                               reason='Missing or nonfinite score')
            continue
        old_normal, old_control, normal, control = values
        direction = -1 if PROFILES[task][1] == 'mae' else 1
        old_gap = direction * (old_normal - old_control)
        gap = direction * (normal - control)
        report_only = task == 'Q15'
        passed = report_only or (old_gap > 0 and gap >= .5 * old_gap)
        tasks[task] = dict(status='PASS' if passed else 'FAIL',
                           policy='report_only' if report_only else 'enforced',
                           reference_gap=old_gap, candidate_gap=gap,
                           minimum_gap=None if report_only else .5 * old_gap)
    return dict(
        status='PASS' if all(v['status'] == 'PASS'
                             for v in tasks.values()) else 'FAIL', tasks=tasks)


def compare_predictions(before, after, *, atol, rtol):
    """Compare every stored coordinate. Labels and ranks must stay exact."""
    if not all(finite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError('Prediction tolerances must be finite and nonnegative')
    ids = before.get('entity_ids', [])
    if (not ids or len(set(ids)) != len(ids) or ids != after.get('entity_ids')
            or before.get('ranked_ids') != after.get('ranked_ids')
            or before.get('decisions') != after.get('decisions')):
        raise ValueError('Prediction population, decisions, or ranks changed')
    left, right = before.get('values', []), after.get('values', [])
    if len(left) != len(ids) or len(right) != len(ids):
        raise ValueError('Prediction population is incomplete')
    maximum = 0.
    for a, b in zip(left, right, strict=True):
        if not isinstance(a, list) or not a or not isinstance(
                b, list) or len(a) != len(b):
            raise ValueError('Prediction coordinates changed')
        for x, y in zip(a, b, strict=True):
            if not finite(x) or not finite(y) or abs(x -
                                                     y) > atol + rtol * abs(x):
                raise ValueError(
                    'Prediction values exceed the approved tolerance')
            maximum = max(maximum, abs(x - y))
    return dict(status='PASS', entities=len(ids), max_abs_difference=maximum,
                atol=atol, rtol=rtol)


def replay_reference(model_path, stored, load, predict, *, atol, rtol):
    model = load(model_path)
    observed = predict(model)
    return compare_predictions(stored, observed, atol=atol, rtol=rtol)


def check_case_result(row, case_id, wheel, *, candidate):
    work = row.get('work', {})
    stages = ('candidate', 'reload',
              'reference_replay') if candidate else ('candidate', 'reload')
    if (row.get('status') != 'PASS' or row.get('case_id') != case_id
            or type(row.get('physical_fits')) is not int
            or row['physical_fits'] != 1 or not row.get('fit_id') or any(
                row.get('runtime', {}).get(k) != v
                for k, v in wheel.items()) or work.get('completed_fits') != 1
            or work.get('optimizer_steps', 0) <= 0
            or work.get('training_batches', 0) <= 0
            or row.get('reload', {}).get('status') != 'PASS' or candidate
            and row.get('reference_replay', {}).get('status') != 'PASS' or any(
                work.get('prediction_batches', {}).get(s, 0) <= 0
                for s in stages)):
        raise ValueError(
            'Case lacks one CUDA fit, complete predictions, or reference replay'
        )


def load_manifest(root, expected, *, verify_files=True):
    path = root / 'manifest.json'
    if digest(path) != expected:
        raise ValueError('Input manifest checksum differs')
    value = read(path)
    cases = value.get('cases', [])
    if (value.get('format') != 'kumo-engine-weekly-input-v2' or len(cases) != 11
            or {c['id']
                for c in cases} != ROSTER
            or not value.get('reference', {}).get('version')
            or not re.fullmatch('[0-9a-f]{64}', value['reference'].get(
                'wheel_sha256', ''))):
        raise ValueError(
            'Use the complete eleven-case manifest and an approved wheel')
    for case in cases:
        task = case['id'].split('-')[0]
        if ((case['dataset'], case['metric']) != PROFILES[task]
                or type(case['seed']) is not int or case['seed'] != 0
                or case['arm'] != case['id'].split('-')[1]):
            raise ValueError('Frozen workload, metric, arm, or seed differs')
        if not all(
                finite(case['tolerance'][k]) and case['tolerance'][k] >= 0
                for k in ('atol', 'rtol')):
            raise ValueError(
                'Approve finite prediction tolerances before execution')
        for key in ('snapshot', 'training_table', 'model_plan',
                    'prediction_table', 'population', 'answers', 'catalog'):
            if case.get(key) is not None:
                data_file(root, case[key],
                          {k: value['files'][k]
                           for k in case['files']})
        if (not isinstance(case['gpu_class']['compute_capability'], list)
                or len(case['gpu_class']['compute_capability']) != 2 or any(
                    type(v) is not int or v < 0
                    for v in case['gpu_class']['compute_capability'])
                or type(case['gpu_class']['minimum_memory_bytes']) is not int
                or case['gpu_class']['minimum_memory_bytes'] <= 0):
            raise ValueError('Bind the GPU class and its minimum memory')
    if verify_files:
        for name in {name for case in cases for name in case['files']}:
            data_file(root, name, value['files'])
    return value


def inventory(root):
    files = {}
    for path in Path(root).rglob('*'):
        if path.is_symlink():
            raise ValueError('Evidence cannot contain symbolic links')
        if path.is_file():
            files[str(path.relative_to(root))] = digest(path)
    return files


def load_reference(root, expected, manifest, manifest_sha256):
    path = data_file(root, 'reference.json', {'reference.json': expected})
    value = read(path)
    if (value.get('format') != 'kumo-engine-weekly-reference-v2'
            or value.get('status') != 'PASS'
            or value.get('manifest_sha256') != manifest_sha256
            or value.get('wheel') != manifest['reference']
            or set(value.get('cases', {})) != ROSTER):
        raise ValueError('Reference approval, wheel, inputs, or roster differs')
    scores = {}
    for case in manifest['cases']:
        record = value['cases'][case['id']]
        if record['status'] != 'PASS' or record['physical_fits'] != 1:
            raise ValueError('Reference case did not complete exactly one fit')
        directory = root / 'cases' / case['id']
        actual = inventory(directory)
        if actual != record['files']:
            raise ValueError('Reference evidence changed')
        if not any(k.startswith('model/')
                   for k in actual) or 'predictions.json' not in actual:
            raise ValueError(
                'Reference model or stored predictions are missing')
        scores[case['id']] = record['score']
    if check_learning(scores, scores)['status'] != 'PASS':
        raise ValueError(
            'Reference does not establish the required learning separation')
    return value


def execute_cases(cases, output, execute):
    rows = []
    for case in cases:
        directory = output / case['id']
        directory.mkdir(parents=True, exist_ok=False)
        try:
            row = execute(case, directory)
        except Exception as error:
            retained = directory / 'worker-result.json'
            row = read(retained) if retained.is_file() else {}
            row.update(status='FAIL', case_id=case['id'],
                       error=f'{type(error).__name__}: {error}',
                       traceback=traceback.format_exc())
        save(directory / 'result.json', row)
        rows.append(row)
    return dict(
        status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
        cases=rows)


def run(args):
    manifest = load_manifest(args.input_root, args.manifest_sha256)
    reference = None
    if args.mode == 'candidate':
        if not args.build_receipt:
            raise ValueError('Candidate work needs its source build receipt')
        check_build(read(args.build_receipt), args.version, args.wheel_sha256)
        reference = load_reference(args.reference_root, args.reference_sha256,
                                   manifest, args.manifest_sha256)
    if digest(args.wheel) != args.wheel_sha256:
        raise ValueError('Selected wheel checksum differs')
    if args.mode == 'reference' and dict(
            version=args.version,
            wheel_sha256=args.wheel_sha256) != manifest['reference']:
        raise ValueError(
            'Reference creation requires the approved manifest wheel')
    if args.output.exists() or args.scratch.exists():
        raise ValueError('Use fresh evidence and scratch directories')
    if (args.scratch.resolve().is_relative_to(args.output.resolve())
            or args.output.resolve().is_relative_to(args.scratch.resolve())):
        raise ValueError('Keep engine scratch outside the evidence tree')
    args.output.mkdir(parents=True)
    if args.build_receipt:
        shutil.copyfile(args.build_receipt, args.output / 'build.json')
    report = dict(
        format='kumo-engine-weekly-run-v2', status='RUNNING', mode=args.mode,
        wheel=dict(version=args.version, wheel_sha256=args.wheel_sha256),
        manifest_sha256=args.manifest_sha256, cases=[])
    started = time.monotonic()
    try:
        with args.host_lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = Path(__file__).with_name('weekly_worker.py')
            report['capacity'] = capacity(args.scratch.parent)
            check_capacity(manifest['cases'], report['capacity'])
            import torch
            properties = torch.cuda.get_device_properties(0)
            if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                    or
                    any(c['gpu_class']['compute_capability'] !=
                        [properties.major, properties.minor] or c['gpu_class']
                        ['minimum_memory_bytes'] > properties.total_memory
                        for c in manifest['cases'])):
                raise ValueError(
                    'The current GPU class does not admit every case')

            def execute(case, directory):
                command = [
                    sys.executable, '-I',
                    str(worker), '--input-root',
                    str(args.input_root), '--manifest-sha256',
                    args.manifest_sha256, '--case', case['id'], '--output',
                    str(directory), '--scratch',
                    str(args.scratch / case['id']), '--wheel',
                    str(args.wheel), '--version', args.version,
                    '--wheel-sha256', args.wheel_sha256
                ]
                if reference is not None:
                    command += ['--reference-root', str(args.reference_root)]
                with (directory / 'worker.log').open('w') as log:
                    result = subprocess.run(command, stdout=log,
                                            stderr=subprocess.STDOUT)
                row = read(directory / 'worker-result.json')
                if result.returncode != 0 or row['status'] != 'PASS':
                    raise ValueError(
                        f'{case["id"]} failed: {row.get("error", "worker exit " + str(result.returncode))}'
                    )
                check_case_result(row, case['id'], report['wheel'],
                                  candidate=reference is not None)
                if reference and row['fit_id'] in {
                        v['fit_id']
                        for v in reference['cases'].values()
                }:
                    raise ValueError(
                        'Candidate reused an approved reference fit')
                return row

            report.update(
                execute_cases(manifest['cases'], args.output / 'cases',
                              execute))
            fits = [r['fit_id'] for r in report['cases'] if r.get('fit_id')]
            if len(set(fits)) != len(fits):
                report['status'] = 'FAIL'
            scores = {r['case_id']: r.get('score') for r in report['cases']}
            baseline = {
                k: v['score']
                for k, v in reference['cases'].items()
            } if reference else scores
            report['learning'] = check_learning(scores, baseline)
            if report['learning']['status'] != 'PASS':
                report['status'] = 'FAIL'
            load_manifest(args.input_root, args.manifest_sha256)
            if reference:
                load_reference(args.reference_root, args.reference_sha256,
                               manifest, args.manifest_sha256)
            if args.mode == 'reference' and report['status'] == 'PASS':
                records = {}
                for row in report['cases']:
                    records[row['case_id']] = dict(
                        row,
                        files=inventory(args.output / 'cases' / row['case_id']))
                save(
                    args.output / 'reference.json',
                    dict(format='kumo-engine-weekly-reference-v2',
                         status='PASS', wheel=manifest['reference'],
                         manifest_sha256=args.manifest_sha256, cases=records))
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.monotonic() - started
        save(args.output / 'weekly-result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'output', 'scratch', 'host-lock', 'wheel'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'wheel-sha256', 'version'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mode', choices=('reference', 'candidate'),
                        default='candidate')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--build-receipt', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if args.mode == 'candidate' and (not args.reference_root
                                     or not args.reference_sha256):
        parser.error(
            'Candidate work needs an approved reference bundle and checksum')
    return 0 if run(args)['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
