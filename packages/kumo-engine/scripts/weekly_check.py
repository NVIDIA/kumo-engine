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


def select_cases(manifest, profile=None):
    if profile not in (None, 'rtx', 'scale'):
        raise ValueError('Unknown weekly execution profile')
    return [
        case for case in manifest['cases']
        if profile is None or (case['id'] == 'P2-fixed') == (profile == 'scale')
    ]


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


def compare_predictions(before, after, *, atol, rtol, rank_atol=0.):
    """Keep IDs and labels exact. Bound any explicitly approved rank swaps."""
    if not all(finite(v) and v >= 0 for v in (atol, rtol, rank_atol)):
        raise ValueError('Prediction tolerances must be finite and nonnegative')
    ids = before.get('entity_ids', [])
    ranks, observed_ranks = before.get('ranked_ids'), after.get('ranked_ids')
    if (not ids or len(set(ids)) != len(ids) or ids != after.get('entity_ids')
            or (ranks is None) != (observed_ranks is None)
            or rank_atol == 0 and ranks != observed_ranks
            or before.get('decisions') != after.get('decisions')):
        raise ValueError('Prediction population, decisions, or ranks changed')
    left, right = before.get('values', []), after.get('values', [])
    if len(left) != len(ids) or len(right) != len(ids):
        raise ValueError('Prediction population is incomplete')
    if ranks is not None and (len(ranks) != len(ids)
                              or len(observed_ranks) != len(ids)):
        raise ValueError('Prediction population is incomplete')
    maximum, changed, inversions, maximum_gap = 0., 0, 0, 0.
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if not isinstance(a, list) or not a or not isinstance(
                b, list) or len(a) != len(b):
            raise ValueError('Prediction coordinates changed')
        if not all(finite(v) for v in (*a, *b)):
            raise ValueError('Prediction values are nonfinite')
        pairs = zip(a, b, strict=True)
        if ranks is not None:
            names, observed_names = ranks[index], observed_ranks[index]
            if (len(names) != len(a) or len(observed_names) != len(b)
                    or len(set(names)) != len(names)
                    or len(set(observed_names)) != len(observed_names)
                    or set(names) != set(observed_names)):
                raise ValueError('Prediction ranking membership changed')
            if any(x < y for row in (a, b) for x, y in zip(row, row[1:])):
                raise ValueError('Prediction scores do not follow their ranks')
            observed = dict(zip(observed_names, b, strict=True))
            if names != observed_names:
                changed += 1
                positions = {name: i for i, name in enumerate(observed_names)}
                for i, first in enumerate(names):
                    for j in range(i + 1, len(names)):
                        second = names[j]
                        if positions[first] > positions[second]:
                            gap = max(a[i] - a[j],
                                      observed[second] - observed[first])
                            if gap > rank_atol:
                                raise ValueError(
                                    'Rank swap exceeds the approved score-gap budget'
                                )
                            maximum_gap = max(maximum_gap, gap)
                            inversions += 1
            pairs = ((value, observed[name])
                     for name, value in zip(names, a, strict=True))
        for x, y in pairs:
            if abs(x - y) > atol + rtol * abs(x):
                raise ValueError(
                    'Prediction values exceed the approved tolerance')
            maximum = max(maximum, abs(x - y))
    return dict(status='PASS', entities=len(ids), max_abs_difference=maximum,
                atol=atol, rtol=rtol, rank_atol=rank_atol,
                rank_changed_entities=changed, rank_inversions=inversions,
                max_reordered_score_gap=maximum_gap)


def replay_reference(model_path, stored, load, predict, *, atol, rtol,
                     rank_atol=0.):
    model = load(model_path)
    observed = predict(model)
    return compare_predictions(stored, observed, atol=atol, rtol=rtol,
                               rank_atol=rank_atol)


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
            or row.get('reload', {}).get('status') != 'PASS'
            or row.get('native_contract', {}).get('status') != 'PASS'
            or candidate
            and row.get('reference_replay', {}).get('status') != 'PASS' or any(
                work.get('prediction_batches', {}).get(s, 0) <= 0
                for s in stages) or any(
                    work.get('restored_tensors', {}).get(s, 0) <= 0
                    for s in stages if s != 'candidate')):
        raise ValueError(
            'Case lacks one CUDA fit, complete predictions, or reference replay'
        )


def load_manifest(root, expected, *, verify_files=True, profile=None,
                  retained_p2=False):
    path = root / 'manifest.json'
    if digest(path) != expected:
        raise ValueError('Input manifest checksum differs')
    value = read(path)
    cases = value.get('cases', [])
    roster = {'P2-fixed'} if retained_p2 else ROSTER
    artifact_format = ('kumo-engine-weekly-p2-input-v1'
                       if retained_p2 else 'kumo-engine-weekly-input-v2')
    if (value.get('format') != artifact_format or len(cases) != len(roster)
            or {c['id']
                for c in cases} != roster
            or not value.get('reference', {}).get('version')
            or not re.fullmatch('[0-9a-f]{64}', value['reference'].get(
                'wheel_sha256', ''))):
        raise ValueError(
            'Use the complete manifest for this check and an approved wheel')
    selected = {c['id'] for c in select_cases(value, profile)}
    for case in cases:
        task = case['id'].split('-')[0]
        if ((case['dataset'], case['metric']) != PROFILES[task]
                or type(case['seed']) is not int or case['seed'] != 0
                or case['arm'] != case['id'].split('-')[1]):
            raise ValueError('Frozen workload, metric, arm, or seed differs')
        required = ('snapshot', 'training_table', 'model_plan',
                    'prediction_table', 'population', 'dataset_manifest')
        if (any(not isinstance(case.get(k), str) or not case[k]
                for k in required)
                or case['id'] != 'P2-fixed' and not case.get('answers')):
            raise ValueError(
                'Freeze every required native input and reserved answer file')
        if case['metric'].startswith('recall@') and (
                not case.get('catalog')
                or type(case.get('raw_limit')) is not int or not int(
                    case['metric'].split('@')[1]) <= case['raw_limit'] <= 500):
            raise ValueError(
                'Freeze the ranking catalog and prediction limit before fitting'
            )
        tolerance = case.get('tolerance')
        if (not isinstance(tolerance, dict)
                or not {'atol', 'rtol'} <= tolerance.keys()
                or not tolerance.keys() <= {'atol', 'rtol', 'rank_atol'}
                or not all(finite(v) and v >= 0 for v in tolerance.values())):
            raise ValueError(
                'Approve finite prediction tolerances before execution')
        for key in ('snapshot', 'training_table', 'model_plan',
                    'prediction_table', 'population', 'answers', 'catalog',
                    'dataset_manifest'):
            if verify_files and case.get(
                    key) is not None and case['id'] in selected:
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
        for name in {
                name
                for case in cases if case['id'] in selected
                for name in case['files']
        }:
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


def reference_header(path, expected, manifest, manifest_sha256, *,
                     retained_p2=False):
    value = read(data_file(path.parent, path.name, {path.name: expected}))
    roster = {'P2-fixed'} if retained_p2 else ROSTER
    artifact_format = ('kumo-engine-weekly-p2-reference-v1'
                       if retained_p2 else 'kumo-engine-weekly-reference-v2')
    if (value.get('format') != artifact_format or value.get('status') != 'PASS'
            or value.get('manifest_sha256') != manifest_sha256
            or value.get('wheel') != manifest['reference']
            or set(value.get('cases', {})) != roster
            or len({v.get('fit_id')
                    for v in value.get('cases', {}).values()}) != len(roster)):
        raise ValueError(
            'Reference approval, wheel, inputs, roster, or fit identities differ'
        )
    scores = {}
    for name, record in value['cases'].items():
        if record['status'] != 'PASS' or record[
                'physical_fits'] != 1 or not record.get('fit_id'):
            raise ValueError('Reference case did not complete exactly one fit')
        scores[name] = record['score']
    if not retained_p2 and check_learning(scores, scores)['status'] != 'PASS':
        raise ValueError(
            'Reference does not establish the required learning separation')
    return value


def load_reference(root, expected, manifest, manifest_sha256, *, profile=None,
                   retained_p2=False):
    value = reference_header(root / 'reference.json', expected, manifest,
                             manifest_sha256, retained_p2=retained_p2)
    for case in select_cases(manifest, profile):
        record = value['cases'][case['id']]
        directory = root / 'cases' / case['id']
        actual = inventory(directory)
        if actual != record['files']:
            raise ValueError('Reference evidence changed')
        if not any(k.startswith('model/')
                   for k in actual) or 'predictions.json' not in actual:
            raise ValueError(
                'Reference model or stored predictions are missing')
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


def collect_shards(paths, manifest, manifest_sha256, output, *,
                   reference_sha256=None):
    if len(paths) != 2 or output.exists():
        raise ValueError('Collect two complete profiles into a fresh directory')
    reports = [read(path / 'weekly-result.json') for path in paths]
    if {r.get('profile') for r in reports} != {'rtx', 'scale'}:
        raise ValueError('Both distinct weekly profiles are required')
    modes = {r['mode'] for r in reports}
    wheels = {(r['wheel']['version'], r['wheel']['wheel_sha256'])
              for r in reports}
    if len(modes) != 1 or len(wheels) != 1:
        raise ValueError('Weekly profiles used different modes or wheels')
    mode = reports[0]['mode']
    wheel = reports[0]['wheel']
    if mode not in ('candidate', 'reference'
                    ) or mode == 'reference' and wheel != manifest['reference']:
        raise ValueError('Reference profile used an unapproved wheel')
    reference = None
    if mode == 'candidate':
        if not reference_sha256:
            raise ValueError(
                'Candidate collection needs the approved reference checksum')
        reference = reference_header(paths[0] / 'reference.json',
                                     reference_sha256, manifest,
                                     manifest_sha256)
        check_build(read(paths[0] / 'build.json'), wheel['version'],
                    wheel['wheel_sha256'])
    rows = []
    for path, report in zip(paths, reports, strict=True):
        expected = {c['id'] for c in select_cases(manifest, report['profile'])}
        if (report.get('format') != 'kumo-engine-weekly-run-v2'
                or report.get('status') != 'PASS'
                or report.get('manifest_sha256') != manifest_sha256
                or digest(path / 'manifest.json') != manifest_sha256
                or len(report['cases']) != len(expected)
                or set(report['case_ids']) != expected
                or {v['case_id']
                    for v in report['cases']} != expected):
            raise ValueError(
                'Weekly profile is incomplete or uses different inputs')
        if reference:
            if report.get('reference_sha256') != reference_sha256:
                raise ValueError(
                    'Weekly profile used a different approved reference')
            reference_header(path / 'reference.json', reference_sha256,
                             manifest, manifest_sha256)
            if read(path / 'build.json') != read(paths[0] / 'build.json'):
                raise ValueError(
                    'Weekly profiles used different source build receipts')
        for row in report['cases']:
            check_case_result(row, row['case_id'], wheel,
                              candidate=mode == 'candidate')
            directory = path / 'cases' / row['case_id']
            if read(directory / 'result.json') != row or read(
                    directory / 'worker-result.json') != row:
                raise ValueError(
                    'Collected row differs from its worker evidence')
            rows.append(row)
    fits = {r['fit_id'] for r in rows}
    if len(fits) != 11 or reference and fits & {
            v['fit_id']
            for v in reference['cases'].values()
    }:
        raise ValueError('Weekly collection needs eleven distinct fresh fits')
    scores = {r['case_id']: r['score'] for r in rows}
    baseline = {
        k: v['score']
        for k, v in reference['cases'].items()
    } if reference else scores
    learning = check_learning(scores, baseline)
    result = dict(format='kumo-engine-weekly-run-v2', mode=mode,
                  status=learning['status'], wheel=wheel,
                  manifest_sha256=manifest_sha256,
                  reference_sha256=reference_sha256, learning=learning,
                  cases=rows, physical_fits=sum(r['physical_fits']
                                                for r in rows),
                  profiles=reports)
    output.mkdir(parents=True)
    # Model bytes stay in the per-profile artifacts. Collection writes metadata only.
    if mode == 'reference' and result['status'] == 'PASS':
        records = {}
        for path, report in zip(paths, reports, strict=True):
            for row in report['cases']:
                source = path / 'cases' / row['case_id']
                records[row['case_id']] = dict(row, files=inventory(source))
        save(
            output / 'reference.json',
            dict(format='kumo-engine-weekly-reference-v2', status='PASS',
                 wheel=wheel, manifest_sha256=manifest_sha256, cases=records))
    save(output / 'weekly-result.json', result)
    return result


def run(args):
    with args.host_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args)


def run_locked(args):
    manifest = load_manifest(args.input_root, args.manifest_sha256,
                             profile=args.profile)
    selected = select_cases(manifest, args.profile)
    reference = None
    if args.mode == 'candidate':
        if not args.build_receipt:
            raise ValueError('Candidate work needs its source build receipt')
        check_build(read(args.build_receipt), args.version, args.wheel_sha256)
        reference = load_reference(args.reference_root, args.reference_sha256,
                                   manifest, args.manifest_sha256,
                                   profile=args.profile)
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
    immutable = [
        args.input_root, *([args.reference_root] if args.reference_root else [])
    ]
    if any(path.resolve().is_relative_to(root.resolve()) for root in immutable
           for path in (args.output, args.scratch)):
        raise ValueError(
            'Keep all new work outside immutable input and reference directories'
        )
    args.output.mkdir(parents=True)
    if args.build_receipt:
        shutil.copyfile(args.build_receipt, args.output / 'build.json')
    if reference:
        shutil.copyfile(args.reference_root / 'reference.json',
                        args.output / 'reference.json')
    shutil.copyfile(args.input_root / 'manifest.json',
                    args.output / 'manifest.json')
    report = dict(
        format='kumo-engine-weekly-run-v2', status='RUNNING', mode=args.mode,
        wheel=dict(version=args.version, wheel_sha256=args.wheel_sha256),
        manifest_sha256=args.manifest_sha256, profile=args.profile,
        reference_sha256=args.reference_sha256,
        case_ids=[c['id'] for c in selected], cases=[])
    started = time.monotonic()
    try:
        worker = Path(__file__).with_name('weekly_worker.py')
        report['capacity'] = capacity(args.scratch.parent)
        check_capacity(selected, report['capacity'])
        import torch
        properties = torch.cuda.get_device_properties(0)
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or any(c['gpu_class']['compute_capability'] !=
                       [properties.major, properties.minor] or c['gpu_class']
                       ['minimum_memory_bytes'] > properties.total_memory
                       for c in selected)):
            raise ValueError('The current GPU class does not admit every case')

        def execute(case, directory):
            command = [
                sys.executable, '-I',
                str(worker), '--input-root',
                str(args.input_root), '--manifest-sha256', args.manifest_sha256,
                '--case', case['id'], '--output',
                str(directory), '--scratch',
                str(args.scratch / case['id']), '--wheel',
                str(args.wheel), '--version', args.version, '--wheel-sha256',
                args.wheel_sha256
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
                raise ValueError('Candidate reused an approved reference fit')
            return row

        report.update(execute_cases(selected, args.output / 'cases', execute))
        fits = [r['fit_id'] for r in report['cases'] if r.get('fit_id')]
        if len(set(fits)) != len(fits):
            report['status'] = 'FAIL'
        scores = {r['case_id']: r.get('score') for r in report['cases']}
        baseline = {
            k: v['score']
            for k, v in reference['cases'].items()
        } if reference else scores
        report['learning'] = check_learning(
            scores, baseline) if args.profile is None else dict(
                status='DEFERRED_TO_COLLECTION')
        if report['learning']['status'] == 'FAIL':
            report['status'] = 'FAIL'
        load_manifest(args.input_root, args.manifest_sha256,
                      profile=args.profile)
        if reference:
            load_reference(args.reference_root, args.reference_sha256, manifest,
                           args.manifest_sha256, profile=args.profile)
        if args.mode == 'reference' and report['status'] == 'PASS':
            records = {}
            for row in report['cases']:
                records[row['case_id']] = dict(
                    row,
                    files=inventory(args.output / 'cases' / row['case_id']))
            save(
                args.output /
                ('reference-shard.json' if args.profile else 'reference.json'),
                dict(format='kumo-engine-weekly-reference-v2', status='PASS',
                     wheel=manifest['reference'],
                     manifest_sha256=args.manifest_sha256, profile=args.profile,
                     cases=records))
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.monotonic() - started
        save(args.output / 'weekly-result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run_parser = commands.add_parser('run')
    for name in ('input-root', 'output', 'scratch', 'host-lock', 'wheel'):
        run_parser.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'wheel-sha256', 'version'):
        run_parser.add_argument('--' + name, required=True)
    run_parser.add_argument('--mode', choices=('reference', 'candidate'),
                            default='candidate')
    run_parser.add_argument('--profile', choices=('rtx', 'scale'))
    run_parser.add_argument('--reference-root', type=Path)
    run_parser.add_argument('--build-receipt', type=Path)
    run_parser.add_argument('--reference-sha256')
    collect_parser = commands.add_parser('collect')
    collect_parser.add_argument('--shards', nargs=2, type=Path, required=True)
    collect_parser.add_argument('--input-root', type=Path, required=True)
    collect_parser.add_argument('--manifest-sha256', required=True)
    collect_parser.add_argument('--reference-sha256')
    collect_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        manifest = load_manifest(args.input_root, args.manifest_sha256,
                                 verify_files=False)
        result = collect_shards(args.shards, manifest, args.manifest_sha256,
                                args.output,
                                reference_sha256=args.reference_sha256)
    else:
        if args.mode == 'candidate' and (not args.reference_root
                                         or not args.reference_sha256):
            parser.error(
                'Candidate work needs an approved reference bundle and checksum'
            )
        result = run(args)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
