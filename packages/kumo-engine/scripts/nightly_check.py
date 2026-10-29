#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the fixed nightly task cases against one installed wheel."""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import platform
import re
import runpy
import sys
import time
import traceback
from contextlib import chdir
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PR = runpy.run_path(str(Path(__file__).with_name('pr_check.py')))
TASKS = tuple(sorted(PR['TASKS']))
CONDITIONS = {'normal': 'learnable', 'scrambled': 'independent'}
digest, save = PR['digest'], PR['save']


def case_roster(tasks, conditions):
    if (not tasks or not conditions or len(set(tasks)) != len(tasks)
            or len(set(conditions)) != len(conditions) or any(task not in TASKS
                                                              for task in tasks)
            or any(condition not in CONDITIONS for condition in conditions)):
        raise ValueError('Select unique known tasks and normal/scrambled cases')
    return [(task, 0, CONDITIONS[condition]) for task in tasks
            for condition in conditions]


def verify_archive(wheel, expected_sha256):
    if (not re.fullmatch('[0-9a-f]{64}', expected_sha256)
            or digest(wheel) != expected_sha256):
        raise ValueError('The selected wheel fingerprint differs')


def gpu_properties(properties):
    if (properties.name != 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
            or type(properties.uuid) is not str or not properties.uuid
            or type(properties.total_memory) is not int
            or properties.total_memory <= 0):
        raise ValueError('Use the fixed RTX PRO 6000 Blackwell GPU profile')
    return dict(gpu=properties.name, gpu_uuid=properties.uuid,
                gpu_memory_bytes=properties.total_memory)


def fixture_sources(fixtures, tasks, conditions):
    sources = {}
    link_modes = set()
    for task in tasks:
        sources[task] = {}
        for condition in dict.fromkeys(['normal', *conditions]):
            directory = fixtures / task / ('normal' if condition == 'normal'
                                           else 'seed-0')
            path = directory / 'descriptor.json'
            descriptor = json.loads(path.read_text())
            seed = None if condition == 'normal' else 0
            identity = descriptor.get('identity', {})
            arm = 'normal' if seed is None else 'control'

            def matching_seed(value):
                return value is None if seed is None else type(
                    value) is int and value == 0

            if (descriptor.get('case') != task
                    or not matching_seed(descriptor.get('seed'))
                    or identity.get('case') != task
                    or not matching_seed(identity.get('control_seed'))
                    or descriptor.get('arm') != arm):
                raise ValueError(
                    'The frozen task or scrambled-answer arm differs')
            mode = identity.get('ranked_links', False)
            if type(mode) is not bool or mode and not task.endswith('link'):
                raise ValueError('The frozen prediction representation differs')
            if task.endswith('link'):
                link_modes.add(mode)
            sources[task][condition] = dict(directory=str(directory),
                                            descriptor_sha256=digest(path))
    if len(link_modes) > 1:
        raise ValueError(
            'Selected link cases need the same output representation')
    return sources, link_modes == {True}


def run(args):
    expected = case_roster(args.tasks, args.conditions)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'work'
    work.mkdir()
    report = dict(
        status='RUNNING', group='all', seeds=[0], runs=[],
        expected_runs=expected, completed_runs=[],
        learning_quality_policy='report_only',
        control_quality_policy='diagnostic', release_approved=False,
        physical_fits=0, source_sha256=digest(Path(__file__)),
        version=args.version, wheel_sha256=args.wheel_sha256,
        expected_prefix=str(args.expected_prefix), host_lock=str(
            args.host_lock), host_lock_acquired=False, host_lock_released=False,
        configuration=dict(device='cuda', seed_profile='nightly',
                           tasks=args.tasks, conditions=args.conditions,
                           scale_bench_rung='100M'))
    result = output / 'nightly-result.json'
    started = time.monotonic()
    fits = output / 'fits.json'
    save(fits, [])
    lock = None
    try:
        lock = args.host_lock.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'GPU host is busy: {args.host_lock}') from error
        report['host_lock_acquired'] = True
        verify_archive(args.wheel, args.wheel_sha256)
        if (not sys.flags.isolated or sys.prefix == sys.base_prefix
                or Path(sys.prefix).resolve() != args.expected_prefix.resolve()
                or platform.machine() != 'x86_64'):
            raise ValueError(
                'Use the declared isolated Linux x64 wheel environment')
        report['installation'] = PR['verify_install'](dict(
            version=args.version, wheel_sha256=args.wheel_sha256), args.wheel)
        PR['run_logged'](
            [sys.executable, '-I', '-m', 'pip', '--isolated', 'check'],
            output / 'pip-check.log', cwd=work)
        sys.path.insert(0, str(ROOT / 'packages/kumo-engine/test'))
        import torch
        import wheel_gpu_learning as gpu
        import wheel_scale_data as data
        import wheel_scale_learning as learning

        packages = {
            re.sub('[-_.]+', '-', item.metadata['Name']).lower(): item.version
            for item in importlib.metadata.distributions() if re.sub(
                '[-_.]+', '-', item.metadata['Name']).lower() != 'kumo-engine'
        }
        gpu.verify_dependencies(packages, args.dependency_profile)
        report.update(dependencies=packages,
                      dependency_profile_sha256=digest(args.dependency_profile),
                      gpu_source_sha256=gpu.source_identity(prepared=True),
                      version=args.version, wheel_sha256=args.wheel_sha256)
        sources, ranked = fixture_sources(args.fixtures, args.tasks,
                                          args.conditions)
        report.update(fixture_sources=sources, profiles={})
        report['configuration'].update(
            ranked_links=ranked,
            **gpu_properties(torch.cuda.get_device_properties(0)),
            cuda=torch.version.cuda, torch=torch.__version__, threads=4)
        torch.set_num_threads(4)
        save(result, report)
        with PR['count_fits'](fits) as physical, chdir(work):
            for task in args.tasks:
                options = {
                    'ranked_links': ranked
                } if task.endswith('link') else {}
                full = data.prepare_profile(
                    args.data / data.CASES[task]['graph'] / '100M',
                    args.scale_source, task,
                    output / 'prepared' / task / 'source', **options)
                normal = data.load_control(
                    full, None, args.fixtures / task / 'normal',
                    output / 'prepared' / task / 'normal')
                report['profiles'][task] = normal['identity']
                del full
                data._PREPARED.clear()
                gc.collect()
                for condition in args.conditions:
                    independent = condition == 'scrambled'
                    name = f'{task}-0-{CONDITIONS[condition]}'
                    report['stage'] = name
                    save(result, report)
                    prepared = (data.load_control(
                        normal, 0, args.fixtures / task / 'seed-0', output /
                        'prepared' / task /
                        'scrambled') if independent else normal)
                    if (prepared['frozen_control']['descriptor_sha256'] !=
                            sources[task][condition]['descriptor_sha256']):
                        raise ValueError('The loaded frozen descriptor changed')
                    data.verify_final_isolation(prepared)
                    row = learning.run_case(prepared, task, 0, independent,
                                            output / name)
                    data.verify_final_isolation(prepared)
                    report['runs'].append(row)
                    report['completed_runs'].append((task, 0, row['case']))
                    report['physical_fits'] = len(physical)
                    save(result, report)
                    del prepared
                    gc.collect()
                    torch.cuda.empty_cache()
                del normal
        report['status'] = 'MEASURED'
        gpu.verify_prepared_report(report, expected_runs=expected)
        if len(physical) != len(expected):
            raise ValueError(
                'Actual model fits differ from the selected nightly cases')
        PR['verify_install'](dict(version=args.version,
                                  wheel_sha256=args.wheel_sha256), args.wheel)
        report['stage'] = 'complete'
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        if lock is not None:
            lock.close()
            report['host_lock_released'] = report['host_lock_acquired']
        report['seconds'] = time.monotonic() - started
        if fits.exists():
            report['physical_fits'] = len(json.loads(fits.read_text()))
        save(result, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wheel', 'data', 'scale-source', 'fixtures',
                 'dependency-profile', 'expected-prefix', 'host-lock',
                 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS,
                        default=list(TASKS))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS,
                        default=list(CONDITIONS))
    args = parser.parse_args()
    for name in ('wheel', 'data', 'scale_source', 'fixtures',
                 'dependency_profile', 'expected_prefix'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.host_lock = args.host_lock.resolve()
    run(args)


if __name__ == '__main__':
    main()
