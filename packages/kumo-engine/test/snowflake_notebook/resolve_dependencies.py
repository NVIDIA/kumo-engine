# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a candidate's Snowflake extra and create a dependency-only archive."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from dependency_inventory import inventory


def runtime_target():
    return dict(python='.'.join(map(str, sys.version_info[:2])),
                system=platform.system(), architecture=platform.machine())


def resolve(wheel, destination, plan, workload_wheels=None):
    target = plan['target']
    if platform.python_implementation() != 'CPython':
        raise ValueError(
            'Resolve with CPython for the configured notebook target')
    if (target['system'], target['architecture']) != ('Linux', 'x86_64'):
        raise ValueError('Only the Linux x86_64 notebook target is supported')
    current = runtime_target()
    if current != {key: target[key] for key in current}:
        raise ValueError(
            'Resolve on the configured Linux x86_64 Python minor version')
    if not wheel.is_file() or wheel.is_symlink():
        raise ValueError('Supply a regular candidate wheel')
    extra_wheels = list(workload_wheels or [])
    if (len(
        {path.name
         for path in extra_wheels}) != len(extra_wheels) or any(
             path.is_symlink() or not path.is_file() or path.name == wheel.name
             for path in extra_wheels)):
        raise ValueError(
            'Workload dependency wheels must be unique regular files')
    if extra_wheels:
        import launch
        launch.workload_tools().check_workload_wheels(wheel, extra_wheels)
    destination.mkdir(parents=True, exist_ok=False)
    wheels = destination / 'wheelhouse'
    wheels.mkdir()
    command = [
        sys.executable, '-I', '-m', 'pip', '--isolated', '--no-cache-dir'
    ]
    for filename in plan['local_wheels']:
        if not filename.endswith('-py3-none-any.whl'):
            raise ValueError(
                'Authored source builds must be pure Python wheels')
        name, version, *_ = filename.split('-')
        requirement = name + '==' + version
        subprocess.run(
            command + [
                'wheel', '--no-deps', '--index-url', plan['source_index'],
                '--wheel-dir',
                str(wheels), requirement
            ], check=True, env=dict(os.environ, SOURCE_DATE_EPOCH='0'))
    platforms = []
    minor = int(target['maximum_manylinux'].removeprefix('2.'))
    for version in range(minor, 4, -1):
        platforms += ['--platform', f'manylinux_2_{version}_x86_64']
    for legacy in ('manylinux2014_x86_64', 'manylinux2010_x86_64',
                   'manylinux1_x86_64'):
        platforms += ['--platform', legacy]
    if any(path.name.endswith('-linux_x86_64.whl') for path in extra_wheels):
        platforms += ['--platform', 'linux_x86_64']
    subprocess.run(
        command + [
            'download', '--only-binary=:all:', '--index-url',
            plan['source_index'], '--python-version', target['python'],
            '--implementation', 'cp', '--abi',
            'cp' + target['python'].replace('.', ''), '--abi', 'abi3', '--abi',
            'none', *platforms, '--find-links',
            str(wheels), '--dest',
            str(wheels),
            str(wheel.resolve()) + '[snowflake]',
            *[str(path.resolve()) for path in extra_wheels]
        ], check=True)
    archive = destination / 'wheelhouse.tar'
    with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT) as bundle:
        for path in sorted(wheels.iterdir()):
            if path.name == wheel.name:
                path.unlink()
                continue
            if not path.is_file() or path.suffix != '.whl':
                raise ValueError('Resolver produced a non-wheel dependency')
            member = tarfile.TarInfo(path.name)
            member.size = path.stat().st_size
            member.mode = 0o644
            with path.open('rb') as source:
                bundle.addfile(member, source)
            path.unlink()
    resolved_plan = dict(plan)
    if extra_wheels:
        resolved_plan['workload_wheels'] = [path.name for path in extra_wheels]
    result = inventory(archive, resolved_plan)
    if not {path.name for path in extra_wheels} <= set(result['wheels']):
        raise ValueError('Dependency archive missed a customer workload wheel')
    (destination /
     'dependencies.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.rmtree(wheels)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--workload-wheel', type=Path, action='append',
        help='Additional workload dependency wheel; repeat as needed')
    parser.add_argument(
        '--plan', type=Path,
        default=Path(__file__).with_name('dependency-plan.json'))
    args = parser.parse_args()
    print(
        resolve(args.wheel, args.output_dir, json.loads(args.plan.read_text()),
                args.workload_wheel))


if __name__ == '__main__':
    main()
