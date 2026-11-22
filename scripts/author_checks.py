# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover and run existing local checks without automatic host provisioning."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__:
    from .author_tools import (
        git_state,
        handle_termination,
        machine,
        new_output,
        run_process,
        write_json,
    )
else:
    from author_tools import (
        git_state,
        handle_termination,
        machine,
        new_output,
        run_process,
        write_json,
    )

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(__file__).with_name('author-checks.json')


def load_catalog():
    catalog = json.loads(CATALOG.read_text())
    if catalog.get('schema_version') != 1:
        raise ValueError('Unsupported author-check catalog version')
    return catalog['profiles']


def parameters(profile, values):
    result = {}
    for argument in values:
        key, separator, value = argument.partition('=')
        if not separator or not value or key in result:
            raise ValueError('--param requires unique NAME=VALUE entries')
        if key not in profile['parameters']:
            raise ValueError(f'Unknown parameter: {key}')
        if '://' in value:
            raise ValueError('Catalog profiles accept local paths only')
        path = Path(value).expanduser().resolve()
        kind = profile['parameters'][key]
        if kind == 'input-dir' and not path.is_dir():
            raise ValueError(f'{key} must be an existing directory')
        if kind == 'new-dir' and path.exists():
            raise ValueError(f'{key} must be a new directory')
        result[key] = str(path)
    missing = profile['parameters'].keys() - result.keys()
    if missing:
        raise ValueError(f'Missing --param: {", ".join(sorted(missing))}')
    return result


def doctor(profile, python):
    checks = [{
        'name': tool,
        'ok': shutil.which(tool) is not None
    } for tool in profile['tools']]
    code = """import importlib.util,json,sys
modules = {}
for name in json.loads(sys.argv[1]):
    try:
        modules[name] = importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        modules[name] = False
print(json.dumps({'python':sys.version, 'python_supported':sys.version_info >= (3,11), 'modules': modules}))
"""
    try:
        probe = subprocess.run(
            [python, '-c', code,
             json.dumps(profile['modules'])], capture_output=True, text=True,
            timeout=30)
        if probe.returncode:
            raise ValueError(
                'Interpreter/module probe failed; prepare the selected environment'
            )
        result = json.loads(probe.stdout)
        checks += [{
            'name': 'Python >=3.11',
            'ok': result['python_supported'],
            'version': result['python']
        }]
        checks += [{
            'name': name,
            'ok': found
        } for name, found in result['modules'].items()]
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        checks.append({
            'name': 'interpreter probe',
            'ok': False,
            'error': str(error)
        })
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'show', 'doctor', 'run'])
    parser.add_argument('profile', nargs='?')
    parser.add_argument('--python', default=sys.executable,
                        help='prepared interpreter; no packages are installed')
    parser.add_argument('--param', action='append', default=[],
                        metavar='NAME=PATH')
    parser.add_argument('--reports', type=Path,
                        help='new directory outside the checkout')
    parser.add_argument(
        '--disk-budget-gib', type=float,
        help='minimum free GiB for data/scratch preflight; not a quota')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='print resolved argv without running or creating output')
    args = parser.parse_args(argv)
    try:
        if sys.version_info < (3, 11):
            raise ValueError('Use Python 3.11 or newer for author tools')
        profiles = load_catalog()
        if args.action == 'list':
            for name, profile in profiles.items():
                print(f'{name}: {profile["description"]}')
            return 0
        if args.profile not in profiles:
            raise ValueError('Choose a profile from the list command')
        profile = profiles[args.profile]
        if args.action == 'show':
            print(json.dumps(profile, indent=2))
            return 0
        python = shutil.which(
            args.python) if '/' not in args.python else os.path.abspath(
                os.path.expanduser(args.python))
        if python is None:
            raise ValueError('Selected interpreter was not found')
        checks = doctor(profile, python)
        if args.action == 'doctor':
            print(
                json.dumps(
                    {
                        'checks': checks,
                        'limits': profile['limits'],
                        'guide': profile['guide']
                    }, indent=2))
            return 0 if all(row['ok'] for row in checks) else 1
        params = parameters(profile, args.param)
        substitutions = dict(params, python=python)
        command = [
            argument.format_map(substitutions) for argument in profile['argv']
        ]
        print(
            json.dumps(
                {
                    'argv': command,
                    'cwd': str(ROOT / profile['cwd']),
                    'timeout_seconds': profile['timeout_seconds'],
                    'limits': profile['limits']
                }, indent=2), flush=True)
        if args.dry_run:
            return 0
        if not all(row['ok'] for row in checks):
            print(json.dumps({'checks': checks}, indent=2), file=sys.stderr)
            return 1
        if args.reports is None:
            raise ValueError(
                'run requires --reports pointing to a new directory')
        budget = args.disk_budget_gib
        if profile.get('disk_budget_required') and budget is None:
            raise ValueError(
                'This profile requires --disk-budget-gib for generated data/scratch'
            )
        disk = []
        if budget is not None:
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError(
                    '--disk-budget-gib must be finite and positive')
            # Check every output filesystem, including scratch; do not create it.
            paths = [
                Path(v) for k, v in params.items()
                if profile['parameters'][k] == 'new-dir'
            ]
            for path in paths or [args.reports]:
                parent = path.resolve().parent
                while not parent.exists():
                    parent = parent.parent
                free = shutil.disk_usage(parent).free
                disk.append({
                    'path': str(path),
                    'free_bytes': free,
                    'requested_budget_gib': budget
                })
                if free < budget * (1 << 30):
                    raise ValueError(
                        f'Insufficient free space for the declared budget at {path}'
                    )
        state = git_state(ROOT)
        output = new_output(args.reports, (ROOT, ))
        receipt = {
            'schema_version': 1,
            'profile': args.profile,
            'definition': profile,
            'parameters': params,
            'checks': checks,
            'machine': machine(),
            'git': state,
            'disk_preflight': disk,
            'status': 'running'
        }
        write_json(output / 'results.json', receipt)
        try:
            receipt['run'] = run_process(command, ROOT / profile['cwd'],
                                         output / 'run.log',
                                         profile['timeout_seconds'])
            receipt['status'] = receipt['run']['status']
        except KeyboardInterrupt:
            receipt['status'] = 'interrupted'
        finally:
            write_json(output / 'results.json', receipt)
        print(f'Receipt: {output / "results.json"}')
        return 0 if receipt['status'] == 'ok' else 1
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    with handle_termination():
        sys.exit(main())
