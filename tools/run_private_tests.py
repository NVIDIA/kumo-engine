# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover private cases and run their declared profiles in fresh processes."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from test.ci_resources import GROUPS, PROFILE_ENV


def redact_download_urls(text):
    for name, value in os.environ.items():
        if value and (name.endswith(
            ('_TOKEN', '_SECRET', '_SECRET_ACCESS_KEY', '_PASSWORD',
             '_PRIVATE_KEY')) or name == 'AWS_ACCESS_KEY_ID'):
            text = text.replace(value, 'REDACTED')
    return re.sub(
        r"https?://[^\s'\"<>]+",
        lambda match: match[0].split('?', 1)[0] + '?REDACTED'
        if '?' in match[0] else match[0], text)


def run_pytest(args, env, output, name, *, coverage=False):
    command = [sys.executable, '-X', 'faulthandler', '-m']
    if coverage:
        command += [
            'coverage', 'run', '--data-file',
            str(output / f'{name}.coverage'), '-m'
        ]
    command += [
        'pytest', '--rootdir=.', '--capture=fd', '--verbosity=0',
        '--disable-warnings', *args
    ]
    with (output / f'{name}.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            line = redact_download_urls(line)
            print(line, end='', flush=True)
            log.write(line)
        result = process.wait()
    junit = output / f'{name}.xml'
    if junit.exists():
        junit.write_text(redact_download_urls(junit.read_text()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=GROUPS[1:])
    parser.add_argument('paths', nargs='*',
                        help='Override component collection roots')
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    output = Path(os.environ.get('CI_TEST_RESULTS', 'ci-test-results'))
    output.mkdir(parents=True, exist_ok=True)
    # Hosted component suites are subdirectories of test/. Root-level script
    # tests and release tooling have separate entry points and prerequisites.
    roots = args.paths or [
        str(path) for path in sorted(Path('test').iterdir()) if path.is_dir()
        and not path.name.startswith('_') and path.name != 'release_tests'
    ]
    discovery = output / 'discovery.json'
    result = run_pytest([
        *roots, '--collect-only', '--resource-group', args.group,
        '--private-discovery', '--resource-manifest',
        str(discovery)
    ], os.environ.copy(), output, 'discovery')
    if result:
        return result
    sessions = {}
    for nodeid, metadata in json.loads(discovery.read_text()).items():
        key = (tuple(metadata['profiles']), metadata['isolated_module'])
        sessions.setdefault(key, {})[nodeid] = metadata
    if not sessions:
        raise RuntimeError(
            f'No tests discovered for private group {args.group}')
    failed = False
    for index, ((profiles, module),
                expected) in enumerate(sorted(sessions.items())):
        name = f'{args.group}-{index}'
        selection = output / f'{name}-selection.json'
        selection.write_text(json.dumps(expected, indent=2) + '\n')
        print(f'CI_SELECTION {name} profiles={profiles} module={module}',
              flush=True)
        if args.collect_only:
            continue
        env = os.environ.copy()
        for profile in profiles:
            env.update(PROFILE_ENV[profile])
        failed |= run_pytest([
            *expected,
            '--resource-group',
            args.group,
            '--require-resource-credentials',
            '--resource-manifest',
            str(selection),
            '--junitxml',
            str(output / f'{name}.xml'),
            '--durations=0',
            '--timeout=1200',
            '--timeout-method=signal',
        ], env, output, name, coverage=True) != 0
        if json.loads(selection.read_text()) != expected:
            print(
                f'CI selection changed between discovery and execution: {name}',
                flush=True)
            failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
