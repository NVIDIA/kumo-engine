# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run local connector suites using only public prerequisites.

See docs/development/public-tests.md for installation and test directory rules.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SUITES = {
    'core': ('packages/kumo-core/tests/sources/unit', 'tests/public'),
    'csv': ('packages/kumo-core/tests/sources/csv',),
    'parquet': ('packages/kumo-core/tests/sources/parquet',),
    's3': ('packages/kumo-s3/tests',),
    'gcs': ('packages/kumo-gcs/tests',),
    'adls': ('test/io_utils/fsutil/adls', 'packages/kumo-adls/tests'),
    'databricks': ('packages/kumo-databricks/tests',),
    'snowflake': ('packages/kumo-snowflake/tests',),
    'bigquery': ('packages/kumo-bigquery/tests',),
}


def select_test_file(parser, test_file, directories, excluded):
    if test_file is None:
        return None
    root = Path(__file__).resolve().parents[1]
    path = (root / test_file).resolve()
    if path.is_file() and path.suffix == '.py':
        for directory in directories:
            base = (root / directory).resolve()
            if path.is_relative_to(base) and not any(
                    path.is_relative_to(base / tier) for tier in excluded):
                return path
    parser.error('--test-file must be a Python file in the selected public group')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=(*SUITES, 'all'), default='csv',
                        help='suite to run (default: csv)')
    parser.add_argument('--junitxml', type=Path,
                        help='XML result path for a single suite')
    parser.add_argument('--test-file', type=Path, help='one repository-relative test file')
    parser.add_argument('-k', help='pytest name expression for one suite')
    args = parser.parse_args()
    if (args.test_file or args.k is not None) and args.suite == 'all':
        parser.error('--test-file and -k require a single --suite')
    if args.junitxml and args.suite == 'all':
        parser.error('--junitxml requires a single suite')
    test_file = select_test_file(
        parser, args.test_file, SUITES.get(args.suite, ()), ('live', 'native'))
    root = Path(__file__).resolve().parents[1]
    # Do not inherit provider credentials, PYTHONPATH, PYTEST_ADDOPTS, or
    # PYTEST_PLUGINS from a developer's hosted environment.
    environment = {
        name: os.environ[name]
        for name in ('PATH', 'SYSTEMROOT', 'WINDIR', 'TMPDIR', 'TEMP', 'TMP',
                     'LANG', 'LC_ALL') if name in os.environ
    }
    environment.update({
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        'AWS_EC2_METADATA_DISABLED': 'true',
    })
    # Block Python sockets before importing pytest or collecting tests.
    # The plugin reapplies this guard for each test. Native libraries and
    # subprocesses are outside its scope; see the validation in docs/development/public-tests.md.
    bootstrap = '''
import pytest_socket
pytest_socket.disable_socket()
import pytest, sys

class CollectionSocketGuard:
    def pytest_collection_finish(self, session):
        # Let pytest-socket apply each test's policy, including allow_hosts.
        pytest_socket.enable_socket()

sys.exit(pytest.main(sys.argv[1:], plugins=[pytest_socket, CollectionSocketGuard()]))
'''
    command = [
        sys.executable,
        '-I',
        '-c',
        bootstrap,
        '-c',
        str(root / 'tests/public/pytest.ini'),
        '--disable-socket',
        '-o',
        'cache_dir=/tmp/pytest_cache',
        '--rootdir',
        str(root),
    ]
    if args.junitxml:
        command += [f'--junitxml={args.junitxml.resolve()}']
    if args.k is not None:
        command += ['-k', args.k]
    selected = SUITES if args.suite == 'all' else (args.suite, )
    exit_code = 0
    for suite in selected:
        # Packages have identically named test modules and `fakes` helpers.
        # Separate interpreters prevent one suite from importing another's.
        paths = [str(root / directory) for directory in SUITES[suite]]
        exclusions = [f'--ignore={path}/{tier}' for path in paths
                      for tier in ('live', 'native')]
        targets = [str(test_file)] if test_file else paths
        result = subprocess.run(command + [f'--confcutdir={paths[0]}']
                                + exclusions + targets,
                                cwd=root, env=environment,
                                check=False)
        exit_code = exit_code or result.returncode
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
