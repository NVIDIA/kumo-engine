#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that a wheel installs from PyPI, alone and with each of its extras.

Resolves every combination against pypi.org for each Python version the
wheel's classifiers claim, and exits non-zero naming each one that does not
resolve.

    check_pypi_requirements.py WHEEL --platform x86_64-manylinux_2_34
"""

import argparse
import re
import subprocess
import sys
import zipfile
from email.parser import HeaderParser
from pathlib import Path

INDEX = 'https://pypi.org/simple'
PYTHON_CLASSIFIER = re.compile(r'^Programming Language :: Python :: (3\.\d+)$')


def read_metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        name = next(name for name in archive.namelist()
                    if name.endswith('.dist-info/METADATA'))
        return HeaderParser().parsestr(archive.read(name).decode())


def python_versions(metadata):
    versions = [
        match.group(1) for classifier in metadata.get_all('Classifier') or []
        if (match := PYTHON_CLASSIFIER.match(classifier))
    ]
    if not versions:
        raise SystemExit('the wheel declares no Python version classifiers')
    return versions


def resolves(requirement, python, platform):
    """Resolve with uv, pinning the interpreter so its ABI is not assumed."""
    result = subprocess.run([
        'uv', 'pip', 'compile', '--no-config', '--default-index', INDEX,
        '--python', sys.executable, '--python-version', python,
        '--python-platform', platform, '-'
    ], input=requirement, capture_output=True, text=True, check=False)
    if result.returncode:
        print(
            f'{requirement.split(" @ ")[0]} on Python {python}:\n'
            f'{result.stderr}', file=sys.stderr)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--platform', required=True,
                        help='uv target, e.g. x86_64-manylinux_2_34')
    args = parser.parse_args()

    metadata = read_metadata(args.wheel)
    url = args.wheel.resolve().as_uri()
    targets = [metadata['Name']] + [
        f'{metadata["Name"]}[{extra}]'
        for extra in metadata.get_all('Provides-Extra') or []
    ]
    versions = python_versions(metadata)
    failures = [
        f'{target} on Python {python}' for python in versions
        for target in targets
        if not resolves(f'{target} @ {url}', python, args.platform)
    ]
    if failures:
        raise SystemExit('Cannot be installed from PyPI:\n  ' +
                         '\n  '.join(failures))
    print(f'{len(targets)} installs resolve from PyPI on Python '
          f'{", ".join(versions)}: {", ".join(targets)}')


if __name__ == '__main__':
    main()
