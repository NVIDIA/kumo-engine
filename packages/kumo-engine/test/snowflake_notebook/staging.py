# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the current candidate dependencies and stage content-addressed wheels."""
import argparse
import email.parser
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import launch


def identity(wheel, source):
    if (wheel.is_symlink() or not wheel.is_file()
            or not re.fullmatch('[a-f0-9]{40}', source)):
        raise ValueError('A regular wheel and full source commit are required')
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith('.dist-info/METADATA')
        ]
        if len(names) != 1:
            raise ValueError('Expected one complete wheel metadata record')
        metadata = email.parser.Parser().parsestr(
            archive.read(names[0]).decode())
    version = metadata['Version']
    if (metadata['Name'] != 'kumo-engine' or not isinstance(version, str)
            or not re.fullmatch(r'[0-9A-Za-z_.]+\+[a-f0-9]{7,40}', version)
            or not source.startswith(version.rsplit('+', 1)[-1])
            or not re.fullmatch(
                r'kumo_engine-' + re.escape(version) +
                r'-cp311-abi3-manylinux_2_[0-9]+_x86_64\.whl', wheel.name)):
        raise ValueError('Wheel identity must match this source revision')
    return dict(wheel_file=wheel.name, wheel_sha256=launch.sha(wheel),
                expected_version=version)


def stage(wheel, source, run, output, key_file=None):
    from resources import require

    require()
    repo = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                   text=True).strip()
    if source != head or (run is not None and
                          (type(run) is not int or run <= 0)):
        raise ValueError('Candidate build must identify this checkout and run')
    candidate = identity(wheel, source)
    from resolve_dependencies import resolve

    with tempfile.TemporaryDirectory(
            prefix='kumo-snowflake-stage-') as temporary:
        work = Path(temporary)
        plan = json.loads(
            Path(__file__).with_name('dependency-plan.json').read_text())
        wheelhouse = resolve(wheel, work / 'dependencies', plan)
        manifest = json.loads(
            wheelhouse.with_name('dependencies.json').read_text())
        with launch.connect(
                key_file) as connection, connection.cursor() as cursor:
            archive = launch.ensure_cached_file(cursor, wheelhouse,
                                                manifest['archive_sha256'],
                                                work, candidates=[])
            asset = launch.ensure_cached_file(cursor, wheel,
                                              candidate['wheel_sha256'], work,
                                              candidates=[])
            for value in (archive, asset):
                value.pop('action', None)
    build = dict(source_commit=source)
    if run is not None:
        build['github_run'] = run
    result = dict(candidate=candidate, build=build, dependencies=manifest,
                  assets={
                      wheel.name: asset,
                      'wheelhouse.tar': archive
                  })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    output.chmod(0o600)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--source-ref', required=True)
    parser.add_argument('--github-run', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--key-file', type=Path,
        help='Encrypted local RSA key already unlocked in the memory cache')
    args = parser.parse_args()
    stage(args.wheel, args.source_ref, args.github_run, args.output,
          args.key_file)
    print('Staged this checkout candidate; dependency cache verified')


if __name__ == '__main__':
    main()
