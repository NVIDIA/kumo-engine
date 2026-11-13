# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage this dispatch's complete wheel and reuse the dedicated dependency cache."""
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


def dependencies(cursor, descriptor, work, wheelhouse=None):
    launch.validate_remote_asset('wheelhouse.tar', descriptor)
    prefix = (launch.STAGE + '/mlops-validation/payload-cache/' +
              descriptor['sha256'] + '/wheelhouse.tar/')
    protected = json.loads(
        Path(__file__).with_name('dependency-asset.json').read_text())
    if (not descriptor['location'].startswith(prefix)
            and descriptor != protected):
        raise ValueError(
            'Dependency archive must use the dedicated payload cache or exact protected asset'
        )
    if wheelhouse is None:
        asset = launch.existing_remote_asset(cursor, 'wheelhouse.tar',
                                             descriptor)
    else:
        if wheelhouse.name != 'wheelhouse.tar':
            raise ValueError('Dependency archive must be named wheelhouse.tar')
        asset = launch.ensure_cached_file(cursor, wheelhouse,
                                          descriptor['sha256'], work,
                                          candidates=[])
    launch.validate_remote_asset('wheelhouse.tar', asset)
    if ((asset['location'] != descriptor['location']
         and not asset['location'].startswith(prefix))
            or asset['sha256'] != descriptor['sha256']
            or asset['size'] != descriptor['size']):
        raise ValueError('Dependency cache returned an unexpected archive')
    asset.pop('action', None)
    return asset


def stage(wheel, source, run, output, descriptor, wheelhouse=None,
          key_file=None):
    from resources import require

    require()
    repo = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                   text=True).strip()
    if source != head or (run is not None and
                          (type(run) is not int or run <= 0)):
        raise ValueError('Candidate build must identify this checkout and run')
    candidate = identity(wheel, source)
    dependency = json.loads(descriptor.read_text())
    manifest = json.loads(
        Path(__file__).with_name('dependencies.json').read_text())
    if dependency.get('sha256') != manifest['archive_sha256']:
        raise ValueError(
            'Dependency archive differs from its package inventory')
    with tempfile.TemporaryDirectory(
            prefix='kumo-snowflake-stage-') as temporary:
        with launch.connect(
                key_file) as connection, connection.cursor() as cursor:
            archive = dependencies(cursor, dependency, Path(temporary),
                                   wheelhouse)
            asset = launch.ensure_cached_file(cursor, wheel,
                                              candidate['wheel_sha256'],
                                              Path(temporary), candidates=[])
            asset.pop('action', None)
    build = dict(source_commit=source)
    if run is not None:
        build['github_run'] = run
    result = dict(candidate=candidate, build=build, assets={
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
        '--dependency-asset', type=Path,
        default=Path(__file__).with_name('dependency-asset.json'))
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument(
        '--key-file', type=Path,
        help='Encrypted local RSA key already unlocked in the memory cache')
    args = parser.parse_args()
    stage(args.wheel, args.source_ref, args.github_run, args.output,
          args.dependency_asset, args.wheelhouse, args.key_file)
    print('Staged this checkout candidate; dependency cache verified')


if __name__ == '__main__':
    main()
