# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject a historical candidate when dispatched product sources have changed."""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from email.parser import BytesParser
from pathlib import Path

PRODUCT_PATHS = ('packages', 'VERSION', 'pyproject.toml', 'setup.py', 'tools',
                 'scripts', 'uv.lock', '.cargo', 'LICENSE', 'THIRD-PARTY.txt',
                 '3rdparty/notices', 'agent-skills/skills',
                 'agent-skills/scripts/package_skill.py')
EXCLUDED = ('test', 'tests', 'docs')


def product_files(repo, revision):
    result = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--full-tree', revision, '--', *PRODUCT_PATHS],
        cwd=repo, text=True)
    files = {}
    for row in result.splitlines():
        metadata, path = row.split('\t', 1)
        if path.startswith('packages/') and any(
                part in EXCLUDED for part in Path(path).parts[2:]):
            continue
        files[path] = metadata.split()[2]
    return files


def verify(repo, candidate, revision='HEAD'):
    build = candidate.get('build', {})
    source = build.get('source_commit')
    if not isinstance(source, str) or not re.fullmatch('[a-f0-9]{40}', source):
        raise ValueError(
            'Candidate lacks committed build provenance; build and record a candidate from the dispatched product sources'
        )
    if build.get('provider_source_commit', source) != source:
        raise ValueError(
            'Candidate providers were built from a different source revision')
    expected, actual = product_files(repo,
                                     source), product_files(repo, revision)
    if not expected or expected != actual:
        raise ValueError(
            'Pinned candidate does not match dispatched product sources; rebuild the candidate before cloud execution'
        )
    return source


def artifact_outputs(candidate):
    build, identity = candidate['build'], candidate['candidate']
    source, run = build['source_commit'], build['github_run']
    filename = identity['wheel_file']
    if (not isinstance(source, str) or not re.fullmatch('[a-f0-9]{40}', source)
            or type(run) is not int or run <= 0
            or not isinstance(filename, str) or
            not re.fullmatch(r'kumo_engine-[A-Za-z0-9_.+\-]+\.whl', filename)):
        raise ValueError('Invalid canonical wheel artifact metadata')
    return dict(run_id=str(run), artifact_name='engine-pr-wheel-' + source,
                wheel_file=filename)


def prepare(wheel, source, output, github_run=None):
    repository = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                   text=True).strip()
    if (source != head or not re.fullmatch('[a-f0-9]{40}', source)
            or wheel.is_symlink() or not wheel.is_file()
            or (github_run is not None and
                (type(github_run) is not int or github_run <= 0))):
        raise ValueError(
            'Require this checkout source and a regular complete wheel')
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith('.dist-info/METADATA')
        ]
        if len(names) != 1:
            raise ValueError('Expected exactly one wheel metadata record')
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    version = metadata['Version']
    if (metadata['Name'] != 'kumo-engine' or not isinstance(version, str)
            or not re.fullmatch(r'[0-9A-Za-z_.]+\+[a-f0-9]{7,40}', version)
            or not source.startswith(version.rsplit('+', 1)[-1])
            or not re.fullmatch(
                r'kumo_engine-' + re.escape(version) +
                r'-cp311-abi3-manylinux_2_[0-9]+_x86_64\.whl', wheel.name)):
        raise ValueError('Wheel metadata and filename must match this source')
    with wheel.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    build = dict(source_commit=source)
    if github_run is not None:
        build['github_run'] = github_run
    result = dict(
        candidate=dict(wheel_file=wheel.name, wheel_sha256=sha,
                       expected_version=version), build=build)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    output.chmod(0o600)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--candidate', type=Path)
    inputs.add_argument('--wheel', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--github-run', type=int)
    parser.add_argument('--source-ref', default='HEAD')
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if args.wheel:
        if args.output is None or args.github_output:
            parser.error(
                '--wheel requires --output and does not use --github-output')
        source = subprocess.check_output(['git', 'rev-parse', args.source_ref],
                                         cwd=repo, text=True).strip()
        prepare(args.wheel, source, args.output, args.github_run)
        print('Verified this checkout wheel identity')
        return
    if args.output is not None or args.github_run is not None:
        parser.error('--output and --github-run require --wheel')
    candidate = json.loads(args.candidate.read_text())
    verify(repo, candidate, args.source_ref)
    if args.github_output:
        with args.github_output.open('a') as output:
            for name, value in artifact_outputs(candidate).items():
                output.write(name + '=' + value + '\n')
    print('Candidate product sources match the selected revision')


if __name__ == '__main__':
    main()
