#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable nightly data and remove owned temporary work."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


def configuration(raw):
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {
            'format', 'min_free_bytes', 'inputs', 'reference', 'policy',
            'input_manifest_sha256'
    } or value['format'] != 'kumo-engine-nightly-job-v1'
            or type(value['min_free_bytes']) is not int
            or value['min_free_bytes'] <= 0 or
            not re.fullmatch('[0-9a-f]{64}', value['input_manifest_sha256'])):
        raise ValueError(
            'Bind reviewed nightly objects and required free space')
    for name in ('inputs', 'reference', 'policy'):
        item = value[name]
        if (name != 'inputs' and isinstance(item, dict)
                and set(item) == {'path', 'sha256'}):
            path, checksum = item['path'], item['sha256']
            if (not isinstance(path, str) or not path or '\\' in path
                    or '\n' in path or '\r' in path or path.startswith('/')
                    or any(p in ('', '.', '..') for p in path.split('/'))
                    or not isinstance(checksum, str)
                    or not re.fullmatch('[0-9a-f]{64}', checksum)):
                raise ValueError(
                    'Local reviewed data need a relative path and checksum')
            continue
        if (not isinstance(item, dict)
                or set(item) != {'bucket', 'region', 'key', 'sha256'}
                or any(not isinstance(v, str) or not v or '\n' in v or '\r' in v
                       for v in item.values())
                or not re.fullmatch('[a-z0-9][a-z0-9.-]*', item['bucket'])
                or not re.fullmatch('[a-z0-9-]+', item['region'])
                or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                or '\\' in item['key'] or item['key'].startswith('/')
                or any(p in ('', '.', '..') for p in item['key'].split('/'))):
            raise ValueError(
                'Nightly objects need regular S3 keys and checksums')
    return value


def create_work(root, minimum_free):
    root = Path(root).resolve()
    free = shutil.disk_usage(root).free
    if free < minimum_free:
        raise ValueError(
            f'Insufficient free disk: {free} < {minimum_free} bytes')
    work = Path(tempfile.mkdtemp(prefix='engine-nightly-', dir=root))
    (work / '.nightly-job').write_text('kumo-engine-nightly-job-v1\n')
    return work


def owned_work(work, runner_temp):
    work = Path(work)
    marker = work / '.nightly-job'
    if (work.is_symlink()
            or work.resolve().parent != Path(runner_temp).resolve()
            or not work.name.startswith('engine-nightly-')
            or not marker.is_file() or marker.is_symlink()
            or marker.read_text() != 'kumo-engine-nightly-job-v1\n'):
        raise ValueError('Cleanup requires this runner\'s marked job directory')
    return work


def cleanup_work(work, runner_temp):
    work = owned_work(work, runner_temp)
    shutil.rmtree(work)


def stage_object(item, target, local_root):
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        if 'path' in item:
            source = local_root / item['path']
            if (source.is_symlink() or not source.is_file() or
                    not source.resolve().is_relative_to(local_root.resolve())):
                raise ValueError(
                    'Local reviewed data must stay inside the repository')
            shutil.copyfile(source, temporary)
        else:
            import boto3
            boto3.client('s3', region_name=item['region']).download_file(
                item['bucket'], item['key'], str(temporary))
        with temporary.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise ValueError('Staged object checksum differs')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(value, work):
    local_root = Path(__file__).parents[1] / 'test/nightly'

    for name in ('inputs', 'reference', 'policy'):
        item = value[name]
        target = work / ('inputs.tar' if name == 'inputs' else name + '.json')
        stage_object(item, target, local_root)
    with tarfile.open(work / 'inputs.tar') as archive:
        members = archive.getmembers()
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise ValueError('Input archive must contain regular data files')
        if sum(m.size for m in members) > shutil.disk_usage(work).free:
            raise ValueError('Insufficient free disk to extract nightly inputs')
        archive.extractall(work / 'inputs', filter='data')
    (work / 'inputs.tar').unlink()


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'build':
        parser = argparse.ArgumentParser(
            description='Build one identified wheel')
        parser.add_argument('--source', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(arguments[1:])
        builder = Path(
            __file__).parents[3] / '.github/actions/build-engine-wheel/build.py'
        print(json.dumps(load_script(builder).build(args.source, args.output)))
        return 0
    if arguments and arguments[0] == 'run':
        parser = argparse.ArgumentParser(
            description='Run one case with repository data bindings')
        parser.add_argument(
            '--configuration', type=Path,
            default=Path(__file__).parents[1] / 'test/nightly/inputs.json')
        parser.add_argument('--build-receipt', type=Path, required=True)
        parser.add_argument('--reference', type=Path)
        parser.add_argument('--policy', type=Path)
        args, forwarded = parser.parse_known_args(arguments[1:])
        value = configuration(args.configuration.read_text())
        bound = [
            '--build-receipt',
            str(args.build_receipt), '--expected-version',
            json.loads(args.build_receipt.read_text())['version'],
            '--input-sha256', value['input_manifest_sha256']
        ]
        for name in ('reference', 'policy'):
            path = getattr(args, name)
            if path is None:
                if 'path' not in value[name]:
                    parser.error('Stage the remote ' + name +
                                 ' and pass its path')
                path = args.configuration.parent / value[name]['path']
            bound += [
                '--' + name,
                str(path), '--' + name + '-sha256', value[name]['sha256']
            ]
        return load_script(
            Path(__file__).with_name('nightly_check.py')).main(forwarded +
                                                               bound)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('create', 'prepare', 'cleanup'))
    args = parser.parse_args(arguments)
    root = Path(os.environ['RUNNER_TEMP'])
    if args.command == 'cleanup':
        work = os.environ.get('NIGHTLY_JOB_ROOT')
        if work:
            owned_work(Path(work), root)
            evidence = Path(work + '-evidence')
            if (os.environ.get('NIGHTLY_EVIDENCE_UPLOADED') == 'true'
                    and evidence.is_dir() and not evidence.is_symlink()
                    and evidence.parent.resolve() == root.resolve()
                    and (Path(work) / '.nightly-evidence').is_file()
                    and not (Path(work) / '.nightly-evidence').is_symlink()
                    and (Path(work) / '.nightly-evidence').read_text()
                    == str(evidence)):
                shutil.rmtree(evidence)
            cleanup_work(Path(work), root)
        return
    value = configuration(os.environ['NIGHTLY_CONFIGURATION'])
    if args.command == 'create':
        work = create_work(root, value['min_free_bytes'])
        evidence = Path(str(work) + '-evidence')
        evidence.mkdir()
        (work / '.nightly-evidence').write_text(str(evidence))
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write(
                f'NIGHTLY_JOB_ROOT={work}\nNIGHTLY_EVIDENCE_ROOT={evidence}\n')
            stream.write('NIGHTLY_MANIFEST_SHA256=' +
                         value['input_manifest_sha256'] + '\n')
            stream.write('NIGHTLY_REFERENCE_SHA256=' +
                         value['reference']['sha256'] + '\n')
            stream.write('NIGHTLY_POLICY_SHA256=' + value['policy']['sha256'] +
                         '\n')
        print(
            json.dumps(
                dict(
                    free_bytes=shutil.disk_usage(root).free,
                    required_free_bytes=value['min_free_bytes'],
                    cpus=os.cpu_count(),
                    system_memory_bytes=os.sysconf('SC_PAGE_SIZE') *
                    os.sysconf('SC_PHYS_PAGES'))))
        return
    work = Path(os.environ['NIGHTLY_JOB_ROOT'])
    owned_work(work, root)
    prepare(value, work)


if __name__ == '__main__':
    raise SystemExit(main())
