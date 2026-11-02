#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable nightly data and remove owned temporary work."""

import argparse
import hashlib
import json
import os
import re
import shutil
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


def prepare(value, work):
    import boto3

    for name in ('inputs', 'reference', 'policy'):
        item = value[name]
        target = work / ('inputs.tar' if name == 'inputs' else name + '.json')
        boto3.client('s3', region_name=item['region']).download_file(
            item['bucket'], item['key'], str(target))
        with target.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise ValueError('Downloaded object checksum differs: ' + name)
    with tarfile.open(work / 'inputs.tar') as archive:
        members = archive.getmembers()
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise ValueError('Input archive must contain regular data files')
        if sum(m.size for m in members) > shutil.disk_usage(work).free:
            raise ValueError('Insufficient free disk to extract nightly inputs')
        archive.extractall(work / 'inputs', filter='data')
    (work / 'inputs.tar').unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('create', 'prepare', 'cleanup'))
    args = parser.parse_args()
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
    main()
