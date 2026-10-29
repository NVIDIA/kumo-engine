#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate and check the fixed small PR172 profiles from pinned source."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source(source, record, *, exact=False):
    source = source.resolve()
    if exact:
        expected = (set(record['copied_files_sha256'])
                    | set(record['generated_profiles_sha256']))
        actual = {
            str(path.relative_to(source))
            for path in source.rglob('*')
            if path.is_file() and path.suffix in ('.py', '.json', '.txt')
        }
        if expected != actual:
            raise ValueError('source inventory differs')
    for name, digest in record['copied_files_sha256'].items():
        path = (source / name).resolve()
        if (not path.is_relative_to(source) or not path.is_file()
                or sha(path) != digest):
            raise ValueError(f'source fingerprint differs: {name}')
    return record


def verify_profiles(source, record):
    for name, digest in record['generated_profiles_sha256'].items():
        if sha(source / name) != digest:
            raise ValueError(f'generated profile fingerprint differs: {name}')


def validate_dataset(root, profiles):
    hashes = {}
    for graph, expected in profiles.items():
        directory = root / graph / 'pr'
        manifest_path = directory / 'manifest.json'
        content = manifest_path.read_bytes()
        manifest = json.loads(content)
        stamp = json.loads((directory / '_VALIDATED.json').read_text())
        digest = hashlib.sha256(content).hexdigest()
        if (stamp.get('ok') is not True
                or stamp.get('checksums_verified') is not True
                or stamp.get('violations') != 0
                or stamp.get('manifest_sha256') != digest):
            raise ValueError(f'invalid validation stamp: {graph}')
        if (manifest.get('graph') != graph or manifest.get('rung') != 'pr'
                or manifest.get('seed') != 20261009
                or manifest.get('shards') != expected['shards']
                or manifest.get('shard_entities') != expected['shard_entities']
                or
            {k: v['rows']
             for k, v in manifest['tables'].items()} != expected['rows']):
            raise ValueError(f'generated profile differs: {graph}')
        for key in ('config_hash', 'profile_digest'):
            if key in expected and manifest.get(key) != expected[key]:
                raise ValueError(f'generated profile differs: {graph}/{key}')
        listed = {(root / entry['path']).resolve()
                  for table in manifest['tables'].values()
                  for entry in table['files']}
        actual = {
            path.resolve()
            for path in directory.rglob('*.parquet') if not any(
                part.startswith('_')
                for part in path.relative_to(directory).parts)
        }
        if actual != listed:
            raise ValueError(f'table inventory differs: {graph}')
        for table in manifest['tables'].values():
            for entry in table['files']:
                path = (root / entry['path']).resolve()
                if (not path.is_relative_to(directory.resolve())
                        or not path.is_file()
                        or path.stat().st_size != entry['bytes']
                        or sha(path) != entry['sha256']):
                    raise ValueError(f'table fingerprint differs: {graph}')
        hashes[graph] = digest
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='Pinned upstream scale-bench directory')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--check', action='store_true',
                        help='Check committed source fingerprints only')
    args = parser.parse_args()
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    if sha(ROOT / 'config.json') != provenance['config_sha256']:
        raise ValueError('source fingerprint differs: config.json')
    upstream = args.source_dir.resolve()
    verify_source(upstream, provenance)
    if args.check:
        print(json.dumps({'source_check': 'PASS'}))
        return
    if args.output_dir is None or args.workers < 1:
        parser.error('--output-dir and positive --workers are required')
    output = args.output_dir.resolve()
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Use Python 3.12 for the frozen generator')
    for package, version in provenance['dependencies'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'generator dependency differs: {package}')
    if output.exists():
        # A complete checked cache is reusable. Partial output is preserved.
        receipt = json.loads((output / 'pr172-inputs.json').read_text())
        if (receipt['data_root'] != str(output / 'data')
                or receipt['source_root'] != str(output / 'source') or
                receipt['provenance_sha256'] != sha(ROOT / 'provenance.json')):
            raise ValueError('cached generator provenance differs')
        verify_source(output / 'source', provenance, exact=True)
        verify_profiles(output / 'source', provenance)
        if receipt['manifest_sha256'] != validate_dataset(
                output / 'data', provenance['expected_profiles']):
            raise ValueError('cached data manifest differs')
        print(json.dumps(receipt))
        return
    output.mkdir(parents=True)
    source = output / 'source'
    data = output / 'data'
    for name in provenance['copied_files_sha256']:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upstream / name, destination)
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in ('PYTHONPATH', 'PYTHONHOME', 'SCALEBENCH_GRAPHS')
    }
    environment['PYTHONPATH'] = str(source)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    started = time.monotonic()
    subprocess.run([
        sys.executable,
        str(source / 'profiles/make_placeholders.py'),
        *provenance['expected_profiles']
    ], cwd=source, env=environment, check=True)
    verify_profiles(source, provenance)
    for graph, profile in provenance['expected_profiles'].items():
        commands = [
            [
                sys.executable, '-m', 'scalebench.generate', 'run', '--graph',
                graph, '--rung', 'pr', '--shards',
                str(profile['shards']), '--workers',
                str(args.workers), '--root',
                str(data), '--config',
                str(ROOT / 'config.json')
            ],
            [
                sys.executable, '-m', 'scalebench.validate', 'check', '--graph',
                graph, '--rung', 'pr', '--root',
                str(data), '--workers',
                str(args.workers), '--duck-mem', '2GB', '--checksums',
                '--row-sample', '20000', '--profile-tol', '1.0',
                '--max-tenure-rate-ratio', '3', '--max-end-density-ratio', '3'
            ],
        ]
        for index, command in enumerate(commands):
            with (output / f'{graph}-{index}.log').open('w') as stream:
                subprocess.run(command, cwd=source, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
    verify_source(source, provenance, exact=True)
    verify_profiles(source, provenance)
    result = {
        'data_root': str(data),
        'source_root': str(source),
        'provenance_sha256': sha(ROOT / 'provenance.json'),
        'manifest_sha256': validate_dataset(data,
                                            provenance['expected_profiles']),
        'generation_seconds': time.monotonic() - started
    }
    (output /
     'pr172-inputs.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
