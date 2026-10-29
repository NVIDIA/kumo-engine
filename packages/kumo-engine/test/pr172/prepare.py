#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract and check the committed small synthetic PR172 fixtures offline."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_member(member):
    if (member.type != tarfile.REGTYPE or member.name.startswith('/')
            or '\\' in member.name
            or any(part in ('', '.', '..') for part in member.name.split('/'))):
        raise ValueError(f'Unsafe archive member: {member.name}')


def extract_archive(archive, destination, integrity):
    expected = integrity['archive']
    if archive.stat().st_size != expected['bytes'] or sha(archive) != expected['sha256']:
        raise ValueError('Bundled archive fingerprint differs')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        names = [member.name for member in members]
        for member in members:
            safe_member(member)
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        if set(names) != set(integrity['files']):
            raise ValueError('Bundled archive inventory differs')
        for member in members:
            if member.size != integrity['files'][member.name]['bytes']:
                raise ValueError(f'Bundled fixture fingerprint differs: {member.name}')
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
            if sha(target) != integrity['files'][member.name]['sha256']:
                raise ValueError(f'Bundled fixture fingerprint differs: {member.name}')


def verify_files(root, integrity):
    if root.is_symlink():
        raise ValueError('Bundled fixture fingerprint differs: linked data root')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(integrity['files']):
        raise ValueError('Bundled fixture inventory differs')
    for name, expected in integrity['files'].items():
        path = root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size != expected['bytes']
                or sha(path) != expected['sha256']):
            raise ValueError(f'Bundled fixture fingerprint differs: {name}')


def validate_dataset(root, profiles):
    import pyarrow.parquet as pq

    hashes = {}
    for graph, expected in profiles.items():
        directory = root / graph / 'pr'
        content = (directory / 'manifest.json').read_bytes()
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
                or {k: v['rows'] for k, v in manifest['tables'].items()}
                != expected['rows']
                or manifest.get('total_rows') != sum(expected['rows'].values())):
            raise ValueError(f'generated profile differs: {graph}')
        for key in ('config_hash', 'profile_digest'):
            if key in expected and manifest.get(key) != expected[key]:
                raise ValueError(f'generated profile differs: {graph}/{key}')
        listed = {(root / entry['path']).resolve()
                  for table in manifest['tables'].values()
                  for entry in table['files']}
        actual = {path.resolve() for path in directory.rglob('*.parquet')}
        if actual != listed:
            raise ValueError(f'table inventory differs: {graph}')
        for table in manifest['tables'].values():
            total = 0
            for entry in table['files']:
                path = (root / entry['path']).resolve()
                if (not path.is_relative_to(directory.resolve())
                        or path.is_symlink() or not path.is_file()
                        or path.stat().st_size != entry['bytes']
                        or sha(path) != entry['sha256']):
                    raise ValueError(f'table fingerprint differs: {graph}')
                rows = pq.read_metadata(path).num_rows
                if rows != entry['rows']:
                    raise ValueError(f'Parquet row count differs: {graph}')
                total += rows
            if total != table['rows']:
                raise ValueError(f'Parquet row count differs: {graph}')
        hashes[graph] = digest
    return hashes


def prepare_data(output):
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    if sha(ROOT / 'config.json') != provenance['config_sha256']:
        raise ValueError('Fixture configuration fingerprint differs')
    metadata = ROOT / provenance['fixture_bundle']['integrity_file']
    if sha(metadata) != provenance['fixture_bundle']['integrity_sha256']:
        raise ValueError('Fixture integrity metadata fingerprint differs')
    integrity = json.loads(metadata.read_text())
    archive = ROOT / integrity['archive']['filename']
    if archive.parent != ROOT or archive.name != 'small-pr172-data.tar.gz':
        raise ValueError('Invalid bundled archive name')
    output = output.resolve()
    data = output / 'data'
    cache = output.exists()
    started = time.monotonic()
    if cache:
        # A checked complete cache is reusable. Partial output stays visible.
        try:
            receipt = json.loads((output / 'pr172-inputs.json').read_text())
        except (OSError, ValueError) as error:
            raise ValueError('Missing or invalid cached preparation receipt') from error
        if archive.stat().st_size != integrity['archive']['bytes'] or sha(archive) != integrity['archive']['sha256']:
            raise ValueError('Bundled archive fingerprint differs')
    else:
        output.mkdir(parents=True)
        extract_archive(archive, data, integrity)
    verify_files(data, integrity)
    hashes = validate_dataset(data, provenance['expected_profiles'])
    result = dict(data_root=str(data), source_root=str(ROOT),
                  provenance_sha256=sha(ROOT / 'provenance.json'),
                  archive_sha256=integrity['archive']['sha256'],
                  manifest_sha256=hashes, cache_reused=cache,
                  rows={graph: sum(profile['rows'].values())
                        for graph, profile in provenance['expected_profiles'].items()},
                  preparation_seconds=time.monotonic() - started)
    if cache:
        identity = ('data_root', 'source_root', 'provenance_sha256',
                    'archive_sha256', 'manifest_sha256', 'rows')
        if (not isinstance(receipt, dict)
                or any(receipt.get(key) != result[key] for key in identity)):
            raise ValueError('Mismatched cached preparation receipt')
    else:
        (output / 'pr172-inputs.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true',
                        help='Extract and check the bundled fixtures in temporary storage')
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix='kumo-pr172-check-') as directory:
            result = prepare_data(Path(directory) / 'inputs')
        print(json.dumps(dict(fixture_check='PASS', rows=result['rows'],
                              archive_sha256=result['archive_sha256'])))
    else:
        if args.output_dir is None:
            parser.error('--output-dir is required')
        print(json.dumps(prepare_data(args.output_dir)))


if __name__ == '__main__':
    main()
