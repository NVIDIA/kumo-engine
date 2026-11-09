# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private local trial copies without shared inodes or page-cache state."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from io_utils import fsutil


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with fsutil.get_binary_handle(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_tree(root: str) -> list[str]:
    result = []
    for path in fsutil.listdir(root):
        if fsutil.isdir(path):
            result.extend(file_tree(path))
        else:
            result.append(path)
    return sorted(result)


def baseline_state(root: str) -> dict:
    """Inventory all files and hash small metadata, including extra extensions."""
    inventory, metadata = [], {}
    for path in file_tree(root):
        relative = str(Path(path).relative_to(root))
        size = fsutil.filesize(path)
        # fsutil has no inode/mtime API; detect same-size writes without a full
        # graph reread inside the trial scope. Full hashes are pinned at clone.
        status = os.stat(path)
        inventory.append({'path': relative, 'size': size,
                          'inode': status.st_ino, 'device': status.st_dev,
                          'mtime_ns': status.st_mtime_ns})
        if path.endswith('.json') and size <= 1 << 20:
            metadata[relative] = file_hash(path)
    return {'baseline_inventory': inventory,
            'baseline_metadata_sha256': metadata}


def verify_baseline(manifest: dict) -> None:
    isolation = manifest['isolation']
    root = isolation['source_root']
    if file_hash(f'{root}/manifest.json') != isolation['source_manifest_sha256']:
        raise ValueError('Prepared baseline manifest changed during the trial')
    actual = baseline_state(root)
    for key in ('baseline_inventory', 'baseline_metadata_sha256'):
        if actual[key] != isolation[key]:
            raise ValueError(f'Prepared baseline changed during the trial: {key}')


def private_copy(source: str, destination: str) -> dict:
    """Copy through fsutil handles; inode checks are specific to this Linux test."""
    fsutil.makedirs(str(Path(destination).parent), exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with fsutil.get_binary_handle(source, 'rb') as reader:
        with fsutil.get_binary_handle(destination, 'wb') as writer:
            for block in iter(lambda: reader.read(8 << 20), b''):
                writer.write(block)
                digest.update(block)
                size += len(block)
    # fsutil has no inode identity API; independent inodes are a benchmark gate.
    left, right = os.stat(source), os.stat(destination)
    if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
        raise ValueError('A trial copy shares its baseline inode')
    if size != fsutil.filesize(source) or size != fsutil.filesize(destination):
        raise ValueError('A prepared file changed during copying')
    return {'size': size, 'sha256': digest.hexdigest()}


def clone_prepared(prepared_dir: str, destination: str,
                   expected_sha256: dict[str, str] | None = None) -> dict:
    root, target = Path(prepared_dir).resolve(), Path(destination).resolve()
    if root == target or target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError('Trial and baseline directories must be disjoint')
    if fsutil.exists(str(target)):
        raise ValueError('Trial directory already exists; refusing to reuse it')
    manifest_path = str(root / 'manifest.json')
    manifest = json.loads(fsutil.read_text(manifest_path))
    if 'isolation' in manifest:
        raise ValueError('Clone a baseline, not an existing trial')
    if expected_sha256 is None and 'pristine_owned_files' in manifest:
        expected_sha256 = {record['path']: record['sha256']
                           for record in manifest['pristine_owned_files']}
    if expected_sha256 is not None:
        expected_paths = {str(Path(path).relative_to(root))
                          for path in manifest['owned_files']}
        if set(expected_sha256) != expected_paths:
            raise ValueError('Pristine hashes must exactly cover the owned-file whitelist')
    started = time.perf_counter()
    state = baseline_state(str(root))
    manifest_sha = file_hash(manifest_path)
    fsutil.makedirs(str(target))

    def remap(path: str) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root) or resolved == root:
            raise ValueError(f'Prepared file escapes the baseline: {path}')
        return str(target / resolved.relative_to(root))

    copied = []
    for source in sorted(set(manifest['owned_files'])):
        if Path(source).is_symlink() or not fsutil.exists(source):
            raise ValueError(f'Invalid prepared file: {source}')
        path = remap(source)
        record = private_copy(source, path)
        relative = str(Path(source).relative_to(root))
        if (expected_sha256 is not None
                and record['sha256'] != expected_sha256[relative]):
            raise ValueError(f'Prepared file differs from pristine hash: {relative}')
        copied.append({'path': relative, **record})
    manifest['snapshot']['index_path'] = remap(manifest['snapshot']['index_path'])
    for key in ('train_table_path', 'manifest_path', 'staging_dir'):
        manifest['training'][key] = remap(manifest['training'][key])
    manifest['inputs'] = remap(manifest['inputs'])
    for key in ('owned_files', 'index_files'):
        manifest[key] = [remap(path) for path in manifest[key]]
    manifest['isolation'] = {
        'source_root': str(root), 'trial_root': str(target),
        'source_manifest_sha256': manifest_sha, **state,
        'pristine_files': copied, 'clone_seconds': time.perf_counter() - started,
        'copy_mode': 'private_inode_full_copy',
    }
    verify_baseline(manifest)
    fsutil.write_text(str(target / 'manifest.json'), json.dumps(
        manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def validate_clone(manifest: dict, directory: str) -> None:
    root = Path(directory).resolve()
    isolation = manifest.get('isolation')
    if isolation is None or Path(isolation['trial_root']) != root:
        raise ValueError('A measured trial requires its own private prepared clone')
    expected = root / Path(manifest['snapshot']['index_path']).relative_to(root)
    if not fsutil.isdir(str(expected)):
        raise ValueError('Private snapshot index is missing')
    for path in manifest['owned_files']:
        relative = Path(path).relative_to(root)
        source = str(Path(isolation['source_root']) / relative)
        left, right = os.stat(source), os.stat(path)
        if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
            raise ValueError('Trial uses a shared baseline inode')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    manifest = clone_prepared(args.prepared_dir, args.destination)
    print(json.dumps({'clone_seconds': manifest['isolation']['clone_seconds'],
                      'files': len(manifest['owned_files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
