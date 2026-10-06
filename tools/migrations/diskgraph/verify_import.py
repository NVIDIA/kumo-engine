#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

IMPORT_PREFIX = 'packages/diskgraph'
PROVENANCE = Path('tools/migrations/diskgraph')


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.rstrip('\n')


def tree(repo: Path, revision: str,
         prefix: str | None = None) -> dict[str, dict]:
    args = ['ls-tree', '-r', revision]
    if prefix is not None:
        args.extend(['--', prefix])
    entries: dict[str, dict] = {}
    for line in git(repo, *args).splitlines():
        metadata, path = line.split('\t', 1)
        mode, kind, blob = metadata.split()
        if prefix is not None:
            path = path.removeprefix(f'{prefix}/')
        entries[path] = {'mode': mode, 'type': kind, 'blob': blob}
    return entries


def index_tree(repo: Path, prefix: str) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    output = git(repo, 'ls-files', '--stage', '--', prefix)
    for line in output.splitlines():
        metadata, path = line.split('\t', 1)
        mode, blob, stage = metadata.split()
        if stage != '0':
            raise SystemExit(f'unmerged DiskGraph path: {path}')
        path = path.removeprefix(f'{prefix}/')
        entries[path] = {'mode': mode, 'type': 'blob', 'blob': blob}
    return entries


def manifest_tree(manifest: dict) -> dict[str, dict]:
    return {
        entry['source_path']: {
            'mode': entry['mode'],
            'type': entry['type'],
            'blob': entry['blob'],
        }
        for entry in manifest['entries']
    }


def compare(source: dict[str, dict], imported: dict[str, dict]) -> None:
    if source == imported:
        return
    source_paths = set(source)
    imported_paths = set(imported)
    differences = sorted(path for path in source_paths & imported_paths
                         if source[path] != imported[path])
    raise SystemExit('DiskGraph mirror differs from its source: '
                     f'missing={sorted(source_paths - imported_paths)}, '
                     f'unexpected={sorted(imported_paths - source_paths)}, '
                     f'changed={differences}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-repo', type=Path)
    parser.add_argument('--kumo-repo', type=Path, default=Path.cwd())
    parser.add_argument('--index', action='store_true')
    args = parser.parse_args()

    kumo_repo = args.kumo_repo.resolve()
    provenance = kumo_repo / PROVENANCE
    source_sha = (provenance / 'source_revision.txt').read_text().strip()
    manifest = json.loads(
        (provenance / 'generated/source_manifest.json').read_text())
    if manifest['source_sha'] != source_sha:
        raise SystemExit('source revision and manifest disagree')

    imported = (index_tree(kumo_repo, IMPORT_PREFIX) if args.index else tree(
        kumo_repo, 'HEAD', IMPORT_PREFIX))
    expected = manifest_tree(manifest)
    compare(expected, imported)
    if args.source_repo is not None:
        source_repo = args.source_repo.resolve()
        compare(tree(source_repo, source_sha), imported)

    print(f'verified {len(imported)} DiskGraph files at {source_sha} '
          f'under {IMPORT_PREFIX}')


if __name__ == '__main__':
    main()
