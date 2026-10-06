#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

IMPORT_PREFIX = Path('packages/diskgraph')
PROVENANCE = Path('tools/migrations/diskgraph')


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=text,
    ).stdout


def source_tree(repo: Path, revision: str) -> dict[str, dict[str, str]]:
    entries = {}
    output = str(git(repo, 'ls-tree', '-r', revision)).rstrip('\n')
    for line in output.splitlines():
        metadata, path = line.split('\t', 1)
        mode, kind, blob = metadata.split()
        if kind != 'blob':
            raise SystemExit(f'unsupported Git object {kind}: {path}')
        entries[path] = {'mode': mode, 'type': kind, 'blob': blob}
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--kumo-repo', type=Path, default=Path.cwd())
    args = parser.parse_args()

    source_repo = args.source_repo.resolve()
    kumo_repo = args.kumo_repo.resolve()
    source_sha = str(
        git(source_repo, 'rev-parse', f'{args.source_sha}^{{commit}}'))
    source_sha = source_sha.strip()
    entries = source_tree(source_repo, source_sha)

    destination = kumo_repo / IMPORT_PREFIX
    shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path, metadata in entries.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = bytes(
            git(source_repo, 'show', f'{source_sha}:{path}', text=False))
        if metadata['mode'] == '120000':
            target.symlink_to(data.decode())
        else:
            target.write_bytes(data)
            target.chmod(0o755 if metadata['mode'] == '100755' else 0o644)

    provenance = kumo_repo / PROVENANCE
    (provenance / 'source_revision.txt').write_text(f'{source_sha}\n')
    manifest = {
        'source_repository':
        'kumo-ai/kumo-diskgraph',
        'source_sha':
        source_sha,
        'source_reachable_commit_count':
        int(str(git(source_repo, 'rev-list', '--count', source_sha))),
        'import_prefix':
        IMPORT_PREFIX.as_posix(),
        'file_count':
        len(entries),
        'entries': [{
            'source_path': path,
            'destination_path': f'{IMPORT_PREFIX.as_posix()}/{path}',
            **entries[path],
        } for path in sorted(entries)],
    }
    manifest_path = provenance / 'generated/source_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'synchronized {len(entries)} files from {source_sha}')


if __name__ == '__main__':
    main()
