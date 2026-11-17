#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check component locations, explicit wheel inputs, and local package references."""
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    errors = []
    workspace = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    if 'project' in workspace:
        errors.append('The repository workspace must not also package Studio.')
    members = workspace['tool']['uv']['workspace']['members']
    names = {}
    for member in members:
        manifest = ROOT / member / 'pyproject.toml'
        if not manifest.is_file():
            errors.append(f'Missing workspace manifest: {member}')
            continue
        name = tomllib.loads(manifest.read_text())['project']['name']
        if name in names:
            errors.append(f'Duplicate distribution {name}: {names[name]}, {member}')
        names[name] = member
    if names.get('kumo') != 'apps/studio':
        errors.append('Studio must be packaged from apps/studio.')
    bundle = []
    for line in (ROOT / 'packages/engine-bundle.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        if line in bundle:
            errors.append(f'Duplicate engine bundle input: {line}')
        bundle.append(line)
        if 'packages/' + line not in members:
            errors.append(f'Engine bundle input is not a workspace member: {line}')
    if any(x in bundle for x in ['clients/kumo-sdk', 'data/kumo-pytorch']) or any('experimental/' in x for x in bundle):
        errors.append('Remote SDK, standalone PyTorch API and experiments must not enter the engine bundle.')
    if 'data/diskgraph' in bundle:
        errors.append('DiskGraph must use the separate native build stage.')
    moves = json.loads((ROOT / 'development/repository-moves.json').read_text())['moves']
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for old in moves:
        if old == 'Makefile':
            continue  # Intentionally retained as a compatibility command entry point.
        if any(path == old or path.startswith(old + '/') for path in tracked if path):
            errors.append(f'Obsolete tracked location: {old}')
    for manifest in ROOT.glob('**/package.json'):
        if 'node_modules' in manifest.parts or 'build' in manifest.parts:
            continue
        data = json.loads(manifest.read_text())
        for group in ['dependencies', 'devDependencies', 'optionalDependencies']:
            for name, version in data.get(group, {}).items():
                if isinstance(version, str) and version.startswith('file:'):
                    target = manifest.parent / version[5:]
                    if not (target / 'package.json').is_file():
                        errors.append(f'{manifest.relative_to(ROOT)}: missing local package {name}: {version}')
    dockerignore = (ROOT / 'development/config/dockerignore').read_bytes()
    for name in ['Dockerfile.kumo_backend_cpu', 'Dockerfile.kumo_backend_gpu']:
        if (ROOT / 'apps/studio/docker' / (name + '.dockerignore')).read_bytes() != dockerignore:
            errors.append(f'{name}.dockerignore differs from development/config/dockerignore')
    for path in tracked:
        p = ROOT / path
        if p.is_symlink() and not p.exists():
            errors.append(f'Dangling tracked symlink: {path}')
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f'Layout valid: {len(members)} workspace projects; {len(bundle)} explicit Python bundle inputs.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
