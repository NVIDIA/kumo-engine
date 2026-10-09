#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the kumo-data-science skill package for one host and check that it resolves.

    python3 agent-skills/scripts/package_skill.py coco             # build into agent-skills/dist/
    python3 agent-skills/scripts/package_skill.py coco --out DIR   # build somewhere else
    python3 agent-skills/scripts/package_skill.py --check          # build every host in a temp dir

The source tree under `skills/` is not the layout that ships. A package holds one skill directory,
`kumo-data-science/`, and every other source directory nests inside it, so the router reaches
`kumoai-reference/`, `hosts/<host>/` and the capability skills by relative path. Each host gets
the portable core plus only its own host files, so an agent never reads another host's adapter.

The build stamps the router's `SKILL-BUILD` marker with the commit the package was built from,
`<commit date>-<short sha>`, suffixed `-dirty` when `skills/` has uncommitted changes. The source
tree carries `SKILL-BUILD: source`, so a session quoting that value read an unpackaged copy.

Every build is checked: each markdown link must resolve relative to the file that holds it, and
each backticked path naming a package directory must resolve relative to the package root. A
package that references a file it does not ship fails the build.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / 'skills'
DIST = ROOT / 'dist'
ROUTER = 'kumo-data-science'

CAPABILITY = [
    'kumo-model-explanations',
    'kumo-training-diagnostics',
    'kumo-model-persistence',
    'kumo-custom-populations',
    'kumo-model-tuning',
]
HOSTS: dict[str, dict] = {
    # The Cortex extension root holds only `skills/`, with one skill directory.
    'coco': dict(prefix='kumo-ds/skills', hosts=['snowflake-coco/adapter.md'], capability=True),
}

MARKER = re.compile(r'SKILL-BUILD: [^`*\s]+')
MD_LINK = re.compile(r'\]\(([^)\s#]+)(?:#[^)]*)?\)')
CODE_PATH = re.compile(r'`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:md|py))`')


def build_id() -> str:
    """Name the source commit: `<commit date>-<short sha>`, `-dirty` if `skills/` has edits."""
    def git(*args: str) -> str:
        return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    try:
        stamp = git('log', '-1', '--format=%cd-%h', '--date=format:%Y-%m-%d')
        dirty = git('status', '--porcelain', '--', str(SKILLS))
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return f'{stamp}-dirty' if dirty else stamp


def git_tracked(src: Path) -> list[Path]:
    """List the files git tracks under `src`, as absolute paths."""
    rel = src.relative_to(ROOT)
    out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--', str(rel)],
                         check=True, capture_output=True, text=True).stdout
    return [ROOT / line for line in out.splitlines() if line]


def copy_tracked(src: Path, dest: Path) -> None:
    """Copy only the files git tracks under `src` into `dest`, preserving structure.

    `shutil.copytree` would also ship whatever else sits in `src` — a stray `.env`, a log, a
    notebook someone left there — since gitignore keeps those out of commits but not out of the
    working tree. Building the file list from `git ls-files` instead means the package can only
    ever contain what the source commit actually tracks.
    """
    for path in git_tracked(src):
        target = dest / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def build(host: str, dist: Path) -> Path:
    """Assemble the package for `host` under `dist` and return its skill directory."""
    spec = HOSTS[host]
    out = dist / host
    if out.exists():
        shutil.rmtree(out)
    pkg = out / spec['prefix'] / ROUTER
    copy_tracked(SKILLS / ROUTER, pkg)
    copy_tracked(SKILLS / 'kumoai-reference', pkg / 'kumoai-reference')
    for rel in spec['hosts']:
        dest = pkg / 'hosts' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SKILLS / 'hosts' / rel, dest)
    if spec['capability']:
        for name in CAPABILITY:
            copy_tracked(SKILLS / name, pkg / name)
    router = pkg / 'SKILL.md'
    text, n = MARKER.subn(f'SKILL-BUILD: {build_id()}', router.read_text())
    if n != 1:
        raise SystemExit(f'expected one SKILL-BUILD marker in {router}, found {n}')
    router.write_text(text)
    return pkg


def unresolved(pkg: Path) -> list[str]:
    """Return every reference in `pkg` that does not resolve to a shipped file."""
    pkg = pkg.resolve()
    roots = {p.name for p in pkg.iterdir()}
    errors = []
    for doc in sorted(pkg.rglob('*.md')):
        text = doc.read_text()
        where = doc.relative_to(pkg)
        for target in MD_LINK.findall(text):
            if '://' in target or target.startswith('mailto:'):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.is_file():
                errors.append(f'{where}: link ({target}) does not resolve')
            elif not resolved.is_relative_to(pkg):
                errors.append(f'{where}: link ({target}) resolves outside the package root')
        for target in CODE_PATH.findall(text):
            head, _, rest = target.partition('/')
            if head == ROUTER:
                head, _, rest = rest.partition('/')
            if head not in roots:
                continue  # not a package path, e.g. a Kumo source module
            if not (pkg / head / rest).is_file():
                errors.append(f'{where}: `{target}` does not resolve from the package root')
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description='Build and check a kumo-data-science skill package.')
    ap.add_argument('host', nargs='?', choices=sorted(HOSTS))
    ap.add_argument('--out', type=Path, default=DIST, help='output root (default: %(default)s)')
    ap.add_argument('--check', action='store_true', help='build and check every host in a temp dir')
    args = ap.parse_args()
    if not args.check and not args.host:
        ap.error('give a host, or --check')

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for host in sorted(HOSTS) if args.check else [args.host]:
            pkg = build(host, Path(tmp) if args.check else args.out)
            errors = unresolved(pkg)
            files = sum(1 for p in pkg.rglob('*') if p.is_file())
            print(f'{host}: {files} files, build {build_id()}, {len(errors)} unresolved reference(s)')
            for e in errors:
                print(f'  {e}')
            failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
