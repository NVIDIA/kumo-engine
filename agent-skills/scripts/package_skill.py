#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the kumo-data-science skill package for one host and check that it resolves.

    python3 agent-skills/scripts/package_skill.py coco             # build into agent-skills/dist/
    python3 agent-skills/scripts/package_skill.py generic          # Claude Code, Codex, other agents
    python3 agent-skills/scripts/package_skill.py genie            # Databricks Genie Code
    python3 agent-skills/scripts/package_skill.py genie --deploy   # + upload to your Genie skills folder
    python3 agent-skills/scripts/package_skill.py coco --out DIR   # build somewhere else
    python3 agent-skills/scripts/package_skill.py --check          # build every host in a temp dir
    python3 agent-skills/scripts/package_skill.py --zip DIR        # one download zip per host

The source tree under `skills/` is not the layout that ships. A package holds one skill directory,
`kumo-data-science/`, and every other source directory nests inside it, so the router reaches
`kumoai-reference/`, `hosts/<host>/` and the capability skills by relative path. Each host gets
the portable core plus only its own host files, so an agent never reads another host's adapter.

The build stamps the router's `SKILL-BUILD` marker with the commit the package was built from,
`<commit date>-<short sha>`, suffixed `-dirty` when `skills/` has uncommitted changes. The source
tree carries `SKILL-BUILD: source`, so a session quoting that value read an unpackaged copy.

`genie --deploy` uploads the package to `/Users/<you>/.assistant/skills/kumo-data-science/`, the
folder Genie Code reads skills from, through the Workspace API, then reads every file back and
compares bytes. It authenticates with `DATABRICKS_HOST` (or `DATABRICKS_URL`) and
`DATABRICKS_TOKEN` from the environment or `--env-file`. Files already in that folder that the
package does not ship are reported, and deleted with `--prune` (only inside
`kumo-data-science/`). A new Genie Code chat picks up the change.

`--zip DIR` writes `kumo-data-science-<host>.zip` for every host, each holding the one
`kumo-data-science/` directory at its root: unzip it into an agent's skills directory, or upload it
where an agent takes a skill zip. The release workflow attaches them to each `kumo-engine-v*`
GitHub release, beside the wheel built from the same commit. Entries are sorted and carry a fixed
timestamp, so one commit always yields the same bytes.

Every build is checked: each markdown link must resolve relative to the file that holds it, and
each backticked path naming a package directory must resolve relative to the package root. A
package that references a file it does not ship fails the build.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
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
    'coco': dict(prefix='kumo-ds/skills',
                 hosts=['snowflake-coco/adapter.md', 'snowflake-coco/runtime.md',
                        'snowflake-coco/ml_job.md'],
                 capability=True),
    # A coding agent outside any data platform. The package root holds the one skill directory,
    # ready to copy into the agent's skills directory.
    'generic': dict(prefix='skills',
                    hosts=['generic/adapter.md', 'generic/runtime.md'],
                    capability=True),
    # Databricks Genie Code. Its skills folder holds the one skill directory, like `generic`.
    'genie': dict(prefix='skills',
                  hosts=['databricks-genie/adapter.md', 'databricks-genie/runtime.md',
                         'databricks-genie/job.md'],
                  capability=True),
}
GENIE_SKILLS = '/Users/{user}/.assistant/skills'

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


SKILL_SUFFIXES = ('.md', '.py')


def git_tracked(src: Path) -> list[Path]:
    """List the files git tracks under `src`, as absolute paths.

    Without git — a source archive, extracted — fall back to the skill's own file types, so a
    stray `.env` or log still cannot ship. `tests/test_package_skill.py` checks that the two lists
    agree on the committed tree.
    """
    rel = src.relative_to(ROOT)
    try:
        out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--', str(rel)],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return skill_files(src)
    return [ROOT / line for line in out.splitlines() if line]


def skill_files(src: Path) -> list[Path]:
    """The files under `src` a skill is made of, found without git."""
    return sorted(p for p in src.rglob('*')
                  if p.is_file() and p.suffix in SKILL_SUFFIXES and '__pycache__' not in p.parts)


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


def bundle(out: Path) -> None:
    """Build every host's package into `out/<host>/kumo-data-science/`, the layout the
    `kumo-engine` wheel ships under `kumo/engine/_skills/` for `kumo.engine.install_skills()`.

    `packages/kumo-engine/setup.py` calls this on every build from a checkout, so a wheel always
    carries the skill built from its own commit. A package with an unresolved reference fails the
    build instead of shipping.
    """
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        for host in sorted(HOSTS):
            pkg = build(host, Path(tmp))
            errors = unresolved(pkg)
            if errors:
                raise SystemExit(f'{host}: {len(errors)} unresolved reference(s): {errors[:3]}')
            shutil.copytree(pkg, out / host / ROUTER)


# A zip entry's timestamp: the earliest a zip can hold, so a rebuild never changes the bytes.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def archive(out: Path) -> list[Path]:
    """Write `out/kumo-data-science-<host>.zip` for every host and return the paths.

    Each zip is the `bundle()` layout for one host: `kumo-data-science/` at its root, the same
    directory `kumo.engine.install_skills()` installs.
    """
    out.mkdir(parents=True, exist_ok=True)
    zips = []
    with tempfile.TemporaryDirectory() as tmp:
        bundle(Path(tmp) / 'bundle')
        for host in sorted(HOSTS):
            root = Path(tmp) / 'bundle' / host
            dest = out / f'{ROUTER}-{host}.zip'
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(p for p in root.rglob('*') if p.is_file()):
                    info = zipfile.ZipInfo(p.relative_to(root).as_posix(), ZIP_EPOCH)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    zf.writestr(info, p.read_bytes())
            zips.append(dest)
    return zips


class Workspace:
    """The few Databricks REST calls a deploy needs, with a personal access token."""

    def __init__(self, host: str, token: str) -> None:
        self.host = host if host.startswith('http') else f'https://{host}'
        self.host = self.host.rstrip('/')
        self.token = token

    def api(self, method: str, path: str, body: dict | None = None,
            query: dict | None = None) -> dict:
        url = self.host + path + ('?' + urllib.parse.urlencode(query) if query else '')
        req = urllib.request.Request(
            url, method=method, data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b'{}')
        except urllib.error.HTTPError as e:
            raise SystemExit(f'{method} {path}: HTTP {e.code} {e.read().decode()[:300]}') from None

    def me(self) -> str:
        return self.api('GET', '/api/2.0/preview/scim/v2/Me')['userName']

    def listing(self, path: str, dirs: bool = False) -> list[str]:
        """Every file under `path`, recursively — or, with `dirs`, every directory, deepest
        first; [] when it does not exist."""
        try:
            objects = self.api('GET', '/api/2.0/workspace/list', query=dict(path=path))
        except SystemExit as e:
            if 'RESOURCE_DOES_NOT_EXIST' in str(e):
                return []
            raise
        out = []
        for o in objects.get('objects', []):
            if o['object_type'] == 'DIRECTORY':
                out += self.listing(o['path'], dirs) + ([o['path']] if dirs else [])
            elif not dirs:
                out.append(o['path'])
        return out

    def delete(self, path: str) -> None:
        """Delete one file or one empty directory; never recursive."""
        self.api('POST', '/api/2.0/workspace/delete', dict(path=path, recursive=False))


def read_env_file(path: Path) -> None:
    """Set KEY=VALUE lines from `path` into the environment, without overriding it."""
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition('=')
        if sep and not key.startswith('#'):
            os.environ.setdefault(key.strip(), value.strip().strip('\'"'))


def deploy_genie(pkg: Path, ws: Workspace, prune: bool = False) -> str:
    """Upload `pkg` to the user's Genie Code skills folder and read every file back.

    Files in the target that the package does not ship — an older build's host directory, say —
    are reported, and with `prune` deleted, along with directories they leave empty. Pruning runs
    before the upload, so a failed delete leaves the old build rather than two host directories.
    Only the `kumo-data-science/` folder is touched; other skills beside it never are, and a listed
    path outside it stops the deploy before anything is deleted. A stale file under `hosts/` is
    refused without `prune` before anything is written: the router expects exactly one host
    directory, so a second one left in place would let Genie read the wrong adapter.
    """
    target = f'{GENIE_SKILLS.format(user=ws.me())}/{ROUTER}'
    files = sorted(p for p in pkg.rglob('*') if p.is_file())
    remote = {f'{target}/{p.relative_to(pkg).as_posix()}': p for p in files}

    def inside(paths: list[str]) -> list[str]:
        outside = [path for path in paths if not path.startswith(target + '/')]
        if outside:
            raise SystemExit(f'workspace listing of {target} returned paths outside it '
                             f'(e.g. {outside[0]}). Nothing was deleted.')
        return paths

    stale = sorted(set(inside(ws.listing(target))) - set(remote))
    stale_hosts = [path for path in stale if path.startswith(f'{target}/hosts/')]
    if stale_hosts and not prune:
        raise SystemExit(f'{len(stale_hosts)} file(s) under {target}/hosts/ are not in this package '
                         f'(e.g. {stale_hosts[0]}); a second host directory would be read. '
                         f'Rerun with --prune to delete them. Nothing was written.')
    if prune:
        for path in stale:
            ws.delete(path)
            print(f'  not in this package, deleted: {path}')
        for d in inside(ws.listing(target, dirs=True)):
            if not any(p.startswith(d + '/') for p in remote) and not ws.listing(d):
                ws.delete(d)
                print(f'  empty directory deleted: {d}')
    for path, p in remote.items():
        ws.api('POST', '/api/2.0/workspace/mkdirs', dict(path=path.rsplit('/', 1)[0]))
        ws.api('POST', '/api/2.0/workspace/import', dict(
            path=path, format='AUTO', overwrite=True,
            content=base64.b64encode(p.read_bytes()).decode()))
    bad = [path for path, p in remote.items()
           if base64.b64decode(ws.api('GET', '/api/2.0/workspace/export',
                                      query=dict(path=path, format='AUTO')).get('content', ''))
           != p.read_bytes()]
    if bad:
        raise SystemExit(f'deploy check failed for {len(bad)} file(s): {bad[:5]}')
    print(f'genie: {len(files)} files deployed and read back byte for byte -> {target}')
    if not prune and stale:
        for path in stale:
            print(f'  not in this package, left in place: {path}')
        print('  --prune deletes these.')
    print('Start a new Genie Code chat (agent mode) to load it; ask it to quote SKILL-BUILD.')
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description='Build and check a kumo-data-science skill package.')
    ap.add_argument('host', nargs='?', choices=sorted(HOSTS))
    ap.add_argument('--out', type=Path, default=DIST, help='output root (default: %(default)s)')
    ap.add_argument('--check', action='store_true', help='build and check every host in a temp dir')
    ap.add_argument('--zip', type=Path, metavar='DIR',
                    help='write kumo-data-science-<host>.zip for every host into DIR')
    ap.add_argument('--deploy', action='store_true',
                    help='genie: upload to your Genie Code skills folder and read it back')
    ap.add_argument('--prune', action='store_true',
                    help='genie --deploy: delete files in the target the package does not ship')
    ap.add_argument('--env-file', type=Path, help='KEY=VALUE file with DATABRICKS_HOST/_TOKEN')
    args = ap.parse_args()
    if args.zip:
        if args.host or args.check or args.deploy:
            ap.error('--zip builds every host; give it alone')
        for z in archive(args.zip):
            with zipfile.ZipFile(z) as zf:
                print(f'{z}: {len(zf.namelist())} files, build {build_id()}')
        return
    if not args.check and not args.host:
        ap.error('give a host, or --check')
    if args.deploy and (args.check or args.host != 'genie'):
        ap.error('--deploy is for the genie host')
    if args.prune and not args.deploy:
        ap.error('--prune goes with --deploy')

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
    if failed:
        sys.exit(1)
    if args.deploy:
        if args.env_file:
            read_env_file(args.env_file)
        host = os.environ.get('DATABRICKS_HOST') or os.environ.get('DATABRICKS_URL')
        token = os.environ.get('DATABRICKS_TOKEN')
        if not host or not token:
            sys.exit('--deploy needs DATABRICKS_HOST (or DATABRICKS_URL) and DATABRICKS_TOKEN')
        deploy_genie(pkg, Workspace(host, token), prune=args.prune)


if __name__ == '__main__':
    main()
