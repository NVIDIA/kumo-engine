#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the copy of the skill that the NVIDIA skills catalog publishes.

    python3 agent-skills/scripts/publish_nvskills.py           # rewrite skills/kumo-data-science/
    python3 agent-skills/scripts/publish_nvskills.py --check   # exit 1 if that copy is out of date

The NVIDIA/skills catalog (github.com/NVIDIA/skills) pulls skills from each product repo
every hour. It reads them from `components.d/<slug>.yml` in the catalog, which names a
directory in this repository. It accepts a skill only if the skill's files match the signature
that NVSkills CI writes. NVSkills CI runs only on pull requests that change `skills/` or
`team-skills/` at the repository root. `agent-skills/skills/` is the source layout, not a
package, so the catalog cannot read it directly. This script writes the `generic` package (the one
for Claude Code, Codex and other coding agents) to `skills/kumo-data-science/`. The
`nvskills-publish` workflow runs it on every change to `main` and opens a pull request with
the result. NVSkills CI then signs that pull request.

Three kinds of file sit in the published directory:

* the package, rebuilt from `agent-skills/skills/` on every run;
* the catalog files that NVSkills requires and that we write: `skill-card.md` and
  `evals/*.json`. Their source is `agent-skills/nvskills/kumo-data-science/`, and they are
  copied in on every run;
* `BENCHMARK.md` and `skill.oms.sig`, which NVSkills CI writes into the pull request. They are
  never edited here. A refresh leaves them in place, so they describe the previous content until
  NVSkills CI runs again. The catalog then keeps serving the last signed version rather than
  dropping the skill.

`--check` ignores the `SKILL-BUILD` stamp, which names the commit that built the package and so
differs on every commit even when no skill file changed.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('package_skill', HERE / 'package_skill.py')
assert spec and spec.loader
package_skill = importlib.util.module_from_spec(spec)
sys.modules.setdefault('package_skill', package_skill)
spec.loader.exec_module(package_skill)

HOST = 'generic'
REPO = package_skill.ROOT.parent
PUBLISHED = REPO / 'skills' / package_skill.ROUTER
CATALOG_FILES = package_skill.ROOT / 'nvskills' / package_skill.ROUTER
# Written by NVSkills CI into the pull request; never generated or deleted here.
SIGNED = frozenset({'BENCHMARK.md', 'skill.oms.sig'})


def build(out: Path, catalog_files: Path = CATALOG_FILES) -> Path:
    """Build the package the catalog publishes into `out` and return its skill directory."""
    pkg = package_skill.build(HOST, out)
    errors = package_skill.unresolved(pkg)
    if errors:
        raise SystemExit(f'{HOST}: {len(errors)} unresolved reference(s): {errors[:3]}')
    if catalog_files.is_dir():
        for path in sorted(catalog_files.rglob('*')):
            if path.is_file():
                rel = path.relative_to(catalog_files)
                if rel.as_posix() in SIGNED:
                    raise SystemExit(f'{path}: {rel} is written by NVSkills CI, not by hand')
                target = pkg / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return pkg


def files(root: Path) -> dict[str, bytes]:
    """Map each file under `root` that we own to its bytes, with the build stamp normalized."""
    if not root.is_dir():
        return {}
    out = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and rel not in SIGNED:
            data = path.read_bytes()
            if rel == 'SKILL.md':
                text = package_skill.MARKER.sub('SKILL-BUILD: x', data.decode('utf-8'))
                data = text.encode('utf-8')
            out[rel] = data
    return out


def stale(dest: Path = PUBLISHED, catalog_files: Path = CATALOG_FILES) -> list[str]:
    """Return what `dest` would gain, lose or change if it were refreshed now."""
    with tempfile.TemporaryDirectory() as tmp:
        want = files(build(Path(tmp), catalog_files))
    have = files(dest)
    return ([f'+ {p}' for p in sorted(want.keys() - have.keys())]
            + [f'- {p}' for p in sorted(have.keys() - want.keys())]
            + [f'~ {p}' for p in sorted(want.keys() & have.keys()) if want[p] != have[p]])


def publish(dest: Path = PUBLISHED, catalog_files: Path = CATALOG_FILES) -> list[str]:
    """Rewrite `dest` with the current package, keeping the files NVSkills CI wrote."""
    changes = stale(dest, catalog_files)
    if not changes:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        pkg = build(Path(tmp), catalog_files)
        if dest.is_dir():
            for path in sorted(dest.rglob('*'), reverse=True):
                rel = path.relative_to(dest).as_posix()
                if path.is_file() and rel not in SIGNED:
                    path.unlink()
                elif path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
        for path in sorted(pkg.rglob('*')):
            if path.is_file():
                target = dest / path.relative_to(pkg)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true',
                    help=f'exit 1 if {PUBLISHED.relative_to(REPO)}/ is out of date')
    args = ap.parse_args()
    where = PUBLISHED.relative_to(REPO)
    changes = stale() if args.check else publish()
    for line in changes:
        print(line)
    if not changes:
        print(f'{where}/ is up to date')
    elif args.check:
        print(f'{where}/ is out of date: run {Path(__file__).relative_to(REPO)}', file=sys.stderr)
        raise SystemExit(1)
    else:
        print(f'{where}/ refreshed: {len(changes)} file(s)')


if __name__ == '__main__':
    main()
