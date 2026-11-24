#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the copy of the skills that the NVIDIA skills catalog publishes.

    python3 agent-skills/scripts/publish_nvskills.py           # rewrite skills/
    python3 agent-skills/scripts/publish_nvskills.py --check   # exit 1 if that copy is out of date

The NVIDIA/skills catalog (github.com/NVIDIA/skills) pulls skills from each product repo
every hour. It reads them from `components.d/<slug>.yml` in the catalog, which names one
directory in this repository per skill. It accepts a skill only if the skill's files match the
signature that NVSkills CI writes. NVSkills CI runs only on pull requests that change `skills/` or
`team-skills/` at the repository root. `agent-skills/skills/` is the source layout, not a
package, so the catalog cannot read it directly. This script writes the `generic` package (the one
for Claude Code, Codex and other coding agents) to `skills/`. A PR that changes the skill runs it
and commits the result, and `tests/test_publish_nvskills.py` fails the PR when it did not.
NVSkills CI signs that same PR, so merging it to `main` is what publishes.

The catalog does not allow a skill inside another skill. The package nests the seven other skills
inside the router, `kumo-data-science/`, so this script lays them out beside it instead: eight
directories under `skills/`, each holding one `SKILL.md`. Their links to one another resolve the
same way in both layouts (`../kumo-model-persistence/SKILL.md`), and a backticked path that starts
with a skill's name is read from the directory holding the skills, so the build checks every
reference again in the flat layout.

The catalog sync deletes any skill directory that lacks its own `skill.oms.sig`, `skill-card.md`
and eval dataset. Three kinds of file sit in each published skill directory:

* the package, rebuilt from `agent-skills/skills/` on every run;
* the catalog files that NVSkills requires and that we write: the skill's `skill-card.md` and
  `evals/*.json`. Their source is `agent-skills/nvskills/<skill>/`, and they are copied in on
  every run. A catalog file must belong to a skill and may not replace a package file. Once that
  source exists, the build fails unless every skill has both, so the sync never drops one.
  The card is the exception to "every run": NVSkills CI rewrites it when it signs, adding the
  evaluation results, so once a skill's directory holds a `skill.oms.sig` its card is NVSkills
  CI's. A refresh then leaves it as it is and `--check` does not compare it. The source card
  seeds a skill's first card; to change a signed one, edit `skills/<skill>/skill-card.md`;
* `BENCHMARK.md` and `skill.oms.sig`, which NVSkills CI writes into the pull request. They are
  never edited here, and the build fails if the package or the catalog files carry either
  name at any depth. A refresh leaves them in place at any depth, so they describe the previous
  content until NVSkills CI runs again. The catalog then keeps serving the last signed version
  rather than dropping the skill.

Everything else under `skills/` is this script's output: a refresh deletes what the build no
longer produces and rewrites only the files whose content changed, so a skill whose files did not
change keeps its bytes, including the router's `SKILL-BUILD` stamp, and is not re-signed.

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
ROUTER = package_skill.ROUTER
REPO = package_skill.ROOT.parent
PUBLISHED = REPO / 'skills'
CATALOG_FILES = package_skill.ROOT / 'nvskills'
# Written by NVSkills CI into the pull request; never generated or deleted here.
SIGNED = frozenset({'BENCHMARK.md', 'skill.oms.sig'})
# What each skill directory needs from the catalog files; the sync drops a skill without them.
CARD = 'skill-card.md'
EVALS = 'evals'


def skill_dirs(root: Path) -> list[str]:
    """Each skill directory of the flat layout under `root`, by name."""
    return sorted(p.parent.name for p in root.glob('*/SKILL.md'))


def nested(root: Path) -> list[str]:
    """Every `SKILL.md` under `root` that is not at the top of its own skill directory."""
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('SKILL.md')
                  if len(p.relative_to(root).parts) != 2)


def owner(rel: Path) -> str | None:
    """The skill a catalog file belongs to, or None if it is not a catalog input."""
    parts = rel.parts
    if len(parts) == 2 and parts[1] == CARD:
        return parts[0]
    if len(parts) > 2 and parts[1] == EVALS:
        return parts[0]
    return None


def missing(root: Path) -> list[str]:
    """The catalog inputs a skill directory under `root` lacks, as `root`-relative paths."""
    out = []
    for skill in skill_dirs(root):
        d = root / skill
        if not (d / CARD).is_file():
            out.append(f'{skill}/{CARD}')
        if not any((d / EVALS).glob('*.json')):
            out.append(f'{skill}/{EVALS}/*.json')
    return out


def unresolved(root: Path) -> list[str]:
    """Return every reference in the flat layout under `root` that does not resolve.

    A markdown link resolves from the file that holds it. A backticked path resolves from `root`
    when it starts with a skill's name, and from its own skill directory when it starts with one
    of that skill's entries. A path that starts with another skill's entry would be read from the
    wrong directory, so it is an error.
    """
    root = root.resolve()
    skills = set(skill_dirs(root))
    entries = {s: {p.name for p in (root / s).iterdir()} for s in skills}
    errors = []
    for doc in sorted(root.rglob('*.md')):
        text = doc.read_text(encoding='utf-8')
        where = doc.relative_to(root).as_posix()
        skill = doc.relative_to(root).parts[0]
        for target in package_skill.MD_LINK.findall(text):
            if '://' in target or target.startswith('mailto:'):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.is_file() or not resolved.is_relative_to(root):
                errors.append(f'{where}: link ({target}) does not resolve')
        for target in package_skill.CODE_PATH.findall(text):
            head = target.split('/', 1)[0]
            if head in skills:
                path = root / target
            elif head in entries.get(skill, ()):
                path = root / skill / target
            elif any(head in names for names in entries.values()):
                errors.append(f'{where}: `{target}` is another skill\'s file: start it with '
                              f'that skill\'s name')
                continue
            else:
                continue  # not a package path, e.g. a Kumo source module
            if not path.is_file():
                errors.append(f'{where}: `{target}` does not resolve')
    return errors


def flatten(pkg: Path, out: Path) -> Path:
    """Lay the package `pkg` out under `out` with every skill beside the router."""
    inner = sorted(p.parent.relative_to(pkg) for p in pkg.rglob('SKILL.md') if p.parent != pkg)
    deep = [p.as_posix() for p in inner if len(p.parts) != 1]
    if deep:
        raise SystemExit(f'{HOST} package nests skills below its top level: {deep}')
    names = {p.name for p in inner}
    shutil.copytree(pkg, out / ROUTER,
                    ignore=lambda d, entries: names & set(entries) if Path(d) == pkg else ())
    for name in sorted(names):
        shutil.copytree(pkg / name, out / name)
    return out


def build(out: Path, catalog_files: Path = CATALOG_FILES) -> Path:
    """Build the flat layout the catalog publishes into `out` and return it."""
    pkg = package_skill.build(HOST, out / 'package')
    errors = package_skill.unresolved(pkg)
    if errors:
        raise SystemExit(f'{HOST}: {len(errors)} unresolved reference(s): {errors[:3]}')
    # A signed name in the package would overwrite what NVSkills CI wrote, unseen by --check.
    signed = [p.relative_to(pkg).as_posix() for p in pkg.rglob('*') if p.name in SIGNED]
    if signed:
        raise SystemExit(f'{HOST} package ships {signed}: written by NVSkills CI, not by hand')
    root = flatten(pkg, out / 'skills')
    errors = unresolved(root)
    if errors:
        raise SystemExit(f'{HOST}, one directory per skill: {len(errors)} unresolved '
                         f'reference(s): {errors[:3]}')
    if catalog_files.is_dir():
        skills = skill_dirs(root)
        for path in sorted(catalog_files.rglob('*')):
            if path.is_file():
                rel = path.relative_to(catalog_files)
                if path.name in SIGNED:
                    raise SystemExit(f'{path}: {rel} is written by NVSkills CI, not by hand')
                if owner(rel) not in skills:
                    raise SystemExit(f'{path}: {rel} is not the {CARD} or {EVALS}/ of a skill '
                                     f'in the {HOST} package ({skills})')
                target = root / rel
                if target.exists():
                    raise SystemExit(f'{path}: {rel} is already in the {HOST} package')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        # The catalog sync deletes a skill directory without its own card and evals, and with
        # it the files the other skills link to. Before the first one is written nothing is
        # published, so the gap is reported (`not_ready`) rather than fatal.
        gaps = missing(root)
        if gaps:
            raise SystemExit(f'{catalog_files}: the catalog would drop these skills; add '
                             f'{gaps}')
    return root


def not_ready(catalog_files: Path = CATALOG_FILES) -> list[str]:
    """Catalog inputs still to write before the first publish; empty once they all exist."""
    with tempfile.TemporaryDirectory() as tmp:
        return missing(build(Path(tmp), catalog_files))


def signed_cards(root: Path) -> frozenset[str]:
    """The cards under `root` that NVSkills CI rewrote when it signed their skill."""
    return frozenset(f'{skill}/{CARD}' for skill in skill_dirs(root)
                     if (root / skill / 'skill.oms.sig').is_file()
                     and (root / skill / CARD).is_file())


def files(root: Path, kept: frozenset[str] = frozenset()) -> dict[str, bytes]:
    """Map each file under `root` that we own to its bytes, with the build stamp normalized.

    `kept` names files NVSkills CI owns besides the signed ones: they are left out.
    """
    if not root.is_dir():
        return {}
    out = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and path.name not in SIGNED and rel not in kept:
            data = path.read_bytes()
            if rel == f'{ROUTER}/SKILL.md':
                text = package_skill.MARKER.sub('SKILL-BUILD: x', data.decode('utf-8'))
                data = text.encode('utf-8')
            out[rel] = data
    return out


def stale(dest: Path = PUBLISHED, catalog_files: Path = CATALOG_FILES) -> list[str]:
    """Return what `dest` would gain, lose or change if it were refreshed now."""
    kept = signed_cards(dest)
    with tempfile.TemporaryDirectory() as tmp:
        want = files(build(Path(tmp), catalog_files), kept)
    have = files(dest, kept)
    return ([f'+ {p}' for p in sorted(want.keys() - have.keys())]
            + [f'- {p}' for p in sorted(have.keys() - want.keys())]
            + [f'~ {p}' for p in sorted(want.keys() & have.keys()) if want[p] != have[p]])


def publish(dest: Path = PUBLISHED, catalog_files: Path = CATALOG_FILES) -> list[str]:
    """Rewrite `dest` with the current skills, keeping the files NVSkills CI wrote."""
    changes = stale(dest, catalog_files)
    if not changes:
        return []
    kept = signed_cards(dest)
    with tempfile.TemporaryDirectory() as tmp:
        root = build(Path(tmp), catalog_files)
        want = files(root, kept)
        have = files(dest, kept)
        # Rewrite only what differs once the build stamp is ignored: an unchanged skill keeps
        # its bytes, so NVSkills CI does not re-evaluate and re-sign it for a new stamp alone.
        for rel in sorted(have.keys() - want.keys()):
            (dest / rel).unlink()
        if dest.is_dir():
            for path in sorted(dest.rglob('*'), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
        for rel in sorted(want):
            if have.get(rel) != want[rel]:
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, target)
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
    todo = not_ready()
    if todo:
        print(f'not publishable yet: the catalog would drop every skill until '
              f'{CATALOG_FILES.relative_to(REPO)}/ supplies {todo}', file=sys.stderr)


if __name__ == '__main__':
    main()
