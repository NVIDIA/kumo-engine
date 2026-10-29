# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'package_skill.py'
spec = importlib.util.spec_from_file_location('package_skill', SCRIPT)
assert spec and spec.loader
package_skill = importlib.util.module_from_spec(spec)
sys.modules['package_skill'] = package_skill
spec.loader.exec_module(package_skill)


@pytest.mark.parametrize('host', sorted(package_skill.HOSTS))
def test_every_reference_resolves(tmp_path: Path, host: str) -> None:
    pkg = package_skill.build(host, tmp_path)
    assert (pkg / 'SKILL.md').is_file()
    assert package_skill.unresolved(pkg) == []


def test_coco_nests_everything_inside_the_router(tmp_path: Path) -> None:
    pkg = package_skill.build('coco', tmp_path)
    assert pkg == tmp_path / 'coco' / 'kumo-ds' / 'skills' / 'kumo-data-science'
    assert [p.name for p in pkg.parent.iterdir()] == ['kumo-data-science']
    for name in ['kumoai-reference', *package_skill.CAPABILITY]:
        assert (pkg / name / 'SKILL.md').is_file()
    assert (pkg / 'hosts' / 'snowflake-coco' / 'adapter.md').is_file()


def test_build_stamps_the_marker(tmp_path: Path) -> None:
    source = (package_skill.SKILLS / 'kumo-data-science' / 'SKILL.md').read_text()
    assert '**`SKILL-BUILD: source`**' in source
    built = (package_skill.build('coco', tmp_path) / 'SKILL.md').read_text()
    marker = package_skill.MARKER.findall(built)
    assert marker == [f'SKILL-BUILD: {package_skill.build_id()}']
    assert re.fullmatch(r'SKILL-BUILD: (\d{4}-\d{2}-\d{2}-[0-9a-f]+(-dirty)?|unknown)', marker[0])


def test_broken_references_are_reported(tmp_path: Path) -> None:
    pkg = tmp_path / 'pkg'
    (pkg / 'accept').mkdir(parents=True)
    (pkg / 'accept' / 'real.md').write_text('')
    (pkg / 'SKILL.md').write_text('\n'.join([
        '[ok](accept/real.md) [gone](accept/missing.md) [web](https://example.com/x.md)',
        '`accept/real.md` `kumo-data-science/accept/real.md` `accept/nope.md`',
        '`kumo/engine/trainer.py` is a source path, not a package path',
    ]))
    assert package_skill.unresolved(pkg) == [
        'SKILL.md: link (accept/missing.md) does not resolve',
        'SKILL.md: `accept/nope.md` does not resolve from the package root',
    ]


def test_generic_ships_only_its_own_host(tmp_path: Path) -> None:
    pkg = package_skill.build('generic', tmp_path)
    assert pkg == tmp_path / 'generic' / 'skills' / 'kumo-data-science'
    assert [p.name for p in (pkg / 'hosts').iterdir()] == ['generic']
    leaks = [str(p.relative_to(pkg)) for p in pkg.rglob('*.md')
             if 'snowflake-coco' in p.read_text() or 'ml_job.md' in p.read_text()]
    assert leaks == []


def test_core_names_no_host() -> None:
    """The portable core reaches its host as `hosts/<host>/`, never by a host's own name."""
    hosts = sorted(p.name for p in (package_skill.SKILLS / 'hosts').iterdir() if p.is_dir())
    named = re.compile(r'hosts/(?:' + '|'.join(map(re.escape, hosts)) + r')\b')
    hits = [f'{p.relative_to(package_skill.SKILLS)}: {m.group(0)}'
            for p in sorted(package_skill.SKILLS.rglob('*.md'))
            if p.relative_to(package_skill.SKILLS).parts[0] != 'hosts'
            for m in named.finditer(p.read_text())]
    assert hits == []


@pytest.mark.parametrize('host', sorted(package_skill.HOSTS))
def test_adapter_covers_what_the_core_defers(host: str) -> None:
    """Preflight §0, the intake contract and batch inference defer these to the adapter."""
    adapter = next(rel for rel in package_skill.HOSTS[host]['hosts'] if rel.endswith('adapter.md'))
    text = (package_skill.SKILLS / 'hosts' / adapter).read_text()
    for section in ['Source scope', 'Access', 'Destinations']:
        assert section in text, f'{adapter} does not cover {section}'
