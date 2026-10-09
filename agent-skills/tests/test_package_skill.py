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
