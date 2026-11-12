# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'publish_nvskills.py'
spec = importlib.util.spec_from_file_location('publish_nvskills', SCRIPT)
assert spec and spec.loader
publish_nvskills = importlib.util.module_from_spec(spec)
sys.modules['publish_nvskills'] = publish_nvskills
spec.loader.exec_module(publish_nvskills)


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    root = tmp_path / 'catalog'
    (root / 'evals').mkdir(parents=True)
    (root / 'evals' / 'evals.json').write_text('{"skill_name": "kumo-data-science"}\n')
    (root / 'skill-card.md').write_text('## Description\n')
    return root


def test_publish_writes_the_generic_package_and_catalog_files(tmp_path: Path,
                                                              catalog: Path) -> None:
    dest = tmp_path / 'skills' / 'kumo-data-science'
    assert publish_nvskills.publish(dest, catalog)
    assert (dest / 'SKILL.md').is_file()
    assert (dest / 'hosts' / 'generic' / 'adapter.md').is_file()
    assert not (dest / 'hosts' / 'snowflake-coco').exists()
    assert (dest / 'evals' / 'evals.json').is_file()
    assert (dest / 'skill-card.md').is_file()
    assert publish_nvskills.stale(dest, catalog) == []


def test_publish_keeps_signed_files_and_drops_files_no_longer_shipped(tmp_path: Path,
                                                                      catalog: Path) -> None:
    dest = tmp_path / 'kumo-data-science'
    publish_nvskills.publish(dest, catalog)
    for name in publish_nvskills.SIGNED:
        (dest / name).write_text('from NVSkills CI\n')
    (dest / 'old' / 'removed.md').parent.mkdir()
    (dest / 'old' / 'removed.md').write_text('gone\n')

    assert publish_nvskills.stale(dest, catalog) == ['- old/removed.md']
    publish_nvskills.publish(dest, catalog)
    assert not (dest / 'old').exists()
    for name in publish_nvskills.SIGNED:
        assert (dest / name).read_text() == 'from NVSkills CI\n'


def test_check_ignores_the_build_stamp_but_not_content(tmp_path: Path, catalog: Path) -> None:
    dest = tmp_path / 'kumo-data-science'
    publish_nvskills.publish(dest, catalog)
    router = dest / 'SKILL.md'
    text = router.read_text(encoding='utf-8')
    router.write_text(publish_nvskills.package_skill.MARKER.sub('SKILL-BUILD: 2000-01-01-0', text),
                      encoding='utf-8')
    assert publish_nvskills.stale(dest, catalog) == []

    router.write_text(text + '\nedited by hand\n', encoding='utf-8')
    assert publish_nvskills.stale(dest, catalog) == ['~ SKILL.md']


@pytest.mark.parametrize('rel', ['skill.oms.sig', 'evals/skill.oms.sig', 'evals/BENCHMARK.md'])
def test_catalog_files_cannot_supply_signed_files(tmp_path: Path, catalog: Path,
                                                  rel: str) -> None:
    (catalog / rel).write_text('forged\n')
    with pytest.raises(SystemExit, match='written by NVSkills CI'):
        publish_nvskills.publish(tmp_path / 'kumo-data-science', catalog)


@pytest.mark.parametrize('rel', ['SKILL.md', 'hosts/generic/adapter.md'])
def test_catalog_files_cannot_replace_package_files(tmp_path: Path, catalog: Path,
                                                    rel: str) -> None:
    (catalog / rel).parent.mkdir(parents=True, exist_ok=True)
    (catalog / rel).write_text('replaced\n')
    with pytest.raises(SystemExit, match='already in the generic package'):
        publish_nvskills.publish(tmp_path / 'kumo-data-science', catalog)


def test_the_package_cannot_ship_signed_files(tmp_path: Path, catalog: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    build = publish_nvskills.package_skill.build

    def with_signature(host: str, dist: Path) -> Path:
        pkg = build(host, dist)
        (pkg / 'kumoai-reference' / 'BENCHMARK.md').write_text('forged\n')
        return pkg

    monkeypatch.setattr(publish_nvskills.package_skill, 'build', with_signature)
    dest = tmp_path / 'kumo-data-science'
    with pytest.raises(SystemExit, match='written by NVSkills CI'):
        publish_nvskills.publish(dest, catalog)
    assert not dest.exists()


def test_the_published_copy_matches_the_source() -> None:
    """The copy NVIDIA/skills syncs from main is rebuilt in the same PR as the change.

    A PR that edits agent-skills/ without refreshing skills/kumo-data-science/ would merge a
    published copy that no longer matches its source, so it fails here instead.
    """
    changes = publish_nvskills.stale()
    assert changes == [], (
        'skills/kumo-data-science/ is out of date; run '
        f'python3 agent-skills/scripts/publish_nvskills.py and commit the result: {changes}'
    )
