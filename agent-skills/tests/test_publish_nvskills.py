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


# The router and every skill the generic package nests inside it, each published beside it.
ROUTER = publish_nvskills.ROUTER
SKILLS = [ROUTER, 'kumoai-reference', *publish_nvskills.package_skill.CAPABILITY]


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    root = tmp_path / 'catalog'
    for skill in SKILLS:
        (root / skill / 'evals').mkdir(parents=True, exist_ok=True)
        (root / skill / 'evals' / 'evals.json').write_text(f'{{"skill_name": "{skill}"}}\n')
        (root / skill / 'skill-card.md').write_text('## Description\n')
    return root


def test_publish_writes_the_generic_package_and_catalog_files(tmp_path: Path,
                                                              catalog: Path) -> None:
    dest = tmp_path / 'skills'
    assert publish_nvskills.publish(dest, catalog)
    assert (dest / ROUTER / 'SKILL.md').is_file()
    assert (dest / ROUTER / 'hosts' / 'generic' / 'adapter.md').is_file()
    assert not (dest / ROUTER / 'hosts' / 'snowflake-coco').exists()
    assert sorted(publish_nvskills.skill_dirs(dest)) == sorted(SKILLS)
    for skill in SKILLS:
        assert (dest / skill / 'evals' / 'evals.json').is_file()
        assert (dest / skill / 'skill-card.md').is_file()
    assert publish_nvskills.missing(dest) == []
    assert publish_nvskills.stale(dest, catalog) == []


def test_no_skill_is_published_inside_another(tmp_path: Path, catalog: Path) -> None:
    """The catalog does not allow a skill inside a skill: each SKILL.md has its own directory."""
    dest = tmp_path / 'skills'
    publish_nvskills.publish(dest, catalog)
    assert publish_nvskills.nested(dest) == []
    assert not (dest / ROUTER / 'kumoai-reference').exists()
    assert (dest / 'kumoai-reference' / 'references' / 'kumo_finetuning_sdk.md').is_file()
    assert publish_nvskills.unresolved(dest) == []


def test_the_package_cannot_nest_a_skill_two_levels_down(tmp_path: Path, catalog: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    build = publish_nvskills.package_skill.build

    def deeper(host: str, dist: Path) -> Path:
        pkg = build(host, dist)
        (pkg / 'kumo-model-tuning' / 'extra').mkdir()
        (pkg / 'kumo-model-tuning' / 'extra' / 'SKILL.md').write_text('---\nname: extra\n---\n')
        return pkg

    monkeypatch.setattr(publish_nvskills.package_skill, 'build', deeper)
    with pytest.raises(SystemExit, match='nests skills below its top level'):
        publish_nvskills.publish(tmp_path / 'skills', catalog)


@pytest.mark.parametrize('text, error', [
    ('read `accept/preflight.md`', "is another skill's file"),
    ('read `kumoai-reference/references/absent.md`', 'does not resolve'),
    ('read `references/absent.md`', 'does not resolve'),
    ('see [plans](../kumo-model-tuning/references/absent.md)', 'does not resolve'),
])
def test_references_are_checked_one_directory_per_skill(tmp_path: Path, catalog: Path,
                                                        text: str, error: str) -> None:
    """A path a nested package resolves can miss once the skills sit side by side."""
    dest = tmp_path / 'skills'
    publish_nvskills.publish(dest, catalog)
    doc = dest / 'kumo-model-tuning' / 'references' / 'plans.md'
    doc.write_text(doc.read_text() + f'\n{text}\n')
    errors = publish_nvskills.unresolved(dest)
    assert len(errors) == 1 and error in errors[0], errors


@pytest.mark.parametrize('rel', ['kumoai-reference/skill-card.md',
                                 'kumo-model-tuning/evals/evals.json',
                                 'kumo-data-science/evals/evals.json'])
def test_every_skill_directory_needs_its_own_card_and_evals(tmp_path: Path, catalog: Path,
                                                            rel: str) -> None:
    """The catalog sync deletes a SKILL.md directory without its own card and evals.

    That also breaks the other skills' links into it, so the build refuses.
    """
    (catalog / rel).unlink()
    dest = tmp_path / 'skills'
    with pytest.raises(SystemExit, match='the catalog would drop these skills'):
        publish_nvskills.publish(dest, catalog)
    assert not dest.exists()


def test_without_catalog_files_every_input_is_reported_missing(tmp_path: Path) -> None:
    absent = tmp_path / 'no-catalog'
    dest = tmp_path / 'skills'
    assert publish_nvskills.publish(dest, absent)
    todo = publish_nvskills.not_ready(absent)
    for skill in SKILLS:
        assert f'{skill}/skill-card.md' in todo
        assert f'{skill}/evals/*.json' in todo
    assert len(todo) == 2 * len(SKILLS)


@pytest.mark.parametrize('rel', ['notes.md', 'skill-card.md', 'evals/evals.json',
                                 'kumo-data-science/SKILL.md',
                                 'kumo-data-science/hosts/generic/adapter.md',
                                 'kumo-data-science/hosts/skill-card.md',
                                 'kumo-data-science/hosts/evals/evals.json',
                                 'kumo-data-science/kumo-model-tuning/skill-card.md',
                                 'kumo-model-tunning/skill-card.md'])
def test_catalog_files_must_belong_to_a_skill(tmp_path: Path, catalog: Path, rel: str) -> None:
    (catalog / rel).parent.mkdir(parents=True, exist_ok=True)
    (catalog / rel).write_text('stray\n')
    with pytest.raises(SystemExit, match='is not the skill-card.md or evals/ of a skill'):
        publish_nvskills.publish(tmp_path / 'skills', catalog)


def test_publish_keeps_signed_files_and_drops_files_no_longer_shipped(tmp_path: Path,
                                                                      catalog: Path) -> None:
    dest = tmp_path / 'skills'
    publish_nvskills.publish(dest, catalog)
    signed = [dest / skill / name for skill in SKILLS for name in publish_nvskills.SIGNED]
    for path in signed:
        path.write_text('from NVSkills CI\n')
    # The nested layout this directory held before it was split into one directory per skill.
    (dest / ROUTER / 'kumoai-reference').mkdir()
    (dest / ROUTER / 'kumoai-reference' / 'SKILL.md').write_text('gone\n')

    assert publish_nvskills.stale(dest, catalog) == [f'- {ROUTER}/kumoai-reference/SKILL.md']
    publish_nvskills.publish(dest, catalog)
    assert not (dest / ROUTER / 'kumoai-reference').exists()
    for path in signed:
        assert path.read_text() == 'from NVSkills CI\n'


def test_a_signed_skill_keeps_the_card_nvskills_ci_wrote(tmp_path: Path,
                                                         catalog: Path) -> None:
    """NVSkills CI rewrites a skill's card when it signs; the source card only seeds it."""
    dest = tmp_path / 'skills'
    publish_nvskills.publish(dest, catalog)
    signed, unsigned = dest / 'kumo-model-tuning', dest / 'kumoai-reference'
    (signed / 'skill.oms.sig').write_text('from NVSkills CI\n')
    for skill in (signed, unsigned):
        (skill / 'skill-card.md').write_text('## Description\n\n## Evaluation Results\n')

    assert publish_nvskills.stale(dest, catalog) == ['~ kumoai-reference/skill-card.md']
    (dest / ROUTER / 'SKILL.md').write_text('edited by hand\n')
    publish_nvskills.publish(dest, catalog)
    assert (signed / 'skill-card.md').read_text() == '## Description\n\n## Evaluation Results\n'
    assert (unsigned / 'skill-card.md').read_text() == '## Description\n'
    assert publish_nvskills.stale(dest, catalog) == []


def test_check_ignores_the_build_stamp_but_not_content(tmp_path: Path, catalog: Path) -> None:
    dest = tmp_path / 'skills'
    publish_nvskills.publish(dest, catalog)
    router = dest / ROUTER / 'SKILL.md'
    text = router.read_text(encoding='utf-8')
    router.write_text(publish_nvskills.package_skill.MARKER.sub('SKILL-BUILD: 2000-01-01-0', text),
                      encoding='utf-8')
    assert publish_nvskills.stale(dest, catalog) == []

    router.write_text(text + '\nedited by hand\n', encoding='utf-8')
    assert publish_nvskills.stale(dest, catalog) == [f'~ {ROUTER}/SKILL.md']


def test_publish_leaves_unchanged_skills_and_their_stamp_alone(tmp_path: Path,
                                                               catalog: Path) -> None:
    """Refreshing one skill must not rewrite another, or NVSkills CI re-signs it for nothing."""
    dest = tmp_path / 'skills'
    publish_nvskills.publish(dest, catalog)
    router = dest / ROUTER / 'SKILL.md'
    old = publish_nvskills.package_skill.MARKER.sub('SKILL-BUILD: 2000-01-01-0',
                                                    router.read_text(encoding='utf-8'))
    router.write_text(old, encoding='utf-8')
    (dest / 'kumo-model-tuning' / 'SKILL.md').write_text('stale\n', encoding='utf-8')

    assert publish_nvskills.publish(dest, catalog) == ['~ kumo-model-tuning/SKILL.md']
    assert router.read_text(encoding='utf-8') == old
    assert publish_nvskills.stale(dest, catalog) == []


@pytest.mark.parametrize('rel', ['kumo-data-science/skill.oms.sig',
                                 'kumo-data-science/evals/skill.oms.sig',
                                 'kumo-data-science/evals/BENCHMARK.md',
                                 'kumoai-reference/skill.oms.sig',
                                 'kumo-model-tuning/BENCHMARK.md'])
def test_catalog_files_cannot_supply_signed_files(tmp_path: Path, catalog: Path,
                                                  rel: str) -> None:
    (catalog / rel).write_text('forged\n')
    with pytest.raises(SystemExit, match='written by NVSkills CI'):
        publish_nvskills.publish(tmp_path / 'skills', catalog)


@pytest.mark.parametrize('rel', ['skill-card.md', 'kumo-model-tuning/evals/evals.json'])
def test_catalog_files_cannot_replace_package_files(tmp_path: Path, catalog: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    rel: str) -> None:
    build = publish_nvskills.package_skill.build

    def shipping(host: str, dist: Path) -> Path:
        pkg = build(host, dist)
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text('from the package\n')
        return pkg

    monkeypatch.setattr(publish_nvskills.package_skill, 'build', shipping)
    with pytest.raises(SystemExit, match='already in the generic package'):
        publish_nvskills.publish(tmp_path / 'skills', catalog)


def test_the_package_cannot_ship_signed_files(tmp_path: Path, catalog: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    build = publish_nvskills.package_skill.build

    def with_signature(host: str, dist: Path) -> Path:
        pkg = build(host, dist)
        (pkg / 'kumoai-reference' / 'BENCHMARK.md').write_text('forged\n')
        return pkg

    monkeypatch.setattr(publish_nvskills.package_skill, 'build', with_signature)
    dest = tmp_path / 'skills'
    with pytest.raises(SystemExit, match='written by NVSkills CI'):
        publish_nvskills.publish(dest, catalog)
    assert not dest.exists()


def test_the_published_copy_matches_the_source() -> None:
    """The copy NVIDIA/skills syncs from main is rebuilt in the same PR as the change.

    A PR that edits agent-skills/ without refreshing skills/ would merge a published copy that
    no longer matches its source, so it fails here instead.
    """
    changes = publish_nvskills.stale()
    assert changes == [], (
        'skills/ is out of date; run '
        f'python3 agent-skills/scripts/publish_nvskills.py and commit the result: {changes}'
    )
