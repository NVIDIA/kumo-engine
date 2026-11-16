# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The skills assume Kumo is already present, and say nothing about how it got there.

This holds even when a sentence is a prohibition. An earlier version told the agent never to emit a
package-file path, and the agent asked the user for one anyway: the placeholder existed only inside
the rule that forbade it. So the words are checked by absence, in every file under `skills/`,
including the host files about Snowflake ML Jobs, where Kumo reaches the job through host-supplied
values (`KUMO_JOB_RUNTIME`, `KUMO_JOB_REQUIREMENTS`) that the skill only names.
"""
import re
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1] / 'skills'

# `pip` as a word or as the `pip_` prefix of a submission keyword; `.zip` / `.whl` as extensions,
# so `zipfile.ZipFile` in the retention script does not count.
SETUP_WORDS = re.compile(r'install|\bpip(?:\b|_)|wheel|\.whl\b|\.zip\b', re.IGNORECASE)


def skill_files() -> list[Path]:
    return sorted(p for p in SKILLS.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts)


def test_files_are_found() -> None:
    names = {p.name for p in skill_files()}
    assert {'SKILL.md', 'adapter.md', 'ml_job.md', 'retain_artifacts.py'} <= names


@pytest.mark.parametrize('path', skill_files(), ids=lambda p: str(p.relative_to(SKILLS)))
def test_no_setup_words(path: Path) -> None:
    hits = [f'{n}: {line.strip()}' for n, line in enumerate(path.read_text().splitlines(), 1)
            if SETUP_WORDS.search(line)]
    assert hits == []


@pytest.mark.parametrize('text', [
    'pip install kumo', 'Do not install anything', 'reinstalled', 'pip_requirements=[...]',
    'kumo_fat_bundle-20261010-cp312-cp312-linux_x86_64.whl', 'a Python wheel', 'bundle.zip',
])
def test_pattern_catches(text: str) -> None:
    assert SETUP_WORDS.search(text)


@pytest.mark.parametrize('text', [
    'with zipfile.ZipFile(handle) as bundle:', 'a data pipeline', 'subprocess.PIPE',
    'KUMO_JOB_REQUIREMENTS', 'runtime_environment=KUMO_JOB_RUNTIME',
])
def test_pattern_spares(text: str) -> None:
    assert not SETUP_WORDS.search(text)
