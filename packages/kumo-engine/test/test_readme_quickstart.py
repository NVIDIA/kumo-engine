# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The README's quickstart runs as published, against the installed package."""

import re
from pathlib import Path

from kumo.engine import datasets

README = Path(__file__).parents[1] / 'README.md'


def _quickstart_code() -> str:
    text = README.read_text(encoding='utf-8')
    section = re.search(r'^## Quickstart\n(.*?)(?=^## |\Z)', text,
                        re.MULTILINE | re.DOTALL)
    assert section, f'{README} has no "## Quickstart" section'
    blocks = re.findall(r'^```python\n(.*?)^```$', section.group(1),
                        re.MULTILINE | re.DOTALL)
    assert blocks, f'the quickstart in {README} has no python code block'
    return '\n'.join(blocks)


def test_readme_quickstart(tmp_path, monkeypatch):
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    shop = Path(datasets.shop())
    shipped = sorted(shop.iterdir())

    namespace = {}
    exec(compile(_quickstart_code(), str(README), 'exec'), namespace)

    frame = namespace['result'].frame
    assert len(frame) == 1000
    assert list(frame) == ['ENTITY', 'TARGET_PRED', 'False_PROB', 'True_PROB']
    # Reads the package data in place; writes only to the scratch directory.
    assert sorted(shop.iterdir()) == shipped
    assert not any(cwd.iterdir())
