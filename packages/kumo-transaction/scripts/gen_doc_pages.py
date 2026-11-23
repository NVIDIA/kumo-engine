# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lay out the repository's documentation for MkDocs (`mkdocs-gen-files`).

Pages keep their repository paths on the site (`docs/concepts/x.md` is served
at `docs/concepts/x/`), so every relative link that works on GitHub also
works on the site, and `mkdocs build --strict` fails on one that does not.
Linked non-Markdown files (configs, scripts, JSON results) are copied as
static files for the same reason. `docs/_site/` holds the site-only pages
(the API reference) and is MkDocs' own `docs_dir`, so it is not copied.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
TREES = ("docs", "recipes", "notebooks", "benchmarks", "configs", "scripts", "src")
ROOT_PAGES = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "SYNC.md",
    "UPSTREAM.json",
)
SUFFIXES = {".md", ".ipynb", ".yaml", ".yml", ".py", ".json", ".toml", ".sh", ".png", ".svg"}


def _repository_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return sorted(set(listed))


def _published(path: str) -> bool:
    if path in ROOT_PAGES:
        return True
    parts = Path(path).parts
    if len(parts) < 2 or parts[0] not in TREES or path.startswith("docs/_site/"):
        return False
    return Path(path).suffix in SUFFIXES and (ROOT / path).is_file()


for path in filter(_published, _repository_files()):
    with mkdocs_gen_files.open(path, "wb") as out:
        out.write((ROOT / path).read_bytes())
