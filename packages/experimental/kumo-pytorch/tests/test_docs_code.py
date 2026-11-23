# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The README and guide pages run as published, and their links resolve.

Every ```` ```python ```` block on a page runs in page order, in one namespace,
in a fresh interpreter whose working directory is a temporary directory. HTML
comments on the lines just before a fence change that:

- ``<!-- test: skip (reason) -->``: the block is a fragment and does not run.
- ``<!-- test: diskgraph -->``: the block needs DiskGraph; without it the block
  is skipped and the page reports a skip (a failure under ``--require-imports``).
- ``<!-- test: fresh-process -->``: the block starts a new interpreter and
  namespace, as a reader would in a separate process.
- ``<!-- test: file NAME -->``: the block is not run; it is written to ``NAME`` in
  the working directory before any block runs, so later blocks can import it.
- ``<!-- test: output -->`` before a ```` ```text ```` block: that block must
  equal what the preceding python block printed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CODE_PAGES = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
NOT_IN_REPOSITORY = (
    "docs/internal/",
    "demos/",
    "benchmarks/",
)
MARKER = re.compile(r"^<!--\s*test:\s*([a-z-]+)(.*?)-->\s*$")
FENCE = re.compile(r"^```(\w*)\s*$")
MARKERS = {"skip", "diskgraph", "fresh-process", "output", "file"}


@dataclass
class Block:
    language: str
    code: str
    line: int
    markers: dict[str, str] = field(default_factory=dict)


def _blocks(path: Path) -> list[Block]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    pending: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = MARKER.match(line)
        if marker:
            name = marker.group(1)
            assert name in MARKERS, f"{path.name}:{index + 1}: unknown test marker {name!r}"
            pending[name] = marker.group(2).strip()
            index += 1
            continue
        fence = FENCE.match(line)
        if fence:
            end = index + 1
            while not lines[end].startswith("```"):
                end += 1
            code = "\n".join(lines[index + 1 : end]) + "\n"
            blocks.append(Block(fence.group(1), code, index + 2, pending))
            pending = {}
            index = end + 1
            continue
        if line.strip():
            assert not pending, f"{path.name}:{index + 1}: a test marker must precede a fence"
        index += 1
    return blocks


RUNNER = """
import contextlib, io, json, sys
blocks = json.load(sys.stdin)
namespace = {"__name__": "__main__"}
outputs = []
for name, line, code in blocks:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile("\\n" * (line - 1) + code, name, "exec"), namespace)
    outputs.append(buffer.getvalue())
print(json.dumps(outputs))
"""


def _run(path: Path, segment: list[Block], cwd: Path) -> list[str]:
    payload = [(str(path.relative_to(ROOT)), block.line, block.code) for block in segment]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", RUNNER],
        input=json.dumps(payload),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"{path.name} failed:\n{result.stderr}"
    return json.loads(result.stdout.splitlines()[-1])


def _segments(blocks: list[Block], diskgraph: bool) -> tuple[list[list[Block]], list[Block]]:
    segments: list[list[Block]] = [[]]
    skipped: list[Block] = []
    for block in blocks:
        if block.language != "python" or {"skip", "file"} & block.markers.keys():
            continue
        if "diskgraph" in block.markers and not diskgraph:
            skipped.append(block)
            continue
        if "fresh-process" in block.markers and segments[-1]:
            segments.append([])
        segments[-1].append(block)
    return [segment for segment in segments if segment], skipped


@pytest.mark.parametrize("path", CODE_PAGES, ids=lambda path: str(path.relative_to(ROOT)))
def test_page_code_runs_in_order(path: Path, tmp_path: Path) -> None:
    blocks = _blocks(path)
    for block in blocks:
        if "skip" in block.markers:
            assert block.markers["skip"], f"{path.name}:{block.line}: say why a block is skipped"
    for block in blocks:
        if "file" in block.markers:
            (tmp_path / block.markers["file"]).write_text(block.code, encoding="utf-8")
    diskgraph = importlib.util.find_spec("diskgraph") is not None
    segments, skipped = _segments(blocks, diskgraph)
    printed: dict[int, str] = {}
    for segment in segments:
        for block, output in zip(segment, _run(path, segment, tmp_path), strict=True):
            printed[block.line] = output
    previous: Block | None = None
    for block in blocks:
        if "output" in block.markers:
            assert previous is not None, f"{path.name}:{block.line}: output follows no python block"
            assert "skip" not in previous.markers, (
                f"{path.name}:{block.line}: output follows a skipped block, so it is never checked"
            )
            if previous.line in printed:
                assert printed[previous.line].rstrip() == block.code.rstrip(), (
                    f"{path.name}:{block.line}: printed output differs from the page"
                )
        elif block.language == "python" and "file" not in block.markers:
            previous = block
    if skipped:
        pytest.importorskip("diskgraph")


def _link_pages() -> list[Path]:
    pages = [ROOT / "README.md", ROOT / "examples" / "README.md"]
    pages += sorted((ROOT / "examples").glob("*/README.md"))
    pages += sorted(path for path in (ROOT / "docs").rglob("*.md") if "internal" not in path.parts)
    return [page for page in pages if page.exists()]


def _notebooks() -> list[Path]:
    return sorted((ROOT / "examples").glob("*/walkthrough.ipynb"))


def _anchor(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
    else:
        text = path.read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.MULTILINE):
        base = _anchor(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def _links(text: str) -> list[str]:
    text = re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"`[^`\n]*`", "", text)
    return re.findall(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text)


def _check_links(source: Path, text: str) -> list[str]:
    problems: list[str] = []
    for target in _links(text):
        where = f"{source.relative_to(ROOT)} -> {target}"
        if re.match(r"https?://(www\.)?github\.com/kumo-ai/", target):
            problems.append(f"{where}: link to a private repository")
            continue
        if re.match(r"[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (source.parent / file_part).resolve() if file_part else source
        relative = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else ""
        if any(relative.startswith(excluded) for excluded in NOT_IN_REPOSITORY):
            problems.append(f"{where}: links to a file that is not in this repository")
        elif not resolved.exists():
            problems.append(f"{where}: no such file")
        elif anchor and resolved.suffix in {".md", ".ipynb"} and anchor not in _anchors(resolved):
            problems.append(f"{where}: no such heading")
    return problems


def test_relative_links_resolve() -> None:
    problems: list[str] = []
    for page in _link_pages():
        problems += _check_links(page, page.read_text(encoding="utf-8"))
    for notebook in _notebooks():
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        text = "\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
        )
        problems += _check_links(notebook, text)
    assert not problems, "\n".join(problems)


def test_user_docs_use_the_package_name() -> None:
    pages = _link_pages() + _notebooks()
    offenders = [
        f"{page.relative_to(ROOT)}:{number}"
        for page in pages
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\bRDL\b|\bRdl[A-Z]", line)
    ]
    assert not offenders, "say 'kumo-pytorch' or 'this package': " + ", ".join(offenders)


PUBLIC_MODULES = {"kumo.pytorch.api", "kumo.pytorch.build", "kumo.pytorch.datasets"}
FROM_IMPORT = re.compile(r"^\s*from\s+(kumo.pytorch[\w.]*)\s+import\s+([\w, ]+)", re.MULTILINE)


@pytest.mark.parametrize("path", CODE_PAGES, ids=lambda path: str(path.relative_to(ROOT)))
def test_page_code_uses_only_public_modules(path: Path) -> None:
    code = "\n".join(block.code for block in _blocks(path) if block.language == "python")
    imported = set(re.findall(r"^\s*import\s+(kumo.pytorch\.[\w.]*)", code, re.MULTILINE))
    for module, names in FROM_IMPORT.findall(code):
        if module != "kumo.pytorch":
            imported.add(module)
            continue
        imported |= {f"kumo.pytorch.{name.strip()}" for name in names.split(",")}
    assert imported <= PUBLIC_MODULES, f"{path.name} imports {sorted(imported - PUBLIC_MODULES)}"
