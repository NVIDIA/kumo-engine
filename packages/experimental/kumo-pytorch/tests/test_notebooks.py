# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The examples: one folder each, committed clean, and reproducing their committed files.

Each ``examples/<name>/`` holds a ``README.md``, a ``walkthrough.ipynb`` committed
without outputs, optional ``.py`` helpers the notebook imports, and the small JSON
files the notebook writes into its working directory (at least ``contract.json``).

Execution is marked ``notebook`` and deselected by default; run it with
``pytest -m notebook``. It runs each walkthrough from a fresh kernel in a copy of
its folder and requires the files it writes to equal the committed ones byte for
byte. Rewrite them with ``python -m tests.test_notebooks --write [name ...]``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NEEDS: dict[str, tuple[str, ...]] = {
    "churn": ("diskgraph",),
    "hm-recommendation": ("torch_geometric",),
    "pragma-fraud": ("diskgraph",),
}
NAMES = sorted(NEEDS)
NOTEBOOK = "walkthrough.ipynb"
SOURCES = ("README.md", NOTEBOOK)
IGNORED = ("__pycache__", ".ipynb_checkpoints")
MAX_FILE_BYTES = 64 * 1024
SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.\n"
    "# All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)
PRIVATE = re.compile(
    r"/Users/|/home/|/private/|/tmp/|/var/folders/|C:\\\\|s3://|gs://|\.internal\b|kumo-ai/|"
    r"\bRDL\b|\bharness/|AGENTS\.md|docs/internal|\bdemos/|\bbenchmarks/|RdlModel|RdlBatch"
)


def _is_source(path: Path) -> bool:
    return path.name in SOURCES or path.suffix == ".py"


def _generated(folder: Path) -> dict[str, Path]:
    """The files in ``folder`` the notebook writes: everything but its sources."""

    return {
        path.name: path
        for path in sorted(folder.iterdir())
        if path.is_file() and not _is_source(path) and path.name not in IGNORED
    }


def _copy_sources(folder: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.is_file() and _is_source(path):
            shutil.copy2(path, into / path.name)


def _execute(folder: Path) -> str:
    """Run ``folder``'s walkthrough from a fresh kernel with ``folder`` as its working
    directory; return what its last cell printed. The notebook file is not written."""

    import nbclient
    import nbformat

    notebook = nbformat.read(folder / NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(folder)}},
    )
    client.execute()
    return "".join(
        output.get("text", "")
        for output in notebook.cells[-1].get("outputs", [])
        if output.output_type == "stream"
    )


def _drift(name: str, produced: dict[str, Path], committed: dict[str, Path]) -> list[str]:
    problems: list[str] = []
    for missing in sorted(committed.keys() - produced.keys()):
        problems.append(f"{missing} is committed but the notebook no longer writes it")
    for extra in sorted(produced.keys() - committed.keys()):
        problems.append(f"the notebook writes {extra}, which is not committed")
    for same in sorted(produced.keys() & committed.keys()):
        new, old = produced[same].read_bytes(), committed[same].read_bytes()
        if new == old:
            continue
        diff = difflib.unified_diff(
            old.decode().splitlines(),
            new.decode().splitlines(),
            f"committed/{same}",
            f"regenerated/{same}",
            lineterm="",
        )
        shown = list(diff)[:40]
        problems.append(f"{same} differs from what the notebook writes:\n" + "\n".join(shown))
    if problems:
        problems.append(
            f"regenerate with: python -m tests.test_notebooks --write {name}  "
            f"(then review and commit examples/{name}/)"
        )
    return problems


def test_every_example_is_listed() -> None:
    folders = sorted(
        path.name
        for path in EXAMPLES.iterdir()
        if path.is_dir() and path.name not in IGNORED and not path.name.startswith(".")
    )
    assert folders == NAMES, "every examples/ folder needs an entry in NEEDS, and vice versa"
    assert sorted(path.name for path in EXAMPLES.glob("*.ipynb")) == [], (
        "notebooks live in an example folder as walkthrough.ipynb"
    )
    index = (EXAMPLES / "README.md").read_text()
    for name in NAMES:
        assert f"({name}/)" in index, f"examples/README.md does not link {name}/"


@pytest.mark.parametrize("name", NAMES)
def test_example_folder_layout(name: str) -> None:
    folder = EXAMPLES / name
    for required in (*SOURCES, "contract.json"):
        assert (folder / required).is_file(), f"examples/{name}/ has no {required}"
    for helper in sorted(folder.glob("*.py")):
        assert helper.read_text().startswith(SPDX_HEADER), (
            f"examples/{name}/{helper.name} must start with the SPDX header"
        )
    for generated, path in _generated(folder).items():
        where = f"examples/{name}/{generated}"
        assert path.suffix == ".json", f"{where}: generated files are JSON; data stays in a tempdir"
        assert path.stat().st_size <= MAX_FILE_BYTES, f"{where} is over {MAX_FILE_BYTES} bytes"
        text = path.read_text()
        json.loads(text)
        found = PRIVATE.search(text)
        assert found is None, f"{where} mentions {found.group()!r}"
    readme = (folder / "README.md").read_text()
    for listed in [NOTEBOOK, *_generated(folder), *(p.name for p in folder.glob("*.py"))]:
        assert f"`{listed}`" in readme or f"({listed})" in readme, (
            f"examples/{name}/README.md does not describe {listed}"
        )


@pytest.mark.parametrize("name", NAMES)
def test_notebook_is_clean(name: str) -> None:
    folder = EXAMPLES / name
    notebook = json.loads((folder / NOTEBOOK).read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"].get("kernelspec", {}).get("name") == "python3"
    for index, cell in enumerate(notebook["cells"]):
        where = f"examples/{name}/{NOTEBOOK} cell {index}"
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"{where} has outputs; clear them before committing"
            assert cell["execution_count"] is None, f"{where} has an execution count"
        found = PRIVATE.search("".join(cell["source"]))
        assert found is None, f"{where} mentions {found.group()!r}"
    for path in [folder / "README.md", *sorted(folder.glob("*.py"))]:
        found = PRIVATE.search(path.read_text())
        assert found is None, f"examples/{name}/{path.name} mentions {found.group()!r}"


@pytest.mark.notebook
@pytest.mark.parametrize("name", NAMES)
def test_notebook_reproduces_its_files(name: str, tmp_path: Path) -> None:
    for module in NEEDS[name]:
        pytest.importorskip(module)
    pytest.importorskip("nbclient")
    pytest.importorskip("nbformat")
    folder = EXAMPLES / name
    copy = tmp_path / name
    _copy_sources(folder, copy)
    assert "checks passed" in _execute(copy)
    problems = _drift(name, _generated(copy), _generated(folder))
    if problems:
        pytest.fail(f"examples/{name}/: " + "\n".join(problems), pytrace=False)


def write(names: list[str]) -> None:
    """Run each walkthrough and replace its folder's generated files with what it wrote.

    The run happens in a copy of the folder, so a failing notebook changes nothing and
    the committed ``walkthrough.ipynb`` stays stripped.
    """

    for name in names:
        folder = EXAMPLES / name
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch) / name
            _copy_sources(folder, copy)
            printed = _execute(copy)
            if "checks passed" not in printed:
                raise SystemExit(f"examples/{name}: the check cell did not pass:\n{printed}")
            produced = _generated(copy)
            for stale in _generated(folder).keys() - produced.keys():
                (folder / stale).unlink()
            for generated, path in produced.items():
                shutil.copyfile(path, folder / generated)
        print(f"examples/{name}: wrote {', '.join(sorted(produced))}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m tests.test_notebooks")
    parser.add_argument(
        "--write",
        action="store_true",
        required=True,
        help="execute the walkthroughs and rewrite their generated files in place",
    )
    parser.add_argument("names", nargs="*", metavar="name", help=f"any of {', '.join(NAMES)}")
    arguments = parser.parse_args(argv)
    unknown = sorted(set(arguments.names) - set(NAMES))
    if unknown:
        parser.error(f"no such example: {', '.join(unknown)}")
    write(arguments.names or NAMES)


if __name__ == "__main__":
    main(sys.argv[1:])
