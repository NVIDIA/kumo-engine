# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a pull request's changes to kumo-pytorch's conformance spec.

Two rules, read from the diff between the head and its merge base with the base
branch (see packages/experimental/kumo-pytorch/harness/rules.md, V3):

- conformance/expectations.json only moves forward: an existing refusal may
  become "answer". Any other change needs a guarantee.md change in the same PR
  that explains it.
- Every golden file of a guarantee the PR touches is linked from that
  guarantee's "Golden examples" list.

`evaluate()` is pure; `gather()` reads its inputs from git.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE = "packages/experimental/kumo-pytorch/"
EXPECTATIONS_PATH = "conformance/expectations.json"
GUARANTEES = "conformance/spec/guarantees/"
GUARANTEE_DIRECTORY = re.compile(r"^conformance/spec/guarantees/([^/]+)/")
GOLDEN = re.compile(r"^conformance/spec/guarantees/([^/]+)/golden/([^/]+)\.json$")
GUARANTEE = re.compile(r"^conformance/spec/guarantees/([^/]+)/guarantee\.md$")
EXAMPLES_HEADING = "## Golden examples"
EXAMPLE_LINK = re.compile(r"^- \[(C-[A-Z0-9-]+)\]\(golden/\1\.json\)")


def _load_parse_expectations() -> Callable[[str], dict]:
    # By path: the package's conformance code needs no installed environment.
    path = REPOSITORY / PACKAGE / "conformance/expectations.py"
    spec = importlib.util.spec_from_file_location("kumo_pytorch_expectations", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_expectations


parse_expectations = _load_parse_expectations()


# --- expectations ------------------------------------------------------------------------


def only_promotions(before: dict, after: dict) -> bool:
    if before.keys() != after.keys():
        return False
    for case_id, decisions in before.items():
        if decisions.keys() != after[case_id].keys():
            return False
        for engine, previous in decisions.items():
            current = after[case_id][engine]
            if current != previous and not (isinstance(previous, dict) and current == "answer"):
                return False
    return True


def only_additions(before: dict, after: dict) -> bool:
    """New cases only: every existing entry stays, or a refusal is promoted to an answer."""
    if not before.keys() <= after.keys():
        return False
    return only_promotions(before, {case_id: after[case_id] for case_id in before})


def expectations_problem(paths: list[str], before: str | None, after: str | None) -> str | None:
    """Expectations only move forward unless a changed guarantee explains the change."""
    if after is None:
        return f"the PR must keep {EXPECTATIONS_PATH}"
    try:
        proposed = parse_expectations(after)
    except ValueError as error:
        return f"invalid {EXPECTATIONS_PATH}: {error}"
    try:
        previous = parse_expectations(before) if before is not None else {}
    except ValueError:
        previous = {}
    if proposed == previous or (previous and only_promotions(previous, proposed)):
        return None
    if any(GUARANTEE.match(path) for path in paths):
        return None
    if previous and only_additions(previous, proposed):
        return (
            f"{EXPECTATIONS_PATH} adds cases: add their golden files to a guarantee and link "
            "them from its guarantee.md in the same PR"
        )
    return (
        f"{EXPECTATIONS_PATH} changes beyond refusal-to-answer promotions must update "
        "a guarantee.md that explains the decision"
    )


# --- golden examples ---------------------------------------------------------------------


def linked_examples(guarantee: str) -> set[str]:
    """Case IDs linked from the list under the guarantee's Golden examples heading."""
    lines = guarantee.splitlines()
    if EXAMPLES_HEADING not in lines:
        return set()
    heading = lines.index(EXAMPLES_HEADING)
    cases: set[str] = set()
    for line in lines[heading + 1:]:
        if line.startswith("#"):
            break
        link = EXAMPLE_LINK.match(line)
        if link:
            cases.add(link[1])
    return cases


def golden_link_problems(
    paths: list[str],
    at_head: Callable[[str], str | None],
    files_at_head: Callable[[str], list[str]],
) -> list[str]:
    """Each touched guarantee links every golden file in its directory."""
    problems = []
    touched = sorted({match[1] for path in paths if (match := GUARANTEE_DIRECTORY.match(path))})
    for guarantee_id in touched:
        directory = f"{GUARANTEES}{guarantee_id}/"
        golden = sorted(
            match[2] for path in files_at_head(directory) if (match := GOLDEN.match(path))
        )
        if not golden:
            continue
        document = at_head(f"{directory}guarantee.md")
        if document is None:
            problems.append(f"{directory}golden has cases but {guarantee_id} has no guarantee.md")
            continue
        missing = [case for case in golden if case not in linked_examples(document)]
        if missing:
            problems.append(
                f"{directory}guarantee.md must link {', '.join(missing)} under "
                f"'{EXAMPLES_HEADING}' as '- [C-...](golden/C-....json): ...'"
            )
    return problems


# --- the decision ------------------------------------------------------------------------


@dataclass
class Change:
    paths: list[str]  # package-relative, changed since the merge base; renames count twice
    expectations_before: str | None
    expectations_after: str | None
    at_head: Callable[[str], str | None] = lambda path: None
    files_at_head: Callable[[str], list[str]] = field(default=lambda directory: [])


def evaluate(change: Change) -> list[str]:
    problems: list[str] = []
    if EXPECTATIONS_PATH in change.paths:
        problem = expectations_problem(
            change.paths, change.expectations_before, change.expectations_after
        )
        if problem:
            problems.append(problem)
    problems += golden_link_problems(change.paths, change.at_head, change.files_at_head)
    return problems


# --- gathering inputs from git -----------------------------------------------------------


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPOSITORY, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _read_at(ref: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{PACKAGE}{path}"], cwd=REPOSITORY, capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def _package_paths(output: str) -> list[str]:
    return [path[len(PACKAGE):] for path in output.split("\0") if path.startswith(PACKAGE)]


def changed(old: str, new: str) -> list[str]:
    # Disabling rename detection reports both the removed and added paths.
    return _package_paths(_git("diff", "--name-only", "--no-renames", "-z", old, new, "--"))


def gather(base: str, head: str) -> Change:
    merge_base = _git("merge-base", base, head).strip()
    return Change(
        paths=changed(merge_base, head),
        expectations_before=_read_at(merge_base, EXPECTATIONS_PATH),
        expectations_after=_read_at(head, EXPECTATIONS_PATH),
        at_head=lambda path: _read_at(head, path),
        files_at_head=lambda directory: _package_paths(
            _git("ls-tree", "-r", "-z", "--name-only", head, "--", f"{PACKAGE}{directory}")
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="the base branch (default: %(default)s)")
    parser.add_argument("--head", default="HEAD", help="the PR's head commit (default: %(default)s)")
    args = parser.parse_args()
    problems = evaluate(gather(args.base, args.head))
    lines = [f"- {problem}" for problem in problems] or ["- all rules pass"]
    report = "\n".join(["### kumo-pytorch content policy", *lines])
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report + "\n")
    if os.environ.get("GITHUB_ACTIONS"):
        for problem in problems:
            print(f"::error title=kumo-pytorch policy::{problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
