# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The content policy the kumo-pytorch-gate check runs on every pull request.

The script lives at .github/scripts/kumo-pytorch-policy.py, outside the package.
"""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from conformance.cases import REPOSITORY
from conformance.expectations import load_expectations

SCRIPT = REPOSITORY.parents[2] / ".github/scripts/kumo-pytorch-policy.py"
_spec = importlib.util.spec_from_file_location("kumo_pytorch_policy", SCRIPT)
policy = importlib.util.module_from_spec(_spec)
sys.modules["kumo_pytorch_policy"] = policy
_spec.loader.exec_module(policy)

EXPECTATIONS = json.dumps(load_expectations())
ROWS = "conformance/spec/guarantees/G-ROWS-001"
GUARANTEE = f"{ROWS}/guarantee.md"
RULE = (REPOSITORY / GUARANTEE).read_text()
GOLDEN = sorted(f"{ROWS}/golden/{path.name}" for path in (REPOSITORY / ROWS / "golden").iterdir())
NEW_CASE = f"{ROWS}/golden/C-ROWS-003.json"
LAST_LINK = "  key being its table's primary key, with the same intended counts.\n"
LINK = "- [C-ROWS-003](golden/C-ROWS-003.json): one more boundary,\n  described on two lines.\n"


def change(paths, **overrides):
    fields = dict(
        paths=paths,
        expectations_before=EXPECTATIONS,
        expectations_after=EXPECTATIONS,
        at_head=lambda path: RULE if path == GUARANTEE else None,
        files_at_head=lambda directory: GOLDEN if directory == f"{ROWS}/" else [],
    )
    fields.update(overrides)
    return policy.Change(**fields)


def with_expectations(edit):
    after = deepcopy(load_expectations())
    edit(after)
    return json.dumps(after)


@pytest.mark.parametrize(
    "paths",
    (
        ["src/kumo/pytorch/engine.py", "tests/test_core_api.py"],
        ["docs/README.md"],
        [GUARANTEE],
        [],
    ),
)
def test_changes_that_leave_the_spec_consistent_pass(paths):
    assert policy.evaluate(change(paths)) == []


# --- expectations only move forward --------------------------------------------------------


def test_refusal_to_answer_promotion_passes_alone():
    after = with_expectations(lambda e: e["C-ROWS-002"].__setitem__("diskgraph", "answer"))
    assert policy.evaluate(change([policy.EXPECTATIONS_PATH], expectations_after=after)) == []


@pytest.mark.parametrize("edit", ("downgrade", "reason", "remove", "add"))
def test_other_expectation_changes_need_a_guarantee_change(edit):
    def apply(expectations):
        if edit == "downgrade":
            expectations["C-ROWS-001"]["diskgraph"] = deepcopy(
                expectations["C-ROWS-002"]["diskgraph"]
            )
        elif edit == "reason":
            expectations["C-ROWS-002"]["diskgraph"]["message_contains"] = "different reason"
        elif edit == "remove":
            expectations.pop("C-ROWS-001")
        else:
            expectations["C-NEW-001"] = deepcopy(expectations["C-ROWS-001"])

    after = with_expectations(apply)
    alone = policy.evaluate(change([policy.EXPECTATIONS_PATH], expectations_after=after))
    assert len(alone) == 1
    assert "guarantee.md" in alone[0]
    explained = change([policy.EXPECTATIONS_PATH, GUARANTEE], expectations_after=after)
    assert policy.evaluate(explained) == []


def test_new_cases_alone_name_the_missing_golden_files():
    after = with_expectations(lambda e: e.__setitem__("C-NEW-001", deepcopy(e["C-ROWS-001"])))
    problems = policy.evaluate(change([policy.EXPECTATIONS_PATH], expectations_after=after))
    assert problems and "adds cases" in problems[0]


def test_a_golden_file_change_does_not_explain_an_expectation_change():
    after = with_expectations(lambda e: e.pop("C-ROWS-001"))
    paths = [policy.EXPECTATIONS_PATH, GOLDEN[0]]
    assert policy.evaluate(change(paths, expectations_after=after))


def test_the_expectations_file_must_remain():
    problems = policy.evaluate(change([policy.EXPECTATIONS_PATH], expectations_after=None))
    assert problems == [f"the PR must keep {policy.EXPECTATIONS_PATH}"]


def test_malformed_expectations_fail_with_a_reason():
    problems = policy.evaluate(change([policy.EXPECTATIONS_PATH], expectations_after="{not json"))
    assert problems and problems[0].startswith("invalid")


def test_unchanged_expectations_are_not_checked():
    # The rule reads the diff: a file the PR doesn't touch is not re-judged.
    assert policy.evaluate(change(["src/a.py"], expectations_after="{not json")) == []


# --- golden examples are linked from their guarantee ---------------------------------------


def adding_a_case(rule):
    return change(
        [NEW_CASE, GUARANTEE],
        at_head=lambda path: rule if path == GUARANTEE else None,
        files_at_head=lambda directory: [*GOLDEN, NEW_CASE] if directory == f"{ROWS}/" else [],
    )


def test_a_new_golden_case_linked_from_its_guarantee_passes():
    linked = RULE.replace(LAST_LINK, LAST_LINK + LINK)
    assert linked != RULE
    assert policy.evaluate(adding_a_case(linked)) == []


@pytest.mark.parametrize(
    "edit",
    (
        lambda rule: rule,
        lambda rule: rule + "\n## Notes\n\n" + LINK,
        lambda rule: rule.replace(LAST_LINK, LAST_LINK + LINK.replace("003.json", "009.json")),
        lambda rule: rule.replace(LAST_LINK, LAST_LINK + "See C-ROWS-003.\n"),
    ),
    ids=("unlinked", "other-section", "wrong-file", "prose"),
)
def test_a_golden_case_not_linked_from_its_examples_list_fails(edit):
    problems = policy.evaluate(adding_a_case(edit(RULE)))
    assert len(problems) == 1
    assert "C-ROWS-003" in problems[0]


def test_a_removed_link_fails_even_if_only_the_guarantee_changed():
    without = RULE.split("- [C-ROWS-002]")[0]
    problems = policy.evaluate(
        change([GUARANTEE], at_head=lambda path: without if path == GUARANTEE else None)
    )
    assert len(problems) == 1
    assert "C-ROWS-002" in problems[0]


def test_golden_cases_without_a_guarantee_fail():
    problems = policy.evaluate(change([NEW_CASE], at_head=lambda path: None))
    assert problems and "no guarantee.md" in problems[0]


def test_every_guarantee_in_the_repository_links_its_golden_cases():
    guarantees = REPOSITORY / "conformance/spec/guarantees"
    paths = [
        path.relative_to(REPOSITORY).as_posix() for path in guarantees.rglob("*") if path.is_file()
    ]

    def at_head(path):
        file = REPOSITORY / path
        return file.read_text() if file.exists() else None

    def files_at_head(directory):
        return [
            path.relative_to(REPOSITORY).as_posix()
            for path in (REPOSITORY / directory).rglob("*")
            if path.is_file()
        ]

    assert policy.golden_link_problems(paths, at_head, files_at_head) == []


# --- git plumbing, against a scratch repository --------------------------------------------


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    def write(path, text):
        file = tmp_path / policy.PACKAGE / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
        git("add", "-A")
        git("commit", "-q", "-m", f"write {path}")
        return git("rev-parse", "HEAD")

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("root\n")
    write(policy.EXPECTATIONS_PATH, EXPECTATIONS)
    write(GUARANTEE, RULE)
    for path in GOLDEN:
        write(path, "{}\n")
    monkeypatch.setattr(policy, "REPOSITORY", tmp_path)
    return git, write


def test_a_rename_reports_both_paths(scratch):
    git, _ = scratch
    base = git("rev-parse", "HEAD")
    git("mv", f"{policy.PACKAGE}{GOLDEN[0]}", f"{policy.PACKAGE}{ROWS}/golden/C-ROWS-009.json")
    git("commit", "-q", "-m", "rename")
    assert set(policy.changed(base, "HEAD")) == {GOLDEN[0], f"{ROWS}/golden/C-ROWS-009.json"}


def test_gather_reads_only_the_branch_since_its_merge_base(scratch):
    git, write = scratch
    git("checkout", "-q", "-b", "feature")
    after = with_expectations(lambda e: e["C-ROWS-002"].__setitem__("diskgraph", "answer"))
    head = write(policy.EXPECTATIONS_PATH, after)
    git("checkout", "-q", "main")
    (policy.REPOSITORY / "other.txt").write_text("outside the package\n")
    write("src/kumo/pytorch/later.py", "x = 1\n")
    found = policy.gather("main", head)
    assert found.paths == [policy.EXPECTATIONS_PATH]
    assert found.expectations_before == EXPECTATIONS
    assert found.expectations_after == after
    assert found.at_head(GUARANTEE) == RULE
    assert sorted(found.files_at_head(f"{ROWS}/")) == sorted([GUARANTEE, *GOLDEN])
    assert policy.evaluate(found) == []


def test_gather_sees_an_unlinked_golden_case(scratch):
    git, write = scratch
    git("checkout", "-q", "-b", "feature")
    head = write(NEW_CASE, "{}\n")
    problems = policy.evaluate(policy.gather("main", head))
    assert len(problems) == 1
    assert "C-ROWS-003" in problems[0]
