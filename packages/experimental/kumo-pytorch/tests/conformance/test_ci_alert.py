# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decisions behind the kumo-pytorch alert workflow on main.

The script lives at .github/scripts/kumo-pytorch-alert.py, outside the package.
"""

import importlib.util
import io
import json
import os
import sys
import zipfile

import pytest

from conformance.cases import REPOSITORY

SCRIPT = REPOSITORY.parents[2] / ".github/scripts/kumo-pytorch-alert.py"
_spec = importlib.util.spec_from_file_location("kumo_pytorch_alert", SCRIPT)
alert = importlib.util.module_from_spec(_spec)
sys.modules["kumo_pytorch_alert"] = alert
_spec.loader.exec_module(alert)


@pytest.mark.parametrize(
    ("conclusion", "attempt", "reported", "outcome"),
    (
        ("failure", 1, "passing", alert.RERUN),
        ("failure", 1, "failing", alert.RERUN),
        ("timed_out", 1, None, alert.RERUN),
        ("failure", 2, "passing", alert.FAILING_NOW),
        ("failure", 2, None, alert.FAILING_NOW),
        ("timed_out", 3, "passing", alert.FAILING_NOW),
        ("failure", 2, "failing", None),
        ("timed_out", 2, "failing", None),
        ("success", 1, "failing", alert.RECOVERED),
        ("success", 2, "failing", alert.RECOVERED),
        ("success", 2, "passing", None),
        ("success", 1, None, None),
        ("cancelled", 1, "passing", None),
        ("skipped", 1, "failing", None),
    ),
)
def test_decide(conclusion, attempt, reported, outcome):
    assert alert.decide(conclusion, attempt, reported) == outcome


def run(number, conclusion, event="push", branch="main", status="completed"):
    return dict(
        run_number=number, conclusion=conclusion, event=event, head_branch=branch, status=status
    )


def test_the_previous_run_skips_unfinished_cancelled_and_other_runs():
    runs = [
        run(12, "failure"),  # the current run
        run(11, "cancelled"),
        run(10, "success", event="workflow_dispatch"),
        run(9, "failure", status="in_progress"),
        run(8, "success", branch="other"),
        run(7, "timed_out"),
        run(6, "success"),
    ]
    assert alert.previous_conclusion(runs, 12) == "timed_out"
    assert alert.previous_conclusion(runs, 7) == "success"
    assert alert.previous_conclusion(runs, 6) is None


def test_only_a_later_main_run_that_succeeded_or_failed_supersedes():
    runs = [
        run(13, "cancelled"),
        run(14, "success", event="workflow_dispatch"),
        run(15, "failure", branch="other"),
        run(12, "failure"),
    ]
    assert not alert.superseded(runs, 12)
    assert alert.superseded([*runs, run(13, "success")], 12)


def test_the_message_has_the_workflow_commit_subject_and_run():
    text = alert.message(
        alert.FAILING_NOW, "kumo-pytorch tests", "0123456789abcdef", "Fix it\n\nDetails", "URL"
    )
    assert text == "kumo-pytorch tests started failing on main at 0123456789 Fix it\nURL"


def state_archive(state):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("state.json", json.dumps(state))
    return buffer.getvalue()


class GitHub:
    """The API calls the script makes, answered from runs and state artifacts."""

    def __init__(self):
        self.runs = [run(12, "failure"), run(11, "success")]
        self.states = []
        self.calls = []

    def gh_bytes(self, *args):
        self.calls.append(args)
        if args[0] != "api":
            return b""
        path = args[1]
        if "/commits/" in path:
            return b"Fix it\n"
        if "/runs?" in path:
            return json.dumps({"workflow_runs": self.runs}).encode()
        if "/artifacts?name=kumo-pytorch-alert-state-7&" in path:
            artifacts = [
                dict(id=index, expired=expired, created_at=f"2026-11-{index + 1:02d}T00:00:00Z")
                for index, (_, expired) in enumerate(self.states)
            ]
            return json.dumps({"artifacts": artifacts}).encode()
        if path.endswith("/zip"):
            index = int(path.split("/")[-2])
            return state_archive(self.states[index][0])
        raise AssertionError(f"unexpected call {args}")


@pytest.fixture
def environment(monkeypatch, tmp_path):
    values = dict(
        GITHUB_REPOSITORY="owner/repo",
        WORKFLOW="kumo-pytorch tests",
        WORKFLOW_ID="7",
        RUN_ID="100",
        RUN_NUMBER="12",
        RUN_ATTEMPT="2",
        RUN_CONCLUSION="failure",
        RUN_EVENT="push",
        RUN_BRANCH="main",
        RUN_SHA="0123456789abcdef",
        RUN_URL="URL",
        SLACK_WEBHOOK_URL="https://hooks.example/1",
        STATE_PATH=str(tmp_path / "state.json"),
        GITHUB_OUTPUT=str(tmp_path / "output"),
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    github, posts = GitHub(), []
    monkeypatch.setattr(alert, "_gh_bytes", github.gh_bytes)
    monkeypatch.setattr(alert, "_post", lambda url, text: posts.append((url, text)))
    return monkeypatch, github, posts


def written_state():
    path = os.environ["STATE_PATH"]
    output = os.environ["GITHUB_OUTPUT"]
    if not os.path.exists(path):
        assert not os.path.exists(output)
        return None
    with open(output) as file:
        assert file.read() == "state-changed=true\n"
    with open(path) as file:
        return json.load(file)


FAILING_NOW_TEXT = "kumo-pytorch tests started failing on main at 0123456789 Fix it\nURL"
RECOVERED_TEXT = "kumo-pytorch tests recovered on main at 0123456789 Fix it\nURL"


def test_a_final_failure_after_a_success_posts_once(environment):
    _, github, posts = environment
    github.states = [(dict(state="passing", run_number=11, run_attempt=1), False)]
    assert alert.main() == 0
    assert posts == [("https://hooks.example/1", FAILING_NOW_TEXT)]
    assert written_state() == dict(state="failing", run_number=12, run_attempt=2)


def test_a_first_failed_attempt_is_rerun_without_posting(environment):
    monkeypatch, github, posts = environment
    monkeypatch.setenv("RUN_ATTEMPT", "1")
    assert alert.main() == 0
    assert github.calls == [("run", "rerun", "100", "--failed", "--repo", "owner/repo")]
    assert posts == []
    assert written_state() is None


def test_a_late_failure_of_an_older_run_is_ignored(environment):
    """(a) 11 and 13 pass, then 12's delayed second attempt fails."""
    _, github, posts = environment
    github.runs = [run(13, "success"), run(12, "failure"), run(11, "success")]
    github.states = [(dict(state="passing", run_number=13, run_attempt=1), False)]
    assert alert.main() == 0
    assert posts == []
    assert written_state() is None


def test_a_late_success_of_an_older_run_is_ignored(environment):
    """(b) 11 and 13 fail, then 12's delayed retry passes."""
    monkeypatch, github, posts = environment
    monkeypatch.setenv("RUN_CONCLUSION", "success")
    github.runs = [run(13, "failure"), run(12, "success"), run(11, "failure")]
    github.states = [(dict(state="failing", run_number=11, run_attempt=2), False)]
    assert alert.main() == 0
    assert posts == []
    assert written_state() is None


def test_a_failure_already_reported_is_not_posted_again(environment):
    """(c) Run 12 reported its failure on attempt 2; a manual attempt 3 fails."""
    monkeypatch, github, posts = environment
    monkeypatch.setenv("RUN_ATTEMPT", "3")
    github.states = [
        (dict(state="passing", run_number=11, run_attempt=1), False),
        (dict(state="failing", run_number=12, run_attempt=2), False),
    ]
    assert alert.main() == 0
    assert posts == []
    assert written_state() is None


def test_a_rerun_that_passes_after_a_reported_failure_recovers(environment):
    """(d) Run 12 reported a failure; a manual rerun of 12 passes, no later run."""
    monkeypatch, github, posts = environment
    monkeypatch.setenv("RUN_ATTEMPT", "3")
    monkeypatch.setenv("RUN_CONCLUSION", "success")
    github.runs = [run(12, "success"), run(11, "success")]
    github.states = [(dict(state="failing", run_number=12, run_attempt=2), False)]
    assert alert.main() == 0
    assert posts == [("https://hooks.example/1", RECOVERED_TEXT)]
    assert written_state() == dict(state="passing", run_number=12, run_attempt=3)


def test_without_a_state_artifact_the_previous_run_decides(environment):
    _, github, posts = environment
    github.states = [(dict(state="failing", run_number=5, run_attempt=2), True)]
    assert alert.main() == 0
    assert posts == [("https://hooks.example/1", FAILING_NOW_TEXT)]
    assert written_state() == dict(state="failing", run_number=12, run_attempt=2)


def test_without_a_state_artifact_a_continued_failure_records_without_posting(environment):
    _, github, posts = environment
    github.runs = [run(12, "failure"), run(11, "timed_out")]
    assert alert.main() == 0
    assert posts == []
    assert written_state() == dict(state="failing", run_number=12, run_attempt=2)


def test_without_a_webhook_nothing_is_posted_but_the_state_is_recorded(environment):
    monkeypatch, github, posts = environment
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "")
    github.states = [(dict(state="passing", run_number=11, run_attempt=1), False)]
    assert alert.main() == 0
    assert posts == []
    assert written_state() == dict(state="failing", run_number=12, run_attempt=2)


def test_a_cancelled_run_changes_nothing(environment):
    monkeypatch, github, posts = environment
    monkeypatch.setenv("RUN_CONCLUSION", "cancelled")
    assert alert.main() == 0
    assert github.calls == [] and posts == []
    assert written_state() is None


@pytest.mark.parametrize(
    ("key", "value"), (("RUN_EVENT", "workflow_dispatch"), ("RUN_BRANCH", "feature"))
)
def test_runs_off_main_or_by_hand_are_ignored(environment, key, value):
    monkeypatch, github, posts = environment
    monkeypatch.setenv(key, value)
    assert alert.main() == 0
    assert github.calls == [] and posts == []
