# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decisions behind the kumo-pytorch alert workflow on main.

The script lives at .github/scripts/kumo-pytorch-alert.py, outside the package.
"""

import importlib.util
import json
import sys

import pytest

from conformance.cases import REPOSITORY

SCRIPT = REPOSITORY.parents[2] / ".github/scripts/kumo-pytorch-alert.py"
_spec = importlib.util.spec_from_file_location("kumo_pytorch_alert", SCRIPT)
alert = importlib.util.module_from_spec(_spec)
sys.modules["kumo_pytorch_alert"] = alert
_spec.loader.exec_module(alert)


@pytest.mark.parametrize(
    ("conclusion", "attempt", "previous", "outcome"),
    (
        ("failure", 1, "success", alert.RERUN),
        ("failure", 1, "failure", alert.RERUN),
        ("timed_out", 1, None, alert.RERUN),
        ("failure", 2, "success", alert.FAILING_NOW),
        ("failure", 2, None, alert.FAILING_NOW),
        ("timed_out", 3, "success", alert.FAILING_NOW),
        ("failure", 2, "failure", None),
        ("failure", 2, "timed_out", None),
        ("success", 1, "failure", alert.RECOVERED),
        ("success", 2, "failure", alert.RECOVERED),
        ("success", 2, "success", None),
        ("success", 1, None, None),
        ("cancelled", 1, "success", None),
        ("skipped", 1, "failure", None),
    ),
)
def test_decide(conclusion, attempt, previous, outcome):
    assert alert.decide(conclusion, attempt, previous) == outcome


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


def test_the_message_has_the_workflow_commit_subject_and_run():
    text = alert.message(
        alert.FAILING_NOW, "kumo-pytorch tests", "0123456789abcdef", "Fix it\n\nDetails", "URL"
    )
    assert text == "kumo-pytorch tests started failing on main at 0123456789 Fix it\nURL"


@pytest.fixture
def environment(monkeypatch):
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
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    calls, posts = [], []

    def gh(*args):
        calls.append(args)
        if args[0] == "api" and "/commits/" in args[1]:
            return "Fix it\n"
        return json.dumps({"workflow_runs": [run(12, "failure"), run(11, "success")]})

    monkeypatch.setattr(alert, "_gh", gh)
    monkeypatch.setattr(alert, "_post", lambda url, text: posts.append((url, text)))
    return monkeypatch, calls, posts


def test_a_final_failure_after_a_success_posts_once(environment):
    _, _, posts = environment
    assert alert.main() == 0
    assert posts == [
        (
            "https://hooks.example/1",
            "kumo-pytorch tests started failing on main at 0123456789 Fix it\nURL",
        )
    ]


def test_a_first_failed_attempt_is_rerun_without_posting(environment):
    monkeypatch, calls, posts = environment
    monkeypatch.setenv("RUN_ATTEMPT", "1")
    assert alert.main() == 0
    assert calls == [("run", "rerun", "100", "--failed", "--repo", "owner/repo")]
    assert posts == []


def test_without_a_webhook_nothing_is_posted(environment):
    monkeypatch, _, posts = environment
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "")
    assert alert.main() == 0
    assert posts == []


@pytest.mark.parametrize(
    ("key", "value"), (("RUN_EVENT", "workflow_dispatch"), ("RUN_BRANCH", "feature"))
)
def test_runs_off_main_or_by_hand_are_ignored(environment, key, value):
    monkeypatch, calls, posts = environment
    monkeypatch.setenv(key, value)
    assert alert.main() == 0
    assert calls == [] and posts == []
