# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report kumo-pytorch workflow runs on main to the team channel.

Run by .github/workflows/kumo-pytorch-alert.yml for each completed run. A run
that fails on its first attempt has its failed jobs rerun once. After that, the
channel hears only about changes: a workflow that starts failing, and one that
recovers. A run that finishes after a later run of the same workflow is ignored.
The state last reported for each workflow is kept in a workflow artifact, so
each change is posted once however often a run is retried. `decide()` is pure;
everything else reads the run from the environment and the GitHub API.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from typing import TypedDict

FAILING = {"failure", "timed_out"}
EVENTS = {"push", "schedule"}
BRANCH = "main"
PASSING_STATE, FAILING_STATE = "passing", "failing"
RERUN, FAILING_NOW, RECOVERED = "rerun", "started failing", "recovered"
STATE_FILE = "state.json"


class Run(TypedDict):
    run_number: int
    conclusion: str | None
    event: str
    head_branch: str
    status: str


class State(TypedDict):
    state: str
    run_number: int
    run_attempt: int


def state_of(conclusion: str | None) -> str | None:
    """`passing` or `failing` for a run that succeeded or failed, else None."""
    if conclusion == "success":
        return PASSING_STATE
    if conclusion in FAILING:
        return FAILING_STATE
    return None


def decide(conclusion: str, attempt: int, reported: str | None) -> str | None:
    """What to do about a completed run, given the state last reported.

    `reported` is `passing`, `failing`, or None if nothing is known yet.
    """
    if conclusion in FAILING and attempt == 1:
        return RERUN
    current = state_of(conclusion)
    if current is None or current == reported:
        return None
    if current == FAILING_STATE:
        return FAILING_NOW
    return RECOVERED if reported == FAILING_STATE else None


def _deciding(runs: list[Run]) -> list[Run]:
    return [
        run
        for run in runs
        if run["event"] in EVENTS
        and run["head_branch"] == BRANCH
        and run["status"] == "completed"
        and state_of(run["conclusion"]) is not None
    ]


def superseded(runs: list[Run], current_number: int) -> bool:
    """Whether a later main run has already succeeded or failed."""
    return any(run["run_number"] > current_number for run in _deciding(runs))


def previous_conclusion(runs: list[Run], current_number: int) -> str | None:
    """The conclusion of the latest earlier main run that succeeded or failed."""
    earlier = [run for run in _deciding(runs) if run["run_number"] < current_number]
    if not earlier:
        return None
    return max(earlier, key=lambda run: run["run_number"])["conclusion"]


def artifact_name(workflow_id: str) -> str:
    return f"kumo-pytorch-alert-state-{workflow_id}"


def message(outcome: str, workflow: str, sha: str, commit_message: str, url: str) -> str:
    subject = commit_message.splitlines()[0] if commit_message else ""
    return f"{workflow} {outcome} on {BRANCH} at {sha[:10]} {subject}\n{url}"


def _gh_bytes(*args: str) -> bytes:
    result = subprocess.run(["gh", *args], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.decode().strip()}")
    return result.stdout


def _gh(*args: str) -> str:
    return _gh_bytes(*args).decode()


def _post(webhook: str, text: str) -> None:
    request = urllib.request.Request(
        webhook,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def reported_state(repo: str, workflow_id: str) -> State | None:
    """The state in the newest unexpired state artifact, or None if there is none."""
    page = json.loads(
        _gh("api", f"repos/{repo}/actions/artifacts?name={artifact_name(workflow_id)}&per_page=20")
    )
    artifacts = [artifact for artifact in page["artifacts"] if not artifact["expired"]]
    if not artifacts:
        return None
    newest = max(artifacts, key=lambda artifact: artifact["created_at"])
    archive = _gh_bytes("api", f"repos/{repo}/actions/artifacts/{newest['id']}/zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as files:
        return json.loads(files.read(STATE_FILE))


def _write_state(state: State) -> None:
    env = os.environ
    with open(env["STATE_PATH"], "w") as file:
        json.dump(state, file)
    if "GITHUB_OUTPUT" in env:
        with open(env["GITHUB_OUTPUT"], "a") as output:
            output.write("state-changed=true\n")


def main() -> int:
    env = os.environ
    repo, run_id, workflow_id = env["GITHUB_REPOSITORY"], env["RUN_ID"], env["WORKFLOW_ID"]
    if env["RUN_EVENT"] not in EVENTS or env["RUN_BRANCH"] != BRANCH:
        print(f"Ignoring a {env['RUN_EVENT']} run on {env['RUN_BRANCH']}.")
        return 0
    conclusion, attempt = env["RUN_CONCLUSION"], int(env["RUN_ATTEMPT"])
    number = int(env["RUN_NUMBER"])
    if conclusion in FAILING and attempt == 1:
        print(f"{env['WORKFLOW']}: {conclusion} on attempt 1: {RERUN}")
        _gh("run", "rerun", run_id, "--failed", "--repo", repo)
        return 0
    current = state_of(conclusion)
    if current is None:
        print(f"{env['WORKFLOW']}: ignoring a {conclusion} run.")
        return 0
    page = _gh(
        "api",
        f"repos/{repo}/actions/workflows/{workflow_id}/runs"
        f"?branch={BRANCH}&status=completed&per_page=50",
    )
    runs: list[Run] = json.loads(page)["workflow_runs"]
    if superseded(runs, number):
        print(f"::notice::Run {number} finished after a later run; ignoring it.")
        return 0
    state = reported_state(repo, workflow_id)
    if state is not None:
        reported = state["state"]
        source = f"reported by run {state['run_number']} attempt {state['run_attempt']}"
    else:
        reported = state_of(previous_conclusion(runs, number))
        source = "from the previous run"
    outcome = decide(conclusion, attempt, reported)
    print(
        f"{env['WORKFLOW']}: {conclusion} on attempt {attempt}, "
        f"previously {reported} ({source}): {outcome}"
    )
    if outcome is not None:
        webhook = env.get("SLACK_WEBHOOK_URL", "")
        if webhook:
            commit = _gh("api", f"repos/{repo}/commits/{env['RUN_SHA']}", "--jq", ".commit.message")
            _post(
                webhook, message(outcome, env["WORKFLOW"], env["RUN_SHA"], commit, env["RUN_URL"])
            )
        else:
            print("::notice::No alert webhook is configured; not posting.")
    if state is None or current != reported:
        _write_state(State(state=current, run_number=number, run_attempt=attempt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
