# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report kumo-pytorch workflow runs on main to the team channel.

Run by .github/workflows/kumo-pytorch-alert.yml for each completed run. A run
that fails on its first attempt has its failed jobs rerun once. After that, the
channel hears only about changes: a workflow that starts failing, and one that
recovers. `decide()` is pure; everything else reads the run from the
environment and the GitHub API.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

FAILING = {"failure", "timed_out"}
EVENTS = {"push", "schedule"}
BRANCH = "main"
RERUN, FAILING_NOW, RECOVERED = "rerun", "started failing", "recovered"


def decide(conclusion: str, attempt: int, previous: str | None) -> str | None:
    """What to do about a completed run, given the last earlier run's conclusion.

    `previous` is the conclusion of the most recent earlier run that succeeded or
    failed, or None if there is none.
    """
    if conclusion in FAILING:
        if attempt == 1:
            return RERUN
        return None if previous in FAILING else FAILING_NOW
    if conclusion == "success" and previous in FAILING:
        return RECOVERED
    return None


def previous_conclusion(runs: list[dict], current_number: int) -> str | None:
    """The conclusion of the latest earlier main run that succeeded or failed."""
    earlier = [
        run
        for run in runs
        if run["run_number"] < current_number
        and run["event"] in EVENTS
        and run["head_branch"] == BRANCH
        and run["status"] == "completed"
        and (run["conclusion"] == "success" or run["conclusion"] in FAILING)
    ]
    if not earlier:
        return None
    return max(earlier, key=lambda run: run["run_number"])["conclusion"]


def message(outcome: str, workflow: str, sha: str, commit_message: str, url: str) -> str:
    subject = commit_message.splitlines()[0] if commit_message else ""
    return f"{workflow} {outcome} on {BRANCH} at {sha[:10]} {subject}\n{url}"


def _gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _post(webhook: str, text: str) -> None:
    request = urllib.request.Request(
        webhook,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def main() -> int:
    env = os.environ
    repo, run_id = env["GITHUB_REPOSITORY"], env["RUN_ID"]
    if env["RUN_EVENT"] not in EVENTS or env["RUN_BRANCH"] != BRANCH:
        print(f"Ignoring a {env['RUN_EVENT']} run on {env['RUN_BRANCH']}.")
        return 0
    conclusion, attempt = env["RUN_CONCLUSION"], int(env["RUN_ATTEMPT"])
    previous = None
    if attempt > 1 or conclusion not in FAILING:
        page = _gh(
            "api",
            f"repos/{repo}/actions/workflows/{env['WORKFLOW_ID']}/runs"
            f"?branch={BRANCH}&status=completed&per_page=50",
        )
        previous = previous_conclusion(json.loads(page)["workflow_runs"], int(env["RUN_NUMBER"]))
    outcome = decide(conclusion, attempt, previous)
    print(f"{env['WORKFLOW']}: {conclusion} on attempt {attempt}, previously {previous}: {outcome}")
    if outcome == RERUN:
        _gh("run", "rerun", run_id, "--failed", "--repo", repo)
        return 0
    if outcome is None:
        return 0
    webhook = env.get("SLACK_WEBHOOK_URL", "")
    if not webhook:
        print("::notice::No alert webhook is configured; not posting.")
        return 0
    commit = _gh("api", f"repos/{repo}/commits/{env['RUN_SHA']}", "--jq", ".commit.message")
    _post(webhook, message(outcome, env["WORKFLOW"], env["RUN_SHA"], commit, env["RUN_URL"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
