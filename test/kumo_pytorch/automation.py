# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record complete nightly job outcomes and verify a later scheduled run."""

import argparse
import json
import os
from pathlib import Path

NIGHTLY_JOBS = frozenset(
    {
        "dataset-tooling",
        "diskgraph-wheel",
        "benchmark-memory",
        "aml-training-acceptance",
        "aml-fitting-memory",
        "benchmark-gpu",
        "benchmark-aml",
        "aml-population",
        "aml-lifecycle",
        "aml-summary",
        "package-regressions",
        "plan",
        "run",
        "aml-model-checks",
        "checks",
        "conformance",
    }
)


def nightly_status(results, *, run_id, attempt, sha, event, ref):
    missing = sorted(NIGHTLY_JOBS - results.keys())
    unexpected = sorted(results.keys() - NIGHTLY_JOBS)
    unsuccessful = {
        name: row.get("result")
        for name, row in results.items()
        if row.get("result") != "success"
    }
    return {
        "status": "failed" if missing or unexpected or unsuccessful else "passed",
        "run_id": str(run_id),
        "run_attempt": int(attempt),
        "head_sha": sha,
        "event": event,
        "ref": ref,
        "missing": missing,
        "unexpected": unexpected,
        "unsuccessful": unsuccessful,
        "jobs": results,
    }


def verify_schedule(run, evidence):
    """A failed test is visible evidence, not proof that the schedule never fired."""
    if run["event"] != "schedule" or run["head_branch"] != "main":
        raise ValueError(
            "Expected an automatic schedule run on main, not a manual/PR run"
        )
    if run["path"] != ".github/workflows/kumo-pytorch-t2b.yml":
        raise ValueError("Wrong workflow for nightly schedule verification")
    if run["status"] != "completed":
        raise ValueError("Nightly run has not completed")
    expected = nightly_status(
        evidence["jobs"],
        run_id=run["id"],
        attempt=run["run_attempt"],
        sha=run["head_sha"],
        event=run["event"],
        ref="refs/heads/main",
    )
    if evidence != expected:
        raise ValueError(
            "Nightly evidence is incomplete or belongs to another revision/run/attempt"
        )
    if evidence["missing"] or evidence["unexpected"]:
        raise ValueError("Nightly evidence does not account for every required job")
    conclusion = "success" if evidence["status"] == "passed" else "failure"
    if run["conclusion"] != conclusion:
        raise ValueError("Workflow conclusion and nightly evidence disagree")
    return {
        "scheduled_trigger": "verified",
        "required_jobs_executed": all(
            row.get("result") in {"success", "failure"}
            for row in evidence["jobs"].values()
        ),
        "tests": evidence["status"],
        "unsuccessful": evidence["unsuccessful"],
        "run_url": run["html_url"],
        "head_sha": run["head_sha"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-schedule")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "record":
        result = nightly_status(
            json.loads(os.environ["JOB_RESULTS"]),
            run_id=os.environ["GITHUB_RUN_ID"],
            attempt=os.environ["GITHUB_RUN_ATTEMPT"],
            sha=os.environ["GITHUB_SHA"],
            event=os.environ["GITHUB_EVENT_NAME"],
            ref=os.environ["GITHUB_REF"],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as stream:
                stream.write(f"## Complete nightly result: {result['status']}\n\n")
                stream.write("```json\n" + json.dumps(result, indent=2) + "\n```\n")
        return int(result["status"] != "passed")
    result = verify_schedule(
        json.loads(args.run.read_text()), json.loads(args.evidence.read_text())
    )
    print(json.dumps(result, indent=2))
    return int(not result["required_jobs_executed"])


if __name__ == "__main__":
    raise SystemExit(main())
