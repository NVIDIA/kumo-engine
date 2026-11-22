# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from tests.e2e import ci_plan as ci

REPO = "NVIDIA/kumo-engine"
PR = {"number": 1, "head": {"sha": "head", "repo": {"full_name": REPO}}, "base": {"sha": "base"}}


def plan(event="pull_request", paths=("docs/README.md",), **payload):
    return ci.make_plan(
        event,
        {"pull_request": PR, **payload} if event == "pull_request" else payload,
        paths,
        revision="candidate",
        ref="refs/heads/main",
        repository=REPO,
        run_id=14,
        attempt=2,
    )


def jobs(p):
    return [
        dict(
            name=name,
            head_sha=p["source_revision"],
            run_id=14,
            run_attempt=2,
            html_url="job",
            conclusion="success" if required else "skipped",
        )
        for name, required in ci.planned_jobs(p).items()
    ]


def test_rename_and_incomplete_lists_select_the_package(monkeypatch):
    old = "packages/experimental/kumo-pytorch/tests/e2e/test_aml.py"
    after = {**PR, "changed_files": 1}
    monkeypatch.setattr(
        ci,
        "gh",
        lambda path, **kw: (
            [[{"filename": "unrelated.py", "previous_filename": old}]] if kw else after
        ),
    )
    moved = ci.changed_paths({"pull_request": PR})
    assert plan(paths=moved)["t2a"]
    assert plan(paths=[ci.WORKFLOW])["t2a"]
    assert plan(paths=["packages/diskgraph/src/lib.rs"])["t2b"]
    after["changed_files"] = 2
    assert ci.changed_paths({"pull_request": PR}) is None and plan(paths=None)["suite"]
    assert not plan()["affected"]
    fork = dict(PR, head={"sha": "head", "repo": {"full_name": "fork/project"}})
    assert not plan(paths=moved, pull_request=fork)["private"]
    for event in ("schedule", "pull_request", "push"):
        assert all(
            r["status"] != "required"
            for r in ci.selection_rows(plan(event, paths=None))
            if r["group"] in {"benchmark", "real 25B"} or r["mode"] == "full"
        )


def test_gate_checks_attempt_membership_and_explicit_exemption(tmp_path):
    p = plan()
    raw = jobs(p)
    evidence = ci.results(p, raw, [], tmp_path)
    ci.gate(p, evidence)
    assert evidence["inventory"] == {"correctness": 12, "training": 11, "lifecycle": 6, "full": 11}
    row = evidence["selections"].pop()
    with pytest.raises(ValueError, match="selection set"):
        ci.gate(p, evidence)
    evidence["selections"].append(row)
    del evidence["jobs"]["plan"]
    with pytest.raises(ValueError, match="job set"):
        ci.gate(p, evidence)
    raw[0].update(run_attempt=1)
    _, errors = ci.account_jobs(p, raw + [raw[0]])
    assert any("stale" in e for e in errors)


def test_single_full_binding_requires_successful_full_source_receipt(tmp_path):
    p = plan("workflow_dispatch", inputs={"mode": "binding-full", "binding": "E01"})
    (row,) = [r for r in ci.selection_rows(p) if r["status"] == "required"]
    assert row["binding"] == "E01" and row["scope"] == "full_source" and row["mode"] == "full"
    name = "binding-full-E01-attempt-2"
    source = tmp_path / name
    source.mkdir()
    (source / "evidence.xml").write_text(
        f'<testsuite><testcase><properties><property name="nodeid" value="{row["nodeid"]}" />'
        "</properties></testcase></testsuite>"
    )
    key = hashlib.sha256(row["nodeid"].encode()).hexdigest()[:16]
    receipt = source / f"binding-full-{key}.json"
    fields = {k: row[k] for k in ("binding", "representation", "selector", "scope", "mode")}
    fields.update(
        outcome="executed",
        stage="complete",
        full_population_verified=True,
        splits={
            s: {"delivered": 5, "eligible": {}, "features": {"graph": 5}}
            for s in ("train", "valid", "test")
        },
        expected=dict(
            counts=dict.fromkeys(("train", "valid", "test"), 5),
            targets={},
            features=["graph"],
            checks={},
        ),
        checks={},
        revision="candidate",
        tracked_diff_sha256=None,
    )
    receipt.write_text(json.dumps(fields))
    artifacts = [
        dict(id=20, name=name, expired=False, workflow_run={"id": 14, "head_sha": "candidate"})
    ]
    report = ci.results(p, jobs(p), artifacts, tmp_path)
    ci.gate(p, report)
    fields.update(scope="bounded", full_population_verified=False)
    receipt.write_text(json.dumps(fields))
    assert ci.results(p, jobs(p), artifacts, tmp_path)["errors"]
    artifacts[0]["name"] = "binding-full-E01"
    assert "current-attempt" in ci.results(p, jobs(p), artifacts, tmp_path)["errors"][0]


def test_scheduled_verification_distinguishes_firing_and_failing(tmp_path):
    p = plan("schedule")
    raw = jobs(p)
    raw[0]["conclusion"] = "failure"
    report = ci.results(p, raw, [], tmp_path)
    for name, result in (("results", "success"), ("kumo-pytorch-gate", "failure")):
        raw.append(dict(raw[0], name=name, conclusion=result))
    run = dict(
        path=ci.WORKFLOW,
        event="schedule",
        head_branch="main",
        head_sha="candidate",
        id=14,
        run_attempt=2,
        status="completed",
        conclusion="failure",
        html_url="https://example.org/run",
    )
    assert ci.scheduled(report, run, raw)["tests"] == "failure"
    with pytest.raises(ValueError):
        ci.scheduled(report, dict(run, event="workflow_dispatch"), raw)
