# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from tests.e2e import ci_plan as ci
from tests.e2e.nightly_report import METRICS

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
    next(j for j in raw if j["name"] == "binding-train")["name"] = (
        "matrix.binding && format('GPU training {0}', matrix.binding) || 'binding-train'"
    )
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


def test_nightly_keeps_correctness_and_trains_only_the_six_full_bindings():
    p = plan("schedule")
    assert p["training_bindings"] == ["S02", "L01", "S01", "M04", "S04", "M03"]
    required = [r for r in p["selections"] if r["status"] == "required"]
    assert {r["binding"] for r in required if r["mode"] == "correctness"} == set(ci.BINDINGS)
    assert [r["binding"] for r in required if r["scope"] == "full_source"] == p["training_bindings"]
    assert not any(
        r["mode"] in {"training", "lifecycle"} or r["binding"] == "T01" for r in required
    )
    assert not p["t2a"] and not p["jobs"]["hm-lifecycle / Nightly H&M training"]
    raw = jobs(p)
    next(j for j in raw if j["name"] == "hm-lifecycle / Nightly H&M training")["name"] = (
        "hm-lifecycle / github.event_name == 'schedule' && 'Nightly H&M training' || "
        "format('Nightly end-to-end {0} seed {1} (size M)', matrix.flow, matrix.seed)"
    )
    assert not ci.account_jobs(p, raw)[1]
    assert all(p["jobs"]["hm-lifecycle / " + name] for name in ci.CONFORMANCE)
    assert plan(paths=None)["training_bindings"] == []
    with pytest.raises(ValueError, match="Unknown training binding"):
        plan("workflow_dispatch", inputs={"mode": "binding-train", "binding": "E01"})


def test_training_report_requires_full_epochs_metrics_and_losses(tmp_path):
    from e2e.datasets.binding_sources import PINS

    p = plan("workflow_dispatch", inputs={"mode": "binding-train", "binding": "M04"})
    name = "binding-train-M04-attempt-2"
    folder = tmp_path / name
    folder.mkdir()
    pins = json.loads((PINS / "mbd-m04.json").read_text())["objects"]
    counts = dict(zip(("train", "valid", "test"), ci.NIGHTLY_TRAINING["M04"], strict=True))

    def split(s, e):
        n = counts[s]
        return dict(
            epoch=e,
            split=s,
            samples=n,
            tokens=0,
            elapsed_ns=10**9,
            optimizer_updates=2 if s == "train" else 0,
            heads={"count": dict(eligible=n, loss_sum=n / 4, mean_loss=0.25)},
        )

    data = dict(
        binding="M04",
        workload="binding-train",
        revision="candidate",
        run_id=14,
        run_attempt=2,
        stage="complete",
        tracked_diff_sha256=None,
        complete=True,
        scope="full_source",
        device="cuda",
        fit_split="train",
        history_limit=64,
        split_counts=counts,
        expected_eligible={"count": counts},
        source_manifest_sha256=pins["source_manifest"]["sha256"],
        anchor_manifest_sha256=pins["anchor_manifest"]["sha256"],
        epochs=[
            dict(split("train", e), gpu=dict(samples=5, mean_percent=15, device_uuid="GPU-1"))
            for e in (1, 2)
        ],
        validation=[split("valid", e) for e in (1, 2)],
        test=split("test", 2),
        metrics=dict(
            ingestion_seconds=5,
            materialization_seconds=1,
            training_seconds=2,
            samples_per_second=counts["train"],
            tokens_per_second=0,
            peak_ram_bytes=2048,
            average_gpu_utilization_percent=15,
            ssd_bytes=4096,
        ),
    )
    artifacts = [
        dict(id=1, name=name, expired=False, workflow_run={"id": 14, "head_sha": "candidate"})
    ]

    def report():
        (folder / "report.json").write_text(json.dumps(data))
        return ci.results(p, jobs(p), artifacts, tmp_path)

    result = report()
    ci.gate(p, result)
    (folder / "report.json").unlink()
    folder.rmdir()
    assert (
        "Missing current-attempt artifact"
        in ci.results(p, jobs(p), artifacts, tmp_path)["errors"][0]
    )
    folder = tmp_path
    flat = report()
    ci.gate(p, flat)
    assert next(r for r in flat["selections"] if r["status"] == "executed")["artifact"] == name
    stale = [{**artifacts[0], "name": "binding-train-M04"}]
    assert (
        "Missing current-attempt artifact" in ci.results(p, jobs(p), stale, tmp_path)["errors"][0]
    )
    markdown = ci.render(result)
    assert "Mean loss" in markdown and all(label in markdown for _, label, _ in METRICS)
    assert "64 recent events" in markdown
    del data["history_limit"]
    assert "history_limit" in report()["errors"][0]
    data["history_limit"] = 64
    data["epochs"][1]["samples"] -= 1
    assert "expected samples=" in report()["errors"][0]
    data["epochs"][1]["samples"] += 1
    data["validation"][0]["heads"].clear()
    assert "loss head" in report()["errors"][0]


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
