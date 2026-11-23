# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
import shlex
from pathlib import Path

import pytest

from tests.e2e import ci_plan as ci
from tests.e2e.nightly_report import METRICS

REPO = "NVIDIA/kumo-engine"
PR = {"number": 1, "head": {"sha": "head", "repo": {"full_name": REPO}}, "base": {"sha": "base"}}


def plan(event="pull_request", **payload):
    return ci.make_plan(
        event,
        {"pull_request": PR, **payload} if event == "pull_request" else payload,
        revision="candidate",
        ref="refs/pull/1/merge" if event == "pull_request" else "refs/heads/main",
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
            started_at="2026-11-01T00:00:00Z",
            completed_at="2026-11-01T01:00:00Z",
        )
        for name, required in ci.planned_jobs(p).items()
    ]


def test_pr_and_nightly_require_identical_package_jobs_and_evidence(monkeypatch):
    monkeypatch.setattr(ci, "time", lambda: 86400 * 100 + 42)
    pr = plan()
    nightly = plan("schedule")
    replay = plan("workflow_dispatch", inputs={})
    for field in ("jobs", "selections", "training_bindings", "seed_start", "torch_backend"):
        assert pr[field] == nightly[field] == replay[field]
    assert ci.evidence_names(pr) == ci.evidence_names(nightly) == ci.evidence_names(replay)
    assert nightly["seed_start"] == 1200
    assert all(pr["jobs"].values())
    assert not {"cuda", "binding-full", "aml-benchmark", "streaming-25b"} & pr["jobs"].keys()
    assert ci.account_jobs(pr, jobs(pr))[1] == []
    assert pr["training_bindings"] == ["S02", "L01", "S01", "M04", "S04", "M03"]
    rows = pr["selections"]
    assert all(row["status"] == "required" for row in rows)
    for mode in ("correctness", "training", "lifecycle"):
        assert {r["binding"] for r in rows if r["mode"] == mode} == set(ci.BINDINGS)
    training_rows = [r for r in rows if r["group"] == "GPU training"]
    assert [r["binding"] for r in training_rows] == pr["training_bindings"]
    assert all(
        (r["mode"], r["scope"], r["artifact"])
        == ("binding-throughput", "extract", f"binding-throughput-{r['binding']}")
        for r in training_rows
    )
    assert all(r["scope"] != "full_source" for r in rows)
    assert pr["jobs"]["Package tests / Package and dataset tests"]
    assert all(pr["jobs"][f"Package tests / Engine conformance {i}/3"] for i in (1, 2, 3))
    training = [name for name in pr["jobs"] if name.startswith("Training · ")]
    assert len(training) == 15
    assert "Training · AML L01 / GPU" in training
    matrix = ci.training_matrix(pr)["include"]
    assert len(matrix) == 15
    assert all(row["seed"] in {"0", "1", "2"} for row in matrix if "tier" in row)
    for dataset, seeds in (("H&M Small", 3), ("H&M Medium", 2)):
        assert pr["jobs"][f"Lifecycle · {dataset} / Check saved models"]
        for flow, count in (("gnn", seeds), ("pragma", 2)):
            for i in range(count):
                assert f"Training · {dataset} · {flow} (seed {i}) / CPU" in training
    raw = jobs(pr)
    inactive = dict(raw[0], name="Training · AML L01 / CPU", conclusion="skipped")
    assert ci.account_jobs(pr, [*raw, inactive])[1] == []
    assert ci.account_jobs(pr, [*raw, dict(inactive, conclusion="success")])[1]
    fork = dict(PR, head={"sha": "head", "repo": {"full_name": "fork/project"}})
    with pytest.raises(ValueError, match="private datasets and a same-repository PR"):
        plan(pull_request=fork)


def test_integrated_suite_runs_full_package_pytest_once():
    root = Path(__file__).resolve().parents[5]
    suite = root / ci.WORKFLOW
    calls = re.findall(r"^    uses: \./(\.github/workflows/\S+\.yml)$", suite.read_text(), re.M)
    assert ".github/workflows/kumo-pytorch-diskgraph.yml" not in calls  # Standalone launcher.
    full_runs = []
    for workflow in [suite, *(root / path for path in calls)]:
        jobs = workflow.read_text().split("\njobs:\n", 1)[1].replace("\\\n", " ")
        job = None
        for line in jobs.splitlines():
            if header := re.fullmatch(r"  ([\w-]+):", line):
                job = header[1]
            if line.lstrip().startswith(("#", "name:", "- name:")) or not (
                cmd := re.search(r"(?:^|\s)pytest(?=\s|$)(.*)", line)
            ):
                continue
            args = shlex.split(cmd[1])
            paths = (arg.removeprefix("./").rstrip("/") for arg in args if not arg.startswith("-"))
            if {"-k", "-m"}.intersection(args) or any(
                path.startswith("tests/") or path.split("::")[0].endswith(".py") for path in paths
            ):
                continue
            full_runs.append((workflow.name, job, args))
    assert [(file, job) for file, job, _ in full_runs] == [
        ("_kumo-pytorch-package-tests.yml", "package")
    ]
    assert {"--complete-fixtures", "--require-imports"} <= set(full_runs[0][2])


def test_gate_checks_attempt_membership_and_requires_every_selected_row(tmp_path):
    p = plan("workflow_dispatch", inputs={"mode": "binding-train", "binding": "M03"})
    raw = jobs(p)
    evidence = ci.results(p, raw, [], tmp_path)
    assert any("Missing current-attempt artifact" in error for error in evidence["errors"])
    with pytest.raises(ValueError):
        ci.gate(p, evidence)
    evidence.update(errors=[], status="passed", selections=[])
    with pytest.raises(ValueError, match="selection set"):
        ci.gate(p, evidence)
    del evidence["jobs"]["Select benchmark"]
    with pytest.raises(ValueError, match="job set"):
        ci.gate(p, evidence)
    _, errors = ci.account_jobs(p, raw + [dict(raw[0], run_attempt=1)])
    assert any("stale" in e for e in errors)


def test_manual_benchmarks_select_only_requested_work():
    manual = plan("workflow_dispatch", inputs={"mode": "binding-train", "binding": "all"})
    assert manual["training_bindings"] == list(ci.NIGHTLY_TRAINING)
    assert [r["binding"] for r in manual["selections"]] == manual["training_bindings"]
    assert all(
        (r["mode"], r["scope"], r["artifact"])
        == ("two-epoch-training", "full_source", f"binding-train-{r['binding']}")
        for r in manual["selections"]
    )
    assert manual["jobs"]["GPU training / MBD M03 / GPU"]
    assert manual["jobs"]["dataset-source"]
    for job in ("Package tests", "H&M Small", "Distribution"):
        assert not any(name.startswith(job) for name in manual["jobs"])
    with pytest.raises(ValueError, match="Unknown training binding"):
        plan("workflow_dispatch", inputs={"mode": "binding-train", "binding": "E01"})
    for mode in ("cuda", *ci.AML, *ci.STREAMING):
        p = plan("workflow_dispatch", inputs={"mode": mode})
        assert p["jobs"]["dataset-source"] and not p["jobs"]["GPU training"]
        assert p["selections"] and all(row["status"] == "required" for row in p["selections"])


def m04_training_report(counts):
    from e2e.datasets.binding_sources import PINS

    pins = json.loads((PINS / "mbd-m04.json").read_text())["objects"]

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

    return dict(
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


def test_training_report_requires_full_epochs_metrics_and_losses(tmp_path):
    p = plan("workflow_dispatch", inputs={"mode": "binding-train", "binding": "M04"})
    name = "binding-train-M04-attempt-2"
    folder = tmp_path / name
    folder.mkdir()
    counts = dict(zip(("train", "valid", "test"), ci.NIGHTLY_TRAINING["M04"], strict=True))
    data = m04_training_report(counts)
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
    assert "| History tokens/s |" in markdown and "| N/A (graph-only) |" in markdown
    assert "64 recent events" in markdown
    assert "First GPU job start to last finish took **1:00:00**" in markdown
    del data["history_limit"]
    assert "history_limit" in report()["errors"][0]
    data["history_limit"] = 64
    data["epochs"][1]["samples"] -= 1
    assert "expected samples=" in report()["errors"][0]
    data["epochs"][1]["samples"] += 1
    data["validation"][0]["heads"].clear()
    assert "loss head" in report()["errors"][0]


def test_nightly_report_accepts_only_the_pinned_training_window(tmp_path):
    from e2e.datasets.binding_sources import PINS

    p = plan("schedule")
    pin = (PINS / "throughput" / "m04.json").read_bytes()
    extract = json.loads(pin)
    data = m04_training_report(extract["split_counts"])
    data.update(
        workload="binding-throughput",
        scope="extract",
        batch_size=32,
        staging_seconds=12,
        extract_seconds=0.01,
        extract_manifest_sha256=hashlib.sha256(pin).hexdigest(),
        expected_eligible=extract["expected_eligible"],
    )
    for row, phase, updates in zip(
        data.pop("epochs"), ("warmup", "measurement"), (10, 100), strict=True
    ):
        n = updates * data["batch_size"]
        row.update(phase=phase, epoch=1, optimizer_updates=updates, samples=n)
        row["heads"]["count"].update(eligible=n, loss_sum=n / 4)
        data[phase] = row
    data["validation"].pop()
    data["test"]["epoch"] = 1
    data["metrics"].update(training_seconds=1, samples_per_second=3200)
    name = "binding-throughput-M04-attempt-2"
    folder = tmp_path / name
    folder.mkdir()
    artifacts = [
        dict(id=1, name=name, expired=False, workflow_run={"id": 14, "head_sha": "candidate"})
    ]

    def selection():
        (folder / "report.json").write_text(json.dumps(data))
        report = ci.results(p, jobs(p), artifacts, tmp_path)
        return next(
            r
            for r in report["selections"]
            if r["binding"] == "M04" and r["scope"] == "extract" and r["group"] == "GPU training"
        )

    assert selection()["status"] == "executed"
    data["measurement"]["optimizer_updates"] -= 1
    assert "optimizer_updates" in selection()["error"]
    data["measurement"]["optimizer_updates"] += 1
    data["metrics"]["training_seconds"] += 1
    assert "aggregate metrics" in selection()["error"]


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
    next(j for j in raw if j["name"] == "Training · Synthetic S02 / GPU")["completed_at"] = (
        "2026-11-01T01:30:00Z"
    )
    report = ci.results(p, raw, [], tmp_path)
    assert report["training_wall_seconds"] == 5400
    for name, result in (("Publish results", "success"), ("kumo-pytorch-gate", "failure")):
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
