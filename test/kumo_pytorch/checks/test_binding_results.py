# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Required LI reports must describe the requested execution, not just green jobs."""

import json
import subprocess
import sys
from test.kumo_pytorch import nightly as binding_results
from test.kumo_pytorch.bindings import (
    load_bindings,
    selected_bindings,
    task_definition,
)
from test.kumo_pytorch.data import ROOT, load_manifest

import pytest


def report(ref, mode):
    binding = load_bindings()[ref]
    targets = binding["targets"] or {task_definition(binding)["kind"]: {}}
    return {
        **{k: binding[k] for k in ("ref", "binding_id", "selector")},
        "feature_coverage": {
            name: dict.fromkeys(("train", "valid", "test"), 3)
            for name in binding.get("feature_checks", {}).get("features", ())
        },
        "family": binding["representation"],
        "revision": "fixture-revision",
        "dirty_checkout": False,
        "mode": mode,
        "status": "passed",
        "stage": "complete",
        "full_population_verified": mode == "full",
        "population": {"mode": "full" if mode == "full" else "smoke"},
        "mapping": {"scope": {"kind": "full_source"}},
        "splits": {
            s: {
                "status": "passed",
                "delivered": 3,
                "expected": 3,
                "eligible_targets": dict.fromkeys(targets, 2),
            }
            for s in ("train", "valid", "test")
        },
        "reproducible": True,
        "repeats": [
            {
                "status": "passed",
                "completed_updates": 100,
                "final_model_sha256": "model",
                "checks": dict.fromkeys(
                    (
                        "100_updates",
                        "every_parameter_has_nonzero_gradient",
                        "every_parameter_updated",
                        "loss_decreased",
                    ),
                    True,
                ),
            }
            for _ in range(2)
        ],
        "evaluation_model_sha256": "model",
        "evaluation": {
            s: {t: {"eligible": 2, "mean_loss": 0.5} for t in targets}
            for s in ("valid", "test")
        },
    }


def reports(root):
    for binding in selected_bindings():
        ref = binding["ref"]
        for mode in (
            m for m in ("full", "training", "smoke") if m in binding["campaigns"]
        ):
            path = root / mode / ref / "report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(report(ref, mode)))


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "failed",
        "stale",
        "binding",
        "bounded",
        "source-filter",
        "dropped",
        "missing-target",
        "empty-target",
        "training",
        "unchecked",
        "evaluation",
        "checkpoint",
        "nonfinite",
        "dirty",
        "bare-refusal",
    ),
)
def test_li_summary_rejects_incomplete_or_wrong_evidence(tmp_path, fault):
    reports(tmp_path)
    assert (
        binding_results.summarize_bindings(tmp_path, "fixture-revision")["status"]
        == "passed"
    )
    mode = (
        "training"
        if fault in {"training", "unchecked", "evaluation", "checkpoint", "nonfinite"}
        else "full"
    )
    path = tmp_path / mode / "L01/report.json"
    data = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "failed":
        data["status"] = "failed"
    elif fault == "stale":
        data["revision"] = "old"
    elif fault == "binding":
        data["binding_id"] = load_bindings()["L02"]["binding_id"]
    elif fault == "bounded":
        data["population"]["mode"] = "smoke"
    elif fault == "source-filter":
        data["mapping"]["scope"]["kind"] = "selected_root_full_histories"
    elif fault == "dropped":
        data["splits"]["test"]["delivered"] -= 1
    elif fault == "missing-target":
        data["splits"]["test"]["eligible_targets"].pop("amount")
    elif fault == "empty-target":
        data["splits"]["test"]["eligible_targets"]["amount"] = 0
    elif fault == "training":
        data["repeats"][0]["completed_updates"] = 99
    elif fault == "unchecked":
        data["repeats"][0]["checks"] = {}
    elif fault == "evaluation":
        data["evaluation"]["test"]["amount"]["eligible"] = 1
    elif fault == "checkpoint":
        data["evaluation_model_sha256"] = "untrained"
    elif fault == "nonfinite":
        data["evaluation"]["test"]["amount"]["mean_loss"] = float("nan")
    elif fault == "dirty":
        data["dirty_checkout"] = True
    elif fault == "bare-refusal":
        data.update(
            outcome="refused",
            stage="prepare",
            full_population_verified=False,
            splits={},
            refusal={},
        )
    if fault != "missing":
        path.write_text(json.dumps(data))
    result = binding_results.summarize_bindings(tmp_path, "fixture-revision")
    assert result["status"] == "failed"
    assert len(result["cases"]) == 24
    assert sum(row["status"] == "failed" for row in result["cases"]) == 1


def test_li_summary_preserves_structured_refusal(tmp_path):
    reports(tmp_path)
    path = tmp_path / "full/L02/report.json"
    data = json.loads(path.read_text())
    data.update(
        outcome="refused",
        stage="prepare",
        full_population_verified=False,
        splits={},
        refusal={
            "outcome": "refused",
            "engine": "diskgraph",
            "stage": "preflight",
            "execution": "not_started",
            "issues": [
                {
                    "code": "DECLARED",
                    "path": "batch.x",
                    "reason": "unsupported",
                    "suggestion": "Use supported input",
                }
            ],
        },
    )
    path.write_text(json.dumps(data))
    result = binding_results.summarize_bindings(tmp_path, "fixture-revision")
    assert result["status"] == "passed"
    assert sum(row["outcome"] == "refused" for row in result["cases"]) == 1
    data["refusal"]["stage"] = "iteration"
    path.write_text(json.dumps(data))
    assert (
        binding_results.summarize_bindings(tmp_path, "fixture-revision")["status"]
        == "failed"
    )


def test_li_summary_cli_needs_no_installed_package_and_writes_failures(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "test.kumo_pytorch.nightly",
            "summary",
            "--directory",
            str(tmp_path),
            "--revision",
            "fixture",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert len(json.loads((tmp_path / "report.json").read_text())["shards"]) == 54
    assert (tmp_path / "report.md").is_file()


def test_li_policy_grants_only_additional_pinned_objects():
    manifest = load_manifest()
    existing = json.loads((ROOT / "datasets/ci-read-policy.json").read_text())[
        "Statement"
    ][0]
    new = json.loads((ROOT / "datasets/ci-li-medium-read-policy.json").read_text())[
        "Statement"
    ][0]
    expected = {
        f"arn:aws:s3:::{manifest['bucket']}/{item['key']}"
        for item in manifest["datasets"]["li-medium"]["files"]
    } - set(existing["Resource"])
    assert set(new["Resource"]) == expected and len(new["Resource"]) == 10
    assert new["Action"] == "s3:GetObject" and new["Effect"] == "Allow"
    assert new["Principal"] == existing["Principal"]
    assert all("*" not in resource for resource in new["Resource"])


def test_jobs_cover_all_bindings_and_retain_reports():
    workflow = (ROOT.parents[1] / ".github/workflows/kumo-pytorch-t2b.yml").read_text()
    job = workflow.split("  dataset-campaigns:\n", 1)[1].split("  aml-lifecycle:\n", 1)[
        0
    ]
    planned = (
        binding_results.campaign_matrix("training")["include"]
        + binding_results.campaign_matrix("population")["include"]
    )
    assert len(planned) == 24
    assert len({(r["binding"], r["mode"]) for r in planned}) == 24
    assert {r["dataset"] for r in planned} == {"hi-small", "li-medium"}
    assert "matrix: ${{ fromJSON(needs.plan.outputs.dataset_campaigns) }}" in job
    assert "fail-fast: false" in job
    assert '--dataset "$DATASET"' in job and "--anchors-per-split" not in job
    assert "python -m test.kumo_pytorch.nightly run" in job
    assert (
        "torch-extra: cpu" in job and "diskgraph-wheel-artifact: diskgraph-wheel" in job
    )
    assert "report.json" in job and "loss-curve.csv" in job
    assert "pattern: kumo-pytorch-dataset-*" in workflow
    assert "python3 -m test.kumo_pytorch.nightly summary" in workflow
    assert "aml-li-medium:" not in workflow


@pytest.mark.parametrize(
    "fault", ("absent", "missing-feature", "unchecked", "missing-split")
)
def test_required_binding_features_cannot_disappear(tmp_path, fault):
    reports(tmp_path)
    path = tmp_path / "smoke/L01/report.json"
    data = json.loads(path.read_text())
    if fault == "absent":
        data.pop("feature_coverage")
    elif fault == "missing-feature":
        data["feature_coverage"].pop("F04")
    elif fault == "unchecked":
        data["feature_coverage"]["F08"]["test"] = 0
    else:
        data["feature_coverage"]["F14"].pop("valid")
    path.write_text(json.dumps(data))
    result = binding_results.summarize_bindings(tmp_path, "fixture-revision")
    assert result["status"] == "failed"
    assert sum(row["status"] == "failed" for row in result["cases"]) == 1
