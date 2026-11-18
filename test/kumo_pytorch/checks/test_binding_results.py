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
        "feature_assertions": {
            name: {
                **declaration,
                "status": "passed",
                "checked_anchors": {
                    "population": dict.fromkeys(("train", "valid", "test"), 3),
                    "training": {
                        "train": 600 if mode == "training" else 0,
                        "valid": 0,
                        "test": 0,
                    },
                    "evaluation": {
                        "train": 0,
                        "valid": 3 if mode == "training" else 0,
                        "test": 3 if mode == "training" else 0,
                    },
                },
            }
            for name, declaration in binding.get("feature_checks", {})
            .get("assertions", {})
            .items()
            if mode in declaration.get("modes", binding["feature_checks"]["modes"])
        },
        "feature_coverage": {
            name: dict.fromkeys(("train", "valid", "test"), 3)
            for name in binding.get("feature_checks", {}).get("features", ())
        },
        "lifecycle": {
            "status": "passed",
            "checks": {
                "inference": {
                    "status": "passed",
                    "no_refitting": True,
                    "targets_absent": True,
                    "pid": 2,
                    "parent_pid": 1,
                },
                "finetuning": {
                    "status": "passed",
                    "input_refitting": False,
                    "resume": {"status": "passed"},
                    "pid": 2,
                    "parent_pid": 1,
                },
                "mid_epoch_resume": {
                    "status": "passed",
                    "remaining": [{"sample_ids": ["tr-c"]}],
                    "pid": 2,
                    "parent_pid": 1,
                },
            },
        },
        "histories": {
            "status": "passed",
            "cases": [
                {
                    "name": name,
                    "status": "passed",
                    "samples": 9,
                    "events_by_input": {
                        f"{keep}_{order}": 9
                        for keep in ("all", "first", "last")
                        for order in ("ascending", "descending")
                    },
                }
                for name in ("windows", "filters-and-windows")
            ],
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
                "observed_train_anchors": 300,
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
            m
            for m in ("full", "training", "smoke", "lifecycle", "histories")
            if m in binding["campaigns"]
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
    assert len(result["cases"]) == 30
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
    assert len(json.loads((tmp_path / "report.json").read_text())["shards"]) == 39
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
    job = (ROOT.parents[1] / ".github/workflows/_kumo-pytorch-binding.yml").read_text()
    planned = (
        binding_results.campaign_matrix("training")["include"]
        + binding_results.campaign_matrix("population")["include"]
        + binding_results.campaign_matrix("lifecycle")["include"]
        + binding_results.campaign_matrix("histories")["include"]
    )
    assert len(planned) == 30
    assert len({(r["binding"], r["mode"]) for r in planned}) == 30
    assert {r["dataset"] for r in planned} == {
        "hi-small",
        "li-medium",
        "synthetic-medium-customers",
    }
    for group in ("correctness", "training", "lifecycle", "population"):
        assert f"matrix: ${{{{ fromJSON(needs.plan.outputs.{group}) }}}}" in workflow
    assert "fail-fast: false" in workflow
    assert '--dataset "$DATASET"' in job and "--anchors-per-split" not in job
    assert "python -m test.kumo_pytorch.nightly run" in job
    assert (
        "torch-extra: cpu" in job
        and "diskgraph-wheel-artifact: ${{ inputs.diskgraph-wheel-artifact }}" in job
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


@pytest.mark.parametrize("mode", ("smoke", "training"))
@pytest.mark.parametrize(
    "fault",
    (
        "absent",
        "missing",
        "failed",
        "meaning",
        "unchecked",
        "phase",
        "partial-training",
    ),
)
def test_individual_feature_evidence_is_required(tmp_path, mode, fault):
    reports(tmp_path)
    path = tmp_path / mode / "L01/report.json"
    data = json.loads(path.read_text())
    assertions = data["feature_assertions"]
    if fault == "absent":
        data.pop("feature_assertions")
    elif fault == "missing":
        assertions.pop("graph_membership")
    elif fault == "failed":
        assertions["graph_membership"]["status"] = "failed"
    elif fault == "meaning":
        assertions["graph_membership"]["reference"] = "another engine"
    elif fault == "unchecked":
        assertions["aggregate_counts"]["checked_anchors"]["population"]["valid"] -= 1
    elif fault == "phase":
        assertions["history_values"]["checked_anchors"]["training"]["train"] = (
            0 if mode == "training" else 1
        )
    else:
        assertions["graph_values"]["checked_anchors"]["training"]["train"] = 599
    path.write_text(json.dumps(data))
    result = binding_results.summarize_bindings(tmp_path, "fixture-revision")
    assert result["status"] == "failed"
    assert sum(row["status"] == "failed" for row in result["cases"]) == 1


def test_declared_feature_assertions_require_handlers_and_feature_mapping():
    from test.kumo_pytorch.feature_checks import (
        FeatureChecks,
        required_assertions,
    )

    requirements = load_bindings()["L01"]["feature_checks"]
    declarations = required_assertions(requirements, "smoke")
    checks = {name: lambda batch: None for name in declarations}
    checks.pop("graph_membership")
    with pytest.raises(AssertionError, match="Missing or unregistered"):
        FeatureChecks(requirements, "smoke", checks, {})
    requirements["features"].append("unmapped-feature")
    with pytest.raises(AssertionError, match="Every required feature"):
        required_assertions(requirements, "smoke")


@pytest.mark.parametrize(
    "mode,fault",
    [
        ("lifecycle", "missing-check"),
        ("lifecycle", "refitting"),
        ("lifecycle", "same-process"),
        ("histories", "missing-probe"),
        ("histories", "late-refusal"),
    ],
)
def test_new_campaigns_require_actual_evidence(tmp_path, mode, fault):
    reports(tmp_path)
    path = tmp_path / mode / "L01/report.json"
    data = json.loads(path.read_text())
    if fault == "missing-check":
        data[mode]["checks"].pop("mid_epoch_resume")
    elif fault == "refitting":
        data[mode]["checks"]["finetuning"]["input_refitting"] = True
    elif fault == "same-process":
        data[mode]["checks"]["inference"]["pid"] = 1
    elif fault == "missing-probe":
        data[mode]["cases"].pop()
    else:
        data[mode]["cases"][1].update(outcome="refused", stage="iteration", refusal={})
    path.write_text(json.dumps(data))
    result = binding_results.summarize_bindings(tmp_path, "fixture-revision")
    assert result["status"] == "failed"
    assert sum(r["status"] == "failed" for r in result["cases"]) == 1
