# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel execution must retain every binding, experiment and repetition."""

import json
import subprocess
import sys
from test.kumo_pytorch import nightly

import pytest

from .test_binding_results import reports as binding_reports
from .test_fitting_memory import runs


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fitting_reports(root, case):
    for repeat in range(3):
        selected = [r for r in runs() if r["repeat"] == repeat]
        write(
            root / "fitting" / case / f"repeat-{repeat}/report.json",
            {
                "status": "passed",
                "selected_repeat": repeat,
                "repetitions": 3,
                "cases": [
                    {
                        "case": case,
                        "status": "passed",
                        "selected_repeat": repeat,
                        "runs": selected,
                    }
                ],
            },
        )


def reports(root):
    binding_reports(root)
    for row in nightly.matrix(campaign="fitting")["include"]:
        fitting_reports(root, row["case"])
    for row in nightly.benchmark_matrix()["include"]:
        case, mode = row["case"], row["mode"]
        base = root / mode / case
        write(
            base / "results.json",
            {"status": "complete", "cases": [{"name": case, "status": "complete"}]},
        )
        write(base / case / "results.json", {"status": "complete", "runs": []})


def test_plan_has_every_binding_and_no_native_import_requirement():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "test.kumo_pytorch.nightly", "matrix"],
        capture_output=True,
        text=True,
        check=True,
    )
    names = [r["case"] for r in json.loads(result.stdout)["include"]]
    assert set(names) == {
        "account-involvement-next24h-graphsage",
        "account-involvement-next24h-combined",
        "account-involvement-next24h-nuformer",
        "masked-value-pretrain-pragma",
        "next-token-pretrain-nuformer",
        "li-medium-combined",
    }
    assert len(names) == 6
    with pytest.raises(ValueError, match="Unknown nightly case"):
        nightly.selected_cases("not-a-binding")


@pytest.mark.parametrize("mode", nightly.MODES)
def test_benchmark_shards_preserve_limits_and_training_scope(mode, tmp_path):
    template = json.loads((nightly.ROOT / "workloads/aml-benchmark.json").read_text())
    for row in nightly.matrix()["include"]:
        case = row["case"]
        enabled = {
            (r["case"], r["mode"]) for r in nightly.benchmark_matrix()["include"]
        }
        if (case, mode) not in enabled:
            with pytest.raises(ValueError, match="does not enable"):
                nightly.benchmark_config(case, mode, tmp_path)
            continue
        config = nightly.benchmark_config(case, mode, tmp_path)
        assert len(config) == 1 and config[0]["name"] == case
        selected = config[0]["config"]
        for name in ("anchors_per_split", "batch_size", "threads", "seed"):
            assert selected[name] == template[name]
        if mode.startswith("timing-"):
            assert selected["backend"] == mode.removeprefix("timing-")
            assert selected["training"] is False
            assert selected["cache_mode"] == "uncontrolled"
        elif mode == "length-planning":
            assert selected["length_planning"] is True and selected["training"] is False
        else:
            assert selected["training"] is True
            assert selected["cache_mode"] == "cold-warm"


def test_summary_recombines_all_experiments_and_fitting_repetitions(tmp_path):
    reports(tmp_path)
    result = nightly.summarize(tmp_path, "fixture-revision")
    assert result["status"] == "passed" and len(result["shards"]) == 49
    case = nightly.matrix()["include"][0]["case"]
    combined = json.loads((tmp_path / "fitting" / case / "report.json").read_text())
    assert len(combined["cases"][0]["runs"]) == 6
    comparison = combined["cases"][0]["comparison"]
    assert comparison["repetitions"] == 3 and comparison["growth_ratio"] == 1.9
    assert (tmp_path / "timings.json").exists()


@pytest.mark.parametrize("fault", [None, "missing-reason", "late", "delivered"])
@pytest.mark.parametrize("empty", [False, True])
def test_summary_retains_refusals_without_claiming_population(tmp_path, fault, empty):
    reports(tmp_path)
    path = tmp_path / "full" / "E01" / "report.json"
    report = json.loads(path.read_text())
    report["full_population_verified"] = False
    selected = report
    selected.update(
        stage="prepare",
        outcome="refused",
        full_population_verified=False,
        splits={"train": {"status": "not_executed", "delivered": 0}},
        refusal={
            "outcome": "refused",
            "engine": "diskgraph",
            "stage": "preflight",
            "execution": "not_started",
            "issues": [
                {
                    "code": "DG-ROW-FILTER",
                    "path": "sample.entity.filters",
                    "reason": "DiskGraph does not support entity row filters",
                    "suggestion": "Remove the filter",
                }
            ],
        },
    )
    if empty:
        selected["refusal"].update(
            stage="ingestion",
            execution="stopped_before_materialization",
            engine="diskgraph",
            source_rows=0,
        )
        selected["refusal"]["issues"][0].update(
            code="DG-EMPTY-BASE-TABLE",
            path="dataset.tables.items",
            reason="Ingest failed: graph source returned zero rows for table items",
            suggestion="Use nonempty base tables",
        )
    if fault == "missing-reason":
        del selected["refusal"]["issues"][0]["reason"]
    elif fault == "late":
        selected["refusal"]["execution"] = "started"
    elif fault == "delivered":
        selected["splits"]["train"]["delivered"] = 1
    write(path, report)
    result = nightly.summarize(tmp_path, "fixture-revision")
    assert result["status"] == ("passed" if fault is None else "failed")
    if fault is None:
        row = next(
            row
            for row in result["shards"]
            if row["case"] == "E01" and row["kind"] == "full"
        )
        assert row["outcome"] == "refused" and row["refusal"] == selected["refusal"]
        assert "passed (refused)" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize(
    "fault",
    [
        "missing-benchmark",
        "wrong-binding",
        "missing-population",
        "partial-population",
        "missing-repeat",
        "duplicate-repeat",
        "different-spec",
    ],
)
def test_summary_rejects_lost_or_substituted_shards(tmp_path, fault):
    reports(tmp_path)
    case = nightly.matrix()["include"][0]["case"]
    benchmark = nightly.benchmark_matrix()["include"][0]["case"]
    if fault == "missing-benchmark":
        (tmp_path / "training" / benchmark / "results.json").unlink()
    elif fault == "wrong-binding":
        write(
            tmp_path / "training" / benchmark / "results.json",
            {
                "status": "complete",
                "cases": [{"name": "another-case"}],
            },
        )
    elif fault == "missing-population":
        (tmp_path / "full" / "E01" / "report.json").unlink()
    elif fault == "partial-population":
        path = tmp_path / "full" / "E01" / "report.json"
        report = json.loads(path.read_text())
        report["full_population_verified"] = False
        write(path, report)
    else:
        path = tmp_path / "fitting" / case / "repeat-2/report.json"
        if fault == "missing-repeat":
            path.unlink()
        else:
            report = json.loads(path.read_text())
            if fault == "duplicate-repeat":
                report["cases"][0]["runs"][0]["repeat"] = 0
            else:
                report["cases"][0]["runs"][0]["spec_sha256"] = "changed-inputs"
            write(path, report)
    result = nightly.summarize(tmp_path, "fixture-revision")
    assert result["status"] == "failed"
    assert len(result["shards"]) == 49  # Failure never suppresses other reports.


def test_li_campaign_scope_does_not_enable_reference_backends_or_fitting():
    assert {
        row["mode"]
        for row in nightly.benchmark_matrix()["include"]
        if row["case"] == "li-medium-combined"
    } == {"training", "index", "timing-diskgraph", "length-planning"}
    assert "li-medium-combined" not in {
        row["case"] for row in nightly.matrix(campaign="fitting")["include"]
    }
    assert len(nightly.benchmark_matrix()["include"]) == 4


@pytest.mark.parametrize(
    "kind",
    [
        "training",
        "index",
        "timing-diskgraph",
        "length-planning",
        "lifecycle",
        "histories",
    ],
)
def test_summary_requires_each_new_campaign(tmp_path, kind):
    reports(tmp_path)
    case = (
        "li-medium-combined"
        if kind in {"training", "index", "timing-diskgraph", "length-planning"}
        else "L01"
    )
    name = "results.json" if case == "li-medium-combined" else "report.json"
    (tmp_path / kind / case / name).unlink()
    result = nightly.summarize(tmp_path, "fixture-revision")
    assert result["status"] == "failed"
    assert [
        (row["case"], row["kind"])
        for row in result["shards"]
        if row["status"] == "failed"
    ] == [(case, kind)]


def test_loader_only_binding_does_not_require_model_training(tmp_path, monkeypatch):
    from copy import deepcopy
    from test.kumo_pytorch import bindings

    catalog = deepcopy(bindings._catalog())
    selected = next(b for b in catalog["bindings"] if b["ref"] == "L01")
    selected["campaigns"].remove("benchmark-training")
    monkeypatch.setattr(bindings, "_catalog", lambda: catalog)
    assert {
        r["mode"]
        for r in nightly.benchmark_matrix()["include"]
        if r["case"] == "li-medium-combined"
    } == {"index", "timing-diskgraph", "length-planning"}
    assert (
        nightly.benchmark_config("li-medium-combined", "timing-diskgraph", tmp_path)[0][
            "config"
        ]["training"]
        is False
    )
    reports(tmp_path)
    assert nightly.summarize(tmp_path, "fixture-revision")["status"] == "passed"
    (tmp_path / "timing-diskgraph/li-medium-combined/results.json").unlink()
    assert nightly.summarize(tmp_path, "fixture-revision")["status"] == "failed"


def test_fitting_only_binding_still_requires_its_repetitions(tmp_path, monkeypatch):
    from copy import deepcopy
    from test.kumo_pytorch import bindings

    catalog = deepcopy(bindings._catalog())
    next(b for b in catalog["bindings"] if b["ref"] == "L01")["campaigns"] = ["fitting"]
    monkeypatch.setattr(bindings, "_catalog", lambda: catalog)
    assert all(
        r["case"] != "li-medium-combined" for r in nightly.benchmark_matrix()["include"]
    )
    reports(tmp_path)
    assert nightly.summarize(tmp_path, "fixture-revision")["status"] == "passed"
    (tmp_path / "fitting/li-medium-combined/repeat-2/report.json").unlink()
    assert nightly.summarize(tmp_path, "fixture-revision")["status"] == "failed"
