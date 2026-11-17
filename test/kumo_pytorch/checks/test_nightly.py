"""Parallel execution must retain every binding, experiment and repetition."""

import json
import subprocess
import sys

import pytest

from test.kumo_pytorch import nightly

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
    for row in nightly.matrix()["include"]:
        case = row["case"]
        selection, family = nightly.selected_cases(case)[0]
        fitting_reports(root, case)
        for mode in nightly.MODES:
            base = root / mode / case
            write(
                base / "results.json",
                {
                    "status": "complete",
                    "cases": [{"name": case, "status": "complete"}],
                },
            )
            write(base / case / "results.json", {"status": "complete", "runs": []})
        for mode in ("smoke", "full"):
            write(
                root / mode / case / "report.json",
                {
                    "status": "passed",
                    "full_population_verified": mode == "full",
                    "cases": [
                        {
                            "selector": selection["selector"],
                            "family": family,
                            "status": "passed",
                        }
                    ],
                },
            )


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
    }
    assert len(names) == 5
    with pytest.raises(ValueError, match="Unknown nightly case"):
        nightly.selected_cases("not-a-binding")


@pytest.mark.parametrize("mode", nightly.MODES)
def test_benchmark_shards_preserve_limits_and_reference_scope(mode, tmp_path):
    template = json.loads((nightly.ROOT / "workloads/aml-benchmark.json").read_text())
    for row in nightly.matrix()["include"]:
        case = row["case"]
        config = nightly.benchmark_config(case, mode, tmp_path)
        assert len(config) == 1 and config[0]["name"] == case
        selected = config[0]["config"]
        for name in ("anchors_per_split", "batch_size", "threads", "seed"):
            assert selected[name] == template[name]
        if mode.startswith("timing-"):
            assert selected["backend"] == mode.removeprefix("timing-")
            assert selected["training"] is False
            assert selected["cache_mode"] == "uncontrolled"
        else:
            assert selected["training"] is True
            assert selected["cache_mode"] == "cold-warm"


def test_summary_recombines_all_experiments_and_fitting_repetitions(tmp_path):
    reports(tmp_path)
    result = nightly.summarize(tmp_path)
    assert result["status"] == "passed" and len(result["shards"]) == 40
    case = nightly.matrix()["include"][0]["case"]
    combined = json.loads((tmp_path / "fitting" / case / "report.json").read_text())
    assert len(combined["cases"][0]["runs"]) == 6
    comparison = combined["cases"][0]["comparison"]
    assert comparison["repetitions"] == 3 and comparison["growth_ratio"] == 1.9
    assert (tmp_path / "timings.json").exists()


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
    if fault == "missing-benchmark":
        (tmp_path / "training" / case / "results.json").unlink()
    elif fault == "wrong-binding":
        write(
            tmp_path / "training" / case / "results.json",
            {
                "status": "complete",
                "cases": [{"name": "another-case"}],
            },
        )
    elif fault == "missing-population":
        (tmp_path / "full" / case / "report.json").unlink()
    elif fault == "partial-population":
        path = tmp_path / "full" / case / "report.json"
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
    result = nightly.summarize(tmp_path)
    assert result["status"] == "failed"
    assert len(result["shards"]) == 40  # Failure never suppresses other reports.
