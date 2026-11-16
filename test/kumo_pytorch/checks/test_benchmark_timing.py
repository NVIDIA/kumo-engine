"""Timing scope and summaries must not silently certify the full metric suite."""

import json

import pytest

from test.kumo_pytorch.benchmarking import run
from test.kumo_pytorch.benchmarking.metrics import METRICS, TIMING_METRICS
from test.kumo_pytorch.benchmarking.timing_report import timing_summary

from .test_benchmark import graph, tabular


def test_timing_scope_is_explicit_and_excludes_memory(tmp_path):
    result = run(
        tabular,
        {},
        tmp_path / "run",
        repetitions=1,
        epochs=2,
        measurement_profile="timing",
        memory_cgroup=tmp_path / "absent",
    )
    assert result["execution_status"] == "completed"
    assert (
        result["status"] == "incomplete"
    )  # Missing preparation timers still fail coverage.
    assert result["measurement_profile"] == "timing"
    assert result["metric_coverage"]["total"] == len(TIMING_METRICS) < len(METRICS)
    assert not any(key.endswith("peak_host_bytes") for key in result["measurements"])
    coverage = json.loads((tmp_path / "run/coverage.json").read_text())["metrics"]
    assert coverage["ingestion.elapsed_ns"]["status"] == "incomplete"
    assert "gpu.utilization_percent" not in coverage


def test_timing_scope_rejects_training(tmp_path):
    result = run(
        graph,
        {},
        tmp_path / "run",
        repetitions=1,
        epochs=1,
        measurement_profile="timing",
    )
    assert result["execution_status"] == "failed"
    assert "CPU loader-only" in result["errors"][0]["traceback"]


def test_unknown_profile_creates_no_output(tmp_path):
    with pytest.raises(ValueError, match="measurement_profile"):
        run(tabular, {}, tmp_path / "run", measurement_profile="typo")
    assert not (tmp_path / "run").exists()


def test_timing_summary_separates_epochs_fit_modes_and_excludes_failures(tmp_path):
    directory = tmp_path / "example"
    directory.mkdir()
    runs = []
    for name, mode, good, value in (
        ("fresh-0", "fresh", True, 10),
        ("fresh-1", "fresh", True, 20),
        ("fresh-failed", "fresh", False, 900),
        ("reused-0", "reused", True, 3),
    ):
        (directory / name).mkdir()
        runs.append({"name": name, "mode": mode, "execution_completed": good})
        measurements = {
            "materialization.elapsed_ns": [
                {"status": "available", "value": value, "scope": {}}
            ],
            "batch_production.elapsed_ns": [
                {
                    "status": "available",
                    "value": value * (epoch + 1),
                    "scope": {"epoch": epoch},
                }
                for epoch in (0, 1)
            ],
        }
        (directory / name / "results.json").write_text(
            json.dumps(
                {
                    "epochs": [{"scope": {"epoch": e}} for e in (0, 1)],
                    "measurements": measurements,
                }
            )
        )
    (directory / "results.json").write_text(
        json.dumps({"status": "failed", "runs": runs})
    )
    lines = timing_summary([{"name": "example"}], tmp_path)
    rows = json.loads((tmp_path / "timings.json").read_text())["rows"]
    assert len(rows) == 6
    prep, first, later = rows[:3]
    assert prep["metrics"]["materialization.elapsed_ns"]["values"] == [10, 20]
    assert prep["metrics"]["materialization.elapsed_ns"]["median"] == 15
    assert prep["successful_runs"] == 2 and prep["expected_runs"] == 3
    assert first["metrics"]["batch_production.elapsed_ns"]["median"] == 15
    assert later["metrics"]["batch_production.elapsed_ns"]["median"] == 30
    assert rows[3]["metrics"]["materialization.elapsed_ns"]["median"] == 3
    assert all(row["status"] == "failed" for row in rows)
    assert "example/report.md" in "\n".join(lines)
