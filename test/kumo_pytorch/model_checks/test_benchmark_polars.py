"""Timing adapter integration; no cross-engine batch-equality assertions."""

import json

import pytest

from test.kumo_pytorch.aml_benchmark import factory
from test.kumo_pytorch.benchmarking import run
from test.kumo_pytorch.benchmarking.campaign import compare_states

from .test_benchmark_aml import CASES, source  # noqa: F401 -- shared local source fixture


@pytest.mark.parametrize("workload,family", CASES)
def test_reference_timing_and_state_reuse(source, workload, family):  # noqa: F811 -- imported fixture
    pl = pytest.importorskip("polars")
    config = {
        "data_dir": str(source),
        "family": family,
        "workload": workload,
        "anchors_per_split": 1,
        "batch_size": 1,
        "threads": pl.thread_pool_size(),
        "backend": "polars",
        "training": False,
        "device": "cpu",
        "save_fitted_state": str(source / "state"),
    }
    fresh = run(
        factory,
        config,
        source / "fresh",
        repetitions=1,
        epochs=2,
        measurement_profile="timing",
    )
    assert fresh["execution_status"] == "completed", fresh["errors"]
    assert fresh["measurement_profile"] == "timing"
    for name in (
        "reference.load",
        "reference.fit",
        "reference.plan",
        "reference.batch",
    ):
        rows = fresh["measurements"][name + ".elapsed_ns"]
        assert rows and all(row["status"] == "available" for row in rows)
    assert not fresh["measurements"].get("training.step.elapsed_ns")
    assert not fresh["measurements"].get("index.build.elapsed_ns")
    coverage = json.loads((source / "fresh/coverage.json").read_text())["metrics"]
    assert {key for key, row in coverage.items() if row["status"] != "complete"} == {
        "later_batch_latency_p50_ns",
        "later_batch_latency_p95_ns",
    }
    config["load_fitted_state"] = config.pop("save_fitted_state")
    reused = run(
        factory,
        config,
        source / "reused",
        repetitions=1,
        epochs=2,
        measurement_profile="timing",
    )
    assert reused["execution_status"] == "completed", reused["errors"]
    assert compare_states(fresh, reused)["status"] == "measured"
    assert all(row["metrics"]["real_samples"] == 1 for row in reused["epochs"])
