# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_three_arm_controls_use_identical_engine_state_and_disjoint_inputs(tmp_path):
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    from benchmarks.ablations.graph_backends.benchmark import Settings, run

    result = run(
        Settings(
            entities=8,
            events_per_entity=4,
            transfers_per_entity=2,
            batch_size=4,
            width=8,
            depth=2,
            warmup=0,
            steps=1,
            device="cpu",
        ),
        tmp_path / "cost.json",
    )
    shared, disjoint, legacy = result["arms"]
    assert shared["status"] == disjoint["status"] == legacy["status"] == "ok"
    assert shared["initial_state_sha256"] == disjoint["initial_state_sha256"]
    assert shared["parameters"] == disjoint["parameters"]
    assert disjoint["nodes"] == legacy["nodes"] and disjoint["edges"] == legacy["edges"]
    assert shared["nodes"] < disjoint["nodes"]
    assert all(arm["samples"] == 4 and len(arm["step_ms"]) == 1 for arm in result["arms"])
    for arm in result["arms"]:
        assert arm["throughput"]["unit"] == "roots/s"
        assert arm["throughput"]["value"] == pytest.approx(
            arm["samples"] * 1000 / arm["step_ms_median"]
        )
        assert "transfer" in arm["throughput"]["scope"] and "excluded" in arm["throughput"]["scope"]


@pytest.mark.parametrize("missing_git", [False, True])
def test_cpu_benchmark_writes_results_with_source_provenance_from_outside_checkout(
    tmp_path, monkeypatch, missing_git
):
    import json

    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    from benchmarks.ablations.graph_backends import benchmark

    from tfm.runtime import common

    expected = common.source_provenance()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "foreign-nonexistent.git"))
    if missing_git:
        (tmp_path / "src/tfm").mkdir(parents=True)
        (tmp_path / "pyproject.toml").write_text("")
        monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
        expected = {"source_revision": None, "source_dirty": None}
    path = tmp_path / "retained.json"
    result = benchmark.run(
        benchmark.Settings(
            entities=8,
            events_per_entity=4,
            transfers_per_entity=2,
            batch_size=4,
            width=8,
            depth=2,
            warmup=0,
            steps=1,
            device="cpu",
        ),
        path,
    )
    for key, value in expected.items():
        assert result[key] == value
    assert len(result["arms"]) == 3
    assert json.loads(path.read_text())["arms"] == result["arms"]


def test_all_cost_drivers_use_shared_unknown_provenance(tmp_path, monkeypatch):
    from benchmarks.ablations.graph_backends import benchmark, execution, views

    from tfm.runtime import common

    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    for driver in (benchmark, execution, views):
        assert driver.source_provenance() == {"source_revision": None, "source_dirty": None}


def test_cuda_drivers_keep_results_without_git_and_report_resources(tmp_path, monkeypatch):
    import argparse
    import json

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    from benchmarks.ablations.graph_backends import conversion, execution
    from benchmarks.ablations.graph_backends.benchmark import Settings

    from tfm.runtime import common

    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    transferred = execution.run(
        Settings(
            entities=8,
            events_per_entity=4,
            transfers_per_entity=2,
            batch_size=4,
            width=8,
            depth=2,
            warmup=0,
            steps=1,
            device="cuda",
        ),
        tmp_path / "transfer.json",
    )
    assert transferred["source_revision"] is None and len(transferred["arms"]) == 2
    for arm in transferred["arms"]:
        assert arm["throughput"]["unit"] == "roots/s"
        assert arm["throughput"]["value"] == pytest.approx(
            len(arm["final_prediction"]) * 1000 / arm["transfer_and_train_ms_median"]
        )
        assert arm["throughput"]["scope"].startswith("blocking batch transfer")
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        result = conversion.run(argparse.Namespace(warmup=0, steps=1, deterministic=False))
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
    assert result["source_revision"] is None and result["source_dirty"] is None
    for arm in result["results"]:
        assert arm["throughput"]["unit"] == "roots/s"
        assert arm["throughput"]["value"] == pytest.approx(
            arm["effective_settings"]["root_count"] * 1000 / arm["median_ms"]["cycle_ms"]
        )
        assert arm["peak_memory"]["allocated_mib"] > 0
        assert arm["peak_memory"]["reserved_mib"] >= arm["peak_memory"]["allocated_mib"]
        assert arm["final_state_sha256"] and arm["final_gradients_sha256"]
    (tmp_path / "resources.json").write_text(json.dumps(result))
