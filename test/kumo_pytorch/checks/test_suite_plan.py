# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The unified execution plan retains all obligations regardless of trigger."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from test.kumo_pytorch import nightly
from test.kumo_pytorch.stage_timing import StageTiming

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_full_plan_partitions_existing_binding_campaigns_without_loss():
    plan = nightly.execution_plan(240)
    grouped = [
        r
        for name in ("correctness", "training", "lifecycle", "population")
        for r in plan[name]["include"]
    ]
    expected = [
        r
        for name in ("training", "population", "lifecycle", "histories")
        for r in nightly.campaign_matrix(name)["include"]
    ]
    identity = lambda r: (r["binding"], r["mode"])
    assert len(grouped) == len(expected) == 42
    assert sorted(map(identity, grouped)) == sorted(map(identity, expected))
    assert len(set(map(identity, grouped))) == len(grouped)
    assert len(plan["fitting"]["include"]) == 15
    assert len(plan["benchmarks"]["include"]) == 4
    assert all(r["dataset"] == "li-medium" for r in plan["benchmarks"]["include"])
    assert {r["repeat"] for r in plan["fitting"]["include"]} == {0, 1, 2}
    assert [r["shard"] for r in plan["conformance"]["include"]] == [0, 1, 2]
    assert all(
        r["seed_start"] == 240 and r["seeds"] == 12
        for r in plan["conformance"]["include"]
    )


def test_all_triggers_resolve_identical_plan_without_native_imports(tmp_path):
    plans = []
    for event in ("pull_request", "schedule", "workflow_dispatch", "push"):
        path = tmp_path / f"{event}.json"
        subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "test.kumo_pytorch.nightly",
                "plan",
                "--seed-start",
                "240",
                "--output",
                str(path),
            ],
            cwd=ROOT,
            env={**os.environ, "GITHUB_EVENT_NAME": event},
            check=True,
        )
        plans.append(json.loads(path.read_text()))
    assert all(plan == plans[0] for plan in plans)


def test_large_binding_uses_shared_worker_and_checks_storage_before_download():
    plan = nightly.execution_plan()
    campaigns = [
        r
        for name in ("correctness", "training", "lifecycle", "population")
        for r in plan[name]["include"]
    ]
    large = [r for r in campaigns if r["binding"] in {"S01", "S02", "S04", "M04"}]
    assert {r["mode"] for r in large} == {"smoke", "training", "lifecycle", "full"}
    for row in large:
        assert json.loads(row["runner"]) == {
            "group": "nv-gpu-amd64-rtxpro6000-1gpu",
            "labels": "linux-amd64-gpu-rtxpro6000-latest-1",
        }
        assert row["min_free_gib"] == 128
    assert all(
        json.loads(r["runner"]) == "ubuntu-24.04"
        for r in campaigns
        if r["binding"] not in {"S01", "S02", "S04", "M04"}
    )
    workflow = (ROOT / ".github/workflows/_kumo-pytorch-binding.yml").read_text()
    assert "runs-on: ${{ fromJSON(inputs.runner) }}" in workflow
    assert "torch-extra: cpu" in workflow
    assert workflow.index("storage_preflight(root,") < workflow.index(
        "data fetch --dataset"
    )
    assert workflow.index("raise SystemExit('Insufficient free space") < workflow.index(
        "data fetch --dataset"
    )


def test_campaign_resource_errors_fail_during_planning():
    with pytest.raises(ValueError, match="Unknown campaign runner"):
        nightly.campaign_resources({"runner": "missing"})
    with pytest.raises(ValueError, match="min_free_gib"):
        nightly.campaign_resources({"min_free_gib": -1})


def test_only_full_suite_has_external_triggers():
    workflows = ROOT / ".github/workflows"
    suite = (workflows / "kumo-pytorch-t2b.yml").read_text()
    assert "name: PyTorch — Full Suite" in suite and "name: kumo-pytorch-gate" in suite
    for event in ("pull_request:", "schedule:", "workflow_dispatch:", "push:"):
        assert f"  {event}" in suite
    assert "github.event_name == 'schedule'" not in suite
    assert not (workflows / "kumo-pytorch-gate.yml").exists()
    for path in workflows.glob("*kumo-pytorch*.yml"):
        if path.name != "kumo-pytorch-t2b.yml":
            text = path.read_text()
            assert "  workflow_call:" in text
            assert all(
                f"  {event}" not in text
                for event in (
                    "pull_request:",
                    "schedule:",
                    "workflow_dispatch:",
                    "push:",
                )
            )


def test_conformance_shards_have_one_owner_and_keep_seed_caches_together():
    path = ROOT / "packages/experimental/kumo-pytorch/conformance/sharding.py"
    spec = importlib.util.spec_from_file_location("sharding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes = [
        f"tests/conformance/test_properties.py::test_property[{name}-diskgraph-seed{seed}]"
        for name in ("delivery", "point_in_time", "sequence_expected")
        for seed in (0, *range(240, 252))
    ]
    nodes += [
        f"tests/conformance/test_guarantees.py::test_public_behavior[C-{i}-diskgraph-answer]"
        for i in range(20)
    ]
    partitions = [
        {n for n in nodes if module.owner(n, 3) == shard} for shard in range(3)
    ]
    assert set.union(*partitions) == set(nodes)
    assert sum(map(len, partitions)) == len(nodes)
    for seed in (0, *range(240, 252)):
        assert (
            len({module.owner(n, 3) for n in nodes if n.endswith(f"-seed{seed}]")}) == 1
        )
    assert all(module.owner(n, 1) == 0 for n in nodes)


def test_stage_timing_preserves_failures_and_partial_evidence(monkeypatch):
    clock = iter((10.0, 12.0, 20.0, 23.0))
    monkeypatch.setattr(
        "test.kumo_pytorch.stage_timing.time.perf_counter", lambda: next(clock)
    )
    report, snapshots = {}, []
    timing = StageTiming(
        report, lambda: snapshots.append(json.loads(json.dumps(report)))
    )
    with timing.span("materialization", checkpoint=True):
        pass
    with (
        pytest.raises(ValueError, match="original failure"),
        timing.span("materialization", checkpoint=True),
    ):
        raise ValueError("original failure")
    assert report["stage_timings"]["materialization"] == {
        "seconds": 5.0,
        "calls": 2,
        "failures": 1,
    }
    assert snapshots[0]["active_timing_stage"] == "materialization"
    assert "active_timing_stage" not in snapshots[-1]
