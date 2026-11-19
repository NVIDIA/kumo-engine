# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The unified execution plan retains all obligations regardless of trigger."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from test.kumo_pytorch import nightly
from test.kumo_pytorch.stage_timing import StageTiming

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def reporting_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_WORKSPACE", str(ROOT))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("KUMO_CI_LABEL", "Workflow command check")
    monkeypatch.setenv("KUMO_CI_REPORTS", "")


def test_full_plan_partitions_existing_binding_campaigns_without_loss():
    plan = nightly.execution_plan(240)
    grouped = [
        r
        for name in ("correctness", "training", "lifecycle", "population")
        for r in plan[name]["include"]
        if r["executor"] == "binding"
    ]
    expected = [
        r
        for name in ("training", "population", "lifecycle", "histories")
        for r in nightly.campaign_matrix(name)["include"]
    ]
    identity = lambda r: (r["binding"], r["mode"])
    assert len(grouped) == len(expected) == 40
    assert all(r["binding"] != "T01" for r in grouped)
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


@pytest.mark.parametrize("campaign", ("all", "smoke", "training", "lifecycle", "full"))
def test_manual_plan_retains_only_requested_t01_campaigns(campaign):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "test.kumo_pytorch.nightly",
            "matrix",
            "--on-demand",
            "--campaign",
            campaign,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)["include"]
    expected = (
        {"smoke", "training", "lifecycle", "full"} if campaign == "all" else {campaign}
    )
    assert {(r["binding"], r["mode"]) for r in rows} == {
        ("T01", mode) for mode in expected
    }
    assert len(rows) == len(expected)
    assert all(r["dataset"] == "streaming-25b" for r in rows)


def test_manual_workflow_publishes_the_shared_matrix(tmp_path):
    import textwrap

    workflow = (ROOT / ".github/workflows/kumo-pytorch-on-demand.yml").read_text()
    command = workflow.split("        run: |\n", 1)[1].split("      - uses:", 1)[0]
    output = tmp_path / "github-output"
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", textwrap.dedent(command)],
        cwd=ROOT,
        env={**os.environ, "CAMPAIGN": "all", "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
        text=True,
    )
    key, matrix = output.read_text().strip().split("=", 1)
    assert key == "matrix"
    assert json.loads(matrix) == nightly.campaign_matrix("all", on_demand=True)
    assert json.loads((tmp_path / "on-demand-plan.json").read_text()) == json.loads(
        matrix
    )


def test_large_binding_uses_shared_worker_and_checks_storage_before_download():
    plan = nightly.execution_plan()
    campaigns = [
        r
        for name in ("correctness", "training", "lifecycle", "population")
        for r in plan[name]["include"]
    ] + nightly.campaign_matrix("all", on_demand=True)["include"]
    large = [
        r
        for r in campaigns
        if r["binding"] in {"S01", "S02", "S04", "M03", "M04", "T01"}
    ]
    assert {r["mode"] for r in large} == {"smoke", "training", "lifecycle", "full"}
    for row in large:
        assert json.loads(row["runner"]) == {
            "group": "nv-gpu-amd64-rtxpro6000-1gpu",
            "labels": "linux-amd64-gpu-rtxpro6000-latest-1",
        }
        assert row["min_free_gib"] == (512 if row["binding"] == "T01" else 128)
    assert all(
        json.loads(r["runner"]) == "ubuntu-24.04"
        for r in campaigns
        if r["binding"] not in {"S01", "S02", "S04", "M03", "M04", "T01"}
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


def test_full_suite_is_automatic_and_manual_bindings_have_a_separate_entry_point():
    workflows = ROOT / ".github/workflows"
    suite = (workflows / "kumo-pytorch-t2b.yml").read_text()
    assert "name: PyTorch — Full Suite" in suite and "name: kumo-pytorch-gate" in suite
    for event in ("pull_request:", "schedule:", "workflow_dispatch:", "push:"):
        assert f"  {event}" in suite
    assert "github.event_name == 'schedule'" not in suite
    assert not (workflows / "kumo-pytorch-gate.yml").exists()
    difficult = (workflows / "kumo-pytorch-difficult-behavior.yml").read_text()
    assert "name: Difficult Behavior" in difficult
    assert "--required-coverage=difficult" in difficult
    assert "continue-on-error" not in difficult
    assert "  diskgraph-wheel:" not in difficult
    assert (
        "diskgraph-wheel-artifact: ${{ inputs.diskgraph-wheel-artifact }}" in difficult
    )
    caller = suite.split("  difficult-behavior:\n", 1)[1].split("\n  conformance:", 1)[
        0
    ]
    assert "needs: diskgraph-wheel" in caller
    assert "uses: ./.github/workflows/kumo-pytorch-difficult-behavior.yml" in caller
    assert "diskgraph-wheel-artifact: diskgraph-wheel" in caller
    for path in workflows.glob("*kumo-pytorch*.yml"):
        if path.name == "kumo-pytorch-on-demand.yml":
            text = path.read_text()
            assert "  workflow_dispatch:" in text
            assert all(
                f"  {event}:" not in text
                for event in ("push", "pull_request", "schedule", "workflow_call")
            )
            assert "matrix --on-demand --campaign" in text
            assert "uses: ./.github/workflows/_kumo-pytorch-binding.yml" in text
            assert "fail-fast: false" in text
            assert "continue-on-error" not in text
            assert "diskgraph-wheel-artifact: diskgraph-wheel" in text
            assert "kumo-pytorch-on-demand.yml" not in suite
            continue
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


def worker_step(name):
    text = (ROOT / ".github/workflows/_kumo-pytorch-binding.yml").read_text()
    return re.search(
        r"^      - name: " + re.escape(name) + r"\n(.*?)(?=^      - |\Z)",
        text,
        re.M | re.S,
    )[1]


@pytest.mark.parametrize(
    "group,mode,step,module,artifact",
    [
        (
            "lifecycle",
            "lifecycle",
            "Check bounded AML HI Small training lifecycle and mid-epoch resume",
            "test.kumo_pytorch.run_aml",
            "kumo-pytorch-aml-lifecycle",
        ),
        (
            "correctness",
            "histories",
            "Check real AML HI Small multi-source histories",
            "test.kumo_pytorch.history_checks",
            "kumo-pytorch-aml-histories",
        ),
        (
            "correctness",
            "smoke",
            "Check real AML HI Small source inputs, targets and losses",
            "test.kumo_pytorch.run_aml",
            "kumo-pytorch-aml-source-correctness",
        ),
    ],
)
def test_hi_small_checks_use_their_own_matrix_with_original_commands(
    tmp_path, monkeypatch, group, mode, step, module, artifact
):
    import textwrap
    from test.kumo_pytorch.bindings import selected_bindings
    from types import SimpleNamespace

    plan = nightly.execution_plan()
    rows = plan[group]["include"]
    grouped = [
        row for row in rows if row["executor"] != "binding" and row["mode"] == mode
    ]
    assert len(grouped) == 1
    row = grouped[0]
    assert (row["executor"], row["binding"], row["dataset"], row["mode"]) == (
        "aml-prototype",
        "hi-small",
        "hi-small",
        mode,
    )
    assert row["batch_size"] == 2 and row["threads"] == 1
    assert {b["ref"] for b in selected_bindings(campaign="legacy-lifecycle")} == {
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
    }
    suite = (ROOT / ".github/workflows/kumo-pytorch-t2b.yml").read_text()
    assert "lifecycle-hi-small" not in suite
    job = re.split(r"\n  (?=\S)", suite.split(f"\n  {group}:\n", 1)[1])[0]
    assert "executor: ${{ matrix.executor }}" in job
    assert "uses: ./.github/workflows/_kumo-pytorch-binding.yml" in job
    assert "fail-fast: false" in job
    assert "needs: [diskgraph-wheel, plan]" in job
    assert all(r["mode"] == "lifecycle" for r in plan["lifecycle"]["include"])
    assert all(
        r["mode"] in {"smoke", "histories"} for r in plan["correctness"]["include"]
    )
    worker = (ROOT / ".github/workflows/_kumo-pytorch-binding.yml").read_text()
    assert (
        "timeout-minutes: ${{ inputs.executor == 'aml-prototype' && 45 || 180 }}"
        in worker
    )
    artifact_step = worker_step(
        "Retain real AML HI Small "
        + {
            "lifecycle": "lifecycle",
            "histories": "history",
            "smoke": "source correctness",
        }[mode]
        + " reports"
    )
    assert f"name: {artifact}" in artifact_step
    assert f"inputs.mode == '{mode}'" in artifact_step
    assert "always() && inputs.executor == 'aml-prototype'" in artifact_step
    if mode == "histories":
        assert "$RUNNER_TEMP/aml-histories/report.json" in worker
        assert "uses: ./.github/actions/kumo-pytorch-results" in worker
    log = tmp_path / "commands.jsonl"
    fake = tmp_path / "python"
    fake.write_text(
        f'#!{sys.executable}\nimport json,os,sys\nwith open(os.environ["COMMAND_LOG"],"a") as stream: stream.write(json.dumps(sys.argv[1:])+"\\n")\n'
    )
    fake.chmod(0o755)
    text = worker_step(step)
    assert (
        f"!cancelled() && inputs.executor == 'aml-prototype' && inputs.mode == '{mode}' && steps.aml-data.outcome == 'success'"
        in text
    )
    command = textwrap.dedent(text.split("        run: |\n", 1)[1])
    subprocess.run(
        ["bash", "-e", "-c", command],
        check=True,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "RUNNER_TEMP": str(tmp_path),
            "BATCH_SIZE": str(row["batch_size"]),
            "THREADS": str(row["threads"]),
            "COMMAND_LOG": str(log),
        },
    )
    args = json.loads(log.read_text().splitlines()[-1])
    assert args[:3] == ["-m", "test.kumo_pytorch.nightly", "group"]
    assert args[args.index("--anchors-per-split") + 1] == "4"
    assert args[args.index("--threads") + 1] == "1"
    if mode == "lifecycle":
        assert args[args.index("--batch-size") + 1] == "2"
        assert args[args.index("--steps") + 1] == "1"
    assert args[args.index("--mode") + 1] == mode
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "test.kumo_pytorch.run_aml",
        SimpleNamespace(run_group=lambda args: calls.append(args) or 23),
    )
    assert nightly.main(args[2:]) == 23
    assert len(calls) == 1 and calls[0].mode == mode
    assert calls[0].anchors_per_split == 4 and calls[0].threads == 1
    assert calls[0].steps == 1 and calls[0].seed == 0
    if mode != "histories":
        assert calls[0].batch_size == 2


@pytest.mark.parametrize(
    "executor,dataset,binding,mode,success",
    [
        ("binding", "li-medium", "L01", "lifecycle", True),
        ("aml-prototype", "hi-small", "hi-small", "lifecycle", True),
        ("aml-prototype", "hi-small", "hi-small", "histories", True),
        ("aml-prototype", "hi-small", "hi-small", "smoke", True),
        ("aml-prototype", "li-medium", "hi-small", "lifecycle", False),
        ("aml-prototype", "li-medium", "hi-small", "histories", False),
        ("aml-prototype", "hi-small", "L01", "lifecycle", False),
        ("aml-prototype", "hi-small", "hi-small", "training", False),
        ("unknown", "hi-small", "hi-small", "lifecycle", False),
    ],
)
def test_matrix_worker_rejects_mismatched_entry_points(
    executor, dataset, binding, mode, success
):
    import textwrap

    text = worker_step("Validate campaign entry point")
    command = textwrap.dedent(text.split("        run: |\n", 1)[1])
    result = subprocess.run(
        ["bash", "-e", "-c", command],
        env={
            **os.environ,
            "EXECUTOR": executor,
            "DATASET": dataset,
            "BINDING": binding,
            "CAMPAIGN": mode,
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == success


def test_only_li_medium_combined_is_in_any_active_campaign(tmp_path):
    from test.kumo_pytorch.bindings import load_bindings

    plan = nightly.execution_plan()
    medium = [
        row
        for section in plan.values()
        for row in section["include"]
        if row.get("dataset") in {"li-medium", "hi-medium"}
    ]
    assert medium and {row["dataset"] for row in medium} == {"li-medium"}
    assert {row.get("binding", row.get("case")) for row in medium} == {
        "L01",
        "li-medium-combined",
    }
    catalog = load_bindings()
    assert all(catalog[ref]["campaigns"] == [] for ref in ("L02", "L03", "L04"))
    for ref in ("L02", "L03", "L04"):
        with pytest.raises(SystemExit) as error:
            nightly.main(
                [
                    "run",
                    "--binding",
                    ref,
                    "--mode",
                    "training",
                    "--data-dir",
                    str(tmp_path / "absent"),
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
        assert error.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("campaign", ["correctness", "training", "lifecycle"])
def test_hm_checks_select_one_owner_and_reuse_saved_runs(tmp_path, campaign):
    import textwrap

    worker = (ROOT / ".github/workflows/kumo-pytorch-t2a.yml").read_text()
    check = re.search(r"      - name: Check the runs\n(.*?)(?=      - )", worker, re.S)[
        1
    ]
    assert "KUMO_PYTORCH_E2E_REUSE_ONLY: '1'" in check
    command = textwrap.dedent(check.split("        run: |\n", 1)[1])
    fake = tmp_path / "python"
    fake.write_text(
        f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    fake.chmod(0o755)
    subprocess.run(
        ["bash", "-e", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "CAMPAIGN": campaign,
            "TIER": "t2a",
            "RUNNER_TEMP": str(tmp_path),
        },
    )
    log = next((tmp_path / "pytorch-ci-logs").glob("*/console.log"))
    args = json.loads(log.read_text())
    assert [arg for arg in args if arg.startswith("tests/")] == [
        f"tests/e2e/test_{campaign}.py"
    ]
    assert "--require-imports" in args and "--t2a" in args
    assert f"--junitxml={tmp_path}/{campaign}.xml" in args
    suite = (ROOT / ".github/workflows/kumo-pytorch-t2b.yml").read_text()
    checks = suite.split("\n  hm-checks:\n", 1)[1].split("\n  population:", 1)[0]
    assert "needs: [diskgraph-wheel, hm-runs]" in checks
    assert "fail-fast: false" in checks
    assert f"campaign: {campaign}" in checks
    assert "name: ${{ matrix.group.label }}" in checks
    assert "inputs.campaign }}-predictions-and-metrics" in worker
