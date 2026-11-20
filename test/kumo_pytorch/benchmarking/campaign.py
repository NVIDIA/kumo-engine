# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeat any benchmark adapter in fresh interpreters and compare fitted state.

Adapters opting into state comparison consume load_fitted_state/save_fitted_state
configuration keys and emit fitting_state records (the Kumo adapter does this).
No cache eviction or warm-cache claim is implicit in a new process or index.
"""

import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

from .cli import main as run_cli
from .cli import worker_command
from .metrics import required_metrics
from .reporting import write_json


def stop(process):
    """Let the cgroup supervisor clean up before a last-resort kill."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    return process.wait()


def compare_states(fresh, reused):
    if any(result["execution_status"] != "completed" for result in (fresh, reused)):
        return {"status": "failed", "reason": "Execution failed"}
    left, right = fresh.get("fitting_states", []), reused.get("fitting_states", [])
    if len(left) != 1 or len(right) != 1:
        return {
            "status": "incomplete",
            "reason": "Adapter did not record one fitted state per run",
        }
    left, right = left[0], right[0]
    if (left["mode"], right["mode"]) != ("fresh", "reused"):
        return {"status": "failed", "reason": "Expected fresh and reused fitting modes"}
    if any(
        left[key] != right[key]
        for key in ("fingerprint", "contract_fingerprint", "anchors_fingerprint")
    ):
        return {"status": "failed", "reason": "Fitted state or contract changed"}
    if not left["saved_to"] or left["saved_to"] != right["loaded_from"]:
        return {
            "status": "failed",
            "reason": "Reuse did not load the fresh run's artifact",
        }
    # Same task and selected input population: compare actual delivered counts,
    # not stochastic masks, graph sampling outcomes, or numerical predictions.
    counts = lambda result: [
        row["metrics"].get("real_samples") for row in result["epochs"]
    ]
    if counts(fresh) != counts(reused):
        return {"status": "failed", "reason": "Selected population delivery changed"}

    def measured(result, metric):
        return [
            row["value"]
            for row in result["measurements"].get(metric, [])
            if row["status"] == "available"
        ]

    fresh_ns = measured(fresh, "materialization.elapsed_ns")
    reused_ns = measured(reused, "materialization.elapsed_ns")
    load_ns = measured(reused, "state.load.elapsed_ns")
    if not fresh_ns or not reused_ns or not load_ns:
        return {"status": "incomplete", "reason": "Missing fitting/loading timing"}
    return {
        "status": "measured",
        "fingerprint": left["fingerprint"],
        "fresh_materialization_ns": fresh_ns,
        "reused_materialization_ns": reused_ns,
        "state_load_ns": load_ns,
        "state_load_is_inside_materialization": True,
        "cache_scope": "See each worker's cache conditions; this comparison alone does not control cache state",
    }


def run_campaign(
    adapter,
    config,
    output,
    *,
    repetitions=3,
    epochs=2,
    compare_fitting=False,
    index_lifecycle=False,
    timeout=3600,
    isolate_memory=False,
    gpu_interval=0.1,
    memory_method="kernel",
    memory_interval=0.01,
    measurement_profile="full",
):
    required_metrics(measurement_profile)
    if index_lifecycle and (repetitions != 3 or compare_fitting):
        raise ValueError(
            "Index lifecycle requires three runs and fresh fitting in each"
        )
    if index_lifecycle and {"index_dir", "index_action"} & config.keys():
        raise ValueError("Index lifecycle owns index paths and actions")
    if (
        type(repetitions) is not int
        or repetitions < 2
        or type(epochs) is not int
        or epochs < 1
    ):
        raise ValueError("Campaign requires at least two repetitions and one epoch")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Worker timeout must be positive")
    if (
        memory_method not in {"kernel", "sampled"}
        or not math.isfinite(memory_interval)
        or memory_interval <= 0
    ):
        raise ValueError("Valid memory method and positive interval required")
    if {"save_fitted_state", "load_fitted_state"} & config.keys():
        raise ValueError("Campaign owns fitted-state paths")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "config.json",
        {
            "measurement_profile": measurement_profile,
            "adapter": adapter,
            "config": config,
            "repetitions": repetitions,
            "epochs": epochs,
            "compare_fitting": compare_fitting,
            "index_lifecycle": index_lifecycle,
            "isolate_memory": isolate_memory,
            "worker_timeout_seconds": timeout,
            "gpu_interval_seconds": gpu_interval,
            "memory_method": memory_method,
            "memory_interval_seconds": memory_interval,
        },
    )
    runs, comparisons = [], []
    index_checks = []
    for repetition in range(repetitions):
        pair = {}
        for mode in ("fresh", "reused") if compare_fitting else ("fresh",):
            name = f"repeat-{repetition}-{mode}"
            selected = dict(config)
            if index_lifecycle:
                selected.setdefault("prepared_dir", str(output / "shared-prepared"))
                selected.update(
                    index_dir=str(output / "shared-index"),
                    index_action="build" if repetition == 0 else "reopen",
                )
            state = output / f"repeat-{repetition}-fitted-state"
            if compare_fitting:
                selected[
                    "save_fitted_state" if mode == "fresh" else "load_fitted_state"
                ] = str(state)
            path = output / f"{name}.json"
            write_json(path, selected)
            command = worker_command(
                adapter=adapter,
                config=path,
                output=output / name,
                repetitions=1,
                epochs=epochs,
                gpu_interval=gpu_interval,
                measurement_profile=measurement_profile,
                memory_method=memory_method,
                memory_interval=memory_interval,
            )
            if isolate_memory:
                command = [
                    "sudo",
                    "--",
                    sys.executable,
                    "-m",
                    "test.kumo_pytorch.benchmarking.cgroup_run",
                    "--uid",
                    str(os.getuid()),
                    "--gid",
                    str(os.getgid()),
                    "--timeout",
                    str(timeout),
                    "--",
                    *command,
                ]
            with (output / f"{name}.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    code = process.wait(
                        timeout=timeout + 15 if isolate_memory else timeout
                    )
                    timed_out = isolate_memory and code == 124
                except subprocess.TimeoutExpired:
                    code = stop(process)
                    timed_out = True
                except BaseException:
                    stop(process)
                    raise
            result_file = output / name / "results.json"
            result = (
                json.loads(result_file.read_text()) if result_file.is_file() else None
            )
            successful_execution = (
                code in (0, 2)
                and result is not None
                and result["execution_status"] == "completed"
            )
            runs.append(
                {
                    "name": name,
                    "repetition": repetition,
                    "mode": mode,
                    "pid": process.pid,
                    "exit_code": code,
                    "timed_out": timed_out,
                    "execution_completed": successful_execution,
                    "benchmark_status": result["status"] if result else "failed",
                    "metric_coverage_complete": bool(result)
                    and result["metric_coverage"]["complete"]
                    == result["metric_coverage"]["total"],
                    "epoch_experiments_complete": bool(result)
                    and all(
                        result["experiments"][name]["status"] == "measured"
                        for name in (
                            "steady_state",
                            "first_and_later_epochs",
                            *(
                                ("cold_and_warm_caches",)
                                if measurement_profile == "full"
                                else ()
                            ),
                        )
                    ),
                    "cache_experiment": result["experiments"]["cold_and_warm_caches"]
                    if result
                    else None,
                }
            )
            if index_lifecycle:
                action = selected["index_action"]
                measurements = result.get("measurements", {}) if result else {}
                wanted = [
                    m
                    for m in measurements.get(f"index.{action}.elapsed_ns", [])
                    if m["status"] == "available"
                ]
                other = "reopen" if action == "build" else "build"
                forbidden = [
                    m
                    for m in measurements.get(f"index.{other}.elapsed_ns", [])
                    if m["status"] == "available"
                ]
                index_checks.append(
                    {
                        "run": name,
                        "action": action,
                        "index_path": selected["index_dir"],
                        "status": "measured"
                        if len(wanted) == 1 and not forbidden
                        else "failed",
                        "timings": wanted,
                    }
                )
            if result is not None:
                pair[mode] = result
                worker_config = json.loads((output / name / "config.json").read_text())
                runs[-1]["supervisor_pid"] = process.pid
                runs[-1]["pid"] = worker_config["runtime"]["pid"]
        if compare_fitting:
            comparisons.append(
                dict(
                    repetition=repetition,
                    **(
                        compare_states(pair["fresh"], pair["reused"])
                        if len(pair) == 2
                        else {"status": "failed", "reason": "Worker report missing"}
                    ),
                )
            )
    completed = all(row["execution_completed"] for row in runs)
    covered = completed and all(
        row["metric_coverage_complete"] and row["epoch_experiments_complete"]
        for row in runs
    )
    fitting_complete = not compare_fitting or all(
        c["status"] == "measured" for c in comparisons
    )
    result = {
        "measurement_profile": measurement_profile,
        "status": "failed"
        if not completed
        or any(c["status"] == "failed" for c in comparisons + index_checks)
        else "complete"
        if covered and fitting_complete
        else "incomplete",
        "process_isolation": "one new interpreter per run; sequential launches; shared OS page cache",
        "independent_repeated_runs": "measured" if completed else "incomplete",
        "cold_and_warm_caches": "measured; selected Linux file pages only"
        if completed
        and all(row["cache_experiment"]["status"] == "measured" for row in runs)
        else "pending; see worker cache-condition coverage",
        "runs": runs,
        "fitting_comparisons": comparisons,
        "index_lifecycle": index_checks,
        "index_cache_scope": "Build once, reopen the same index twice; OS page-cache state is recorded separately",
    }
    if measurement_profile == "timing":
        result["cold_and_warm_caches"] = (
            "outside timing-comparison scope; caches uncontrolled"
        )
    write_json(output / "results.json", result)
    lines = [
        "# Benchmark campaign",
        "",
        f"Status: **{result['status']}**. Measurement profile: {measurement_profile}.",
        "",
        "Each run starts a fresh interpreter. Process isolation does not itself control OS caches; see worker conditions.",
        "",
        "| Run | PID | Exit | Benchmark status | Report |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in runs:
        name = row["name"]
        lines.append(
            f"| {name} | {row['pid']} | {row['exit_code']} | {row['benchmark_status']} | [report]({name}/report.md) |"
        )
    lines += [
        "",
        "## Index build and reuse",
        "",
        "```json",
        json.dumps(index_checks, indent=2),
        "```",
        "",
        "## Fitting comparisons",
        "",
        "```json",
        json.dumps(comparisons, indent=2),
        "```",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    return run_cli(run_campaign, argv, scope="campaign", description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
