"""Repeat any benchmark adapter in fresh interpreters and compare fitted state.

Adapters opting into state comparison consume load_fitted_state/save_fitted_state
configuration keys and emit fitting_state records (the Kumo adapter does this).
No cache eviction or warm-cache claim is implicit in a new process or index.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

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
    timeout=3600,
    isolate_memory=False,
    gpu_interval=0.1,
):
    if (
        type(repetitions) is not int
        or repetitions < 2
        or type(epochs) is not int
        or epochs < 1
    ):
        raise ValueError("Campaign requires at least two repetitions and one epoch")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Worker timeout must be positive")
    if {"save_fitted_state", "load_fitted_state"} & config.keys():
        raise ValueError("Campaign owns fitted-state paths")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "config.json",
        {
            "adapter": adapter,
            "config": config,
            "repetitions": repetitions,
            "epochs": epochs,
            "compare_fitting": compare_fitting,
            "isolate_memory": isolate_memory,
            "worker_timeout_seconds": timeout,
            "gpu_interval_seconds": gpu_interval,
        },
    )
    runs, comparisons = [], []
    for repetition in range(repetitions):
        pair = {}
        for mode in ("fresh", "reused") if compare_fitting else ("fresh",):
            name = f"repeat-{repetition}-{mode}"
            selected = dict(config)
            state = output / f"repeat-{repetition}-fitted-state"
            if compare_fitting:
                selected[
                    "save_fitted_state" if mode == "fresh" else "load_fitted_state"
                ] = str(state)
            path = output / f"{name}.json"
            write_json(path, selected)
            command = [
                sys.executable,
                "-m",
                "test.kumo_pytorch.benchmark",
                "--adapter",
                adapter,
                "--config",
                str(path),
                "--output",
                str(output / name),
                "--repetitions",
                "1",
                "--epochs",
                str(epochs),
                "--gpu-interval",
                str(gpu_interval),
            ]
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
                            "cold_and_warm_caches",
                        )
                    ),
                    "cache_experiment": result["experiments"]["cold_and_warm_caches"]
                    if result
                    else None,
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
    fitting_complete = compare_fitting and all(
        c["status"] == "measured" for c in comparisons
    )
    result = {
        "status": "failed"
        if not completed or any(c["status"] == "failed" for c in comparisons)
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
    }
    write_json(output / "results.json", result)
    lines = [
        "# Benchmark campaign",
        "",
        f"Status: **{result['status']}**.",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--compare-fitting", action="store_true")
    parser.add_argument("--worker-timeout", type=float, default=3600)
    parser.add_argument("--isolate-memory", action="store_true")
    parser.add_argument("--gpu-interval", type=float, default=0.1)
    args = parser.parse_args(argv)
    result = run_campaign(
        args.adapter,
        json.loads(args.config.read_text()),
        args.output,
        repetitions=args.repetitions,
        epochs=args.epochs,
        compare_fitting=args.compare_fitting,
        timeout=args.worker_timeout,
        isolate_memory=args.isolate_memory,
        gpu_interval=args.gpu_interval,
    )
    print(f"{result['status']}: {args.output / 'report.md'}")
    return {"complete": 0, "failed": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
