# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare fresh fitting memory at two anchor counts using existing cgroup collectors.

The JSON specification supplies a Kumo contract, Parquet sources, anchor column
bindings and two anchor files. No dataset or model is built into this runner.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

from .campaign import stop
from .memory import CgroupMemory, SampledAnonymousMemory, SampledCgroupMemory
from .recording import Recorder


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker(spec_path, output, *, count=None, method="sampled", interval=0.05):
    """No count builds the index; counted workers must reopen it without ingestion."""
    import importlib.metadata

    import torch

    import kumo.pytorch.api as r

    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(spec_path.read_text())
    report = {
        "status": "failed",
        "count": count,
        "pid": os.getpid(),
        "spec_sha256": digest(spec_path),
        "stage": "index" if count is None else "fitting",
    }
    try:
        torch.set_num_threads(spec["threads"])
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "pyarrow", "diskgraph")
        }
        contract = r.DataLoaderSpec.from_json(Path(spec["contract"]).read_text())
        assert contract.fingerprint == spec["contract_fingerprint"]
        dataset = r.Dataset(
            {name: r.Parquet(path) for name, path in spec["sources"].items()}
        )
        index = Path(spec["index"])
        if count is None:
            assert not index.exists(), "Index construction requires a new destination"
            with r.prepare(
                contract,
                dataset,
                r.DiskGraphEngine(index=index, num_threads=spec["threads"]),
            ):
                pass
            report["status"] = "passed"
        else:
            group = os.environ.get("KUMO_BENCHMARK_CGROUP")
            if not group:
                raise RuntimeError(
                    "Fitting measurements require a dedicated Linux cgroup"
                )
            entry = spec["anchors"][str(count)]
            assert digest(entry["path"]) == entry["sha256"], "Anchor file changed"
            columns = dict(spec["anchor_columns"])
            if "metadata" in columns:
                columns["metadata"] = tuple(columns["metadata"])
            anchors = r.AnchorTable(r.Parquet(entry["path"]), **columns)
            memory = (
                SampledCgroupMemory(group, interval=interval)
                if method == "sampled"
                else CgroupMemory(group)
            )
            with (output / "observations.jsonl").open("w") as stream:
                recorder = Recorder(
                    stream,
                    observers=(
                        memory,
                        SampledAnonymousMemory(group, interval=interval),
                    ),
                )
                # Imports, identity checks, index construction and reopen are
                # outside the materialization boundary. No fitted state is reused.
                with r.prepare(
                    contract,
                    dataset,
                    r.DiskGraphEngine(
                        index=index, ingest="never", num_threads=spec["threads"]
                    ),
                ) as plan:
                    with recorder.span("materialization"):
                        task = plan.materialize(
                            anchors,
                            fit_split=spec["fit_split"],
                            encoder_cutoff=spec["encoder_cutoff_seconds"],
                        )
                    with task:
                        report["actual_count"] = task.count(spec["fit_split"])
                        assert report["actual_count"] == count, (
                            "Fitted population count differs"
                        )
                        report["fitted_state_sha256"] = task.fitted_state.fingerprint
                        report["anchors_sha256"] = task.fingerprints["anchors"]
            observations = [
                json.loads(line)
                for line in (output / "observations.jsonl").read_text().splitlines()
            ]
            peak = next(
                row
                for row in observations
                if row.get("metric") == "materialization.peak_host_bytes"
            )
            anonymous = next(
                row
                for row in observations
                if row.get("metric") == "materialization.peak_anonymous_bytes"
            )
            report["anonymous_memory"] = anonymous
            report["peak_anonymous_bytes"] = anonymous["value"]
            assert anonymous["status"] == "available", (
                "Anonymous fitting memory is unavailable"
            )
            elapsed = next(
                row
                for row in observations
                if row.get("metric") == "materialization.elapsed_ns"
            )
            report["memory"] = {k: v for k, v in peak.items() if k != "details"}
            report["memory"]["details"] = {
                k: v for k, v in peak["details"].items() if k != "samples"
            }
            samples = peak["details"].get("samples", [])
            report["stage_start_bytes"] = samples[0]["value"] if samples else None
            report["elapsed_ns"] = elapsed["value"]
            assert peak["status"] == elapsed["status"] == "available", (
                "Fitting measurement is incomplete"
            )
            assert peak["value"] > 0, "A zero peak is not a valid memory measurement"
            report["peak_host_bytes"] = peak["value"]
            report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - required failures remain failed reports
        report.update(error=str(error), traceback=traceback.format_exc())
    save(output / "report.json", report)
    return report


def compare(runs, *, counts, repetitions):
    """Report growth only after validating every required measurement."""
    if len(counts) != 2 or not 0 < counts[0] < counts[1] or repetitions < 1:
        raise ValueError("Two increasing counts and positive repetitions are required")
    result = {
        "status": "failed",
        "counts": list(counts),
        "repetitions": repetitions,
        "aggregation": "median absolute stage peak per count",
        "peaks_bytes": {},
        "anonymous_peaks_bytes": {},
        "growth_ratio": None,
    }
    if len(runs) != repetitions * 2:
        return {**result, "reason": "Missing fitting workers"}
    methods, scopes, specs, groups = set(), set(), set(), set()
    for count in counts:
        selected = [run for run in runs if run.get("count") == count]
        if len(selected) != repetitions or {
            run.get("repeat") for run in selected
        } != set(range(repetitions)):
            return {**result, "reason": "Missing or duplicate size/repetition"}
        peaks, anonymous_peaks = [], []
        for run in selected:
            details = run.get("memory", {}).get("details", {})
            peak = run.get("peak_host_bytes")
            if (
                run.get("status") != "passed"
                or run.get("returncode") != 0
                or run.get("actual_count") != count
                or type(peak) is not int
                or peak <= 0
                or run.get("memory", {}).get("status") != "available"
                or not details.get("completed_operation")
                or not details.get("cgroup")
            ):
                return {
                    **result,
                    "reason": "Failed worker, wrong population, or unavailable memory",
                }
            methods.add(details.get("method"))
            scopes.add(details.get("memory_scope"))
            specs.add(run.get("spec_sha256"))
            groups.add(details["cgroup"])
            anon = run.get("peak_anonymous_bytes")
            if (
                type(anon) is not int
                or anon < 0
                or run.get("anonymous_memory", {}).get("status") != "available"
            ):
                return {**result, "reason": "Unavailable anonymous fitting memory"}
            anonymous_peaks.append(anon)
            peaks.append(peak)
        result["peaks_bytes"][str(count)] = peaks
        result["anonymous_peaks_bytes"][str(count)] = anonymous_peaks
    if (
        None in methods
        or None in scopes
        or None in specs
        or len(methods) != 1
        or len(scopes) != 1
        or len(specs) != 1
        or len(groups) != len(runs)
    ):
        return {**result, "reason": "Incomparable measurements or reused worker cgroup"}
    medians = [statistics.median(result["peaks_bytes"][str(n)]) for n in counts]
    ratio = medians[1] / medians[0]
    result.update(
        growth_ratio=ratio,
        median_anonymous_bytes={
            str(n): statistics.median(result["anonymous_peaks_bytes"][str(n)])
            for n in counts
        },
        median_peak_bytes=dict(zip(map(str, counts), medians, strict=True)),
        method=next(iter(methods)),
        memory_scope=next(iter(scopes)),
        status="passed",
    )
    return result


def launch(command, log_path, *, timeout, isolate=False):
    if isolate:
        command = [
            "sudo",
            "-n",
            "--preserve-env=PATH,PYTHONPATH,OMP_NUM_THREADS,MKL_NUM_THREADS,OPENBLAS_NUM_THREADS,RAYON_NUM_THREADS,POLARS_MAX_THREADS",
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
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout + 15)
        except subprocess.TimeoutExpired:
            stop(process)
            return 124
        except BaseException:
            stop(process)
            raise


def campaign(
    spec_path,
    output,
    *,
    repetitions=3,
    method="sampled",
    interval=0.05,
    timeout=3600,
    repeat=None,
):
    spec_path, output = spec_path.resolve(), output.resolve()
    spec = json.loads(spec_path.read_text())
    counts = sorted(map(int, spec["anchors"]))
    compare([], counts=counts, repetitions=repetitions)
    if repeat is not None and (
        type(repeat) is not int or not 0 <= repeat < repetitions
    ):
        raise ValueError("Selected repetition is outside the campaign")
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("Positive finite timeout and sample interval required")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "case": spec["id"],
        "runs": [],
        "spec": spec,
        "counts": counts,
        "repetitions": repetitions,
        "selected_repeat": repeat,
        "all_repetitions_executed": repeat is None,
        "cache_state": "uncontrolled; same prebuilt index, alternating size order",
        "measurement_scope": "native anchor binding and fresh fitting inside Plan.materialize; excludes index build/reopen",
    }
    save(output / "report.json", result)
    command = [
        sys.executable,
        "-m",
        "test.kumo_pytorch.benchmarking.fitting_memory",
        "--worker",
        "--spec",
        str(spec_path),
    ]
    try:
        index_output = output / "index-build"
        code = launch(
            [*command, "--output", str(index_output)],
            output / "index-build.log",
            timeout=timeout,
        )
        result["index_returncode"] = code
        if code:
            raise RuntimeError("Index construction failed; see index-build.log")
        expected_spec = digest(spec_path)
        for repetition in range(repetitions) if repeat is None else (repeat,):
            for count in counts if repetition % 2 == 0 else reversed(counts):
                run_output = output / f"{count}-repeat-{repetition}"
                code = launch(
                    [
                        *command,
                        "--output",
                        str(run_output),
                        "--count",
                        str(count),
                        "--memory-method",
                        method,
                        "--memory-interval",
                        str(interval),
                    ],
                    output / f"{run_output.name}.log",
                    timeout=timeout,
                    isolate=True,
                )
                report_path = run_output / "report.json"
                run = (
                    json.loads(report_path.read_text())
                    if report_path.exists()
                    else {"status": "failed", "error": "Worker produced no report"}
                )
                run.update(count=count, repeat=repetition, returncode=code)
                if run.get("spec_sha256") != expected_spec:
                    run["status"] = "failed"
                    run["error"] = "Worker used a different configuration"
                result["runs"].append(run)
                save(output / "report.json", result)
        result["comparison"] = compare(
            result["runs"]
            if repeat is None
            else [dict(run, repeat=0) for run in result["runs"]],
            counts=counts,
            repetitions=repetitions if repeat is None else 1,
        )
        result["status"] = result["comparison"]["status"]
    except Exception as error:  # noqa: BLE001 - keep partial runs and fail closed
        result.update(error=str(error), traceback=traceback.format_exc())
    save(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--count", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--repeat",
        type=int,
        help="Select one paired repetition for external orchestration",
    )
    parser.add_argument(
        "--memory-method", choices=("sampled", "kernel"), default="sampled"
    )
    parser.add_argument("--memory-interval", type=float, default=0.05)
    parser.add_argument("--worker-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    if args.worker:
        report = worker(
            args.spec,
            args.output,
            count=args.count,
            method=args.memory_method,
            interval=args.memory_interval,
        )
    else:
        report = campaign(
            args.spec,
            args.output,
            repetitions=args.repetitions,
            method=args.memory_method,
            interval=args.memory_interval,
            timeout=args.worker_timeout,
            repeat=args.repeat,
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
