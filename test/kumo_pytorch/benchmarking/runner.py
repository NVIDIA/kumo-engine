"""Generic complete-epoch loop; no dataset/model-specific imports or inspection."""

import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from importlib.metadata import distributions
from pathlib import Path

from .interface import BatchCounts, Workload
from .metrics import required_metrics, validate_inapplicable
from .recording import Recorder
from .reporting import report, write_json


def _runtime(factory):
    source = inspect.getsourcefile(factory)
    root = Path(__file__).resolve().parents[3]
    git = shutil.which("git")
    revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if git
        else subprocess.CompletedProcess([], 127, stdout="")
    )
    status = (
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if git
        else subprocess.CompletedProcess([], 127, stdout="")
    )
    diff = (
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if git
        else subprocess.CompletedProcess([], 127, stdout=b"")
    )
    implementation = sorted(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).parents[1] / "benchmark.py"
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
        if Path("/proc/cpuinfo").is_file()
        else platform.processor(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "physical_memory_bytes": os.sysconf("SC_PHYS_PAGES")
        * os.sysconf("SC_PAGE_SIZE")
        if "SC_PHYS_PAGES" in os.sysconf_names
        else None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "RAYON_NUM_THREADS",
                "POLARS_MAX_THREADS",
            )
        },
        "hostname": platform.node(),
        "pid": os.getpid(),
        "clock": "monotonic_ns",
        "process_isolation": False,
        "cache_state": "uncontrolled",
        "source_revision": revision.stdout.strip() or None,
        "worktree_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest()
        if diff.returncode == 0
        else None,
        "benchmark_sources": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in implementation
        },
        "adapter_source_sha256": (
            hashlib.sha256(Path(source).read_bytes()).hexdigest()
            if source and Path(source).is_file()
            else None
        ),
        "dependencies": {
            d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]
        },
    }


def _percentiles(values):
    ordered = sorted(values)
    return statistics.median(ordered), ordered[math.ceil(0.95 * len(ordered)) - 1]


def _epoch(
    workload,
    recorder,
    epoch,
    warmup_batches=2,
    steady_min_batches=4,
    synchronize_batches=False,
):
    workload.synchronize()
    totals = dict.fromkeys(BatchCounts.__dataclass_fields__, 0)
    absent = set()
    produced, wait_ns = 0, 0
    latencies = []
    complete, steps, updates = False, 0, 0
    first_end = None
    window = {}
    startup = {}
    steady_start = None
    steady_counts = None
    try:
        with recorder.span("execution") as window:
            start = window["start_ns"]
            with recorder.defer(), recorder.span("iterator.startup") as startup:
                iterator = iter(workload.batches(epoch))
            while True:
                with recorder.defer(), recorder.span("batch.request") as request:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        request["outcome"] = "exhausted"
                        complete = True
                    else:
                        request["outcome"] = "produced"
                    if synchronize_batches:
                        workload.synchronize()
                wait_ns += request["elapsed_ns"]
                if complete:
                    break
                first_end = request["end_ns"] if first_end is None else first_end
                latencies.append(request["elapsed_ns"])
                counted = workload.count(batch)
                if not isinstance(counted, BatchCounts):
                    raise TypeError("count must return BatchCounts")
                observed = asdict(counted)
                for name, value in observed.items():
                    if value is None:
                        absent.add(name)
                    else:
                        totals[name] += value
                recorder.write("batch", batch_index=produced, counts=observed)
                produced += 1
                if workload.step is not None:
                    with recorder.span("training.step"):
                        updated = workload.step(batch, recorder)
                        if type(updated) is not bool:
                            raise TypeError(
                                "step must return whether an optimizer update ran"
                            )
                    steps += 1
                    updates += int(updated)
                del batch
                if produced == warmup_batches:
                    workload.synchronize()
                    steady_start = recorder.clock()
                    steady_counts = dict(totals)
            workload.synchronize()
    finally:
        # A failed window has evidence but never receives successful throughput.
        success = complete and window.get("outcome") == "completed"
        values = {"batches": produced, "batch_wait_ns": wait_ns}
        for key, value in totals.items():
            if key not in absent and (
                produced or key in {"samples", "padding_samples"}
            ):
                values[key] = value
        if "samples" in values and "padding_samples" in values:
            values["real_samples"] = values["samples"] - values["padding_samples"]
        if first_end is not None:
            values.update(
                first_batch_ns=first_end - start, first_request_ns=latencies[0]
            )
        if latencies:
            values["batch_latency_p50_ns"], values["batch_latency_p95_ns"] = (
                _percentiles(latencies)
            )
        if len(latencies) > 1:
            (
                values["later_batch_latency_p50_ns"],
                values["later_batch_latency_p95_ns"],
            ) = _percentiles(latencies[1:])
        if workload.step is not None:
            values.update(steps=steps, optimizer_updates=updates)
        elapsed = window.get("elapsed_ns", 0)
        if success:
            recorder.measure(
                "batch_production.elapsed_ns",
                elapsed if workload.step is None else startup["elapsed_ns"] + wait_ns,
                timing_scope="complete_loader_window"
                if workload.step is None
                else "iterator_and_request_intervals",
            )
        if success and elapsed > 0:
            for count, rate in (
                ("real_samples", "samples"),
                ("tokens", "tokens"),
                ("nodes", "nodes"),
                ("edges", "edges"),
                ("steps", "steps"),
            ):
                if count in values:
                    values[rate + "_per_second"] = values[count] * 1e9 / elapsed
        for name, value in values.items():
            recorder.measure(name, value, completed_window=success)
        recorder.write(
            "epoch_result",
            complete=success,
            metrics=values,
            elapsed_ns=elapsed,
            percentile_method="nearest_rank_p95",
            later_batches_are_not_proven_steady_state=True,
        )
        if (
            success
            and steady_start is not None
            and produced - warmup_batches >= steady_min_batches
        ):
            durations = latencies[warmup_batches:]
            p50, p95 = _percentiles(durations)
            duration = window["end_ns"] - steady_start
            samples = (
                totals["samples"]
                - totals["padding_samples"]
                - steady_counts["samples"]
                + steady_counts["padding_samples"]
            )
            counts = {
                name: value - steady_counts[name]
                for name, value in totals.items()
                if name not in absent
            }
            counts["real_samples"] = samples
            rates = {
                f"{name}_per_second": counts[count] * 1e9 / duration
                for name, count in (
                    ("samples", "real_samples"),
                    ("tokens", "tokens"),
                    ("nodes", "nodes"),
                    ("edges", "edges"),
                )
                if count in counts and duration > 0
            }
            recorder.write(
                "steady_window",
                counts=counts,
                rates=rates,
                start_ns=steady_start,
                end_ns=window["end_ns"],
                elapsed_ns=duration,
                warmup_batches=warmup_batches,
                measured_batches=len(durations),
                real_samples=samples,
                samples_per_second=samples * 1e9 / duration if duration else None,
                p50_ns=p50,
                p95_ns=p95,
                latency_cv=statistics.pstdev(durations) / statistics.mean(durations)
                if sum(durations)
                else None,
                scope_description="explicit post-warmup window; coefficient of variation reported, no equilibrium assumed",
            )


def run(
    factory,
    config,
    output,
    *,
    repetitions=3,
    epochs=2,
    clock=time.monotonic_ns,
    memory_cgroup=None,
    memory_method="kernel",
    memory_interval=0.01,
    gpu_interval=1.0,
    warmup_batches=2,
    steady_min_batches=4,
    measurement_profile="full",
):
    """Run a trusted ``factory(config, workdir) -> Workload`` and retain evidence.

    Each repetition gets a new adapter and workspace, but uses this OS process.
    Existing output directories are never overwritten. Failures retain reports;
    callers must inspect status (CLI uses nonzero for incomplete coverage).
    """
    required_metrics(measurement_profile)
    for name, value in (
        ("repetitions", repetitions),
        ("epochs", epochs),
        ("warmup_batches", warmup_batches),
        ("steady_min_batches", steady_min_batches),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        type(gpu_interval) not in (int, float)
        or not math.isfinite(gpu_interval)
        or gpu_interval <= 0
    ):
        raise ValueError("gpu_interval must be positive and finite")
    if memory_method not in {"kernel", "sampled"}:
        raise ValueError("memory_method must be kernel or sampled")
    if not math.isfinite(memory_interval) or memory_interval <= 0:
        raise ValueError("memory_interval must be positive and finite")
    output = Path(output)
    frozen = {
        "adapter_config": config,
        "measurement_profile": measurement_profile,
        "repetitions": repetitions,
        "epochs": epochs,
        "runtime": _runtime(factory),
        "memory_cgroup": str(memory_cgroup) if memory_cgroup is not None else None,
        "gpu_interval_seconds": gpu_interval,
        "memory_method": memory_method,
        "memory_interval_seconds": memory_interval,
        "warmup_batches": warmup_batches,
        "steady_min_batches": steady_min_batches,
    }
    # Reject non-JSON settings before creating the run directory.
    frozen = json.loads(json.dumps(frozen, allow_nan=False))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", frozen)
    identities, errors = {}, []
    with (output / "observations.jsonl").open("x") as stream:
        from .memory import CgroupMemory, SampledAnonymousMemory, SampledCgroupMemory

        memory = (
            (
                SampledCgroupMemory(memory_cgroup, interval=memory_interval)
                if memory_method == "sampled"
                else CgroupMemory(memory_cgroup)
            )
            if memory_cgroup is not None and measurement_profile == "full"
            else None
        )

        recorder = Recorder(
            stream,
            clock=clock,
            observers=(
                memory,
                SampledAnonymousMemory(memory_cgroup, interval=memory_interval),
            )
            if memory is not None
            else (),
        )
        for repetition in range(repetitions):
            workload = None
            observers = recorder.observers
            recorder.scope = {"repetition": repetition}
            workdir = output / "work" / str(repetition)
            workdir.mkdir(parents=True)
            try:
                workload = factory(
                    json.loads(json.dumps(frozen["adapter_config"])), workdir
                )
                if not isinstance(workload, Workload):
                    raise TypeError("factory must return Workload")
                workload.validate()
                if measurement_profile == "timing" and workload.step is not None:
                    raise ValueError("Timing comparisons require loader-only workloads")
                if not workload.identity.get("dataset") or not workload.identity.get(
                    "workload"
                ):
                    raise ValueError("Adapter identity requires dataset and workload")
                validate_inapplicable(workload.inapplicable)
                identity = {
                    "identity": workload.identity,
                    "inapplicable": workload.inapplicable,
                    "device": workload.device,
                }
                identities[repetition] = json.loads(
                    json.dumps(identity, allow_nan=False)
                )
                recorder.write("workload", **identities[repetition])
                workload.prepare(recorder)
                if workload.device != "cpu" and measurement_profile == "full":
                    from .gpu import GPUCollector

                    recorder.observers = (
                        *observers,
                        GPUCollector(workload.device, interval=gpu_interval),
                    )
                for epoch in range(epochs):
                    recorder.scope = {"repetition": repetition, "epoch": epoch}
                    workload.before_epoch(epoch, recorder)
                    _epoch(
                        workload,
                        recorder,
                        epoch,
                        warmup_batches,
                        steady_min_batches,
                        synchronize_batches=measurement_profile == "timing"
                        and workload.device != "cpu",
                    )
            except Exception:  # noqa: BLE001 -- retain arbitrary adapter failures in the report
                errors.append(
                    {
                        "repetition": repetition,
                        "scope": dict(recorder.scope),
                        "traceback": traceback.format_exc(),
                    }
                )
                recorder.write("failure", error=errors[-1])
            finally:
                recorder.observers = observers
                if isinstance(workload, Workload):
                    try:
                        workload.close()
                    except Exception:  # noqa: BLE001 -- preserve both execution and cleanup failures
                        errors.append(
                            {
                                "repetition": repetition,
                                "operation": "close",
                                "traceback": traceback.format_exc(),
                            }
                        )
                        recorder.write("failure", error=errors[-1])
                stream.flush()
    return report(output, frozen, identities, errors)
