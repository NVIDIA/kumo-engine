# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one flow and seed of a tier, so CI can train each in its own job.

``python -m tests.e2e.runs TIER FLOW SEED --e2e-dir DIR`` prepares the tier's data
under ``DIR/<tier>`` and trains into ``DIR/<tier>/<flow>-seed<seed>``, the layout
``test_lifecycle.py`` reads. A finished run (``run.json``) on the same tables and
training anchors is reused, not retrained: after changing a flow, use a fresh
``DIR``. With ``KUMO_PYTORCH_E2E_REUSE_ONLY=1`` (the CI checks job) a missing run
is an error instead.

``python -m tests.e2e.runs loader --e2e-dir DIR [--repetitions N]`` times the
33-anchor CPU fixture in fresh pytest workers and prints the timing report path.
Each worker records setup once, fit stages per mode and batch stages per epoch.
Stage times include nested calls. A missing stage did not run on that workload.
On Linux, ``loader --cold-warm-cache`` verifies that the base index's file pages
are cold before epoch 0 and warm before epoch 1 of each fit, outside epoch timing.
Use ``loader --host-memory {kernel,sampled} --cgroup-parent DIR`` on Linux with
a delegated cgroup v2 memory controller. Each worker runs unprivileged in its own
cgroup. A root supervisor uses SUDO_UID/SUDO_GID to run pytest unprivileged.
Its output directory must be accessible to that user. Kernel mode resets
memory.peak per interval. Sampled mode reads memory.current every 10 ms.
Both sample anonymous memory and include descendant charges. Epoch wall time
includes collection overhead.
Use ``banking`` in place of ``loader`` to time one CPU PyG training step per
worker on the package banking fixture. Its step includes batch adaptation through
the optimizer update and excludes model initialization and batch delivery.
Use ``cuda`` to profile two native 33-anchor training epochs on the current GPU.
The same worker runs the three CUDA regressions and records their JUnit results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from importlib.metadata import distributions
from pathlib import Path
from xml.etree import ElementTree

from e2e.datasets.hm import HMDataset, generate, sample_anchors
from tests.e2e.tiers import BY_NAME, Tier

REPOSITORY = Path(__file__).resolve().parents[2]
FLOWS = {"gnn": "e2e.flows.hm_gnn", "pragma": "e2e.flows.hm_pragma"}
MARKER = "run.json"
TIMING_CASES = {
    "loader": "tests/test_native_anchor_handoff.py::test_native_cpu_loader_timing",
    "cuda": "tests/test_native_anchor_handoff.py::test_native_cuda_epochs",
    "banking": "tests/test_diskgraph_backend.py::"
    "test_synthetic_banking_materialization_sampling_and_reopen_are_reproducible",
}
CUDA_REGRESSIONS = (
    "tests/test_masking.py::test_masking_is_bitwise_equal_on_cpu_and_cuda",
    "tests/test_model.py::test_packed_event_encoder_runs_with_cuda_fp16",
    "tests/test_varlen_metadata.py::test_varlen_attention_consumes_the_metadata_directly",
)
LOADER_BATCHING = {"split": "train", "batch_size": 5}


def dataset(root: Path, tier: Tier) -> HMDataset:
    return generate(tier.scale, 0, root / f"hm-{tier.scale.lower()}")


def training_data(hm: HMDataset, root: Path, tier: Tier) -> HMDataset:
    """What the flows train on: the dataset, or the tier's sample of its anchors."""

    if not tier.sample:
        return hm
    return sample_anchors(hm, tier.sample, tier.sample_seed, root / "sampled")


def run(data: HMDataset, root: Path, flow: str, seed: int) -> tuple[Path, float]:
    """Train ``flow`` with ``seed`` unless a finished run on the same data is already there.

    The same data means the same table hashes and the same ``anchors.parquet`` bytes, so a
    run on another sample of the anchors is not reused.
    """

    work = root / f"{flow}-seed{seed}"
    marker = work / MARKER
    identity = {
        "flow": flow,
        "seed": seed,
        "data": dict(data.metadata.table_sha256),
        "anchors": hashlib.sha256(data.anchors.read_bytes()).hexdigest(),
    }
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if {key: recorded.get(key) for key in identity} == identity:
            return work, float(recorded["seconds"])
    if os.environ.get("KUMO_PYTORCH_E2E_REUSE_ONLY") == "1":
        raise RuntimeError(f"no finished run on this data at {work}")
    shutil.rmtree(work, ignore_errors=True)
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", FLOWS[flow], "--data", str(data.root), "--work", str(work)]
        + ["--seed", str(seed)],
        cwd=REPOSITORY,
        check=True,
    )
    seconds = time.perf_counter() - started
    marker.write_text(json.dumps({**identity, "seconds": seconds}) + "\n")
    return work, seconds


def time_loader(
    plan, table, work: Path, make_batches, *, setup, cache=None, **batch_options
) -> dict:
    """Measure two complete train epochs for a fresh fit and its saved artifact."""
    from kumo.pytorch import api
    from tests.e2e.host_memory import memory_interval
    from tests.e2e.loader_timing import install_stages, measure_epoch, stage_observer

    work.mkdir(parents=True, exist_ok=True)
    options = LOADER_BATCHING | batch_options
    results = {"setup": setup}
    artifact = work / "artifact"
    try:
        for mode in ("fresh", "reloaded"):
            results[mode] = {}
            with ExitStack() as tasks:
                with (
                    stage_observer(install_stages) as profile,
                    memory_interval(results[mode], "materialization"),
                ):
                    started = time.perf_counter_ns()
                    fit = {"fit_split": options["split"]}
                    if mode == "reloaded":
                        with profile.measure("artifact.load"):
                            fit = {"fitted": api.Artifact.load(artifact).fitted_state}
                    task = tasks.enter_context(plan.materialize(table, **fit))
                    materialize_ns = time.perf_counter_ns() - started
                results[mode] |= {
                    "materialize_ns": materialize_ns,
                    "stages": profile.stages,
                    "fitted_state": task.fitted_state.fingerprint,
                    "anchors": task.fingerprints["anchors"],
                    "anchor_sha256": setup["anchor_sha256"],
                    "contract": task.fingerprints["contract"],
                    "num_threads": plan.engine.num_threads,
                    "epochs": [],
                }
                if mode == "fresh":
                    task.to_artifact().save(artifact)
                for epoch in range(2):
                    with stage_observer(install_stages) as profile:
                        record = {"epoch": epoch, "complete": False, "stages": profile.stages}
                        results[mode]["epochs"].append(record)
                        if cache is not None:
                            record["file_page_cache"] = {}
                            cache.condition(
                                "cold" if epoch == 0 else "warm", record["file_page_cache"]
                            )
                        measure_epoch(task, make_batches, record, profile, **options)
    finally:
        (work / "results.json").write_text(json.dumps(results) + "\n")
    return results


def _ranges(rows):
    """Keep each completed worker's value alongside the median and range."""
    metrics = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if all(isinstance(v, dict) for v in values):
            metrics[key] = _ranges(values)
        elif all(type(v) in (int, float) for v in values):
            metrics[key] = {
                "values": values,
                "median": statistics.median(values),
                "range": [min(values), max(values)],
            }
    return metrics


def _cuda_junit_error(path, expected):
    """Require one passing JUnit case for every selected CUDA test."""
    try:
        cases = ElementTree.parse(path).findall(".//testcase")
    except (OSError, ElementTree.ParseError) as error:
        return str(error)
    passed = []
    for case in cases:
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            return f"CUDA case did not pass: {case.get('classname')}::{case.get('name')}"
        module = case.get("classname", "").replace(".", "/")
        passed.append(f"{module}.py::{case.get('name')}")
    if sorted(passed) != sorted(expected):
        return f"Expected CUDA cases {sorted(expected)}, observed {sorted(passed)}"
    return None


def _timing_campaign(
    workload: str,
    e2e_dir: Path,
    repetitions: int,
    cold_warm_cache=False,
    host_memory=None,
    cgroup_parent=None,
) -> Path:
    e2e_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{workload}-", dir=e2e_dir.resolve()))
    case = TIMING_CASES[workload]
    report = root / "timings.json"
    completed = []
    workers = []
    cache_args = ["--cold-warm-cache"] if cold_warm_cache else []
    env = {**os.environ, f"KUMO_PYTORCH_PROFILE_{workload.upper()}": "1"}
    env.pop("KUMO_PYTORCH_HOST_MEMORY", None)
    if host_memory:
        from tests.e2e.memory_worker import run_isolated, worker_identity

        supervisor_root = os.geteuid() == 0
        if supervisor_root:
            _, gid = worker_identity()
            os.chown(root, -1, gid)
            root.chmod(0o710)
    for index in range(repetitions):
        work = root / str(index)
        command = [sys.executable, "-m", "pytest", "-q", case, "--e2e-dir", str(work), *cache_args]
        if workload == "cuda":
            junit = work / "cuda.xml"
            command.extend([*CUDA_REGRESSIONS, f"--junitxml={junit}"])
        if host_memory:
            try:
                if supervisor_root:
                    work.mkdir()
                    os.chown(work, -1, gid)
                    work.chmod(0o770)
                worker = run_isolated(
                    command, parent=cgroup_parent, method=host_memory, cwd=REPOSITORY, env=env
                )
            except (OSError, RuntimeError, ValueError) as error:
                workers.append({"worker": index, "complete": False, "error": str(error)})
                break
        else:
            worker = subprocess.run(command, cwd=REPOSITORY, env=env)
        result = work / workload / "results.json"
        ok = worker.returncode == 0 and result.is_file()
        evidence = {}
        if workload == "cuda":
            evidence["junit"] = str(junit.relative_to(root))
            if error := _cuda_junit_error(junit, (case, *CUDA_REGRESSIONS)):
                evidence["error"] = error
                ok = False
        workers.append(
            {
                "worker": index,
                "returncode": worker.returncode,
                "complete": ok,
                "results": str(result.relative_to(root)) if result.is_file() else None,
                **evidence,
            }
        )
        if ok:
            completed.append(json.loads(result.read_text()))

    summary = {}
    for mode in completed[0] if completed else ():
        fits = [run[mode] for run in completed]
        summary[mode] = _ranges(fits)
        if "epochs" not in fits[0]:
            continue
        summary[mode]["epochs"] = [
            _ranges(rows) for rows in zip(*(fit["epochs"] for fit in fits), strict=True)
        ]
    sources = [
        Path(__file__),
        Path(__file__).with_name("loader_timing.py"),
        REPOSITORY / case.split("::")[0],
    ]
    if workload == "loader":
        sources.append(Path(__file__).with_name("file_cache.py"))
    if workload == "cuda":
        sources.append(Path(__file__).with_name("cuda_metrics.py"))
        sources.extend(REPOSITORY / node.split("::")[0] for node in CUDA_REGRESSIONS)
    if host_memory:
        sources.extend(Path(__file__).with_name(f) for f in ("host_memory.py", "memory_worker.py"))
    revision = tracked_diff = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPOSITORY)
        tracked_diff = hashlib.sha256(diff).hexdigest() if diff else None
    except (OSError, subprocess.CalledProcessError):
        pass  # Source distributions can run outside a Git checkout.
    metadata = {
        "workload": case,
        **({"file_page_cache": "cold-warm"} if cold_warm_cache else {}),
        **({"host_memory": host_memory} if host_memory else {}),
        "device": "cuda" if workload == "cuda" else "cpu",
        "stage_timing": "Inclusive caller wall time. Nested and native timings can overlap.",
        **({"batch_size": 2} if workload == "banking" else LOADER_BATCHING),
        "revision": revision,
        "tracked_diff_sha256": tracked_diff,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            d.name: d.version
            for d in distributions()
            if d.name in ("torch", "pyarrow", "diskgraph", "torch-geometric")
        },
        "workers": workers,
        "summary": summary,
    }
    report.write_text(json.dumps(metadata, indent=2) + "\n")
    if len(completed) != repetitions:
        raise SystemExit(f"{len(completed)}/{repetitions} workers completed. See {report}")
    return report


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--e2e-dir", type=Path, required=True)
    if argv and argv[0] in TIMING_CASES:
        parser.add_argument("--repetitions", type=int, default=3)
        if argv[0] == "loader":
            parser.add_argument("--host-memory", choices=("kernel", "sampled"))
            parser.add_argument(
                "--cgroup-parent", type=Path, help="delegated Linux cgroup v2 parent"
            )
            parser.add_argument(
                "--cold-warm-cache",
                action="store_true",
                help="verify index page residency before each epoch (Linux)",
            )
        args = parser.parse_args(argv[1:])
        if args.repetitions < 1:
            parser.error("--repetitions must be positive")
        options = vars(args)
        if bool(options.get("host_memory")) != bool(options.get("cgroup_parent")):
            parser.error("--host-memory and --cgroup-parent must be supplied together")
        if options.get("host_memory") and sys.platform != "linux":
            parser.error("--host-memory requires a delegated Linux cgroup")
        print(_timing_campaign(argv[0], **options))
        return
    parser.add_argument("tier", choices=sorted(BY_NAME))
    parser.add_argument("flow", choices=sorted(FLOWS))
    parser.add_argument("seed", type=int)
    args = parser.parse_args(argv)
    tier = BY_NAME[args.tier]
    root = args.e2e_dir / tier.name
    root.mkdir(parents=True, exist_ok=True)
    work, seconds = run(training_data(dataset(root, tier), root, tier), root, args.flow, args.seed)
    print(f"{work}: {seconds:.0f}s")


if __name__ == "__main__":
    main()
