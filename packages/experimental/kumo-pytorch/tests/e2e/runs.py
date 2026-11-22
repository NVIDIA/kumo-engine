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
Use ``l01`` with the pinned LI Medium files in ``KUMO_PYTORCH_DATA_ROOT`` to time
two CPU loader epochs per fit. Each repetition fits in one worker and reloads
its artifact in another. The workers share the prepared 512 anchors per split.
``l01 --index-lifecycle`` builds one index and reopens it twice in separate
CPU workers. Each fits fresh and runs both epochs.
``l01 --length-planning`` buckets those CPU batches by history length and
checks native cache reuse on uncontrolled file pages.
``l01 --cuda --cgroup-parent DIR`` trains on those batches with sampled host
memory and verified cold/warm index pages on a Linux GPU worker.
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
from contextlib import ExitStack, nullcontext
from importlib.metadata import distributions
from itertools import product
from pathlib import Path
from xml.etree import ElementTree

from e2e.datasets.hm import HMDataset, generate, sample_anchors
from tests.e2e.tiers import BY_NAME, Tier

REPOSITORY = Path(__file__).resolve().parents[2]
FLOWS = {"gnn": "e2e.flows.hm_gnn", "pragma": "e2e.flows.hm_pragma"}
MARKER = "run.json"
TIMING_CASES = {
    "loader": "tests/test_native_anchor_handoff.py::test_native_cpu_loader_timing",
    "l01": "tests/e2e/test_aml.py::test_li_medium_l01_loader",
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
L01_BATCHING = LOADER_BATCHING | {"batch_size": 4}


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
    plan,
    table,
    work: Path,
    make_batches,
    *,
    setup,
    cache=None,
    modes=("fresh", "reloaded"),
    artifact=None,
    make_step=None,
    **batch_options,
) -> dict:
    """Measure two complete train epochs for a fresh fit and its saved artifact."""
    from kumo.pytorch import api
    from tests.e2e.host_memory import memory_interval
    from tests.e2e.loader_timing import install_stages, measure_epoch, stage_observer

    work.mkdir(parents=True, exist_ok=True)
    options = LOADER_BATCHING | batch_options
    results = {"setup": setup}
    artifact = artifact or work / "artifact"
    try:
        for mode in modes:
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
                previous_caches = None
                step = None
                if make_step is not None:
                    import torch

                    from tests.e2e.cuda_metrics import cuda_epoch

                    device = torch.device("cuda", torch.cuda.current_device())
                    step = make_step(task, device)
                for epoch in range(2):
                    with stage_observer(install_stages) as profile:
                        record = {"epoch": epoch, "complete": False, "stages": profile.stages}
                        results[mode]["epochs"].append(record)
                        if cache is not None:
                            record["file_page_cache"] = {}
                            cache.condition(
                                "cold" if epoch == 0 else "warm", record["file_page_cache"]
                            )
                        with (
                            cuda_epoch(device, record) if step is not None else nullcontext() as gpu
                        ):
                            measure_epoch(
                                task, make_batches, record, profile, gpu=gpu, step=step, **options
                            )
                        if task.contract.batch.batching.bucket_by_length:
                            current = task._anchors._length_caches
                            record["length_cache_reused"] = (
                                bool(current) and current == previous_caches
                            )
                            previous_caches = current.copy()
                            record["lengths_computed"], record["length_cache_hits"] = (
                                task._anchors._last_length_planning_stats
                            )
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
    cuda=False,
    index_lifecycle=False,
    length_planning=False,
) -> Path:
    e2e_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{workload}-", dir=e2e_dir.resolve()))
    case = TIMING_CASES[workload]
    report = root / "timings.json"
    completed = []
    workers = []
    env = {**os.environ, f"KUMO_PYTORCH_PROFILE_{workload.upper()}": "1"}
    env.pop("KUMO_PYTORCH_HOST_MEMORY", None)
    env.pop("KUMO_PYTORCH_L01_CUDA", None)
    env.pop("KUMO_PYTORCH_L01_INDEX_ACTION", None)
    env["KUMO_PYTORCH_L01_LENGTH_PLANNING"] = str(int(length_planning))
    if cuda:
        env["KUMO_PYTORCH_L01_CUDA"] = "1"
    modes = ("",)
    setup = {}
    if workload == "l01":
        from e2e.datasets.aml import prepare_l01_benchmark

        prepared = root / "prepared"
        started = time.perf_counter_ns()
        prepare_l01_benchmark(Path(env["KUMO_PYTORCH_DATA_ROOT"]), prepared)
        setup = {"shared_population_prepare_ns": time.perf_counter_ns() - started}
        modes = ("fresh",) if index_lifecycle else ("fresh", "reloaded")
    if host_memory:
        from tests.e2e.memory_worker import run_isolated, worker_identity

        supervisor_root = os.geteuid() == 0
        if supervisor_root:
            _, gid = worker_identity()
            os.chown(root, -1, gid)
            root.chmod(0o750 if cuda else 0o710)
            if cuda:
                for path in (prepared, *prepared.iterdir()):
                    os.chown(path, -1, gid)
                    path.chmod(0o750 if path.is_dir() else 0o640)
    for index, mode in product(range(repetitions), modes):
        if index_lifecycle:
            env["KUMO_PYTORCH_L01_INDEX_ACTION"] = "build" if index == 0 else "reopen"
        work = root / str(index) / mode
        node = f"{case}[{mode}]" if mode else case
        command = [sys.executable, "-m", "pytest", "-q", node, "--e2e-dir", str(work)]
        if cold_warm_cache:
            command.append("--cold-warm-cache")
        expected_cases = [node, *CUDA_REGRESSIONS] if workload == "cuda" else [node]
        if workload == "cuda" or cuda:
            junit = work / "cuda.xml"
            command.extend([*expected_cases[1:], f"--junitxml={junit}"])
        if host_memory:
            try:
                if supervisor_root:
                    if mode == "fresh":
                        work.parent.mkdir()
                        os.chown(work.parent, -1, gid)
                        work.parent.chmod(0o750)
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
        evidence = {"pair": index, "fit": mode} if mode else {}
        if index_lifecycle:
            evidence = {"index_action": env["KUMO_PYTORCH_L01_INDEX_ACTION"], "fit": mode}
        if workload == "cuda" or cuda:
            evidence["junit"] = str(junit.relative_to(root))
            if error := _cuda_junit_error(junit, expected_cases):
                evidence["error"] = error
                ok = False
        workers.append(
            {
                "worker": len(workers),
                "returncode": worker.returncode,
                "complete": ok,
                "results": str(result.relative_to(root)) if result.is_file() else None,
                **evidence,
            }
        )
        if ok:
            completed.append(json.loads(result.read_text()))
        elif mode:
            break  # A failed fit cannot supply its paired worker or a complete campaign.

    summary = {}
    for mode in dict.fromkeys(key for run in completed for key in run):
        fits = [run[mode] for run in completed if mode in run]
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
    if workload == "loader" or cuda:
        sources.append(Path(__file__).with_name("file_cache.py"))
    if workload == "l01":
        sources.append(REPOSITORY / "e2e/datasets/aml.py")
    if workload == "cuda" or cuda:
        sources.append(Path(__file__).with_name("cuda_metrics.py"))
    if cuda:
        sources.append(Path(__file__).with_name("l01_model.py"))
    if workload == "cuda":
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
        **({"length_planning": True, "training": False} if length_planning else {}),
        **setup,
        **({"seed": 0, "file_page_cache": "uncontrolled"} if workload == "l01" else {}),
        **({"file_page_cache": "cold-warm"} if cold_warm_cache else {}),
        **({"host_memory": host_memory} if host_memory else {}),
        "device": "cuda" if workload == "cuda" or cuda else "cpu",
        "stage_timing": "Inclusive caller wall time. Nested and native timings can overlap.",
        **({"banking": {"batch_size": 2}, "l01": L01_BATCHING}.get(workload, LOADER_BATCHING)),
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
    expected = repetitions * len(modes)
    if len(completed) != expected:
        raise SystemExit(f"{len(completed)}/{expected} workers completed. See {report}")
    return report


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--e2e-dir", type=Path, required=True)
    if argv and argv[0] in TIMING_CASES:
        parser.add_argument("--repetitions", type=int, default=3)
        if argv[0] == "l01":
            group = parser.add_mutually_exclusive_group()
            group.add_argument("--cuda", action="store_true")
            group.add_argument("--index-lifecycle", action="store_true")
            group.add_argument("--length-planning", action="store_true")
        if argv[0] in ("loader", "l01"):
            parser.add_argument(
                "--cgroup-parent", type=Path, help="delegated Linux cgroup v2 parent"
            )
        if argv[0] == "loader":
            parser.add_argument("--host-memory", choices=("kernel", "sampled"))
            parser.add_argument(
                "--cold-warm-cache",
                action="store_true",
                help="verify index page residency before each epoch (Linux)",
            )
        args = parser.parse_args(argv[1:])
        if args.repetitions < 1:
            parser.error("--repetitions must be positive")
        options = vars(args)
        if options.get("index_lifecycle") and args.repetitions != 3:
            parser.error("--index-lifecycle requires three workers (one build and two reopens)")
        if options.get("cuda"):
            options.update(host_memory="sampled", cold_warm_cache=True)
        if bool(options.get("host_memory")) != bool(options.get("cgroup_parent")):
            parser.error("--cgroup-parent must accompany --host-memory or l01 --cuda")
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
