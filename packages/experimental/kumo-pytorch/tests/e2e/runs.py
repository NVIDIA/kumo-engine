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

from e2e.datasets.hm import HMDataset, generate, sample_anchors
from tests.e2e.tiers import BY_NAME, Tier

REPOSITORY = Path(__file__).resolve().parents[2]
FLOWS = {"gnn": "e2e.flows.hm_gnn", "pragma": "e2e.flows.hm_pragma"}
MARKER = "run.json"
LOADER_CASE = "tests/test_native_anchor_handoff.py::test_native_cpu_loader_timing"
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


def time_loader(plan, table, work: Path, make_batches, *, setup, **batch_options) -> dict:
    """Measure two complete train epochs for a fresh fit and its saved artifact."""
    from kumo.pytorch import api
    from tests.e2e.loader_timing import install_stages, measure_epoch, stage_observer

    work.mkdir(parents=True, exist_ok=True)
    options = LOADER_BATCHING | batch_options
    results = {"setup": setup}
    artifact = work / "artifact"
    try:
        for mode in ("fresh", "reloaded"):
            with ExitStack() as tasks:
                with stage_observer(install_stages) as profile:
                    started = time.perf_counter_ns()
                    fit = {"fit_split": options["split"]}
                    if mode == "reloaded":
                        with profile.measure("artifact.load"):
                            fit = {"fitted": api.Artifact.load(artifact).fitted_state}
                    task = tasks.enter_context(plan.materialize(table, **fit))
                    materialize_ns = time.perf_counter_ns() - started
                results[mode] = {
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
                        record = measure_epoch(task, make_batches, epoch=epoch, **options)
                    results[mode]["epochs"].append({**record, "stages": profile.stages})
    finally:
        (work / "results.json").write_text(json.dumps(results) + "\n")
    return results


def _ranges(rows):
    """Summarize numeric metrics shared by the workers. Raw observations stay per worker."""
    metrics = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if all(isinstance(v, dict) for v in values):
            metrics[key] = _ranges(values)
        elif all(isinstance(v, (int, float)) for v in values):
            metrics[key] = {
                "median": statistics.median(values),
                "range": [min(values), max(values)],
            }
    return metrics


def _loader_campaign(directory: Path, repetitions: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="loader-", dir=directory.resolve()))
    report = root / "timings.json"
    completed = []
    workers = []
    for index in range(repetitions):
        work = root / str(index)
        worker = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", LOADER_CASE, "--e2e-dir", str(work)],
            cwd=REPOSITORY,
            env={**os.environ, "KUMO_PYTORCH_PROFILE_LOADER": "1"},
        )
        result = work / "loader" / "results.json"
        ok = worker.returncode == 0 and result.is_file()
        workers.append(
            {
                "worker": index,
                "returncode": worker.returncode,
                "complete": ok,
                "results": str(result.relative_to(root)) if result.is_file() else None,
            }
        )
        if ok:
            completed.append(json.loads(result.read_text()))

    summary = {}
    for mode in completed[0] if completed else ():
        fits = [run[mode] for run in completed]
        summary[mode] = _ranges(fits)
        if mode == "setup":
            continue
        summary[mode]["epochs"] = [
            _ranges(rows) for rows in zip(*(fit["epochs"] for fit in fits), strict=True)
        ]
    sources = [
        Path(__file__),
        Path(__file__).with_name("loader_timing.py"),
        REPOSITORY / LOADER_CASE.split("::")[0],
    ]
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
        "workload": LOADER_CASE,
        "device": "cpu",
        "stage_timing": "Inclusive caller wall time. Nested and native timings can overlap.",
        **LOADER_BATCHING,
        "revision": revision,
        "tracked_diff_sha256": tracked_diff,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            d.name: d.version
            for d in distributions()
            if d.name in ("torch", "pyarrow", "diskgraph")
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
    if argv[:1] == ["loader"]:
        parser.add_argument("--repetitions", type=int, default=3)
        args = parser.parse_args(argv[1:])
        if args.repetitions < 1:
            parser.error("--repetitions must be positive")
        print(_loader_campaign(args.e2e_dir, args.repetitions))
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
