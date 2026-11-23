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
Use ``hi-small --host-memory sampled --cgroup-parent DIR`` to compare fresh fits
on 25k and 400k train anchors for all five HI Small bindings (three pairs each).
Use ``aml --cgroup-parent DIR`` for all four L01 modes and 15 HI Small pairs
on one clean candidate revision. Pass ``--sudo`` if the two cgroup supervisors
require root. The linked measurements are descriptive.
Use ``binding-train BINDING --e2e-dir DIR --output FILE`` on Linux to train the
full registered training split for two epochs on CUDA. Stage the verified
source under ``KUMO_PYTORCH_DATA_ROOT`` first. Ingestion includes source staging
(``KUMO_PYTORCH_STAGE_SECONDS``, zero for pre-staged sources), re-verification,
projection and DiskGraph preparation. RAM is process peak RSS.
SSD usage includes the staged source and live prepared tables and index.
Use ``binding-throughput BINDING`` with the same arguments to measure 100 eligible
updates after 10 warmup updates on a pinned extract. Its report times extract staging,
verification, DiskGraph preparation and fitting separately.
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
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from e2e import E01_MANIFEST, L01_MANIFEST
from tests.e2e.tiers import BY_NAME, Tier

if TYPE_CHECKING:
    from e2e.datasets.hm import HMDataset

REPOSITORY = Path(__file__).resolve().parents[2]
FLOWS = {"gnn": "e2e.flows.hm_gnn", "pragma": "e2e.flows.hm_pragma"}
MARKER = "run.json"
TIMING_CASES = {
    "loader": "tests/test_native_anchor_handoff.py::test_native_cpu_loader_timing",
    "l01": "tests/e2e/test_aml.py::test_li_medium_l01_loader",
    "hi-small": "tests/e2e/test_aml.py::test_hi_small_fitting_memory",
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
AML_MODES = {
    "l01-cpu": ("l01",),
    "l01-index": ("l01", "--index-lifecycle"),
    "l01-length": ("l01", "--length-planning"),
    "l01-cuda": ("l01", "--cuda"),
    "hi-small": ("hi-small", "--host-memory", "sampled"),
}
HI_SMALL_BINDINGS = ("E01", "E02", "E03", "E04", "E05")


def dataset(root: Path, tier: Tier) -> HMDataset:
    from e2e.datasets.hm import generate

    return generate(tier.scale, 0, root / f"hm-{tier.scale.lower()}")


def training_data(hm: HMDataset, root: Path, tier: Tier) -> HMDataset:
    """What the flows train on: the dataset, or the tier's sample of its anchors."""

    if not tier.sample:
        return hm
    from e2e.datasets.hm import sample_anchors

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
    from tests.e2e.loader_timing import correctness_stage

    shutil.rmtree(work, ignore_errors=True)
    started = time.perf_counter()
    with correctness_stage("training"):
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


def _revision():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPOSITORY)
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "tracked_diff_sha256": None}
    return {
        "revision": revision,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def _provenance(case):
    versions = {
        d.name: d.version
        for d in distributions()
        if d.name in ("torch", "pyarrow", "diskgraph", "torch-geometric")
    }
    return {**_revision(), "workload": case, "platform": platform.platform(), "versions": versions}


def _hi_small_fits(e2e_dir, cgroup_parent, host_memory):
    from tests.e2e.memory_worker import run_isolated, worker_identity

    e2e_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="hi-small-", dir=e2e_dir.resolve()))
    report = root / "timings.json"
    output = report.open("x")
    cleanup_user = {}
    if os.geteuid() == 0:
        uid, gid = worker_identity()
        os.fchown(output.fileno(), uid, gid)
        os.chown(root, uid, gid)
        cleanup_user = {"user": uid, "group": gid, "extra_groups": []}
    env = dict(os.environ, KUMO_PYTORCH_PROFILE_HI_SMALL="1")
    case = TIMING_CASES["hi-small"]
    data = {
        **_provenance(case),
        "num_threads": 1,
        "source_pin_sha256": hashlib.sha256(E01_MANIFEST.read_bytes()).hexdigest(),
        "host_memory": host_memory,
        "sample_interval_ns": 50_000_000,
        "complete": False,
        "workers": [],
        "summary": {},
    }
    groups = set()
    sizes = (25_000, 400_000)
    metrics = ("peak_host_bytes", "peak_anonymous_bytes")
    try:
        for binding in HI_SMALL_BINDINGS:
            identity = None
            anchor_hashes = {}
            peaks = {size: [] for size in sizes}
            for repeat in range(3):
                pair = root / binding / str(repeat)
                for size in ("index-setup", *(sizes if repeat % 2 == 0 else reversed(sizes))):
                    work = pair / str(size)
                    result = work / "results.json"
                    command = [sys.executable, "-m", "pytest", "-q", case, "--e2e-dir", str(work)]
                    print(f"{binding} pair {repeat + 1}, {size}", flush=True)
                    worker = run_isolated(
                        command,
                        parent=cgroup_parent,
                        method=host_memory,
                        cwd=REPOSITORY,
                        env=env,
                        timeout=7200,
                    )
                    data["workers"].append(
                        {"results": str(result.relative_to(root)), "returncode": worker.returncode}
                    )
                    if worker.returncode or not result.is_file():
                        raise RuntimeError(
                            f"Missing successful {binding}/{repeat}/{size} worker: {report}"
                        )
                    row = json.loads(result.read_text())
                    common = tuple(
                        row[k]
                        for k in ("source_sha256", "contract", "prefix_sha256", "encoder_cutoff")
                    )
                    if identity is None:
                        identity = common
                    assert row["binding"] == binding and identity == common
                    if size == "index-setup":
                        continue
                    assert row["anchors"] == size
                    digest = row["anchor_sha256"]
                    assert anchor_hashes.setdefault(size, digest) == digest
                    memory = row["host_memory"]
                    assert memory["complete"] and memory["method"] == host_memory
                    assert memory["sample_interval_ns"] == data["sample_interval_ns"]
                    assert memory["cgroup"] not in groups
                    groups.add(memory["cgroup"])
                    assert all(memory[k] > 0 for k in metrics)
                    peaks[size].append({k: memory[k] for k in metrics})
                subprocess.run(["/bin/rm", "-r", "--", pair / "index"], check=True, **cleanup_user)
            small, large = (_ranges(peaks[size]) for size in sizes)
            data["summary"][binding] = {
                "25000": small,
                "400000": large,
                "growth_ratio": {k: large[k]["median"] / small[k]["median"] for k in metrics},
            }
        data["complete"] = True
    finally:
        with output:
            json.dump(data, output, indent=2)
    return report


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
    metadata = {
        **_provenance(case),
        **({"length_planning": True, "training": False} if length_planning else {}),
        **setup,
        **({"seed": 0, "file_page_cache": "uncontrolled"} if workload == "l01" else {}),
        **({"file_page_cache": "cold-warm"} if cold_warm_cache else {}),
        **({"host_memory": host_memory} if host_memory else {}),
        "device": "cuda" if workload == "cuda" or cuda else "cpu",
        "stage_timing": "Inclusive caller wall time. Nested and native timings can overlap.",
        **({"banking": {"batch_size": 2}, "l01": L01_BATCHING}.get(workload, LOADER_BATCHING)),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "python": platform.python_version(),
        "workers": workers,
        "summary": summary,
    }
    report.write_text(json.dumps(metadata, indent=2) + "\n")
    expected = repetitions * len(modes)
    if len(completed) != expected:
        raise SystemExit(f"{len(completed)}/{expected} workers completed. See {report}")
    return report


def _aml_evidence(mode, path, revision):
    from tests.e2e.binding_report import require_fields

    data = json.loads(path.read_text())
    if (
        not revision
        or data["revision"] != revision
        or data["tracked_diff_sha256"] is not None
        or data["workload"] != TIMING_CASES[AML_MODES[mode][0]]
    ):
        raise ValueError(f"Wrong candidate or workload in {path}")
    workers = data["workers"]
    if mode == "hi-small":
        prepared = json.loads((path.parent / "prepared/identity.json").read_text())
        pin = hashlib.sha256(E01_MANIFEST.read_bytes()).hexdigest()
        require_fields(data, {"source_pin_sha256": pin}, path)
        require_fields(prepared, {"source_manifest_sha256": pin}, path)
        expected = {
            f"{binding}/{rep}/{stage}/results.json"
            for binding, rep, stage in product(
                HI_SMALL_BINDINGS, range(3), ("index-setup", "25000", "400000")
            )
        }
        ok = data["complete"] and set(data["summary"]) == set(HI_SMALL_BINDINGS)
    else:
        require_fields(data, L01_BATCHING, path)
        lifecycle = mode == "l01-index"
        fits = ("fresh",) if lifecycle else ("fresh", "reloaded")
        expected = {f"{rep}/{fit}/l01/results.json" for rep, fit in product(range(3), fits)}
        ok = data["device"] == ("cuda" if mode == "l01-cuda" else "cpu")
        ok &= data.get("length_planning", False) == (mode == "l01-length")
        if lifecycle:
            ok &= [w.get("index_action") for w in workers] == ["build", "reopen", "reopen"]
    if not ok or len(workers) != len(expected) or {w["results"] for w in workers} != expected:
        raise ValueError(f"Incomplete {mode} coverage in {path}")
    for worker in workers:
        result = path.parent / worker["results"]
        if worker["returncode"] or not result.is_file():
            raise ValueError(f"Missing successful worker for {result}")
        raw = json.loads(result.read_text())
        if mode == "hi-small":
            binding, _, stage, _ = Path(worker["results"]).parts
            identity = prepared["bindings"][binding]
            require_fields(
                raw,
                {
                    "binding": binding,
                    "prefix_sha256": identity["prefix_sha256"],
                    "encoder_cutoff": identity["encoder_cutoff_seconds"],
                },
                result,
            )
            if stage != "index-setup":
                require_fields(
                    raw,
                    {"anchors": int(stage), "anchor_sha256": identity["anchors_sha256"][stage]},
                    result,
                )
                require_fields(
                    raw["host_memory"], {"complete": True, "method": data["host_memory"]}, result
                )
        else:
            rep, fit, _, _ = Path(worker["results"]).parts
            require_fields(worker, {"fit": fit}, result)
            expected_setup = {
                "source_sha256": hashlib.sha256(L01_MANIFEST.read_bytes()).hexdigest(),
                "split_counts": dict.fromkeys(("train", "valid", "test"), 512),
            }
            if mode == "l01-index":
                expected_setup.update(
                    index_ingests=int(rep == "0"), index_action="build" if rep == "0" else "reopen"
                )
            require_fields(raw["setup"], expected_setup, result)
            if not worker["complete"]:
                raise ValueError(f"Incomplete worker in {result}")
            check_l01_epochs(
                raw[fit]["epochs"],
                cuda=mode == "l01-cuda",
                length_planning=mode == "l01-length",
                where=result,
            )


def check_l01_epochs(epochs, *, cuda, length_planning, where):
    from tests.e2e.binding_report import require_fields

    if [e["epoch"] for e in epochs] != [0, 1]:
        raise ValueError(f"Incomplete epochs in {where}")
    for i, epoch in enumerate(epochs):
        expected = dict(complete=True, samples=512, batches=128, padded=0, batch_sizes=[4] * 128)
        if not epoch["native_totals"] or epoch["post_warmup"]["batches"] != 126:
            raise ValueError(f"Missing native or post-warmup evidence in {where}")
        if length_planning:
            if "file_page_cache" in epoch or "train_step_ns" in epoch:
                raise ValueError(f"Unexpected cache conditioning or training in {where}")
            expected.update(
                lengths_computed=(512, 0)[i],
                length_cache_hits=(0, 512)[i],
                length_cache_reused=bool(i),
                merged_history_batches=128,
                events=10_293,
            )
        if cuda:
            expected.update(eligible_amount=512, eligible_currency=512, optimizer_updates=128)
            require_fields(
                epoch["file_page_cache"], dict(mode=("cold", "warm")[i], status="verified"), where
            )
            require_fields(epoch["cuda"], {"complete": True}, where)
            if epoch["cuda"]["steady_samples"] < 2 or len(epoch["cuda"]["transfers"]) != 128:
                raise ValueError(f"Incomplete GPU measurements in {where}")
            require_fields(epoch["host_memory"], {"complete": True}, where)
        require_fields(epoch, expected, where)


def _aml_campaign(e2e_dir, cgroup_parent, sudo=False):
    candidate = _revision()
    revision = candidate["revision"]
    if (
        not revision
        or candidate["tracked_diff_sha256"]
        or os.getenv("GITHUB_SHA", revision) != revision
    ):
        raise ValueError("The AML campaign requires a clean checkout of the candidate revision")
    e2e_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="aml-", dir=e2e_dir.resolve()))
    report = {"revision": revision, "complete": False, "reports": {}, "hi_small_pairs": []}
    try:
        for mode, args in AML_MODES.items():
            directory = root / mode
            command = [sys.executable, "-m", "tests.e2e.runs", *args, "--e2e-dir", str(directory)]
            if mode in ("hi-small", "l01-cuda"):
                command.extend(["--cgroup-parent", str(cgroup_parent)])
                if sudo:
                    command = [
                        "sudo",
                        "-n",
                        "--",
                        "env",
                        f"PATH={os.environ['PATH']}",
                        f"KUMO_PYTORCH_DATA_ROOT={os.environ['KUMO_PYTORCH_DATA_ROOT']}",
                        *command,
                    ]
            print(f"Running {mode}", flush=True)
            subprocess.run(command, cwd=REPOSITORY, check=True)
            (path,) = directory.glob("*/timings.json")
            _aml_evidence(mode, path, revision)
            report["reports"][mode] = str(path.relative_to(root))
        report["hi_small_pairs"] = [
            {"binding": binding, "repetition": rep}
            for binding, rep in product(HI_SMALL_BINDINGS, range(3))
        ]
        report["complete"] = True
    finally:
        (root / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return root


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--e2e-dir", type=Path, required=True)
    if argv and argv[0] in ("binding-train", "binding-throughput"):
        from tests.e2e.binding_plan import NIGHTLY_TRAINING
        from tests.e2e.full_binding_run import run_full_binding

        parser.add_argument("binding", choices=NIGHTLY_TRAINING)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        run_full_binding(**vars(args), bounded=argv[0] == "binding-throughput")
        print(args.output)
        return
    if argv and argv[0] == "aml":
        parser.add_argument("--cgroup-parent", type=Path, required=True)
        parser.add_argument(
            "--sudo", action="store_true", help="elevate the two cgroup supervisors"
        )
        args = parser.parse_args(argv[1:])
        if sys.platform != "linux":
            parser.error("aml requires a Linux GPU worker with delegated memory cgroups")
        try:
            print(_aml_campaign(**vars(args)) / "campaign.json")
        except subprocess.CalledProcessError as error:
            code = error.returncode
            raise SystemExit(code if code >= 0 else 128 - code) from error
        return
    if argv and argv[0] in TIMING_CASES:
        parser.add_argument("--repetitions", type=int, default=3)
        if argv[0] == "l01":
            group = parser.add_mutually_exclusive_group()
            group.add_argument("--cuda", action="store_true")
            group.add_argument("--index-lifecycle", action="store_true")
            group.add_argument("--length-planning", action="store_true")
        if argv[0] in ("loader", "l01", "hi-small"):
            parser.add_argument(
                "--cgroup-parent", type=Path, help="delegated Linux cgroup v2 parent"
            )
        if argv[0] in ("loader", "hi-small"):
            parser.add_argument(
                "--host-memory", choices=("kernel", "sampled"), required=argv[0] == "hi-small"
            )
        if argv[0] == "loader":
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
        if argv[0] == "hi-small":
            if args.repetitions != 3:
                parser.error("hi-small requires three pairs per binding")
            print(_hi_small_fits(args.e2e_dir, args.cgroup_parent, args.host_memory))
            return
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
