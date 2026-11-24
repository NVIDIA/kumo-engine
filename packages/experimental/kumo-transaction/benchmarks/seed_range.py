# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one recipe config end to end at several seeds and summarize the metrics.

The seed changes pretraining, the probe's entity split, and fine-tuning, so the
range covers the whole pipeline, not just the probe.

    export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
    uv run python benchmarks/seed_range.py \\
        --config recipes/blueprint_v2/config.yaml --seeds 42 43 44 \\
        --run-root outputs/blueprint_v2/seeds \\
        --summary benchmarks/results/blueprint_v2_seeds.json

Run directories stay under `--run-root` (not committed); `--summary` gets the
per-seed metrics, their min / median / max, and the hardware and timings.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from tfm.config import load_recipe_config
from tfm.runtime.pipeline import run_evaluate, run_report, run_train, setup_run


def _commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def _seed_config(config_path: Path, seed: int, overrides: dict[str, Any], out: Path) -> Path:
    raw = load_recipe_config(config_path).model_dump(mode="json")
    raw["seed"] = seed
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


def run_seed(
    config_path: Path,
    seed: int,
    run_root: Path,
    overrides: dict[str, Any],
    device: str | None,
    *,
    reuse_checkpoint: bool = False,
) -> dict[str, Any]:
    run_dir = run_root / f"seed_{seed}"
    seeded = _seed_config(config_path, seed, overrides, run_dir / "seed_config.yaml")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    artifacts = setup_run(config_path=seeded, run_dir=run_dir, device=device)
    prepared = time.perf_counter()
    stats_file = run_dir / "train_stats.json"
    if reuse_checkpoint and (run_dir / "model.pt").exists() and stats_file.exists():
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
    else:
        stats = run_train(artifacts)
    trained = time.perf_counter()
    metrics = run_evaluate(artifacts)
    run_report(artifacts, metrics)
    finished = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "train_stats": stats,
        "seconds": {
            "prepare": round(prepared - started, 1),
            "train": round(trained - prepared, 1),
            "evaluate": round(finished - trained, 1),
        },
        "peak_gpu_memory_gib": None if peak is None else round(peak, 2),
        "n_parameters": sum(p.numel() for p in artifacts.model.parameters()),
        "metrics": metrics,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """min / median / max per task and metric, over the seeds that report it."""
    values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for task, metrics in run["metrics"].items():
            for name, value in metrics.items():
                if isinstance(value, int | float):
                    values.setdefault(task, {}).setdefault(name, []).append(float(value))
    return {
        task: {
            name: {
                "n": len(xs),
                "min": min(xs),
                "median": statistics.median(xs),
                "max": max(xs),
            }
            for name, xs in metrics.items()
        }
        for task, metrics in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config key for every seed, e.g. training.max_steps=20 (YAML value).",
    )
    parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="Skip training for a seed whose run directory already has model.pt; re-evaluate it.",
    )
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    overrides = {
        key: yaml.safe_load(value) for key, value in (item.split("=", 1) for item in args.set)
    }
    if args.worker_out is not None:
        (seed,) = args.seeds
        run = run_seed(
            args.config,
            seed,
            args.run_root,
            overrides,
            args.device,
            reuse_checkpoint=args.reuse_checkpoint,
        )
        args.worker_out.write_text(json.dumps(run, default=str), encoding="utf-8")
        return
    runs = []
    for seed in args.seeds:
        # One process per seed: host memory from a seed's data, probes, and
        # baselines is returned before the next, and timings do not depend on
        # the seeds that ran first.
        worker_out = args.run_root / f"seed_{seed}" / "seed_range_run.json"
        worker_out.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, "--config", str(args.config), "--seeds", str(seed)]
        command += ["--run-root", str(args.run_root), "--summary", str(args.summary)]
        command += ["--worker-out", str(worker_out)]
        command += [] if args.device is None else ["--device", args.device]
        command += ["--reuse-checkpoint"] if args.reuse_checkpoint else []
        for item in args.set:
            command += ["--set", item]
        subprocess.run(command, check=True)
        runs.append(json.loads(worker_out.read_text(encoding="utf-8")))
        # Write after every seed so a long sweep leaves partial evidence.
        record = {
            "config": str(args.config),
            "commit": _commit(),
            "overrides": overrides,
            "seeds": [r["seed"] for r in runs],
            "hardware": {
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            "summary": summarize(runs),
            "runs": runs,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
