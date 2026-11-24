# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score one checkpoint on every proxy task at several probe-split seeds.

The pretraining seed is fixed; what varies is which rows the probes and the
tabular arms train on, which is the cheap part of the variance. How a split is
resampled depends on how the task is split to begin with:

- A forecast task (`churn_6m`, ...) is entity-disjoint at shared evaluation
  points, so each seed draws a fresh entity split (`forecast_task(seed=...)`).
- `fraud` and `history_window` train on train-side histories and evaluate on
  test-side histories of held-out entities, fixed when the splice was drawn.
  Mixing the sides would train on post-cutoff rows, so each seed keeps the
  evaluation rows and trains on a seeded share of the training entities.

Every arm is the proxy's (`proxy_eval.py`) with the recipe's search budget
(`--xgboost-trials`, 30 by default, as the tabular arms use). `--raw-views`
adds `xgboost_raw_embedding:<view>` arms with other readouts. `--fixed-xgboost`
scores the raw arms with one fixed setting instead, as the lab notebooks do.
`--finetune <config>` also fine-tunes a copy of the model on each task with
that config's `fraud_task.finetune` (arm `finetune:<arm>`), on the same rows
and with fine-tuning seed `config.seed + split_seed`.

    uv run --all-extras python benchmarks/final_eval.py \\
        --run-dir outputs/proxy/stage5/causal_4x --out outputs/proxy/final.json \\
        --raw-views last_surprise
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from proxy_eval import DEFAULT_TASKS, proxy_eval, proxy_tasks
from seed_range import _commit

from tfm.adaptation.batching import select_rows
from tfm.adaptation.finetune import finetune_binary
from tfm.config import RecipeConfig, load_recipe_config
from tfm.runtime.common import apply_device_override
from tfm.runtime.data import PreparedData, prepare_data
from tfm.runtime.evaluate import ProbeTask, finetune_spec, forecast_task
from tfm.runtime.lab import LAB_XGBOOST_PARAMS
from tfm.runtime.pipeline import build_model
from tfm.training.checkpointing import load_checkpoint
from tfm.training.device import resolve_device

HEADLINE_METRICS = ("pr_auc", "roc_auc")


def train_subsample(task: ProbeTask, fraction: float, seed: int) -> ProbeTask:
    """`task` trained on a seeded `fraction` of its training entities, all their rows."""
    entities = sorted({seq.entity_id for seq in task.train_sequences})
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(entities, size=max(1, round(fraction * len(entities))), replace=False))
    rows = [i for i, seq in enumerate(task.train_sequences) if seq.entity_id in keep]
    return dataclasses.replace(
        task,
        train_batch=select_rows(task.train_batch, rows),
        y_train=task.y_train[rows],
        train_sequences=tuple(task.train_sequences[i] for i in rows),
        train_vectors=None,
        eval_vectors=None,
    )


def split_tasks(
    data: PreparedData,
    config: RecipeConfig,
    base: dict[str, ProbeTask],
    split_seed: int,
    fraction: float,
) -> dict[str, ProbeTask]:
    out: dict[str, ProbeTask] = {}
    for name, task in base.items():
        if name in data.label_rows:
            fresh = forecast_task(
                data, config, name, data.label_rows[name], seed=config.seed + split_seed
            )
            if fresh is not None:
                out[name] = fresh
        else:
            out[name] = train_subsample(task, fraction, seed=split_seed)
    return out


def _arms(entry: dict[str, Any]) -> dict[str, dict[str, float]]:
    arms = {f"probe:{view}": m for view, m in entry["probe"].items()}
    for arm, metrics in entry.items():
        if arm.startswith(("xgboost_raw", "tabular_", "finetune:")) and isinstance(metrics, dict):
            arms[arm] = metrics
    return arms


def summarize(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Median, min, max per task, arm, and metric; lift and gain recovered per seed."""
    out: dict[str, Any] = {}
    tasks = sorted({t for r in per_seed for t in r["tasks"]})
    for task in tasks:
        entries = [r["tasks"][task] for r in per_seed if task in r["tasks"]]
        arms: dict[str, dict[str, list[float]]] = {}
        for entry in entries:
            for arm, metrics in _arms(entry).items():
                for metric in HEADLINE_METRICS:
                    arms.setdefault(arm, {}).setdefault(metric, []).append(float(metrics[metric]))
        table = {
            arm: {
                metric: {
                    "median": float(np.median(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "values": v,
                }
                for metric, v in metrics.items()
            }
            for arm, metrics in arms.items()
        }
        lifts: dict[str, Any] = {}
        raw = arms.get("xgboost_raw", {}).get("pr_auc")
        tab = arms.get("tabular_xgboost", {}).get("pr_auc")
        for arm, metrics in arms.items():
            if not arm.startswith(("xgboost_raw_embedding", "finetune:")) or raw is None:
                continue
            lifts[arm] = {
                "pr_auc_lift": [e - r for e, r in zip(metrics["pr_auc"], raw, strict=True)]
            }
            if tab is not None:
                lifts[arm]["gain_recovered"] = [
                    (e - r) / (t - r) if t != r else float("nan")
                    for e, r, t in zip(metrics["pr_auc"], raw, tab, strict=True)
                ]
        out[task] = {"arms": table, "lift": lifts, "n_seeds": len(entries)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Holds seed_config.yaml.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--xgboost-trials", type=int, default=30)
    parser.add_argument(
        "--fixed-xgboost",
        action="store_true",
        help="Score every raw arm with tfm.runtime.lab.LAB_XGBOOST_PARAMS, no search "
        "(the lab notebooks' protocol); skips the hand-built history arms.",
    )
    parser.add_argument(
        "--finetune",
        type=Path,
        default=None,
        help="Recipe config whose fraud_task.finetune to fine-tune on every task.",
    )
    parser.add_argument("--raw-views", nargs="*", default=["last_surprise"])
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/proxy/cache_final"))
    parser.add_argument("--random-init", action="store_true", help="Skip loading model.pt.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = apply_device_override(
        load_recipe_config(args.run_dir / "seed_config.yaml"), args.device
    )
    device = resolve_device(config.runtime.device)
    started = time.perf_counter()
    data = prepare_data(config, fixture_dir=args.out.parent / f"{args.out.stem}_book")
    prepare_seconds = round(time.perf_counter() - started, 1)
    model = build_model(config, data.tokenizer.vocab_size)
    if not args.random_init:
        load_checkpoint(model, args.run_dir / "model.pt")
    base = proxy_tasks(data, config, args.tasks)
    missing = sorted(set(args.tasks) - set(base))
    if missing:
        raise SystemExit(f"Cannot build the probes for {missing} on this data.")
    fixed = dict(LAB_XGBOOST_PARAMS) if args.fixed_xgboost else None
    ft = finetune_spec(load_recipe_config(args.finetune)) if args.finetune else None
    per_seed: list[dict[str, Any]] = []
    for i, split_seed in enumerate(args.split_seeds):
        tasks = split_tasks(data, config, base, split_seed, args.train_fraction)
        t0 = time.perf_counter()
        result = proxy_eval(
            model,
            data,
            config,
            device,
            tasks=args.tasks,
            cache_dir=args.cache_dir / f"split{split_seed}_frac{args.train_fraction}",
            xgboost_trials=args.xgboost_trials,
            probe_tasks=tasks,
            raw_views=args.raw_views,
            diagnostics=i == 0,
            xgboost_params=fixed,
        )
        if ft is not None:
            for name, task in tasks.items():
                t1 = time.perf_counter()
                metrics, manifest = finetune_binary(
                    model,
                    task.train_batch,
                    task.eval_batch,
                    task.y_train,
                    task.y_eval,
                    ft,
                    device=device,
                    seed=config.seed + split_seed,
                    k=config.evaluation.precision_at_k,
                    precision=config.training.precision,
                )
                manifest.pop("adapted_modules", None)
                manifest["seconds"] = round(time.perf_counter() - t1, 1)
                result["tasks"][name][f"finetune:{ft.arm}"] = metrics
                result["tasks"][name]["finetune_manifest"] = manifest
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        result["split_seed"] = split_seed
        result["seconds"] = round(time.perf_counter() - t0, 1)
        per_seed.append(result)
        record = {
            "run_dir": str(args.run_dir),
            "random_init": args.random_init,
            "commit": _commit(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "prepare_seconds": prepare_seconds,
            "train_fraction": args.train_fraction,
            "xgboost_trials": 0 if fixed else args.xgboost_trials,
            "xgboost_params": fixed,
            "finetune": dataclasses.asdict(ft) if ft else None,
            "summary": summarize(per_seed),
            "per_seed": per_seed,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"split_seed": split_seed, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
