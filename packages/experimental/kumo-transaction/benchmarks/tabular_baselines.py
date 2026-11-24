# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the tabular baselines on a recipe's probe rows at several seeds, with no model.

The baselines never read an encoder, so one run per seed serves every record
built from the same config's data: the rows, labels, entity splits, and
histories are exactly those `evaluate_run` probes. Needs the `baselines` extra.

    export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
    uv run --extra baselines python benchmarks/tabular_baselines.py \\
        --config benchmarks/ablations/blueprint_v2_lab.yaml --seeds 42 43 44 \\
        --summary benchmarks/results/blueprint_v2_hybrid_lab_splice_tabular_baselines.json

`--summary` gets the per-seed metrics, their min / median / max, the selected
XGBoost trial per task, and timings. Nothing else is written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path
from typing import Any

from seed_range import _commit, summarize

from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import InsufficientLabelSupportError
from tfm.runtime.data import prepare_data
from tfm.runtime.evaluate import (
    ProbeTask,
    forecast_task,
    fraud_entity_split,
    fraud_task,
    history_window_tasks,
    tabular_baselines,
)


def _config(path: Path, seed: int, trials: int, n_jobs: int) -> RecipeConfig:
    raw = load_recipe_config(path).model_dump(mode="json")
    raw["seed"] = seed
    raw["evaluation"]["tabular_baselines"] = {
        "enabled": True,
        "top_k": raw["evaluation"]["tabular_baselines"]["top_k"],
        "xgboost_trials": trials,
        "n_jobs": n_jobs,
    }
    return RecipeConfig.model_validate(raw)


def run_seed(
    path: Path, seed: int, trials: int, n_jobs: int, fixture_dir: Path | None
) -> dict[str, Any]:
    config = _config(path, seed, trials, n_jobs)
    started = time.perf_counter()
    data = prepare_data(config, fixture_dir)
    prepared = time.perf_counter()
    tasks: dict[str, ProbeTask] = {}
    skipped: dict[str, str] = {}
    try:
        split = fraud_entity_split(data, config.seed)
    except InsufficientLabelSupportError as exc:
        split, skipped["fraud"] = None, str(exc)
    if split is not None:
        tasks["fraud"] = fraud_task(data, split)
    for name, rows in data.label_rows.items():
        probe = forecast_task(data, config, name, rows, seed=config.seed)
        if probe is None:
            skipped[name] = "insufficient support or rows predate train cutoff"
        else:
            tasks[name] = probe
    if config.evaluation.history_window.enabled and split is not None:
        history = history_window_tasks(data, config, split).get("history_window")
        if history is not None:
            tasks["history_window"] = history
    metrics, manifest = tabular_baselines(tasks, data, config)
    finished = time.perf_counter()
    return {
        "seed": seed,
        "seconds": {
            "prepare": round(prepared - started, 1),
            "baselines": round(finished - prepared, 1),
        },
        "tasks": {name: task.summary() for name, task in tasks.items()},
        "skipped": skipped,
        "tabular_baselines": manifest,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per task.")
    parser.add_argument("--n-jobs", type=int, default=8, help="XGBoost threads.")
    parser.add_argument(
        "--fixture-dir", type=Path, default=None, help="Where a generated fixture book goes."
    )
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            runs.append(run_seed(args.config, seed, args.trials, args.n_jobs, args.fixture_dir))
        record = {
            "config": str(args.config),
            "commit": _commit(),
            "trials": args.trials,
            "seeds": [r["seed"] for r in runs],
            "hardware": {
                "cpu_count": os.cpu_count(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "summary": summarize(runs),
            "runs": runs,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
