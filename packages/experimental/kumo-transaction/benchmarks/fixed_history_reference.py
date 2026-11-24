# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-built history features under the lab's fixed XGBoost setting.

`final_eval.py --fixed-xgboost` skips the hand-built history arm. This scores
it on the same tasks, probe splits, and seeds, with `LAB_XGBOOST_PARAMS`:
`TabularFeaturizer(history=True)` (counts, amount statistics, and recency over
the window, plus the last event and the profile), early-stopped on the same
inner fold. It is context for the model arms, not one of them.

    uv run --all-extras python benchmarks/fixed_history_reference.py \\
        --run-dir outputs/proxy/stage3/causal --out outputs/proxy/history_reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from final_eval import split_tasks
from proxy_eval import proxy_tasks

from tfm.config import load_recipe_config
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.evaluation.tabular import TabularFeaturizer, _fit_predict_xgboost, fixed_xgboost
from tfm.runtime.data import prepare_data
from tfm.runtime.lab import LAB_XGBOOST_PARAMS

TASKS = ("history_window", "churn_6m", "ltv_positive", "default_12m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Holds seed_config.yaml.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    config = load_recipe_config(args.run_dir / "seed_config.yaml")
    data = prepare_data(config, fixture_dir=args.out.parent / f"{args.out.stem}_book")
    base = proxy_tasks(data, config, args.tasks)
    missing = sorted(set(args.tasks) - set(base))
    if missing:
        raise SystemExit(f"Cannot build the probes for {missing} on this data.")
    profiles = {p.entity_id: p for p in data.profiles}
    cfg = config.evaluation.tabular_baselines
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for split_seed in args.split_seeds:
        tasks = split_tasks(data, config, base, split_seed, args.train_fraction)
        for name, task in tasks.items():
            features = TabularFeaturizer.fit(
                task.train_sequences,
                profiles,
                top_k=cfg.top_k,
                coded_keys=tuple(config.features.categorical_keys),
                history=True,
            )
            x_tr = features.transform(task.train_sequences, profiles)
            x_ev = features.transform(task.eval_sequences, profiles)
            groups = [seq.entity_id for seq in task.train_sequences]
            tuned = fixed_xgboost(
                x_tr, task.y_train, groups, LAB_XGBOOST_PARAMS, seed=config.seed, n_jobs=cfg.n_jobs
            )
            scores = _fit_predict_xgboost(
                x_tr, task.y_train, x_ev, tuned, seed=config.seed, n_jobs=cfg.n_jobs
            )
            metrics = classification_metrics(
                task.y_eval, scores, k=config.evaluation.precision_at_k
            )
            row = {
                "split_seed": split_seed,
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "n_features": int(x_tr.shape[1]),
                "n_estimators": tuned.n_estimators,
            }
            per_seed.setdefault(name, []).append(row)
            print(json.dumps({"task": name, **row}), flush=True)
    summary = {
        name: {
            metric: {
                "median": float(np.median([r[metric] for r in rows])),
                "min": float(np.min([r[metric] for r in rows])),
                "max": float(np.max([r[metric] for r in rows])),
            }
            for metric in ("pr_auc", "roc_auc")
        }
        for name, rows in per_seed.items()
    }
    record = {
        "run_dir": str(args.run_dir),
        "arm": "tabular_xgboost",
        "xgboost_params": dict(LAB_XGBOOST_PARAMS),
        "train_fraction": args.train_fraction,
        "summary": summary,
        "per_seed": per_seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
