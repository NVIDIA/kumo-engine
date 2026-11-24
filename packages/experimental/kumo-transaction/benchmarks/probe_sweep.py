# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score probe protocol and readout variants offline from a `probe_cache.py` cache.

Every variant is XGBoost on the cached rows and scored on the full `eval` side
(PR-AUC, weighted by the cache's `w` when the side was sampled). Axes: legitimate rows per fraud on the train side (subsets of the
cached ratio), early stopping (`inner`: the recipe's user-disjoint 20% of the
train side, then a refit; `time`: the latest 20% of the train side by event
time, then a refit; `period`: the later `stop` side, no refit; `tuned`: the
kit's `tune_xgboost` search, `--trials` trials on the `inner` split, then a
refit; `period_tuned`: the same search space, each trial early-stopped and
scored on the `stop` side, the best trial's model kept as fitted), the embedding readout, PCA to 64 dimensions (fitted on the train side), and tree depth. The
`_noid` bases drop the high-cardinality merchant-name, merchant-city, and Zip
columns, the values a deep tree can memorize.

    uv run python benchmarks/probe_sweep.py --cache outputs/probe_cache/blueprint_v1_seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from tfm.evaluation.tabular import _fit_predict_xgboost, tune_xgboost

V1_PARAMS: dict[str, Any] = {
    "max_depth": 7,
    "learning_rate": 0.02,
    "min_child_weight": 2.8,
    "subsample": 0.87,
    "colsample_bytree": 0.76,
    "reg_lambda": 0.43,
    "reg_alpha": 0.09,
}


def _subsample(y: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = min(len(neg), int(round(ratio * len(pos))))
    neg = np.random.default_rng(seed).choice(neg, size=keep, replace=False)
    return np.sort(np.concatenate([pos, neg]))


ID_COLUMNS = ("cat:merchant_name", "cat:merchant_city", "num:zip")


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    x_stop: np.ndarray | None,
    y_stop: np.ndarray | None,
    seed: int,
    params: dict[str, Any],
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        **params,
        n_estimators=2000,
        early_stopping_rounds=50 if x_stop is not None else None,
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=seed,
    )
    if x_stop is None:
        model.fit(x, y, verbose=False)
    else:
        model.fit(x, y, eval_set=[(x_stop, y_stop)], verbose=False)
    return model


def _period_tuned(
    x: np.ndarray,
    y: np.ndarray,
    x_stop: np.ndarray,
    y_stop: np.ndarray,
    *,
    seed: int,
    trials: int,
) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    """`tune_xgboost`'s search space, selected on the later `stop` side."""
    import optuna

    ratio = float((y == 0).sum()) / max(float(y.sum()), 1.0)
    best: dict[str, Any] = {}

    def objective(trial: Any) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "scale_pos_weight": ratio
            if trial.suggest_categorical("balanced", [False, True])
            else 1.0,
        }
        model = _fit(x, y, x_stop, y_stop, seed, params)
        score = float(average_precision_score(y_stop, model.predict_proba(x_stop)[:, 1]))
        if score > best.get("validation_pr_auc", -1.0):
            best.update(model=model, params=params, validation_pr_auc=score)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return best["model"], {**best["params"], "validation_pr_auc": best["validation_pr_auc"]}


def _score(
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    stop: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    stopping: str,
    seed: int,
    params: dict[str, Any],
    trials: int = 0,
    chosen: dict[str, Any] | None = None,
) -> tuple[float, int]:
    chosen = {} if chosen is None else chosen
    x, y, users, times = train
    if stopping == "tuned":
        tuned = tune_xgboost(
            x, y, list(users), seed=seed, n_trials=trials, n_jobs=-1, device="cuda"
        )
        pred = _fit_predict_xgboost(x, y, test[0], tuned, seed=seed, n_jobs=-1, device="cuda")
        chosen.update(tuned.params, validation_pr_auc=tuned.validation_pr_auc)
        return float(average_precision_score(test[1], pred, sample_weight=test[2])), (
            tuned.n_estimators
        )
    if stopping == "period_tuned":
        model, picked = _period_tuned(x, y, *stop, seed=seed, trials=trials)
        chosen.update(picked)
        pred = model.predict_proba(test[0])[:, 1]
        return float(average_precision_score(test[1], pred, sample_weight=test[2])), int(
            model.best_iteration + 1
        )
    if stopping == "period":
        model = _fit(x, y, *stop, seed, params)
        trees = model.best_iteration + 1
    else:
        if stopping == "time":
            order = np.argsort(times, kind="stable")
            cut = int(len(order) * 0.8)
            fit_idx, stop_idx = order[:cut], order[cut:]
        else:
            folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            fit_idx, stop_idx = next(folds.split(x, y, users))
        fitted = _fit(x[fit_idx], y[fit_idx], x[stop_idx], y[stop_idx], seed, params)
        trees = fitted.best_iteration + 1
        model = xgb.XGBClassifier(
            **params,
            n_estimators=trees,
            tree_method="hist",
            device="cuda",
            random_state=seed,
        )
        model.fit(x, y, verbose=False)
    pred = model.predict_proba(test[0])[:, 1]
    return float(average_precision_score(test[1], pred, sample_weight=test[2])), int(trees)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ratios", default="5,10,20,40")
    parser.add_argument("--stopping", default="inner,period")
    parser.add_argument("--arms", help="comma-separated; default: the full list below")
    parser.add_argument("--max-depth", type=int, default=V1_PARAMS["max_depth"])
    parser.add_argument(
        "--fit-side",
        choices=("train", "stop"),
        default="train",
        help="`stop` fits on the validation period, the side nearest the test",
    )
    parser.add_argument("--trials", type=int, default=20, help="search trials per arm (`tuned`)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sides = {s: dict(np.load(args.cache / f"{s}.npz")) for s in ("train", "stop", "eval")}
    meta = json.loads((args.cache / "meta.json").read_text())
    seed = meta["seed"]
    params = {**V1_PARAMS, "max_depth": args.max_depth}
    for side in sides.values():
        for base in ("raw", "full"):
            names = meta["names"][base]
            keep = [i for i, n in enumerate(names) if not n.startswith(ID_COLUMNS)]
            side[f"{base}_noid"] = side[base][:, keep]

    pca: dict[str, PCA] = {}

    def reduce(readout: str, vec: np.ndarray) -> np.ndarray:
        if readout not in pca:
            fit = sides["train"][readout].astype(np.float32)
            pca[readout] = PCA(64, random_state=seed).fit(fit)
        return pca[readout].transform(vec)

    def features(side: str, arm: str) -> np.ndarray:
        data = sides[side]
        parts = []
        base, _, emb = arm.partition("+")
        parts.append(data[base])
        for name in filter(None, emb.split("&")):
            readout, _, how = name.partition("@")
            vec = data[readout].astype(np.float32)
            parts.append(reduce(readout, vec) if how == "pca64" else vec)
        return np.hstack(parts).astype(np.float32)

    arms = (
        args.arms.split(",")
        if args.arms
        else [
            "raw",
            "raw+hist_last",
            "raw+hist_last@pca64",
            "raw+hist_event",
            "raw+single_last",
            "raw+single_last@pca64",
            "raw+hist_last&single_last",
            "full",
            "full+hist_last",
            "full+single_last",
        ]
    )
    results: list[dict[str, Any]] = []
    for ratio in [float(r) for r in args.ratios.split(",")]:
        fit = args.fit_side
        idx = _subsample(sides[fit]["y"], ratio, seed)
        for stopping in args.stopping.split(","):
            if fit == "stop" and stopping.startswith("period"):
                raise SystemExit("--fit-side stop leaves no later side for period stopping.")
            bare: dict[str, float] = {}
            for arm in arms:
                train = (
                    features(fit, arm)[idx],
                    sides[fit]["y"][idx],
                    sides[fit]["user"][idx],
                    sides[fit]["time"][idx],
                )
                stop = (features("stop", arm), sides["stop"]["y"])
                test = (
                    features("eval", arm),
                    sides["eval"]["y"],
                    sides["eval"].get("w", np.ones(len(sides["eval"]["y"]))),
                )
                chosen: dict[str, Any] = {}
                ap, trees = _score(
                    train,
                    stop,
                    test,
                    stopping=stopping,
                    seed=seed,
                    params=params,
                    trials=args.trials,
                    chosen=chosen,
                )
                family = arm.partition("+")[0]
                if "+" not in arm:
                    bare.setdefault(family, ap)
                base = bare.get(family, bare.get("raw"))
                lift = (ap / base - 1) * 100 if base else float("nan")
                row = {"ratio": ratio, "stopping": stopping, "arm": arm, "pr_auc": ap}
                row.update(trees=trees, lift_pct=lift, seed=seed, fit_side=fit)
                row.update(chosen or {"max_depth": args.max_depth})
                results.append(row)
                print(
                    f"seed={seed} neg={ratio:>4g} stop={stopping:<6} "
                    f"depth={row['max_depth']} {arm:<30} "
                    f"PR-AUC {ap:.4f}  lift {lift:+6.1f}%  trees {trees}",
                    flush=True,
                )
    out = args.out or args.cache / "sweep.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
