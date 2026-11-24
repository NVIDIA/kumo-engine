# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast proxy evaluation of a pretrained encoder, for choosing between configs.

The full evaluation (every probe, LoRA on every task, 30-trial XGBoost searches
per arm) takes about 26 minutes at lab size. A search over pretraining settings
needs a cheaper signal. For each checkpoint this computes:

- **held-out losses**: each objective's loss on the validation-side sequences,
  whose last events postdate `train_end` and so were never pretrained on, and
  the next-event cross-entropy per key;
- **readout probes**: standardized, class-balanced logistic regression on
  several readouts of one forward pass (last valid slot, mean, profile slot,
  last+mean, and the last slot of the penultimate layer), per task. A hybrid
  model also gets `surprise`, its own next-event cross-entropy per key on each
  row's latest 1 and 8 events (each scored from its prefix only), and
  `last_surprise`, that joined to the last slot;
- **`xgboost_raw_embedding`**: a short Optuna search on the raw fields plus
  the embedding.

The model-independent arms (`xgboost_raw`, `tabular_xgboost`) are computed
once per data config and cached. Selection uses the proxy; a result is
recorded only from the full evaluation.

    uv run --all-extras python benchmarks/proxy_eval.py \\
        --run-dir outputs/cache_epoch/tail/seed_42 --out outputs/proxy/tail.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tfm.adaptation.batching import minibatches
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import InsufficientLabelSupportError
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.evaluation.surprise import next_event_token_ce, surprise_features
from tfm.evaluation.tabular import (
    TabularFeaturizer,
    _fit_predict_xgboost,
    evaluate_raw_embedding_arms,
    tune_xgboost,
)
from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
from tfm.layers.profile_history import ProfileHistoryModel
from tfm.objectives.mlm import apply_mlm_mask
from tfm.objectives.next_event_fields import next_event_targets
from tfm.pooling.last_token import LastTokenPoolModule
from tfm.pooling.mean import MeanPoolModule
from tfm.runtime.data import PreparedData
from tfm.runtime.evaluate import (
    ProbeTask,
    forecast_task,
    fraud_entity_split,
    fraud_task,
    history_window_tasks,
    tabular_baselines,
)

DEFAULT_TASKS = ("history_window", "fraud", "churn_6m")
_LAST = LastTokenPoolModule()
_MEAN = MeanPoolModule()


def proxy_tasks(
    data: PreparedData, config: RecipeConfig, names: Sequence[str]
) -> dict[str, ProbeTask]:
    """The same rows `evaluate_run` probes, for the requested tasks only."""
    tasks: dict[str, ProbeTask] = {}
    try:
        split = fraud_entity_split(data, config.seed)
    except InsufficientLabelSupportError:
        split = None
    if split is not None and "fraud" in names:
        tasks["fraud"] = fraud_task(data, split)
    if split is not None and "history_window" in names:
        history = history_window_tasks(data, config, split).get("history_window")
        if history is not None:
            tasks["history_window"] = history
    for name, rows in data.label_rows.items():
        if name in names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                probe = forecast_task(data, config, name, rows, seed=config.seed)
            if probe is not None:
                tasks[name] = probe
    return {name: tasks[name] for name in names if name in tasks}


def readouts(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    *,
    minibatch_size: int = 64,
    key_ids: Sequence[int] = (),
) -> dict[str, np.ndarray]:
    """Several readouts from one forward pass, in fp32.

    A `ProfileEventDecoderModel` gives `last` (its default), `mean`, `profile`,
    `last_mean`, and `penultimate_last`, and with `key_ids` also `surprise`
    and `last_surprise` (see `tfm.evaluation.surprise`). A `ProfileHistoryModel` gives PRAGMA's
    `usr`, the final event's `last`, both as `usr_last`, and `mean`. Any other
    model gives `default`, its own configured readout.
    """
    model.eval()
    model.to(device)
    out: dict[str, list[np.ndarray]] = {}
    hybrid = isinstance(model, ProfileEventDecoderModel)
    if isinstance(model, ProfileHistoryModel):
        # Its forward computes MLM logits over the vocabulary for every token.
        minibatch_size = min(minibatch_size, 16)
    captured: list[torch.Tensor] = []
    hook = None
    if hybrid and len(model.decoder.blocks) >= 2:
        hook = model.decoder.blocks[-2].register_forward_hook(lambda _m, _i, o: captured.append(o))
    try:
        with torch.no_grad():
            for chunk in minibatches(batch, minibatch_size):
                chunk = chunk.to(device)
                captured.clear()
                if hybrid:
                    states, mask, _tokens = model.encode(chunk)
                    last, mean = _LAST(states, mask), _MEAN(states, mask)
                    views = {
                        "last": last,
                        "mean": mean,
                        "profile": states[:, 0],
                        "last_mean": torch.cat([last, mean], dim=-1),
                    }
                    if captured:
                        views["penultimate_last"] = _LAST(model.decoder.norm(captured[0]), mask)
                    if key_ids:
                        views["surprise"] = surprise_features(model, chunk, states, key_ids)
                        views["last_surprise"] = torch.cat([last, views["surprise"]], dim=-1)
                elif isinstance(model, ProfileHistoryModel):
                    states = model(chunk, compute_loss=False).hidden_states
                    assert states is not None
                    events, event_mask = states[:, 1:], chunk.event_mask
                    last = _LAST(events, event_mask)
                    views = {
                        "usr": states[:, 0],
                        "last": last,
                        "mean": _MEAN(events, event_mask),
                        "usr_last": torch.cat([states[:, 0], last], dim=-1),
                    }
                else:
                    output = model(chunk, compute_loss=False)
                    assert output.embeddings is not None
                    views = {"default": output.embeddings}
                for name, value in views.items():
                    out.setdefault(name, []).append(value.float().cpu().numpy())
    finally:
        if hook is not None:
            hook.remove()
    return {name: np.concatenate(parts).astype(np.float32) for name, parts in out.items()}


def heldout_losses(
    model: nn.Module,
    batch: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    max_rows: int = 1024,
    minibatch_size: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean loss per objective, and next-event cross-entropy per key, on `batch`.

    MLM masking is seeded, so two checkpoints are scored on the same masks.
    Per-key cross-entropy is reported on unmasked inputs, which is what an
    embedding sees and what the copy and unigram floors assume, and on masked
    inputs, which is what training optimizes.
    """
    from tfm.adaptation.batching import select_rows

    rows = list(range(min(max_rows, batch.batch_size)))
    batch = select_rows(batch, rows)
    model.eval()
    model.to(device)
    torch.manual_seed(seed)
    totals: dict[str, float] = {}
    n = 0
    per_key: dict[str, tuple[dict[int, float], dict[int, int]]] = {
        "unmasked": ({}, {}),
        "masked": ({}, {}),
    }
    with torch.no_grad():
        for chunk in minibatches(batch, minibatch_size):
            chunk = chunk.to(device)
            output = model(chunk, compute_loss=True)
            for name, value in (output.losses or {"loss": output.loss}).items():
                assert value is not None
                totals[name] = totals.get(name, 0.0) + float(value)
            n += 1
            if not isinstance(model, ProfileEventDecoderModel):
                continue
            target = next_event_targets(chunk)
            inputs = {"unmasked": chunk}
            if model.mlm_spec is not None:
                inputs["masked"] = apply_mlm_mask(chunk, model.mlm_spec)
            for view, work in inputs.items():
                states, _mask, _tokens = model.encode(work)
                ce = next_event_token_ce(model, states, target)
                sums, counts = per_key[view]
                for key in target.key_ids.unique().tolist():
                    sel = target.key_ids == key
                    sums[key] = sums.get(key, 0.0) + float(ce[sel].sum())
                    counts[key] = counts.get(key, 0) + int(sel.sum())
    names = getattr(tokenizer, "id_to_token", {})

    def by_key(view: str) -> dict[str, float]:
        sums, counts = per_key[view]
        return {
            str(names.get(k, k)).removeprefix("KEY_"): sums[k] / counts[k] for k in sorted(sums)
        }

    return {
        "losses": {name: total / max(n, 1) for name, total in totals.items()},
        "next_event_ce_by_key": by_key("unmasked"),
        "next_event_ce_by_key_masked": by_key("masked"),
        "rows": len(rows),
    }


def _effective_rank(x: np.ndarray) -> float:
    """Participation ratio of the centered covariance spectrum: 1 is collapse, D is isotropy."""
    centered = x - x.mean(0, keepdims=True)
    eig = np.clip(np.linalg.eigvalsh(np.cov(centered, rowvar=False)), 0, None)
    return float(eig.sum() ** 2 / max(float((eig**2).sum()), 1e-12))


def _mean_cosine(x: np.ndarray) -> float:
    unit = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    sims = unit @ unit.T
    n = len(x)
    return float((sims.sum() - n) / max(n * (n - 1), 1))


def _ridge_r2(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 6)))
    from sklearn.model_selection import KFold

    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(model, x, y, cv=folds, scoring="r2")))


def embedding_health(
    model: nn.Module,
    data: PreparedData,
    config: RecipeConfig,
    device: torch.device,
    *,
    n_users: int = 1000,
    window: int = 64,
) -> dict[str, Any]:
    """Collapse, anisotropy, same-user retrieval, and recoverable aggregates.

    Each sampled user contributes their last two disjoint `window`-event
    windows before `train_end`. Retrieval ranks, for each user's older window,
    every user's newer window by centered cosine similarity. Both windows share
    the user's profile, so the `profile` readout's retrieval is the floor that
    profile identity alone gives. Needs a token cache.
    """
    from tfm.runtime.common import parse_duration

    cache = data.token_cache
    if cache is None:
        return {"skipped": "no token cache"}
    lookback = parse_duration(config.data.observation_window)
    everyone = cache.train_windows(
        train_end=data.cutoffs.train, lookback=lookback, window_events=window
    )
    full = everyone.hi - everyone.lo == window
    users, counts = np.unique(everyone.user[full], return_counts=True)
    eligible = users[counts >= 2]
    rng = np.random.default_rng(config.seed)
    chosen = set(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False).tolist())
    newer: dict[int, int] = {}
    older: dict[int, int] = {}
    for i in range(len(everyone) - 1, -1, -1):
        u = int(everyone.user[i])
        if u not in chosen or not full[i] or u in older:
            continue
        if u not in newer:
            newer[u] = i
        else:
            older[u] = i
    pairs = [(older[u], newer[u]) for u in sorted(older)]
    a_rows = everyone.take([p[0] for p in pairs])
    b_rows = everyone.take([p[1] for p in pairs])
    tokenizer: Any = data.tokenizer
    max_length = config.data.max_tokens
    a = readouts(model, cache.collate(a_rows, tokenizer, max_length=max_length), device)
    b_batch = cache.collate(b_rows, tokenizer, max_length=max_length)
    b = readouts(model, b_batch, device)
    out: dict[str, Any] = {"users": len(pairs), "window_events": window, "views": {}}
    targets = b_batch.numeric_features.numpy() if b_batch.numeric_features is not None else None
    for view in b:
        va, vb = a[view], b[view]
        center = np.concatenate([va, vb]).mean(0, keepdims=True)
        ua = (va - center) / np.clip(
            np.linalg.norm(va - center, axis=1, keepdims=True), 1e-12, None
        )
        ub = (vb - center) / np.clip(
            np.linalg.norm(vb - center, axis=1, keepdims=True), 1e-12, None
        )
        sims = ua @ ub.T
        ranks = (sims > np.diag(sims)[:, None]).sum(1)
        entry: dict[str, Any] = {
            "effective_rank": _effective_rank(vb),
            "dim": int(vb.shape[1]),
            "mean_cosine": _mean_cosine(vb),
            "retrieval_top1": float((ranks == 0).mean()),
            "retrieval_top10": float((ranks < 10).mean()),
            "retrieval_median_rank": float(np.median(ranks)),
        }
        if targets is not None and targets.size:
            entry["aggregate_r2"] = [
                _ridge_r2(vb, np.log1p(np.abs(targets[:, j])), config.seed)
                for j in range(targets.shape[1])
            ]
        out["views"][view] = entry
    return out


def _logistic(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(x_train, y_train.astype(int))
    return np.asarray(model.predict_proba(x_eval)[:, 1], dtype=np.float64)


def _data_key(config: RecipeConfig, names: Sequence[str]) -> str:
    raw = config.model_dump(mode="json")
    payload = {
        "data": raw["data"],
        "features": raw["features"],
        "seed": config.seed,
        "tasks": sorted(names),
        "tabular": raw["evaluation"]["tabular_baselines"],
        "history_rows": raw["evaluation"]["history_window"]["max_rows"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def reference_arms(
    tasks: Mapping[str, ProbeTask],
    data: PreparedData,
    config: RecipeConfig,
    cache_dir: Path,
) -> dict[str, dict[str, float]]:
    """`xgboost_raw`, `tabular_xgboost`, `tabular_logistic` per task, cached per data config."""
    path = cache_dir / f"tabular_{_data_key(config, list(tasks))}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["metrics"]
    bare = {
        name: ProbeTask(
            t.train_batch, t.eval_batch, t.y_train, t.y_eval, t.train_sequences, t.eval_sequences
        )
        for name, t in tasks.items()
    }
    metrics, manifest = tabular_baselines(bare, data, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metrics": metrics, "manifest": manifest}, indent=2, default=str),
        encoding="utf-8",
    )
    return metrics


def proxy_eval(
    model: nn.Module,
    data: PreparedData,
    config: RecipeConfig,
    device: torch.device,
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    cache_dir: Path,
    xgboost_trials: int = 8,
    probe_tasks: Mapping[str, ProbeTask] | None = None,
    raw_views: Sequence[str] = (),
    diagnostics: bool = True,
    xgboost_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The proxy metrics of one model; see the module docstring.

    `xgboost_raw_embedding` appends the model's default readout to the raw row.
    Each of `raw_views` adds `xgboost_raw_embedding:<view>` with that readout
    instead, searched and scored the same way. With `xgboost_params`, every
    raw arm, `xgboost_raw` included, uses that one setting instead of a search
    (the lab notebooks' protocol), and the hand-built history arms are skipped.
    `diagnostics=False` skips the held-out losses and embedding health, which
    do not depend on the tasks.
    """
    started = time.perf_counter()
    chosen = dict(probe_tasks) if probe_tasks is not None else proxy_tasks(data, config, tasks)
    reference = {} if xgboost_params else reference_arms(chosen, data, config, cache_dir)
    k = config.evaluation.precision_at_k
    profiles = {p.entity_id: p for p in data.profiles}
    coded = tuple(config.features.categorical_keys)
    top_k = config.evaluation.tabular_baselines.top_k
    result: dict[str, Any] = {"tasks": {}, "reference": reference}
    key_ids = sorted(getattr(data.tokenizer, "key_vocab", {}).values())
    for name, task in chosen.items():
        tr = readouts(model, task.train_batch, device, key_ids=key_ids)
        ev = readouts(model, task.eval_batch, device, key_ids=key_ids)
        entry: dict[str, Any] = {"summary": task.summary(), "probe": {}}
        for view in tr:
            scores = _logistic(tr[view], task.y_train, ev[view], config.seed)
            entry["probe"][view] = classification_metrics(task.y_eval, scores, k=k)
        default = next(v for v in ("usr", "last", "default") if v in tr)
        if xgboost_params:
            views = {"": default, **{v: v for v in raw_views if v in tr}}
            fixed, _ = evaluate_raw_embedding_arms(
                task.train_sequences,
                task.y_train,
                task.eval_sequences,
                task.y_eval,
                profiles=profiles,
                embedding_views={name: (tr[v], ev[v]) for name, v in views.items()},
                xgboost_params=xgboost_params,
                seed=config.seed,
                top_k=top_k,
                coded_keys=coded,
                exclude_keys=tuple(config.evaluation.tabular_baselines.exclude_keys),
                n_jobs=config.evaluation.tabular_baselines.n_jobs,
                k=k,
            )
            entry.update(fixed)
        elif xgboost_trials > 0:
            raw = TabularFeaturizer.fit(
                task.train_sequences, profiles, top_k=top_k, coded_keys=coded, history=False
            )
            raw_tr = raw.transform(task.train_sequences, profiles)
            raw_ev = raw.transform(task.eval_sequences, profiles)
            groups = [seq.entity_id for seq in task.train_sequences]
            n_jobs = config.evaluation.tabular_baselines.n_jobs
            arms = {"xgboost_raw_embedding": default}
            arms.update({f"xgboost_raw_embedding:{v}": v for v in raw_views if v in tr})
            for arm, view in arms.items():
                x_tr = np.concatenate([raw_tr, tr[view]], 1)
                x_ev = np.concatenate([raw_ev, ev[view]], 1)
                tuned = tune_xgboost(
                    x_tr,
                    task.y_train,
                    groups,
                    seed=config.seed,
                    n_trials=xgboost_trials,
                    n_jobs=n_jobs,
                )
                scores = _fit_predict_xgboost(
                    x_tr, task.y_train, x_ev, tuned, seed=config.seed, n_jobs=n_jobs
                )
                entry[arm] = classification_metrics(task.y_eval, scores, k=k)
        stem = "" if name == "fraud" else f"{name}_"
        for arm in ("xgboost_raw", "tabular_xgboost", "tabular_logistic"):
            if f"{stem}{arm}" in reference:
                entry[arm] = reference[f"{stem}{arm}"]
        result["tasks"][name] = entry
    if diagnostics:
        validation = data.batches[1]
        result["heldout"] = heldout_losses(
            model, validation, data.tokenizer, device, seed=config.seed
        )
        result["embedding_health"] = embedding_health(model, data, config, device)
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def summary_row(result: Mapping[str, Any]) -> dict[str, float]:
    """Flat scalars for a sweep table: PR-AUC per task and readout, and held-out losses."""
    row: dict[str, float] = {}
    for task, entry in result["tasks"].items():
        for view, metrics in entry["probe"].items():
            row[f"{task}/{view}/pr_auc"] = metrics["pr_auc"]
            row[f"{task}/{view}/roc_auc"] = metrics["roc_auc"]
        raw_arms = [a for a in entry if a.startswith("xgboost_raw_embedding")]
        for arm in (*raw_arms, "xgboost_raw", "tabular_xgboost"):
            if arm in entry:
                row[f"{task}/{arm}/pr_auc"] = entry[arm]["pr_auc"]
                row[f"{task}/{arm}/roc_auc"] = entry[arm]["roc_auc"]
    for name, value in result.get("heldout", {}).get("losses", {}).items():
        row[f"heldout/{name}"] = value
    for view, health in result.get("embedding_health", {}).get("views", {}).items():
        for name in ("effective_rank", "mean_cosine", "retrieval_top1"):
            row[f"health/{view}/{name}"] = health[name]
    return row


def main() -> None:
    from tfm.runtime.pipeline import setup_run
    from tfm.training.checkpointing import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Holds seed_config.yaml.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Defaults to the run's.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/proxy/cache"))
    parser.add_argument("--xgboost-trials", type=int, default=8)
    parser.add_argument("--random-init", action="store_true", help="Skip loading model.pt.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = args.config or args.run_dir / "seed_config.yaml"
    config = load_recipe_config(config_path)
    artifacts = setup_run(
        config=config, run_dir=args.out.parent / f"{args.out.stem}_run", device=args.device
    )
    if not args.random_init:
        load_checkpoint(artifacts.model, args.run_dir / "model.pt")
    result = proxy_eval(
        artifacts.model,
        artifacts.data,
        artifacts.config,
        artifacts.device,
        tasks=args.tasks,
        cache_dir=args.cache_dir,
        xgboost_trials=args.xgboost_trials,
    )
    result["run_dir"] = str(args.run_dir)
    result["random_init"] = args.random_init
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary_row(result), indent=1))


if __name__ == "__main__":
    main()
