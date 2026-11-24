# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-engineered tabular baselines: the reference a lift is measured against.

A foundation model's lift is reported against the task-specific model it would
replace; PRAGMA reports relative PR-AUC over production baselines. Here that
model is fitted on aggregates of the same truncated history the encoder reads,
plus the profile fields its tokenizer sees, so a comparison changes only the
representation:

- `tabular_logistic`: logistic regression on the standardized features.
- `tabular_xgboost`: XGBoost tuned with Optuna on an entity-disjoint inner
  split of the train side (needs the `baselines` extra).
- `xgboost_raw`: the same search on the last event's own fields and the
  profile only, Blueprint v1's raw arm: the reference for what an encoder adds
  without any hand-engineered history feature.

Features are built per key from the event contract (counts, recency, numeric
statistics, category shares, how often the last event's value was seen
before), so no dataset-specific field is named here. Nothing reads a label, the
evaluation side during fitting, or an event after a row's cutoff; profile
milestones are kept only when dated by the cutoff, as the tokenizer does.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tfm.contracts.errors import InsufficientLabelSupportError
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.extras.capability import require_extra

_DAY = 86_400.0
_RECENT_DAYS = (1, 7, 30)
_ARMS = ("tabular_logistic", "tabular_xgboost", "tabular_xgboost_embedding")
_RAW_ARMS = ("xgboost_raw", "xgboost_raw_embedding")

XGBoostDevice = Literal["cpu", "cuda"]


def _is_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _code_order(values: Counter[str]) -> dict[str, int]:
    """Rank every train-side value: numerically when all are numbers, else as text."""
    numeric = all(_is_number(v) for v in values)
    ordered = sorted(values, key=lambda v: (float(v), v) if numeric else (0.0, v))
    return {value: i for i, value in enumerate(ordered)}


def _categories(event: TransactionEvent, coded: tuple[str, ...]) -> dict[str, str]:
    values = dict(event.categorical)
    for key in coded:
        if key in event.numeric:
            values[key] = f"{event.numeric[key]:g}"
    return values


@dataclass(frozen=True)
class TabularFeaturizer:
    """Feature layout fitted on train-side rows; `transform` never refits it.

    `coded_keys` are numeric-valued codes (such as `mcc`) treated as categories,
    matching `features.categorical_keys` for the tokenizer. With `history`
    off, a row is the last event's own fields plus the profile: no count,
    recency, or aggregate over earlier events, as in Blueprint v1's raw arm.
    `exclude_keys` names event or profile keys, such as a user id, that no
    feature is built from.

    Beside the top-`top_k` one-hots, the last event's value of every categorical
    key is also one integer code (`cat:<key>:last_code`), ranked over every
    train-side value, so a tree can split on values outside the top `top_k`.
    A value the train side never saw is missing (NaN).
    """

    numeric_keys: tuple[str, ...]
    categorical_values: Mapping[str, tuple[str, ...]]
    text_keys: tuple[str, ...]
    profile_numeric: tuple[str, ...]
    profile_values: Mapping[str, tuple[str, ...]]
    lifelong_keys: tuple[str, ...]
    coded_keys: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    history: bool = True
    category_codes: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        sequences: Sequence[TransactionSequence],
        profiles: Mapping[str, EntityProfile],
        *,
        top_k: int,
        coded_keys: Sequence[str] = (),
        history: bool = True,
        exclude_keys: Sequence[str] = (),
    ) -> TabularFeaturizer:
        coded = tuple(coded_keys)
        excluded = set(exclude_keys)
        numeric: set[str] = set()
        categorical: dict[str, Counter[str]] = {}
        text: set[str] = set()
        for seq in sequences:
            for event in seq.events:
                numeric.update(k for k in event.numeric if k not in coded and k not in excluded)
                text.update(k for k in event.text if k not in excluded)
                for key, value in _categories(event, coded).items():
                    if key not in excluded:
                        categorical.setdefault(key, Counter())[value] += 1
        attributes: dict[str, Counter[str]] = {}
        lifelong: set[str] = set()
        for entity in sorted({seq.entity_id for seq in sequences}):
            profile = profiles.get(entity)
            if profile is None:
                continue
            for key, value in profile.attributes.items():
                if key not in excluded:
                    attributes.setdefault(key, Counter())[str(value)] += 1
            lifelong.update(item.key for item in profile.lifelong if item.key not in excluded)
        profile_numeric = tuple(
            sorted(k for k, c in attributes.items() if all(_is_number(v) for v in c))
        )
        layout = cls(
            numeric_keys=tuple(sorted(numeric)),
            categorical_values={
                key: tuple(v for v, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])
                for key, c in sorted(categorical.items())
            },
            text_keys=tuple(sorted(text)),
            profile_numeric=profile_numeric,
            profile_values={
                key: tuple(v for v, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])
                for key, c in sorted(attributes.items())
                if key not in profile_numeric
            },
            lifelong_keys=tuple(sorted(lifelong)),
            coded_keys=coded,
            history=history,
            category_codes={key: _code_order(c) for key, c in sorted(categorical.items())},
        )
        names: set[str] = set()
        for seq in sequences:
            names.update(layout._row(seq, profiles.get(seq.entity_id)))
        return cls(**{**layout.__dict__, "names": tuple(sorted(names))})

    def transform(
        self,
        sequences: Sequence[TransactionSequence],
        profiles: Mapping[str, EntityProfile],
    ) -> NDArray[np.float32]:
        """One row per sequence; a feature the row lacks is NaN."""
        out = np.full((len(sequences), len(self.names)), np.nan, dtype=np.float32)
        index = {name: i for i, name in enumerate(self.names)}
        for r, seq in enumerate(sequences):
            for name, value in self._row(seq, profiles.get(seq.entity_id)).items():
                col = index.get(name)
                if col is not None:
                    out[r, col] = value
        return out

    def _row(self, seq: TransactionSequence, profile: EntityProfile | None) -> dict[str, float]:
        cutoff = seq.cutoff_time.timestamp()
        events = seq.events
        n = len(events)
        if not self.history:
            row = self._last_event(events[-1]) if n else {}
        else:
            row = {"n_events": float(n), "log_n_events": math.log1p(n)}
        if n and self.history:
            ts = np.array([e.event_timestamp.timestamp() for e in events], dtype=np.float64)
            age = (cutoff - ts) / _DAY
            row["days_since_last"] = float(age[-1])
            row["days_since_first"] = float(age[0])
            gaps = np.log1p(np.diff(ts))
            if len(gaps):
                row["log_gap_mean"] = float(gaps.mean())
                row["log_gap_min"] = float(gaps.min())
                row["log_gap_last"] = float(gaps[-1])
            for days in _RECENT_DAYS:
                row[f"events_{days}d"] = float((age <= days).sum())
            last_time = events[-1].event_timestamp
            row["last_hour"] = float(last_time.hour)
            row["last_weekday"] = float(last_time.weekday())
            for key in self.numeric_keys:
                vals = np.array(
                    [e.numeric[key] for e in events if key in e.numeric], dtype=np.float64
                )
                if not len(vals):
                    continue
                row[f"num:{key}:mean"] = float(vals.mean())
                row[f"num:{key}:std"] = float(vals.std())
                row[f"num:{key}:min"] = float(vals.min())
                row[f"num:{key}:max"] = float(vals.max())
                recent = [
                    e.numeric[key]
                    for e, a in zip(events, age, strict=True)
                    if key in e.numeric and a <= 7
                ]
                row[f"num:{key}:sum_7d"] = float(sum(recent))
                if key in events[-1].numeric:
                    row[f"num:{key}:last"] = float(events[-1].numeric[key])
            categories = [_categories(e, self.coded_keys) for e in events]
            for key, tops in self.categorical_values.items():
                counts = Counter(c[key] for c in categories if key in c)
                row[f"cat:{key}:distinct"] = float(len(counts))
                for value in tops:
                    row[f"cat:{key}={value}:share"] = counts.get(value, 0) / n
                last = categories[-1].get(key)
                if last is not None:
                    row[f"cat:{key}:last_prior_count"] = float(counts[last] - 1)
                    row[f"cat:{key}:last_code"] = self._code(key, last)
                    for value in tops:
                        row[f"cat:{key}={value}:last"] = float(last == value)
            for key in self.text_keys:
                seen: Counter[str] = Counter()
                new_ages: list[float] = []
                present = 0
                for event, a in zip(events, age, strict=True):
                    value = event.text.get(key)
                    if value is None:
                        continue
                    present += 1
                    if value not in seen:
                        new_ages.append(float(a))
                    seen[value] += 1
                if not present:
                    continue
                row[f"text:{key}:distinct_share"] = len(seen) / present
                for days in _RECENT_DAYS:
                    row[f"text:{key}:new_{days}d"] = float(sum(a <= days for a in new_ages))
                last = events[-1].text.get(key)
                if last is not None:
                    row[f"text:{key}:last_prior_count"] = float(seen[last] - 1)
        if profile is not None:
            for key, value in profile.attributes.items():
                if key in self.profile_numeric:
                    if _is_number(str(value)):
                        row[f"profile:{key}"] = float(value)
                elif key in self.profile_values:
                    for top in self.profile_values[key]:
                        row[f"profile:{key}={top}"] = float(str(value) == top)
            for item in profile.lifelong:
                if item.key in self.lifelong_keys and item.timestamp <= seq.cutoff_time:
                    row[f"lifelong:{item.key}:log_days"] = math.log1p(
                        max(cutoff - item.timestamp.timestamp(), 0.0) / _DAY
                    )
        return row

    def _code(self, key: str, value: str) -> float:
        code = self.category_codes.get(key, {}).get(value)
        return math.nan if code is None else float(code)

    def _last_event(self, event: TransactionEvent) -> dict[str, float]:
        """The last event's own fields, under the same names the history row uses."""
        row = {
            "last_hour": float(event.event_timestamp.hour),
            "last_weekday": float(event.event_timestamp.weekday()),
        }
        for key in self.numeric_keys:
            if key in event.numeric:
                row[f"num:{key}:last"] = float(event.numeric[key])
        categories = _categories(event, self.coded_keys)
        for key, tops in self.categorical_values.items():
            last = categories.get(key)
            if last is not None:
                row[f"cat:{key}:last_code"] = self._code(key, last)
                for value in tops:
                    row[f"cat:{key}={value}:last"] = float(last == value)
        return row


def _constant(y_train: NDArray[np.floating], n: int) -> NDArray[np.float64]:
    return np.full(n, float(y_train.mean() if len(y_train) else 0.0))


def _fit_predict_logistic(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_eval: NDArray[np.floating],
    seed: int,
) -> NDArray[np.float64]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if np.unique(y_train).size < 2:
        return _constant(y_train, len(x_eval))
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(x_train, y_train.astype(int))
    return np.asarray(model.predict_proba(x_eval)[:, 1], dtype=np.float64)


@dataclass(frozen=True)
class TunedXGBoost:
    """The selected trial: parameters, tree count, and its inner-validation PR-AUC."""

    params: Mapping[str, Any]
    n_estimators: int
    validation_pr_auc: float
    n_trials: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": dict(self.params),
            "n_estimators": self.n_estimators,
            "validation_pr_auc": self.validation_pr_auc,
            "n_trials": self.n_trials,
        }


def _inner_fold(
    x: NDArray[np.floating], labels: NDArray[np.integer], groups: Sequence[str], seed: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The 80/20 entity-disjoint fit/validation split of the train side."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(folds.split(x, labels, groups=np.asarray(groups)))
    for name, idx in (("fit", fit_idx), ("validation", val_idx)):
        if np.unique(labels[idx]).size < 2:
            raise InsufficientLabelSupportError(
                f"The XGBoost baseline's inner {name} fold holds one class."
            )
    return fit_idx, val_idx


def xgboost_classifier(
    params: Mapping[str, Any],
    *,
    n_estimators: int,
    seed: int,
    n_jobs: int,
    device: XGBoostDevice = "cpu",
    early_stopping: bool = False,
) -> Any:
    """The one XGBoost constructor every arm uses; `device="cuda"` trains on the GPU."""
    require_extra("baselines", "xgboost")
    import xgboost as xgb  # type: ignore[import-not-found]

    stopping: dict[str, Any] = (
        {"early_stopping_rounds": 50, "eval_metric": "aucpr"} if early_stopping else {}
    )
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
        **stopping,
        **dict(params),
    )


def fixed_xgboost(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    groups: Sequence[str],
    params: Mapping[str, Any],
    *,
    seed: int,
    n_jobs: int,
    device: XGBoostDevice = "cpu",
) -> TunedXGBoost:
    """`params` as given, with only the tree count chosen, by early stopping.

    The inner fold is `tune_xgboost`'s, so a fixed setting and a searched one
    see the same data; `n_trials` is 0. The evaluation side is never seen.
    """
    from sklearn.metrics import average_precision_score

    labels = np.asarray(y).astype(int)
    fit_idx, val_idx = _inner_fold(x, labels, groups, seed)
    clf = xgboost_classifier(
        params, n_estimators=2000, seed=seed, n_jobs=n_jobs, device=device, early_stopping=True
    )
    clf.fit(x[fit_idx], labels[fit_idx], eval_set=[(x[val_idx], labels[val_idx])], verbose=False)
    return TunedXGBoost(
        params=dict(params),
        n_estimators=int(clf.best_iteration) + 1,
        validation_pr_auc=float(
            average_precision_score(labels[val_idx], clf.predict_proba(x[val_idx])[:, 1])
        ),
        n_trials=0,
    )


def tune_xgboost(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    groups: Sequence[str],
    *,
    seed: int,
    n_trials: int,
    n_jobs: int,
    device: XGBoostDevice = "cpu",
) -> TunedXGBoost:
    """Optuna TPE search over XGBoost on an 80/20 entity-disjoint split of the train side.

    Trials maximize PR-AUC on the inner validation fold, with early stopping
    choosing the tree count. The evaluation side is never seen. Raises
    `InsufficientLabelSupportError` when a fold would hold one class.
    """
    require_extra("baselines", "optuna")
    import optuna  # type: ignore[import-not-found]
    from sklearn.metrics import average_precision_score

    labels = np.asarray(y).astype(int)
    fit_idx, val_idx = _inner_fold(x, labels, groups, seed)
    ratio = float((labels[fit_idx] == 0).sum()) / max(float(labels[fit_idx].sum()), 1.0)

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
        clf = xgboost_classifier(
            params,
            n_estimators=2000,
            seed=seed,
            n_jobs=n_jobs,
            device=device,
            early_stopping=True,
        )
        clf.fit(
            x[fit_idx], labels[fit_idx], eval_set=[(x[val_idx], labels[val_idx])], verbose=False
        )
        trial.set_user_attr("n_estimators", int(clf.best_iteration) + 1)
        trial.set_user_attr("params", params)
        return float(average_precision_score(labels[val_idx], clf.predict_proba(x[val_idx])[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    return TunedXGBoost(
        params=dict(best.user_attrs["params"]),
        n_estimators=int(best.user_attrs["n_estimators"]),
        validation_pr_auc=float(study.best_value),
        n_trials=n_trials,
    )


def _fit_predict_xgboost(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_eval: NDArray[np.floating],
    tuned: TunedXGBoost,
    *,
    seed: int,
    n_jobs: int,
    device: XGBoostDevice = "cpu",
) -> NDArray[np.float64]:
    clf = xgboost_classifier(
        tuned.params, n_estimators=tuned.n_estimators, seed=seed, n_jobs=n_jobs, device=device
    )
    clf.fit(x_train, np.asarray(y_train).astype(int))
    return np.asarray(clf.predict_proba(x_eval)[:, 1], dtype=np.float64)


def _raw_blocks(
    train_sequences: Sequence[TransactionSequence],
    eval_sequences: Sequence[TransactionSequence],
    profiles: Mapping[str, EntityProfile],
    views: Mapping[str, tuple[NDArray[np.floating], NDArray[np.floating]]],
    *,
    top_k: int,
    coded_keys: Sequence[str],
    exclude_keys: Sequence[str],
) -> tuple[dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]], int]:
    """`xgboost_raw`'s rows (last event plus profile) and one `xgboost_raw_embedding` per view."""
    raw = TabularFeaturizer.fit(
        train_sequences,
        profiles,
        top_k=top_k,
        coded_keys=coded_keys,
        history=False,
        exclude_keys=exclude_keys,
    )
    r_train = raw.transform(train_sequences, profiles)
    r_eval = raw.transform(eval_sequences, profiles)
    blocks = {_RAW_ARMS[0]: (r_train, r_eval)}
    for name, (view_train, view_eval) in views.items():
        blocks[_RAW_ARMS[1] + (f":{name}" if name else "")] = (
            np.concatenate([r_train, np.asarray(view_train, np.float32)], axis=1),
            np.concatenate([r_eval, np.asarray(view_eval, np.float32)], axis=1),
        )
    return blocks, len(raw.names)


def _score_xgboost_arms(
    blocks: Mapping[str, tuple[NDArray[np.floating], NDArray[np.floating]]],
    y_train: NDArray[np.floating],
    y_eval: NDArray[np.floating],
    groups: Sequence[str],
    *,
    xgboost_trials: int,
    xgboost_params: Mapping[str, Any] | None,
    seed: int,
    n_jobs: int,
    device: XGBoostDevice,
    k: int,
) -> tuple[dict[str, dict[str, float]], dict[str, TunedXGBoost]]:
    """Fit every arm the same way: its own search, or the one fixed setting."""
    metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, TunedXGBoost] = {}
    for arm, (x_tr, x_ev) in blocks.items():
        if xgboost_params is not None:
            tuned = fixed_xgboost(
                x_tr, y_train, groups, xgboost_params, seed=seed, n_jobs=n_jobs, device=device
            )
        else:
            tuned = tune_xgboost(
                x_tr,
                y_train,
                groups,
                seed=seed,
                n_trials=xgboost_trials,
                n_jobs=n_jobs,
                device=device,
            )
        scores = _fit_predict_xgboost(
            x_tr, y_train, x_ev, tuned, seed=seed, n_jobs=n_jobs, device=device
        )
        metrics[arm] = classification_metrics(y_eval, scores, k=k)
        fitted[arm] = tuned
    return metrics, fitted


def evaluate_tabular_baselines(
    train_sequences: Sequence[TransactionSequence],
    y_train: NDArray[np.floating],
    eval_sequences: Sequence[TransactionSequence],
    y_eval: NDArray[np.floating],
    *,
    profiles: Mapping[str, EntityProfile],
    seed: int,
    top_k: int = 10,
    coded_keys: Sequence[str] = (),
    exclude_keys: Sequence[str] = (),
    xgboost_trials: int = 30,
    n_jobs: int = 8,
    k: int = 5,
    prefix: str = "",
    train_embeddings: NDArray[np.floating] | None = None,
    eval_embeddings: NDArray[np.floating] | None = None,
    embedding_views: Mapping[str, tuple[NDArray[np.floating], NDArray[np.floating]]] | None = None,
    xgboost_params: Mapping[str, Any] | None = None,
    device: XGBoostDevice = "cpu",
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Fit the tabular arms on train rows and score evaluation rows.

    Returns `(metrics, manifest)`. Metric keys are `{prefix}_tabular_logistic`
    and, when `xgboost_trials > 0` or `xgboost_params` is set,
    `{prefix}_tabular_xgboost` and `{prefix}_xgboost_raw` (the last event's
    fields and the profile, with no history aggregates); the manifest records
    the feature counts and each arm's XGBoost setting. With frozen embeddings
    for both sides, `{prefix}_tabular_xgboost_embedding` and
    `{prefix}_xgboost_raw_embedding` are fitted the same way with the embedding
    appended: what the encoder adds to each reference model rather than what
    it replaces. Each `embedding_views[name] = (train, eval)` adds
    `{prefix}_xgboost_raw_embedding:{name}`, so several readouts are compared
    against one set of reference arms. Each arm runs its own `xgboost_trials`
    search, or, with `xgboost_params`, uses that setting unchanged
    (`fixed_xgboost`), as `evaluate_raw_embedding_arms` does.
    """
    if xgboost_params is not None and xgboost_trials > 0:
        raise ValueError("Pass xgboost_params or xgboost_trials > 0, not both.")
    if len(train_sequences) != len(y_train) or len(eval_sequences) != len(y_eval):
        raise ValueError("Each side needs one label per sequence.")
    if (train_embeddings is None) != (eval_embeddings is None):
        raise ValueError("Pass embeddings for both sides or neither.")
    views = dict(embedding_views or {})
    if train_embeddings is not None and eval_embeddings is not None:
        views = {"": (train_embeddings, eval_embeddings), **views}
    for view_train, view_eval in views.values():
        if len(view_train) != len(y_train) or len(view_eval) != len(y_eval):
            raise ValueError("Each side needs one embedding per sequence.")
    stem = f"{prefix}_" if prefix else ""
    featurizer = TabularFeaturizer.fit(
        train_sequences, profiles, top_k=top_k, coded_keys=coded_keys, exclude_keys=exclude_keys
    )
    x_train = featurizer.transform(train_sequences, profiles)
    x_eval = featurizer.transform(eval_sequences, profiles)
    y_tr = np.asarray(y_train, dtype=np.float32)
    y_ev = np.asarray(y_eval, dtype=np.float32)
    metrics = {
        f"{stem}{_ARMS[0]}": classification_metrics(
            y_ev, _fit_predict_logistic(x_train, y_tr, x_eval, seed), k=k
        )
    }
    manifest: dict[str, Any] = {
        "n_features": len(featurizer.names),
        "top_k": top_k,
        "exclude_keys": sorted(exclude_keys),
    }
    if xgboost_trials > 0 or xgboost_params is not None:
        raw_blocks, manifest["n_raw_features"] = _raw_blocks(
            train_sequences,
            eval_sequences,
            profiles,
            views,
            top_k=top_k,
            coded_keys=coded_keys,
            exclude_keys=exclude_keys,
        )
        blocks = {_ARMS[1]: (x_train, x_eval)}
        if "" in views:
            e_train, e_eval = (np.asarray(v, np.float32) for v in views[""])
            blocks[_ARMS[2]] = (
                np.concatenate([x_train, e_train], axis=1),
                np.concatenate([x_eval, e_eval], axis=1),
            )
        blocks.update(raw_blocks)
        arm_metrics, fitted = _score_xgboost_arms(
            blocks,
            y_tr,
            y_ev,
            [seq.entity_id for seq in train_sequences],
            xgboost_trials=xgboost_trials,
            xgboost_params=xgboost_params,
            seed=seed,
            n_jobs=n_jobs,
            device=device,
            k=k,
        )
        for arm, tuned in fitted.items():
            metrics[f"{stem}{arm}"] = arm_metrics[arm]
            manifest[arm.removeprefix("tabular_")] = tuned.as_dict()
    return metrics, manifest


def evaluate_raw_embedding_arms(
    train_sequences: Sequence[TransactionSequence],
    y_train: NDArray[np.floating],
    eval_sequences: Sequence[TransactionSequence],
    y_eval: NDArray[np.floating],
    *,
    profiles: Mapping[str, EntityProfile],
    embedding_views: Mapping[str, tuple[NDArray[np.floating], NDArray[np.floating]]],
    xgboost_params: Mapping[str, Any],
    seed: int,
    top_k: int = 10,
    coded_keys: Sequence[str] = (),
    exclude_keys: Sequence[str] = (),
    n_jobs: int = 8,
    k: int = 5,
    device: XGBoostDevice = "cpu",
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Raw features against raw features plus each model view, with one fixed XGBoost setting.

    The raw row is `xgboost_raw`'s: the last event's own fields and the
    profile. `embedding_views[name] = (train, eval)` adds
    `xgboost_raw_embedding:{name}`, the raw row with that view appended, and
    `name=""` adds plain `xgboost_raw_embedding`, as in
    `evaluate_tabular_baselines`. Every
    arm uses `xgboost_params` unchanged (`fixed_xgboost`), so arms differ only
    in their features. Returns `(metrics, manifest)`; the manifest records the
    feature counts and each arm's tree count.
    """
    if len(train_sequences) != len(y_train) or len(eval_sequences) != len(y_eval):
        raise ValueError("Each side needs one label per sequence.")
    for view_train, view_eval in embedding_views.values():
        if len(view_train) != len(y_train) or len(view_eval) != len(y_eval):
            raise ValueError("Each side needs one embedding per sequence.")
    blocks, n_raw = _raw_blocks(
        train_sequences,
        eval_sequences,
        profiles,
        embedding_views,
        top_k=top_k,
        coded_keys=coded_keys,
        exclude_keys=exclude_keys,
    )
    metrics, fitted = _score_xgboost_arms(
        blocks,
        np.asarray(y_train, dtype=np.float32),
        np.asarray(y_eval, dtype=np.float32),
        [seq.entity_id for seq in train_sequences],
        xgboost_trials=0,
        xgboost_params=xgboost_params,
        seed=seed,
        n_jobs=n_jobs,
        device=device,
        k=k,
    )
    manifest: dict[str, Any] = {"n_raw_features": n_raw, "xgboost_params": dict(xgboost_params)}
    for arm, tuned in fitted.items():
        manifest[arm] = {"n_features": blocks[arm][0].shape[1], "n_estimators": tuned.n_estimators}
    return metrics, manifest


__all__ = [
    "TabularFeaturizer",
    "TunedXGBoost",
    "XGBoostDevice",
    "evaluate_raw_embedding_arms",
    "evaluate_tabular_baselines",
    "fixed_xgboost",
    "tune_xgboost",
    "xgboost_classifier",
]
