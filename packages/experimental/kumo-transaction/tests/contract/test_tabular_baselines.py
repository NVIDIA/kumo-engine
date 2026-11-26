# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular baselines: fitted on train rows only, cutoff-safe, and deterministic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from random import Random

import numpy as np
import pytest
import torch

from tfm.contracts.schema import (
    EntityProfile,
    EventLabel,
    LifelongItem,
    TransactionEvent,
    TransactionSequence,
)
from tfm.evaluation.tabular import (
    TabularFeaturizer,
    evaluate_raw_embedding_arms,
    evaluate_tabular_baselines,
    fixed_xgboost,
)
from tfm.extras.capability import extra_available

START = datetime(2025, 1, 1, tzinfo=UTC)
CUTOFF = START + timedelta(days=60)


def _sequence(entity: str, merchants: list[str], amounts: list[float]) -> TransactionSequence:
    events = tuple(
        TransactionEvent(
            entity_id=entity,
            event_id=f"{entity}:{i}",
            event_timestamp=START + timedelta(days=i),
            categorical={"txn_type": "card_payment" if i % 2 else "transfer"},
            numeric={"amount": amount},
            text={"merchant": merchant},
        )
        for i, (merchant, amount) in enumerate(zip(merchants, amounts, strict=True))
    )
    return TransactionSequence(entity, events, CUTOFF, "90d")


def _profile(entity: str, opened_days: int) -> EntityProfile:
    return EntityProfile(
        entity_id=entity,
        as_of=CUTOFF + timedelta(days=365),
        attributes={"country": "GB", "age": "41"},
        lifelong=(LifelongItem("account_opened", START + timedelta(days=opened_days)),),
    )


def _row(featurizer: TabularFeaturizer, seq: TransactionSequence, profiles: dict) -> dict:
    values = featurizer.transform([seq], profiles)[0]
    return dict(zip(featurizer.names, values.tolist(), strict=True))


def test_the_last_events_novelty_and_profile_fields_are_features() -> None:
    seen = _sequence("u1", ["a", "b", "a"], [1.0, 2.0, 3.0])
    new = _sequence("u2", ["a", "b", "c"], [1.0, 2.0, 3.0])
    profiles = {"u1": _profile("u1", 5), "u2": _profile("u2", 5)}
    featurizer = TabularFeaturizer.fit([seen, new], profiles, top_k=5)
    assert _row(featurizer, seen, profiles)["text:merchant:last_prior_count"] == 1.0
    assert _row(featurizer, new, profiles)["text:merchant:last_prior_count"] == 0.0
    row = _row(featurizer, seen, profiles)
    assert row["num:amount:max"] == 3.0 and row["n_events"] == 3.0
    assert row["profile:age"] == 41.0 and row["profile:country=GB"] == 1.0
    assert row["lifelong:account_opened:log_days"] == pytest.approx(np.log1p(55.0))


def test_the_raw_row_holds_the_last_event_and_profile_only() -> None:
    short = _sequence("u1", ["a", "b"], [1.0, 3.0])
    long = _sequence("u2", ["a", "c", "a", "b"], [9.0, 9.0, 9.0, 3.0])
    profiles = {"u1": _profile("u1", 5), "u2": _profile("u2", 5)}
    raw = TabularFeaturizer.fit([short, long], profiles, top_k=5, history=False)
    assert all(
        name.startswith(("last_", "profile:", "lifelong:"))
        or name.endswith((":last", ":last_code"))
        for name in raw.names
    )
    assert not any(name.startswith("text:") for name in raw.names)
    a, b = _row(raw, short, profiles), _row(raw, long, profiles)
    assert a["num:amount:last"] == 3.0 and a["profile:age"] == 41.0
    same_last = {k: v for k, v in a.items() if not k.startswith("last_")}
    assert same_last == {k: v for k, v in b.items() if not k.startswith("last_")}
    full = TabularFeaturizer.fit([short, long], profiles, top_k=5)
    assert set(raw.names) < set(full.names)


def test_excluded_keys_build_no_feature() -> None:
    seq = _sequence("u1", ["a", "b", "a"], [1.0, 2.0, 3.0])
    profiles = {"u1": _profile("u1", 5)}
    for history in (True, False):
        featurizer = TabularFeaturizer.fit(
            [seq],
            profiles,
            top_k=5,
            history=history,
            exclude_keys=("amount", "merchant", "country", "account_opened"),
        )
        assert not any(
            key in name
            for name in featurizer.names
            for key in ("amount", "merchant", "country", "account_opened")
        )
        assert any(name.startswith("cat:txn_type") for name in featurizer.names)
        assert "profile:age" in featurizer.names


def test_coded_keys_become_category_shares_instead_of_statistics() -> None:
    seq = _sequence("u1", ["a", "b", "c"], [1.0, 2.0, 3.0])
    coded = TransactionSequence(
        "u1",
        tuple(
            TransactionEvent(
                entity_id="u1",
                event_id=e.event_id,
                event_timestamp=e.event_timestamp,
                numeric={**e.numeric, "mcc": 5411.0 if i < 2 else 5812.0},
                text=e.text,
            )
            for i, e in enumerate(seq.events)
        ),
        CUTOFF,
        "90d",
    )
    featurizer = TabularFeaturizer.fit([coded], {}, top_k=5, coded_keys=("mcc",))
    row = _row(featurizer, coded, {})
    assert row["cat:mcc=5411:share"] == pytest.approx(2 / 3)
    assert row["cat:mcc=5812:last"] == 1.0
    assert not any(name.startswith("num:mcc") for name in featurizer.names)


def _with_mcc(seq: TransactionSequence, codes: list[float]) -> TransactionSequence:
    events = tuple(
        TransactionEvent(
            entity_id=e.entity_id,
            event_id=e.event_id,
            event_timestamp=e.event_timestamp,
            numeric={**e.numeric, "mcc": code},
        )
        for e, code in zip(seq.events, codes, strict=True)
    )
    return TransactionSequence(seq.entity_id, events, CUTOFF, "90d")


@pytest.mark.parametrize("history", [True, False])
def test_every_train_value_gets_a_code_beyond_the_top_k(history: bool) -> None:
    base = _sequence("u1", ["a"] * 4, [1.0] * 4)
    train = _with_mcc(base, [5812.0, 742.0, 5411.0, 742.0])
    featurizer = TabularFeaturizer.fit([train], {}, top_k=1, coded_keys=("mcc",), history=history)
    assert featurizer.category_codes["mcc"] == {"742": 0, "5411": 1, "5812": 2}
    for last, code in ((5411.0, 1.0), (5812.0, 2.0)):
        row = _row(featurizer, _with_mcc(base, [742.0, 742.0, 742.0, last]), {})
        assert row["cat:mcc:last_code"] == code
        assert f"cat:mcc={last:g}:last" not in row
    unseen = _row(featurizer, _with_mcc(base, [742.0, 742.0, 742.0, 9999.0]), {})
    assert np.isnan(unseen["cat:mcc:last_code"])


def test_a_milestone_after_the_cutoff_is_not_a_feature() -> None:
    seq = _sequence("u1", ["a", "b"], [1.0, 2.0])
    early = {"u1": _profile("u1", 5)}
    late = {"u1": _profile("u1", 90)}
    featurizer = TabularFeaturizer.fit([seq], early, top_k=5)
    assert np.isnan(_row(featurizer, seq, late)["lifelong:account_opened:log_days"])


def test_labels_never_change_features_and_evaluation_rows_never_refit() -> None:
    seq = _sequence("u1", ["a", "b"], [1.0, 2.0])
    labelled = TransactionSequence(
        "u1",
        tuple(
            TransactionEvent(
                entity_id=e.entity_id,
                event_id=e.event_id,
                event_timestamp=e.event_timestamp,
                categorical=e.categorical,
                numeric=e.numeric,
                text=e.text,
                labels={"fraud": EventLabel("fraud", 1.0, e.event_timestamp)},
            )
            for e in seq.events
        ),
        CUTOFF,
        "90d",
    )
    featurizer = TabularFeaturizer.fit([seq], {}, top_k=5)
    np.testing.assert_array_equal(
        featurizer.transform([seq], {}), featurizer.transform([labelled], {})
    )
    unseen_event = TransactionEvent(
        entity_id="u9",
        event_id="u9:0",
        event_timestamp=START,
        categorical={"txn_type": "never_seen", "brand_new_key": "x"},
        numeric={"amount": 5.0},
    )
    unseen = TransactionSequence("u9", (unseen_event,), CUTOFF, "90d")
    assert featurizer.transform([unseen], {}).shape == (1, len(featurizer.names))
    assert not any("never_seen" in n or "brand_new_key" in n for n in featurizer.names)


def _task(n: int, seed: int) -> tuple[list[TransactionSequence], np.ndarray]:
    rng = Random(seed)
    sequences, labels = [], []
    for i in range(n):
        positive = i % 4 == 0
        merchants = [f"m{rng.randrange(3)}" for _ in range(9)]
        merchants.append(f"new{i}" if positive else merchants[0])
        amounts = [rng.uniform(5, 50) for _ in range(9)] + [
            rng.uniform(200, 400) if positive else rng.uniform(5, 50)
        ]
        sequences.append(_sequence(f"s{seed}u{i}", merchants, amounts))
        labels.append(float(positive))
    return sequences, np.array(labels, dtype=np.float32)


def test_the_logistic_arm_separates_a_separable_task() -> None:
    train, y_train = _task(80, 0)
    evaluation, y_eval = _task(40, 1)
    metrics, manifest = evaluate_tabular_baselines(
        train, y_train, evaluation, y_eval, profiles={}, seed=0, xgboost_trials=0, prefix="toy"
    )
    assert set(metrics) == {"toy_tabular_logistic"}
    assert metrics["toy_tabular_logistic"]["roc_auc"] > 0.95
    assert manifest["n_features"] > 10 and "xgboost" not in manifest


def test_one_label_per_sequence_is_required() -> None:
    train, y_train = _task(8, 0)
    with pytest.raises(ValueError, match="one label per sequence"):
        evaluate_tabular_baselines(
            train, y_train[:-1], train, y_train, profiles={}, seed=0, xgboost_trials=0
        )


def test_embeddings_must_come_for_both_sides_one_row_each() -> None:
    train, y_train = _task(8, 0)
    emb = np.zeros((len(train), 4), np.float32)
    with pytest.raises(ValueError, match="both sides"):
        evaluate_tabular_baselines(
            train, y_train, train, y_train, profiles={}, seed=0, train_embeddings=emb
        )
    with pytest.raises(ValueError, match="one embedding per sequence"):
        evaluate_tabular_baselines(
            train,
            y_train,
            train,
            y_train,
            profiles={},
            seed=0,
            train_embeddings=emb[:-1],
            eval_embeddings=emb,
        )


@pytest.mark.skipif(
    not (extra_available("optuna") and extra_available("xgboost")),
    reason="needs Optuna and XGBoost from the baselines extra",
)
def test_the_embedding_arm_uses_signal_only_the_embedding_carries() -> None:
    # Labels independent of the sequences; one embedding column encodes them.
    rng = np.random.default_rng(0)
    train, _ = _task(120, 0)
    evaluation, _ = _task(60, 1)
    y_train = (rng.random(len(train)) < 0.3).astype(np.float32)
    y_eval = (rng.random(len(evaluation)) < 0.3).astype(np.float32)

    def embed(y: np.ndarray) -> np.ndarray:
        return np.stack([y + rng.normal(0, 0.1, len(y)), rng.normal(size=len(y))], 1)

    metrics, manifest = evaluate_tabular_baselines(
        train,
        y_train,
        evaluation,
        y_eval,
        profiles={},
        seed=0,
        xgboost_trials=2,
        n_jobs=1,
        train_embeddings=embed(y_train),
        eval_embeddings=embed(y_eval),
    )
    assert metrics["tabular_xgboost_embedding"]["roc_auc"] > 0.95
    assert metrics["tabular_xgboost"]["roc_auc"] < 0.8
    assert metrics["xgboost_raw_embedding"]["roc_auc"] > 0.95
    assert metrics["xgboost_raw"]["roc_auc"] < 0.8
    assert set(manifest) >= {"xgboost", "xgboost_embedding", "xgboost_raw", "xgboost_raw_embedding"}
    assert manifest["n_raw_features"] < manifest["n_features"]


@pytest.mark.skipif(
    not (extra_available("optuna") and extra_available("xgboost")),
    reason="needs Optuna and XGBoost from the baselines extra",
)
def test_each_named_view_gets_its_own_raw_embedding_arm() -> None:
    rng = np.random.default_rng(0)
    train, _ = _task(120, 0)
    evaluation, _ = _task(60, 1)
    y_train = (rng.random(len(train)) < 0.3).astype(np.float32)
    y_eval = (rng.random(len(evaluation)) < 0.3).astype(np.float32)
    informative = (
        np.stack([y_train + rng.normal(0, 0.1, len(y_train))], 1),
        np.stack([y_eval + rng.normal(0, 0.1, len(y_eval))], 1),
    )
    noise = (rng.normal(size=(len(train), 2)), rng.normal(size=(len(evaluation), 2)))
    metrics, _ = evaluate_tabular_baselines(
        train,
        y_train,
        evaluation,
        y_eval,
        profiles={},
        seed=0,
        xgboost_trials=2,
        n_jobs=1,
        embedding_views={"signal": informative, "noise": noise},
    )
    assert "xgboost_raw_embedding" not in metrics
    assert "tabular_xgboost_embedding" not in metrics
    assert metrics["xgboost_raw_embedding:signal"]["roc_auc"] > 0.95
    assert metrics["xgboost_raw_embedding:noise"]["roc_auc"] < 0.8
    with pytest.raises(ValueError, match="one embedding per sequence"):
        evaluate_tabular_baselines(
            train,
            y_train,
            evaluation,
            y_eval,
            profiles={},
            seed=0,
            embedding_views={"short": (noise[0][:-1], noise[1])},
        )


@pytest.mark.skipif(
    not (extra_available("optuna") and extra_available("xgboost")),
    reason="needs Optuna and XGBoost from the baselines extra",
)
def test_the_tuned_xgboost_arm_is_deterministic() -> None:
    train, y_train = _task(120, 0)
    evaluation, y_eval = _task(40, 1)

    def run() -> tuple[dict, dict]:
        return evaluate_tabular_baselines(
            train, y_train, evaluation, y_eval, profiles={}, seed=3, xgboost_trials=3, n_jobs=1
        )

    first, second = run(), run()
    assert first == second
    assert first[0]["tabular_xgboost"]["roc_auc"] > 0.9
    assert first[1]["xgboost"]["n_trials"] == 3


FIXED = {"max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0}


@pytest.mark.skipif(not extra_available("xgboost"), reason="needs the baselines extra")
def test_fixed_arms_share_one_setting_and_differ_only_in_features() -> None:
    rng = np.random.default_rng(0)
    train, _ = _task(120, 0)
    evaluation, _ = _task(60, 1)
    y_train = (rng.random(len(train)) < 0.3).astype(np.float32)
    y_eval = (rng.random(len(evaluation)) < 0.3).astype(np.float32)
    informative = (
        np.stack([y_train + rng.normal(0, 0.1, len(y_train))], 1),
        np.stack([y_eval + rng.normal(0, 0.1, len(y_eval))], 1),
    )
    noise = (rng.normal(size=(len(train), 2)), rng.normal(size=(len(evaluation), 2)))

    def run() -> tuple[dict, dict]:
        return evaluate_raw_embedding_arms(
            train,
            y_train,
            evaluation,
            y_eval,
            profiles={},
            embedding_views={"signal": informative, "noise": noise},
            xgboost_params=FIXED,
            seed=0,
            n_jobs=1,
        )

    metrics, manifest = run()
    assert run() == (metrics, manifest)
    assert set(metrics) == {
        "xgboost_raw",
        "xgboost_raw_embedding:signal",
        "xgboost_raw_embedding:noise",
    }
    assert metrics["xgboost_raw_embedding:signal"]["roc_auc"] > 0.95
    assert metrics["xgboost_raw"]["roc_auc"] < 0.8
    assert manifest["xgboost_params"] == FIXED
    raw_width = manifest["n_raw_features"]
    assert manifest["xgboost_raw"]["n_features"] == raw_width
    assert manifest["xgboost_raw_embedding:noise"]["n_features"] == raw_width + 2
    with pytest.raises(ValueError, match="one embedding per sequence"):
        evaluate_raw_embedding_arms(
            train,
            y_train,
            evaluation,
            y_eval,
            profiles={},
            embedding_views={"short": (noise[0][:-1], noise[1])},
            xgboost_params=FIXED,
            seed=0,
        )


@pytest.mark.skipif(
    not (extra_available("xgboost") and torch.cuda.is_available()),
    reason="needs the baselines extra and a CUDA device",
)
def test_gpu_xgboost_scores_the_same_arms_as_cpu() -> None:
    train, y_train = _task(120, 0)
    evaluation, y_eval = _task(60, 1)

    def run(device: str) -> dict:
        metrics, _ = evaluate_raw_embedding_arms(
            train,
            y_train,
            evaluation,
            y_eval,
            profiles={},
            embedding_views={},
            xgboost_params=FIXED,
            seed=0,
            n_jobs=1,
            device=device,  # type: ignore[arg-type]
        )
        return metrics

    cpu, gpu = run("cpu"), run("cuda")
    assert set(cpu) == set(gpu) == {"xgboost_raw"}
    assert abs(cpu["xgboost_raw"]["roc_auc"] - gpu["xgboost_raw"]["roc_auc"]) < 0.05


@pytest.mark.skipif(not extra_available("xgboost"), reason="needs the baselines extra")
def test_the_recipes_fixed_arms_equal_the_notebooks_raw_arms() -> None:
    rng = np.random.default_rng(1)
    train, y_train = _task(120, 0)
    evaluation, y_eval = _task(60, 1)
    embedding = (rng.normal(size=(len(train), 3)), rng.normal(size=(len(evaluation), 3)))
    surprise = (rng.normal(size=(len(train), 2)), rng.normal(size=(len(evaluation), 2)))
    recipe, manifest = evaluate_tabular_baselines(
        train,
        y_train,
        evaluation,
        y_eval,
        profiles={},
        seed=0,
        n_jobs=1,
        xgboost_trials=0,
        xgboost_params=FIXED,
        train_embeddings=embedding[0],
        eval_embeddings=embedding[1],
        embedding_views={"surprise": surprise},
        prefix="churn_6m",
    )
    notebook, _ = evaluate_raw_embedding_arms(
        train,
        y_train,
        evaluation,
        y_eval,
        profiles={},
        embedding_views={"": embedding, "surprise": surprise},
        xgboost_params=FIXED,
        seed=0,
        n_jobs=1,
    )
    assert set(notebook) == {
        "xgboost_raw",
        "xgboost_raw_embedding",
        "xgboost_raw_embedding:surprise",
    }
    for arm, metrics in notebook.items():
        assert recipe[f"churn_6m_{arm}"] == metrics
    assert {"churn_6m_tabular_xgboost", "churn_6m_tabular_xgboost_embedding"} <= set(recipe)
    assert manifest["xgboost"]["params"] == FIXED
    assert manifest["xgboost"]["n_trials"] == 0
    with pytest.raises(ValueError, match="not both"):
        evaluate_tabular_baselines(
            train, y_train, evaluation, y_eval, profiles={}, seed=0, xgboost_params=FIXED
        )


@pytest.mark.skipif(not extra_available("xgboost"), reason="needs the baselines extra")
def test_a_fixed_setting_is_kept_and_only_the_tree_count_is_chosen() -> None:
    train, y_train = _task(120, 0)
    x = TabularFeaturizer.fit(train, {}, top_k=10).transform(train, {})
    fitted = fixed_xgboost(x, y_train, [s.entity_id for s in train], FIXED, seed=0, n_jobs=1)
    assert dict(fitted.params) == FIXED
    assert fitted.n_trials == 0
    assert 1 <= fitted.n_estimators <= 2000
