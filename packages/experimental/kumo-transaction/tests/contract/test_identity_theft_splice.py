# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spliced identity-theft task keeps dataset strategy's labels and leakage rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

import pytest

from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.schema import EntityProfile, EventLabel, TransactionEvent
from tfm.data.adapters.pragmatiq import load_pragmatiq_dataset, write_pragmatiq_dataset
from tfm.data.adapters.registry import load_adapter_events
from tfm.data.labels.windows import side_label_windows
from tfm.data.tasks.identity_theft import (
    CARD_TXN_TYPES,
    IdentityTheftSpec,
    splice_identity_theft,
)
from tfm.runtime.data import prepare_data
from tfm.runtime.evaluate import fraud_entity_split, fraud_task, history_window_tasks

START = datetime(2024, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2024, 7, 1, tzinfo=UTC)
VALIDATION_END = datetime(2024, 9, 1, tzinfo=UTC)
DAYS = 300
N_USERS = 40
NATIVE_FRAUD_USER = f"u{N_USERS - 1:02d}"
SPEC = IdentityTheftSpec(victim_fraction=0.3, min_prior_card_events=20)


def _book_records() -> dict[str, Any]:
    """Daily card payments, app sessions every third day, and a monthly salary per user."""
    rng = Random(0)
    events: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    fraud: list[dict[str, Any]] = []
    for u in range(N_USERS):
        user = f"u{u:02d}"
        country = "PL" if u % 5 == 0 else "GB"
        profiles.append(
            {
                "user_id": user,
                "as_of": START,
                "attributes": {"country": country},
                "lifelong": [("account_opened", START)],
            }
        )
        for day in range(DAYS):
            base = START + timedelta(days=day)
            ts = base + timedelta(hours=9 + rng.randrange(10), minutes=rng.randrange(60))
            events.append(
                {
                    "user_id": user,
                    "ts": ts,
                    "source": "transaction",
                    "fields": {
                        "amount": f"{rng.uniform(10, 40):.2f}",
                        "currency": "PLN" if country == "PL" else "GBP",
                        "mcc": "5411",
                        "merchant": f"SHOP {u}-{rng.randrange(6)}",
                        "txn_type": "atm_withdrawal" if day % 11 == 0 else "card_payment",
                        "channel": "contactless",
                        "country": country,
                    },
                }
            )
            if day % 3 == 0:
                events.append(
                    {
                        "user_id": user,
                        "ts": ts - timedelta(minutes=5),
                        "source": "app",
                        "fields": {
                            "screen": "home",
                            "device_id": f"dev_{u}",
                            "geo_country": country,
                        },
                    }
                )
            if day % 30 == 0:
                events.append(
                    {
                        "user_id": user,
                        "ts": base + timedelta(hours=6),
                        "source": "transaction",
                        "fields": {
                            "amount": "2000.00",
                            "currency": "GBP",
                            "mcc": "0000",
                            "merchant": "EMPLOYER",
                            "txn_type": "credit_transfer",
                            "channel": "bank",
                            "country": country,
                        },
                    }
                )
            if user == NATIVE_FRAUD_USER and day == 200:
                fraud.append({"user_id": user, "ts": ts, "label": 1})
    return {"events": events, "profiles": profiles, "labels": {"fraud": fraud}}


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_pragmatiq_dataset(tmp_path_factory.mktemp("theft_book"), _book_records())


def _loaded(book: Path) -> tuple[tuple[TransactionEvent, ...], tuple[EntityProfile, ...], Any]:
    dataset = load_pragmatiq_dataset(book, attach_fraud_labels=False)
    return dataset.events, dataset.profiles, dataset


def _windows(events: tuple[TransactionEvent, ...]) -> Any:
    last = max(e.event_timestamp for e in events)
    return side_label_windows(TRAIN_END, VALIDATION_END, last, timedelta(days=730))


def _splice(book: Path, seed: int = 0, spec: IdentityTheftSpec = SPEC) -> Any:
    events, profiles, _ = _loaded(book)
    return splice_identity_theft(
        events,
        profiles,
        windows=_windows(events),
        spec=spec,
        seed=seed,
        exclude={NATIVE_FRAUD_USER},
    )


def test_the_splice_is_deterministic(book: Path) -> None:
    assert _splice(book, seed=3).manifest() == _splice(book, seed=3).manifest()
    assert _splice(book, seed=3).manifest() != _splice(book, seed=4).manifest()


def test_victim_and_donor_share_a_side_and_each_user_takes_one_role(book: Path) -> None:
    splice = _splice(book)
    side_of = {e: "train" for e in splice.split.train} | {
        e: "evaluation" for e in splice.split.evaluation
    }
    assert splice.records
    assert {r.side for r in splice.records} == {"train", "evaluation"}
    roles = [r.victim for r in splice.records] + [r.donor for r in splice.records]
    assert len(roles) == len(set(roles))
    for record in splice.records:
        assert side_of[record.victim] == side_of[record.donor] == record.side
    assert NATIVE_FRAUD_USER not in side_of


def test_matched_donors_share_the_home_country(book: Path) -> None:
    splice = _splice(book)
    home = {p.entity_id: p.attributes["country"] for p in _loaded(book)[1]}
    assert all(home[r.victim] == home[r.donor] for r in splice.records)


def test_theft_days_lie_in_their_side_window_and_labels_are_event_level(book: Path) -> None:
    events, _, _ = _loaded(book)
    windows = _windows(events)
    splice = _splice(book)
    by_donor_day: dict[str, list[TransactionEvent]] = {}
    for event in events:
        by_donor_day.setdefault(event.entity_id, []).append(event)
    for record in splice.records:
        window = windows["train" if record.side == "train" else "test"]
        day_end = record.theft_day + timedelta(days=1)
        assert window.contains(record.theft_day + timedelta(seconds=1))
        assert window.contains(day_end - timedelta(microseconds=1))
        assert record.theft_day >= window.end - SPEC.theft_lookback - timedelta(days=1)
        history = splice.victim_events[record.victim]
        copies = [e for e in history if ":theft" in e.event_id]
        labelled = [e for e in history if "fraud" in e.labels]
        assert all(record.theft_day <= e.event_timestamp < day_end for e in copies)
        assert all(e.entity_id == record.victim for e in copies)
        for event in labelled:
            label = event.labels["fraud"]
            assert label.level == "event"
            assert event.categorical["txn_type"] in CARD_TXN_TYPES
            assert window.contains(event.event_timestamp)
            if ":theft" in event.event_id:
                assert label.value == 1.0
            else:
                assert label.value == 0.0
                assert record.theft_day - SPEC.negative_window <= event.event_timestamp < day_end
        # Card spending and app sessions are copied; the donor's salary is not.
        donor_day = [
            e for e in by_donor_day[record.donor] if record.theft_day <= e.event_timestamp < day_end
        ]
        expected = [
            e
            for e in donor_day
            if e.categorical.get("source") == "app"
            or e.categorical.get("txn_type") in CARD_TXN_TYPES
        ]
        assert len(copies) == len(expected)
        assert sum(1 for e in copies if "fraud" in e.labels) == record.n_fraud > 0
        assert sum(1 for e in labelled if e.labels["fraud"].value == 0.0) == record.n_legitimate


def test_the_input_book_is_left_unspliced(book: Path) -> None:
    events, _, _ = _loaded(book)
    splice = _splice(book)
    assert not any("fraud" in e.labels for e in events)
    applied = splice.apply(events)
    for victim, history in splice.victim_events.items():
        assert [e for e in applied if e.entity_id == victim] == list(history)
    untouched = {e.entity_id for e in events} - set(splice.victim_events)
    assert [e for e in applied if e.entity_id in untouched] == [
        e for e in events if e.entity_id in untouched
    ]


def test_a_book_already_carrying_fraud_labels_is_rejected(book: Path) -> None:
    events, profiles, _ = _loaded(book)
    first = events[0]
    labelled = TransactionEvent(
        entity_id=first.entity_id,
        event_id=first.event_id,
        event_timestamp=first.event_timestamp,
        categorical=first.categorical,
        numeric=first.numeric,
        labels={"fraud": EventLabel("fraud", 1.0, first.event_timestamp)},
    )
    with pytest.raises(SchemaError, match="already carries"):
        splice_identity_theft(
            (labelled, *events[1:]), profiles, windows=_windows(events), spec=SPEC, seed=0
        )


def _config(book: Path, **data: Any) -> RecipeConfig:
    raw = load_recipe_config(Path("recipes/blueprint_v2/config.smoke.yaml")).model_dump()
    raw["data"].update(
        {
            "path": str(book),
            "observation_window": "730d",
            "sequence_length": 64,
            "max_entities": N_USERS,
            "fraud_labels": "spliced_identity_theft",
            "identity_theft": {"victim_fraction": 0.3},
            "temporal_split": {
                "train_end": TRAIN_END.isoformat(),
                "validation_end": VALIDATION_END.isoformat(),
                "label_delay": "0h",
            },
            **data,
        }
    )
    raw["evaluation"] = {**raw["evaluation"], "history_window": {"enabled": True}}
    return RecipeConfig.model_validate(raw)


def test_prepare_data_keeps_pretraining_and_forecasts_unspliced(book: Path) -> None:
    data = prepare_data(_config(book))
    assert data.fraud is not None
    victims = {r["victim"] for r in data.fraud.manifest["records"]}
    for sequences in (data.pretrain_sequences, data.train_sequences, data.test_sequences):
        for seq in sequences:
            assert not any(":theft" in e.event_id or "fraud" in e.labels for e in seq.events)
    assert not any(":theft" in e.event_id for e in data.events)
    assert any(":theft" in e.event_id for e in data.fraud.events)
    assert victims

    split = fraud_entity_split(data, seed=0)
    assert split == data.fraud.split
    task = fraud_task(data, split)
    train_victims = {r["victim"] for r in data.fraud.manifest["records"] if r["side"] == "train"}
    eval_victims = {
        r["victim"] for r in data.fraud.manifest["records"] if r["side"] == "evaluation"
    }
    train_ids = task.train_batch.entity_ids
    eval_ids = task.eval_batch.entity_ids
    assert set(train_ids) <= set(split.train) and set(eval_ids) <= set(split.evaluation)
    assert {e for e, y in zip(train_ids, task.y_train, strict=True) if y == 1} == train_victims
    assert {e for e, y in zip(eval_ids, task.y_eval, strict=True) if y == 1} == eval_victims

    rows = history_window_tasks(data, _config(book), split)["history_window"]
    assert rows.y_train.sum() > 0 and rows.y_eval.sum() > 0
    assert (rows.y_train == 0).sum() > 0 and (rows.y_eval == 0).sum() > 0

    # The tabular baselines read the same spliced histories as the encoder, row for row.
    for probe in (task, rows):
        assert [s.entity_id for s in probe.train_sequences] == list(probe.train_batch.entity_ids)
        assert [s.entity_id for s in probe.eval_sequences] == list(probe.eval_batch.entity_ids)
    victim_rows = [s for s in task.train_sequences if s.entity_id in train_victims]
    assert victim_rows and all(any(":theft" in e.event_id for e in s.events) for s in victim_rows)


def test_flat_field_aliasing_gives_copies_the_victims_identity(book: Path) -> None:
    loaded = load_adapter_events(_config(book), flat_fields=True)
    assert loaded.fraud_splice is not None
    for victim, history in loaded.fraud_splice.victim_events.items():
        assert {e.categorical["source"] for e in history} == {"transaction"}
        assert len({e.numeric["user"] for e in history}) == 1
        own = next(e for e in loaded.events if e.entity_id == victim)
        assert history[0].numeric["user"] == own.numeric["user"]


def test_the_splice_needs_a_pragmatiq_book() -> None:
    raw = load_recipe_config(Path("tests/fixtures/configs/flat_decoder.yaml")).model_dump()
    raw["data"]["fraud_labels"] = "spliced_identity_theft"
    with pytest.raises(ValueError, match="pragmatiq"):
        RecipeConfig.model_validate(raw)


def test_a_jointly_trained_head_cannot_use_spliced_labels(book: Path) -> None:
    raw = _config(book).model_dump()
    raw["model"]["composition"]["granularity"]["type"] = "flat"
    raw["fraud_task"]["supervised_head"] = {"enabled": True}
    with pytest.raises(ValueError, match="no spliced labels"):
        RecipeConfig.model_validate(raw)


def test_the_generated_fixture_is_too_small_to_splice(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("TFM_PRAGMATIQ_ROOT", raising=False)
    config = _config(tmp_path, path=None)
    with pytest.raises(UnsupportedCombinationError, match="too small"):
        prepare_data(config, fixture_dir=tmp_path / "book")
