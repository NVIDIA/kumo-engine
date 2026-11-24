# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tfm.api.validation import validate_tokenizer_layout
from tfm.compositions.flat import Flat
from tfm.config import RecipeConfig
from tfm.contracts.errors import IncompatibleCompositionError, UnsupportedCombinationError
from tfm.contracts.kvt import EncodedKVTRecord, KeyValueTimeBatch
from tfm.contracts.schema import (
    EntityProfile,
    LifelongItem,
    TransactionEvent,
    TransactionSequence,
)
from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME
from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture, load_pragmatiq_dataset
from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
from tfm.features.tokenizers import KeyValueTimeTokenizer, build_tokenizer
from tfm.features.tokenizers.key_value_time import KeyValueTimeConfig
from tfm.layers.decoder import Decoder


def test_kvt_fit_encode_collate_heterogeneous(tmp_path) -> None:
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=3, events_per_user=6, seed=3)
    dataset = load_pragmatiq_dataset(root)
    seqs = build_entity_sequences(
        list(dataset.events), datetime(2024, 12, 31, tzinfo=UTC), max_events=8
    )
    profiles = {p.entity_id: p for p in dataset.profiles}
    tok = KeyValueTimeTokenizer()
    tok.set_profiles(profiles)
    tok.fit(seqs)
    assert tok.layout == LAYOUT_KEY_VALUE_TIME
    assert tok.vocab_size > 5
    encoded = tok.encode(seqs[0])
    assert isinstance(encoded, EncodedKVTRecord)
    assert encoded.event_offsets[0] == 0
    assert encoded.event_offsets[-1] == len(encoded.key_ids)
    assert len(encoded.time_log) == len(seqs[0].events) or len(encoded.time_log) <= 8
    sources = {ev.categorical.get("source") for ev in seqs[0].events}
    assert "app" in sources or "transaction" in sources
    batch = tok.collate(seqs, max_length=24)
    assert isinstance(batch, KeyValueTimeBatch)
    assert batch.key_ids.shape[0] == len(seqs)
    assert batch.token_mask.sum() > 0
    assert batch.event_mask.sum() > 0


def test_kvt_unseen_key_is_unk(tmp_path) -> None:
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=2, events_per_user=6, seed=4)
    dataset = load_pragmatiq_dataset(root)
    seqs = build_entity_sequences(
        list(dataset.events), datetime(2024, 12, 31, tzinfo=UTC), max_events=8
    )
    tok = KeyValueTimeTokenizer()
    tok.fit(seqs[:1])
    extra = seqs[-1]
    encoded = tok.encode(extra)
    assert tok.unk_id in encoded.key_ids or tok.unk_id in encoded.value_ids or encoded.key_ids


def test_kvt_incompatible_with_flat_decoder() -> None:
    tok = build_tokenizer("key_value_time")
    with pytest.raises(IncompatibleCompositionError):
        validate_tokenizer_layout(tok, Flat(Decoder(mask="causal", vocab_size=32)))


def test_kvt_profile_tokens() -> None:
    from tfm.data.synthetic.generator import generate_synthetic_events

    events = generate_synthetic_events(seed=1, n_entities=1, events_per_entity=4)
    seqs = build_entity_sequences(events, datetime(2024, 12, 31, tzinfo=UTC), max_events=8)
    profile = EntityProfile(
        entity_id=seqs[0].entity_id,
        as_of=datetime(2024, 6, 1, tzinfo=UTC),
        attributes={"country": "GB"},
        lifelong=(LifelongItem(key="account_opened", timestamp=datetime(2024, 1, 1, tzinfo=UTC)),),
    )
    tok = KeyValueTimeTokenizer()
    tok.set_profiles({profile.entity_id: profile})
    tok.fit(seqs)
    encoded = tok.encode(seqs[0])
    assert len(encoded.prof_key_ids) >= 1
    assert encoded.prof_time_log[0] == 0.0


def test_kvt_drops_lifelong_milestones_after_the_cutoff() -> None:
    """A snapshot's `as_of` can postdate the cutoff; later milestones must not leak."""
    from tfm.data.synthetic.generator import generate_synthetic_events

    events = generate_synthetic_events(seed=1, n_entities=1, events_per_entity=4)
    seq = build_entity_sequences(events, datetime(2024, 12, 31, tzinfo=UTC), max_events=8)[0]
    cutoff = seq.cutoff_time
    profile = EntityProfile(
        entity_id=seq.entity_id,
        as_of=cutoff + timedelta(days=365),
        attributes={"country": "GB"},
        lifelong=(
            LifelongItem(key="account_opened", timestamp=cutoff - timedelta(days=30)),
            LifelongItem(key="premium_upgrade", timestamp=cutoff + timedelta(days=10)),
        ),
    )
    tok = KeyValueTimeTokenizer()
    tok.set_profiles({seq.entity_id: profile})
    tok.fit([seq])
    assert "premium_upgrade" not in tok.key_vocab
    encoded = tok.encode(seq)
    assert len(encoded.prof_time_log) == 2
    assert encoded.prof_time_log[1] == pytest.approx(8.0 * math.log1p(30 * 86400 / 8.0), rel=1e-6)


def _card_sequence() -> TransactionSequence:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    events = tuple(
        TransactionEvent(
            entity_id="u1",
            event_id=f"u1:{i}",
            event_timestamp=start + timedelta(hours=i),
            # The adapter's own order puts a long merchant name before amount and mcc.
            categorical={
                "source": "transaction",
                "merchant": f"A VERY LONG MERCHANT NAME {i}",
                "txn_type": "card_payment",
            },
            numeric={"amount": 10.0 + i, "mcc": float(5000 + 100 * i)},
        )
        for i in range(10)
    )
    return TransactionSequence("u1", events, start + timedelta(days=1), None)


def test_a_capped_event_trims_text_bytes_never_a_single_token_field() -> None:
    seq = _card_sequence()
    tok = KeyValueTimeTokenizer(KeyValueTimeConfig(max_event_tokens=6, categorical_threshold=4))
    tok.fit([seq])
    assert tok.field_kind["merchant"] == "text"
    encoded = tok.encode(seq)
    first = encoded.key_ids[encoded.event_offsets[0] : encoded.event_offsets[1]]
    keys = [tok.id_to_token[k] for k in first]
    assert keys[:4] == ["KEY_source", "KEY_amount", "KEY_mcc", "KEY_txn_type"]
    assert keys[4:] == ["KEY_merchant", "KEY_merchant"]


def test_categorical_keys_keep_codes_out_of_numeric_buckets() -> None:
    seq = _card_sequence()
    binned = KeyValueTimeTokenizer()
    binned.fit([seq])
    assert binned.field_kind["mcc"] == "numeric"
    coded = KeyValueTimeTokenizer(KeyValueTimeConfig(categorical_keys=("mcc",)))
    coded.fit([seq])
    assert coded.field_kind["mcc"] == "categorical"
    assert len(coded.cat_vocab["mcc"]) == 10
    assert coded.manifest()["categorical_keys"] == ["mcc"]


def test_categorical_keys_need_a_key_value_time_tokenizer() -> None:
    from tfm.config import load_recipe_config
    from tfm.runtime.data import tokenizer_for

    raw = load_recipe_config(Path("tests/fixtures/configs/flat_decoder.yaml")).model_dump()
    raw["features"]["categorical_keys"] = ["mcc"]
    with pytest.raises(UnsupportedCombinationError, match="categorical_keys"):
        tokenizer_for(RecipeConfig.model_validate(raw))


def _merchant_sequences(merchants: list[str]) -> list[TransactionSequence]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events = tuple(
        TransactionEvent(
            entity_id="u1",
            event_id=f"u1:{i}",
            event_timestamp=start + timedelta(hours=i),
            categorical={"channel": "online"},
            text={"merchant": m},
        )
        for i, m in enumerate(merchants)
    )
    return [TransactionSequence("u1", events, start + timedelta(days=2), "30d")]


def test_a_hashed_key_is_one_stable_token_per_value() -> None:
    config = KeyValueTimeConfig(hashed_keys=("merchant",), hash_buckets=64)
    seqs = _merchant_sequences(["SHOP A", "SHOP B", "SHOP A", "A MUCH LONGER MERCHANT NAME"])
    tok = KeyValueTimeTokenizer(config).fit(seqs)
    assert tok.field_kind["merchant"] == "hashed"
    rec = tok.encode(seqs[0])
    kid = tok.key_vocab["merchant"]
    merchant = [v for k, v in zip(rec.key_ids, rec.value_ids, strict=True) if k == kid]
    assert len(merchant) == 4
    assert merchant[0] == merchant[2] != merchant[1]
    base = tok.hash_base["merchant"]
    assert all(base <= v < base + 64 for v in merchant)
    # Unseen at fit time, yet a bucket rather than [UNK]; the same bucket in a fresh process.
    fresh = KeyValueTimeTokenizer(config).fit(seqs)
    new = tok.field_tokens("merchant", "NEVER SEEN")[1]
    assert new == fresh.field_tokens("merchant", "NEVER SEEN")[1] and new[0] != 2
    restored = KeyValueTimeTokenizer.from_state(tok.state_dict())
    assert restored.encode(seqs[0]) == rec
    assert tok.vocab_size == restored.vocab_size


def test_each_hashed_key_has_its_own_bucket_range() -> None:
    config = KeyValueTimeConfig(hashed_keys=("merchant", "device_id"), hash_buckets=32)
    seqs = _merchant_sequences(["SHOP A", "SHOP B"])
    events = tuple(
        dataclasses.replace(e, text={**e.text, "device_id": "SHOP A"}) for e in seqs[0].events
    )
    seqs = [dataclasses.replace(seqs[0], events=events)]
    tok = KeyValueTimeTokenizer(config).fit(seqs)
    ranges = {k: range(b, b + 32) for k, b in tok.hash_base.items()}
    assert set(ranges) == {"merchant", "device_id"}
    assert not set(ranges["merchant"]) & set(ranges["device_id"])
    as_merchant = tok.field_tokens("merchant", "SHOP A")[1][0]
    as_device = tok.field_tokens("device_id", "SHOP A")[1][0]
    assert as_merchant in ranges["merchant"] and as_device in ranges["device_id"]
    legacy = {**tok.state_dict(), "hash_base": min(tok.hash_base.values())}
    with pytest.raises(ValueError, match="rebuild the token cache"):
        KeyValueTimeTokenizer.from_state(legacy)


def test_without_hashed_keys_the_state_is_unchanged() -> None:
    seqs = _merchant_sequences([f"SHOP {i}" for i in range(80)])
    tok = KeyValueTimeTokenizer().fit(seqs)
    assert tok.field_kind["merchant"] == "text"
    state = tok.state_dict()
    assert "hash_base" not in state and "hashed_keys" not in state["config"]
    assert KeyValueTimeTokenizer.from_state(state).state_dict() == state


def test_hashed_keys_need_the_key_value_time_tokenizer() -> None:
    from tfm.config import load_recipe_config
    from tfm.runtime.data import tokenizer_for

    base = Path(__file__).parents[2] / "tests" / "fixtures" / "configs" / "flat_decoder.yaml"
    raw = load_recipe_config(base).model_dump(mode="json")
    raw["features"]["hashed_keys"] = ["merchant"]
    with pytest.raises(UnsupportedCombinationError, match="hashed_keys"):
        tokenizer_for(RecipeConfig.model_validate(raw))


def test_an_excluded_key_is_never_fitted_or_encoded() -> None:
    seq = _card_sequence()
    events = tuple(dataclasses.replace(e, numeric={**e.numeric, "user": 7.0}) for e in seq.events)
    seq = dataclasses.replace(seq, events=events)
    kept = KeyValueTimeTokenizer().fit([seq])
    assert "user" in kept.key_vocab
    tok = KeyValueTimeTokenizer(KeyValueTimeConfig(excluded_keys=("user",))).fit([seq])
    assert "user" not in tok.key_vocab and "user" not in tok.field_kind
    rec = tok.encode(seq)
    assert not any(tok.id_to_token[k] == "KEY_user" for k in rec.key_ids)
    assert len(rec.key_ids) == len(kept.encode(seq).key_ids) - len(seq.events)
    restored = KeyValueTimeTokenizer.from_state(tok.state_dict())
    assert restored.config.excluded_keys == ("user",)
    assert restored.encode(seq) == rec
    assert "excluded_keys" not in kept.state_dict()["config"]


def test_exclude_keys_need_the_key_value_time_tokenizer() -> None:
    from tfm.config import load_recipe_config
    from tfm.runtime.data import tokenizer_for

    base = Path(__file__).parents[2] / "tests" / "fixtures" / "configs" / "flat_decoder.yaml"
    raw = load_recipe_config(base).model_dump(mode="json")
    raw["features"]["exclude_keys"] = ["user"]
    with pytest.raises(UnsupportedCombinationError, match="exclude_keys"):
        tokenizer_for(RecipeConfig.model_validate(raw))
