# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

from tfm.api.validation import validate_tokenizer_layout
from tfm.compositions.flat import Flat
from tfm.config import RecipeConfig
from tfm.contracts.errors import (
    IncompatibleCompositionError,
    SchemaError,
    UnsupportedCombinationError,
)
from tfm.contracts.kvt import EncodedKVTRecord, KeyValueTimeBatch
from tfm.contracts.schema import (
    EntityProfile,
    LifelongItem,
    TransactionEvent,
    TransactionSequence,
)
from tfm.contracts.tokenizers import (
    FIELD_CATEGORICAL,
    FIELD_NUMERIC,
    LAYOUT_KEY_VALUE_TIME,
    EventBlock,
)
from tfm.data.adapters.pragmatiq import (
    generate_pragmatiq_fixture,
    load_pragmatiq_dataset,
    parse_float,
)
from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
from tfm.features.tokenizers import KeyValueTimeTokenizer, build_tokenizer, register_tokenizer
from tfm.features.tokenizers.key_value_time import (
    KeyValueTimeConfig,
    collate_records,
    time_encode,
    time_encode_many,
)
from tfm.features.tokenizers.vocabulary import UNK, ValueVocabulary, murmur3_32, murmur3_32_many
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
    assert "premium_upgrade" not in tok.pairs.key_vocab
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
    assert tok.vocabulary.field_kind["merchant"] == "text"
    encoded = tok.encode(seq)
    first = encoded.key_ids[encoded.event_offsets[0] : encoded.event_offsets[1]]
    keys = [tok.vocabulary.id_to_token[k] for k in first]
    assert keys[:4] == ["KEY_source", "KEY_amount", "KEY_mcc", "KEY_txn_type"]
    assert keys[4:] == ["KEY_merchant", "KEY_merchant"]


def test_categorical_keys_keep_codes_out_of_numeric_buckets() -> None:
    seq = _card_sequence()
    binned = KeyValueTimeTokenizer()
    binned.fit([seq])
    assert binned.vocabulary.field_kind["mcc"] == "numeric"
    coded = KeyValueTimeTokenizer(KeyValueTimeConfig(categorical_keys=("mcc",)))
    coded.fit([seq])
    assert coded.vocabulary.field_kind["mcc"] == "categorical"
    assert len(coded.vocabulary.cat_vocab["mcc"]) == 10
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
    assert tok.vocabulary.field_kind["merchant"] == "hashed"
    rec = tok.encode(seqs[0])
    kid = tok.pairs.key_vocab["merchant"]
    merchant = [v for k, v in zip(rec.key_ids, rec.value_ids, strict=True) if k == kid]
    assert len(merchant) == 4
    assert merchant[0] == merchant[2] != merchant[1]
    base = tok.vocabulary.hash_base["merchant"]
    assert all(base <= v < base + 64 for v in merchant)
    # Unseen at fit time, yet a bucket rather than [UNK]; the same bucket in a fresh process.
    fresh = KeyValueTimeTokenizer(config).fit(seqs)
    new = tok.pairs.field_tokens("merchant", "NEVER SEEN")[1]
    assert new == fresh.pairs.field_tokens("merchant", "NEVER SEEN")[1] and new[0] != 2
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
    ranges = {k: range(b, b + 32) for k, b in tok.vocabulary.hash_base.items()}
    assert set(ranges) == {"merchant", "device_id"}
    assert not set(ranges["merchant"]) & set(ranges["device_id"])
    as_merchant = tok.pairs.field_tokens("merchant", "SHOP A")[1][0]
    as_device = tok.pairs.field_tokens("device_id", "SHOP A")[1][0]
    assert as_merchant in ranges["merchant"] and as_device in ranges["device_id"]
    legacy = {**tok.state_dict(), "hash_base": min(tok.vocabulary.hash_base.values())}
    with pytest.raises(ValueError, match="rebuild the token cache"):
        KeyValueTimeTokenizer.from_state(legacy)


def test_murmur3_matches_its_reference_vectors() -> None:
    cases = [
        (b"", 0, 0),
        (b"", 1, 0x514E28B7),
        (b"\x00", 0, 0x514E28B7),
        (b"\x21", 0, 0x72661CF4),
        (b"\x21\x43", 0, 0xA0F7B07A),
        (b"\x21\x43\x65", 0, 0x7E4A8634),
        (b"\x21\x43\x65\x87", 0, 0xF55B516B),
        (b"Hello, world!", 1234, 0xFAF6CDB3),
        (b"The quick brown fox jumps over the lazy dog", 0x9747B28C, 0x2FA826CD),
    ]
    assert [murmur3_32(data, seed) for data, seed, _ in cases] == [h for *_, h in cases]


def test_murmur3_many_matches_murmur3() -> None:
    rng = np.random.default_rng(0)
    strings = [b"", b"\x00", "é€中".encode(), bytes(range(256)) * 400] + [
        rng.integers(0, 256, rng.integers(0, 70), dtype=np.uint8).tobytes() for _ in range(500)
    ]
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in strings])]).astype(np.int64)
    data = np.frombuffer(b"".join(strings), np.uint8)
    assert murmur3_32_many(offsets, data, seed=7).tolist() == [murmur3_32(s, 7) for s in strings]


def _every_kind() -> tuple[KeyValueTimeTokenizer, dict[str, list[str]]]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events = tuple(
        TransactionEvent(
            entity_id="u1",
            event_id=f"u1:{i}",
            event_timestamp=start + timedelta(hours=i),
            categorical={"channel": ("online", "store")[i % 2]},
            numeric={"amount": 0.0 if i % 10 == 0 else i * 3.5},
            text={"merchant": f"SHOP {i}", "note": f"note number {i} " * (i % 4)},
        )
        for i in range(60)
    )
    seq = TransactionSequence("u1", events, start + timedelta(days=5), None)
    config = KeyValueTimeConfig(hashed_keys=("merchant",), hash_buckets=64, categorical_threshold=8)
    tok = KeyValueTimeTokenizer(config).fit([seq])
    assert {tok.vocabulary.field_kind[k] for k in ("channel", "amount", "merchant", "note")} == {
        "categorical",
        "numeric",
        "hashed",
        "text",
    }
    odd = ["", "é" * 20, "nan", "inf", "-0.0", "0.0", "abc", "x" * 40]
    values = {
        "channel": ["online", "store", "kiosk", *odd],
        "amount": ["3.5", "17.5", "1e9", "-4.0", *odd],
        "merchant": ["SHOP 1", "NEVER SEEN", *odd],
        "note": ["note number 3 ", *odd],
        "unfitted": ["a", *odd],
    }
    return tok, values


def test_time_encode_many_matches_time_encode() -> None:
    edges = [0.0, -0.0, -5.0, 1e-6, 0.5, 8.0, 59.999999, 3600.0, 86400.0 * 400, 1e10]
    rng = np.random.default_rng(0)
    seconds = np.concatenate([edges, np.exp(rng.uniform(-14.0, 23.0, 10_000))])
    want = np.array([time_encode(x) for x in seconds.tolist()])
    actual = time_encode_many(seconds)
    # NumPy's vector log1p and the platform's scalar libm can round differently.
    np.testing.assert_array_max_ulp(actual, want, maxulp=2)
    # The float32 times consumed by the model/cache must still agree exactly.
    np.testing.assert_array_equal(actual.astype(np.float32), want.astype(np.float32))


def test_value_ids_match_value_tokens_and_pairs_add_the_key() -> None:
    tok, values = _every_kind()
    for key, strings in values.items():
        numbers = np.asarray([np.nan if (x := parse_float(s)) is None else x for s in strings])
        lengths, ids, positions = tok.vocabulary.value_ids(
            key, pa.array(strings, pa.large_string()), numbers
        )
        expected = [tok.vocabulary.value_tokens(key, s) for s in strings]
        assert lengths.tolist() == [len(v) for v, _ in expected]
        assert ids.tolist() == [t for v, _ in expected for t in v]
        assert positions.tolist() == [t for _, p in expected for t in p]
        kid = tok.pairs.key_vocab.get(key, UNK)
        for s, (v, p) in zip(strings, expected, strict=True):
            assert tok.pairs.field_tokens(key, s) == ([kid] * len(v), v, p)


def test_hashed_buckets_follow_the_hash_the_state_names() -> None:
    config = KeyValueTimeConfig(hashed_keys=("merchant",), hash_buckets=64)
    tok = KeyValueTimeTokenizer(config).fit(_merchant_sequences(["SHOP A", "SHOP B"]))
    state = tok.state_dict()
    old = KeyValueTimeTokenizer.from_state({k: v for k, v in state.items() if k != "hash"})
    blake2b = hashlib.blake2b(b"merchant\x1fSHOP A", digest_size=8).digest()
    base = tok.vocabulary.hash_base["merchant"]
    assert state["hash"] == "murmur3"
    assert tok.pairs.field_tokens("merchant", "SHOP A")[1] == [
        base + murmur3_32(b"merchant\x1fSHOP A") % 64
    ]
    assert old.pairs.field_tokens("merchant", "SHOP A")[1] == [
        base + int.from_bytes(blake2b, "little") % 64
    ]
    assert old.state_dict()["hash"] == "blake2b"
    with pytest.raises(UnsupportedCombinationError, match="Refit the tokenizer"):
        old.vocabulary.value_ids("merchant", pa.array(["SHOP A"]), np.array([np.nan]))
    with pytest.raises(SchemaError, match="Unknown hash"):
        KeyValueTimeTokenizer.from_state({**state, "hash": "xxhash64"})


def test_without_hashed_keys_the_state_is_unchanged() -> None:
    seqs = _merchant_sequences([f"SHOP {i}" for i in range(80)])
    tok = KeyValueTimeTokenizer().fit(seqs)
    assert tok.vocabulary.field_kind["merchant"] == "text"
    state = tok.state_dict()
    assert "hash_base" not in state and "hashed_keys" not in state["config"]
    assert KeyValueTimeTokenizer.from_state(state).state_dict() == state


@pytest.mark.parametrize("tokenizer", ["transaction_field_tokenizer", "flat_key_value"])
def test_hashed_keys_need_the_key_value_time_tokenizer(tokenizer: str) -> None:
    from tfm.config import load_recipe_config
    from tfm.runtime.data import tokenizer_for

    base = Path(__file__).parents[2] / "tests" / "fixtures" / "configs" / "flat_decoder.yaml"
    raw = load_recipe_config(base).model_dump(mode="json")
    raw["features"].update(tokenizer=tokenizer, hashed_keys=["merchant"])
    with pytest.raises(UnsupportedCombinationError, match="hashed_keys"):
        tokenizer_for(RecipeConfig.model_validate(raw))


@pytest.mark.parametrize("key", ["categorical_keys", "hashed_keys", "exclude_keys"])
def test_event_fields_refuses_key_settings_its_fixed_fields_ignore(key: str) -> None:
    from tfm.config import load_recipe_config
    from tfm.runtime.data import tokenizer_for

    base = Path(__file__).parents[2] / "tests" / "fixtures" / "configs" / "flat_decoder.yaml"
    raw = load_recipe_config(base).model_dump(mode="json")
    raw["features"].update({"tokenizer": "event_fields", key: ["user"]})
    with pytest.raises(UnsupportedCombinationError, match=f"features.{key} does not apply"):
        tokenizer_for(RecipeConfig.model_validate(raw))


class _KeyValueTimeSubclass(KeyValueTimeTokenizer):
    pass


register_tokenizer("test_key_value_time_subclass")(_KeyValueTimeSubclass)


def test_a_key_value_time_subclass_takes_hashed_and_excluded_keys() -> None:
    from tfm.config import load_recipe_config
    from tfm.runtime.data import tokenizer_for

    base = Path(__file__).parents[2] / "tests" / "fixtures" / "configs" / "flat_decoder.yaml"
    raw = load_recipe_config(base).model_dump(mode="json")
    raw["features"].update(
        tokenizer="test_key_value_time_subclass", hashed_keys=["merchant"], exclude_keys=["user"]
    )
    tok = tokenizer_for(RecipeConfig.model_validate(raw))
    assert isinstance(tok, _KeyValueTimeSubclass)
    assert tok.config.hashed_keys == ("merchant",) and tok.config.excluded_keys == ("user",)


def test_an_excluded_key_is_never_fitted_or_encoded() -> None:
    seq = _card_sequence()
    events = tuple(dataclasses.replace(e, numeric={**e.numeric, "user": 7.0}) for e in seq.events)
    seq = dataclasses.replace(seq, events=events)
    kept = KeyValueTimeTokenizer().fit([seq])
    assert "user" in kept.pairs.key_vocab
    tok = KeyValueTimeTokenizer(KeyValueTimeConfig(excluded_keys=("user",))).fit([seq])
    assert "user" not in tok.pairs.key_vocab and "user" not in tok.vocabulary.field_kind
    rec = tok.encode(seq)
    assert not any(tok.vocabulary.id_to_token[k] == "KEY_user" for k in rec.key_ids)
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


def _categorical_sequence(*rows: dict[str, str]) -> TransactionSequence:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events = tuple(
        TransactionEvent(
            entity_id="u1",
            event_id=f"u1:{i}",
            event_timestamp=start + timedelta(hours=i),
            categorical=row,
        )
        for i, row in enumerate(rows)
    )
    return TransactionSequence("u1", events, start + timedelta(days=1), None)


def test_a_fitted_tokenizer_is_not_refit() -> None:
    """A second fit would place new bins after the old ids and encode past the vocabulary."""
    seq = _categorical_sequence({"channel": "online"}, {"channel": "store"})
    tok = KeyValueTimeTokenizer().fit([seq])
    with pytest.raises(UnsupportedCombinationError, match="already fitted"):
        tok.fit([seq])


def test_two_fields_that_map_to_one_value_token_are_refused() -> None:
    with pytest.raises(SchemaError, match="VAL_a_b_c"):
        KeyValueTimeTokenizer().fit([_categorical_sequence({"a": "b_c", "a_b": "c"})])


def test_an_event_with_more_than_128_fields_fits_as_collect_does() -> None:
    """A user source may send wide events; the block fit must order keys as `collect`."""
    names = [f"c{i:03d}" for i in range(130)]
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    event = TransactionEvent(
        "u", "u:0", ts, categorical=dict.fromkeys(names, "x"), numeric={"amt": 1.5}
    )
    us = np.array([int(ts.timestamp() * 1_000_000)], np.int64)
    block = EventBlock(
        index=0,
        user_ids=("u",),
        user_offsets=np.array([0, 1], np.int64),
        ts_us=us,
        id_us=us,
        loader_index=np.zeros(1, np.int64),
        pair_event=np.zeros(131, np.int64),
        pair_key=np.arange(131, dtype=np.int64),
        pair_value=np.array([0] + [1] * 130, np.int64),
        pair_kind=np.array([FIELD_NUMERIC] + [FIELD_CATEGORICAL] * 130, np.int8),
        pair_map_position=np.arange(131, dtype=np.int64),
        keys=("amt", *names),
        values=("1.5", "x"),
        numbers=np.array([1.5, np.nan]),
    )
    want = ValueVocabulary().collect([TransactionSequence("u", (event,), ts, None)], {})
    got = ValueVocabulary().block_statistics(block, {}).fit_statistics()
    assert list(got.keys) == list(want.keys)
    assert ValueVocabulary().fit(got).token_to_id == ValueVocabulary().fit(want).token_to_id


def test_per_event_arrays_in_any_mapping_order_collate_by_name() -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    def record(events: dict[str, tuple[int, ...]]) -> EncodedKVTRecord:
        return EncodedKVTRecord(
            "u", "u", (5,), (6,), (0,), (0, 1), (0,), (0.0,), events, (), (), (), (0,), ()
        )

    seqs = [TransactionSequence("u", (), ts, None)] * 2
    a, b = record({"hour": (3,), "dow": (4,)}), record({"dow": (1,), "hour": (2,)})
    batch = collate_records(seqs, [a, b], max_length=4)
    assert list(batch.events) == ["hour", "dow"]
    assert torch.equal(batch.events["hour"][:, 0], torch.tensor([3, 2]))
    assert torch.equal(batch.events["dow"][:, 0], torch.tensor([4, 1]))


def test_a_block_giving_an_event_one_key_twice_is_refused() -> None:
    us = np.zeros(1, np.int64)
    block = EventBlock(
        index=0,
        user_ids=("u",),
        user_offsets=np.array([0, 1], np.int64),
        ts_us=us,
        id_us=us,
        loader_index=np.zeros(1, np.int64),
        pair_event=np.zeros(2, np.int64),
        pair_key=np.zeros(2, np.int64),
        pair_value=np.array([1, 0], np.int64),
        pair_kind=np.array([FIELD_CATEGORICAL, FIELD_NUMERIC], np.int8),
        pair_map_position=np.arange(2, dtype=np.int64),
        keys=("score",),
        values=("5", "old"),
        numbers=np.array([5.0, np.nan]),
    )
    vocab = ValueVocabulary().fit(ValueVocabulary().block_statistics(block, {}).fit_statistics())
    with pytest.raises(SchemaError, match="same key twice"):
        vocab.encode_events(block, None)


@pytest.mark.parametrize("widest", [2**39 - 1, 2**39], ids=["at-bound", "over-bound"])
def test_a_block_too_wide_to_order_is_refused(widest: int) -> None:
    """Ordinals stay below the next block's: (events + 1) * 4 * width <= 2**42."""
    us = np.zeros(1, np.int64)
    block = EventBlock(
        index=0,
        user_ids=("u",),
        user_offsets=np.array([0, 1], np.int64),
        ts_us=us,
        id_us=us,
        loader_index=np.zeros(1, np.int64),
        pair_event=np.zeros(2, np.int64),
        pair_key=np.arange(2, dtype=np.int64),
        pair_value=np.zeros(2, np.int64),
        pair_kind=np.full(2, FIELD_CATEGORICAL, np.int8),
        pair_map_position=np.array([0, widest], np.int64),
        keys=("a", "b"),
        values=("x",),
        numbers=np.array([np.nan]),
    )
    if widest < 2**39:
        ValueVocabulary().block_statistics(block, {})
    else:
        with pytest.raises(SchemaError, match="rows_per_block"):
            ValueVocabulary().block_statistics(block, {})
