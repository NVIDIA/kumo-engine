# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta

import pytest
from tests.conftest import make_event

from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import SchemaError
from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
from tfm.data.splits.temporal import TemporalSplitSpec, assign_temporal_split
from tfm.data.synthetic.generator import generate_synthetic_events
from tfm.data.transforms.ordering import order_events
from tfm.data.validation.schema import validate_event
from tfm.features.tokenizers.financial import TransactionFieldTokenizer
from tfm.layers.masks import causal_mask


def test_validate_event_rejects_missing_entity() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(SchemaError):
        validate_event(make_event("", "e1", ts))


def test_validate_event_rejects_label_like_input() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    event = make_event("ent", "e1", ts)
    object.__setattr__(event, "numeric", {"amount": 1.0, "is_fraud": 1.0})
    with pytest.raises(SchemaError):
        validate_event(event)


def test_order_events_is_deterministic() -> None:
    events = generate_synthetic_events(seed=7, n_entities=3, events_per_entity=5)
    assert [e.event_id for e in order_events(events)] == [
        e.event_id for e in order_events(list(reversed(events)))
    ]


def test_temporal_split_partitions() -> None:
    events = generate_synthetic_events(seed=42, n_entities=6, events_per_entity=16)
    spec = TemporalSplitSpec(
        train_end=datetime(2024, 1, 4, tzinfo=UTC),
        validation_end=datetime(2024, 1, 7, tzinfo=UTC),
        label_delay=timedelta(hours=12),
    )
    split = assign_temporal_split(events, spec)
    assert split.train and split.validation and split.test
    assert all(e.event_timestamp < spec.train_end for e in split.train)
    assert all(spec.train_end <= e.event_timestamp < spec.validation_end for e in split.validation)
    assert all(e.event_timestamp >= spec.validation_end for e in split.test)


def test_sequence_builder_max_events() -> None:
    events = generate_synthetic_events(seed=1, n_entities=2, events_per_entity=10)
    cutoff = datetime(2024, 12, 31, tzinfo=UTC)
    seqs = build_entity_sequences(events, cutoff, max_events=3)
    assert seqs
    assert all(len(s.events) <= 3 for s in seqs)


def test_tokenizer_vocab_and_encode() -> None:
    tok = TransactionFieldTokenizer()
    assert tok.pad_id == 0
    assert tok.layout == "flat_fields"
    assert tok.vocab_size > 1000
    events = generate_synthetic_events(seed=0, n_entities=1, events_per_entity=4)
    seqs = build_entity_sequences(events, datetime(2024, 12, 31, tzinfo=UTC), max_events=8)
    encoded = tok.encode(seqs[0])
    assert encoded.layout == "flat_fields"
    assert encoded.token_ids[0] == tok.bos_id
    ids, attn = tok.encode_sequence(seqs[0], max_length=64)
    assert len(ids) == 64
    assert ids[0] == tok.bos_id
    assert sum(attn) < 64 or attn[-1] == 1


def test_build_tokenizer_from_registry() -> None:
    from tfm.features.tokenizers import build_tokenizer

    tok = build_tokenizer("transaction_field_tokenizer")
    assert tok.name == "transaction_field_tokenizer"
    assert tok.layout == "flat_fields"


def test_causal_mask_lower_triangular() -> None:
    import torch

    mask = causal_mask(4, torch.device("cpu"))
    assert mask[0, 0] and not mask[0, 1]
    assert mask[3].all()


def test_load_recipe_config() -> None:
    from pathlib import Path

    path = Path("tests/fixtures/configs/flat_decoder.yaml")
    cfg = load_recipe_config(path)
    assert isinstance(cfg, RecipeConfig)
    assert cfg.seed == 42
    assert cfg.model.composition.granularity.type == "flat"


def test_removed_config_keys_name_their_replacement() -> None:
    import pytest
    from pydantic import ValidationError

    from tfm.config import ModelConfig

    with pytest.raises(ValidationError, match="composition.granularity.type"):
        ModelConfig.model_validate({"architecture": "decoder_only"})
    with pytest.raises(ValidationError, match="composition.readout"):
        ModelConfig.model_validate({"pooling": "last_valid_token"})


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("data", "pretrain_window_events", "data.pretrain_windows"),
        ("data", "pretrain_max_tokens", "data.pretrain_windows"),
        ("training", "packing_seq_length", "max_seq_len"),
        ("fraud_task", "train_head", "fraud_task.supervised_head"),
        ("evaluation", "history_window_probe", "evaluation.history_window"),
        ("evaluation", "history_window_max_rows", "evaluation.history_window"),
    ],
)
def test_keys_moved_into_sub_sections_name_their_new_home(
    section: str, key: str, replacement: str
) -> None:
    from pydantic import ValidationError

    raw = load_recipe_config("tests/fixtures/configs/flat_decoder.yaml").model_dump(mode="json")
    raw[section][key] = 1
    with pytest.raises(ValidationError, match=replacement.replace(".", r"\.")):
        RecipeConfig.model_validate(raw)

    raw = load_recipe_config("tests/fixtures/configs/flat_decoder.yaml").model_dump(mode="json")
    raw["training"]["optimizer"]["muon_learning_rate"] = 0.01
    with pytest.raises(ValidationError, match=r"training\.optimizer\.muon"):
        RecipeConfig.model_validate(raw)


def test_unknown_config_keys_are_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    from tfm.config import FeaturesConfig

    with pytest.raises(ValidationError, match="Extra inputs"):
        FeaturesConfig.model_validate({"tokenizer": "key_value_time", "tokeniser_typo": 1})


def test_tokenizer_fields_select_and_order_the_layout() -> None:
    from tfm.features.tokenizers.financial import FIELD_ORDER

    full = TransactionFieldTokenizer()
    fields = tuple(f for f in FIELD_ORDER if f != "CUST")
    tok = TransactionFieldTokenizer(fields=fields)
    assert tok.vocab_size == full.vocab_size - (full.max_cust + 1)
    assert not any(token.startswith("CUST_") for token in tok.token_to_id)
    assert sorted(tok.id_to_token) == list(range(tok.vocab_size))
    event = generate_synthetic_events(seed=0, n_entities=1, events_per_entity=1)[0]
    assert tok.event_tokens(event) == [
        t for t in full.event_tokens(event) if not t.startswith("CUST_")
    ]
    assert tok.manifest()["field_order"] == list(fields)
    reordered = TransactionFieldTokenizer(fields=("HOUR", "AMT"))
    assert [t.split("_")[0] for t in reordered.event_tokens(event)] == ["HOUR", "AMT"]


@pytest.mark.parametrize("fields", [(), ("AMT", "AMT"), ("AMT", "NOPE")])
def test_tokenizer_rejects_invalid_fields(fields: tuple[str, ...]) -> None:
    from tfm.contracts.errors import UnsupportedCombinationError

    with pytest.raises(UnsupportedCombinationError, match="fields"):
        TransactionFieldTokenizer(fields=fields)


def test_truncation_keeps_bos_and_the_most_recent_whole_events() -> None:
    tok = TransactionFieldTokenizer(fields=("AMT", "HOUR"))
    events = generate_synthetic_events(seed=3, n_entities=1, events_per_entity=6)
    seq = build_entity_sequences(events, datetime(2024, 12, 31, tzinfo=UTC), max_events=6)[0]
    full = list(tok.encode(seq).token_ids)
    assert len(full) == 1 + 6 * 2 + 5 + 1
    ids, attn = tok.encode_sequence(seq, max_length=11)
    kept = [i for i, a in zip(ids, attn, strict=True) if a]
    # <bos>, then the last three events (2 fields each, <sep> between), then <eos>.
    assert kept[0] == tok.bos_id
    assert kept[1:] == full[-9:]
    assert kept[-1] == tok.eos_id
    assert kept[1] != tok.sep_id
    # A cut that lands on an event boundary keeps that event.
    ids, attn = tok.encode_sequence(seq, max_length=10)
    assert [i for i, a in zip(ids, attn, strict=True) if a] == [tok.bos_id, *full[-9:]]


def test_features_fields_reach_the_tokenizer() -> None:
    from tfm.runtime.data import tokenizer_for

    cfg = load_recipe_config("tests/fixtures/configs/flat_decoder.yaml")
    edited = cfg.model_copy(
        update={"features": cfg.features.model_copy(update={"fields": ["AMT", "MCC"]})}
    )
    tok = tokenizer_for(edited)
    assert isinstance(tok, TransactionFieldTokenizer)
    assert tok.fields == ("AMT", "MCC")


@pytest.mark.parametrize("key", ["pretrain_window_split_by", "pretrain_observation_window"])
def test_removed_pretrain_keys_explain_themselves(key: str) -> None:
    from pydantic import ValidationError

    from tfm.config import DataConfig

    with pytest.raises(ValidationError, match="ADR 0017"):
        DataConfig.model_validate(
            {
                key: "card",
                "temporal_split": {
                    "train_end": "2024-01-04T00:00:00Z",
                    "validation_end": "2024-01-07T00:00:00Z",
                },
            }
        )
