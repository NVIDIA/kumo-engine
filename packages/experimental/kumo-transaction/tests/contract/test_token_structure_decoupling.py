# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token scheme and sequence structure are independent slots (building-block contracts).

Key-value tokens can feed the flat decoder (`flat_key_value`), and the 12 flat
field tokens can feed the hierarchical encoders (`event_fields`). Rule 4 still
holds: every layout carries an event boundary its structure can read.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from tfm.api.validation import validate_tokenizer_layout
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.schema import TransactionEvent, TransactionSequence
from tfm.features.tokenizers.event_fields import EventFieldsTokenizer
from tfm.features.tokenizers.financial import FIELD_ORDER, TransactionFieldTokenizer
from tfm.features.tokenizers.flat_key_value import FlatKeyValueTokenizer
from tfm.features.tokenizers.key_value_time import EVT, PAD, USR
from tfm.runtime.data import prepare_data
from tfm.runtime.pipeline import build_model

V2 = Path("recipes/blueprint_v2/config.smoke.yaml")
FLAT = Path("tests/fixtures/configs/flat_decoder.yaml")


def _with(path: Path, **updates: object) -> RecipeConfig:
    raw = load_recipe_config(path).model_dump(mode="json")
    for dotted, value in updates.items():
        node = raw
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return RecipeConfig.model_validate(raw)


def _flat_kv_config() -> RecipeConfig:
    """Blueprint v2's book through the flat decoder."""
    v2 = load_recipe_config(V2).model_dump(mode="json")
    flat = load_recipe_config(FLAT).model_dump(mode="json")
    flat["data"] = {**v2["data"], "max_tokens": 160}
    flat["features"] = {"tokenizer": "flat_key_value"}
    return RecipeConfig.model_validate(flat)


def test_flat_key_value_feeds_the_flat_decoder(tmp_path: Path) -> None:
    config = _flat_kv_config()
    data = prepare_data(config, fixture_dir=tmp_path / "book")
    assert isinstance(data.tokenizer, FlatKeyValueTokenizer)
    batch = data.batches[0]
    assert isinstance(batch, TransactionBatch)
    model = build_model(config, data.tokenizer.vocab_size)
    validate_tokenizer_layout(data.tokenizer, model.granularity)  # type: ignore[arg-type]
    out = model(batch)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.embeddings is not None and out.embeddings.shape == (batch.batch_size, 64)


def test_flat_key_value_rows_start_with_the_profile_and_mark_every_event(tmp_path: Path) -> None:
    data = prepare_data(_flat_kv_config(), fixture_dir=tmp_path / "book")
    batch = data.batches[2]
    seq = data.test_sequences[0]
    row = batch.input_ids[0][batch.attention_mask[0].bool()].tolist()
    assert row[0] == USR
    assert row.count(EVT) == len(seq.events)
    assert PAD not in row


def test_flat_key_value_truncation_keeps_the_most_recent_whole_events(tmp_path: Path) -> None:
    data = prepare_data(_flat_kv_config(), fixture_dir=tmp_path / "book")
    tokenizer = data.tokenizer
    assert isinstance(tokenizer, FlatKeyValueTokenizer)
    seq = data.test_sequences[0]
    full = list(tokenizer.encode(seq).token_ids)
    short = tokenizer.encode_truncated(seq, max_length=40)
    assert len(short) <= 40
    first_event = short.index(EVT)
    assert full[-(len(short) - first_event) :] == short[first_event:]


def test_event_fields_feed_the_hierarchical_decoder(tmp_path: Path) -> None:
    config = _with(V2, features__tokenizer="event_fields", data__max_tokens=12)
    data = prepare_data(config, fixture_dir=tmp_path / "book")
    assert isinstance(data.tokenizer, EventFieldsTokenizer)
    batch = data.batches[0]
    assert isinstance(batch, KeyValueTimeBatch)
    assert batch.key_ids.shape[2] == len(FIELD_ORDER)
    model = build_model(config, data.tokenizer.vocab_size)
    out = model(batch)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_event_fields_values_are_the_flat_tokens() -> None:
    event = TransactionEvent(
        entity_id="e0",
        event_id="x0",
        event_timestamp=datetime(2024, 1, 1, 9, tzinfo=UTC),
        numeric={"amount": 42.0, "card": 1.0, "zip": 94105.0, "user": 7.0},
        categorical={"merchant_name": "ACME", "merchant_state": "CA", "mcc": "5411"},
    )
    tok = EventFieldsTokenizer().fit([])
    record = tok.encode(
        TransactionSequence(
            entity_id="e0",
            events=(event,),
            cutoff_time=event.event_timestamp,
            observation_window=None,
        )
    )
    flat = TransactionFieldTokenizer().event_tokens(event)
    assert [tok.vocabulary.id_to_token[v] for v in record.value_ids] == flat
    assert [tok.vocabulary.id_to_token[k] for k in record.key_ids] == [
        f"KEY_{f}" for f in FIELD_ORDER
    ]


def test_flat_stream_into_a_hierarchy_is_refused_by_rule_4(tmp_path: Path) -> None:
    config = _with(V2, features__tokenizer="flat_key_value")
    data = prepare_data(config, fixture_dir=tmp_path / "book")
    model = build_model(config, data.tokenizer.vocab_size)
    with pytest.raises(IncompatibleCompositionError, match="rule 4"):
        validate_tokenizer_layout(data.tokenizer, model.spec)  # type: ignore[union-attr]
