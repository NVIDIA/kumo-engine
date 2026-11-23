# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta

import pytest
from tests.conftest import make_event

from tfm.contracts.errors import IncompatibleCompositionError, MissingLabelError, SchemaError
from tfm.contracts.registry import advertised_capabilities, get_capability
from tfm.data.labels.horizon import event_label_at_cutoff
from tfm.data.sequence_builders.entity_sequences import events_at_or_before
from tfm.data.transforms.ordering import order_events
from tfm.data.validation.schema import validate_event


def test_invalid_schema_rejected() -> None:
    with pytest.raises(SchemaError):
        validate_event(make_event("ent", "", datetime(2024, 1, 1, tzinfo=UTC)))


def test_tied_timestamps_order_by_event_id() -> None:
    ts = datetime(2024, 1, 1, 12, tzinfo=UTC)
    a = make_event("e", "b-id", ts)
    b = make_event("e", "a-id", ts)
    ordered = order_events([a, b])
    assert [ev.event_id for ev in ordered] == ["a-id", "b-id"]


def test_cutoff_excludes_future_events() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    events = [
        make_event("e", "1", ts0),
        make_event("e", "2", ts0 + timedelta(days=2)),
        make_event("e", "3", ts0 + timedelta(days=5)),
    ]
    kept = events_at_or_before(events, ts0 + timedelta(days=2))
    assert [ev.event_id for ev in kept] == ["1", "2"]


def test_delayed_labels_excluded() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    event = make_event("e", "1", ts, fraud=1.0, delay=timedelta(days=7))
    cutoff = ts + timedelta(days=1)
    with pytest.raises(MissingLabelError):
        event_label_at_cutoff(event, "fraud", cutoff)
    available = event_label_at_cutoff(event, "fraud", ts + timedelta(days=7))
    assert available.value == 1.0


def test_registry_advertises_m1_components() -> None:
    caps = advertised_capabilities()
    assert get_capability("decoder").causal is True
    assert get_capability("next_event").name in caps
    assert get_capability("transaction_field_tokenizer").support_tier == "reference"
    assert caps["eager_attention"].support_tier == "reference"


def test_incompatible_composition_fails_loudly() -> None:
    from tfm.api.validation import validate_composition
    from tfm.attention.eager import EagerAttention
    from tfm.compositions.flat import Flat
    from tfm.layers.decoder import Decoder
    from tfm.pooling.last_token import LastTokenPool
    from tfm.position.none import NoPE

    with pytest.raises(IncompatibleCompositionError):
        validate_composition(
            Flat(Decoder(mask="causal")),
            NoPE(),
            EagerAttention(),
            object(),
            LastTokenPool(),
            None,
        )


def test_incompatible_tokenizer_layout_fails_loudly() -> None:
    from tfm.api.validation import validate_tokenizer_layout
    from tfm.compositions.flat import Flat
    from tfm.layers.decoder import Decoder

    class _KvtStub:
        name = "key_value_time"
        layout = "key_value_time"

    with pytest.raises(IncompatibleCompositionError):
        validate_tokenizer_layout(_KvtStub(), Flat(Decoder(mask="causal", vocab_size=32)))
