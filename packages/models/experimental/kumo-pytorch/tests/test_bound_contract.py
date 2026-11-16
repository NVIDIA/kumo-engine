# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""H2 — a fine-tune reuses a pretrained plan's fitted state instead of re-fitting.

The failure this prevents is silent: refit on different data and the same token
string maps to a different ID, so a reused embedding table is being indexed by a
vocabulary it was not trained with. Nothing raises; the model is just wrong.
"""

from __future__ import annotations

import pytest

from kumo.pytorch.api import (
    NON_TEMPORAL,
    Batching,
    BatchSpec,
    Buckets,
    Columns,
    DataLoaderSpec,
    DataSchema,
    DType,
    EntitySource,
    EntitySpec,
    Field,
    IdSpace,
    InMemoryEngine,
    Input,
    Padded,
    RelationSpec,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    Tokens,
    UseEncoding,
    Vocabulary,
    ZScore,
    prepare_batch_spec,
)
from kumo.pytorch.contract.batch import Encode, Fields
from tests.helpers import Plans as _Plans
from tests.helpers import in_memory_dataset

USERS = TableSpec(
    name="users",
    primary_key="user_id",
    columns={"user_id": DType("int64"), "country": DType("string"), "balance": DType("float64")},
    time_column=NON_TEMPORAL,
)
EVENTS = TableSpec(
    name="events",
    primary_key="event_id",
    columns={
        "event_id": DType("int64"),
        "user_id": DType("int64"),
        "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
    },
    time_column="occurred_at",
)
OWNS = RelationSpec(
    name="user_events",
    left_table="users",
    left_key="user_id",
    right_table="events",
    right_key="user_id",
)
SCHEMA = DataSchema(tables=(USERS, EVENTS), relations=(OWNS,))
PRETRAIN_ROWS = {
    "users": [
        {"user_id": 1, "country": "US", "balance": 10.0},
        {"user_id": 2, "country": "US", "balance": 20.0},
        {"user_id": 3, "country": "GB", "balance": 30.0},
        {"user_id": 4, "country": "DE", "balance": 40.0},
        {"user_id": 5, "country": "FR", "balance": 50.0},
    ],
    "events": [],
}
FINETUNE_ROWS = {
    "users": [
        {"user_id": 1, "country": "FR", "balance": 1000.0},
        {"user_id": 2, "country": "FR", "balance": 2000.0},
        {"user_id": 3, "country": "FR", "balance": 3000.0},
        {"user_id": 4, "country": "DE", "balance": 4000.0},
        {"user_id": 5, "country": "US", "balance": 5000.0},
    ],
    "events": [],
}
CONTRACT = DataLoaderSpec(
    schema=SCHEMA,
    sample=SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={"country": Field(column="country"), "balance": Field(column="balance")},
    ),
    batch=BatchSpec(
        inputs={
            "values": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(field="country", encoding=Vocabulary()),
                        Encode(field="balance", encoding=ZScore()),
                    ),
                    row=Columns(),
                ),
                layout=Separate(),
            )
        },
        batching=Batching(max_samples=8),
    ),
)
REQUESTS = tuple(
    SampleRequest(f"s{index}", entity_id=index, as_of_time=1000) for index in range(1, 6)
)


def _prepare(rows: dict[str, list[dict[str, object]]], contract: DataLoaderSpec):
    return _Plans(SCHEMA, rows).prepare(contract, REQUESTS)


def _country_ids(plan) -> dict[str, int]:
    return dict(plan.output(name="values").field(name="country").encoder.entries)


class TestBoundContract:
    def test_refitting_on_new_data_moves_token_ids(self) -> None:
        pretrained = _prepare(PRETRAIN_ROWS, CONTRACT)
        refit = _prepare(FINETUNE_ROWS, CONTRACT)
        assert _country_ids(pretrained) != _country_ids(refit)

    def test_binding_pins_ids_across_corpora(self) -> None:
        pretrained = _prepare(PRETRAIN_ROWS, CONTRACT)
        bound = pretrained.bound_contract()
        finetuned = _prepare(FINETUNE_ROWS, bound)
        assert _country_ids(finetuned) == _country_ids(pretrained)

    def test_binding_pins_numerical_state_too(self) -> None:
        pretrained = _prepare(PRETRAIN_ROWS, CONTRACT)
        finetuned = _prepare(FINETUNE_ROWS, pretrained.bound_contract())
        before = pretrained.output(name="values").field(name="balance").encoder.state_dict()
        after = finetuned.output(name="values").field(name="balance").encoder.state_dict()
        assert before == after

    def test_a_bound_contract_carries_its_state_in_the_spec(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, CONTRACT).bound_contract()
        representation = bound.batch.inputs["values"].representation
        fields = {use.field: use.encoding for use in representation.fields}
        assert fields["country"].values
        assert fields["balance"].mean is not None

    def test_binding_changes_the_fingerprint(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, CONTRACT).bound_contract()
        assert bound.fingerprint != CONTRACT.fingerprint

    def test_a_bound_contract_round_trips_through_json(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, CONTRACT).bound_contract()
        assert DataLoaderSpec.from_json(bound.to_json()) == bound

    def test_binding_is_idempotent(self) -> None:
        once = _prepare(PRETRAIN_ROWS, CONTRACT).bound_contract()
        twice = _prepare(FINETUNE_ROWS, once).bound_contract()
        assert twice == once

    def test_a_bound_contract_still_validates(self) -> None:
        _prepare(PRETRAIN_ROWS, CONTRACT).bound_contract().validate()

    def test_binding_needs_a_complete_contract(self) -> None:
        source = InMemoryEngine().open(in_memory_dataset(PRETRAIN_ROWS), SCHEMA)
        built = source.build(CONTRACT.sample, REQUESTS)
        plan = prepare_batch_spec(CONTRACT.sample, CONTRACT.batch, built)
        with pytest.raises(ValueError, match="complete DataLoaderSpec"):
            plan.bound_contract()


SHARED_CONTRACT = DataLoaderSpec(
    schema=SCHEMA,
    sample=SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={"country": Field(column="country"), "balance": Field(column="balance")},
    ),
    batch=BatchSpec(
        encodings={"places": Vocabulary()},
        inputs={
            "values": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(field="country", encoding=UseEncoding("places")),
                        Encode(field="balance", encoding=ZScore()),
                    ),
                    row=Columns(),
                ),
                layout=Separate(),
            )
        },
        batching=Batching(max_samples=8),
    ),
)


class TestSharedEncodings:
    """A shared encoding is a claim that fields share one ID space.

    Inlining fitted values over a `UseEncoding` would silently split that space
    and orphan the shared entry, so binding keeps the reference and pins the
    shared encoding instead.
    """

    def test_binding_a_shared_scalar_encoding_round_trips(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, SHARED_CONTRACT).bound_contract()
        bound.validate()
        assert DataLoaderSpec.from_json(bound.to_json()) == bound

    def test_the_shared_reference_survives_binding(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, SHARED_CONTRACT).bound_contract()
        fields = {
            use.field: use.encoding for use in bound.batch.inputs["values"].representation.fields
        }
        assert fields["country"] == UseEncoding("places")
        assert fields["balance"].mean is not None

    def test_the_shared_encoding_itself_is_pinned(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, SHARED_CONTRACT).bound_contract()
        assert bound.batch.encodings["places"].values

    def test_a_shared_vocabulary_is_stable_across_corpora(self) -> None:
        pretrained = _prepare(PRETRAIN_ROWS, SHARED_CONTRACT)
        finetuned = _prepare(FINETUNE_ROWS, pretrained.bound_contract())
        before = pretrained.encoders["places"].state_dict()
        after = finetuned.encoders["places"].state_dict()
        assert before == after

    def test_refitting_a_shared_vocabulary_would_have_moved_it(self) -> None:
        pretrained = _prepare(PRETRAIN_ROWS, SHARED_CONTRACT)
        refit = _prepare(FINETUNE_ROWS, SHARED_CONTRACT)
        assert pretrained.encoders["places"].entries != refit.encoders["places"].entries


TOKEN_CONTRACT = DataLoaderSpec(
    schema=SCHEMA,
    sample=SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={"country": Field(column="country"), "balance": Field(column="balance")},
    ),
    batch=BatchSpec(
        encodings={"countries": Vocabulary(), "balances": Buckets(bins=4)},
        id_spaces={
            "tokens": IdSpace(
                specials={
                    "pad": "[PAD]",
                    "unknown": "[UNK]",
                    "missing": "[MISSING]",
                    "mask": "[MASK]",
                }
            )
        },
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(
                            field="country",
                            encoding=UseEncoding(name="countries"),
                            id_space="tokens",
                        ),
                        Encode(
                            field="balance",
                            encoding=UseEncoding(name="balances"),
                            id_space="tokens",
                        ),
                    ),
                    row=Tokens(max_tokens=8),
                ),
                layout=Padded(),
            )
        },
        batching=Batching(max_samples=8),
    ),
)


class TestSharedFieldTokens:
    """The PRAGMA shape: entity and event fields sharing one token ID space."""

    def test_binding_a_shared_field_tokenizer_round_trips(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, TOKEN_CONTRACT).bound_contract()
        bound.validate()
        assert DataLoaderSpec.from_json(bound.to_json()) == bound

    def test_each_fitted_token_encoder_is_pinned_explicitly(self) -> None:
        bound = _prepare(PRETRAIN_ROWS, TOKEN_CONTRACT).bound_contract()
        assert bound.batch.encodings["countries"].values
        assert bound.batch.encodings["balances"].boundaries is not None

    def test_token_ids_are_stable_across_corpora(self) -> None:
        pretrained = _prepare(PRETRAIN_ROWS, TOKEN_CONTRACT)
        finetuned = _prepare(FINETUNE_ROWS, pretrained.bound_contract())
        before = pretrained.encoders["countries"].entries
        after = finetuned.encoders["countries"].entries
        assert before == after
        assert pretrained.spaces["tokens"].state_dict() == finetuned.spaces["tokens"].state_dict()

    def test_refitting_token_ids_would_have_moved_them(self) -> None:
        pretrained = _prepare(PRETRAIN_ROWS, TOKEN_CONTRACT)
        refit = _prepare(FINETUNE_ROWS, TOKEN_CONTRACT)
        assert pretrained.encoders["countries"].entries != refit.encoders["countries"].entries


def test_a_fine_tune_of_a_bound_contract_fits_its_own_targets() -> None:
    from dataclasses import replace

    from kumo.pytorch.api import ExternalTarget, LabelTarget, TargetValue

    labelled = replace(
        CONTRACT,
        sample=replace(CONTRACT.sample, targets={"tier": ExternalTarget(dtype=DType("string"))}),
        batch=replace(CONTRACT.batch, targets={"tier": LabelTarget("tier", Vocabulary())}),
    )

    def requests(*tiers: str) -> tuple[SampleRequest, ...]:
        return tuple(
            SampleRequest(f"s{i}", i, 1000, targets=(TargetValue("tier", tier),))
            for i, tier in enumerate(tiers, start=1)
        )

    pretrained = _Plans(SCHEMA, PRETRAIN_ROWS).prepare(labelled, requests("gold", "silver"))
    finetuned = _Plans(SCHEMA, FINETUNE_ROWS).prepare(
        pretrained.bound_contract(), requests("bronze", "platinum")
    )
    tiers = finetuned.target("tier").field(name="value").encoder.entries
    assert {value for value, _ in tiers} == {"bronze", "platinum"}
