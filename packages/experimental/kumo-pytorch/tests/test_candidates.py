# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named caller-supplied candidates stay aligned through batching."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    BatchSpec,
    CandidateRequest,
    CandidateSetSpec,
    CandidateSource,
    CandidateTarget,
    DataLoaderSpec,
    DataSchema,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    ExternalTarget,
    Field,
    Fields,
    IdSpace,
    Input,
    Packed,
    Padded,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TargetValue,
    TimeWindow,
    ValidationError,
    Vocabulary,
)
from tests.helpers import Plans

SCHEMA = DataSchema(
    (
        TableSpec(
            "users",
            "user_id",
            {"user_id": DType("int64"), "country": DType("string")},
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "items",
            "item_id",
            {"item_id": DType("int64"), "category": DType("string")},
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "views",
            "view_id",
            {
                "view_id": DType("int64"),
                "item_id": DType("int64"),
                "at": DType("timestamp", unit="s", timezone="UTC"),
            },
            time_column="at",
        ),
    ),
    (RelationSpec("item_views", "items", "item_id", "views", "item_id"),),
)
TABLES = {
    "users": [{"user_id": 1, "country": "US"}, {"user_id": 2, "country": "CA"}],
    "items": [
        {"item_id": 10, "category": "book"},
        {"item_id": 11, "category": "game"},
        {"item_id": 12, "category": "book"},
    ],
    "views": [
        {"view_id": 1, "item_id": 10, "at": 80},
        {"view_id": 2, "item_id": 10, "at": 120},
        {"view_id": 3, "item_id": 11, "at": 70},
        {"view_id": 4, "item_id": 11, "at": 90},
    ],
}
PLANS = Plans(SCHEMA, TABLES)


def _candidate_spec() -> CandidateSetSpec:
    return CandidateSetSpec(
        entity=EntitySpec("items", entity_type="item"),
        values={
            "item_id": Field(column="item_id"),
            "category": Field(column="category"),
            "prior_views": Aggregate(
                Rows(("item_views",), TimeWindow("at")), "count", dtype=DType("int64")
            ),
        },
        targets={
            "clicked": ExternalTarget(dtype=DType("bool")),
            "future_views": Aggregate(
                Rows(("item_views",), TimeWindow("at", 0, 50)), "count", dtype=DType("int64")
            ),
        },
    )


def _contract(layout: Padded | Packed = Padded()) -> DataLoaderSpec:
    sample = SampleSpec(
        EntitySpec("users", entity_type="user"),
        entity_values={"country": Field(column="country")},
        candidate_sets={"items": _candidate_spec(), "alternatives": _candidate_spec()},
    )
    batch = BatchSpec(
        inputs={
            "user": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(field="country", encoding=Vocabulary(values=(("US", 2), ("CA", 3)))),
                    )
                ),
                layout=Separate(),
            ),
            "items": Input(
                CandidateSource("items"),
                Fields(
                    fields=(
                        Encode(
                            field="item_id", encoding=Vocabulary(values=((10, 2), (11, 3), (12, 4)))
                        ),
                        Encode(
                            field="category", encoding=Vocabulary(values=(("book", 2), ("game", 3)))
                        ),
                        Encode(field="prior_views", encoding=Raw()),
                    )
                ),
                layout,
            ),
            "alternatives": Input(
                CandidateSource("alternatives"),
                Fields(
                    fields=(
                        Encode(
                            field="category", encoding=Vocabulary(values=(("book", 2), ("game", 3)))
                        ),
                    )
                ),
                layout,
            ),
        },
        targets={
            "clicked": CandidateTarget("items", "clicked", Raw()),
            "future_views": CandidateTarget("items", "future_views", Raw()),
        },
    )
    return DataLoaderSpec(SCHEMA, sample, batch)


def _candidate(entity_id: int, clicked: bool) -> CandidateRequest:
    return CandidateRequest(entity_id, targets=(TargetValue("clicked", clicked),))


REQUESTS = (
    SampleRequest(
        "s1",
        1,
        100,
        candidates={
            "items": (_candidate(10, True), _candidate(11, False)),
            "alternatives": (CandidateRequest(12),),
        },
    ),
    SampleRequest("s2", 2, 100, candidates={"items": (_candidate(12, False),), "alternatives": ()}),
)


def test_padded_candidates_keep_metadata_values_labels_and_validity_aligned() -> None:
    contract = _contract()
    with PLANS.prepare(contract, REQUESTS) as plan:
        batch = next(plan.batches(REQUESTS, batch_size=2))
    items = batch.inputs["items"]
    assert "entity_ids" not in items
    assert batch.metadata["candidate_entity_ids"] == {
        "items": ((10, 11), (12,)),
        "alternatives": ((12,), ()),
    }
    assert items["valid"].tolist() == [[True, True], [True, False]]
    assert items["item_id"]["values"].tolist() == [[2, 3], [4, 0]]
    assert items["category"]["values"].tolist() == [[2, 3], [2, 0]]
    assert items["prior_views"]["values"].tolist() == [[1.0, 2.0], [0.0, 0.0]]
    assert batch.targets["clicked"]["values"].tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert batch.targets["clicked"]["candidate_valid"].tolist() == items["valid"].tolist()
    assert batch.targets["future_views"]["values"].tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_padded_candidate_target_uses_declared_pad_id() -> None:
    contract = _contract()
    contract = replace(
        contract,
        batch=replace(
            contract.batch,
            targets={
                **contract.batch.targets,
                "clicked": CandidateTarget("items", "clicked", Vocabulary(), id_space="labels"),
            },
            id_spaces={
                "labels": IdSpace(
                    specials={"pad": "pad", "unknown": "unknown", "missing": "missing"},
                    special_ids={"pad": 5, "unknown": 1, "missing": 2},
                )
            },
        ),
    )
    with PLANS.prepare(contract, REQUESTS) as plan:
        target = next(plan.batches(REQUESTS, batch_size=2)).targets["clicked"]
    assert target["values"][1, 1].item() == 5
    assert not target["candidate_valid"][1, 1]
    assert not target["valid"][1, 1]


def test_multiple_groups_and_empty_candidates_have_an_explicit_axis() -> None:
    contract = _contract()
    with PLANS.prepare(contract, REQUESTS) as plan:
        batch = next(plan.batches(REQUESTS, batch_size=2))
    alternatives = batch.inputs["alternatives"]
    assert "entity_ids" not in alternatives
    assert alternatives["valid"].tolist() == [[True], [False]]


def test_all_empty_candidate_groups_produce_empty_padded_axes() -> None:
    requests = (
        SampleRequest("s1", 1, 100, candidates={"items": ()}),
        SampleRequest("s2", 2, 100, candidates={"items": ()}),
    )
    contract = _contract()
    with PLANS.prepare(contract, requests) as plan:
        batch = next(plan.batches(requests, batch_size=2))
    assert batch.metadata["candidate_entity_ids"]["items"] == ((), ())
    assert "entity_ids" not in batch.inputs["items"]
    assert tuple(batch.inputs["items"]["valid"].shape) == (2, 0)
    assert tuple(batch.inputs["items"]["category"]["values"].shape) == (2, 0)
    assert tuple(batch.targets["clicked"]["values"].shape) == (2, 0)


def test_packed_candidates_use_offsets_instead_of_padding() -> None:
    contract = _contract(Packed())
    with PLANS.prepare(contract, REQUESTS) as plan:
        batch = next(plan.batches(REQUESTS, batch_size=2))
    assert "entity_ids" not in batch.inputs["items"]
    assert batch.metadata["candidate_entity_ids"]["items"] == ((10, 11), (12,))
    assert batch.inputs["items"]["offsets"].tolist() == [0, 2, 3]
    assert batch.targets["clicked"]["offsets"].tolist() == [0, 2, 3]


def test_unknown_candidate_groups_and_duplicate_candidates_are_refused() -> None:
    contract = _contract()
    unknown = SampleRequest("bad", 1, 100, candidates={"missing": (CandidateRequest(10),)})
    with pytest.raises(ValueError, match="unknown candidate set"):
        PLANS.prepare(contract, (unknown,))
    duplicate = SampleRequest(
        "bad", 1, 100, candidates={"items": (CandidateRequest(10), CandidateRequest(10))}
    )
    with pytest.raises(ValueError, match="candidate IDs must be unique"):
        PLANS.prepare(contract, (duplicate,))


def test_candidate_source_and_target_references_are_validated() -> None:
    contract = _contract()
    bad_input = DataLoaderSpec(
        SCHEMA,
        contract.sample,
        BatchSpec(
            inputs={
                "missing": Input(
                    CandidateSource("missing"),
                    Fields(fields=(Encode(field="category", encoding=Raw()),)),
                    Padded(),
                )
            }
        ),
    )
    with pytest.raises(ValidationError, match="unknown candidate set"):
        bad_input.validate()
    with pytest.raises(ValueError, match="incompatible"):
        BatchSpec(
            inputs=contract.batch.inputs,
            targets={"bad": CandidateTarget(input="user", source="clicked", encoding=Raw())},
        )


def test_candidate_contract_round_trips() -> None:
    contract = _contract()
    assert DataLoaderSpec.from_json(contract.to_json()) == contract
