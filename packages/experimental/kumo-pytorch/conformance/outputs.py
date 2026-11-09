# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readers and case variants for targets, batches, anchors and fitted state.

Pure helpers for the functional-area properties in `conformance.properties`:
they decode outputs into comparable answers and build the variant inputs; the
properties judge them.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from conformance.fixtures import values as fixture_values
from conformance.generate import (
    ACCOUNT_EVENTS,
    AHEAD,
    CLOCK,
    TARGET_FIELDS,
    TARGET_SPAN,
    TIMESTAMP,
    GeneratedCase,
    Row,
    Scalar,
    Tables,
    case_contract,
    entity_cutoffs,
    event_row,
)
from kumo.pytorch.api import (
    Batch,
    Batching,
    BatchSpec,
    Buckets,
    Columns,
    DataLoaderSpec,
    Encode,
    EntitySource,
    EntitySpec,
    EventSource,
    EventSpec,
    EventTimeDelta,
    Field,
    FieldNames,
    Fields,
    IdSpace,
    Input,
    Nested,
    Packed,
    Padded,
    Raw,
    Rows,
    SampleRequest,
    Separate,
    TimeWindow,
    Tokens,
    UseEncoding,
    Vocabulary,
    ZScore,
)

Values = dict[str, Scalar]
#: One event of a decoded sequence input: offset from the cutoff, event ID, amount.
#: The offset is the negated EventTimeDelta the history input carries as ``AGE``.
Token = tuple[int | None, Scalar, Scalar]
TARGET = "target:"
#: Planted rows relative to one cutoff per entity: both sides of every target edge.
LABEL_OFFSETS = (
    -1,
    0,
    1,
    AHEAD[0],
    AHEAD[0] + 1,
    AHEAD[1],
    AHEAD[1] + 1,
    TARGET_SPAN,
    TARGET_SPAN + 1,
    10_000,
)


# -- Decoding ---------------------------------------------------------------


def flat_targets(targets: Mapping[str, object], cutoff: int) -> Values:
    """Preview targets as flat values; a row target's time is kept relative to the cutoff."""

    flat: Values = {}
    for name, value in fixture_values(dict(targets)).items():
        if isinstance(value, Mapping):
            flat[f"{TARGET}{name}.offset"] = int(value["timestamp"]) - cutoff
            for field in value["fields"]:
                if field["value"] is not None:
                    flat[f"{TARGET}{name}.{field['key']}"] = field["value"]
        elif value is not None:
            flat[f"{TARGET}{name}"] = value
    return flat


def entity_only(values: Mapping[str, Scalar]) -> Values:
    return {name: value for name, value in values.items() if not name.startswith(TARGET)}


def _column(field: Mapping[str, torch.Tensor], index: int) -> Scalar:
    return field["values"][index].item() if bool(field["valid"][index]) else None


def decode_batches(batches: Iterable[Batch]) -> list[tuple[str, Values]]:
    """Entity columns and targets of every row, in delivery order, as raw numbers."""

    rows: list[tuple[str, Values]] = []
    for batch in batches:
        for index, sample_id in enumerate(batch.metadata["sample_ids"]):
            values: Values = {
                name: _column(field, index)
                for name, field in batch.inputs.get("entity", {}).items()
            }
            for name, target in batch.targets.items():
                if "values" in target:
                    values[f"{TARGET}{name}"] = _column(target, index)
                else:
                    for field_name, field in target.items():
                        values[f"{TARGET}{name}.{field_name}"] = _column(field, index)
            rows.append((str(sample_id), values))
    return rows


def sequence_tokens(batch: Batch, name: str) -> list[tuple[str, tuple[Token, ...]]]:
    """Each row's events of a Padded or Packed Columns input, in layout order."""

    data = batch.inputs[name]
    ids = [str(sample_id) for sample_id in batch.metadata["sample_ids"]]

    def value(field: Mapping[str, torch.Tensor], position: tuple[int, ...]) -> Scalar:
        return field["values"][position].item() if bool(field["valid"][position]) else None

    def token(position: tuple[int, ...]) -> Token:
        age = value(data[AGE], position)
        return (
            None if age is None else -int(age),
            value(data["event_id"], position),
            value(data["amount"], position),
        )

    tokens: list[tuple[str, tuple[Token, ...]]] = []
    if "offsets" in data:
        bounds = data["offsets"].tolist()
        for row, sample_id in enumerate(ids):
            tokens.append(
                (sample_id, tuple(token((i,)) for i in range(bounds[row], bounds[row + 1])))
            )
        return tokens
    for row, sample_id in enumerate(ids):
        columns = range(data["valid"].shape[1])
        tokens.append(
            (sample_id, tuple(token((row, i)) for i in columns if bool(data["valid"][row, i])))
        )
    return tokens


def batch_cost(batch: Batch, name: str) -> int:
    """Token slots one batch materializes for ``name``, padding included."""

    data = batch.inputs[name]
    if "offsets" in data:
        return int(data["offsets"][-1])
    rows, width = data["valid"].shape
    return int(rows) * int(width)


# -- Label windows ----------------------------------------------------------


@dataclass(frozen=True)
class Planted:
    """Rows planted around one cutoff per entity, and how often a path reaches each."""

    tables: Tables
    #: entity -> (absolute time, event ID) of each planted row
    rows: Mapping[int, tuple[tuple[int, int], ...]]
    multiplicity: Mapping[int, int]


def plant_label_rows(case: GeneratedCase, rng: random.Random) -> Planted:
    tables = {name: [dict(row) for row in rows] for name, rows in case.tables.items()}
    events, shape = tables["events"], str(case.dimensions["schema_shape.path"])
    accounts = tables.get("accounts", [])
    planted: dict[int, tuple[tuple[int, int], ...]] = {}
    multiplicity: dict[int, int] = {}
    for entity, cutoffs in sorted(entity_cutoffs(case.requests).items()):
        owned = [int(row["account_id"]) for row in accounts if row["user_id"] == entity]  # type: ignore[arg-type]
        via = owned[:1] if shape in ACCOUNT_EVENTS else []
        threshold = rng.choice(cutoffs)
        rows = []
        for offset in LABEL_OFFSETS:
            row = event_row(rng, len(events) + 1, entity, threshold + offset, via, "none")
            row.update(amount=1.0, category="a")
            events.append(row)
            rows.append((threshold + offset, int(row["event_id"])))  # type: ignore[arg-type]
            if "links" in tables:
                link = len(tables["links"]) + 1
                tables["links"].append(
                    {"link_id": link, "user_id": entity, "event_id": row["event_id"]}
                )
        planted[entity] = tuple(rows)
        repeated = shape == "depth3_static" and not case.dimensions["contract.distinct"]
        multiplicity[entity] = len(owned) if repeated else 1
    return Planted(tables, planted, multiplicity)


def expected_targets(
    before: Mapping[str, Scalar], rows: Sequence[tuple[int, int]], cutoff: int, repeat: int
) -> Values:
    """Targets after planting ``rows``, by the (start, stop] rule; a tied row's time is
    judged, not which tied row is chosen.
    """

    expected: Values = {}
    offsets = {time - cutoff: event_id for time, event_id in rows}
    for name, (start, stop) in (("next", (0, TARGET_SPAN)), ("ahead", AHEAD)):
        inside = [offset for offset in offsets if start < offset <= stop]
        added = repeat * len(inside)
        count = before.get(f"{TARGET}count_{name}")
        expected[f"{TARGET}count_{name}"] = int(count or 0) + added
        total = before.get(f"{TARGET}sum_{name}")
        expected[f"{TARGET}sum_{name}"] = float(total or 0) + added if added else total
    inside = [offset for offset in offsets if 0 < offset <= TARGET_SPAN]
    for selection, pick in (("first", min), ("last", max)):
        key = f"{TARGET}{selection}_next"
        current = before.get(f"{key}.offset")
        names = (f"{key}.offset", *(f"{key}.{field}" for field in TARGET_FIELDS))
        if inside:
            chosen = pick(inside)
            if current is not None and chosen == current:
                expected[names[0]] = chosen  # which tied row is engine-defined; its time is not
                continue
            if current is None or pick(chosen, current) == chosen:
                expected.update({names[0]: chosen, names[1]: offsets[chosen], names[2]: 1.0})
                continue
        expected.update({name: before.get(name) for name in names})
    return expected


# -- Contract variants ------------------------------------------------------


def model_contract(case: GeneratedCase) -> DataLoaderSpec:
    """The case's contract over its shape's own schema; Raw keeps each declared dtype on
    both output formats.
    """

    return case_contract(str(case.dimensions["schema_shape.path"]), case.dimensions)


HISTORY = "history"
AGE = "age"


def packing_contract(
    case: GeneratedCase,
    layout: Padded | Packed,
    *,
    max_samples: int,
    max_tokens: int | None,
    keep: str,
    max_events: int,
) -> DataLoaderSpec:
    """The model contract plus a bounded event-history input that drives the batching."""

    base = model_contract(case)
    events = EventSpec(
        Rows(case.path, TimeWindow(CLOCK, None, 0), distinct=True),
        {**{name: Field(name) for name in TARGET_FIELDS}, AGE: EventTimeDelta("seconds")},
    )
    history = Input(
        EventSource((HISTORY,), max_events=max_events, keep=keep),
        Fields(tuple(Encode(name, Raw()) for name in (*TARGET_FIELDS, AGE))),
        layout,
    )
    return DataLoaderSpec(
        base.schema,
        replace(base.sample, events={HISTORY: events}),
        BatchSpec(
            inputs={**base.batch.inputs, HISTORY: history},
            targets=base.batch.targets,
            batching=Batching(
                max_samples=max_samples,
                max_tokens=max_tokens,
                length_input=HISTORY,
                bucket_by_length=max_tokens is not None,
            ),
        ),
    )


def packing_requests(case: GeneratedCase) -> tuple[SampleRequest, ...]:
    """The case's requests, plus anchors whose history is empty, one row and two rows
    long: for a few entities, cutoffs just before and at their first two event times.
    """

    owners = {row["account_id"]: row["user_id"] for row in case.tables.get("accounts", [])}
    times: dict[int, set[int]] = {}
    for row in case.tables["events"]:
        owner = row["user_id"] if row.get("user_id") is not None else owners[row["account_id"]]
        times.setdefault(int(str(owner)), set()).add(int(str(row[CLOCK])))
    extra = []
    for user in sorted(times)[:4]:
        first, *later = sorted(times[user])
        cutoffs = {"empty": first - 1, "one": first, "two": later[0] if later else None}
        extra += [
            SampleRequest(f"p{user}-{label}", user, cutoff)
            for label, cutoff in cutoffs.items()
            if cutoff is not None
        ]
    return (*case.requests, *extra)


def fitting_contract(case: GeneratedCase) -> DataLoaderSpec:
    """Entity columns whose encoders are fitted: a vocabulary, a z-score and quantiles."""

    base = case.contract
    columns = Fields(
        (
            Encode("mode", Vocabulary()),
            Encode("sum", ZScore()),
            Encode("count", Buckets(bins=4)),
            Encode("max", Raw()),
        ),
        row=Columns(),
    )
    return replace(
        base,
        batch=replace(
            base.batch,
            inputs={"entity": Input(base.batch.inputs["entity"].source, columns, Separate())},
        ),
    )


#: Entity values and history fields a typed-fitting contract tokenizes.
TYPED_ENTITY = ("mode", "sum", "count", "max")
#: event_id is never null: an event with no value token can't be packed.
TYPED_EVENTS = ("event_id", "amount", "category")
#: Fields a typed-fitting contract encodes with the shared category vocabulary.
TYPED_CATEGORIES = ("mode", "category")
TOKENS = "tokens"
TOKEN_ROLES = ("pad", "unknown", "missing")


def _typed_tokens(names: Sequence[str], max_tokens: int) -> Fields:
    return Fields(
        tuple(
            Encode(
                name,
                UseEncoding("categories")
                if name in TYPED_CATEGORIES
                else Buckets(bins=4, separate_zero=True),
                id_space=TOKENS,
            )
            for name in names
        ),
        row=Tokens(max_tokens=max_tokens),
        field_names="fields",
        field_space=TOKENS,
    )


def typed_fitting_contract(case: GeneratedCase) -> DataLoaderSpec:
    """Field tokens over entity values and an event history, with no targets: DiskGraph
    fits them straight from typed columns (``build_model``), the reference from records.
    """

    base = case.contract
    fields = {name: Field(name) for name in TYPED_EVENTS}
    events = EventSpec(Rows(case.path, TimeWindow(CLOCK, None, 0), distinct=True), fields)
    return DataLoaderSpec(
        base.schema,
        replace(base.sample, targets={}, events={HISTORY: events}),
        BatchSpec(
            inputs={
                "entity": Input(EntitySource(), _typed_tokens(TYPED_ENTITY, 16), Padded()),
                HISTORY: Input(
                    EventSource((HISTORY,)),
                    _typed_tokens(TYPED_EVENTS, 64),
                    Nested(Padded(), Padded()),
                ),
            },
            encodings={"categories": Vocabulary()},
            id_spaces={TOKENS: IdSpace({role: f"[{role.upper()}]" for role in TOKEN_ROLES})},
            field_names={"fields": FieldNames()},
            batching=Batching(max_samples=8),
        ),
    )


#: Fitted-state keys whose values map entries to encoded IDs; compared as entries.
_ID_MAPS = ("vocabulary", "vocab")
Structure = None | bool | int | float | str | list["Structure"] | dict[str, "Structure"]


def fitted_structure(state: object) -> Structure:
    """Fitted state without encoded IDs: vocabularies become their sorted entries and
    serialized tokenizer state is parsed.
    """

    if isinstance(state, Mapping):
        return {
            str(key): sorted(map(str, value))
            if key in _ID_MAPS and isinstance(value, Mapping)
            else fitted_structure(value)
            for key, value in state.items()
        }
    if isinstance(state, list | tuple):
        return [fitted_structure(item) for item in state]
    if isinstance(state, str) and state.startswith("{"):
        try:
            return fitted_structure(json.loads(state))
        except json.JSONDecodeError:
            return state
    if state is None or isinstance(state, bool | int | float | str):
        return state
    return repr(state)


def state_differences(
    reference: object, actual: object, tolerance: float, path: str = "state"
) -> list[str]:
    """Where two fitted states differ: structure and types exactly, floats within
    ``tolerance``.
    """

    if isinstance(reference, Mapping) and isinstance(actual, Mapping):
        if reference.keys() != actual.keys():
            return [f"{path}: keys {sorted(reference)} != {sorted(actual)}"]
        return [
            difference
            for key in sorted(reference)
            for difference in state_differences(
                reference[key], actual[key], tolerance, f"{path}.{key}"
            )
        ]
    if isinstance(reference, list | tuple) and isinstance(actual, list | tuple):
        if len(reference) != len(actual):
            return [f"{path}: {len(reference)} items != {len(actual)}"]
        return [
            difference
            for index, (left, right) in enumerate(zip(reference, actual, strict=True))
            for difference in state_differences(left, right, tolerance, f"{path}[{index}]")
        ]
    if type(reference) is not type(actual):
        return [f"{path}: {reference!r} ({type(reference).__name__}) != {actual!r}"]
    if isinstance(reference, float) and isinstance(actual, float):
        close = math.isclose(reference, actual, rel_tol=tolerance, abs_tol=tolerance)
        return [] if close else [f"{path}: {reference} != {actual}"]
    return [] if reference == actual else [f"{path}: {reference!r} != {actual!r}"]


# -- Anchors and roots ------------------------------------------------------

SPLITS = ("train", "valid", None, "test", "train")
#: Split selections a task is iterated with: one name, unions, and the null split.
SELECTIONS: tuple[str | frozenset[str | None] | None, ...] = (
    "train",
    frozenset({"train", "test"}),
    frozenset({None}),
    frozenset({"valid", None}),
    None,
)


def split_anchor_rows(case: GeneratedCase) -> list[Row]:
    return [
        {
            "sample_id": str(request.sample_id),
            "entity_id": int(request.entity_id),
            "as_of_time": int(request.as_of_time),
            "split": SPLITS[position % len(SPLITS)],
        }
        for position, request in enumerate(case.requests)
    ]


def selected(rows: Sequence[Row], selection: str | frozenset[str | None] | None) -> list[str]:
    """Sample IDs a selection holds, in anchor-table order."""

    names = {selection} if isinstance(selection, str) else selection
    return [str(row["sample_id"]) for row in rows if names is None or row["split"] in names]


JOINED = "joined_at"


def availability_case(case: GeneratedCase) -> tuple[DataLoaderSpec, Tables, int]:
    """The case with ``users.joined_at`` as the entity's ``available_from``; all users
    join at one time before every anchor, which is returned.
    """

    joined = min(int(request.as_of_time) for request in case.requests) - 100
    schema = case.contract.schema
    tables = tuple(
        replace(table, columns={**table.columns, JOINED: TIMESTAMP})
        if table.name == "users"
        else table
        for table in schema.tables
    )
    entity = EntitySpec("users", entity_type="user", available_from=JOINED)
    contract = replace(
        case.contract,
        schema=replace(schema, tables=tables),
        sample=replace(case.contract.sample, entity=entity),
    )
    rows = {name: [dict(row) for row in rows] for name, rows in case.tables.items()}
    for row in rows["users"]:
        row[JOINED] = joined
    return contract, rows, joined
