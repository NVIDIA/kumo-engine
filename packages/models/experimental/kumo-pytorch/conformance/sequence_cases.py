# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated event sequences and their properties, on both output formats.

A case merges two event components into one input and caps a third, so the
model path pushes the cap down to DiskGraph. Sequences are read from real
`Padded` batches; the model format is decoded to the same per-sample events as
the inspectable one. Self-contained: no other model-path collector is shared.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import torch

from conformance.assertions import bound_tables, require
from conformance.generate import (
    GeneratedCase,
    Row,
    Scalar,
    Scale,
    Tables,
    case_digest,
    covering_rows,
    permuted,
)
from conformance.registry import SEQUENCE_DIMENSIONS, DimensionValue
from kumo.pytorch.api import (
    NON_TEMPORAL,
    BatchSpec,
    Constant,
    DataLoaderSpec,
    DataSchema,
    DiskGraphEngine,
    DType,
    Encode,
    EntitySpec,
    EventSource,
    EventSpec,
    EventTimeDelta,
    Field,
    Fields,
    Input,
    Padded,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    TableSpec,
    Task,
    TimeWindow,
    check,
    prepare,
)

OutputFormat = Literal["inspectable", "model"]
#: One delivered event: timestamp, source component, and (field, valid, value) per field.
Event = tuple[int, str, tuple[tuple[str, bool, Scalar], ...]]
#: Per sample, per input: its events in delivered order.
SequenceAnswer = dict[str, dict[str, tuple[Event, ...]]]

BASE = 1_700_000_000
SPAN = 1_200
GRID = (400, 700, 1_000)
TABLES = ("orders", "clicks")
#: Each event component and the table it reads; a constant field names the component.
COMPONENT_TABLES: Mapping[str, str] = {"orders": "orders", "clicks": "clicks", "recent": "orders"}
COMPONENT = "component"
#: Users, anchors, and mean random rows per user and table.
SEQUENCE_SCALES: Mapping[Scale, tuple[int, int, int]] = {"XS": (6, 8, 3), "S": (12, 16, 6)}
EVENT_LIMIT = 2
SECONDS_PER_UNIT: Mapping[str, int] = {"seconds": 1, "minutes": 60}
MERGED_MAX = 3
PUSHED_MAX = 2
#: Every anchor has more eligible orders than any cap, so each cap selects.
ELIGIBLE = 4


def sequence_contract(dims: Mapping[str, DimensionValue]) -> DataLoaderSpec:
    schema = DataSchema(
        tables=(
            TableSpec("users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL),
            *(
                TableSpec(
                    table,
                    f"{table[:-1]}_id",
                    {
                        f"{table[:-1]}_id": DType("int64"),
                        "user_id": DType("int64"),
                        "amount": DType("float64"),
                        "at": DType("timestamp", unit="s", timezone="UTC"),
                    },
                    time_column="at",
                )
                for table in TABLES
            ),
        ),
        relations=tuple(
            RelationSpec(f"user_{table}", "users", "user_id", table, "user_id") for table in TABLES
        ),
    )
    unit: Literal["seconds", "minutes"] = (
        "minutes" if dims["contract.event_delta_unit"] == "minutes" else "seconds"
    )
    order: Literal["ascending", "descending"] = (
        "descending" if dims["contract.event_order"] == "descending" else "ascending"
    )

    def events(
        component: str,
        limit: int | None = None,
        order: Literal["ascending", "descending"] = "ascending",
    ) -> EventSpec:
        table = COMPONENT_TABLES[component]
        fields: dict[str, Field | EventTimeDelta | Constant] = {
            "id": Field(f"{table[:-1]}_id"),
            "amount": Field("amount"),
            "delta": EventTimeDelta(unit),
            COMPONENT: Constant(tuple(COMPONENT_TABLES).index(component), DType("int64")),
        }
        rows = Rows((f"user_{table}",), TimeWindow("at", None, 0))
        return EventSpec(rows, fields, limit, order)

    keep = str(dims["contract.event_keep"])
    columns = Fields(tuple(Encode(name, Raw()) for name in ("id", "amount", "delta", COMPONENT)))
    merged_max = MERGED_MAX if dims["contract.event_max"] == "limited" else None
    return DataLoaderSpec(
        schema,
        SampleSpec(
            EntitySpec("users", entity_type="user"),
            events={
                "orders": events(
                    "orders",
                    limit=EVENT_LIMIT if dims["contract.event_limit"] == "limited" else None,
                    order=order,
                ),
                "clicks": events("clicks"),
                "recent": events("recent"),
            },
        ),
        BatchSpec(
            inputs={
                "merged": Input(
                    EventSource(("orders", "clicks"), max_events=merged_max, keep=keep),
                    columns,
                    Padded(),
                ),
                "pushed": Input(
                    EventSource(("recent",), max_events=PUSHED_MAX, keep=keep), columns, Padded()
                ),
            }
        ),
    )


# -- Rows --------------------------------------------------------------------


def generate_sequence_case(
    seed: int, index: int, dims: Mapping[str, DimensionValue], scale: Scale
) -> GeneratedCase:
    rng = random.Random(f"sequence:{seed}:{index}")
    count, anchor_count, per_user = SEQUENCE_SCALES[scale]
    users = list(range(1, count + 1))
    tied = dims["time.ties"] == "tied"
    grid = tuple(BASE + offset for offset in GRID)
    anchors: list[tuple[int, int]] = []
    for position in range(anchor_count):
        user = users[position % count]
        cutoff = grid[1]
        if dims["time.cutoffs"] == "mixed":
            cutoff = grid[(position // count + user) % len(grid)]
            cutoff += rng.randrange(0, 50) if position >= count else 0
        anchors.append((user, cutoff))

    rows: dict[str, list[Row]] = {table: [] for table in TABLES}
    taken: set[tuple[int, int]] = set()

    def add(table: str, user: int, time: int) -> None:
        # Untied cases date every row of a user apart, across both components.
        if tied or (user, time) not in taken:
            taken.add((user, time))
            amount = None if rng.random() < 0.2 else float(rng.choice((1, 2, 3, 5, 8)))
            key = f"{table[:-1]}_id"
            rows[table].append(
                {key: len(rows[table]) + 1, "user_id": user, "amount": amount, "at": time}
            )

    for user in users:
        for table in TABLES:
            for _ in range(rng.randint(0, 2 * per_user)):
                add(table, user, rng.randrange(BASE, BASE + SPAN))
    for position, (user, cutoff) in enumerate(anchors):
        for table, time in (("orders", cutoff), ("clicks", cutoff + 1), ("orders", cutoff - 1)):
            add(table, user, time)
        if tied and position < 2:
            # Ties within a component at the limit, and across the two components.
            for table in ("orders", "orders", "orders", "clicks"):
                add(table, user, cutoff - 2)
        while (
            sum(r["user_id"] == user and int(str(r["at"])) < cutoff for r in rows["orders"])
            < ELIGIBLE
        ):
            add("orders", user, rng.randrange(BASE, cutoff - 1))
    for table in TABLES:
        rows[table][0]["amount"] = 1.0  # never an all-null column
    tables: Tables = {"users": [{"user_id": user} for user in users], **rows}
    requests = tuple(
        SampleRequest(f"a{position}", entity_id=user, as_of_time=cutoff)
        for position, (user, cutoff) in enumerate(anchors)
    )
    return GeneratedCase(
        f"q{seed}-c{index}", seed, scale, dict(dims), sequence_contract(dims), tables, requests
    )


def sequence_batch(seed: int, engine: str, property_name: str) -> list[GeneratedCase]:
    scale: Scale = "XS" if engine == "in_memory" else "S"
    rows = covering_rows(seed, engine, SEQUENCE_DIMENSIONS, property_name)
    return [generate_sequence_case(seed, index, row, scale) for index, row in rows]


# -- Running a case ----------------------------------------------------------


def sequence_answers(
    contract: DataLoaderSpec, tables: Tables, requests: Sequence[SampleRequest], engine: str
) -> dict[OutputFormat, SequenceAnswer]:
    """Sequences per output format the engine serves (DiskGraph: both), from real batches."""

    clocks = {
        (table, row[f"{table[:-1]}_id"]): int(str(row["at"]))
        for table in TABLES
        for row in tables[table]
    }
    with bound_tables(contract, tables, engine) as (dataset, bound):
        issues = check(contract, dataset, bound)
        require(not issues, f"{engine} refused a generated sequence contract: {issues}")
        formats: tuple[OutputFormat, ...] = (
            ("inspectable", "model") if isinstance(bound, DiskGraphEngine) else ("inspectable",)
        )
        with prepare(contract, dataset, bound) as plan:
            task = plan.materialize(requests)
            return {output: collect_sequences(task, requests, output, clocks) for output in formats}


def collect_sequences(
    task: Task,
    requests: Sequence[SampleRequest],
    output: OutputFormat,
    clocks: Mapping[tuple[str, Scalar], int],
) -> SequenceAnswer:
    """Each sample's delivered events; ``clocks`` dates a row by its table and ID."""

    result: SequenceAnswer = {}
    for batch in task.batches(requests, batch_size=8, output_format=output):
        for position, sample_id in enumerate(batch.metadata["sample_ids"]):
            require(str(sample_id) not in result, f"{output}: duplicate sample {sample_id}")
            result[str(sample_id)] = {
                name: _events(value, position, clocks)
                for name, value in sorted(batch.inputs.items())
            }
    expected = {str(request.sample_id) for request in requests}
    require(result.keys() == expected, f"{output}: samples {sorted(result)} != {sorted(expected)}")
    return result


def _events(
    padded: Mapping[str, object], position: int, clocks: Mapping[tuple[str, Scalar], int]
) -> tuple[Event, ...]:
    """One sample's delivered events from a Padded input, in slot order.

    An event's time is its source row's clock, found by the component's table and the
    delivered ID, so it doesn't depend on the engine's own time arithmetic.
    """

    fields = {
        name: (_row(value["valid"], position), _row(value["values"], position))
        for name, value in sorted(padded.items())
        if isinstance(value, Mapping)
    }
    _, components = fields.pop(COMPONENT)
    ids_valid, ids = fields["id"]
    events = []
    for slot, present in enumerate(_row(padded["valid"], position)):
        if not present:
            continue
        source = tuple(COMPONENT_TABLES)[int(str(components[slot]))]
        require(bool(ids_valid[slot]), f"{source} event in slot {slot} has no ID")
        events.append(
            (
                clocks[(COMPONENT_TABLES[source], ids[slot])],
                source,
                tuple(
                    (name, bool(valid[slot]), values[slot] if valid[slot] else None)
                    for name, (valid, values) in fields.items()
                ),
            )
        )
    return tuple(events)


def _row(value: object, position: int) -> list[Scalar]:
    row = value[position] if isinstance(value, torch.Tensor | list | tuple) else None
    if isinstance(row, torch.Tensor):
        return row.tolist()
    require(isinstance(row, list | tuple), f"unexpected Padded layout {type(value).__name__}")
    return list(row)  # type: ignore[arg-type]


#: One base run per (binding, engine, case) serves every property; permutation reruns.
_RUNS: dict[tuple[Callable[..., object], str, str], dict[OutputFormat, SequenceAnswer]] = {}


def _base(case: GeneratedCase, engine: str) -> dict[OutputFormat, SequenceAnswer]:
    key = (bound_tables, engine, case_digest(case))
    if key not in _RUNS:
        _RUNS[key] = sequence_answers(case.contract, case.tables, case.requests, engine)
    return _RUNS[key]


def _same_sequences(before: SequenceAnswer, after: SequenceAnswer, label: str) -> None:
    differences = [
        f"{key}: {before.get(key)} != {after.get(key)}"
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    ]
    require(not differences, f"{label} changed sequences:\n  " + "\n  ".join(differences[:3]))


# -- Properties --------------------------------------------------------------


def sequence_differential(case: GeneratedCase, engine: str, position: int) -> None:
    """DiskGraph delivers the reference's sequences, on each output format."""

    reference = _base(case, "in_memory")["inspectable"]
    for output, answer in _base(case, engine).items():
        _same_sequences(reference, answer, f"{engine} {output} vs reference")


def with_future_events(case: GeneratedCase, rng: random.Random) -> tuple[Tables, set[str]]:
    """Rows of both components dated after one cutoff per entity; the anchors they guard."""

    tables = {name: [dict(row) for row in rows] for name, rows in case.tables.items()}
    cutoffs: dict[int, list[int]] = {}
    for request in case.requests:
        cutoffs.setdefault(int(request.entity_id), []).append(int(request.as_of_time))
    thresholds = {entity: rng.choice(times) for entity, times in sorted(cutoffs.items())}
    for entity, threshold in thresholds.items():
        for table in TABLES:
            for time in (threshold + 1, threshold + rng.randint(2, 100)):
                key = f"{table[:-1]}_id"
                row: Row = {key: len(tables[table]) + 1, "user_id": entity}
                tables[table].append(row | {"amount": 4.0, "at": time})
    guarded = {
        str(r.sample_id) for r in case.requests if int(r.as_of_time) <= thresholds[int(r.entity_id)]
    }
    return tables, guarded


def sequence_point_in_time(case: GeneratedCase, engine: str, position: int) -> None:
    """No delivered event is dated after its cutoff, and future rows change nothing.

    Each event's EventTimeDelta is exactly (cutoff - timestamp) in the declared unit.
    """

    base = _base(case, engine)
    cutoffs = {str(request.sample_id): int(request.as_of_time) for request in case.requests}
    unit = SECONDS_PER_UNIT[str(case.dimensions["contract.event_delta_unit"])]
    for output, answer in base.items():
        late = [
            f"{sample_id}: {name} event at {event[0]} > cutoff {cutoffs[sample_id]}"
            for sample_id, inputs in sorted(answer.items())
            for name, events in inputs.items()
            for event in events
            if event[0] > cutoffs[sample_id]
        ]
        require(not late, f"{output}: events after the cutoff:\n  " + "\n  ".join(late[:6]))
        wrong = [
            f"{sample_id}: {name} event at {event[0]} has delta {delta}, "
            f"not {(cutoffs[sample_id] - event[0]) / unit}"
            for sample_id, inputs in sorted(answer.items())
            for name, events in inputs.items()
            for event in events
            if (delta := dict((f, v) for f, _, v in event[2]).get("delta"))
            != (cutoffs[sample_id] - event[0]) / unit
        ]
        require(not wrong, f"{output}: wrong time deltas:\n  " + "\n  ".join(wrong[:6]))
    tables, guarded = with_future_events(case, random.Random(f"{case.id}:future"))
    require(bool(guarded), "no anchor is guarded by the future rows")
    after = sequence_answers(case.contract, tables, case.requests, engine)
    for output, answer in base.items():
        _same_sequences(
            {key: answer[key] for key in guarded},
            {key: after[output][key] for key in guarded},
            f"{output}: future rows",
        )


def sequence_permutation(case: GeneratedCase, engine: str, position: int) -> None:
    """A second run, and row and request order, deliver the same sequences (determinism)."""

    base = _base(case, engine)
    again = sequence_answers(case.contract, case.tables, case.requests, engine)
    shuffled = permuted(case, random.Random(f"{case.id}:permute"))
    moved = sequence_answers(shuffled.contract, shuffled.tables, shuffled.requests, engine)
    for output, answer in base.items():
        _same_sequences(answer, again[output], f"{output}: a second run")
        _same_sequences(answer, moved[output], f"{output}: row and request order")


SEQUENCE_CHECKS = {
    "sequence_differential": sequence_differential,
    "sequence_point_in_time": sequence_point_in_time,
    "sequence_permutation": sequence_permutation,
}
