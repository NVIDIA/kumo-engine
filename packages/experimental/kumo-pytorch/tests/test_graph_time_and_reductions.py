# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal safety of graph traversal, uniform sampling, and the reduction vocabulary."""

from __future__ import annotations

import pytest

from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    Batching,
    BatchSpec,
    Columns,
    DataLoaderSpec,
    DataSchema,
    DiskGraphEngine,
    EntitySource,
    EntitySpec,
    Graph,
    GraphLayer,
    GraphSample,
    Input,
    NeighborhoodSource,
    NeighborhoodSpec,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TimeWindow,
    ValidationError,
)
from kumo.pytorch.contract.batch import Encode, Fields
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import ALL_REDUCTIONS, Reduce
from kumo.pytorch.engines.in_memory.graph import GraphRoot, InMemoryGraphSampler
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder

USERS = TableSpec(
    name="users",
    primary_key="user_id",
    columns={"user_id": DType("int64")},
    time_column=NON_TEMPORAL,
)
EVENTS_WITH_CLOCK = TableSpec(
    name="events",
    primary_key="event_id",
    columns={
        "event_id": DType("int64"),
        "user_id": DType("int64"),
        "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
        "amount": DType("float64"),
    },
    time_column="occurred_at",
)
EVENTS_NO_CLOCK = TableSpec(
    name="events",
    primary_key="event_id",
    columns={
        "event_id": DType("int64"),
        "user_id": DType("int64"),
        "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
        "amount": DType("float64"),
    },
    time_column=NON_TEMPORAL,
)
OWNS = RelationSpec(
    name="user_events",
    left_table="users",
    left_key="user_id",
    right_table="events",
    right_key="user_id",
)
SCHEMA = DataSchema(tables=(USERS, EVENTS_WITH_CLOCK), relations=(OWNS,))
SCHEMA_NO_CLOCK = DataSchema(tables=(USERS, EVENTS_NO_CLOCK), relations=(OWNS,))
CUTOFF = 100
ROWS = {
    "users": [{"user_id": 1}],
    "events": [
        {"event_id": 1, "user_id": 1, "occurred_at": 10, "amount": 10.0},
        {"event_id": 2, "user_id": 1, "occurred_at": 40, "amount": 30.0},
        {"event_id": 3, "user_id": 1, "occurred_at": 900, "amount": 999.0},
    ],
}


def _neighborhood(**kwargs: object) -> NeighborhoodSpec:
    return NeighborhoodSpec(
        layers=(GraphLayer((GraphSample("user_events", fanout=10, **kwargs),)),)
    )


def _sampled_event_ids(neighborhood: NeighborhoodSpec, schema: DataSchema = SCHEMA) -> set[int]:
    sampler = InMemoryGraphSampler(schema, ROWS)
    node_rows, _owners, _edges = sampler._collect(
        "users",
        neighborhood,
        (GraphRoot(sample_id="s1", entity_id=1, cutoff=CUTOFF),),
        seed=0,
        epoch=0,
    )
    return {row["event_id"] for row in node_rows.get("events", ())}


def _contract(neighborhood: NeighborhoodSpec, schema: DataSchema = SCHEMA) -> DataLoaderSpec:
    return DataLoaderSpec(
        schema=schema,
        sample=SampleSpec(
            entity=EntitySpec("users", entity_type="user"), neighborhoods={"history": neighborhood}
        ),
        batch=BatchSpec(
            inputs={
                "neighborhood": Input(
                    source=NeighborhoodSource("history"),
                    representation=Graph(node_fields={}),
                    layout=None,
                )
            },
            batching=Batching(max_samples=2),
        ),
    )


class TestGraphTemporalSafety:
    """D2 — the default traversal must not read the anchor's future."""

    def test_default_traversal_inherits_the_declared_clock(self) -> None:
        assert _sampled_event_ids(_neighborhood()) == {1, 2}

    def test_explicit_time_column_still_wins(self) -> None:
        assert _sampled_event_ids(_neighborhood(time_column="occurred_at")) == {1, 2}

    def test_reading_the_future_requires_saying_so(self) -> None:
        assert _sampled_event_ids(_neighborhood(temporal="static")) == {1, 2, 3}

    def test_being_unfiltered_is_never_something_that_just_happens(self) -> None:
        with pytest.raises(ValidationError) as error:
            _contract(_neighborhood(), SCHEMA_NO_CLOCK).validate()
        issue = next(item for item in error.value.issues if item.path.endswith(".temporal"))
        assert "does not say it is intentionally non-temporal" in issue.message
        assert 'temporal="static"' in (issue.suggestion or "")

    def test_declaring_static_makes_it_expressible_again(self) -> None:
        static = _neighborhood(strategy="uniform", temporal="static")
        _contract(static, SCHEMA_NO_CLOCK).validate()
        assert _sampled_event_ids(static, SCHEMA_NO_CLOCK) == {1, 2, 3}

    def test_recent_without_any_clock_is_refused(self) -> None:
        with pytest.raises(ValidationError) as error:
            _contract(_neighborhood(strategy="recent"), SCHEMA_NO_CLOCK).validate()
        issue = next(item for item in error.value.issues if "strategy" in item.path)
        assert "no time column" in issue.message
        assert "TableSpec" in (issue.suggestion or "")

    def test_recent_inherits_a_declared_clock_without_repeating_it(self) -> None:
        _contract(_neighborhood(strategy="recent")).validate()
        assert _sampled_event_ids(_neighborhood(strategy="recent")) == {1, 2}

    def test_ordering_and_filtering_are_separate(self) -> None:
        neighborhood = _neighborhood(strategy="recent", temporal="static")
        _contract(neighborhood).validate()
        assert _sampled_event_ids(neighborhood) == {1, 2, 3}

    def test_recent_with_no_clock_at_all_is_refused_not_raised(self) -> None:
        with pytest.raises(ValidationError) as error:
            _contract(
                _neighborhood(strategy="recent", temporal="static"), SCHEMA_NO_CLOCK
            ).validate()
        assert any(item.path.endswith(".strategy") for item in error.value.issues)

    def test_an_unknown_temporal_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="temporal must be"):
            GraphSample("user_events", fanout=2, temporal="sometimes")

    def test_a_declared_time_column_must_exist(self) -> None:
        with pytest.raises(ValueError, match="not a declared column"):
            TableSpec("events", "event_id", {"event_id": DType("int64")}, time_column="missing")


class TestUniformSampling:
    """Uniform graph sampling is a valid contract, refused by DiskGraph before data is read."""

    def test_the_contract_is_valid(self) -> None:
        _contract(_neighborhood(strategy="uniform")).validate()

    def test_diskgraph_refuses_it_with_its_code(self) -> None:
        contract = _contract(_neighborhood(strategy="uniform"))
        issues = DiskGraphEngine(index="/i").check_contract(contract.resolve())
        issue = next(item for item in issues if item.path.endswith(".strategy"))
        assert issue.code == "DG-UNIFORM-GRAPH"
        assert "uniform" in issue.message
        assert issue.suggestion


class TestReductions:
    """C3 — std, variance and recency complete the reduction vocabulary."""

    def _single(self, operation: str) -> float | None:
        """Reduce a window containing exactly one eligible row."""
        spec = SampleSpec(
            entity=EntitySpec("users", entity_type="user"),
            entity_values={
                "value": Aggregate(
                    source=Rows(("user_events",), TimeWindow("occurred_at", None, -70)),
                    operation=operation,
                    column="amount",
                    dtype=DType("float64"),
                )
            },
        )
        engine = InMemorySampleBuilder(SCHEMA, ROWS)
        built = engine.build(spec, (SampleRequest("s1", entity_id=1, as_of_time=CUTOFF),))[0]
        return next((v.value for v in built.history.entity.values if v.key == "value"), None)

    def _value(self, operation: str) -> float | None:
        column = "occurred_at" if operation == "recency" else "amount"
        spec = SampleSpec(
            entity=EntitySpec("users", entity_type="user"),
            entity_values={
                "value": Aggregate(
                    source=Rows(("user_events",), TimeWindow("occurred_at", None, 0)),
                    operation=operation,
                    column=column,
                    dtype=DType("float64"),
                )
            },
        )
        engine = InMemorySampleBuilder(SCHEMA, ROWS)
        built = engine.build(spec, (SampleRequest("s1", entity_id=1, as_of_time=CUTOFF),))[0]
        return next(value.value for value in built.history.entity.values if value.key == "value")

    def test_population_and_sample_are_both_expressible(self) -> None:
        assert self._value("var_population") == pytest.approx(100.0)
        assert self._value("var_sample") == pytest.approx(200.0)
        assert self._value("std_population") == pytest.approx(10.0)
        assert self._value("std_sample") == pytest.approx(14.142135623730951)

    def test_the_two_conventions_differ_by_the_bessel_factor(self) -> None:
        ratio = self._value("std_sample") / self._value("std_population")
        assert ratio == pytest.approx(2**0.5)

    def test_a_single_observation_has_no_sample_spread(self) -> None:
        assert self._single("std_sample") is None
        assert self._single("var_sample") is None
        assert self._single("std_population") == pytest.approx(0.0)

    def test_recency_is_seconds_from_the_latest_row_to_the_cutoff(self) -> None:
        assert self._value("recency") == pytest.approx(60.0)

    def test_empty_sum_is_zero_but_all_null_sum_is_missing(self) -> None:
        assert (
            InMemorySampleBuilder._reduce([], Reduce("sum", "amount", dtype=DType("float64")), 100)
            == 0.0
        )
        assert (
            InMemorySampleBuilder._reduce(
                [{"amount": None}], Reduce("sum", "amount", dtype=DType("float64")), 100
            )
            is None
        )

    def test_new_reductions_are_rejected_without_a_column(self) -> None:
        with pytest.raises(ValueError, match="requires a column"):
            Aggregate(source=Rows(("user_events",)), operation="std_sample", dtype=DType("float64"))

    def test_an_ambiguous_bare_spelling_is_not_accepted(self) -> None:
        for spelling in ("std", "variance"):
            with pytest.raises(ValueError, match="unsupported reduction"):
                Aggregate(
                    source=Rows(("user_events",)),
                    operation=spelling,
                    column="amount",
                    dtype=DType("float64"),
                )


class TestRecencySemantics:
    """C3 — `recency` is historical-only, with stated null and unit behaviour."""

    def _aggregate(self, role: str) -> DataLoaderSpec:
        spec = Aggregate(
            source=Rows(("user_events",), TimeWindow("occurred_at", None, 0)),
            operation="recency",
            column="occurred_at",
            dtype=DType("float64"),
        )
        entity = EntitySpec("users", entity_type="user")
        if role == "target":
            sample = SampleSpec(
                entity=entity,
                targets={
                    "when": Aggregate(
                        source=Rows(("user_events",), TimeWindow("occurred_at", 0, 86400)),
                        operation="recency",
                        column="occurred_at",
                        dtype=DType("float64"),
                    )
                },
                entity_values={"any": spec},
            )
        else:
            sample = SampleSpec(entity=entity, entity_values={"any": spec})
        return DataLoaderSpec(
            schema=SCHEMA,
            sample=sample,
            batch=BatchSpec(
                inputs={
                    "values": Input(
                        source=EntitySource(),
                        representation=Fields(
                            fields=(Encode(field="any", encoding=Raw()),), row=Columns()
                        ),
                        layout=Separate(),
                    )
                },
                batching=Batching(max_samples=2),
            ),
        )

    def test_recency_is_valid_over_history(self) -> None:
        self._aggregate("context").validate()

    def test_recency_over_a_future_window_is_refused(self) -> None:
        with pytest.raises(ValidationError) as error:
            self._aggregate("target").validate()
        issue = next(item for item in error.value.issues if item.path.endswith(".operation"))
        assert "historical-only" in issue.message
        assert issue.suggestion

    def test_an_empty_window_yields_no_recency_value(self) -> None:
        spec = SampleSpec(
            entity=EntitySpec("users", entity_type="user"),
            entity_values={
                "gap": Aggregate(
                    source=Rows(("user_events",), TimeWindow("occurred_at", -5, -4)),
                    operation="recency",
                    column="occurred_at",
                    dtype=DType("float64"),
                )
            },
        )
        engine = InMemorySampleBuilder(SCHEMA, ROWS)
        built = engine.build(spec, (SampleRequest("s1", entity_id=1, as_of_time=CUTOFF),))[0]
        assert not [v for v in built.history.entity.values if v.key == "gap"]


class TestExternalStandardDeviationParity:
    """Pin the external reference values, not just internal consistency.

    These numbers match DuckDB's `stddev_samp` on the same inputs. If
    `std_sample` ever drifts toward the population convention, this fails rather
    than a downstream feature table silently disagreeing.
    """

    SESSION_MINUTES = (174, 163)
    DUCKDB_STDDEV_SAMP = 7.7781745930520225
    POPULATION_STDDEV = 5.5

    def _std(self, operation: str) -> float | None:
        schema = DataSchema(
            tables=(
                USERS,
                TableSpec(
                    "sessions",
                    "session_id",
                    {
                        "session_id": DType("int64"),
                        "user_id": DType("int64"),
                        "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                        "session_minutes": DType("int64"),
                    },
                    time_column="occurred_at",
                ),
            ),
            relations=(RelationSpec("user_sessions", "users", "user_id", "sessions", "user_id"),),
        )
        rows = {
            "users": [{"user_id": 1}],
            "sessions": [
                {
                    "session_id": index,
                    "user_id": 1,
                    "occurred_at": 10 * index,
                    "session_minutes": minutes,
                }
                for index, minutes in enumerate(self.SESSION_MINUTES, 1)
            ],
        }
        spec = SampleSpec(
            entity=EntitySpec("users", entity_type="user"),
            entity_values={
                "value": Aggregate(
                    source=Rows(("user_sessions",), TimeWindow("occurred_at", None, 0)),
                    operation=operation,
                    column="session_minutes",
                    dtype=DType("float64"),
                )
            },
        )
        built = InMemorySampleBuilder(schema, rows).build(
            spec, (SampleRequest("s1", entity_id=1, as_of_time=CUTOFF),)
        )[0]
        return next(v.value for v in built.history.entity.values if v.key == "value")

    def test_std_sample_matches_the_recorded_duckdb_value(self) -> None:
        assert self._std("std_sample") == pytest.approx(self.DUCKDB_STDDEV_SAMP)

    def test_std_population_reproduces_the_value_that_disagreed(self) -> None:
        assert self._std("std_population") == pytest.approx(self.POPULATION_STDDEV)

    def test_the_two_are_not_interchangeable_at_this_scale(self) -> None:
        assert self._std("std_sample") != pytest.approx(self._std("std_population"))


class TestReducerDispatch:
    """Every reduction is dispatched by name, and unknown names raise.

    The reducer previously implemented the standard deviation in its
    fall-through branch, so any operation that reached it unrecognised was
    silently answered with a standard deviation instead of failing.
    """

    @staticmethod
    def _unchecked(operation: str) -> Reduce:
        spec = object.__new__(Reduce)
        object.__setattr__(spec, "operation", operation)
        object.__setattr__(spec, "column", "x")
        return spec

    def test_an_unknown_operation_raises_rather_than_guessing(self) -> None:
        rows = [{"x": 10}, {"x": 30}]
        with pytest.raises(ValueError, match="unsupported reduction operation 'median'"):
            InMemorySampleBuilder._reduce(rows, self._unchecked("median"), 0)

    def test_the_construction_guard_is_still_the_first_defence(self) -> None:
        with pytest.raises(ValueError, match="unsupported reduction operation"):
            Reduce("median", "x", dtype=DType("float64"))

    def test_every_declared_reduction_is_reachable(self) -> None:
        rows = [{"x": 10}, {"x": 30}]
        for operation in sorted(ALL_REDUCTIONS):
            column = None if operation == "count" else "x"
            InMemorySampleBuilder._reduce(
                rows, Reduce(operation, column, dtype=DType("float64")), 100
            )
