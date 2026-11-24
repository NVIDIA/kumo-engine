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
    neighborhood = _contract(neighborhood, schema).resolve().sample.neighborhoods["history"]
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


class TestGraphSampling:
    def test_timed_default_is_recent_and_filters_the_anchor_future(self):
        contract = _contract(_neighborhood()).resolve()
        traversal = contract.sample.neighborhoods["history"].layers[0].samples[0]
        assert traversal.strategy == "recent"
        assert _sampled_event_ids(_neighborhood()) == {1, 2}

    def test_clockless_default_is_uniform(self):
        contract = _contract(_neighborhood(), SCHEMA_NO_CLOCK).resolve()
        contract.validate()
        traversal = contract.sample.neighborhoods["history"].layers[0].samples[0]
        assert traversal.strategy == "uniform"
        assert _sampled_event_ids(_neighborhood(), SCHEMA_NO_CLOCK) == {1, 2, 3}

    def test_in_memory_rejects_unresolved_strategy(self):
        sampler = InMemoryGraphSampler(SCHEMA, ROWS)
        with pytest.raises(ValueError, match="resolve GraphSample.strategy"):
            sampler._collect(
                "users",
                _neighborhood(),
                (GraphRoot(sample_id="s1", entity_id=1, cutoff=CUTOFF),),
                seed=0,
                epoch=0,
            )

    def test_serving_reads_reject_unresolved_strategy(self):
        from kumo.pytorch.serving_rows import _BatchNames, _collect

        contract = _contract(_neighborhood())
        with pytest.raises(ValueError, match="resolve GraphSample.strategy"):
            _collect(contract, contract.sample, _BatchNames(contract.sample, contract.batch))

    @pytest.mark.parametrize("schema, expected", ((SCHEMA, "recent"), (SCHEMA_NO_CLOCK, "uniform")))
    def test_default_uses_the_traversal_destination_in_reverse(self, schema, expected):
        schema = DataSchema(
            tables=schema.tables,
            relations=(RelationSpec("user_events", "events", "user_id", "users", "user_id"),),
        )
        contract = _contract(_neighborhood(direction="reverse"), schema).resolve()
        assert contract.sample.neighborhoods["history"].layers[0].samples[0].strategy == expected

    @pytest.mark.parametrize("schema", (SCHEMA, SCHEMA_NO_CLOCK))
    def test_explicit_uniform_ignores_real_time(self, schema):
        neighborhood = _neighborhood(strategy="uniform")
        _contract(neighborhood, schema).validate()
        assert _sampled_event_ids(neighborhood, schema) == {1, 2, 3}

    def test_explicit_recent_requires_a_declared_clock(self):
        with pytest.raises(ValidationError) as error:
            _contract(_neighborhood(strategy="recent"), SCHEMA_NO_CLOCK).validate()
        assert any(issue.path.endswith(".strategy") for issue in error.value.issues)

    def test_clock_override_does_not_make_a_clockless_table_temporal(self):
        with pytest.raises(ValidationError):
            _contract(
                _neighborhood(strategy="recent", time_column="occurred_at"), SCHEMA_NO_CLOCK
            ).validate()

    def test_explicit_time_column_still_works(self):
        assert _sampled_event_ids(_neighborhood(time_column="occurred_at")) == {1, 2}

    def test_resolved_strategy_is_serialized(self):
        contract = _contract(_neighborhood(), SCHEMA_NO_CLOCK)
        loaded = DataLoaderSpec.from_json(contract.to_json())
        assert loaded.sample.neighborhoods["history"].layers[0].samples[0].strategy == "uniform"
        assert loaded.fingerprint == contract.fingerprint

    @pytest.mark.parametrize("strategy", (None, "recent", "uniform"))
    def test_serving_requests_every_candidate_for_uniform(self, strategy):
        from kumo.pytorch.api import ServingRowsSpec

        rows = ServingRowsSpec.from_contract(_contract(_neighborhood(strategy=strategy)))
        events = rows.tables["events"]
        if strategy == "uniform":
            assert events.time_column is None
            assert events.limit is None
        else:
            assert events.time_column == "occurred_at"
            assert events.limit.count == 10

    def test_removed_temporal_argument_is_not_silently_ignored(self):
        with pytest.raises(TypeError, match="temporal"):
            GraphSample("user_events", fanout=2, temporal="static")

    def test_diskgraph_accepts_uniform_before_reading_data(self):
        assert not DiskGraphEngine().check_contract(
            _contract(_neighborhood(strategy="uniform")).resolve()
        )


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
