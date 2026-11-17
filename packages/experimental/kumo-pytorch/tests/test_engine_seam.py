# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seam: a dataset says where bytes are, an engine says who reads them.

Each test states which of the two it is about, because the point of the design
is that they vary independently and that collapsing them is the failure mode.
"""

from __future__ import annotations

import pytest

from kumo.pytorch.api import (
    NON_TEMPORAL,
    Batching,
    BatchSpec,
    Columns,
    Dataset,
    DiskGraphEngine,
    DType,
    Engine,
    EntitySource,
    EntitySpec,
    Field,
    InMemoryEngine,
    InMemoryRows,
    Input,
    Parquet,
    RelationSpec,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    ValidationError,
    Vocabulary,
    available_engines,
    check,
    engine_from_name,
    prepare,
    register_engine,
)
from kumo.pytorch.assembly import fit_plan
from kumo.pytorch.contract.batch import Encode, Fields
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue

USERS = TableSpec(
    "users",
    "user_id",
    {"user_id": DType("int64"), "country": DType("string")},
    time_column=NON_TEMPORAL,
)
EVENTS = TableSpec(
    "events",
    "event_id",
    {
        "event_id": DType("int64"),
        "user_id": DType("int64"),
        "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
    },
    time_column="occurred_at",
)
SCHEMA = DataSchema(
    tables=(USERS, EVENTS), relations=(RelationSpec("ue", "users", "user_id", "events", "user_id"),)
)
CONTRACT = DataLoaderSpec(
    schema=SCHEMA,
    sample=SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={"country": Field(column="country")},
    ),
    batch=BatchSpec(
        inputs={
            "values": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(Encode(field="country", encoding=Vocabulary()),), row=Columns()
                ),
                layout=Separate(),
            )
        },
        batching=Batching(max_samples=4),
    ),
)
PARQUET = Dataset(
    {"users": Parquet("/warehouse/users"), "events": Parquet("/lake/events/*.parquet")}
)


class _Stub:
    """An engine with no checks of its own, so only the shared checks apply."""

    name = "stub"

    def open(self, dataset, schema):
        raise AssertionError("unreachable")


class _RefusesEverything:
    """An engine whose own check refuses every contract, before open."""

    name = "refuses_everything"

    def check_contract(self, contract):
        return (ValidationIssue("sample", "this engine runs nothing", "use another contract"),)

    def open(self, dataset, schema):
        raise AssertionError("unreachable: refused during validation")

    def materialize_anchors(self, table):
        raise AssertionError("unreachable: refused during validation")


class TestDatasetDescribesDataNotEngines:
    """The correction that shapes everything else: a dataset names bytes."""

    def test_the_same_dataset_is_checked_by_each_engine(self) -> None:
        assert check(CONTRACT, PARQUET, InMemoryEngine())
        assert check(CONTRACT, PARQUET, DiskGraphEngine(index="/i")) == ()

    def test_a_source_renames_physical_columns_to_logical_ones(self) -> None:
        source = Parquet("/lake/events", columns={"ts": "occurred_at"})
        assert source.columns == {"ts": "occurred_at"}

    def test_colliding_renames_are_refused(self) -> None:
        with pytest.raises(ValueError, match="renames collide"):
            Parquet("/lake/events", columns={"ts": "occurred_at", "t2": "occurred_at"})

    def test_source_kinds_are_visible_without_an_engine(self) -> None:
        assert PARQUET.kinds == frozenset({"parquet"})
        assert Dataset({"users": InMemoryRows([{"user_id": 1}])}).kinds == frozenset({"in_memory"})

    def test_in_memory_rows_refuse_serialization_rather_than_pretending(self) -> None:
        with pytest.raises(TypeError, match="they are the data"):
            Dataset({"users": InMemoryRows([{"user_id": 1}])}).to_dict()

    def test_a_parquet_dataset_serializes(self) -> None:
        assert PARQUET.to_dict()["users"] == {
            "kind": "parquet",
            "path": "/warehouse/users",
            "columns": {},
        }


class TestBindingIsCheckedBeforeAnyDataIsRead:
    def test_an_unbound_table_is_refused_with_the_name(self) -> None:
        partial = Dataset({"users": Parquet("/warehouse/users")})
        issues = check(CONTRACT, partial, _Stub())
        assert any("events" in issue.message for issue in issues)

    def test_a_binding_the_schema_does_not_declare_is_refused(self) -> None:
        extra = Dataset({**PARQUET.tables, "orders": Parquet("/warehouse/orders")})
        issues = check(CONTRACT, extra, _Stub())
        assert any("orders" in issue.message for issue in issues)

    def test_an_engine_that_cannot_read_the_format_is_refused(self) -> None:
        issues = check(CONTRACT, IN_MEMORY, DiskGraphEngine(index="/i"))
        issue = next(item for item in issues if "does not read" in item.message)
        assert "in_memory" in issue.message
        assert issue.code == "DG-SOURCE-KIND"
        assert "Parquet" in (issue.suggestion or "")

    def test_an_engine_without_its_own_checks_over_a_matching_dataset_passes(self) -> None:
        assert check(CONTRACT, PARQUET, _Stub()) == ()


class TestEnginesRefuseEarly:
    """An engine refuses what it can't run before open."""

    def test_an_engine_check_refuses_with_a_suggestion(self) -> None:
        issues = check(CONTRACT, PARQUET, _RefusesEverything())
        assert issues
        assert all(issue.suggestion for issue in issues)

    def test_diskgraph_lists_what_it_does_not_support_read_only(self) -> None:
        unsupported = DiskGraphEngine(index="/i").unsupported
        assert unsupported is DiskGraphEngine.unsupported
        assert all(code == entry.code for code, entry in unsupported.items())
        with pytest.raises(TypeError):
            unsupported["DG-NEW"] = unsupported["DG-ROW-FILTER"]  # type: ignore[index]

    def test_diskgraph_accepts_fitted_direct_entity_values(self) -> None:
        assert check(CONTRACT, PARQUET, DiskGraphEngine(index="/i")) == ()

    def test_prepare_refuses_rather_than_raising_not_implemented(self) -> None:
        with pytest.raises(ValidationError):
            prepare(CONTRACT, PARQUET, _RefusesEverything())

    def test_the_in_memory_engine_declares_nothing_unsupported(self) -> None:
        assert not hasattr(InMemoryEngine(), "unsupported")
        assert check(CONTRACT, IN_MEMORY, InMemoryEngine()) == ()


class TestSelection:
    def test_engines_register_themselves(self) -> None:
        assert set(available_engines()) >= {"diskgraph", "in_memory"}

    def test_a_name_resolves_to_the_same_class_as_construction(self) -> None:
        assert isinstance(engine_from_name("diskgraph", index="/i"), DiskGraphEngine)

    def test_an_unknown_name_lists_what_is_available(self) -> None:
        with pytest.raises(ValueError, match="unknown engine 'cugraph'"):
            engine_from_name("cugraph")

    def test_registering_a_second_class_under_one_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_engine("in_memory", DiskGraphEngine)

    def test_re_registering_the_same_class_is_harmless(self) -> None:
        register_engine("in_memory", InMemoryEngine)

    def test_engines_satisfy_the_engine_protocol(self) -> None:
        for engine in (InMemoryEngine(), DiskGraphEngine(index="/i"), _RefusesEverything()):
            assert isinstance(engine, Engine)


ROWS = [{"user_id": index, "country": value} for index, value in enumerate(["US", "GB", "US"], 1)]
IN_MEMORY = Dataset({"users": InMemoryRows(ROWS), "events": InMemoryRows([])})
REQUESTS = tuple(SampleRequest(f"s{i}", entity_id=i, as_of_time=100) for i in (1, 2, 3))


class TestTheInMemoryEngineRunsThroughTheSeam:
    """The first engine behind the boundary, and the proof the boundary works."""

    def test_a_materialized_task_emits_batches(self) -> None:
        task = prepare(CONTRACT, IN_MEMORY, InMemoryEngine()).materialize(REQUESTS)
        batch = next(iter(task.batches(REQUESTS)))
        assert list(batch.inputs) == ["values"]

    def test_model_output_requires_an_engine_optimized_source(self) -> None:
        task = prepare(CONTRACT, IN_MEMORY, InMemoryEngine()).materialize(REQUESTS)
        with pytest.raises(ValueError, match="does not implement"):
            next(iter(task.batches(REQUESTS, output_format="model")))

    def test_output_format_is_explicit(self) -> None:
        task = prepare(CONTRACT, IN_MEMORY, InMemoryEngine()).materialize(REQUESTS)
        with pytest.raises(ValueError, match="inspectable.*model"):
            next(iter(task.batches(REQUESTS, output_format="fast")))

    def test_preparing_twice_is_deterministic(self) -> None:
        first = prepare(CONTRACT, IN_MEMORY, InMemoryEngine()).materialize(REQUESTS)
        second = prepare(CONTRACT, IN_MEMORY, InMemoryEngine()).materialize(REQUESTS)
        assert first.fitted_state == second.fitted_state
        assert first.manifest["artifact_fingerprint"] == second.manifest["artifact_fingerprint"]
        assert first.fingerprints == second.fingerprints

    def test_an_engine_refuses_a_source_kind_it_does_not_read(self) -> None:
        issues = check(CONTRACT, PARQUET, InMemoryEngine())
        issue = next(item for item in issues if "does not read" in item.message)
        assert "parquet" in issue.message

    def test_opening_a_mismatched_source_is_typed_not_guessed(self) -> None:
        with pytest.raises(TypeError, match="in-memory rows only"):
            InMemoryEngine().open(PARQUET, SCHEMA)

    def test_a_source_closes_without_holding_anything(self) -> None:
        source = InMemoryEngine().open(IN_MEMORY, SCHEMA)
        source.close()
        source.close()


class TestTheOpenedSourceIsOwned:
    """`Engine.open` acquires; something must release. The plan does."""

    class _Counting:
        """An engine whose source records how often it was closed."""

        name = "counting"

        def __init__(self, fail: bool = False):
            self.closed = 0
            self._fail = fail

        def open(self, dataset, schema):
            engine = self

            class _Source:
                def __init__(self):
                    self._inner = InMemoryEngine().open(dataset, schema)

                def build(self, spec, requests):
                    if engine._fail:
                        raise RuntimeError("fitting blew up")
                    return self._inner.build(spec, requests)

                def fit_graph_fields(self, *args):
                    return self._inner.fit_graph_fields(*args)

                def sample_graph(self, *args):
                    return self._inner.sample_graph(*args)

                def close(self):
                    engine.closed += 1

            return _Source()

    def test_a_plan_closes_the_source_it_owns(self) -> None:
        engine = self._Counting()
        plan = prepare(CONTRACT, IN_MEMORY, engine)
        assert engine.closed == 0
        plan.close()
        assert engine.closed == 1

    def test_closing_twice_is_harmless(self) -> None:
        engine = self._Counting()
        plan = prepare(CONTRACT, IN_MEMORY, engine)
        plan.close()
        plan.close()
        assert engine.closed == 1

    def test_a_plan_is_a_context_manager(self) -> None:
        engine = self._Counting()
        with prepare(CONTRACT, IN_MEMORY, engine) as plan:
            assert next(iter(plan.materialize(REQUESTS).batches(REQUESTS))) is not None
        assert engine.closed == 1

    def test_a_failed_materialization_leaves_the_plan_to_its_owner(self) -> None:
        engine = self._Counting(fail=True)
        with prepare(CONTRACT, IN_MEMORY, engine) as plan:
            with pytest.raises(RuntimeError, match="fitting blew up"):
                plan.materialize(REQUESTS)
            assert engine.closed == 0
        assert engine.closed == 1

    def test_a_closed_plan_refuses_to_materialize(self) -> None:
        plan = prepare(CONTRACT, IN_MEMORY, InMemoryEngine())
        plan.close()
        with pytest.raises(ValueError, match="closed"):
            plan.materialize(REQUESTS)

    def test_a_plan_built_from_a_borrowed_source_does_not_close_it(self) -> None:
        source = InMemoryEngine().open(IN_MEMORY, SCHEMA)
        plan = fit_plan(CONTRACT.sample, CONTRACT.batch, REQUESTS, source, contract=CONTRACT)
        plan.close()
        assert next(iter(plan.batches(REQUESTS))) is not None
