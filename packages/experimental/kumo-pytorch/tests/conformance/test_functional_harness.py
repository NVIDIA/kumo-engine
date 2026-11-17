# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional-area checks fail on an engine with the defect, and pass without it.

Stand-ins rewrite what the reference is asked (a window edge, a selection, a
GroupBy key) so they run without the native package; checks of DiskGraph's own
paths (model batches, fitted state, a fresh process) patch one DiskGraph method.
"""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from test_property_harness import CASES, Mutant, _binding

from conformance.assertions import require_engine
from conformance.generate import (
    LAG,
    SHAPED,
    generate_batch,
    schema_for,
    target_edges,
)
from conformance.outputs import (
    entity_only,
    expected_targets,
    fitted_structure,
    state_differences,
)
from conformance.properties import PROPERTIES, PropertyFailure, clear_runs
from conformance.registry import ENGINE_REFUSALS, INAPPLICABLE, blocked_regions
from conformance.variants import VARIANTS, variant_for
from kumo.pytorch import assembly
from kumo.pytorch.api import (
    Aggregate,
    GroupBy,
    PreparedBatchPlan,
    Rows,
    RowTarget,
    SampleSpec,
    Task,
)
from kumo.pytorch.engines.diskgraph import DiskGraphSource
from kumo.pytorch.engines.in_memory.anchors import InMemoryAnchors

# -- Stand-ins that change what the reference is asked ----------------------


def _rows(rows: Rows, defect: str, target: bool) -> Rows:
    time = rows.time
    if target and time is not None and time.stop is not None and time.stop > 0:
        if defect == "target_stop_exclusive":
            time = replace(time, stop=time.stop - 1)
        if defect == "target_start_inclusive" and time.start is not None and time.start > 0:
            time = replace(time, start=time.start - 1)
        if defect == "target_start_ignored":
            time = replace(time, start=0)
    if not target and defect == "lag_stop_ignored" and time is not None and (time.stop or 0) < 0:
        time = replace(time, stop=0)
    distinct = {"distinct_forced": True, "distinct_ignored": False}.get(defect, rows.distinct)
    return Rows(rows.path, time, rows.filters, rows.active, distinct)


def _source(source: Rows | GroupBy, defect: str, target: bool) -> Rows | GroupBy:
    if isinstance(source, Rows):
        return _rows(source, defect, target)
    keys = source.keys[:1] if defect == "group_keys_dropped" else source.keys
    return GroupBy(_source(source.source, defect, target), keys, source.values)


def _asked(spec: SampleSpec, defect: str) -> SampleSpec:
    values = {
        name: replace(value, source=_source(value.source, defect, False))
        if isinstance(value, Aggregate)
        else value
        for name, value in spec.entity_values.items()
    }
    targets = {}
    for name, target in spec.targets.items():
        if isinstance(target, Aggregate):
            targets[name] = replace(target, source=_source(target.source, defect, True))
        else:
            selection = target.selection
            if defect == "row_target_swapped":
                selection = "last" if selection == "first" else "first"
            targets[name] = RowTarget(_rows(target.rows, defect, True), target.fields, selection)
    entity = spec.entity
    if defect == "available_at_cutoff":
        entity = replace(entity, available_from=None)
    return replace(spec, entity=entity, entity_values=values, targets=targets)


class Asked(Mutant):
    """The reference, answering a slightly different question than it was asked."""

    def open(self, dataset, schema):
        source = super().open(dataset, schema)
        build = source.build

        def defective(spec, requests):
            try:
                return build(_asked(spec, self.defect), requests)
            except ValueError as error:
                if self.defect == "missing_root_bare" and "expected one" in str(error):
                    raise RuntimeError("lookup failed") from None
                raise

        source.build = defective
        return source

    def materialize_anchors(self, table):
        anchors = InMemoryAnchors(table)
        rows = anchors._rows

        def defective(selection):
            chosen = rows(selection)
            splits = [anchors._split_of(row) for row in chosen]
            if self.defect == "null_split_dropped":
                return tuple(row for row, split in zip(chosen, splits, strict=True) if split)
            if self.defect == "union_unsorted":
                return tuple(sorted(chosen, key=lambda row: str(anchors._split_of(row))))
            return chosen

        anchors._rows = defective
        return anchors


def _case(**dimensions):
    return next(
        (position, case)
        for position, case in enumerate(CASES)
        if all(case.dimensions[name] == value for name, value in dimensions.items())
    )


def _run(name: str, dimensions: dict, defect: str, kind: str = "reference") -> None:
    position, case = _case(**dimensions)
    if kind == "reference":
        binding, engine = _binding(reference=Asked(defect)), "in_memory"
    else:
        binding, engine = _binding(native=Asked(defect)), "diskgraph"
    with patch("conformance.properties.bound_tables", binding):
        PROPERTIES[name].check(case, engine, position)


SIBLING = {"schema_shape.path": "depth2_sibling"}
#: (property, case dimensions, stand-in kind, defect, expected failure, sweep mutation)
STAND_INS = (
    ("label_window", {}, "reference", "target_stop_exclusive", "label window", "M11"),
    ("label_window", {}, "reference", "target_start_inclusive", "label window", "M12"),
    ("label_window", {}, "reference", "target_start_ignored", "label window", "M29"),
    ("differential", {}, "native", "row_target_swapped", "vs reference", "M13"),
    ("differential", {}, "native", "target_stop_exclusive", "vs reference", "M11"),
    ("differential", {}, "native", "lag_stop_ignored", "vs reference", "lag windows"),
    (
        "differential",
        {**SIBLING, "contract.distinct": True},
        "native",
        "distinct_ignored",
        "vs reference",
        "M03b",
    ),
    (
        "differential",
        {**SIBLING, "contract.distinct": False},
        "native",
        "distinct_forced",
        "vs reference",
        "M03",
    ),
    (
        "differential",
        {"contract.group_by": "nested"},
        "native",
        "group_keys_dropped",
        "vs reference",
        "M16",
    ),
    ("anchor_splits", {}, "reference", "null_split_dropped", "count|served", "M21"),
    ("anchor_splits", {}, "reference", "union_unsorted", "served", "M19"),
    ("root_availability", {}, "reference", "available_at_cutoff", "was answered", "M14"),
    ("root_availability", {}, "reference", "missing_root_bare", "names neither", "C6"),
)


@pytest.mark.parametrize(
    "name,dimensions,kind,defect,expected,mutation",
    STAND_INS,
    ids=[f"{row[0]}-{row[3]}" for row in STAND_INS],
)
def test_a_wrong_engine_fails_a_functional_property(
    name, dimensions, kind, defect, expected, mutation
):
    with pytest.raises(PropertyFailure, match=expected):
        _run(name, dimensions, defect, kind)
    _run(name, dimensions, "none", kind)


def test_distinct_and_group_keys_change_answers_on_their_cases():
    """The cases the differential relies on actually reach a row twice / group twice."""

    for dimensions in (
        {**SIBLING, "contract.distinct": True},
        {**SIBLING, "contract.distinct": False},
        {"contract.group_by": "nested"},
    ):
        defect = "group_keys_dropped" if "contract.group_by" in dimensions else "distinct_forced"
        if dimensions.get("contract.distinct") is True:
            defect = "distinct_ignored"
        with pytest.raises(PropertyFailure):
            _run("differential", dimensions, defect, "native")


@contextmanager
def _patched(target: object, name: str, wrap: Callable) -> Iterator[None]:
    original = getattr(target, name)
    clear_runs()
    try:
        with patch.object(target, name, wrap(original)):
            yield
    finally:
        clear_runs()


def _ignore_budget(original):
    def compose(self, order, lengths, policy, max_samples):
        return original(self, order, lengths, replace(policy, max_tokens=None), max_samples)

    return compose


def _lose_last(original):
    def compose(self, order, lengths, policy, max_samples):
        return original(self, order, lengths, policy, max_samples)[:-1] or [[]]

    return compose


def _never_drop(original):
    def batches(self, *arguments, drop_last=None, **keywords):
        return original(self, *arguments, drop_last=False, **keywords)

    return batches


PACKING_DEFECTS = (
    (PreparedBatchPlan, "_compose_batches", _ignore_budget, "tokens"),
    (PreparedBatchPlan, "_compose_batches", _lose_last, "not a permutation"),
    (PreparedBatchPlan, "batches", _never_drop, "drop_last"),
)


@pytest.mark.parametrize("target,name,wrap,expected", PACKING_DEFECTS, ids=lambda x: str(x)[:20])
def test_packing_fails_on_a_wrong_planner(target, name, wrap, expected):
    position, case = _case()
    with _patched(target, name, wrap), pytest.raises(PropertyFailure, match=expected):
        PROPERTIES["packing"].check(case, "in_memory", position)
    PROPERTIES["packing"].check(case, "in_memory", position)


def _drop_every_short_batch(original):
    def batches(self, *arguments, drop_last=None, **keywords):
        size = keywords.get("batch_size") or self.spec.batching.max_samples
        drop = self.spec.batching.drop_last if drop_last is None else drop_last
        for batch in original(self, *arguments, drop_last=False, **keywords):
            if not (drop and len(batch.metadata["sample_ids"]) < size):
                yield batch

    return batches


def test_packing_checks_drop_last_under_a_token_budget():
    """Dropping every batch the budget closed early, not only the final one, fails."""

    position, case = _case()
    budgeted = replace(case, dimensions={**case.dimensions, "execution.drop_last": "token_budget"})
    with (
        _patched(PreparedBatchPlan, "batches", _drop_every_short_batch),
        pytest.raises(PropertyFailure, match="drop_last under max_tokens=12 kept 0 of"),
    ):
        PROPERTIES["packing"].check(budgeted, "in_memory", position)
    PROPERTIES["packing"].check(budgeted, "in_memory", position)


def _ranks_ignored(original):
    def batches(self, *arguments, rank=0, world_size=1, **keywords):
        return original(self, *arguments, **keywords)

    return batches


def _unclear_refusal(original):
    def batches(self, *arguments, world_size=1, **keywords):
        if world_size > 1:
            raise ValueError("unsupported")
        return original(self, *arguments, world_size=world_size, **keywords)

    return batches


@pytest.mark.parametrize(
    "wrap,expected",
    ((_ranks_ignored, "delivered before refusing"), (_unclear_refusal, "refused unclearly")),
)
def test_refusal_timing_requires_a_declared_late_refusal(wrap, expected):
    """max_tokens with world_size > 1 is a batches() argument: refused there, clearly."""

    position, case = _case(**{"contract.strategy": "max_tokens_world_size"})
    with _patched(Task, "batches", wrap), pytest.raises(PropertyFailure, match=expected):
        PROPERTIES["refusal_timing"].check(case, "in_memory", position)
    PROPERTIES["refusal_timing"].check(case, "in_memory", position)


def test_refusal_timing_fails_when_a_declared_refusal_is_accepted():
    """M25–M28: DiskGraph accepting a strategy it declares refused fails the property."""

    declared = {r.strategy for r in ENGINE_REFUSALS if r.engine == "diskgraph"}
    for strategy in ("active_interval", "clock_override", "dated_values", "static_traversal"):
        assert strategy in declared
        case = next(case for case in CASES if case.dimensions["contract.strategy"] == strategy)
        with patch("conformance.properties.bound_tables", _binding(native=Asked("none"))):
            with pytest.raises(PropertyFailure, match=f"accepted {strategy}, declared refused"):
                PROPERTIES["refusal_timing"].check(case, "diskgraph", 0)


def test_every_variant_is_a_registered_strategy_and_valid_on_the_reference():
    from conformance.registry import BY_NAME

    assert set(VARIANTS) <= set(BY_NAME["contract.strategy"].values)
    position, case = _case()
    for strategy in VARIANTS:
        variant_for(case, strategy).contract.validate()


def test_regions_keep_refusal_strategies_generated_and_inapplicable_values_out():
    regions = blocked_regions("diskgraph")
    assert not any(r in regions for r in ENGINE_REFUSALS if r.strategy == "clock_override")
    for engine in ("in_memory", "diskgraph"):
        values = {
            value
            for case in generate_batch(0, engine)
            for value in (case.dimensions["execution.output_format"],)
        }
        assert ("model" in values) == (engine == "diskgraph")
    assert all(rule.engine == "in_memory" or rule.refused_at for rule in INAPPLICABLE)


@pytest.mark.parametrize("case", [c for c in CASES if c.dimensions["schema_shape.path"] in SHAPED])
def test_new_shapes_realize_what_they_promise(case):
    shape = case.dimensions["schema_shape.path"]
    tables = case.tables
    if shape == "depth2_sibling":
        linked = [row["event_id"] for row in tables["links"]]
        assert len(set(linked)) < len(linked), "some event is linked twice"
        assert set(linked) == {row["event_id"] for row in tables["events"]}
    if shape == "depth1_parent":
        segments = {row["segment_id"] for row in tables["users"]}
        assert None in segments or len(segments) > 1
    if shape == "depth3_static":
        owners = {row["user_id"] for row in tables["accounts"]}
        assert owners == {row["user_id"] for row in tables["users"]}
    assert "users" in {table.name for table in schema_for(shape).tables}


def test_every_case_plants_rows_at_every_target_and_lag_edge():
    for case in CASES:
        times = {row["at"] for row in case.tables["events"]}
        for request in case.requests:
            assert {request.as_of_time + edge for edge in target_edges()} <= times, case.id
        lagged = {r.as_of_time - LAG for r in case.requests} | {
            r.as_of_time - LAG + 1 for r in case.requests
        }
        assert len(lagged & times) >= 2, case.id


def test_a_tied_selection_is_judged_by_its_time():
    before = {"target:last_next.offset": 300, "target:last_next.event_id": 7}
    before |= {"target:last_next.amount": 2.0}
    expected = expected_targets(before, [(100 + 300, 50), (100 + 301, 51)], 100, 1)
    assert expected["target:last_next.offset"] == 300
    assert "target:last_next.event_id" not in expected


def test_fitted_state_is_compared_by_entries_and_types():
    left = {"vocabulary": {"a": 0, "b": 1}, "boundaries": [1.0, 2.0]}
    right = {"vocabulary": {"b": 0, "a": 1}, "boundaries": [1.0, 2.0]}
    assert state_differences(fitted_structure(left), fitted_structure(right), 1e-12) == []
    assert state_differences({"x": 3}, {"x": 3.0}, 1e-12) == ["state.x: 3 (int) != 3.0"]
    tokenizer = '{"model": {"vocab": {"a": 0, "b": 1}}}'
    assert fitted_structure({"json": tokenizer}) == {"json": {"model": {"vocab": ["a", "b"]}}}


def test_expected_targets_follow_the_exclusive_start_inclusive_stop_rule():
    before = {"target:count_next": 2, "target:sum_next": 3.0, "target:count_ahead": 0}
    before |= {"target:first_next.offset": 5, "target:first_next.event_id": 1}
    before |= {"target:first_next.amount": 2.0}
    rows = [(100 + offset, 50 + offset) for offset in (0, 1, 60, 240, 301)]
    expected = expected_targets(before, rows, 100, 1)
    assert expected["target:count_next"] == 5 and type(expected["target:count_next"]) is int
    assert expected["target:sum_next"] == 6.0
    assert expected["target:count_ahead"] == 1
    assert expected["target:first_next.offset"] == 1
    assert expected["target:first_next.event_id"] == 51
    assert expected["target:last_next.offset"] == 240
    assert entity_only({"a": 1, "target:b": 2}) == {"a": 1}


# -- DiskGraph's own paths, with one method patched -------------------------

NATIVE_ONLY = ("model_delivery", "model_epoch_trend", "fitted_state", "contract_transfer")


@pytest.fixture(scope="module")
def native_cases():
    require_engine("diskgraph")
    cases = generate_batch(0, "diskgraph")
    model = next(c for c in cases if c.dimensions["execution.output_format"] == "model")
    return cases, model


def _reverse_entity_values(original):
    def adapt(self, *arguments, **keywords):
        result = original(self, *arguments, **keywords)
        values = {name: array[::-1] for name, array in result.entity_values.items()}
        return replace(result, entity_values=values)

    return adapt


def _shift_model_cutoffs(original):
    def build_model(self, spec, requests, **keywords):
        later = [replace(request, as_of_time=request.as_of_time + 10**6) for request in requests]
        result = original(self, spec, later, **keywords)
        return replace(result, evaluation_times=tuple(r.as_of_time for r in requests))

    return build_model


def _growing_model_work(original):
    calls = []

    def build_model(self, *arguments, **keywords):
        result = original(self, *arguments, **keywords)
        calls.append(1)
        self._last_temporal_profile = {"native_stats": {"profiled": len(calls)}}
        return result

    return build_model


def _perturb_sum(original):
    def build_model(self, spec, requests, **keywords):
        result = original(self, spec, requests, **keywords)
        values = dict(result.entity_values)
        if "sum" in values:
            values["sum"] = pc.add(values["sum"], 1)
        return replace(result, entity_values=values)

    return build_model


def _unbound(original):
    def bound_contract(self):
        return self.contract

    return bound_contract


def _longer_lengths(original):
    def lengths(self, *arguments, **keywords):
        return [length + 1 for length in original(self, *arguments, **keywords)]

    return lengths


NATIVE_DEFECTS = (
    ("differential", "_adapt_model_result", _reverse_entity_values, "vs reference", "M22"),
    ("point_in_time", "build_model_ordered", _shift_model_cutoffs, "future rows", "model PIT"),
    ("fitted_state", "build_model", _shift_model_cutoffs, "tokens: fit inputs", "token fit"),
    ("model_delivery", "_adapt_model_result", _reverse_entity_values, "differs", "M22"),
    ("model_epoch_trend", "build_model_batch", _growing_model_work, "!= epoch 2", "model work"),
    ("fitted_state", "build_model", _perturb_sum, "columns: fit inputs", "column fit"),
    ("contract_transfer", "bound_contract", _unbound, "fresh process", "lost state"),
    ("packing", "model_lengths", _longer_lengths, "differ by format", "M17"),
)


@pytest.mark.engine("diskgraph")
@pytest.mark.parametrize(
    "name,method,wrap,expected,mutation", NATIVE_DEFECTS, ids=[row[0] for row in NATIVE_DEFECTS]
)
def test_a_wrong_diskgraph_fails_its_own_path(native_cases, name, method, wrap, expected, mutation):
    cases, model = native_cases
    target = Task if method == "bound_contract" else DiskGraphSource
    case = model
    with ExitStack() as stack:
        stack.enter_context(_patched(target, method, wrap))
        with pytest.raises(PropertyFailure, match=expected):
            PROPERTIES[name].check(case, "diskgraph", 0)
    PROPERTIES[name].check(case, "diskgraph", 0)


def _swap_float_columns(original):
    def rows(source, values, declarations):
        typed = original(source, values, declarations)
        floats = [
            name for name, column in typed.columns.items() if pa.types.is_floating(column.type)
        ]
        columns = dict(typed.columns)
        for target, moved in zip(floats, floats[1:] + floats[:1], strict=True):
            columns[target] = typed.columns[moved]
        return replace(typed, columns=columns)

    return rows


@pytest.mark.engine("diskgraph")
def test_a_wrong_typed_fitting_path_fails_fitted_state(native_cases):
    """DiskGraph fits field tokens from typed columns; a wrong field mapping is caught."""

    cases, _ = native_cases
    with _patched(assembly, "_model_rows", _swap_float_columns):
        with pytest.raises(PropertyFailure, match="tokens: fitted state differs"):
            PROPERTIES["fitted_state"].check(cases[0], "diskgraph", 0)
    PROPERTIES["fitted_state"].check(cases[0], "diskgraph", 0)


#: Every functional-area property, with the tests above that break it; index_layout's
#: are in test_fake_ingest.py.
FUNCTIONAL_COVERED = (
    {row[0] for row in STAND_INS} | {row[0] for row in NATIVE_DEFECTS} | {"packing", "index_layout"}
) - {"differential", "point_in_time"}


def test_every_functional_property_has_a_wrong_engine():
    assert set(NATIVE_ONLY) <= FUNCTIONAL_COVERED <= PROPERTIES.keys()
