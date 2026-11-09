# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph and sequence properties: wrong engines fail them, generators realize their rows."""

import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest

from conformance.assertions import PropertyFailure, bound_tables
from conformance.generate import GeneratedCase, Row, _pairs_of, required_pairs
from conformance.graph_cases import (
    EXPANSION,
    GRAPH_CHECKS,
    PATHS,
    REVISITING,
    graph_batch,
    traversals,
)
from conformance.properties import PROPERTIES, run_obligation
from conformance.registry import (
    BY_NAME,
    GRAPH_DIMENSIONS,
    KNOWN_GAPS,
    SEQUENCE_DIMENSIONS,
    blocked_regions,
)
from conformance.sequence_cases import (
    ELIGIBLE,
    SEQUENCE_CHECKS,
    TABLES,
    sequence_batch,
)
from kumo.pytorch.api import (
    DataLoaderSpec,
    DataSchema,
    Dataset,
    EventTimeDelta,
    InMemoryEngine,
    InMemoryRows,
    SampleSpec,
)
from kumo.pytorch.contract.sample import GraphSample
from kumo.pytorch.engine import GraphRoot
from kumo.pytorch.engines.in_memory.graph import InMemoryGraphSampler


class WrongSampler(InMemoryGraphSampler):
    """The reference graph sampler with exactly one deliberate defect."""

    def __init__(self, schema: DataSchema, tables: Mapping[str, Sequence[Row]], engine: "Wrong"):
        super().__init__(schema, tables)
        self.engine = engine

    def _neighbors(self, row, current_table, traversal: GraphSample, root: GraphRoot, seed, epoch):
        defect = self.engine.defect
        if defect == "future_visible":
            root = replace(root, cutoff=root.cutoff + 10**6)
        if defect == "fanout_plus_one":
            traversal = replace(traversal, fanout=traversal.fanout + 1)
        if defect == "static_taints_shops":
            if traversal.temporal == "static":
                self.engine.tainted = True
            if self.engine.tainted and current_table == "shops":
                traversal = replace(traversal, temporal="static")
        if defect in ("tied_second_hop_dropped", "late_beyond_tied_first_hop"):
            if current_table == "users":
                self._note_tie(row, current_table, traversal, root, seed, epoch)
            elif root.sample_id in self.engine.tied:
                if defect == "tied_second_hop_dropped":
                    return super()._neighbors(row, current_table, traversal, root, seed, epoch)[
                        0
                    ], []
                if self.engine.opened > 1:
                    root = replace(root, cutoff=root.cutoff + 10**6)
        if defect not in ("row_order_ties", "unstable_ties", "older_on_ties"):
            return super()._neighbors(row, current_table, traversal, root, seed, epoch)
        everything = replace(traversal, fanout=10**6)
        table, rows = super()._neighbors(row, current_table, everything, root, seed, epoch)
        position = {id(candidate): i for i, candidate in enumerate(self._tables[table])}
        flip = -1 if defect == "unstable_ties" and self.engine.sampled % 2 else 1
        clock = self._traversal_clock(traversal, table)
        rows.sort(
            key=lambda candidate: (candidate[clock], -flip * position[id(candidate)]), reverse=True
        )
        fanout = traversal.fanout
        if defect == "older_on_ties" and len(rows) > fanout:
            limit = rows[fanout - 1][clock]
            if rows[fanout][clock] == limit:
                rows = [candidate for candidate in rows if candidate[clock] != limit]
        return table, rows[:fanout]

    def _note_tie(self, row, current_table, traversal: GraphSample, root: GraphRoot, seed, epoch):
        everything = replace(traversal, fanout=10**6)
        table, rows = super()._neighbors(row, current_table, everything, root, seed, epoch)
        clocks = [candidate[self._traversal_clock(traversal, table)] for candidate in rows]
        fanout = traversal.fanout
        if len(clocks) > fanout and clocks[fanout - 1] == clocks[fanout]:
            self.engine.tied.add(root.sample_id)

    def batch(self, *arguments, **keywords):
        self.engine.sampled += 1
        return super().batch(*arguments, **keywords)

    def _edge_output(self, spec, edge_pairs):
        if self.engine.defect == "repeated_edges":
            edge_pairs = {name: pairs + pairs for name, pairs in edge_pairs.items()}
        output = super()._edge_output(spec, edge_pairs)
        if self.engine.defect == "reversed_edges":
            for edge in output.values():
                edge["index"] = edge["index"].flip(0)
                edge["source_type"], edge["destination_type"] = (
                    edge["destination_type"],
                    edge["source_type"],
                )
        return output


class Wrong(InMemoryEngine):
    """The reference engine with one defect in graph sampling or event retrieval."""

    def __init__(self, defect: str):
        super().__init__()
        self.defect = defect
        self.opened = self.sampled = 0
        self.tainted = False
        self.tied: set[str] = set()

    def open(self, dataset: Dataset, schema: DataSchema):
        self.opened += 1
        tables = {name: list(source.rows) for name, source in dataset.tables.items()}
        dropped = self.defect == "first_order_dropped" or (
            self.defect == "unstable_rows" and self.opened % 2
        )
        if dropped and "orders" in tables:
            tables["orders"] = tables["orders"][1:]
        source = super().open(Dataset({n: InMemoryRows(r) for n, r in tables.items()}), schema)
        source._graph = WrongSampler(schema, tables, self)
        build = source.build
        shift = {"future_visible": 10**6, "off_by_one": 1}.get(self.defect, 0)

        def shifted(spec, requests):
            if self.defect == "delta_in_seconds":
                spec = _in_seconds(spec)
            return build(spec, [replace(r, as_of_time=r.as_of_time + shift) for r in requests])

        source.build = shifted
        return source


def _in_seconds(spec: SampleSpec) -> SampleSpec:
    """Every EventTimeDelta measured in seconds, whatever unit it declares."""

    events = {
        name: replace(
            event,
            fields={
                key: EventTimeDelta("seconds") if isinstance(field, EventTimeDelta) else field
                for key, field in event.fields.items()
            },
        )
        for name, event in spec.events.items()
    }
    return replace(spec, events=events)


def _binding(stand_in: Wrong, replaces: str):
    """Stand in for one engine of `bound_tables`; the other stays the real reference."""

    @contextmanager
    def binding(
        contract: DataLoaderSpec, tables, engine_name: str, *, refusal=False, parts=1, types=None
    ) -> Iterator[tuple[Dataset, InMemoryEngine]]:
        with bound_tables(contract, tables, "in_memory", refusal=refusal) as (dataset, engine):
            yield dataset, stand_in if engine_name == replaces else engine

    return binding


def _graph_case(name: str, **dimensions: object) -> GeneratedCase:
    return next(
        case
        for case in graph_batch(0, "in_memory", name)
        if all(case.dimensions[key] == value for key, value in dimensions.items())
    )


UNTIED = {"time.graph_ties": "distinct", "contract.graph_temporal": "inherit"}
TIED = {"time.graph_ties": "tied", "contract.graph_temporal": "inherit"}
#: Property -> (case dimensions, engine replaced, defect, expected failure).
WRONG_ENGINES = {
    "graph_differential": (UNTIED, "diskgraph", "fanout_plus_one", "vs reference changed graphs"),
    "graph_inference_exact": (
        UNTIED,
        "diskgraph",
        "future_visible",
        "future rows changed diskgraph's graphs",
    ),
    "graph_point_in_time": (UNTIED, "in_memory", "future_visible", "after the cutoff"),
    "graph_structure": (
        {"schema_shape.graph_path": "one_hop"},
        "in_memory",
        "fanout_plus_one",
        "exceed fanout",
    ),
    "graph_edge_uniqueness": ({}, "in_memory", "repeated_edges", "edges repeat within a sample"),
    "graph_determinism": (TIED, "in_memory", "unstable_ties", "the same index again changed"),
    "graph_tie_invariance": (TIED, "in_memory", "row_order_ties", "row and request order changed"),
    "sequence_differential": ({}, "diskgraph", "off_by_one", "vs reference changed sequences"),
    "sequence_point_in_time": (
        {"contract.event_delta_unit": "minutes"},
        "in_memory",
        "future_visible",
        "after the cutoff",
    ),
    "sequence_permutation": ({}, "in_memory", "first_order_dropped", "row and request order"),
}
MORE_DEFECTS = (
    ("graph_structure", "reversed_edges", "break orientation"),
    ("graph_structure", "older_on_ties", "recent kept other than the most recent"),
    ("graph_differential", "older_on_ties", "changed tied samples' first-hop clocks"),
    ("graph_point_in_time", "static_taints_shops", "after the cutoff"),
    ("sequence_point_in_time", "delta_in_seconds", "wrong time deltas"),
    ("sequence_permutation", "unstable_rows", "a second run changed sequences"),
)


#: Defects that need other case dimensions than their property's row above.
DEFECT_CASES = {
    ("graph_structure", "older_on_ties"): TIED,
    ("graph_differential", "older_on_ties"): TIED,
    ("graph_point_in_time", "static_taints_shops"): {
        "schema_shape.graph_path": "fan_in",
        "contract.graph_temporal": "static",
    },
}


def _check(name: str, defect: str) -> None:
    dimensions, replaces, _, _ = WRONG_ENGINES[name]
    dimensions = DEFECT_CASES.get((name, defect), dimensions)
    if name.startswith("graph"):
        case, module = _graph_case(name, **dimensions), "conformance.graph_cases"
    else:
        case = next(
            case
            for case in sequence_batch(0, "in_memory", name)
            if all(case.dimensions[key] == value for key, value in dimensions.items())
        )
        module = "conformance.sequence_cases"
    with patch(f"{module}.bound_tables", _binding(Wrong(defect), replaces)):
        PROPERTIES[name].check(case, replaces, 0)


#: Tied samples' later hops deleted (the review's g0-c10), or dated after the cutoff in the
#: run with future rows: first-hop clocks stay the same, the layer-by-layer expansion doesn't.
LATER_HOPS = (
    ("graph_structure", "g0-c10", "in_memory", "tied_second_hop_dropped", EXPANSION),
    (
        "graph_differential",
        "g0-c10",
        "diskgraph",
        "tied_second_hop_dropped",
        f"diskgraph vs reference: diskgraph: {EXPANSION}",
    ),
    (
        "graph_point_in_time",
        "g0-c15",
        "in_memory",
        "late_beyond_tied_first_hop",
        f"future rows: with future rows: {EXPANSION}",
    ),
)


@pytest.mark.parametrize("name,case_id,replaces,defect,expected", LATER_HOPS)
def test_tied_samples_are_checked_at_every_layer(name, case_id, replaces, defect, expected):
    case = next(case for case in graph_batch(0, "diskgraph", name) if case.id == case_id)
    assert case.dimensions["time.graph_ties"] == "tied"
    with patch("conformance.graph_cases.bound_tables", _binding(Wrong("none"), replaces)):
        PROPERTIES[name].check(case, replaces, 0)
    stand_in = Wrong(defect)
    with patch("conformance.graph_cases.bound_tables", _binding(stand_in, replaces)):
        with pytest.raises(PropertyFailure, match=re.escape(expected)):
            PROPERTIES[name].check(case, replaces, 0)
    assert stand_in.tied, "the defect touched no tied sample"


def test_every_sampling_property_has_a_wrong_engine():
    assert WRONG_ENGINES.keys() == GRAPH_CHECKS.keys() | SEQUENCE_CHECKS.keys()


@pytest.mark.parametrize("name", sorted(WRONG_ENGINES))
def test_the_stand_in_passes_without_its_defect(name):
    _check(name, "none")


@pytest.mark.parametrize(
    "name,defect,expected",
    [(name, defect, expected) for name, (_, _, defect, expected) in sorted(WRONG_ENGINES.items())]
    + list(MORE_DEFECTS),
)
def test_a_wrong_engine_fails_the_property(name, defect, expected):
    with pytest.raises(PropertyFailure, match=expected):
        _check(name, defect)


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("name", sorted(GRAPH_CHECKS.keys() | SEQUENCE_CHECKS.keys()))
def test_batches_cover_every_pair_outside_their_own_gaps(name, engine):
    batch = graph_batch if name.startswith("graph") else sequence_batch
    cases = batch(0, engine, name)
    names = GRAPH_DIMENSIONS if name.startswith("graph") else SEQUENCE_DIMENSIONS
    regions = [
        region for region in blocked_regions(engine) if getattr(region, "property", name) == name
    ]
    domains = {n: BY_NAME[n].values for n in names}
    required = {
        pair
        for pair in required_pairs({name: names}, domains)
        if not any(region.covers(dict(pair)) for region in regions)
    }
    assert required <= set().union(*(_pairs_of(case.dimensions) for case in cases))
    assert not any(region.covers(case.dimensions) for region in regions for case in cases)
    assert [c.to_dict() for c in batch(0, engine, name)] == [c.to_dict() for c in cases]


def test_a_gap_blocks_only_its_own_property():
    gaps = {gap.property: gap for gap in KNOWN_GAPS if gap.engine == "diskgraph"}
    assert gaps["graph_tie_invariance"].when == {"time.graph_ties": ("tied",)}

    def ties(name: str) -> set[object]:
        return {case.dimensions["time.graph_ties"] for case in graph_batch(0, "diskgraph", name)}

    assert ties("graph_tie_invariance") == {"distinct"}
    assert ties("graph_structure") == {"distinct", "tied"}


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
def test_edge_uniqueness_covers_paths_that_revisit_a_relation(engine):
    paths = {
        case.dimensions["schema_shape.graph_path"]
        for case in graph_batch(0, engine, "graph_edge_uniqueness")
    }
    assert set(REVISITING) <= paths == set(PATHS)


def _eligible(case: GeneratedCase, table: str, user: int, cutoff: int) -> list[int]:
    return [
        int(row["at"])
        for row in case.tables[table]
        if row["user_id"] == user and int(row["at"]) <= cutoff
    ]


@pytest.mark.parametrize("case", graph_batch(0, "in_memory", "graph_structure"), ids=lambda c: c.id)
def test_graph_cases_realize_their_dimensions(case):
    dims = case.dimensions
    fanout = dims["contract.graph_fanout"]
    events = case.tables["events"]
    for request in case.requests:
        times = _eligible(case, "events", request.entity_id, request.as_of_time)
        assert len(times) > fanout
        assert any(
            r["user_id"] == request.entity_id and r["at"] == request.as_of_time for r in events
        )
    groups = Counter((row["user_id"], row["at"]) for row in events)
    groups.update((-int(row["shop_id"]), row["at"]) for row in events)
    tied = any(count > 1 for count in groups.values())
    assert tied == (dims["time.graph_ties"] == "tied")
    if tied:
        first = case.requests[0]
        top = sorted(_eligible(case, "events", first.entity_id, first.as_of_time))[-(fanout + 1) :]
        assert len(set(top)) == 1, "the fanout limit falls inside a tie"
    cutoffs = {request.as_of_time for request in case.requests}
    assert (len(cutoffs) > 1) == (dims["time.cutoffs"] == "mixed")
    samples = traversals(case.contract)
    assert len(case.contract.sample.neighborhoods["g"].layers) == len(
        PATHS[dims["schema_shape.graph_path"]]
    )
    assert {s.fanout for s in samples} == {fanout}
    to_events = {("user_events", "forward"), ("event_user", "reverse")}
    static = {s.temporal for s in samples if (s.relation, s.direction) in to_events}
    assert static == {"static" if dims["contract.graph_temporal"] == "static" else "inherit"}
    relations = {r.name: r for r in case.contract.schema.relations}
    shop = relations.get("event_shop") or relations["shop_event"]
    assert (shop.left_table == "events") == (dims["schema_shape.graph_orientation"] == "aligned")
    users = relations.get("user_events") or relations["event_user"]
    assert (users.left_table == "users") == (dims["contract.graph_direction"] == "forward")


@pytest.mark.parametrize(
    "case", sequence_batch(0, "in_memory", "sequence_permutation"), ids=lambda c: c.id
)
def test_sequence_cases_realize_their_dimensions(case):
    dims = case.dimensions
    for request in case.requests:
        assert len(_eligible(case, "orders", request.entity_id, request.as_of_time)) >= ELIGIBLE
    times = Counter((row["user_id"], row["at"]) for table in TABLES for row in case.tables[table])
    assert any(count > 1 for count in times.values()) == (dims["time.ties"] == "tied")
    orders = case.contract.sample.events["orders"]
    assert orders.order == dims["contract.event_order"]
    assert (orders.limit is not None) == (dims["contract.event_limit"] == "limited")
    merged = case.contract.batch.inputs["merged"].source
    assert merged.keep == dims["contract.event_keep"]
    assert (merged.max_events is not None) == (dims["contract.event_max"] == "limited")
    assert case.contract.batch.inputs["pushed"].source.max_events is not None
    cutoffs = {request.as_of_time for request in case.requests}
    assert (len(cutoffs) > 1) == (dims["time.cutoffs"] == "mixed")


@pytest.mark.parametrize("dependent", (True, False))
def test_sequence_obligations_run_each_case_at_its_own_shards(dependent, tmp_path):
    """A DiskGraph whose sequences depend on num_shards fails; one that doesn't passes."""

    @contextmanager
    def binding(contract, tables, engine_name, *, refusal=False, parts=1, types=None):
        with patch("conformance.assertions.require_engine"):
            with bound_tables(contract, tables, "diskgraph", refusal=True) as (_, native):
                shards = native.num_shards
        defect = "first_order_dropped" if dependent and shards == 3 else "none"
        with bound_tables(contract, tables, "in_memory", refusal=refusal) as (dataset, engine):
            yield dataset, engine if engine_name == "in_memory" else Wrong(defect)

    with patch("conformance.sequence_cases.bound_tables", binding):
        if not dependent:
            run_obligation("sequence_differential", "diskgraph", 0, tmp_path)
            return
        with pytest.raises(PropertyFailure, match="vs reference changed sequences"):
            run_obligation("sequence_differential", "diskgraph", 0, tmp_path)


@pytest.mark.parametrize(
    ("shards", "threads", "checked"), [(1, 1, True), (1, 4, True), (3, 1, True), (3, 4, False)]
)
def test_inference_exact_checks_tied_samples_where_builds_are_deterministic(
    shards, threads, checked
):
    from conformance.graph_cases import _layout_deterministic

    case = _graph_case("graph_inference_exact", **TIED)
    dims = {**case.dimensions, "execution.num_shards": shards, "execution.num_threads": threads}
    assert _layout_deterministic(replace(case, dimensions=dims)) is checked
