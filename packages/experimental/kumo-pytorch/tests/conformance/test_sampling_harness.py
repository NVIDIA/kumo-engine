# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph and sequence properties: wrong engines fail them, generators realize their rows."""

import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest

from conformance.assertions import PropertyFailure, bound_tables
from conformance.generate import GeneratedCase, _pairs_of, required_pairs
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
    Dataset,
    DiskGraphEngine,
    EventTimeDelta,
    SampleSpec,
)
from tests.native_fixtures import directory, parquet_dataset


class Wrong(DiskGraphEngine):
    """Real native execution with one deliberate retrieval or output fault."""

    def __init__(self, defect):
        super().__init__(index=directory() / "index", num_threads=1, num_shards=1)
        self.defect = defect
        self.opened = self.sampled = 0
        self.tied = set()
        self.graph_calls = Counter()

    def open(self, dataset, schema):
        import pyarrow.parquet as pq

        self.opened += 1
        tables = {
            name: pq.read_table(binding.path).to_pylist()
            for name, binding in dataset.tables.items()
        }
        dropped = self.defect == "first_order_dropped" or (
            self.defect == "unstable_rows" and self.opened % 2
        )
        if dropped and "orders" in tables:
            tables["orders"] = tables["orders"][1:]
            dataset = parquet_dataset(schema, tables, root=directory())
        graph_future = "orders" not in tables and (
            self.defect == "future_visible"
            or (self.defect == "future_after_reopen" and self.opened > 1)
        )
        if graph_future:
            from datetime import timedelta

            for table in schema.tables:
                if table.time_column in table.columns:
                    for row in tables[table.name]:
                        row[table.time_column] -= timedelta(seconds=10**6)
            dataset = parquet_dataset(schema, tables, root=directory())
        source = super().open(dataset, schema)
        build, graph, sampling_plan = source.build, source.sample_graph_batch, source._sampling_plan

        def shifted(spec, requests):
            from datetime import datetime, timedelta

            if self.defect == "delta_in_seconds":
                spec = _in_seconds(spec)
            shift = {"future_visible": 10**6, "off_by_one": 1}.get(self.defect, 0)
            return build(
                spec,
                [
                    replace(
                        r,
                        as_of_time=r.as_of_time
                        + (
                            timedelta(seconds=shift)
                            if isinstance(r.as_of_time, datetime)
                            else shift
                        ),
                    )
                    for r in requests
                ],
            )

        def altered_plan(neighborhood):
            layers = list(neighborhood.layers)
            if self.defect == "fanout_plus_one":
                layers = [
                    replace(
                        layer,
                        samples=tuple(replace(hop, fanout=hop.fanout + 1) for hop in layer.samples),
                    )
                    for layer in layers
                ]
            if self.defect == "tied_second_hop_dropped":
                layers = layers[:1]
            return sampling_plan(replace(neighborhood, layers=tuple(layers)))

        def corrupted(sample, spec, state, runtime):
            self.sampled += 1
            call_key = tuple(runtime["task"]["tt_stage_node_ids"].to_pylist())
            self.graph_calls[call_key] += 1
            if self.defect == "tied_second_hop_dropped" or graph_future:
                self.tied.update(call_key)
            output = graph(sample, spec, state, runtime)
            if self.defect == "late_after_shops" or (
                self.defect == "late_beyond_tied_first_hop" and self.opened > 1
            ):
                import torch

                # Append a real joined future event at hop three. The users/events
                # edges and their first-hop clocks remain byte-for-byte unchanged.
                events, shops = output["nodes"]["events"], output["nodes"]["shops"]
                cutoffs = runtime["task"]["seed_timestamps"].to_pylist()
                edge = next(
                    edge
                    for edge in output["edges"].values()
                    if {edge["source_type"], edge["destination_type"]} == {"shops", "events"}
                )
                for shop_index, (shop_id, owner) in enumerate(
                    zip(shops["ids"].tolist(), shops["batch"].tolist(), strict=True)
                ):
                    selected = {
                        identity
                        for identity, batch_owner in zip(
                            events["ids"].tolist(), events["batch"].tolist(), strict=True
                        )
                        if batch_owner == owner
                    }
                    future = next(
                        (
                            row
                            for row in tables["events"]
                            if row["shop_id"] == shop_id
                            and row["at"].timestamp() > cutoffs[owner]
                            and row["event_id"] not in selected
                        ),
                        None,
                    )
                    if future is None:
                        continue
                    event_index = len(events["ids"])
                    events["ids"] = torch.cat(
                        (
                            events["ids"],
                            torch.tensor([future["event_id"]], dtype=events["ids"].dtype),
                        )
                    )
                    events["batch"] = torch.cat(
                        (events["batch"], torch.tensor([owner], dtype=events["batch"].dtype))
                    )
                    pair = (
                        [shop_index, event_index]
                        if edge["source_type"] == "shops"
                        else [event_index, shop_index]
                    )
                    edge["index"] = torch.cat(
                        (edge["index"], torch.tensor(pair, dtype=edge["index"].dtype)[:, None]),
                        dim=1,
                    )
                    self.tied.add(call_key[owner])
                    break
            if self.defect == "older_on_ties":
                events = output["nodes"]["events"]
                rows = {row["event_id"]: row for row in tables["events"]}
                owners = events["batch"].tolist()
                ids = events["ids"].tolist()
                for owner in set(owners):
                    positions = [i for i, value in enumerate(owners) if value == owner]
                    selected = [rows[ids[i]] for i in positions]
                    oldest = min(row["at"] for row in selected)
                    if not any(
                        row["user_id"] == selected[0]["user_id"]
                        and row["at"] == oldest
                        and row["event_id"] not in ids
                        for row in tables["events"]
                    ):
                        continue
                    older = [
                        row
                        for row in tables["events"]
                        if row["user_id"] == selected[0]["user_id"]
                        and row["at"] < oldest
                        and row["event_id"] not in ids
                    ]
                    if older:
                        events["ids"][positions[-1]] = min(older, key=lambda row: row["at"])[
                            "event_id"
                        ]
                        break
            for edge in output["edges"].values():
                if self.defect == "repeated_edges":
                    import torch

                    edge["index"] = torch.cat([edge["index"], edge["index"]], dim=1)
                elif self.defect == "reversed_edges":
                    edge["index"] = edge["index"].flip(0)
                    edge["source_type"], edge["destination_type"] = (
                        edge["destination_type"],
                        edge["source_type"],
                    )
                elif (self.defect == "unstable_ties" and self.graph_calls[call_key] % 2) or (
                    self.defect == "row_order_ties" and self.opened > 1
                ):
                    edge["index"] = edge["index"][:, :-1]
            return output

        source.build = shifted
        source._sampling_plan = altered_plan
        source.sample_graph_batch = corrupted
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
    """Bind the real native engine with a deliberate fault."""

    @contextmanager
    def binding(
        contract: DataLoaderSpec, tables, engine_name: str, *, refusal=False, parts=1, types=None
    ) -> Iterator[tuple[Dataset, DiskGraphEngine]]:
        with bound_tables(
            contract, tables, "diskgraph", refusal=refusal, parts=parts, types=types
        ) as (dataset, engine):
            stand_in.index = engine.index
            stand_in.num_shards, stand_in.num_threads = engine.num_shards, engine.num_threads
            yield dataset, stand_in

    return binding


def _graph_case(name: str, **dimensions: object) -> GeneratedCase:
    return next(
        case
        for case in graph_batch(0, "diskgraph", name)
        if all(case.dimensions[key] == value for key, value in dimensions.items())
    )


UNTIED = {"time.graph_ties": "distinct"}
TIED = {"time.graph_ties": "tied"}
#: Property -> (case dimensions, engine replaced, defect, expected failure).
WRONG_ENGINES = {
    "graph_expected": (
        UNTIED,
        "diskgraph",
        "fanout_plus_one",
        "exceed fanout|most recent|layer-by-layer",
    ),
    "graph_inference_exact": (
        UNTIED,
        "diskgraph",
        "future_after_reopen",
        "future rows changed diskgraph's graphs",
    ),
    "graph_point_in_time": (UNTIED, "diskgraph", "future_visible", "after the cutoff"),
    "graph_structure": (
        {"schema_shape.graph_path": "one_hop"},
        "diskgraph",
        "fanout_plus_one",
        "exceed fanout",
    ),
    "graph_edge_uniqueness": ({}, "diskgraph", "repeated_edges", "edges repeat within a sample"),
    "graph_determinism": (TIED, "diskgraph", "unstable_ties", "the same index again changed"),
    "graph_tie_invariance": (
        UNTIED,
        "diskgraph",
        "row_order_ties",
        "row and request order.*changed",
    ),
    "sequence_expected": ({}, "diskgraph", "off_by_one", "fixture answers changed sequences"),
    "sequence_point_in_time": (
        {"contract.event_delta_unit": "minutes"},
        "diskgraph",
        "future_visible",
        "after the cutoff",
    ),
    "sequence_permutation": (
        {},
        "diskgraph",
        "first_order_dropped",
        "original|row and request order",
    ),
}


def _tied_sequence_fixture():
    from conformance.sequence_cases import sequence_contract
    from kumo.pytorch.api import SampleRequest

    template = sequence_batch(0, "diskgraph", "sequence_expected")[0]
    dims = {
        **template.dimensions,
        "contract.event_order": "ascending",
        "contract.event_limit": "limited",
        "contract.event_max": "limited",
        "contract.event_keep": "first",
        "contract.event_delta_unit": "seconds",
    }
    case = replace(
        template,
        contract=sequence_contract(dims),
        requests=(SampleRequest("a", 1, 9),),
        tables={
            "users": [{"user_id": 1}],
            "orders": [
                {"order_id": i, "user_id": 1, "amount": float(i), "at": t}
                for i, t in ((1, 7), (2, 8), (3, 8), (4, 8), (5, 10))
            ],
            "clicks": [{"click_id": 10, "user_id": 1, "amount": 10.0, "at": 8}],
        },
    )

    def event(time, source, identity):
        return (
            time,
            source,
            (("amount", True, float(identity)), ("delta", True, 9 - time), ("id", True, identity)),
        )

    answer = {
        "a": {
            "merged": (event(7, "orders", 1), event(8, "orders", 2), event(8, "clicks", 10)),
            "pushed": (event(7, "recent", 1), event(8, "recent", 2)),
        }
    }
    return case, answer, event


MORE_DEFECTS = (
    ("graph_structure", "reversed_edges", "break orientation"),
    ("graph_structure", "older_on_ties", "recent kept other than the most recent"),
    ("graph_expected", "older_on_ties", "recent kept other than the most recent|layer-by-layer"),
    ("graph_point_in_time", "late_after_shops", "after the cutoff"),
    ("sequence_point_in_time", "delta_in_seconds", "wrong time deltas"),
    ("sequence_permutation", "unstable_rows", "original|a second preparation"),
)


#: Defects that need other case dimensions than their property's row above.
DEFECT_CASES = {
    ("graph_structure", "older_on_ties"): {**TIED, "schema_shape.graph_path": "one_hop"},
    ("graph_expected", "older_on_ties"): {**TIED, "schema_shape.graph_path": "one_hop"},
    ("graph_point_in_time", "late_after_shops"): {
        "schema_shape.graph_path": "fan_in",
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
            for case in sequence_batch(0, "diskgraph", name)
            if all(case.dimensions[key] == value for key, value in dimensions.items())
        )
        module = "conformance.sequence_cases"
    with patch(f"{module}.bound_tables", _binding(Wrong(defect), replaces)):
        PROPERTIES[name].check(case, replaces, 0)


#: Tied samples' later hops deleted, or dated after the cutoff in the
#: run with future rows: first-hop clocks stay the same, the layer-by-layer expansion doesn't.
LATER_HOPS = (
    ("graph_structure", "fan_in", "diskgraph", "tied_second_hop_dropped", EXPANSION),
    (
        "graph_expected",
        "fan_in",
        "diskgraph",
        "tied_second_hop_dropped",
        EXPANSION,
    ),
    (
        "graph_point_in_time",
        "fan_in",
        "diskgraph",
        "late_beyond_tied_first_hop",
        f"future rows: with future rows: {EXPANSION}",
    ),
)


@pytest.mark.parametrize("name,path,replaces,defect,expected", LATER_HOPS)
def test_tied_samples_are_checked_at_every_layer(name, path, replaces, defect, expected):
    case = next(
        case
        for case in graph_batch(0, "diskgraph", name)
        if case.dimensions["schema_shape.graph_path"] == path
        and case.dimensions["time.graph_ties"] == "tied"
    )
    assert case.dimensions["time.graph_ties"] == "tied"
    # The generator deliberately places fanout-boundary ties at a0/a1. Other
    # roots must not satisfy this probe by failing an untied expansion instead.
    case = replace(case, requests=case.requests[:2])
    fanout = int(case.dimensions["contract.graph_fanout"])
    for request in case.requests:
        clocks = sorted(
            (
                row["at"]
                for row in case.tables["events"]
                if row["user_id"] == request.entity_id and row["at"] <= request.as_of_time
            ),
            reverse=True,
        )
        assert clocks[fanout - 1] == clocks[fanout]
    with patch("conformance.graph_cases.bound_tables", _binding(Wrong("none"), replaces)):
        PROPERTIES[name].check(case, replaces, 0)
    stand_in = Wrong(defect)
    with patch("conformance.graph_cases.bound_tables", _binding(stand_in, replaces)):
        with pytest.raises(
            PropertyFailure,
            match=re.escape(expected),
        ):
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


@pytest.mark.parametrize("engine", ("diskgraph",))
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


@pytest.mark.parametrize("engine", ("diskgraph",))
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


@pytest.mark.parametrize("case", graph_batch(0, "diskgraph", "graph_structure"), ids=lambda c: c.id)
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
    relations = {r.name: r for r in case.contract.schema.relations}
    shop = relations.get("event_shop") or relations["shop_event"]
    assert (shop.left_table == "events") == (dims["schema_shape.graph_orientation"] == "aligned")
    users = relations.get("user_events") or relations["event_user"]
    assert (users.left_table == "users") == (dims["contract.graph_direction"] == "forward")


@pytest.mark.parametrize(
    "case", sequence_batch(0, "diskgraph", "sequence_permutation"), ids=lambda c: c.id
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
        with bound_tables(
            contract, tables, "diskgraph", refusal=refusal, parts=parts, types=types
        ) as (dataset, engine):
            mutant = Wrong(defect)
            mutant.index = engine.index
            mutant.num_shards = shards
            yield dataset, mutant

    # This self-test isolates settings propagation; the full generated corpus
    # remains exercised by the ordinary sequence obligations.
    base = next(
        case
        for case in sequence_batch(0, "diskgraph", "sequence_expected")
        if case.dimensions["time.ties"] == "distinct"
    )
    cases = [
        replace(base, dimensions={**base.dimensions, "execution.num_shards": shards})
        for shards in (1, 3)
    ]
    with (
        patch("conformance.sequence_cases.bound_tables", binding),
        patch.dict(
            PROPERTIES,
            sequence_expected=replace(
                PROPERTIES["sequence_expected"], cases=lambda seed, engine: cases
            ),
        ),
    ):
        if not dependent:
            run_obligation("sequence_expected", "diskgraph", 0, tmp_path)
            return
        with pytest.raises(PropertyFailure, match="fixture answers changed sequences"):
            run_obligation("sequence_expected", "diskgraph", 0, tmp_path)


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
