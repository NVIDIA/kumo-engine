# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated graph neighborhoods and the properties of G-GRAPH-001.

Cases are `GeneratedCase` values, so repro files, known gaps and replay work as
they do for aggregates. Graphs are read from real batches and compared per
sample: node IDs per table and (left ID, right ID) edges per relation.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

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
from conformance.registry import GRAPH_DIMENSIONS, DimensionValue
from kumo.pytorch.api import (
    Batching,
    BatchSpec,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DiskGraphEngine,
    DType,
    EntitySpec,
    Graph,
    GraphLayer,
    GraphSample,
    InMemoryEngine,
    Input,
    NeighborhoodSource,
    NeighborhoodSpec,
    RelationSpec,
    SampleRequest,
    SampleSpec,
    TableSpec,
    Task,
    check,
    prepare,
)

Key = int | str
Edge = tuple[Key, Key]
Hop = tuple[str, str]

BASE = 1_700_000_000
SPAN = 1_200
GRID = (400, 700, 1_000)
NEIGHBORHOOD = "g"
#: Users (and shops), anchors, and mean random events per user.
GRAPH_SCALES: Mapping[Scale, tuple[int, int, int]] = {"XS": (6, 8, 3), "S": (12, 16, 6)}
CLOCKS: Mapping[str, str] = {"users": "joined", "events": "at", "shops": "opened"}
#: Each path is layers of (from table, to table) hops out of a users root.
PATHS: Mapping[str, tuple[tuple[Hop, ...], ...]] = {
    "one_hop": ((("users", "events"),),),
    "many_to_one": ((("users", "events"),), (("events", "shops"),)),
    "wide": ((("users", "events"), ("users", "shops")), (("events", "shops"),)),
    "three_hop": ((("users", "events"),), (("events", "shops"),), (("shops", "users"),)),
    "wide_back": ((("users", "events"),), (("events", "users"), ("events", "shops"))),
    "fan_in": ((("users", "events"),), (("events", "shops"),), (("shops", "events"),)),
    "back_and_forth": ((("users", "events"),), (("events", "users"),), (("users", "events"),)),
}
KEY = DType("int64")
CLOCK = DType("timestamp", unit="s", timezone="UTC")
#: Paths that traverse some relation again in a later layer.
REVISITING = ("wide_back", "fan_in", "back_and_forth")


@dataclass(frozen=True)
class SampleGraph:
    """One sample's neighborhood, in output order (repeats kept)."""

    nodes: Mapping[str, tuple[Key, ...]]
    edges: Mapping[str, tuple[Edge, ...]]
    #: Each relation's output (source type, destination type).
    ends: Mapping[str, Hop]

    def distinct(self) -> tuple[dict[str, frozenset[Key]], dict[str, frozenset[Edge]]]:
        return (
            {table: frozenset(ids) for table, ids in self.nodes.items()},
            {name: frozenset(pairs) for name, pairs in self.edges.items()},
        )


GraphAnswer = dict[str, SampleGraph]


# -- Contracts ---------------------------------------------------------------


def graph_schema(dims: Mapping[str, DimensionValue]) -> DataSchema:
    user_events = (
        RelationSpec("user_events", "users", "user_id", "events", "user_id")
        if dims["contract.graph_direction"] == "forward"
        else RelationSpec("event_user", "events", "user_id", "users", "user_id")
    )
    event_shops = (
        RelationSpec("event_shop", "events", "shop_id", "shops", "shop_id")
        if dims["schema_shape.graph_orientation"] == "aligned"
        else RelationSpec("shop_event", "shops", "shop_id", "events", "shop_id")
    )
    return DataSchema(
        tables=(
            TableSpec("users", "user_id", {"user_id": KEY, "joined": CLOCK}, time_column="joined"),
            TableSpec(
                "events",
                "event_id",
                {"event_id": KEY, "user_id": KEY, "shop_id": KEY, "at": CLOCK},
                time_column="at",
            ),
            TableSpec(
                "shops",
                "shop_id",
                {"shop_id": KEY, "owner_id": KEY, "opened": CLOCK},
                time_column="opened",
            ),
        ),
        relations=(
            user_events,
            event_shops,
            RelationSpec("shop_owner", "shops", "owner_id", "users", "user_id"),
        ),
    )


def traversal(schema: DataSchema, hop: Hop, fanout: int, static: bool) -> GraphSample:
    relation = next(r for r in schema.relations if {r.left_table, r.right_table} == set(hop))
    return GraphSample(
        relation.name,
        fanout,
        direction="forward" if relation.left_table == hop[0] else "reverse",
        strategy="recent",
        temporal="static" if static and hop == ("users", "events") else "inherit",
    )


def graph_contract(
    dims: Mapping[str, DimensionValue], *, static: bool | None = None
) -> DataLoaderSpec:
    schema = graph_schema(dims)
    static = dims["contract.graph_temporal"] == "static" if static is None else static
    fanout = int(dims["contract.graph_fanout"])
    layers = tuple(
        GraphLayer(tuple(traversal(schema, hop, fanout, static) for hop in layer))
        for layer in PATHS[str(dims["schema_shape.graph_path"])]
    )
    return DataLoaderSpec(
        schema,
        SampleSpec(
            EntitySpec("users", entity_type="user"),
            neighborhoods={NEIGHBORHOOD: NeighborhoodSpec(layers)},
        ),
        BatchSpec(
            inputs={NEIGHBORHOOD: Input(NeighborhoodSource(NEIGHBORHOOD), Graph(), None)},
            batching=Batching(max_samples=8),
        ),
    )


def traversals(contract: DataLoaderSpec) -> list[GraphSample]:
    neighborhood = contract.sample.neighborhoods[NEIGHBORHOOD]
    return [sample for layer in neighborhood.layers for sample in layer.samples]


def _hop(schema: DataSchema, sample: GraphSample) -> tuple[RelationSpec, str, str, str, str]:
    """(relation, from table, from key, to table, to key) of one traversal."""

    relation = next(r for r in schema.relations if r.name == sample.relation)
    left = (relation.left_table, relation.left_key)
    right = (relation.right_table, relation.right_key)
    source, target = (left, right) if sample.direction == "forward" else (right, left)
    return relation, *source, *target


# -- Rows --------------------------------------------------------------------


class _Events:
    """Event rows; untied cases keep every candidate group (per user, per shop) dated apart."""

    def __init__(self, rng: random.Random, shops: int, tied: bool):
        self.rng, self.shops, self.tied = rng, shops, tied
        self.rows: list[Row] = []
        self.taken: set[tuple[str, int, int]] = set()

    def add(self, user: int, time: int, shop: int | None = None) -> bool:
        if not self.tied and ("user", user, time) in self.taken:
            return False
        choices = (
            [shop] if shop is not None else self.rng.sample(range(1, self.shops + 1), self.shops)
        )
        for candidate in choices:
            if self.tied or ("shop", candidate, time) not in self.taken:
                self.taken |= {("user", user, time), ("shop", candidate, time)}
                row: Row = {"event_id": len(self.rows) + 1, "user_id": user}
                self.rows.append(row | {"shop_id": candidate, "at": time})
                return True
        return False

    def eligible(self, user: int, cutoff: int) -> int:
        return sum(row["user_id"] == user and int(str(row["at"])) <= cutoff for row in self.rows)


def generate_graph_case(
    seed: int, index: int, dims: Mapping[str, DimensionValue], scale: Scale
) -> GeneratedCase:
    rng = random.Random(f"graph:{seed}:{index}")
    count, anchor_count, per_user = GRAPH_SCALES[scale]
    users = list(range(1, count + 1))
    fanout = int(dims["contract.graph_fanout"])
    tied = dims["time.graph_ties"] == "tied"
    grid = tuple(BASE + offset for offset in GRID)

    anchors: list[tuple[int, int]] = []
    for position in range(anchor_count):
        user = users[position % count]
        cutoff = grid[1]
        if dims["time.cutoffs"] == "mixed":
            cutoff = grid[(position // count + user) % len(grid)]
            cutoff += rng.randrange(0, 50) if position >= count else 0
        anchors.append((user, cutoff))

    events = _Events(rng, count, tied)
    for user in users:
        for _ in range(rng.randint(1, 2 * per_user)):
            events.add(user, rng.randrange(BASE, BASE + SPAN))
    for position, (user, cutoff) in enumerate(anchors):
        for time in (cutoff, cutoff + 1, cutoff - 1):
            events.add(user, time)
        if tied and position < 2:
            # More candidates than the fanout share the most recent eligible second.
            shop = rng.randint(1, count)
            for _ in range(fanout + 1):
                events.add(user, cutoff, shop)
        while events.eligible(user, cutoff) <= fanout:
            events.add(user, rng.randrange(BASE, cutoff - 1))

    # One user joins, and one shop opens, exactly at an early cutoff.
    tables: Tables = {
        "users": [
            {"user_id": user, "joined": grid[0] if user == count - 1 else BASE - 9_000 - user}
            for user in users
        ],
        "events": events.rows,
        "shops": [
            {
                "shop_id": shop,
                "owner_id": rng.choice(users),
                "opened": grid[0] if shop == count else BASE - 5_000 - shop,
            }
            for shop in range(1, count + 1)
        ],
    }
    requests = tuple(
        SampleRequest(f"a{position}", entity_id=user, as_of_time=cutoff)
        for position, (user, cutoff) in enumerate(anchors)
    )
    return GeneratedCase(
        f"g{seed}-c{index}", seed, scale, dict(dims), graph_contract(dims), tables, requests
    )


def graph_batch(seed: int, engine: str, property_name: str) -> list[GeneratedCase]:
    scale: Scale = "XS" if engine == "in_memory" else "S"
    rows = covering_rows(seed, engine, GRAPH_DIMENSIONS, property_name)
    return [generate_graph_case(seed, index, row, scale) for index, row in rows]


# -- Running a case ----------------------------------------------------------


@contextmanager
def _bound(
    contract: DataLoaderSpec,
    tables: Tables,
    engine: str,
    *,
    parts: int = 1,
    shards: int | None = None,
) -> Iterator[tuple[Dataset, InMemoryEngine | DiskGraphEngine]]:
    """The case's binding; ``shards`` overrides the case's DiskGraph shard count only."""

    with bound_tables(contract, tables, engine, parts=parts) as (dataset, bound):
        if shards is not None and isinstance(bound, DiskGraphEngine):
            bound = DiskGraphEngine(
                index=bound.index, num_threads=bound.num_threads, num_shards=shards
            )
        yield dataset, bound


def graph_answer(
    contract: DataLoaderSpec,
    tables: Tables,
    requests: Sequence[SampleRequest],
    engine: str,
    *,
    parts: int = 1,
    shards: int | None = None,
) -> GraphAnswer:
    """Each sample's graph, from real batches through the public path."""

    with _bound(contract, tables, engine, parts=parts, shards=shards) as (dataset, bound):
        issues = check(contract, dataset, bound)
        require(not issues, f"{engine} refused a generated graph contract: {issues}")
        with prepare(contract, dataset, bound) as plan:
            return collect_graphs(plan.materialize(requests), requests)


def collect_graphs(task: Task, requests: Sequence[SampleRequest]) -> GraphAnswer:
    result: GraphAnswer = {}
    for batch in task.batches(requests):
        graph = batch.inputs[NEIGHBORHOOD]
        sample_ids = [str(sample_id) for sample_id in batch.metadata["sample_ids"]]
        require(not any(batch.metadata["is_padding"].tolist()), "padding in a request batch")
        ids = {table: list(_ids(node["ids"])) for table, node in graph["nodes"].items()}
        owners = {table: node["batch"].tolist() for table, node in graph["nodes"].items()}
        nodes: dict[int, dict[str, list[Key]]] = {i: {} for i in range(len(sample_ids))}
        edges: dict[int, dict[str, list[Edge]]] = {i: {} for i in range(len(sample_ids))}
        for table in ids:
            for key, owner in zip(ids[table], owners[table], strict=True):
                nodes[owner].setdefault(table, []).append(key)
        ends = {
            name: (e["source_type"], e["destination_type"]) for name, e in graph["edges"].items()
        }
        for name, edge in graph["edges"].items():
            left, right = ends[name]
            for source, target in edge["index"].t().tolist():
                inside = 0 <= source < len(ids[left]) and 0 <= target < len(ids[right])
                same = inside and owners[left][source] == owners[right][target]
                require(
                    same, f"{name}: edge ({source}, {target}) does not join two nodes of one sample"
                )
                pair = (ids[left][source], ids[right][target])
                edges[owners[left][source]].setdefault(name, []).append(pair)
        for position, sample_id in enumerate(sample_ids):
            require(sample_id not in result, f"duplicate sample {sample_id}")
            result[sample_id] = SampleGraph(
                {table: tuple(keys) for table, keys in nodes[position].items()},
                {name: tuple(pairs) for name, pairs in edges[position].items()},
                ends,
            )
    expected = {str(request.sample_id) for request in requests}
    require(result.keys() == expected, f"samples {sorted(result)} != requests {sorted(expected)}")
    return result


def _ids(values: torch.Tensor | tuple[Key, ...]) -> list[Key]:
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)


#: One base run per (binding, engine, case) serves every property; determinism reruns.
_RUNS: dict[tuple[Callable[..., object], str, str], GraphAnswer] = {}


def _base(case: GeneratedCase, engine: str) -> GraphAnswer:
    key = (bound_tables, engine, case_digest(case))
    if key not in _RUNS:
        _RUNS[key] = graph_answer(case.contract, case.tables, case.requests, engine)
    return _RUNS[key]


def _same_graphs(before: GraphAnswer, after: GraphAnswer, label: str) -> None:
    differences = _differences(before, after)
    require(not differences, f"{label} changed graphs:\n  " + "\n  ".join(differences[:4]))


def _differences(before: GraphAnswer, after: GraphAnswer) -> list[str]:
    """Samples whose node sets or distinct edges differ."""

    return [
        f"{key}: {_describe(before.get(key))} != {_describe(after.get(key))}"
        for key in sorted(before.keys() | after.keys())
        if (before.get(key) is None) != (after.get(key) is None)
        or before[key].distinct() != after[key].distinct()
    ]


def _describe(graph: SampleGraph | None) -> str:
    if graph is None:
        return "missing"
    nodes, edges = graph.distinct()
    return str({name: sorted(values, key=repr) for name, values in {**nodes, **edges}.items()})


PRIMARY_KEYS: Mapping[str, str] = {"users": "user_id", "events": "event_id", "shops": "shop_id"}


def _rows(tables: Tables) -> dict[str, dict[Scalar, Row]]:
    return {name: {row[PRIMARY_KEYS[name]]: row for row in rows} for name, rows in tables.items()}


def _clock(row: Row, table: str) -> int:
    return int(str(row[CLOCKS[table]]))


def candidate_clocks(case: GeneratedCase, sample: GraphSample, key: Key, cutoff: int) -> list[int]:
    """Clocks of the rows one traversal from row ``key`` may keep, newest first (rule 1)."""

    _, source, source_key, target, target_key = _hop(case.contract.schema, sample)
    value = _rows(case.tables)[source][key][source_key]
    timed = sample.temporal == "inherit"
    clocks = [
        _clock(row, target)
        for row in case.tables[target]
        if row[target_key] == value and not (timed and _clock(row, target) > cutoff)
    ]
    return sorted(clocks, reverse=True)


def first_hops(
    case: GeneratedCase, graph: SampleGraph, entity: Key, tables: Tables | None = None
) -> dict[str, list[int]]:
    """Per first-layer traversal, the sorted clocks of the root's distinct neighbors.

    In every generated path, later layers add no other root edge of these relations.
    """

    rows = _rows(case.tables if tables is None else tables)
    layer = case.contract.sample.neighborhoods[NEIGHBORHOOD].layers[0]
    kept: dict[str, list[int]] = {}
    for sample in layer.samples:
        relation, source, _, target, _ = _hop(case.contract.schema, sample)
        root = 0 if relation.left_table == source else 1
        neighbors = {
            pair[1 - root] for pair in graph.edges.get(relation.name, ()) if pair[root] == entity
        }
        kept[f"{sample.relation}.{sample.direction}"] = sorted(
            _clock(rows[target][key], target) for key in neighbors
        )
    return kept


def tied_samples(case: GeneratedCase, graphs: GraphAnswer) -> set[str]:
    """Samples where a traversal from one of their nodes has a tie at the fanout limit."""

    cutoffs = {str(r.sample_id): int(r.as_of_time) for r in case.requests}
    tied = set()
    for sample_id, graph in graphs.items():
        for sample in traversals(case.contract):
            source = _hop(case.contract.schema, sample)[1]
            for key in set(graph.nodes.get(source, ())):
                clocks = candidate_clocks(case, sample, key, cutoffs[sample_id])
                if (
                    len(clocks) > sample.fanout
                    and clocks[sample.fanout - 1] == clocks[sample.fanout]
                ):
                    tied.add(sample_id)
    return tied


# -- Layer-wise expansion ----------------------------------------------------

Node = tuple[str, Key]


class _Candidates:
    """Rule 1's candidates of one traversal from one row, over indexed tables."""

    def __init__(self, schema: DataSchema, tables: Tables):
        self.schema, self.tables, self.rows = schema, tables, _rows(tables)
        self.joins: dict[tuple[str, str], dict[Scalar, list[Row]]] = {}

    def of(self, sample: GraphSample, key: Key, cutoff: int) -> dict[Key, int]:
        """Each candidate's key and clock."""

        _, source, source_key, target, target_key = _hop(self.schema, sample)
        if (target, target_key) not in self.joins:
            index: dict[Scalar, list[Row]] = {}
            for row in self.tables[target]:
                index.setdefault(row[target_key], []).append(row)
            self.joins[(target, target_key)] = index
        timed = sample.temporal == "inherit"
        found: dict[Key, int] = {}
        for row in self.joins[(target, target_key)].get(self.rows[source][key][source_key], ()):
            if not (timed and _clock(row, target) > cutoff):
                found[_key(row[PRIMARY_KEYS[target]])] = _clock(row, target)
        return found


def _key(value: Scalar) -> Key:
    require(isinstance(value, int | str), f"generated key {value!r} is not an int or str")
    return value if isinstance(value, int | str) else str(value)


def expansion_problems(
    contract: DataLoaderSpec,
    candidates: _Candidates,
    graph: SampleGraph,
    entity: Key,
    cutoff: int,
) -> list[str]:
    """Where one sample's graph is not a layer-by-layer expansion of its root (rule 1).

    The expansion is rebuilt from the graph itself: layer 0's frontier is the root;
    a frontier node's kept neighbors are the distinct edges of the traversal's
    relation at it, and the next frontier is every kept neighbor, including one reached
    before (the reference expands it again). At each frontier node the kept
    neighbors must be candidates (keys join; dated at or before the cutoff unless the
    traversal is static), and their sorted clocks must be the ``fanout`` most recent
    candidates' clocks: count and recency, with the tied neighbor left free.
    Nodes are the root and the ends of edges, and every edge leaves the frontier.

    A relation traversed more than once (``REVISITING`` paths) doesn't say which
    traversal made an edge, so for it the check is weaker: the edges at a frontier
    node must include the ``fanout`` most recent candidates' clocks (more are
    allowed), every edge must be a candidate no older than the fanout limit of some
    traversal of it from its frontier, and a neighbor tied at the limit is a frontier
    node of the next layer only when the tie is resolved by the count kept.
    """

    neighborhood = contract.sample.neighborhoods[NEIGHBORHOOD]
    uses = Counter(sample.relation for sample in traversals(contract))
    root: Node = (contract.sample.entity.table, entity)
    certain, possible = {root}, {root}
    explained: dict[str, set[Edge]] = {}
    problems: list[str] = []
    for depth, layer in enumerate(neighborhood.layers):
        next_certain: set[Node] = set()
        next_possible: set[Node] = set()
        for sample in layer.samples:
            relation, source, _, target, _ = _hop(contract.schema, sample)
            side = 0 if relation.left_table == source else 1
            adjacent: dict[Key, set[Key]] = {}
            for pair in set(graph.edges.get(relation.name, ())):
                adjacent.setdefault(pair[side], set()).add(pair[1 - side])
            shared = uses[relation.name] > 1
            frontier = {key for table, key in possible if table == source}
            hop = f"layer {depth} {relation.name}.{sample.direction}"
            if not shared and (strays := sorted(adjacent.keys() - frontier, key=repr)):
                problems.append(f"{hop}: edges leave {source} {strays}, outside the frontier")
            for key in sorted(frontier, key=repr):
                eligible = candidates.of(sample, key, cutoff)
                top = sorted(eligible.values(), reverse=True)[: sample.fanout]
                limit = top[-1] if len(eligible) > sample.fanout else None
                kept = adjacent.get(key, set())
                sure = (source, key) in certain
                where = f"{hop} from {source} {key}"
                if not shared:
                    if strays := sorted(kept - eligible.keys(), key=repr):
                        problems.append(f"{where}: kept {target} {strays}, not candidates")
                    clocks = sorted(eligible[n] for n in kept if n in eligible)
                    if (sure or kept) and clocks != sorted(top):
                        problems.append(
                            f"{where}: kept clocks {clocks}, "
                            f"the {sample.fanout} most recent candidates {sorted(top)}"
                        )
                    reached = {(target, n) for n in kept if n in eligible}
                    next_possible |= reached
                    next_certain |= reached if sure or kept else set()
                    continue
                chosen = {
                    n for n in kept if n in eligible and (limit is None or eligible[n] >= limit)
                }
                explained.setdefault(relation.name, set()).update(
                    (key, n) if side == 0 else (n, key) for n in chosen
                )
                missing = Counter(top) - Counter(eligible[n] for n in chosen)
                if sure and missing:
                    problems.append(
                        f"{where}: no neighbor dated {sorted(missing.elements())}, "
                        f"among the {sample.fanout} most recent candidates {sorted(top)}"
                    )
                next_possible |= {(target, n) for n in chosen}
                at_limit = sum(eligible[n] == limit for n in chosen)
                resolved = limit is None or at_limit == top.count(limit)
                if sure:
                    next_certain |= {
                        (target, n) for n in chosen if resolved or eligible[n] != limit
                    }
        certain, possible = next_certain, next_possible
    ends = {root}
    for name, pairs in graph.edges.items():
        left, right = graph.ends[name]
        ends |= {(left, a) for a, _ in pairs} | {(right, b) for _, b in pairs}
        if uses[name] > 1 and (stray := sorted(set(pairs) - explained.get(name, set()), key=repr)):
            problems.append(f"{name}: edges {stray[:4]} no traversal from the frontier keeps")
    nodes = {(table, key) for table, keys in graph.nodes.items() for key in keys}
    if loose := sorted(nodes - ends, key=repr):
        problems.append(f"nodes {loose[:4]} are neither the root nor joined by an edge")
    return problems


def _expansion(case: GeneratedCase, graphs: GraphAnswer, tables: Tables | None = None) -> list[str]:
    """Every sample's expansion problems, against ``tables`` (the case's by default)."""

    candidates = _Candidates(case.contract.schema, case.tables if tables is None else tables)
    problems: list[str] = []
    for request in case.requests:
        sample_id = str(request.sample_id)
        if sample_id in graphs:
            found = expansion_problems(
                case.contract,
                candidates,
                graphs[sample_id],
                _key(request.entity_id),
                int(request.as_of_time),
            )
            problems += [f"{sample_id}: {problem}" for problem in found]
    return problems


EXPANSION = "graphs are not layer-by-layer expansions of their roots (rule 1)"


# -- Properties --------------------------------------------------------------


def _same_up_to_ties(
    case: GeneratedCase,
    before: GraphAnswer,
    after: GraphAnswer,
    label: str,
    tables: Tables | None = None,
    sides: tuple[str, str] = ("before", "after"),
) -> None:
    """Untied samples compare directly. Which tied neighbor is kept at a fanout limit
    is unspecified (G-GRAPH-001 rule 2; an allowed discrepancy, see
    conformance/README.md, Differential testing),
    so a tied sample compares on the clocks of the root's
    first-layer neighbors, which rule 1 fixes. Then each side must be a layer-by-layer
    expansion of its roots: ``before`` over the case's tables, ``after`` over ``tables``.
    """

    tied = tied_samples(case, before)
    untied = (before.keys() | after.keys()) - tied
    _same_graphs(
        {k: before[k] for k in untied if k in before},
        {k: after[k] for k in untied if k in after},
        label,
    )
    entities = {str(r.sample_id): r.entity_id for r in case.requests}
    differences = [
        f"{key}: {hops} != {other}"
        for key in sorted(tied)
        if (hops := first_hops(case, before[key], entities[key], tables))
        != (other := key in after and first_hops(case, after[key], entities[key], tables))
    ]
    require(
        not differences,
        f"{label} changed tied samples' first-hop clocks:\n  " + "\n  ".join(differences[:4]),
    )
    for side, graphs, data in ((sides[0], before, None), (sides[1], after, tables)):
        problems = _expansion(case, graphs, data)
        require(not problems, f"{label}: {side}: {EXPANSION}:\n  " + "\n  ".join(problems[:4]))


def graph_differential(case: GeneratedCase, engine: str, position: int) -> None:
    """DiskGraph has the reference's node sets and distinct edges per sample, up to
    which tied neighbors it keeps at a fanout limit; both are layer-by-layer expansions.
    """

    _same_up_to_ties(
        case,
        _base(case, "in_memory"),
        _base(case, engine),
        f"{engine} vs reference",
        sides=("reference", engine),
    )


def with_future_graph_rows(case: GeneratedCase, rng: random.Random) -> tuple[Tables, set[str]]:
    """Events and owned shops dated after an entity's cutoff; the anchors they can't reach.

    A static case gets no future events: a static traversal may reach them.
    """

    tables = {name: [dict(row) for row in rows] for name, rows in case.tables.items()}
    static = case.dimensions["contract.graph_temporal"] == "static"
    cutoffs: dict[int, list[int]] = {}
    for request in case.requests:
        cutoffs.setdefault(int(request.entity_id), []).append(int(request.as_of_time))
    dated: list[int] = []
    for entity, times in sorted(cutoffs.items()):
        threshold = rng.choice(times)
        shop = len(tables["shops"]) + 1
        tables["shops"].append({"shop_id": shop, "owner_id": entity, "opened": threshold + 1})
        dated.append(threshold + 1)
        if not static:
            for time in (threshold + 1, threshold + rng.randint(2, 100)):
                event = {"event_id": len(tables["events"]) + 1, "user_id": entity}
                tables["events"].append(event | {"shop_id": rng.choice([shop, 1]), "at": time})
    # Fan-in reaches other entities' rows, so guard only anchors before every new row.
    guarded = {str(r.sample_id) for r in case.requests if int(r.as_of_time) < min(dated)}
    return tables, guarded


def graph_point_in_time(case: GeneratedCase, engine: str, position: int) -> None:
    """No neighbor is dated after its cutoff (rule 1); future rows change nothing.

    Events reached through the static users-events relation are exempt from the
    date check; events reached any other way are not.
    """

    base = _base(case, engine)
    rows = _rows(case.tables)
    relations = {relation.name: relation for relation in case.contract.schema.relations}
    static = {
        sample.relation for sample in traversals(case.contract) if sample.temporal == "static"
    }
    cutoffs = {str(r.sample_id): (int(r.entity_id), int(r.as_of_time)) for r in case.requests}
    late = []
    for sample_id, graph in sorted(base.items()):
        entity, cutoff = cutoffs[sample_id]
        unfiltered = {
            pair[0 if relations[name].left_table == "events" else 1]
            for name in static
            for pair in graph.edges.get(name, ())
        }
        for table, keys in graph.nodes.items():
            for key in keys:
                if table == "events" and key in unfiltered:
                    continue
                dated = int(str(rows[table][key][CLOCKS[table]]))
                if dated > cutoff and (table, key) != ("users", entity):
                    late.append(f"{sample_id}: {table} {key} dated {dated} > cutoff {cutoff}")
    require(not late, "nodes after the cutoff:\n  " + "\n  ".join(late[:6]))
    tables, guarded = with_future_graph_rows(case, random.Random(f"{case.id}:future"))
    require(bool(guarded), "no anchor is guarded by the future rows")
    after = graph_answer(case.contract, tables, case.requests, engine)
    _same_up_to_ties(
        case,
        {k: base[k] for k in guarded},
        {k: after[k] for k in guarded},
        "future rows",
        tables,
        sides=("without future rows", "with future rows"),
    )


def graph_structure(case: GeneratedCase, engine: str, position: int) -> None:
    """Fanout bound and recency (rule 1), unique nodes and edge orientation (rule 4),
    and each graph a layer-by-layer expansion of its root (``expansion_problems``).

    The fanout bound is exact for relations traversed in one direction only; a
    relation traversed both ways also gains edges from the other side. "recent"
    keeps the root's ``fanout`` most recent eligible neighbors per first-layer hop.
    """

    contract = case.contract
    graphs = _base(case, engine)
    relations = {relation.name: relation for relation in contract.schema.relations}
    rows = _rows(case.tables)
    directions: dict[str, Counter[str]] = {}
    for sample in traversals(contract):
        directions.setdefault(sample.relation, Counter())[sample.direction] += 1
    for sample_id, graph in sorted(graphs.items()):
        for table, keys in graph.nodes.items():
            repeated = sorted(key for key, n in Counter(keys).items() if n > 1)
            require(not repeated, f"{sample_id}: {table} nodes {repeated} repeat in one sample")
        for name, pairs in graph.edges.items():
            relation = relations[name]
            declared = (relation.left_table, relation.right_table)
            require(
                graph.ends[name] == declared,
                f"{name}: output edges {graph.ends[name]} break orientation: declared {declared}",
            )
            for left, right in pairs:
                joined = (
                    rows[relation.left_table][left][relation.left_key]
                    == rows[relation.right_table][right][relation.right_key]
                )
                require(
                    joined,
                    f"{sample_id}: {name} edge ({relation.left_table} {left} -> "
                    f"{relation.right_table} {right}) breaks orientation: keys don't join",
                )
            if len(directions[name]) != 1:
                continue
            ((direction, times),) = directions[name].items()
            side = 0 if direction == "forward" else 1
            bound = int(case.dimensions["contract.graph_fanout"]) * times
            degree = Counter(pair[side] for pair in set(pairs))
            over = sorted((key, n) for key, n in degree.items() if n > bound)
            require(
                not over, f"{sample_id}: {name} {direction} neighbors {over} exceed fanout {bound}"
            )
    _most_recent(case, graphs)
    problems = _expansion(case, graphs)
    require(not problems, f"{EXPANSION}:\n  " + "\n  ".join(problems[:4]))
    if engine == "diskgraph":
        _static_refused(case)


def _most_recent(case: GeneratedCase, graphs: GraphAnswer) -> None:
    layer = case.contract.sample.neighborhoods[NEIGHBORHOOD].layers[0]
    stale = []
    for request in case.requests:
        sample_id, entity = str(request.sample_id), request.entity_id
        kept = first_hops(case, graphs[sample_id], entity)
        for sample in layer.samples:
            eligible = candidate_clocks(case, sample, entity, int(request.as_of_time))
            expected = sorted(eligible[: sample.fanout])
            hop = f"{sample.relation}.{sample.direction}"
            if kept[hop] != expected:
                stale.append(f"{sample_id}: {hop} kept {kept[hop]}, most recent {expected}")
    require(
        not stale, "recent kept other than the most recent neighbors:\n  " + "\n  ".join(stale[:4])
    )


def _static_refused(case: GeneratedCase) -> None:
    """The declared DiskGraph refusal of static traversals happens at check()."""

    contract = graph_contract(case.dimensions, static=True)
    with bound_tables(contract, case.tables, "diskgraph", refusal=True) as (dataset, bound):
        issues = check(contract, dataset, bound)
    require(
        bool(issues) and all(i.path.endswith(".temporal") and i.suggestion for i in issues),
        f"static traversal not refused at check(): {issues}",
    )


def graph_edge_uniqueness(case: GeneratedCase, engine: str, position: int) -> None:
    """An edge appears at most once per sample (rule 3)."""

    for sample_id, graph in sorted(_base(case, engine).items()):
        for name, pairs in graph.edges.items():
            repeated = sorted((pair, n) for pair, n in Counter(pairs).items() if n > 1)
            require(
                not repeated,
                f"{sample_id}: {name} edges repeat within a sample: "
                f"{len(pairs)} edges, {len(set(pairs))} distinct; first {repeated[:2]}",
            )


#: Samplings of the prepared index after the first.
RERUNS = 2


def graph_determinism(case: GeneratedCase, engine: str, position: int) -> None:
    """Sampling one prepared index again gives the same graphs, in the same order.

    Separately built indexes may lay rows out differently; that the graphs still
    follow from the data (rule 2) is graph_tie_invariance's check.
    """

    with _bound(case.contract, case.tables, engine) as (dataset, bound):
        with prepare(case.contract, dataset, bound) as plan:
            first = collect_graphs(plan.materialize(case.requests), case.requests)
            for _ in range(RERUNS):
                again = collect_graphs(plan.materialize(case.requests), case.requests)
                differences = [key for key in sorted(first) if first[key] != again.get(key)]
                require(
                    not differences,
                    f"sampling the same index again changed samples {differences[:6]}",
                )


def graph_tie_invariance(case: GeneratedCase, engine: str, position: int) -> None:
    """Row order, file splits and another shard count don't change the graphs (rule 2).

    Which tied neighbors are kept is unspecified but must follow from the data.
    """

    base = _base(case, engine)
    shuffled = permuted(case, random.Random(f"{case.id}:ties"))
    order = "row and request order"
    variants: dict[str, Callable[[], GraphAnswer]] = {order: lambda: _base(shuffled, engine)}
    if engine == "diskgraph":
        other = 1 if int(case.dimensions.get("execution.num_shards", 1)) == 3 else 3
        variants = {
            f"{order}, 2 files per table": lambda: graph_answer(
                shuffled.contract, shuffled.tables, shuffled.requests, engine, parts=2
            ),
            f"num_shards={other}": lambda: graph_answer(
                case.contract, case.tables, case.requests, engine, shards=other
            ),
        }
    failures = []
    for label, run in variants.items():
        differences = _differences(base, run())
        if differences:
            failures.append(f"{label} changed graphs:\n  " + "\n  ".join(differences[:3]))
    require(not failures, "\n".join(failures))


def _layout_deterministic(case: GeneratedCase) -> bool:
    """Whether fresh index builds of this case lay rows out the same way. With several
    shards and threads they don't, so tie choices vary (GAP-GRAPH-TIES-DG)."""

    dims = case.dimensions
    return (
        int(dims.get("execution.num_shards", 1)) == 1
        or int(dims.get("execution.num_threads", 1)) == 1
    )


def graph_inference_exact(case: GeneratedCase, engine: str, position: int) -> None:
    """What batch inference needs of graphs (G-INFERENCE-001): rows dated after every
    cutoff change none of the engine's graphs, tied neighbors included.

    Where index builds aren't deterministic, a sample with a tie at a fanout limit is
    left out: its tie choice follows the layout, the known gap GAP-GRAPH-TIES-DG.
    """

    base = _base(case, engine)
    tables, guarded = with_future_graph_rows(case, random.Random(f"{case.id}:future"))
    require(bool(guarded), "no anchor is guarded by the future rows")
    if not _layout_deterministic(case):
        guarded -= tied_samples(case, base)
    after = graph_answer(case.contract, tables, case.requests, engine)
    differences = _differences({k: base[k] for k in guarded}, {k: after[k] for k in guarded})
    require(
        not differences,
        f"future rows changed {engine}'s graphs:\n  " + "\n  ".join(differences[:3]),
    )


GRAPH_CHECKS = {
    "graph_differential": graph_differential,
    "graph_inference_exact": graph_inference_exact,
    "graph_point_in_time": graph_point_in_time,
    "graph_structure": graph_structure,
    "graph_edge_uniqueness": graph_edge_uniqueness,
    "graph_determinism": graph_determinism,
    "graph_tie_invariance": graph_tie_invariance,
}
