# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The relation routes a contract's sample reads rows along, from its entity's row.

A route is a sequence of hops; each hop follows one relation from the rows the
route has reached so far to the rows whose column equals theirs. Splitting a
dataset by root entity (`test.kumo_pytorch.reference_engine.gpu.split`) copies into each partition every
row its root rows reach along these routes, so that a sample built from one
partition reads the same rows as one built from the whole dataset.

The routes follow upstream Kumo PyTorch's own rules (``kumo.pytorch.serving_rows``,
which lists the rows serving needs for one request):

- a `Rows` path, a `Field` path, an aggregate's source and an event's or a
  target's rows each follow their relations in order; a relation is followed
  forward (left table to right table) when the route stands on its left
  table, and backward otherwise;
- an event's fields, a dated value's fields and a row target's fields start
  from the rows their `Rows` reached;
- a neighborhood's layer applies each `GraphSample` to every route of the
  previous layer that stands on the sample's starting table, in the sample's
  stated direction.

Time windows, filters, fanouts and limits narrow which of these rows a sample
keeps, per request; a split must hold every row any request could keep, so
they are ignored here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.contract.sample import SampleSpec
    from kumo.pytorch.contract.schema import DataSchema

__all__ = ["Hop", "Route", "candidate_routes", "route_table", "sample_routes"]


@dataclass(frozen=True)
class Hop:
    """Rows of `table` whose `column` equals `from_column` of the rows reached in `from_table`."""

    relation: str
    from_table: str
    from_column: str
    table: str
    column: str


Route = tuple[Hop, ...]


def sample_routes(schema: DataSchema, sample: SampleSpec) -> tuple[Route, ...]:
    """Every route the sample reads along from its entity's row, targets included, each once.

    A route's prefixes are read too; they are listed only when something ends there.
    Candidate sets start from their own entity: `candidate_routes`.
    """

    routes = _Routes(schema)
    entity = sample.entity.table
    routes.values(entity, (), sample.entity_values)
    for dated in sample.dated_entity_values.values():
        reached = routes.rows(entity, (), dated.rows)
        routes.field(entity, reached, dated.field)
        routes.values(entity, reached, dated.fields)
    for events in (sample.events, sample.current_events):
        for event in events.values():
            routes.values(entity, routes.rows(entity, (), event.rows), event.fields)
    for neighborhood in sample.neighborhoods.values():
        routes.neighborhood(entity, neighborhood.layers)
    routes.targets(entity, sample.targets)
    return tuple(routes.found)


def candidate_routes(schema: DataSchema, sample: SampleSpec) -> dict[str, tuple[str, tuple[Route, ...]]]:
    """Each candidate set's entity table and the routes its values and targets read from a candidate's row."""

    found = {}
    for name, candidates in sample.candidate_sets.items():
        routes = _Routes(schema)
        table = candidates.entity.table
        routes.values(table, (), candidates.values)
        routes.targets(table, candidates.targets)
        found[name] = (table, tuple(routes.found))
    return found


def route_table(root: str, route: Route) -> str:
    """The table a route stands on: its last hop's, or the root's when it has none."""

    return route[-1].table if route else root


class _Routes:
    def __init__(self, schema: DataSchema) -> None:
        self.relations = {relation.name: relation for relation in schema.relations}
        self.found: dict[Route, None] = {}

    def hop(self, root: str, route: Route, name: str, *, reverse: bool | None = None) -> Route:
        relation = self.relations.get(name)
        if relation is None:
            raise ValueError(f"the sample follows relation {name!r}, which the schema does not declare")
        current = route_table(root, route)
        if reverse is None:
            if current not in (relation.left_table, relation.right_table):
                raise ValueError(f"relation {name!r} cannot be followed from {current!r}")
            reverse = current != relation.left_table
        if reverse:
            hop = Hop(name, relation.right_table, relation.right_key, relation.left_table, relation.left_key)
        else:
            hop = Hop(name, relation.left_table, relation.left_key, relation.right_table, relation.right_key)
        return (*route, hop)

    def follow(self, root: str, route: Route, path: Iterable[str]) -> Route:
        start = route
        for name in path:
            route = self.hop(root, route, name)
        if route != start:
            self.found[route] = None
        return route

    def rows(self, root: str, route: Route, rows: Any) -> Route:
        return self.follow(root, route, rows.path)

    def field(self, root: str, route: Route, value: Any) -> None:
        from kumo.pytorch.contract.sample import Field

        if isinstance(value, Field):
            self.follow(root, route, value.path)

    def source(self, root: str, route: Route, source: Any) -> None:
        from kumo.pytorch.contract.sample import GroupBy

        while isinstance(source, GroupBy):
            source = source.source
        self.rows(root, route, source)

    def values(self, root: str, route: Route, values: Mapping[str, Any]) -> None:
        from kumo.pytorch.contract.sample import Aggregate, Compare

        for value in values.values():
            if isinstance(value, Aggregate):
                self.source(root, route, value.source)
            elif isinstance(value, Compare):
                self.source(root, route, value.left.source)
            else:
                self.field(root, route, value)

    def targets(self, root: str, targets: Mapping[str, Any]) -> None:
        from kumo.pytorch.contract.sample import Aggregate, Compare, RowTarget

        for target in targets.values():
            if isinstance(target, RowTarget):
                self.values(root, self.rows(root, (), target.rows), target.fields)
            elif isinstance(target, Compare):
                self.source(root, (), target.left.source)
            elif isinstance(target, Aggregate):
                self.source(root, (), target.source)

    def neighborhood(self, root: str, layers: Iterable[Any]) -> None:
        frontier: list[Route] = [()]
        for layer in layers:
            following: list[Route] = []
            for traversal in layer.samples:
                relation = self.relations.get(traversal.relation)
                if relation is None:
                    raise ValueError(f"the sample follows relation {traversal.relation!r}, "
                                     "which the schema does not declare")
                reverse = traversal.direction == "reverse"
                start = relation.right_table if reverse else relation.left_table
                for current in frontier:
                    if route_table(root, current) != start:
                        continue
                    route = self.hop(root, current, traversal.relation, reverse=reverse)
                    self.found[route] = None
                    if route not in following:
                        following.append(route)
            frontier = following
