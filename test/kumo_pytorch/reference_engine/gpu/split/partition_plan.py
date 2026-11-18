# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which tables follow the root entity, along which chains, and which routes a split serves."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from test.kumo_pytorch.reference_engine.gpu.sampling.paths import (
    Hop,
    candidate_routes,
    sample_routes,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.contract.sample import SampleSpec
    from kumo.pytorch.contract.schema import DataSchema
    from kumo.pytorch.dataset import Parquet


__all__ = ['PartitionPlan', 'Step', 'plan_partitions']


@dataclass(frozen=True)
class Step:
    """One pointer toward the root: ``child.key`` equals ``parent.parent_key``, the parent's primary key."""

    child: str
    key: str
    parent: str
    parent_key: str


@dataclass(frozen=True)
class PartitionPlan:
    """Which tables follow the root entity, by which chains, and which are shared.

    :param root: the root entity's table.
    :param owned: table name to its chains of steps up to the root; the root's
        own entry is one empty chain.
    :param shared: tables no chain connects to the root.
    :param primary_keys: every table's primary key, from the schema.
    """

    root: str
    owned: Mapping[str, tuple[tuple[Step, ...], ...]]
    shared: tuple[str, ...]
    primary_keys: Mapping[str, str]


def plan_partitions(schema: DataSchema, root: str) -> PartitionPlan:
    """Each table's chains of relations up to `root`, or "shared" where there is none.

    A chain visits each table at most once and ends at the root. The root's own
    rows are assigned by its primary key alone, even when a relation of the root
    to itself (``users.referrer_id = users.id``) exists. A relation whose keys can
    never be equal (by their declared types, which upstream requires of a
    relation's keys: an integer and a text, a date and a timestamp) is no step of
    a chain: it owns nothing, so a table it alone would link to the root stays
    shared.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.keys import never_equal

    tables = {table.name: table for table in schema.tables}
    if root not in tables:
        raise ValueError(f"the schema declares no root table {root!r}")

    def kind(table: str, column: str) -> DType | None:
        return (tables[table].columns or {}).get(column)

    def meets(child: str, child_key: str, parent: str, parent_key: str) -> bool:
        left, right = kind(child, child_key), kind(parent, parent_key)
        return left is None or right is None or not never_equal(left, right)

    up: dict[str, list[Step]] = {name: [] for name in tables}
    for relation in schema.relations:
        left, right = tables[relation.left_table], tables[relation.right_table]
        ends = (relation.left_table, relation.left_key, relation.right_table, relation.right_key)
        if relation.right_key == right.primary_key and meets(*ends):
            up[left.name].append(Step(left.name, relation.left_key, right.name, relation.right_key))
        if relation.left_key == left.primary_key and meets(*ends[2:], *ends[:2]):
            up[right.name].append(Step(right.name, relation.right_key, left.name, relation.left_key))

    def chains(table: str, visited: frozenset[str]) -> list[tuple[Step, ...]]:
        found = []
        for step in up[table]:
            if step.parent == root:
                found.append((step,))
            elif step.parent not in visited:
                found.extend((step, *rest) for rest in chains(step.parent, visited | {step.parent}))
        return found

    owned: dict[str, tuple[tuple[Step, ...], ...]] = {root: ((),)}
    shared = []
    for name in sorted(tables):
        if name == root:
            continue
        found = tuple(dict.fromkeys(chains(name, frozenset({name}))))
        if found:
            owned[name] = found
        else:
            shared.append(name)
    return PartitionPlan(root, owned, tuple(shared), {name: spec.primary_key for name, spec in tables.items()})


def _check_keys(plan: PartitionPlan, sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]]) -> None:
    for table, chains in plan.owned.items():
        for chain in chains:
            for step in chain:
                for name, key in ((step.child, step.key), (step.parent, step.parent_key)):
                    if key not in sources[name][1]:
                        raise ValueError(f"{name}.{key} links {table} to the root but is not a loaded column")


def _downward(owned: Mapping[str, tuple[tuple[Step, ...], ...]], prefix: tuple[Hop, ...]) -> bool:
    """Whether `prefix` walks from the root down one of its last table's chains, reaching only a partition's own rows.

    ``users -> accounts -> cards`` along ``accounts.user_id`` and ``cards.account_id``
    reaches the cards whose chain ends at the starting user: they are in that
    user's partition already.
    """

    chain = tuple(Step(hop.table, hop.column, hop.from_table, hop.from_column) for hop in reversed(prefix))
    return chain in owned.get(prefix[-1].table, ())


def _sample_routes(schema: DataSchema, plan: PartitionPlan,
                   sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]],
                   sample: SampleSpec | None) -> tuple[tuple[Hop, ...], ...]:
    """The routes `sample` reads from the root, checked against the split, before anything is read."""

    if sample is None:
        return ()
    if sample.entity.table != plan.root:
        raise ValueError(f"the sample is about {sample.entity.table!r}, but the split's root is {plan.root!r}")
    _refuse_split_candidates(schema, sample, set(plan.owned), plan.root)
    routes = sample_routes(schema, sample)
    for route in routes:
        for hop in route:
            for table, column in ((hop.from_table, hop.from_column), (hop.table, hop.column)):
                if column not in sources[table][1]:
                    raise ValueError(f"{table}.{column} joins relation {hop.relation!r} but is not a loaded column")
    return routes


def _refuse_split_candidates(schema: DataSchema, sample: SampleSpec, split: set[str], root: str) -> None:
    """A candidate is any row of its table: a candidate set that reads split tables cannot be served by partitions."""

    for name, (table, candidate) in candidate_routes(schema, sample).items():
        found = sorted({table, *(hop.table for route in candidate for hop in route)} & split)
        if found:
            raise ValueError(
                f"candidate set {name!r} reads {', '.join(found)}, which the split divides by {root}: "
                "a candidate's rows could be in any partition, so only candidate sets that read shared "
                "tables alone can be split"
            )
