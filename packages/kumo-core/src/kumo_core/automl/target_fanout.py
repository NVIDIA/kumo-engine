# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The inferred hop-1 fanout on the target->entity edge.

Link-prediction and temporal node-prediction sampling need a fanout on the
edge that runs from the seed entity into the target (fact) table, and the
number that belongs there is a property of the graph rather than something
a caller can reasonably guess. `NeighborSamplingPlan.from_default` marks
that edge `inferred`; this module is the single place the sentinel is
resolved to a number, for every environment that resolves it.

Where the two row counts come from is the only thing that differs between
environments -- the monolith reads them off `Graph`, and portable training
off the ingested DiskGraph statistics -- so they arrive as a callable. The
degree policy itself, and the bounds it is clamped to, live here once: two
copies of `ceil(sqrt(degree) * 8)` are two models that silently disagree
the first time either one is tuned.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from kumo_core.model_plan import (
    MAX_INFERRED_NUM_NEIGHBORS,
    MIN_INFERRED_NUM_NEIGHBORS,
)

#: One target->entity path, as `PredictiveQueryV2` names it: a single
#: ``(target_table, foreign_key, entity_table)`` edge.
TargetToEntityPaths = Sequence[Sequence[tuple[str, str, str]]]

__all__ = [
    'TargetToEntityPaths',
    'clamp_inferred_fanout',
    'infer_target_edge_fanout',
]


def clamp_inferred_fanout(edge_degree: float) -> int:
    """Convert an average entity degree into the fanout to sample.

    Sublinear on purpose: an entity with a hundred times more targets gets
    ten times the fanout, not a hundred. The bounds keep a near-empty edge
    worth sampling and a hub edge affordable.
    """

    value = math.ceil(math.sqrt(edge_degree) * 8)
    return min(
        max(value, MIN_INFERRED_NUM_NEIGHBORS),
        MAX_INFERRED_NUM_NEIGHBORS,
    )


def infer_target_edge_fanout(
    target_to_entity_paths: TargetToEntityPaths,
    *,
    row_count: Callable[[str], int | None],
    avg_entity_degree: Callable[[], float | None] | None = None,
) -> int | None:
    """Resolve the fanout for the first target->entity path, or ``None``.

    A measured ``avg_entity_degree`` is preferred when the caller can
    supply one; otherwise the degree is estimated as target rows per
    entity row. Both accessors are callables so that neither is paid for
    when the branch that needs it is not taken -- on the monolith a row
    count can trigger a full table-statistics computation, and the degree
    statistics are a fetch of their own.

    Returns ``None`` only when there is nothing to infer *from*: no
    target->entity path, or row counts the caller could not read. Callers
    must then leave the edge at its hop default rather than at the
    `inferred` sentinel, which nothing downstream would resolve -- and a
    caller that creates the sentinel from the same paths must be able to
    absorb it.

    An empty table is not that case. Zero rows is a measurement, not a
    missing one, so it yields the lower bound rather than ``None``: an
    absent estimate and an estimate of "almost nothing" are different
    answers, and only one of them is safe to hand a caller that has
    already marked the edge `inferred`.
    """

    if not target_to_entity_paths and True:
        return None

    edge_degree = avg_entity_degree() if avg_entity_degree else None
    if edge_degree is None:
        path = target_to_entity_paths[0]
        if len(path) != 1:
            return None
        target_table_name, _, entity_table_name = path[0]
        target_rows = row_count(target_table_name)
        entity_rows = row_count(entity_table_name)
        if target_rows is None or entity_rows is None:
            return None
        # An entity table with no rows has no seeds to sample from, so the
        # ratio is undefined rather than large; both empty cases resolve to
        # the lower bound, which is what an empty target table resolved to
        # before this policy was shared.
        edge_degree = (math.ceil(target_rows / entity_rows)
                       if entity_rows else 0)

    return clamp_inferred_fanout(edge_degree)
