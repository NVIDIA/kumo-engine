# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Larger fan-outs/degree shapes reuse the existing graph conformance oracles.

DiskGraph is checked against independent source semantics; no reference-engine
batch ordering or physical layout is required.
"""

from collections import Counter

import pytest

from conformance.generate import GeneratedCase
from conformance.graph_cases import (
    BASE,
    graph_contract,
    graph_edge_uniqueness,
    graph_point_in_time,
    graph_structure,
)
from kumo.pytorch.api import SampleRequest


def degree_case(profile, path, fanout):
    cutoff = BASE + 700
    dimensions = {
        "schema_shape.graph_path": path,
        "schema_shape.graph_orientation": "aligned",
        "contract.graph_direction": "forward",
        "contract.graph_fanout": fanout,
        "contract.graph_temporal": "inherit",
        "time.graph_ties": "distinct",
        "time.cutoffs": "shared",
    }
    degrees = [192] * 6 if profile == "uniform" else [384, 3, 3, 3, 3, 3]
    events = []
    for user, degree in enumerate(degrees, 1):
        for position in range(degree):
            events.append(
                dict(
                    event_id=user * 10000 + position,
                    user_id=user,
                    shop_id=position % 6 + 1,
                    # Globally distinct seconds; one eligible row exactly at T.
                    at=cutoff - 6 * position - (user - 1),
                )
            )
        events.append(
            dict(event_id=user * 10000 + degree, user_id=user, shop_id=1, at=cutoff + user)
        )
    assert Counter(row["user_id"] for row in events if row["at"] <= cutoff) == dict(
        enumerate(degrees, 1)
    )
    # Row order does not supply the reference's recency order.
    tables = {
        "users": [dict(user_id=i, joined=BASE - 9000 - i) for i in range(1, 7)],
        "events": events[::-1],
        "shops": [dict(shop_id=i, owner_id=i, opened=BASE - 5000 - i) for i in range(1, 7)],
    }
    requests = tuple(SampleRequest(f"root-{i}", i, cutoff) for i in range(1, 7))
    return GeneratedCase(
        f"degree-{profile}-{path}-{fanout}",
        0,
        "S",
        dimensions,
        graph_contract(dimensions),
        tables,
        requests,
    )


@pytest.mark.parametrize("engine", ("diskgraph",))
@pytest.mark.parametrize("profile", ("uniform", "skewed"))
@pytest.mark.parametrize("path", ("one_hop", "many_to_one", "three_hop"))
@pytest.mark.parametrize("fanout", (2, 32, 128))
def test_graph_fanout_and_degree_sweeps(engine, profile, path, fanout):
    case = degree_case(profile, path, fanout)
    # These existing properties check all layers, exact most-recent membership
    # for distinct clocks, node/edge ownership and cutoff/future perturbations.
    graph_structure(case, engine, 0)
    graph_edge_uniqueness(case, engine, 0)
    graph_point_in_time(case, engine, 0)
