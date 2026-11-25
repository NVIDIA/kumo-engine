# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from kumo.pytorch import api


def walk_contract(*walks, topology="relations"):
    integer = api.DType("int64")
    schema = api.DataSchema(
        tables=(
            api.TableSpec("pins", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
            api.TableSpec("boards", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "links", None, {"pin": integer, "board": integer}, time_column=api.NON_TEMPORAL
            ),
        ),
        relations=(
            api.RelationSpec("pin_link", "pins", "id", "links", "pin"),
            api.RelationSpec("board_link", "links", "board", "boards", "id"),
        ),
    )
    sample = api.SampleSpec(
        api.EntitySpec("pins"),
        neighborhoods={"graph": api.NeighborhoodSpec((api.GraphLayer(walks),))},
    )
    inputs = {
        "g": api.Input(api.NeighborhoodSource("graph"), api.Graph({}, topology=topology), None)
    }
    return api.DataLoaderSpec(schema, sample, api.BatchSpec(inputs))


def test_walk_contract_round_trips_and_freezes_quotas():
    quotas = {"pins": 25, "boards": 75}
    walk = api.WalkSample("via_boards", ("links",), quotas, restart=0.25, budget=20)
    contract = walk_contract(walk)
    quotas["pins"] = 1
    contract.validate()
    assert walk.keep["pins"] == 25
    assert api.DataLoaderSpec.from_json(contract.to_json()) == contract
    pair = replace(walk, edges=(("board_link", "pin_link"),))
    walk_contract(pair).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("restart", 0),
        ("restart", 1),
        ("restart", float("nan")),
        ("restart", 1e-18),
        ("budget", 2**32),
        ("budget", 0),
        ("edges", ()),
        ("keep", {"pins": -1}),
    ],
)
def test_walk_rejects_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        replace(api.WalkSample("walk", ("links",), {"pins": 2}, 0.5, 100), **{field: value})


def test_walk_contract_rejects_ambiguous_outputs_and_invalid_schema():
    walk = api.WalkSample("via_boards", ("links",), {"pins": 25}, 0.25, 20)
    for contract, message in (
        (walk_contract(walk, walk), "duplicate walk edge set"),
        (walk_contract(walk, topology="sampled"), "relations topology"),
        (walk_contract(replace(walk, keep={"links": 5})), "cannot reach kept table"),
        (walk_contract(replace(walk, edges=("missing",))), "link table"),
        (
            walk_contract(replace(walk, edges=("pin_link",), keep={"boards": 5})),
            "cannot reach kept table",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            contract.validate()


def test_serving_walk_refused():
    contract = walk_contract(api.WalkSample("via_boards", ("links",), {"pins": 5}, 0.5, 100))
    with pytest.raises(ValueError, match="serving rows do not support scored walk"):
        api.ServingRowsSpec.from_contract(contract)
