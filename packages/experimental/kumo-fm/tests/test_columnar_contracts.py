# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FM assembly consumes the public ColumnarGraph ownership and topology contract."""
from dataclasses import replace

import pyarrow as pa
import pytest
from kumo.fm._assembly import assemble_graph
from kumo.fm._contract import _Budgets, _PayloadSpec, _SchemaView, _TargetSpec
from kumo.pytorch import api


def payload(*, parent_first=False, budgets=None):
    view = _SchemaView(tables={
        name: {"primary_key": "id", "time_column": None, "columns": {
            "id": {"dtype": "int64", "nullable": False}}}
        for name in ("users", "events")}, relations={
            "link": {"child": "events", "parent": "users", "foreign_key": "user",
                     "parent_is_source": parent_first}})
    return _PayloadSpec(schema_view=view, entity="users", kind="regression",
                        target=_TargetSpec("float64"), table_names=frozenset(view.tables),
                        relation_names=frozenset(view.relations),
                        projections={"users": ["id"], "events": ["id"]},
                        budgets=budgets or _Budgets())


def graph(*, parent_first=False, repeat_edges=False, empty=False):
    def nodes(keys, owners):
        return api.ColumnarNodes(ids=pa.array(keys, pa.int64()), owners=pa.array(owners, pa.uint64()),
                                 features=pa.table({"id": pa.array(keys, pa.int64())}))
    # The same source entity occurs twice under distinct owners. An isolated
    # third occurrence must remain in the root table without fabricated edges.
    sources = [] if empty else [0, 1, 2]
    destinations = [] if empty else [0, 1, 0]
    if repeat_edges:
        sources *= 2
        destinations *= 2
    if parent_first:
        sources, destinations = destinations, sources
    return api.ColumnarGraph(root_type="users", nodes={
        "users": nodes([10, 10, 99], [0, 1, 2]),
        "events": nodes([] if empty else [30, 30, 40], [] if empty else [0, 1, 0])},
        edges={"link": api.ColumnarEdges(
            "users" if parent_first else "events", "events" if parent_first else "users",
            pa.table({"source": pa.array(sources, pa.uint64()),
                      "destination": pa.array(destinations, pa.uint64())}))})


def seeds():
    return [{"entity_key": key, "instance_id": identifier}
            for key, identifier in zip([10, 10, 99], [-7, 2**53 + 1, 0], strict=True)]


@pytest.mark.parametrize("parent_first", [False, True])
@pytest.mark.parametrize("repeat_edges", [False, True])
def test_assembly_preserves_owners_keys_and_selected_parent_values(parent_first, repeat_edges):
    result = assemble_graph(payload(parent_first=parent_first), seeds(),
                            graph(parent_first=parent_first, repeat_edges=repeat_edges))
    assert result["users"].to_pydict() == {
        "instance_id": [-7, 2**53 + 1, 0], "id": [10, 10, 99]}
    assert result["events"].to_pydict() == {
        "instance_id": [-7, 2**53 + 1, -7], "id": [30, 30, 40], "__fm_edge_link": [10, 10, 10]}


@pytest.mark.parametrize("parent_first", [False, True])
def test_empty_related_tables_keep_their_typed_schema(parent_first):
    result = assemble_graph(payload(parent_first=parent_first), seeds(),
                            graph(parent_first=parent_first, empty=True))
    assert result["events"].num_rows == 0
    assert result["events"].column_names == ["instance_id", "id", "__fm_edge_link"]
    assert all(field.type == pa.int64() for field in result["events"].schema)
    assert result["users"]["id"].to_pylist() == [10, 10, 99]


@pytest.mark.parametrize("limit,accepted", [(8, False), (9, True)])
def test_candidate_budget_counts_nodes_and_edges_including_duplicate_pairs(limit, accepted):
    # Three root nodes, three related nodes and three selected edges.
    work = payload(budgets=_Budgets(max_candidate_rows=limit))
    if accepted:
        assert len(assemble_graph(work, seeds(), graph())["events"]) == 3
    else:
        with pytest.raises(ValueError, match="node-plus-edge budget"):
            assemble_graph(work, seeds(), graph())


def test_row_budget_is_checked_before_assembly():
    work = payload(budgets=_Budgets(max_rows_per_table=2))
    with pytest.raises(ValueError, match="row budget"):
        assemble_graph(work, seeds(), graph())


def test_wrong_root_type_is_rejected():
    with pytest.raises(ValueError, match="rooted at the FM entity"):
        assemble_graph(payload(), seeds(), replace(graph(), root_type="events"))
