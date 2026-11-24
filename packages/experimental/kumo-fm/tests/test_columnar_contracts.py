# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise graph ownership, endpoint and assembly checks with public Arrow batches."""
from types import SimpleNamespace

import pyarrow as pa
import pytest
from kumo.fm._assembly import GraphAssembler
from kumo.fm._columnar import adapt_columnar_batch
from kumo.fm._snapshot import _schema_view
from kumo.pytorch import api


def fixture():
    schema = api.DataSchema(tables=(
        api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL),
        api.TableSpec("events", "id", {"id": api.DType("int64"), "user": api.DType("int64")},
                      time_column=api.NON_TEMPORAL)), relations=(
                          api.RelationSpec("link", "events", "user", "users", "id"),))
    bundle = _schema_view(schema)
    work = SimpleNamespace(bundle=bundle, entity="users", table_names={"users", "events"},
                           relation_names={"link"}, projections={"users": ["id"], "events": ["id"]},
                           kind="regression", target={"dtype": "float64"},
                           budgets={"max_candidate_rows": 100, "max_rows_per_table": 100})
    adapter = SimpleNamespace(bundle=bundle, node_types=list(bundle.tables),
                              assembler=GraphAssembler(bundle), metrics={"native_candidate_rows": 0})
    return work, adapter


def graph(*, user_owners=(0, 1), event_owners=(0, 1), user_keys=(10, 20),
          event_keys=(30, 40), endpoints=(0, 1)):
    def nodes(keys, owners):
        return api.ColumnarNodes(ids=pa.array(keys), owners=pa.array(owners, pa.uint64()),
                                 features=pa.table({"id": pa.array(keys, pa.int64())}))
    return api.ColumnarGraph(root_type="users", nodes={
        "users": nodes(user_keys, user_owners), "events": nodes(event_keys, event_owners)},
        edges={"link": api.ColumnarEdges("events", "users", pa.table({
            "source": pa.array([0, 1], pa.uint64()),
            "destination": pa.array(endpoints, pa.uint64())}))})


def run(candidate, *, permutation=(0, 1), seeds=None):
    work, adapter = fixture()
    return adapt_columnar_batch(adapter, work, seeds or [
        {"entity_key": 10, "instance_id": -7}, {"entity_key": 20, "instance_id": 2**53 + 1}],
        SimpleNamespace(inputs={"graph": candidate}, metadata={"sample_ids": permutation}))


def test_permuted_anchor_batches_remap_roots_owners_and_edges():
    result = run(graph(user_keys=(20, 10), event_keys=(40, 30)), permutation=(1, 0))
    assert result["users"].to_pydict() == {"instance_id": [-7, 2**53 + 1], "id": [10, 20]}
    assert result["events"].to_pydict() == {
        "instance_id": [-7, 2**53 + 1], "id": [30, 40], "__fm_edge_link": [10, 20]}


@pytest.mark.parametrize("candidate,match", [
    (graph(user_owners=(0, 2)), "owner is out of bounds"),
    (graph(event_owners=(1, 0)), "crosses seed"),
    (graph(endpoints=(0, 99)), "edge position out of bounds"),
    (graph(user_keys=(20, 10)), "root prefix"),
    (graph(event_keys=(30, 30), event_owners=(0, 0), endpoints=(0, 0)), "duplicate or null"),
])
def test_malformed_native_graph_is_rejected(candidate, match):
    with pytest.raises(ValueError, match=match):
        run(candidate)


@pytest.mark.parametrize("permutation", [(0, 0), (0, 2), (0,), (0, 1, 2)])
def test_malformed_anchor_permutation_is_rejected(permutation):
    with pytest.raises(ValueError, match="permutation"):
        run(graph(), permutation=permutation)
