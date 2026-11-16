# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pybind smoke tests for intlist (int64_array) and stringlist (string_array)
feature columns.

These tests verify:
1. The wheel can load an index that contains list-type columns.
2. fetch_features returns a RecordBatch with the correct Arrow types.
3. The decoded values match the source Parquet data.
4. Empty lists round-trip correctly.
"""

import json
import pyarrow as pa
import diskgraph as diskgraph_py


# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_all_features(index_path: str, node_type: str, node_ids: list) -> pa.RecordBatch:
    """Load index, call fetch_features, return the first feature batch."""
    dg = diskgraph_py.SamplerEngine(index_path)
    req = [{"node_type": node_type, "node_ids": node_ids, "feature_names": []}]
    batches = [e["batch"] for e in dg.fetch_features(req) if e["type"] == "feature"]
    assert batches, "fetch_features returned no feature events"
    if len(batches) == 1:
        return batches[0]
    import pyarrow as pa
    combined = pa.Table.from_batches(batches)
    return combined.to_batches()[0]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_list_features_index_loads(list_features_graph):
    """Index with list columns can be opened by the SamplerEngine."""
    dg = diskgraph_py.SamplerEngine(list_features_graph)
    mapping = json.loads(dg.get_type_mapping())
    assert "items" in mapping["node_types"], "items table should be present"


def test_list_features_schema(list_features_graph):
    """fetch_features returns a batch with the expected Arrow schema."""
    batch = fetch_all_features(list_features_graph, "items", [0])
    schema = batch.schema
    field_names = schema.names

    assert "tag_ids" in field_names, f"tag_ids missing from schema: {field_names}"
    assert "tags"    in field_names, f"tags missing from schema: {field_names}"

    tag_ids_field = schema.field("tag_ids")
    tags_field    = schema.field("tags")

    # Both should be list types
    assert pa.types.is_list(tag_ids_field.type), \
        f"tag_ids should be list type, got {tag_ids_field.type}"
    assert pa.types.is_list(tags_field.type), \
        f"tags should be list type, got {tags_field.type}"

    # Inner types
    assert pa.types.is_integer(tag_ids_field.type.value_type), \
        f"tag_ids inner type should be integer, got {tag_ids_field.type.value_type}"
    assert pa.types.is_string(tags_field.type.value_type) \
        or pa.types.is_large_string(tags_field.type.value_type), \
        f"tags inner type should be string, got {tags_field.type.value_type}"


def test_intlist_values(list_features_graph):
    """
    tag_ids values match source data:
      item 0 → [10, 20, 30]
      item 1 → [40]
      item 2 → []
      item 3 → [1, 2, 3, 4, 5]
    """
    batch = fetch_all_features(list_features_graph, "items", [0, 1, 2, 3])
    got = sorted(tuple(value) for value in batch.column("tag_ids").to_pylist())
    expected = sorted([
        (10, 20, 30),
        (40,),
        (),
        (1, 2, 3, 4, 5),
    ])
    assert got == expected


def test_stringlist_values(list_features_graph):
    """
    tags values match source data:
      item 0 → ["sports", "sale"]
      item 1 → ["kids"]
      item 2 → []
      item 3 → ["pro", "audio"]
    """
    batch = fetch_all_features(list_features_graph, "items", [0, 1, 2, 3])
    got = sorted(tuple(value) for value in batch.column("tags").to_pylist())
    expected = sorted([
        ("sports", "sale"),
        ("kids",),
        (),
        ("pro", "audio"),
    ])
    assert got == expected


def test_empty_list_rows(list_features_graph):
    """The row with both empty list features survives Encode."""
    batch = fetch_all_features(list_features_graph, "items", [0, 1, 2, 3])
    rows = zip(
        batch.column("tag_ids").to_pylist(),
        batch.column("tags").to_pylist(),
        strict=True,
    )
    assert any(tag_ids == [] and tags == [] for tag_ids, tags in rows)


# ── tests: inner-null drop-null contract (Python pybind e2e) ────────────────

def test_intlist_inner_nulls_dropped(list_features_graph_with_inner_nulls):
    """
    Null elements are dropped from the persisted intlist. Legitimate zeros
    are preserved; only the null bit is the signal for dropping.

      source tag_ids           → decoded tag_ids
      [1, null, 3]             → [1, 3]
      [null, null]             → []
      [0, null, 7]             → [0, 7]          (zero preserved)
      [10, 20]                 → [10, 20]
    """
    batch = fetch_all_features(list_features_graph_with_inner_nulls, "items", [0, 1, 2, 3])
    got = sorted(tuple(value) for value in batch.column("tag_ids").to_pylist())
    expected = sorted([(1, 3), (), (0, 7), (10, 20)])
    assert got == expected


def test_stringlist_inner_nulls_dropped_preserves_empty_string(list_features_graph_with_inner_nulls):
    """
    Null elements are dropped from the persisted stringlist. Legitimate empty
    strings are preserved.

      source tags              → decoded tags
      ["a", null, "b"]         → ["a", "b"]
      [null, null]             → []
      ["x", "", "y"]           → ["x", "", "y"]     (empty string preserved)
      ["p", "", null, "q"]     → ["p", "", "q"]     (null dropped, "" kept)
    """
    batch = fetch_all_features(list_features_graph_with_inner_nulls, "items", [0, 1, 2, 3])
    got = sorted(tuple(value) for value in batch.column("tags").to_pylist())
    expected = sorted([("a", "b"), (), ("x", "", "y"), ("p", "", "q")])
    assert got == expected
