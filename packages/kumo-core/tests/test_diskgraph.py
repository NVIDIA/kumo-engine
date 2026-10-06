# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from kumo_core.diskgraph.ingest import (
    DiskgraphInputError,
    normalize_legacy_column_aliases,
    prepare_diskgraph_converter_config,
    validate_canonical_diskgraph_graph,
    validate_diskgraph_ingest_config,
    validate_resolved_diskgraph_inputs,
)


def _canonical_graph():
    return {
        "name":
        "features",
        "tables": [{
            "table_name":
            "events",
            "source_table_name":
            "main.features.events",
            "connector_id":
            "databricks-native",
            "pkey": {
                "name": "event_id",
                "dtype": "int",
                "stype": "id",
            },
            "cols": [{
                "name": "embedding",
                "dtype": "floatlist",
                "stype": "sequence",
            }, {
                "name": "created_at",
                "dtype": "string",
                "stype": "timestamp",
                "timestamp_format": "%Y/%m/%d",
            }],
            "fkeys": [{
                "name": "user_id",
                "dtype": "int",
                "stype": "id",
                "dst_table": "users",
                "dst_key": "user_id",
            }],
        }, {
            "table_name": "users",
            "source_table_name": "main.features.users",
            "connector_id": "databricks-native",
            "pkey": {
                "name": "user_id",
                "dtype": "int",
                "stype": "id",
            },
            "cols": [],
            "fkeys": [],
        }]
    }


def test_accepts_only_canonical_graph_without_mutating_it():
    graph = _canonical_graph()
    original = copy.deepcopy(graph)
    validate_canonical_diskgraph_graph(graph)

    prepared = prepare_diskgraph_converter_config(graph)

    assert graph == original
    assert prepared is not graph
    assert prepared["name"] == "features"
    for source_table, converter_table in zip(graph["tables"],
                                             prepared["tables"]):
        assert "connector_id" not in converter_table
        assert converter_table["data_source"] == source_table["connector_id"]


def test_converter_input_preserves_multicategorical_delimiters():
    graph = _canonical_graph()
    graph["tables"][0]["cols"].append({
        "name": "tags",
        "dtype": "string",
        "stype": "multicategorical",
        "multicategorical_delimiter": "|",
    })

    prepared = prepare_diskgraph_converter_config(graph)
    columns = {
        column["name"]: column
        for column in prepared["tables"][0]["cols"]
    }

    assert columns["tags"]["multicategorical_delimiter"] == "|"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.__setitem__("source_name", "events"),
         "source_name is a legacy alias"),
        (lambda table: table.__setitem__("data_source", "warehouse"),
         "data_source is a legacy alias"),
        (lambda table: table.__setitem__(
            "source_table", {
                "source_table_name": "main.features.events",
                "connector_id": "databricks-native",
            }), "source_table is a legacy alias"),
        (lambda table: table.pop("source_table_name"),
         "source_table_name must be a non-empty string"),
        (lambda table: table["fkeys"][0].__setitem__("dst_table",
                                                     "warehouse.users"),
         "dst_table must be the exact graph table_name"),
    ],
)
def test_rejects_each_legacy_graph_fallback(mutation, message):
    graph = _canonical_graph()
    mutation(graph["tables"][0])

    with pytest.raises(DiskgraphInputError, match=message):
        validate_canonical_diskgraph_graph(graph)


def test_string_sequence_column_is_prepared_as_a_float_array():
    graph = _canonical_graph()
    graph["tables"][0]["cols"][0].update(dtype="string", stype="sequence")

    prepared = prepare_diskgraph_converter_config(graph)

    assert prepared["tables"][0]["cols"][0]["dtype"] == "floatlist"
    assert graph["tables"][0]["cols"][0]["dtype"] == "string"


def test_ts_format_metadata_is_prepared_as_a_timestamp_format():
    graph = _canonical_graph()
    column = graph["tables"][0]["cols"][1]
    column.pop("timestamp_format")
    column["metadata"] = {"TS_FORMAT": "%Y/%m/%d"}

    prepared = prepare_diskgraph_converter_config(graph)

    prepared_column = prepared["tables"][0]["cols"][1]
    assert prepared_column["timestamp_format"] == "%Y/%m/%d"
    assert "metadata" not in prepared_column
    assert graph["tables"][0]["cols"][1]["metadata"] == {
        "TS_FORMAT": "%Y/%m/%d"
    }


def test_ts_format_metadata_survives_beside_other_metadata_keys():
    config = {
        "tables": [{
            "cols": [{
                "name": "evt",
                "metadata": {
                    "TS_FORMAT": "%y%m%d",
                    "OTHER": "keep",
                },
            }]
        }]
    }

    normalize_legacy_column_aliases(config)

    column = config["tables"][0]["cols"][0]
    assert column["timestamp_format"] == "%y%m%d"
    assert column["metadata"] == {"OTHER": "keep"}


def test_ts_format_echoing_the_unit_is_not_injected_as_a_format():
    # kumo overloads TS_FORMAT: when it equals the column's timestamp_unit it
    # is a unit sentinel for a numeric-epoch column, not a strftime format,
    # and DiskGraph rejects a bare unit token as an invalid chrono format. The
    # unit already rides timestamp_unit.
    config = {
        "tables": [{
            "cols": [{
                "name": "epoch_ts",
                "timestamp_unit": "s",
                "metadata": {
                    "TS_FORMAT": "s"
                },
            }]
        }]
    }

    normalize_legacy_column_aliases(config)

    column = config["tables"][0]["cols"][0]
    assert "timestamp_format" not in column
    assert column["timestamp_unit"] == "s"


def test_declared_timestamp_format_wins_over_the_metadata_alias():
    config = {
        "tables": [{
            "cols": [{
                "name": "evt",
                "timestamp_format": "%Y-%m-%d",
                "metadata": {
                    "TS_FORMAT": "%y%m%d"
                },
            }]
        }]
    }

    normalize_legacy_column_aliases(config)

    assert config["tables"][0]["cols"][0]["timestamp_format"] == "%Y-%m-%d"


def test_blank_or_absent_ts_format_writes_no_timestamp_format():
    config = {
        "tables": [{
            "cols": [{
                "name": "evt",
                "metadata": {
                    "TS_FORMAT": "   "
                },
            }, {
                "name": "amount",
            }],
        }]
    }

    normalize_legacy_column_aliases(config)

    for column in config["tables"][0]["cols"]:
        assert "timestamp_format" not in column


def test_keys_and_pkeys_are_left_alone():
    # pkey/fkey are stype=ID by product constraint and cannot carry a
    # TS_FORMAT; normalization is scoped to `cols` and never reaches them.
    config = {
        "tables": [{
            "pkey": {
                "name": "id",
                "metadata": {
                    "TS_FORMAT": "%Y-%m-%d"
                },
            },
            "fkeys": [{
                "name": "account_id",
                "metadata": {
                    "TS_FORMAT": "%Y-%m-%d"
                },
            }],
            "cols": [],
        }]
    }

    normalize_legacy_column_aliases(config)

    assert "timestamp_format" not in config["tables"][0]["pkey"]
    assert "timestamp_format" not in config["tables"][0]["fkeys"][0]


@pytest.mark.parametrize("config", [{
    "tables": None
}, {
    "tables": [{}]
}, {
    "tables": [{
        "cols": None
    }]
}])
def test_a_graph_without_columns_normalizes_to_itself(config):
    original = copy.deepcopy(config)

    normalize_legacy_column_aliases(config)

    assert config == original


def test_resolved_inputs_require_exact_table_alias_keys_and_strings():
    graph = _canonical_graph()
    versions = {"events": "delta-v7", "users": "delta-v3"}
    paths = {"events": "/staged/events", "users": "/staged/users"}
    validate_resolved_diskgraph_inputs(graph, versions, paths)

    with pytest.raises(DiskgraphInputError,
                       match="table_data_versions keys must exactly match"):
        validate_resolved_diskgraph_inputs(graph, {
            "main.features.events": "delta-v7",
            "users": "delta-v3",
        }, paths)

    invalid_versions = copy.deepcopy(versions)
    invalid_versions["events"] = 7
    with pytest.raises(DiskgraphInputError,
                       match=r"table_data_versions\['events'\]"):
        validate_resolved_diskgraph_inputs(graph, invalid_versions, paths)


def test_resolved_inputs_reject_default_connector_and_missing_path():
    graph = _canonical_graph()
    graph["tables"][0]["connector_id"] = "default"
    versions = {"events": "v1", "users": "v2"}
    paths = {"events": "/events", "users": "/users"}
    with pytest.raises(DiskgraphInputError, match="'default' is not canonical"):
        validate_resolved_diskgraph_inputs(graph, versions, paths)

    graph["tables"][0]["connector_id"] = "databricks-native"
    paths.pop("events")
    with pytest.raises(DiskgraphInputError,
                       match="source_paths keys must exactly match"):
        validate_resolved_diskgraph_inputs(graph, versions, paths)

    with pytest.raises(DiskgraphInputError,
                       match="source_paths must be a mapping"):
        validate_resolved_diskgraph_inputs(graph, versions,
                                           None)  # type: ignore[arg-type]


def test_validates_diskgraph_multicategorical_storage_shape():
    valid = {
        "tables": [{
            "name":
            "EVENTS",
            "columns": [{
                "name": "tags",
                "semantic_type": "multicategorical",
                "ingested_dtype": "string_array",
            }],
        }]
    }
    validate_diskgraph_ingest_config(valid)

    valid["tables"][0]["columns"][0]["ingested_dtype"] = "string"
    with pytest.raises(ValueError, match="EVENTS.*tags"):
        validate_diskgraph_ingest_config(valid)
