# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import replace

import pytest
from kumo_core.task_dataset import (
    DiskgraphTaskManifest,
    TaskDatasetMaterializeSpec,
    _parse_stats,
    _task_dataset_stats,
    materialize_task_dataset,
    resolve_target_encoder_input,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def setup_extension(tmp_path):
    graph_path = tmp_path / "graph" / "v1"
    extension = graph_path / "train_tables" / "tt-1"
    extension.mkdir(parents=True)
    stages = {
        name: {
            "table_name": f"{name}_seeds",
            "index_path": str(extension),
            "row_count": count,
        }
        for name, count in {
            "train": 80,
            "val": 10,
            "test": 10
        }.items()
    }
    task_dataset = {
        "graph_index_path": str(graph_path),
        "tt_extension_path": str(extension),
        "stages": stages,
        "columns": {
            "timestamp": None,
            "weight": None
        },
    }
    write_json(
        extension / "diskgraph_manifest.json",
        {
            "tt_id": "tt-1",
            "graph_index_path": str(graph_path),
            "tt_extension_path": str(extension),
            "diskgraph_manifest_path": str(
                extension / "diskgraph_manifest.json"),
            "task_dataset": task_dataset,
        },
    )
    write_json(
        extension / "stats.json",
        {
            "version": 2,
            "kind": "tt_stats",
            "num_total_rows": 100,
            "num_train_rows": 80,
            "num_entities": 20,
            "num_targets": 2,
        },
    )
    write_json(
        extension / "train_table_col_stats.json",
        {
            "version": 1,
            "kind": "tt_train_table_col_stats",
            "stats_format": "kumo_col_stats_v1",
            "stats": {
                "TARGET1__kumo.categorical__CATEGORY_COUNTS__False": {
                    "value": [[0, 80], [1, 20]]
                }
            },
        },
    )
    return graph_path, extension


def test_stats_parser_requires_current_bounded_contract(tmp_path) -> None:
    path = tmp_path / "stats.json"
    write_json(
        path,
        {
            "version": 1,
            "kind": "tt_stats",
            "num_total_rows": 100,
            "num_train_rows": 80,
            "num_entities": 20,
            "num_targets": 2,
        },
    )

    with pytest.raises(RuntimeError, match="version/kind mismatch"):
        _parse_stats(str(path))


def test_link_task_stats_drop_cardinality_sized_metadata() -> None:
    huge_distribution = {
        "count": {
            f"target-{index}": index
            for index in range(10_000)
        },
        "percentage": {
            f"target-{index}": 0.0001
            for index in range(10_000)
        },
    }
    stats = {
        "version": 2,
        "kind": "tt_stats",
        "num_total_rows": 100,
        "num_train_rows": 80,
        "num_entities": 20,
        "num_targets": 10_000,
        "num_links": 40,
        "target_distribution": huge_distribution,
        "payload": {
            "row_sized_diagnostic": list(range(10_000))
        },
    }

    projected = _task_dataset_stats("temporal_link_prediction", stats)

    assert projected == {
        "version": 2,
        "kind": "tt_stats",
        "num_total_rows": 100,
        "num_train_rows": 80,
        "num_entities": 20,
        "num_targets": 10_000,
        "num_links": 40,
    }
    assert len(json.dumps(projected)) < 256


def test_node_task_stats_preserve_target_metadata() -> None:
    stats = {
        "version": 2,
        "kind": "tt_stats",
        "num_total_rows": 100,
        "num_train_rows": 80,
        "num_entities": 20,
        "num_targets": 2,
        "target_distribution": {
            "count": {
                "false": 60,
                "true": 20
            }
        },
    }

    assert _task_dataset_stats("binary_classification", stats) == stats


def test_diskgraph_task_manifest_loads_training_sidecars(tmp_path):
    bundle = tmp_path / "bundle"
    write_json(
        bundle / "manifest.json",
        {
            "artifact_root": str(bundle),
            "metadata_path": "metadata.json",
            "feature_schema_path": "feature_schema.json",
            "graph_runtime_metadata_path": "graph.json",
            "stats_path": "stats.json",
            "split_info_path": "split.json",
        },
    )
    write_json(
        bundle / "metadata.json",
        {
            "task_type": "static_link_prediction",
            "train_table_name": "transactions",
            "seed_node_type": "users",
            "target_node_type": "items",
            "train_table_columns": {
                "target_column": "TARGET"
            },
        },
    )
    write_json(bundle / "feature_schema.json", {"node_types": {}})
    write_json(bundle / "graph.json", {"graph_index_path": "/graph"})
    write_json(bundle / "stats.json", {"num_targets": 10})
    write_json(
        bundle / "split.json",
        {
            "stage_counts": {
                "train": 80,
                "val": 10
            },
            "stage_seed_times": {
                "train": [1, 9]
            },
        },
    )
    write_json(bundle / "runtime_manifest.json", {"stages": {}})

    manifest = DiskgraphTaskManifest.load(str(bundle / "manifest.json"))

    assert manifest.task_type == "static_link_prediction"
    assert manifest.train_table_name == "transactions"
    assert manifest.seed_node_type == "users"
    assert manifest.target_node_type == "items"
    assert manifest.train_table_columns == {"target_column": "TARGET"}
    assert manifest.stage_counts == {"train": 80, "val": 10}
    assert manifest.stage_seed_times == {"train": (1, 9)}


def make_spec(tmp_path, extension):
    return TaskDatasetMaterializeSpec(
        tt_manifest_path=str(extension / "manifest.json"),
        job_id="job-1",
        artifact_root=str(tmp_path / "bundle"),
        task_family="node_prediction",
        task_type="binary_classification",
        is_temporal=False,
        seed_node_type="users",
        train_table_name="train_table",
        entity_table_name="users",
        feature_schema={
            "node_types": {
                "users": {
                    "feature_columns": ["age"],
                    "mapper_key":
                    "users",
                    "time_column":
                    None,
                    "column_specs": [{
                        "column_name": "age",
                        "dtype": "int64",
                        "stype": "numerical",
                        "encoder": {
                            "name": "Numerical"
                        },
                        "stats": {
                            "MEAN": 42
                        },
                    }],
                }
            },
            "target_encoder_input": None,
        },
        graph_node_types=("users", ),
        graph_edge_types=(),
        required_stages=("train", "val"),
    )


def test_materialize_task_dataset_writes_trainer_bundle(tmp_path) -> None:
    graph_path, extension = setup_extension(tmp_path)
    spec = make_spec(tmp_path, extension)

    result = materialize_task_dataset(
        graph_index_id="graph/v1",
        graph_index_path=str(graph_path),
        spec=spec,
    )

    assert result.storage_kind == "diskgraph_table"
    assert os.path.isfile(result.manifest_path)
    assert os.path.isfile(result.runtime_manifest_path)
    split = json.loads((tmp_path / "bundle" / "split_info.json").read_text())
    assert split["stage_counts"] == {"train": 80, "val": 10, "test": 10}


def test_materialize_task_dataset_rejects_an_empty_required_stage(
        tmp_path) -> None:
    graph_path, extension = setup_extension(tmp_path)
    manifest_path = extension / "diskgraph_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["task_dataset"]["stages"]["test"]["row_count"] = 0
    write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="required stage.*'test'.*no rows"):
        materialize_task_dataset(
            graph_index_id="graph/v1",
            graph_index_path=str(graph_path),
            spec=replace(make_spec(tmp_path, extension),
                         required_stages=("train", "val", "test")),
        )


def test_materialize_link_task_writes_bounded_stats(tmp_path) -> None:
    graph_path, extension = setup_extension(tmp_path)
    source_stats_path = extension / "stats.json"
    source_stats = json.loads(source_stats_path.read_text())
    source_stats["num_links"] = 40
    source_stats["target_distribution"] = {
        "count": {
            f"item-{index}": index
            for index in range(10_000)
        }
    }
    source_stats["payload"] = {"unbounded": list(range(10_000))}
    write_json(source_stats_path, source_stats)
    node_schema = make_spec(tmp_path, extension).feature_schema["node_types"]
    spec = TaskDatasetMaterializeSpec(
        tt_manifest_path=str(extension / "manifest.json"),
        job_id="job-lp",
        artifact_root=str(tmp_path / "link-bundle"),
        task_family="link_prediction",
        task_type="temporal_link_prediction",
        is_temporal=True,
        seed_node_type="users",
        rhs_node_type="items",
        train_table_name="train_table",
        entity_table_name="users",
        feature_schema={
            "node_types": {
                "users": node_schema["users"],
                "items": {
                    "feature_columns": [],
                    "mapper_key": "items",
                    "time_column": None,
                    "column_specs": [],
                },
            },
            "target_encoder_input": None,
        },
        graph_node_types=("users", "items"),
        graph_edge_types=(),
    )

    materialize_task_dataset(
        graph_index_id="graph/v1",
        graph_index_path=str(graph_path),
        spec=spec,
    )

    written = json.loads((tmp_path / "link-bundle" / "stats.json").read_text())
    assert written["num_links"] == 40
    assert "target_distribution" not in written
    assert "payload" not in written
    assert os.path.getsize(tmp_path / "link-bundle" / "stats.json") < 4096


_NUMERICAL_TARGET_STATS = {
    "COUNT": 100,
    "NUM_NA": 0,
    "MEAN": 4.2,
    "MEDIAN": 4.2,
    "STD": 1.0,
    "MIN": 3.0,
    "MAX": 5.0,
    "KURTOSIS": 0.0,
    "QUANTILES": [3.5, 4.2, 4.8],
}
_SKEWED_TARGET_STATS = {
    **_NUMERICAL_TARGET_STATS,
    "MEAN": 30.0,
    "MEDIAN": 2.0,
    "STD": 50.0,
    "MIN": 0.0,
    "MAX": 900.0,
}


def write_target_stats(extension, stats, stype="kumo.numerical"):
    write_json(
        extension / "train_table_col_stats.json",
        {
            "version": 1,
            "kind": "tt_train_table_col_stats",
            "stats_format": "kumo_col_stats_v1",
            "stats": {
                f"TARGET1__{stype}__{name}__False": {
                    "value": value
                }
                for name, value in stats.items()
            },
        },
    )


def target_spec(tmp_path, extension, task_type, task_family):
    spec = make_spec(tmp_path, extension)
    schema = dict(spec.feature_schema)
    if task_family == "forecasting":
        schema["node_types"] = {
            **schema["node_types"],
            "train_table": {
                "feature_columns": [],
                "mapper_key": "train_table",
                "time_column": None,
                "column_specs": [],
            },
        }
    return replace(spec, task_type=task_type, task_family=task_family,
                   feature_schema=schema)


@pytest.mark.parametrize(
    ("task_type", "task_family", "stats", "encoder_name"),
    [
        ("multiclass_classification", "node_prediction", {
            "CATEGORY_COUNTS": [["a", 80], ["b", 20]]
        }, "Index"),
        ("regression", "node_prediction", _NUMERICAL_TARGET_STATS, "Numerical"),
        ("multilabel_ranking", "node_prediction", {
            "MULTI_CATEGORY_COUNTS": [["a", 80], ["b", 20]]
        }, "MultiCategorical"),
        ("forecasting", "forecasting", _NUMERICAL_TARGET_STATS, "Numerical"),
        ("forecasting", "forecasting", _SKEWED_TARGET_STATS, "MaxLogNumerical"),
    ],
)
def test_target_encoder_comes_from_materialized_target_not_source_column(
        tmp_path, task_type, task_family, stats, encoder_name) -> None:
    _, extension = setup_extension(tmp_path)
    write_target_stats(extension, stats, stype="kumo.generated")
    spec = target_spec(tmp_path, extension, task_type, task_family)

    target = resolve_target_encoder_input(spec, {}, str(extension))

    assert target["column_name"] == "TARGET"
    assert target["encoder"]["name"] == encoder_name
    assert target["stats"] == stats


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed_node_type", "catalog.schema.users"),
        ("entity_table_name", "uc.users"),
        ("rhs_node_type", "catalog.products"),
    ],
)
def test_task_dataset_spec_rejects_qualified_name_shortcuts(
        tmp_path, field, value) -> None:
    _, extension = setup_extension(tmp_path)
    values = make_spec(tmp_path, extension).__dict__.copy()
    values[field] = value

    with pytest.raises(ValueError, match="exact unqualified portable name"):
        TaskDatasetMaterializeSpec(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_type", "static_link_prediction", "does not belong"),
        ("feature_schema", {}, "keys must exactly match"),
        ("graph_edge_types",
         (("users", "owns", "products"), ), "must reference graph_node_types"),
    ],
)
def test_task_dataset_spec_rejects_unresolved_schema(tmp_path, field, value,
                                                     message) -> None:
    _, extension = setup_extension(tmp_path)
    values = make_spec(tmp_path, extension).__dict__.copy()
    values[field] = value

    with pytest.raises(ValueError, match=message):
        TaskDatasetMaterializeSpec(**values)


@pytest.mark.parametrize("stats", [{}, {"stats": None}])
def test_spec_rejects_malformed_target(tmp_path, stats) -> None:
    _, extension = setup_extension(tmp_path)
    target = {
        "column_name": "TARGET",
        "dtype": "float",
        "stype": "numerical",
        "encoder": {
            "name": "Numerical"
        },
        **stats,
    }
    values = make_spec(tmp_path, extension).__dict__.copy()
    values["feature_schema"] = {
        **values["feature_schema"], "target_encoder_input": target
    }

    with pytest.raises(ValueError, match="target_encoder_input must be None"):
        TaskDatasetMaterializeSpec(**values)


def test_task_dataset_spec_accepts_exact_forecasting_schema(tmp_path) -> None:
    _, extension = setup_extension(tmp_path)
    values = make_spec(tmp_path, extension).__dict__.copy()
    values.update(
        task_family="forecasting",
        task_type="forecasting",
        feature_schema={
            "node_types": {
                "users": {
                    "feature_columns": ["age"],
                    "mapper_key":
                    "users",
                    "time_column":
                    None,
                    "column_specs": [{
                        "column_name": "age",
                        "dtype": "int64",
                        "stype": "numerical",
                        "encoder": {
                            "name": "Numerical"
                        },
                        "stats": {
                            "MEAN": 42
                        },
                    }],
                },
                "train_table": {
                    "feature_columns": ["future_labels"],
                    "mapper_key":
                    "train_table",
                    "time_column":
                    None,
                    "column_specs": [{
                        "column_name": "future_labels",
                        "dtype": "floatlist",
                        "stype": "sequence",
                        "encoder": {
                            "name": "NumericalList"
                        },
                        "stats": {},
                    }],
                },
            },
            "target_encoder_input": None,
        },
    )

    TaskDatasetMaterializeSpec(**values)
