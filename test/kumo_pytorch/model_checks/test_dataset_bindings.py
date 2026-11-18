# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unrelated tables, identities and clocks must use the same execution path."""

import hashlib
from datetime import datetime, timezone
from test.kumo_pytorch.data import prepare_binding
from test.kumo_pytorch.nightly import run_case

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def shop(tmp_path, family):
    root = tmp_path / "shop"
    root.mkdir()
    stamp = datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc).timestamp()
    pq.write_table(
        pa.table(
            {
                "customer": [10, 20],
                "canonical": ["customer-a", "customer-b"],
                "joined": pa.array([stamp - 86400] * 2, type=pa.timestamp("s")),
            }
        ),
        root / "customers.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "purchase": [1, 2, 3, 4],
                "buyer": [10, 20, 10, 20],
                "when": pa.array(
                    [stamp - n for n in (100, 80, 50, 20)], type=pa.timestamp("s")
                ),
                "when_seconds": [int(stamp) - n for n in (100, 80, 50, 20)],
                "value": [3.0, 5.0, 7.0, 9.0],
                "category": ["a", "b", "b", "a"],
            }
        ),
        root / "purchases.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "example": ["tr-a", "tr-b", "tr-c", "va-a", "te-b"],
                "who": [
                    "customer-a",
                    "customer-b",
                    "customer-a",
                    "customer-a",
                    "customer-b",
                ],
                "prediction": pa.array(
                    [int(stamp * 1000)] * 5, type=pa.timestamp("ms")
                ),
                "partition": ["train", "train", "train", "valid", "test"],
                "will_return": pa.array([1, 0, 1, 1, 0], type=pa.int32()),
            }
        ),
        root / "cohort.parquet",
    )
    files = {
        p.name: {
            "key": p.name,
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in root.glob("*.parquet")
    }
    binding = {
        "ref": "fixture-shop",
        "binding_id": "fixture:shop-return",
        "campaigns": ["training", "full"],
        "selector": {
            "dataset_id": "shop",
            "dataset_version": "one",
            "population": "customers",
            "anchor_version": "release-A",
            "workload_id": "return-prediction",
        },
        "input_variant": "purchases",
        "representation": family,
        "split_column": "partition",
        "targets": {
            "label": {
                "column": "will_return",
                "dtype": "INTEGER",
                "non_null_required": True,
                "all_true_columns": [],
            }
        },
        "task": {
            "kind": "stored",
            "targets": {
                "label": {"column": "will_return", "dtype": "INTEGER", "kind": "binary"}
            },
        },
        "anchors": {
            "sample_id": "example",
            "entity_id": "who",
            "time": "prediction",
            "time_unit": "ms",
            "split": "partition",
            "version": "release-A",
            "execution_offset_seconds": -1
            if family in {"graphsage", "combined"}
            else 0,
        },
        "inputs": {
            "variant": "purchases",
            "root_table": "customers",
            "root_type": "customer",
            "families": [
                [
                    "purchase",
                    ["purchases"],
                    "when",
                    [
                        ["value", "value", "numerical"],
                        ["category", "category", "categorical"],
                    ],
                ]
            ],
            "graph": {
                "node_table": "purchases",
                "relations": ["purchases"],
                "numerical": ["value"],
                "categorical": ["category"],
                "timestamp": "when_seconds",
            },
            "aggregates": [["purchase_count", ["purchases"], "when", None]],
        },
        "dataset": {
            "inventory": "shop",
            "identity_path": "customers.parquet",
            "identity_column": "canonical",
            "identity_id_column": "customer",
            "anchor_path": "cohort.parquet",
            "history_bound": {"table": "purchases", "entity": "buyer"},
            "tables": {
                "customers": {
                    "path": "customers.parquet",
                    "key": "customer",
                    "time": "joined",
                },
                "purchases": {
                    "path": "purchases.parquet",
                    "key": "purchase",
                    "time": "when",
                },
            },
            "relations": [["purchases", "customers", "customer", "purchases", "buyer"]],
        },
    }
    if family in {"pragma", "nuformer"}:
        del binding["inputs"]["graph"]
        del binding["inputs"]["aggregates"]
    elif family == "graphsage":
        del binding["inputs"]["families"]
        del binding["inputs"]["aggregates"]
    return root, files, binding


@pytest.mark.parametrize("family", ("graphsage", "combined", "nuformer", "pragma"))
def test_unrelated_dataset_anchors_and_models_share_runner(tmp_path, family):
    root, files, binding = shop(tmp_path, family)
    output = tmp_path / "output"
    output.mkdir()
    dataset, anchors, population, mapping, capacity = prepare_binding(
        root, output, binding, files, limit=None
    )
    projected = pq.read_table(output / "anchors.parquet").to_pylist()
    assert [r["entity_id"] for r in projected] == [10, 20, 10, 10, 20]
    assert [r["label"] for r in projected] == [1, 0, 1, 1, 0]
    assert projected[0]["prediction_time_ns"] % (86400 * 10**9) != 0
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="training",
        capacity=capacity,
        mapping=mapping,
        batch_size=3,
        seed=17,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    assert report["reproducible"]
    assert [r["completed_updates"] for r in report["repeats"]] == [100, 100]
    assert {s: row["delivered"] for s, row in report["splits"].items()} == {
        "train": 3,
        "valid": 1,
        "test": 1,
    }


@pytest.mark.parametrize("family", ("graphsage", "combined", "nuformer", "pragma"))
def test_unrelated_dataset_uses_shared_benchmark(tmp_path, monkeypatch, family):
    from test.kumo_pytorch import bindings, data
    from test.kumo_pytorch.benchmarking import run
    from test.kumo_pytorch.benchmarking.kumo import factory

    root, files, binding = shop(tmp_path, family)
    monkeypatch.setattr(bindings, "load_bindings", lambda: {binding["ref"]: binding})
    monkeypatch.setattr(
        data,
        "load_manifest",
        lambda: {
            "datasets": {
                "shop": {
                    "dataset_id": "shop",
                    "dataset_version": "one",
                    "population": "customers",
                    "files": list(files.values()),
                }
            }
        },
    )
    result = run(
        factory,
        {
            "binding": binding["ref"],
            "data_dir": str(root),
            "anchors_per_split": 1,
            "batch_size": 1,
            "threads": 1,
            "training": True,
        },
        tmp_path / "benchmark",
        repetitions=1,
        epochs=2,
    )
    assert result["execution_status"] == "completed", result["errors"]
    assert all(row["metrics"]["real_samples"] == 1 for row in result["epochs"])
    assert all(row["metrics"]["events"] > 0 for row in result["epochs"])
    assert all(row["complete"] for row in result["native_iterations"])


@pytest.mark.parametrize("ref", ("E01", "E02", "E03", "E04", "E05"))
def test_existing_bindings_use_shared_population_runner(tmp_path, monkeypatch, ref):
    from test.kumo_pytorch import data
    from test.kumo_pytorch.bindings import load_bindings

    from .test_benchmark_aml import source

    root = source.__wrapped__(tmp_path, monkeypatch)
    binding = load_bindings()[ref]
    output = tmp_path / "shared-population"
    output.mkdir()
    dataset, anchors, population, mapping, capacity = prepare_binding(
        root, output, binding, data.verify_dataset(root, binding), limit=None
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="full",
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    assert report["full_population_verified"]
    assert {split: row["delivered"] for split, row in report["splits"].items()} == {
        "train": 1,
        "valid": 1,
        "test": 1,
    }


@pytest.mark.parametrize("variant", ("same-relation", "other-table", "other-join"))
def test_multiple_history_families_keep_all_events(tmp_path, monkeypatch, variant):
    from copy import deepcopy
    from test.kumo_pytorch import bindings, data
    from test.kumo_pytorch.benchmarking import run
    from test.kumo_pytorch.benchmarking.kumo import factory

    root, files, binding = shop(tmp_path, "nuformer")
    other = deepcopy(binding["inputs"]["families"][0])
    other[0] = "second_purchase"
    if variant == "other-table":
        path = root / "other.parquet"
        path.write_bytes((root / "purchases.parquet").read_bytes())
        files[path.name] = {
            "key": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        binding["dataset"]["tables"]["other"] = {
            **binding["dataset"]["tables"]["purchases"],
            "path": path.name,
        }
        binding["dataset"]["relations"].append(
            ["other_purchases", "customers", "customer", "other", "buyer"]
        )
        other[1] = ["other_purchases"]
    if variant == "other-join":
        path = root / "purchases.parquet"
        table = pq.read_table(path).append_column("seller", pa.array([10, 10, 10, 10]))
        pq.write_table(table, path)
        files[path.name].update(
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        binding["dataset"]["relations"].append(
            ["sold", "customers", "customer", "purchases", "seller"]
        )
        other[1] = ["sold"]
    binding["inputs"]["families"].append(other)
    monkeypatch.setattr(bindings, "load_bindings", lambda: {binding["ref"]: binding})
    monkeypatch.setattr(
        data,
        "load_manifest",
        lambda: {
            "datasets": {
                "shop": {
                    "dataset_id": "shop",
                    "dataset_version": "one",
                    "population": "customers",
                    "files": list(files.values()),
                }
            }
        },
    )
    result = run(
        factory,
        {
            "binding": binding["ref"],
            "data_dir": str(root),
            "anchors_per_split": 1,
            "batch_size": 1,
            "threads": 1,
            "training": False,
        },
        tmp_path / "benchmark",
        repetitions=1,
        epochs=2,
    )
    assert result["execution_status"] == "completed", result["errors"]
    # Two rows in each family: six encoded fields per row plus three markers.
    assert [row["metrics"]["events"] for row in result["epochs"]] == (
        [6, 6] if variant == "other-join" else [4, 4]
    )
    assert [row["metrics"]["tokens"] for row in result["epochs"]] == (
        [41, 41] if variant == "other-join" else [27, 27]
    )


@pytest.mark.parametrize("ref,family", (("E03", "nuformer"), ("E04", "pragma")))
def test_task_definition_cannot_change_with_model(ref, family):
    from test.kumo_pytorch.bindings import load_bindings
    from test.kumo_pytorch.population_anchors import validate_binding

    binding = load_bindings()[ref]
    binding["representation"] = family
    with pytest.raises(ValueError, match="incompatible"):
        validate_binding(binding)


def test_graph_bound_uses_its_actual_relation(tmp_path):
    root, files, binding = shop(tmp_path, "graphsage")
    path = root / "purchases.parquet"
    table = pq.read_table(path).append_column("seller", pa.array([10, 10, 10, 10]))
    pq.write_table(table, path)
    files[path.name].update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    binding["dataset"]["relations"].append(
        ["sold", "customers", "customer", "purchases", "seller"]
    )
    binding["inputs"]["graph"]["relations"] = ["sold"]
    output = tmp_path / "graph"
    output.mkdir()
    *_, capacity = prepare_binding(root, output, binding, files, limit=None)
    assert capacity == 4  # buyer degree is only two; seller's four rows must survive.


def test_unrelated_multitable_graph_uses_the_shared_model_and_runner(tmp_path):
    root, files, binding = shop(tmp_path, "combined")
    path = root / "refunds.parquet"
    path.write_bytes((root / "purchases.parquet").read_bytes())
    files[path.name] = {
        "key": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    binding["dataset"]["tables"]["refunds"] = {
        **binding["dataset"]["tables"]["purchases"],
        "path": path.name,
    }
    binding["dataset"]["relations"].append(
        ["refunds", "customers", "customer", "refunds", "buyer"]
    )
    fields = binding["inputs"]["graph"]
    binding["inputs"]["graph"] = {
        "nodes": {"purchases": fields, "refunds": fields},
        "relations": {"purchases": "purchases", "refunds": "refunds"},
        "reverse_hop": False,
    }
    output = tmp_path / "multi-graph"
    output.mkdir()
    dataset, anchors, population, mapping, capacity = prepare_binding(
        root, output, binding, files, limit=None
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="training",
        capacity=capacity,
        mapping=mapping,
        batch_size=3,
        seed=17,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    assert report["reproducible"]
    assert [row["completed_updates"] for row in report["repeats"]] == [100, 100]
