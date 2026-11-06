# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank product relationships without a future prediction window.

Use the original prepared CSV files and local scratch. Use the suggested FAST plan for the static recommendation task.

Install with `pip install kumo-engine`. Use Linux with GPU compute and local SSD scratch.
Prediction files retain entity IDs for alignment with the scoring population.
See https://docs.nvidia.com/sdgm/kumo-engine/ for setup and modeling guidance.
"""

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from kumoapi.graph import Edge
from kumoapi.jobs import PredictionArtifactType, WriteMode
from kumoapi.model_plan import RunMode
from kumoapi.typing import Stype

from kumo.engine import (
    FileConnector,
    Graph,
    PredictiveQuery,
    Table,
    Trainer,
    configure,
)
from kumo.engine.trainer.output_config import (
    ModelOutputConfig,
    PredictionOutputConfig,
)

QUERY = "PREDICT LIST_DISTINCT(SalesOrderDetail.ProductID) RANK TOP 5 FOR EACH Customer.CustomerID"
TABLES = ("Customer", "Product", "SalesOrderHeader", "SalesOrderDetail")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def source_hashes(root):
    return {
        f"{name}.csv":
        hashlib.sha256((root / f"{name}.csv").read_bytes()).hexdigest()
        for name in TABLES
    }


def model_hashes(root):
    return {
        str(path.relative_to(root)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def build_graph(input_dir):
    connector = FileConnector(root_dir=str(input_dir))
    customers = Table.from_source_table(
        connector["Customer"], ["CustomerID", "StoreID", "TerritoryID"],
        primary_key="CustomerID")
    customers = customers.with_column("StoreID", stype=Stype.categorical)
    customers = customers.with_column("TerritoryID", stype=Stype.categorical)
    products = Table.from_source_table(connector["Product"], [
        "ProductID", "Name", "Color", "ProductLine", "Class", "StandardCost",
        "ListPrice"
    ], primary_key="ProductID")
    products = products.with_column("Name", stype=Stype.text)
    headers = Table.from_source_table(connector["SalesOrderHeader"], [
        "SalesOrderID", "CustomerID", "OnlineOrderFlag", "TerritoryID",
        "SubTotal", "TotalDue"
    ], primary_key="SalesOrderID")
    headers = headers.with_column("CustomerID", stype=Stype.ID)
    details = Table.from_source_table(connector["SalesOrderDetail"], [
        "SalesOrderDetailID", "SalesOrderID", "CustomerID", "ProductID",
        "OrderQty", "UnitPrice", "LineTotal"
    ], primary_key="SalesOrderDetailID")
    details = details.with_column("SalesOrderID", stype=Stype.ID)
    details = details.with_column("CustomerID", stype=Stype.ID)
    details = details.with_column("ProductID", stype=Stype.ID)
    return Graph(
        tables=[customers, products, headers, details],
        edges=[
            Edge(src_table="SalesOrderHeader", fkey="CustomerID",
                 dst_table="Customer"),
            Edge(src_table="SalesOrderDetail", fkey="SalesOrderID",
                 dst_table="SalesOrderHeader"),
            Edge(src_table="SalesOrderDetail", fkey="CustomerID",
                 dst_table="Customer"),
            Edge(src_table="SalesOrderDetail", fkey="ProductID",
                 dst_table="Product"),
        ],
    )


def check_labels(training_table, source, customers, products):
    training = pd.read_parquet(training_table.train_table_path)
    details = pd.read_csv(source / "SalesOrderDetail.csv")
    purchases = details
    expected = purchases.groupby("CustomerID").ProductID.agg(set)
    assert set(training.ENTITY1).issubset(customers)
    assert not training.ENTITY1.duplicated().any()
    assert training.TIMESTAMP.eq(pd.Timestamp("1970-01-01")).all()
    for _, row in training.iterrows():
        actual = set(row.TARGET1)
        assert actual == expected.get(row.ENTITY1, set())
        assert actual <= products
        frequency = dict(
            zip(row.TARGET1, row["__kumo_internal_frequency_column__"]))
        expected_frequency = (
            purchases.loc[purchases.CustomerID.eq(row.ENTITY1),
                          "ProductID"].value_counts().to_dict())
        assert frequency == expected_frequency
    return {
        "rows": len(training),
        "distinct_links": sum(map(len, training.TARGET1)),
        "source_events": len(details),
        "label_mismatches": 0,
        "frequency_mismatches": 0,
        "timestamp": "Unix epoch sentinel, no temporal window",
    }


def run(args):
    source, scratch, output = (getattr(args, key).expanduser().resolve()
                               for key in ("input_dir", "scratch_dir",
                                           "output_dir"))
    model = args.model_dir.expanduser().resolve(
    ) if args.mode == "predict" else None
    paths = [source, scratch, output]
    if model is not None:
        if not model.is_dir():
            raise ValueError("The saved model directory must exist.")
        paths.append(model)
    for first, second in combinations(paths, 2):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError(
                "Input, scratch, model, and output paths must be separate.")
    if args.num_classes_to_return <= 0:
        raise ValueError("Use a positive --num-classes-to-return value.")
    before = source_hashes(source)
    model_before = model_hashes(model) if model else None
    users = set(pd.read_csv(source / "Customer.csv").CustomerID)
    products = set(pd.read_csv(source / "Product.csv").ProductID)
    output.mkdir(parents=True, exist_ok=False)
    configure(scratch_dir=scratch)
    graph = build_graph(source)
    graph.validate()
    snapshot = graph.snapshot()
    query = PredictiveQuery(graph, QUERY)
    query.validate()
    destination = FileConnector(root_dir=str(output))
    summary = {
        "mode": args.mode,
        "query": QUERY,
        "task_type": str(query.get_task_type()),
        "snapshot_id": snapshot.id,
        "snapshot_index": str(snapshot.index_path),
        "source_sha256": before,
    }
    if args.mode == "fit":
        training_table = query.generate_training_table(snapshot)
        labels = check_labels(training_table, source, users, products)
        save_json(output / "training-labels.json", labels)
        plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
        save_json(output / "suggested-plan.json", plan.as_json())
        save_json(output / "training-plan.json", plan.as_json())
        trainer = Trainer(plan)
        trained = trainer.fit(
            training_table,
            snapshot=snapshot,
            export_config=ModelOutputConfig(connector=destination,
                                            path="model"),
        )
        assert trained.metrics and all(
            math.isfinite(float(v)) for v in trained.metrics.values())
        save_json(output / "training-metrics.json", trained.metrics)
        summary.update(
            training_rows=labels["rows"],
            metrics=trained.metrics,
            model_dir=str(output / "model"),
        )
    else:
        trainer = Trainer.load(model)
        summary["model_dir"] = str(model)
    # Batch predictions
    # Generate label-free scoring rows for the query population.
    table = query.generate_prediction_table(snapshot)
    generated = pd.read_parquet(table.table_path)
    assert len(generated) >= 2, "Choose at least two scoring rows."
    assert len(generated) == len(users) and set(generated.ENTITY1) == users
    assert generated.TIMESTAMP.eq(pd.Timestamp("1970-01-01")).all()
    config = PredictionOutputConfig(
        connector=destination,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name="predictions.parquet",
        write_mode=WriteMode.OVERWRITE,
    )
    frame = trainer.predict(table, snapshot=snapshot,
                            num_classes_to_return=args.num_classes_to_return,
                            output_config=config).frame
    print(frame.head(8).to_string(index=False), flush=True)
    assert list(frame.columns) == ["ENTITY", "CLASS", "SCORE"]
    assert frame.notna().all().all()
    assert set(frame.ENTITY) == users
    assert frame.groupby("ENTITY").size().eq(args.num_classes_to_return).all()
    assert not frame.duplicated(["ENTITY", "CLASS"]).any()
    assert set(frame.CLASS) <= products
    assert np.isfinite(frame.SCORE.to_numpy()).all()
    assert all(group.SCORE.is_monotonic_decreasing
               for _, group in frame.groupby("ENTITY"))
    pd.testing.assert_frame_equal(
        frame, pd.read_parquet(output / "predictions.parquet"))
    output_hash = hashlib.sha256(
        (output / "predictions.parquet").read_bytes()).hexdigest()
    controls = []
    for topk in (0, -1):
        try:
            trainer.predict(table, snapshot=snapshot,
                            num_classes_to_return=topk, output_config=config)
        except ValueError as error:
            assert "positive" in str(error)
            assert (output_hash == hashlib.sha256(
                (output / "predictions.parquet").read_bytes()).hexdigest())
            controls.append({
                "top_k": topk,
                "error": str(error),
                "output_unchanged": True
            })
        else:
            raise AssertionError("Expected nonpositive candidate-count refusal")
    save_json(output / "controls.json", controls)
    assert source_hashes(source) == before
    if model is not None:
        assert model_hashes(model) == model_before
    summary.update(
        prediction_rows=len(frame),
        prediction_columns=list(frame.columns),
        candidates_per_customer=args.num_classes_to_return,
        candidate_pool_size=len(products),
        score_min=float(frame.SCORE.min()),
        score_max=float(frame.SCORE.max()),
        source_files_unchanged=True,
    )
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("fit", "predict"):
        command = modes.add_parser(mode)
        command.add_argument("--input-dir", type=Path, required=True)
        command.add_argument("--scratch-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--num-classes-to-return", type=int, default=5)
        if mode == "predict":
            command.add_argument("--model-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
