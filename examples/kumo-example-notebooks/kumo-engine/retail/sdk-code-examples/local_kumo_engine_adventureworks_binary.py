# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict whether an active customer orders again in the next 90 days.

This script uses the local Python API from the Linux setup. Use the four
prepared AdventureWorks CSV files, local scratch, and a new output
directory. Training uses the suggested FAST model plan.

Install with `pip install kumo-engine`. Use Linux with GPU compute and local SSD scratch.
Prediction files retain entity IDs for alignment with the scoring population.
See https://docs.nvidia.com/sdgm/kumo-engine/ for setup and modeling guidance.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from kumoapi.graph import Edge
from kumoapi.jobs import MetadataField, PredictionArtifactType, WriteMode
from kumoapi.model_plan import RunMode
from kumoapi.typing import Dtype, Stype

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

TABLES = ("Customer", "Product", "SalesOrderHeader", "SalesOrderDetail")
QUERY = ("PREDICT COUNT(SalesOrderHeader.*, 0, 90, days) > 0 "
         "FOR EACH Customer.CustomerID "
         "WHERE COUNT(SalesOrderHeader.*, -365, 0, days) > 0")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def build_graph(input_dir):
    connector = FileConnector(root_dir=str(input_dir))
    customer = Table.from_source_table(
        connector["Customer"],
        ["CustomerID", "StoreID", "TerritoryID"],
        primary_key="CustomerID",
    )
    customer = customer.with_column("StoreID", stype=Stype.categorical)
    customer = customer.with_column("TerritoryID", stype=Stype.categorical)
    product = Table.from_source_table(
        connector["Product"],
        [
            "ProductID", "Name", "Color", "ProductLine", "Class",
            "StandardCost", "ListPrice"
        ],
        primary_key="ProductID",
    )
    product = product.with_column("Name", stype=Stype.text)
    for column in ("Color", "ProductLine", "Class"):
        product = product.with_column(column, stype=Stype.categorical)
    for column in ("StandardCost", "ListPrice"):
        product = product.with_column(column, stype=Stype.numerical)
    header = Table.from_source_table(
        connector["SalesOrderHeader"],
        [
            "SalesOrderID", "CustomerID", "OrderDate", "OnlineOrderFlag",
            "TerritoryID", "SubTotal", "TotalDue"
        ],
        primary_key="SalesOrderID",
        time_column="OrderDate",
    )
    header = header.with_column("CustomerID", stype=Stype.ID)
    header = header.with_column("OrderDate", dtype=Dtype.date)
    header = header.with_column("OnlineOrderFlag", stype=Stype.categorical)
    header = header.with_column("TerritoryID", stype=Stype.categorical)
    header = header.with_column("SubTotal", stype=Stype.numerical)
    header = header.with_column("TotalDue", stype=Stype.numerical)
    detail = Table.from_source_table(
        connector["SalesOrderDetail"],
        [
            "SalesOrderDetailID", "SalesOrderID", "ProductID", "ModifiedDate",
            "OrderQty", "UnitPrice", "LineTotal"
        ],
        primary_key="SalesOrderDetailID",
        time_column="ModifiedDate",
    )
    detail = detail.with_column("SalesOrderID", stype=Stype.ID)
    detail = detail.with_column("ProductID", stype=Stype.ID)
    detail = detail.with_column("ModifiedDate", dtype=Dtype.date)
    for column in ("OrderQty", "UnitPrice", "LineTotal"):
        detail = detail.with_column(column, stype=Stype.numerical)
    return Graph(
        tables=[customer, product, header, detail],
        edges=[
            Edge(src_table="SalesOrderHeader", fkey="CustomerID",
                 dst_table="Customer"),
            Edge(src_table="SalesOrderDetail", fkey="SalesOrderID",
                 dst_table="SalesOrderHeader"),
            Edge(src_table="SalesOrderDetail", fkey="ProductID",
                 dst_table="Product"),
        ],
    )


def active_customers(orders, timestamp):
    start = timestamp - pd.Timedelta(days=365)
    mask = (orders.OrderDate > start) & (orders.OrderDate <= timestamp)
    return set(orders.loc[mask, "CustomerID"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-time", default="2025-06-30T00:00:00")
    args = parser.parse_args()
    source, scratch, output = [
        path.expanduser().resolve()
        for path in (args.input_dir, args.scratch_dir, args.output_dir)
    ]
    for first, second in ((source, scratch), (source, output), (scratch,
                                                                output)):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError(
                "Input, scratch, and output directories must be separate.")
    prediction_time = datetime.fromisoformat(args.prediction_time)
    if prediction_time.tzinfo is not None:
        raise ValueError("Use a naive UTC timestamp matching the CSV dates.")
    output.mkdir(parents=True, exist_ok=False)
    source_files = [source / f"{name}.csv" for name in TABLES]
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    configure(scratch_dir=scratch)
    graph = build_graph(source)
    graph.validate()
    snapshot = graph.snapshot()
    query = PredictiveQuery(graph, QUERY)
    query.validate()
    print("Task type:", query.get_task_type(), flush=True)
    training_table = query.generate_training_table(snapshot)
    training = pd.read_parquet(training_table.train_table_path)
    orders = pd.read_csv(source / "SalesOrderHeader.csv",
                         parse_dates=["OrderDate"])
    customers = pd.read_csv(source / "Customer.csv")
    assert not training.duplicated(["ENTITY1", "TIMESTAMP"]).any()
    windows = []
    for timestamp, rows in training.groupby("TIMESTAMP"):
        end = timestamp + pd.Timedelta(days=90)
        future = orders.loc[(orders.OrderDate > timestamp) & (
            orders.OrderDate <= end)].groupby("CustomerID").size()
        expected = rows.ENTITY1.map(future).fillna(0).gt(0)
        assert expected.eq(rows.TARGET1).all()
        assert set(rows.ENTITY1) == active_customers(orders, timestamp)
        windows.append({
            "timestamp": str(timestamp),
            "rows": len(rows),
            "target_sum": int(rows.TARGET1.sum()),
            "mismatches": 0
        })
    assert windows
    save_json(
        output / "training-labels.json", {
            "rows": len(training),
            "columns": list(training.columns),
            "windows": windows,
            "interval": "(timestamp, timestamp + 90 days]",
            "source_sha256": source_hashes,
        })
    # Batch predictions
    # Generate label-free scoring rows for the query population.
    prediction_table = query.generate_prediction_table(
        snapshot, prediction_time=prediction_time)
    generated = pd.read_parquet(prediction_table.table_path)
    assert len(generated) >= 2, "Choose at least two scoring rows."
    expected_population = active_customers(orders,
                                           pd.Timestamp(prediction_time))
    assert set(generated.ENTITY1) == expected_population
    assert set(generated.ENTITY1).issubset(set(customers.CustomerID))
    assert generated.TIMESTAMP.eq(pd.Timestamp(prediction_time)).all()
    plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
    save_json(output / "suggested-plan.json", plan.as_json())
    save_json(output / "training-plan.json", plan.as_json())
    destination = FileConnector(root_dir=str(output))
    trainer = Trainer(plan)
    trained = trainer.fit(
        training_table,
        snapshot=snapshot,
        export_config=ModelOutputConfig(connector=destination, path="model"),
    )
    assert trained.metrics
    assert all(
        math.isfinite(float(value)) for value in trained.metrics.values())
    save_json(output / "training-metrics.json", trained.metrics)
    prediction_runs = []
    for threshold in (0.0, 0.5, 1.0):
        filename = f"predictions-threshold-{threshold:.1f}.parquet"
        predicted = trainer.predict(
            prediction_table,
            snapshot=snapshot,
            binary_classification_threshold=threshold,
            output_config=PredictionOutputConfig(
                connector=destination,
                artifact_type=PredictionArtifactType.PREDICTIONS,
                extra_fields=[MetadataField.ANCHOR_TIMESTAMP],
                table_name=filename, write_mode=WriteMode.OVERWRITE),
        ).frame
        print(predicted.head(8).to_string(index=False), flush=True)
        pd.testing.assert_frame_equal(predicted,
                                      pd.read_parquet(output / filename))
        assert set(predicted.ENTITY) == expected_population
        assert predicted.ANCHOR_TIMESTAMP.eq(
            pd.Timestamp(prediction_time)).all()
        assert np.isfinite(predicted.True_PROB.to_numpy()).all()
        assert np.allclose(predicted.True_PROB + predicted.False_PROB, 1,
                           atol=1e-6)
        assert predicted.TARGET_PRED.eq(predicted.True_PROB.gt(threshold)).all()
        prediction_runs.append({
            "threshold": threshold,
            "file": filename,
            "rows": len(predicted),
            "true_labels": int(predicted.TARGET_PRED.sum())
        })
    assert source_hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    summary = {
        "query": QUERY,
        "task_type": str(query.get_task_type()),
        "prediction_time": prediction_time.isoformat(),
        "training_rows": len(training),
        "metrics": trained.metrics,
        "prediction_runs": prediction_runs,
        "source_files_unchanged": True,
        "working_model": str(trained.bundle_dir),
        "published_model": str(output / "model"),
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
