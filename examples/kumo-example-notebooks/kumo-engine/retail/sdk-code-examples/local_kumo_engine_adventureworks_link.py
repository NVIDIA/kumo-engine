"""Rank products each customer may purchase in the next 90 days.

This script uses the local Python API from the Linux setup.
Use the four prepared AdventureWorks CSV files, local scratch, and a new output
directory. Training uses the suggested FAST model plan.

Install with `pip install kumo-engine`. Use Linux with GPU compute and local SSD scratch.
Prediction files retain entity IDs for alignment with the scoring population.
See https://docs.nvidia.com/sdgm/kumo-engine/ for setup and modeling guidance.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from kumo.engine import FileConnector, Graph, PredictiveQuery, Table, Trainer, configure
from kumo.engine.trainer.output_config import ModelOutputConfig, PredictionOutputConfig
from kumoapi.graph import Edge
from kumoapi.jobs import MetadataField, PredictionArtifactType, WriteMode
from kumoapi.model_plan import RunMode
from kumoapi.typing import Dtype, Stype


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def build_graph(input_dir):
    connector = FileConnector(root_dir=str(input_dir))
    customer = Table.from_source_table(connector["Customer"], ["CustomerID", "StoreID", "TerritoryID"], primary_key="CustomerID")
    customer = customer.with_column("StoreID", stype=Stype.categorical)
    customer = customer.with_column("TerritoryID", stype=Stype.categorical)
    product = Table.from_source_table(connector["Product"], ["ProductID", "Name", "Color", "ProductLine", "Class", "StandardCost", "ListPrice"], primary_key="ProductID")
    product = product.with_column("Name", stype=Stype.text)
    for column in ("Color", "ProductLine", "Class"):
        product = product.with_column(column, stype=Stype.categorical)
    for column in ("StandardCost", "ListPrice"):
        product = product.with_column(column, stype=Stype.numerical)
    header = Table.from_source_table(connector["SalesOrderHeader"], ["SalesOrderID", "CustomerID", "OrderDate", "OnlineOrderFlag", "TerritoryID", "SubTotal", "TotalDue"], primary_key="SalesOrderID", time_column="OrderDate")
    header = header.with_column("CustomerID", stype=Stype.ID)
    header = header.with_column("OrderDate", dtype=Dtype.date)
    header = header.with_column("OnlineOrderFlag", stype=Stype.categorical)
    header = header.with_column("TerritoryID", stype=Stype.categorical)
    header = header.with_column("SubTotal", stype=Stype.numerical)
    header = header.with_column("TotalDue", stype=Stype.numerical)
    detail = Table.from_source_table(connector["SalesOrderDetail"], ["SalesOrderDetailID", "SalesOrderID", "CustomerID", "ProductID", "ModifiedDate", "OrderQty", "UnitPrice", "LineTotal"], primary_key="SalesOrderDetailID", time_column="ModifiedDate")
    detail = detail.with_column("SalesOrderID", stype=Stype.ID)
    detail = detail.with_column("CustomerID", stype=Stype.ID)
    detail = detail.with_column("ProductID", stype=Stype.ID)
    detail = detail.with_column("ModifiedDate", dtype=Dtype.date)
    for column in ("OrderQty", "UnitPrice", "LineTotal"):
        detail = detail.with_column(column, stype=Stype.numerical)
    return Graph(
        tables=[customer, product, header, detail],
        edges=[
            Edge(src_table="SalesOrderHeader", fkey="CustomerID", dst_table="Customer"),
            Edge(src_table="SalesOrderDetail", fkey="SalesOrderID", dst_table="SalesOrderHeader"),
            Edge(src_table="SalesOrderDetail", fkey="CustomerID", dst_table="Customer"),
            Edge(src_table="SalesOrderDetail", fkey="ProductID", dst_table="Product"),
        ],
    )



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-time", default="2025-06-30T00:00:00")
    args = parser.parse_args()
    source, scratch, output = [path.expanduser().resolve() for path in
                               (args.input_dir, args.scratch_dir, args.output_dir)]
    for first, second in ((source, scratch), (source, output), (scratch, output)):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError("Input, scratch, and output directories must be separate.")
    prediction_time = datetime.fromisoformat(args.prediction_time)
    if prediction_time.tzinfo is not None:
        raise ValueError("Use a naive timestamp matching the sample CSV timestamps.")
    output.mkdir(parents=True, exist_ok=False)
    source_files = [source / f"{name}.csv" for name in
                    ("Customer", "Product", "SalesOrderHeader", "SalesOrderDetail")]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    configure(scratch_dir=scratch)
    graph = build_graph(source)
    graph.validate()
    snapshot = graph.snapshot()
    text = ("PREDICT LIST_DISTINCT(SalesOrderDetail.ProductID, 0, 90, days) "
            "RANK TOP 5 FOR EACH Customer.CustomerID")
    query = PredictiveQuery(graph, text)
    query.validate()
    print("Task type:", query.get_task_type(), flush=True)
    training_table = query.generate_training_table(snapshot)
    training = pd.read_parquet(training_table.train_table_path)
    details = pd.read_csv(source / "SalesOrderDetail.csv", parse_dates=["ModifiedDate"])
    customers = pd.read_csv(source / "Customer.csv")
    assert not training.duplicated(["ENTITY1", "TIMESTAMP"]).any()
    classes = set(pd.read_csv(source / "Product.csv").ProductID)
    empty_labels = 0
    multi_labels = 0
    windows = []
    for timestamp, rows in training.groupby("TIMESTAMP"):
        end = timestamp + pd.Timedelta(days=90)
        mask = (details.ModifiedDate > timestamp) & (details.ModifiedDate <= end)
        expanded = details.loc[mask]
        expected = expanded.groupby("CustomerID").ProductID.agg(set)
        decoded = rows.TARGET1.map(lambda value: set(value))
        for entity, actual in zip(rows.ENTITY1, decoded):
            assert actual == expected.get(entity, set())
            assert actual <= classes
        empty_labels += int(decoded.map(len).eq(0).sum())
        multi_labels += int(decoded.map(len).gt(1).sum())
        assert set(rows.ENTITY1).issubset(set(customers.CustomerID))
        windows.append({"timestamp": str(timestamp), "rows": len(rows),
                        "label_count": int(decoded.map(len).sum()), "mismatches": 0,
                        "events_at_excluded_start": int(details.ModifiedDate.eq(timestamp).sum()),
                        "events_at_included_end": int(details.ModifiedDate.eq(end).sum())})
    assert windows
    # Record both boundaries for independent review of interval semantics.
    save_json(output / "training-labels.json", {"rows": len(training),
        "columns": list(training.columns), "windows": windows,
        "empty_label_rows": empty_labels, "multiple_label_rows": multi_labels,
        "classes": sorted(classes),
        "interval": "(timestamp, timestamp + 90 days]", "source_sha256": source_hashes})
    # Batch predictions
    # Generate label-free scoring rows for the query population.
    prediction_table = query.generate_prediction_table(snapshot, prediction_time=prediction_time)
    generated = pd.read_parquet(prediction_table.table_path)
    assert len(generated) >= 2, "Choose at least two scoring rows."
    expected_population = set(generated.ENTITY1)
    assert expected_population.issubset(set(customers.CustomerID))
    assert generated.TIMESTAMP.eq(pd.Timestamp(prediction_time)).all()
    plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
    save_json(output / "suggested-plan.json", plan.as_json())
    save_json(output / "training-plan.json", plan.as_json())
    save_json(output / "query.json", {"query": text, "task_type": str(query.get_task_type()),
        "prediction_time": prediction_time.isoformat(), "snapshot_id": snapshot.id,
        "training_table": str(training_table.train_table_path),
        "prediction_table": str(prediction_table.table_path), "snapshot_index": str(snapshot.index_path)})
    destination = FileConnector(root_dir=str(output))
    trainer = Trainer(plan)
    trained = trainer.fit(training_table, snapshot=snapshot, export_config=ModelOutputConfig(
        connector=destination, path="model"))
    assert trained.metrics
    assert all(math.isfinite(float(value)) for value in trained.metrics.values())
    save_json(output / "training-metrics.json", trained.metrics)
    # Explicitly request five product candidates for each customer in the output.
    config = PredictionOutputConfig(
        connector=destination,
        artifact_type=PredictionArtifactType.PREDICTIONS,
                extra_fields=[MetadataField.ANCHOR_TIMESTAMP],
        table_name="predictions.parquet", write_mode=WriteMode.OVERWRITE)
    predicted = trainer.predict(
        prediction_table, snapshot=snapshot,
        num_classes_to_return=5, output_config=config,
    ).frame
    print(predicted.head(8).to_string(index=False), flush=True)
    pd.testing.assert_frame_equal(predicted, pd.read_parquet(output / "predictions.parquet"))
    assert set(predicted.ENTITY) == expected_population
    assert predicted.groupby("ENTITY").size().eq(5).all()
    assert not predicted.duplicated(["ENTITY", "ANCHOR_TIMESTAMP", "CLASS"]).any()
    assert predicted.ANCHOR_TIMESTAMP.eq(pd.Timestamp(prediction_time)).all()
    assert np.isfinite(predicted.SCORE.to_numpy()).all()
    assert all(group.SCORE.is_monotonic_decreasing for _, group in predicted.groupby("ENTITY"))
    assert set(predicted.CLASS) <= classes
    print("Prediction columns:", list(predicted.columns), flush=True)
    print(predicted.head(10).to_string(index=False), flush=True)
    # A link prediction requires a positive candidate count. Verify refusal
    # preserves the successful output file.
    output_hash = hashlib.sha256((output / "predictions.parquet").read_bytes()).hexdigest()
    try:
        trainer.predict(
            prediction_table, snapshot=snapshot,
            num_classes_to_return=0, output_config=config)
    except ValueError as error:
        assert "positive" in str(error)
        save_json(output / "top-k-error.json", {
            "type": type(error).__name__, "message": str(error),
            "existing_output_unchanged": output_hash == hashlib.sha256(
                (output / "predictions.parquet").read_bytes()).hexdigest()})
    else:
        raise AssertionError("Zero link prediction candidate count should be rejected.")
    assert output_hash == hashlib.sha256((output / "predictions.parquet").read_bytes()).hexdigest()
    assert source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(output.rglob("*")) if p.is_file()]
    save_json(output / "checksums.json", manifest)
    summary = {"query": text, "task_type": str(query.get_task_type()),
        "prediction_time": prediction_time.isoformat(), "training_rows": len(training),
        "training_times": [w["timestamp"] for w in windows], "metrics": trained.metrics,
        "prediction_rows": len(predicted), "prediction_columns": list(predicted.columns),
        "candidates_per_customer": 5, "candidate_pool_size": len(classes),
        "score_min": float(predicted.SCORE.min()), "score_max": float(predicted.SCORE.max()),
        "source_files_unchanged": True, "working_model": str(trained.bundle_dir),
        "published_model": str(output / "model"), "output_directory": str(output)}
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
