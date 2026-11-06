"""Forecast three 30-day order counts for each customer.

Training uses the suggested FAST model plan. Keep local_kumo_engine_adventureworks_temporal.py
beside this file for its graph and JSON helpers.

Install with `pip install kumo-engine`. Use Linux with GPU compute and local SSD scratch.
Prediction files retain entity IDs for alignment with the scoring population.
See https://docs.nvidia.com/sdgm/kumo-engine/ for setup and modeling guidance.
"""

import argparse
from datetime import datetime
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from kumo.engine import FileConnector, PredictiveQuery, Trainer, configure
from kumo.engine.trainer.output_config import ModelOutputConfig, PredictionOutputConfig
from kumoapi.jobs import MetadataField, PredictionArtifactType, WriteMode
from kumoapi.model_plan import RunMode

from local_kumo_engine_adventureworks_temporal import build_graph, save_json

QUERY = (
    "PREDICT COUNT(SalesOrderHeader.*, 0, 30, days) "
    "FORECAST 3 TIMEFRAMES FOR EACH Customer.CustomerID"
)
TABLES = ("Customer", "Product", "SalesOrderHeader", "SalesOrderDetail")


def separate_paths(*paths):
    for first, second in combinations(paths, 2):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError(
                "Input, prepared, scratch, model, and output paths must be separate."
            )


def file_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def source_hashes(root):
    return {
        f"{name}.csv": hashlib.sha256((root / f"{name}.csv").read_bytes()).hexdigest()
        for name in TABLES
    }


def prepare(source, prepared):
    separate_paths(source, prepared)
    before = source_hashes(source)
    original = pd.read_csv(source / "SalesOrderHeader.csv", parse_dates=["OrderDate"])
    if (
        len(original) != 31465
        or original.OrderDate.isna().any()
        or original.OrderDate.min() != pd.Timestamp("2022-05-30")
        or original.OrderDate.max() != pd.Timestamp("2025-06-29")
    ):
        raise ValueError(
            "Use the checksum-verified AdventureWorks sample and its original dates."
        )
    prepared.mkdir(parents=True, exist_ok=False)
    for name in TABLES:
        (prepared / f"{name}.csv").write_bytes((source / f"{name}.csv").read_bytes())
    assert before == source_hashes(source)
    result = {
        "synthetic_history": False,
        "order_rows": len(original),
        "order_date_min": str(original.OrderDate.min()),
        "order_date_max": str(original.OrderDate.max()),
        "source_files_unchanged": True,
        "source_sha256": before,
        "prepared_sha256": source_hashes(prepared),
    }
    save_json(prepared / "preparation.json", result)
    print(json.dumps(result, indent=2), flush=True)


def run(args):
    source, scratch, output = (
        getattr(args, key).expanduser().resolve()
        for key in ("input_dir", "scratch_dir", "output_dir")
    )
    paths = [source, scratch, output]
    model = args.model_dir.expanduser().resolve() if args.mode == "predict" else None
    if model is not None:
        if not model.is_dir():
            raise ValueError("The saved model directory must exist.")
        paths.append(model)
    separate_paths(*paths)
    anchor = datetime.fromisoformat(args.prediction_time)
    if anchor.tzinfo is not None:
        raise ValueError("Use a timezone-free datetime, interpreted as UTC.")
    before = source_hashes(source)
    model_before = file_hashes(model) if model else None
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
        "prediction_time": anchor,
        "snapshot_id": snapshot.id,
        "snapshot_index": str(snapshot.index_path),
        "source_sha256": before,
    }
    if args.mode == "fit":
        training_table = query.generate_training_table(snapshot)
        plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
        save_json(output / "suggested-plan.json", plan.as_json())
        plan.training_job.metrics = ["mae", "mse", "rmse"]
        plan.training_job.tune_metric = "mae"
        save_json(output / "training-plan.json", plan.as_json())
        trainer = Trainer(plan)
        trained = trainer.fit(
            training_table,
            snapshot=snapshot,
            export_config=ModelOutputConfig(connector=destination, path="model"),
        )
        assert trained.metrics and all(
            math.isfinite(float(v)) for v in trained.metrics.values()
        )
        save_json(output / "training-metrics.json", trained.metrics)
        summary.update(
            training_rows=len(pd.read_parquet(training_table.train_table_path)),
            model_dir=str(output / "model"),
            metrics=trained.metrics,
        )
    else:
        trainer = Trainer.load(model)
        summary["model_dir"] = str(model)
    # Batch predictions
    # Generate label-free scoring rows for the query population.
    table = query.generate_prediction_table(snapshot, prediction_time=anchor)
    population = pd.read_parquet(table.table_path)
    assert len(population) >= 2
    assert population.ENTITY1.is_unique
    # Request the optional anchor column for the checks below.
    # TIMESTAMP remains the start of each forecast window.
    frame = trainer.predict(
        table,
        snapshot=snapshot,
        output_config=PredictionOutputConfig(
            connector=destination,
            artifact_type=PredictionArtifactType.PREDICTIONS,
            table_name="predictions.parquet",
            write_mode=WriteMode.OVERWRITE,
            extra_fields=[MetadataField.ANCHOR_TIMESTAMP],
        ),
    ).frame
    print(frame.head(8).to_string(index=False), flush=True)
    assert set(frame.ENTITY) == set(population.ENTITY1)
    assert len(frame) == 3 * len(population)
    assert frame.groupby("ENTITY").size().eq(3).all()
    assert not frame.duplicated(["ENTITY", "TIMESTAMP"]).any()
    assert set(frame.TIMESTAMP) == {
        pd.Timestamp(anchor) + pd.Timedelta(days=30 * offset)
        for offset in range(3)
    }
    assert frame.ANCHOR_TIMESTAMP.eq(pd.Timestamp(anchor)).all()
    assert np.isfinite(frame.TARGET_PRED.to_numpy()).all()
    pd.testing.assert_frame_equal(
        frame, pd.read_parquet(output / "predictions.parquet")
    )
    assert source_hashes(source) == before
    if model is not None:
        assert file_hashes(model) == model_before
    summary.update(
        prediction_rows=len(frame),
        entities=frame.ENTITY.nunique(),
        columns=list(frame.columns),
        source_files_unchanged=True,
        training_invoked=args.mode == "fit",
        predictions_finite=True,
        horizon_dates=sorted(str(t) for t in frame.TIMESTAMP.unique()),
        anchors_correct=True,
        parquet_matches_returned_frame=True,
    )
    save_json(output / "summary.json", summary)
    save_json(output / "checksums.json", file_hashes(output))
    print(json.dumps(summary, indent=2, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    preparation = commands.add_parser(
        "prepare", help="Validate and copy the AdventureWorks sample."
    )
    preparation.add_argument("--input-dir", type=Path, required=True)
    preparation.add_argument("--prepared-dir", type=Path, required=True)
    for mode in ("fit", "predict"):
        command = commands.add_parser(mode)
        command.add_argument("--input-dir", type=Path, required=True)
        command.add_argument("--scratch-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--prediction-time", default="2025-06-30T00:00:00")
        if mode == "predict":
            command.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(
            args.input_dir.expanduser().resolve(),
            args.prepared_dir.expanduser().resolve(),
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
