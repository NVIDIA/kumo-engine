"""Train an AdventureWorks repeat-order model and score active customers.

This script preserves the original local prediction workflow: independent
prepared inputs, an unfitted-trainer control, model publication, predictions,
embeddings, retained labels, and retrospective scoring. The labels check alignment with retained outcomes.
These dates are not excluded from training; accuracy below is an in-sample
alignment diagnostic, not an estimate of held-out model quality.

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
from kumo.engine import FileConnector, PredictiveQuery, Trainer, configure
from kumo.engine.trainer.output_config import ModelOutputConfig, PredictionOutputConfig
from kumoapi.jobs import MetadataField, PredictionArtifactType, WriteMode
from kumoapi.model_plan import RunMode

from local_kumo_engine_adventureworks_binary import TABLES, build_graph


QUERY = (
    "PREDICT COUNT(SalesOrderHeader.*, 0, 90, days) > 0 "
    "FOR EACH Customer.CustomerID "
    "WHERE COUNT(SalesOrderHeader.*, -365, 0, days) > 0"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--prediction-time", default="2025-03-31T00:00:00")
    args = parser.parse_args()
    source, prepared, scratch, output = [
        path.expanduser().resolve()
        for path in (args.input_dir, args.prepared_dir, args.scratch_dir, args.output_dir)
    ]
    directories = (source, prepared, scratch, output)
    for index, first in enumerate(directories):
        for second in directories[index + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError(
                    "Input, prepared, scratch, and output directories must be separate."
                )
    if prepared.exists() or output.exists():
        raise FileExistsError(
            "Choose new prepared and output directories to preserve prior results."
        )
    prediction_time = datetime.fromisoformat(args.prediction_time)
    if prediction_time.tzinfo is not None:
        raise ValueError("Use a timezone-naive UTC prediction time.")
    files = [source / f"{name}.csv" for name in TABLES]
    original_hashes = {path.name: sha256(path) for path in files}
    frames = {path.stem: pd.read_csv(path, keep_default_na=False) for path in files}
    if not frames["Customer"].CustomerID.is_unique:
        raise ValueError("Customer.CustomerID must be unique.")
    prepared.mkdir(parents=True)
    output.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(prepared / f"{name}.parquet", index=False)

    configure(scratch_dir=scratch)
    graph = build_graph(prepared)
    graph.validate()
    snapshot = graph.snapshot()
    query = PredictiveQuery(graph, QUERY)
    query.validate()
    training_table = query.generate_training_table(snapshot)
    # Batch predictions
    # Generate label-free scoring rows for the query population.
    prediction_table = query.generate_prediction_table(
        snapshot, prediction_time=prediction_time
    )
    train_rows = pd.read_parquet(training_table.train_table_path)
    prediction_rows = pd.read_parquet(prediction_table.table_path)
    orders = pd.read_csv(source / "SalesOrderHeader.csv", parse_dates=["OrderDate"])
    population = set(prediction_rows.ENTITY1)
    past = (
        (orders.OrderDate > pd.Timestamp(prediction_time) - pd.Timedelta(days=365))
        & (orders.OrderDate <= pd.Timestamp(prediction_time))
    )
    assert population == set(orders.loc[past, "CustomerID"])
    if not 0 < args.holdout_count <= len(population):
        raise ValueError("holdout-count must be positive and no larger than the population.")
    selected = sorted(population)[:args.holdout_count]
    future = (
        (orders.OrderDate > pd.Timestamp(prediction_time))
        & (orders.OrderDate <= pd.Timestamp(prediction_time) + pd.Timedelta(days=90))
    )
    repeaters = set(orders.loc[future, "CustomerID"])
    truth = pd.DataFrame({
        "CustomerID": selected,
        "repeat_order": [customer in repeaters for customer in selected],
    })
    truth.to_parquet(output / "retrospective-labels.parquet", index=False)

    plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
    save_json(output / "training-plan.json", plan.as_json())
    destination = FileConnector(root_dir=str(output))
    prediction_config = PredictionOutputConfig(
        connector=destination,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name="predictions.parquet",
        write_mode=WriteMode.OVERWRITE,
        extra_fields=[MetadataField.JOB_TIMESTAMP],
    )
    try:
        Trainer(plan).predict(
            prediction_table, snapshot=snapshot,
            binary_classification_threshold=0.5,
            output_config=prediction_config,
        )
    except RuntimeError as error:
        assert "Trainer.fit()" in str(error)
        save_json(output / "unfitted-error.json", {
            "type": type(error).__name__, "message": str(error)
        })
    else:
        raise AssertionError("Unfitted prediction should have failed.")
    trainer = Trainer(plan)
    trained = trainer.fit(
        training_table, snapshot=snapshot,
        export_config=ModelOutputConfig(connector=destination, path="model"),
    )
    assert trained.metrics and all(
        math.isfinite(float(value)) for value in trained.metrics.values()
    )
    save_json(output / "training-metrics.json", trained.metrics)
    predicted = trainer.predict(
        prediction_table, snapshot=snapshot,
        binary_classification_threshold=0.5,
        output_config=prediction_config,
    ).frame
    print(predicted.head(8).to_string(index=False), flush=True)
    embedded = trainer.predict(
        prediction_table, snapshot=snapshot,
        binary_classification_threshold=0.5,
        output_config=PredictionOutputConfig(
            connector=destination,
            artifact_type=PredictionArtifactType.EMBEDDINGS,
            table_name="embeddings.parquet",
            write_mode=WriteMode.OVERWRITE,
        ),
    ).frame
    save_json(output / "output-shapes.json", {
        "predictions": {"rows": len(predicted), "columns": list(predicted.columns)},
        "embeddings": {"rows": len(embedded), "columns": list(embedded.columns)},
    })
    pd.testing.assert_frame_equal(predicted, pd.read_parquet(output / "predictions.parquet"))
    pd.testing.assert_frame_equal(embedded, pd.read_parquet(output / "embeddings.parquet"))
    assert set(predicted.ENTITY) == population
    assert predicted.ENTITY.is_unique
    assert np.isfinite(predicted.True_PROB.to_numpy()).all()
    assert np.allclose(predicted.True_PROB + predicted.False_PROB, 1, atol=1e-6)
    assert predicted.TARGET_PRED.eq(predicted.True_PROB.gt(0.5)).all()
    scored = truth.merge(
        predicted[["ENTITY", "TARGET_PRED"]],
        left_on="CustomerID", right_on="ENTITY", validate="one_to_one",
    )
    scored["correct"] = scored.repeat_order.eq(scored.TARGET_PRED)
    scored.to_parquet(output / "external-scoring.parquet", index=False)
    matrix = np.stack(embedded.EMBEDDING.to_numpy())
    assert matrix.shape[0] == len(population) and matrix.shape[1] > 0
    assert np.isfinite(matrix).all()
    assert set(embedded.ENTITY) == population and embedded.ENTITY.is_unique
    assert original_hashes == {path.name: sha256(path) for path in files}
    manifest = []
    for path in sorted((output / "model").rglob("*")):
        if path.is_file():
            original = trained.bundle_dir / path.relative_to(output / "model")
            assert not path.is_symlink() and not path.samefile(original)
            assert sha256(path) == sha256(original)
            manifest.append({"path": str(path.relative_to(output)),
                             "bytes": path.stat().st_size, "sha256": sha256(path)})
    for name in ("predictions.parquet", "embeddings.parquet",
                 "external-scoring.parquet", "retrospective-labels.parquet"):
        path = output / name
        manifest.append({"path": name, "bytes": path.stat().st_size,
                         "sha256": sha256(path)})
    save_json(output / "checksums.json", manifest)
    summary = {
        "training_rows": len(train_rows),
        "prediction_entities": len(prediction_rows),
        "prediction_rows": len(predicted),
        "prediction_columns": list(predicted.columns),
        "embedding_rows": len(embedded),
        "embedding_columns": list(embedded.columns),
        "embedding_shape": list(matrix.shape),
        "retrospective_accuracy": float(scored.correct.mean()),
        "holdout_rows": len(scored),
        "source_csv_sha256": original_hashes,
        "source_files_unchanged": True,
        "snapshot_id": snapshot.id,
        "snapshot_index": str(snapshot.index_path),
        "working_model": str(trained.bundle_dir),
        "output_directory": str(output),
        "runtime_reload_verified": False,
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
