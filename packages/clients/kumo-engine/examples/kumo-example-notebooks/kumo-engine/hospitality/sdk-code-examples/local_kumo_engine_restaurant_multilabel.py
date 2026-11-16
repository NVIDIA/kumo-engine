# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict which interaction types each user will have in the next day.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_hospitality/generate.py --dataset restaurant --output-dir ./synthetic-restaurant
Use that directory as --source-root or --input-dir. Metrics are demonstration only.
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


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def build_graph(input_dir):
    connector = FileConnector(root_dir=str(input_dir))
    users = Table.from_source_table(connector["users"], primary_key="user_id")
    restaurants = Table.from_source_table(connector["restaurants"],
                                          primary_key="restaurant_id")
    interactions = Table.from_source_table(connector["interactions"],
                                           time_column="timestamp")
    interactions = interactions.with_column(
        "timestamp", dtype=Dtype.time, timestamp_format="%Y-%m-%d %H:%M:%S")
    interactions = interactions.with_column("user_id", stype=Stype.ID)
    interactions = interactions.with_column("restaurant_id", stype=Stype.ID)
    social = Table.from_source_table(connector["social_graph"])
    social = social.with_column("user_a", stype=Stype.ID)
    social = social.with_column("user_b", stype=Stype.ID)
    return Graph(
        tables=[users, restaurants, interactions, social],
        edges=[
            Edge(src_table="interactions", fkey="user_id", dst_table="users"),
            Edge(src_table="interactions", fkey="restaurant_id",
                 dst_table="restaurants"),
            Edge(src_table="social_graph", fkey="user_a", dst_table="users"),
            Edge(src_table="social_graph", fkey="user_b", dst_table="users"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-time", default="2024-01-08T00:00:00")
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
        raise ValueError(
            "Use a naive timestamp matching the sample CSV timestamps.")
    output.mkdir(parents=True, exist_ok=False)
    source_files = [
        source / f"{name}.csv"
        for name in ("users", "restaurants", "interactions", "social_graph")
    ]
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    configure(scratch_dir=scratch)
    graph = build_graph(source)
    graph.validate()
    snapshot = graph.snapshot()
    text = "PREDICT LIST_DISTINCT(interactions.interaction_type, 0, 1, days) CLASSIFY TOP 4 FOR EACH users.user_id"
    query = PredictiveQuery(graph, text)
    query.validate()
    print("Task type:", query.get_task_type(), flush=True)
    training_table = query.generate_training_table(snapshot)
    training = pd.read_parquet(training_table.train_table_path)
    interactions = pd.read_csv(source / "interactions.csv",
                               parse_dates=["timestamp"])
    users = pd.read_csv(source / "users.csv")
    assert not training.duplicated(["ENTITY1", "TIMESTAMP"]).any()
    classes = set(interactions.interaction_type)
    assert len(classes) == 4
    empty_labels = 0
    multi_labels = 0
    windows = []
    for timestamp, rows in training.groupby("TIMESTAMP"):
        end = timestamp + pd.Timedelta(days=1)
        mask = (interactions.timestamp > timestamp) & (interactions.timestamp
                                                       <= end)
        expected = interactions.loc[mask].groupby(
            "user_id").interaction_type.agg(set)
        decoded = rows.TARGET1.map(lambda value: set(value.split("|"))
                                   if value else set())
        for entity, actual in zip(rows.ENTITY1, decoded):
            assert actual == expected.get(entity, set())
            assert actual <= classes
        empty_labels += int(decoded.map(len).eq(0).sum())
        multi_labels += int(decoded.map(len).gt(1).sum())
        assert set(rows.ENTITY1) == set(users.user_id)
        assert timestamp >= interactions.timestamp.min()
        assert end <= interactions.timestamp.max()
        windows.append({
            "timestamp":
            str(timestamp),
            "rows":
            len(rows),
            "label_count":
            int(decoded.map(len).sum()),
            "mismatches":
            0,
            "events_at_excluded_start":
            int(interactions.timestamp.eq(timestamp).sum()),
            "events_at_included_end":
            int(interactions.timestamp.eq(end).sum())
        })
    assert windows
    # Record both boundaries for independent review of interval semantics.
    assert sum(w["events_at_excluded_start"] for w in windows) > 0
    assert sum(w["events_at_included_end"] for w in windows) > 0
    save_json(
        output / "training-labels.json", {
            "rows": len(training),
            "columns": list(training.columns),
            "windows": windows,
            "empty_label_rows": empty_labels,
            "multiple_label_rows": multi_labels,
            "classes": sorted(classes),
            "interval": "(timestamp, timestamp + 1 day]",
            "source_sha256": source_hashes
        })
    # Batch predictions
    # Generate label-free scoring rows for the query population.
    prediction_table = query.generate_prediction_table(
        snapshot, prediction_time=prediction_time)
    generated = pd.read_parquet(prediction_table.table_path)
    assert len(generated) >= 2, "Choose at least two scoring rows."
    assert set(generated.ENTITY1) == set(users.user_id)
    assert len(generated) == len(users)
    assert generated.TIMESTAMP.eq(pd.Timestamp(prediction_time)).all()
    plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
    save_json(output / "suggested-plan.json", plan.as_json())
    save_json(output / "training-plan.json", plan.as_json())
    save_json(
        output / "query.json", {
            "query": text,
            "task_type": str(query.get_task_type()),
            "prediction_time": prediction_time.isoformat(),
            "snapshot_id": snapshot.id,
            "training_table": str(training_table.train_table_path),
            "prediction_table": str(prediction_table.table_path),
            "snapshot_index": str(snapshot.index_path)
        })
    destination = FileConnector(root_dir=str(output))
    trainer = Trainer(plan)
    trained = trainer.fit(
        training_table, snapshot=snapshot,
        export_config=ModelOutputConfig(connector=destination, path="model"))
    assert trained.metrics
    assert all(
        math.isfinite(float(value)) for value in trained.metrics.values())
    save_json(output / "training-metrics.json", trained.metrics)
    # Keep all four labels. Multilabel scores are independent probabilities;
    # they do not need to sum to one, and more than one label can be predicted.
    config = PredictionOutputConfig(
        connector=destination, artifact_type=PredictionArtifactType.PREDICTIONS,
        extra_fields=[MetadataField.ANCHOR_TIMESTAMP],
        table_name="predictions.parquet", write_mode=WriteMode.OVERWRITE)
    predicted = trainer.predict(
        prediction_table,
        snapshot=snapshot,
        num_classes_to_return=len(classes),
        output_config=config,
    ).frame
    print(predicted.head(8).to_string(index=False), flush=True)
    pd.testing.assert_frame_equal(
        predicted, pd.read_parquet(output / "predictions.parquet"))
    assert set(predicted.ENTITY) == set(users.user_id)
    assert predicted.groupby("ENTITY").size().eq(len(classes)).all()
    assert not predicted.duplicated(["ENTITY", "ANCHOR_TIMESTAMP", "CLASS"
                                     ]).any()
    assert predicted.ANCHOR_TIMESTAMP.eq(pd.Timestamp(prediction_time)).all()
    assert np.isfinite(predicted.SCORE.to_numpy()).all()
    assert predicted.SCORE.between(0, 1).all()
    assert predicted.PREDICTED.dtype == bool
    assert all(group.SCORE.is_monotonic_decreasing
               for _, group in predicted.groupby("ENTITY"))
    assert all(
        set(group.CLASS) == classes for _, group in predicted.groupby("ENTITY"))
    print("Prediction columns:", list(predicted.columns), flush=True)
    print(predicted.head(8).to_string(index=False), flush=True)
    # Reject a threshold outside [0, 1] before replacing an existing output.
    output_hash = hashlib.sha256(
        (output / "predictions.parquet").read_bytes()).hexdigest()
    try:
        trainer.predict(prediction_table, snapshot=snapshot,
                        binary_classification_threshold=1.5,
                        output_config=config)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
        save_json(
            output / "threshold-error.json", {
                "type":
                type(error).__name__,
                "message":
                str(error),
                "existing_output_unchanged":
                output_hash == hashlib.sha256(
                    (output / "predictions.parquet").read_bytes()).hexdigest()
            })
    else:
        raise AssertionError(
            "Out-of-range multilabel threshold should be rejected.")
    assert output_hash == hashlib.sha256(
        (output / "predictions.parquet").read_bytes()).hexdigest()
    prediction_runs = [{
        "file":
        "predictions.parquet",
        "rows":
        len(predicted),
        "columns":
        list(predicted.columns),
        "classes":
        sorted(classes),
        "predicted_labels":
        int(predicted.PREDICTED.sum()),
        "positive_labels_per_user":
        predicted.groupby(
            "ENTITY").PREDICTED.sum().value_counts().sort_index().to_dict(),
        "probability_min":
        float(predicted.SCORE.min()),
        "probability_max":
        float(predicted.SCORE.max()),
        "probability_sum_min":
        float(predicted.groupby("ENTITY").SCORE.sum().min()),
        "probability_sum_max":
        float(predicted.groupby("ENTITY").SCORE.sum().max()),
    }]
    assert source_hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    manifest = [{
        "path": str(p.relative_to(output)),
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
    } for p in sorted(output.rglob("*")) if p.is_file()]
    save_json(output / "checksums.json", manifest)
    summary = {
        "query": text,
        "task_type": str(query.get_task_type()),
        "prediction_time": prediction_time.isoformat(),
        "training_rows": len(training),
        "training_times": [w["timestamp"] for w in windows],
        "metrics": trained.metrics,
        "prediction_runs": prediction_runs,
        "source_files_unchanged": True,
        "working_model": str(trained.bundle_dir),
        "published_model": str(output / "model"),
        "output_directory": str(output)
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
