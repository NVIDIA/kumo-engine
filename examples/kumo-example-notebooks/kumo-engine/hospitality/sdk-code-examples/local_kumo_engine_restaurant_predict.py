# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train a small restaurant model and score twenty held-out cuisine labels.
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

TABLES = ("users", "restaurants", "interactions", "social_graph")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-count", type=int, default=20)
    args = parser.parse_args()
    source, prepared, scratch, output = [
        path.expanduser().resolve()
        for path in (args.input_dir, args.prepared_dir, args.scratch_dir,
                     args.output_dir)
    ]
    directories = (source, prepared, scratch, output)
    for i, first in enumerate(directories):
        for second in directories[i + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError(
                    "Input, prepared, scratch, and output directories must be separate."
                )
    if prepared.exists() or output.exists():
        raise FileExistsError(
            "Choose new prepared and output directories to preserve prior results."
        )
    files = [source / f"{name}.csv" for name in TABLES]
    original_hashes = {path.name: sha256(path) for path in files}
    # Keep string values intact. Only the selected target labels become null.
    frames = {
        path.stem: pd.read_csv(path, keep_default_na=False)
        for path in files
    }
    users = frames["users"]
    if not users["user_id"].is_unique:
        raise ValueError("users.user_id must be unique.")
    if not users["preferred_cuisine"].ne("").all():
        raise ValueError("Start with fully labeled preferred_cuisine values.")
    if not 0 < args.holdout_count < len(users):
        raise ValueError(
            "holdout-count must be positive and smaller than the user count.")
    selected = users.nsmallest(args.holdout_count, "user_id").index
    truth = users.loc[selected, ["user_id", "preferred_cuisine"]].copy()
    users.loc[selected, "preferred_cuisine"] = pd.NA
    prepared.mkdir(parents=True)
    output.mkdir(parents=True)
    truth.to_parquet(output / "holdout-labels.parquet", index=False)
    for name, frame in frames.items():
        frame.to_parquet(prepared / f"{name}.parquet", index=False)
    assert pd.read_parquet(
        prepared / "users.parquet").preferred_cuisine.isna().sum() == len(truth)
    configure(scratch_dir=scratch)
    graph = build_graph(prepared)
    graph.validate()
    snapshot = graph.snapshot()
    query = PredictiveQuery(
        graph, "PREDICT users.preferred_cuisine FOR EACH users.user_id")
    query.validate()
    training_table = query.generate_training_table(snapshot)
    # Batch predictions
    # Generate label-free scoring rows for the query population.
    prediction_table = query.generate_prediction_table(snapshot)
    train_rows = pd.read_parquet(training_table.train_table_path)
    prediction_rows = pd.read_parquet(prediction_table.table_path)
    assert len(train_rows) == len(users) - len(truth)
    assert set(train_rows.ENTITY1).isdisjoint(truth.user_id)
    assert set(prediction_rows.ENTITY1) == set(truth.user_id)
    assert len(prediction_rows) == len(truth)
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
    # Check that a new Trainer cannot score a model trained by another instance.
    try:
        Trainer(plan).predict(prediction_table, snapshot=snapshot,
                              output_config=prediction_config)
    except RuntimeError as error:
        assert "Trainer.fit()" in str(error)
        save_json(output / "unfitted-error.json", {
            "type": type(error).__name__,
            "message": str(error)
        })
    else:
        raise AssertionError("Unfitted prediction should have failed.")
    trainer = Trainer(plan)
    trained = trainer.fit(
        training_table, snapshot=snapshot,
        export_config=ModelOutputConfig(connector=destination, path="model"))
    assert trained.metrics and all(
        math.isfinite(float(x)) for x in trained.metrics.values())
    save_json(output / "training-metrics.json", trained.metrics)
    predicted = trainer.predict(
        prediction_table, snapshot=snapshot, output_config=prediction_config,
        num_classes_to_return=users.preferred_cuisine.nunique()).frame
    print(predicted.head(8).to_string(index=False), flush=True)
    print("Prediction columns:", list(predicted.columns), flush=True)
    print(predicted.head().to_string(index=False), flush=True)
    embedded = trainer.predict(
        prediction_table,
        snapshot=snapshot,
        output_config=PredictionOutputConfig(
            connector=destination,
            artifact_type=PredictionArtifactType.EMBEDDINGS,
            table_name="embeddings.parquet",
            write_mode=WriteMode.OVERWRITE,
        ),
    ).frame
    print("Embedding columns:", list(embedded.columns), flush=True)
    save_json(
        output / "output-shapes.json", {
            "predictions": {
                "rows": len(predicted),
                "columns": list(predicted.columns)
            },
            "embeddings": {
                "rows": len(embedded),
                "columns": list(embedded.columns)
            },
        })
    pd.testing.assert_frame_equal(
        predicted, pd.read_parquet(output / "predictions.parquet"))
    pd.testing.assert_frame_equal(
        embedded, pd.read_parquet(output / "embeddings.parquet"))
    # Classification output has one row for each retained class and user.
    assert set(predicted.ENTITY) == set(truth.user_id)
    assert predicted.groupby("ENTITY").size().eq(
        users.preferred_cuisine.nunique()).all()
    assert not predicted.duplicated(["ENTITY", "CLASS"]).any()
    assert np.isfinite(predicted.SCORE.to_numpy()).all()
    assert predicted.SCORE.between(0, 1).all()
    assert predicted.PREDICTED.dtype == bool
    assert np.allclose(predicted.groupby("ENTITY").SCORE.sum(), 1, atol=1e-6)
    expected_classes = set(users.preferred_cuisine.dropna())
    assert all(
        set(group.CLASS) == expected_classes
        for _, group in predicted.groupby("ENTITY"))
    assert predicted.groupby("ENTITY").PREDICTED.sum().eq(1).all()
    best = predicted.loc[predicted.PREDICTED, ["ENTITY", "CLASS"]]
    maxima = predicted.groupby("ENTITY").SCORE.max()
    winner_scores = predicted.loc[predicted.PREDICTED].set_index("ENTITY").SCORE
    assert winner_scores.eq(maxima.reindex(winner_scores.index)).all()
    scored = truth.merge(best, left_on="user_id", right_on="ENTITY",
                         validate="one_to_one")
    scored["correct"] = scored.preferred_cuisine.eq(scored.CLASS)
    scored.to_parquet(output / "external-scoring.parquet", index=False)
    assert len(scored) == len(truth)
    matrix = np.stack(embedded.EMBEDDING.to_numpy())
    assert matrix.shape[0] == len(truth) and matrix.shape[1] > 0
    assert np.isfinite(matrix).all()
    assert set(embedded.ENTITY) == set(truth.user_id)
    assert embedded.ENTITY.is_unique
    assert original_hashes == {path.name: sha256(path) for path in files}
    manifest = []
    for path in sorted((output / "model").rglob("*")):
        if path.is_file():
            original = trained.bundle_dir / path.relative_to(output / "model")
            assert not path.is_symlink() and not path.samefile(original)
            assert sha256(path) == sha256(original)
            manifest.append({
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path)
            })
    for name in ("predictions.parquet", "embeddings.parquet",
                 "external-scoring.parquet", "holdout-labels.parquet"):
        path = output / name
        manifest.append({
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path)
        })
    save_json(output / "checksums.json", manifest)
    summary = {
        "training_rows":
        len(train_rows),
        "prediction_entities":
        len(prediction_rows),
        "prediction_rows":
        len(predicted),
        "prediction_columns":
        list(predicted.columns),
        "class_count":
        users.preferred_cuisine.nunique(),
        "score_min":
        float(predicted.SCORE.min()),
        "score_max":
        float(predicted.SCORE.max()),
        "probability_sum_max_error":
        float(abs(predicted.groupby("ENTITY").SCORE.sum() - 1).max()),
        "embedding_rows":
        len(embedded),
        "embedding_columns":
        list(embedded.columns),
        "embedding_shape":
        list(matrix.shape),
        "top_1_accuracy":
        float(scored.correct.mean()),
        "holdout_rows":
        len(scored),
        "source_csv_sha256":
        original_hashes,
        "source_files_unchanged":
        True,
        "snapshot_id":
        snapshot.id,
        "snapshot_index":
        str(snapshot.index_path),
        "working_model":
        str(trained.bundle_dir),
        "output_directory":
        str(output),
        "runtime_reload_verified":
        False,
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
