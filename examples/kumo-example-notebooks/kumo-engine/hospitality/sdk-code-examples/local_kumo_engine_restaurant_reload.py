# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reload a trusted restaurant model and score its unchanged original snapshot.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_hospitality/generate.py --dataset restaurant --output-dir ./synthetic-restaurant
Run the matching predict script first. Set KUMO_PREDICTION_INPUT and KUMO_PREDICTION_OUTPUT to its prepared input and retained model directories, then choose KUMO_SCRATCH and KUMO_RELOAD_OUTPUT. Metrics are demonstration only.
See https://docs.nvidia.com/sdgm/kumo-engine/ for setup and modeling guidance.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from kumoapi.graph import Edge
from kumoapi.jobs import PredictionArtifactType, WriteMode
from kumoapi.typing import Dtype, Stype

from kumo.engine import (
    FileConnector,
    Graph,
    PredictiveQuery,
    Table,
    Trainer,
    configure,
)
from kumo.engine.trainer.output_config import PredictionOutputConfig

prepared, scratch, published, output = [
    Path(os.environ[name]).expanduser().resolve() for name in (
        "KUMO_PREDICTION_INPUT",
        "KUMO_SCRATCH",
        "KUMO_PREDICTION_OUTPUT",
        "KUMO_RELOAD_OUTPUT",
    )
]
for existing in (prepared, scratch, published):
    if not existing.is_dir():
        raise FileNotFoundError(f"Retain the original directory: {existing}")
    if output.is_relative_to(existing) or existing.is_relative_to(output):
        raise ValueError("Choose a separate reload output directory.")
if output.exists():
    raise FileExistsError("Choose a new reload output directory for each run.")
configure(scratch_dir=scratch)
trainer = Trainer.load(published / "model")

connector = FileConnector(root_dir=str(prepared))
users = Table.from_source_table(connector["users"], primary_key="user_id")
restaurants = Table.from_source_table(connector["restaurants"],
                                      primary_key="restaurant_id")
interactions = Table.from_source_table(connector["interactions"],
                                       time_column="timestamp")
interactions = interactions.with_column("timestamp", dtype=Dtype.time,
                                        timestamp_format="%Y-%m-%d %H:%M:%S")
interactions = interactions.with_column("user_id", stype=Stype.ID)
interactions = interactions.with_column("restaurant_id", stype=Stype.ID)
social = Table.from_source_table(connector["social_graph"])
social = social.with_column("user_a", stype=Stype.ID)
social = social.with_column("user_b", stype=Stype.ID)
graph = Graph(
    tables=[users, restaurants, interactions, social],
    edges=[
        Edge(src_table="interactions", fkey="user_id", dst_table="users"),
        Edge(src_table="interactions", fkey="restaurant_id",
             dst_table="restaurants"),
        Edge(src_table="social_graph", fkey="user_a", dst_table="users"),
        Edge(src_table="social_graph", fkey="user_b", dst_table="users"),
    ],
)
graph.validate()
snapshot = graph.snapshot()
query = PredictiveQuery(
    graph, "PREDICT users.preferred_cuisine FOR EACH users.user_id")
query.validate()

# Batch predictions
# Generate label-free scoring rows for the query population.
prediction_table = query.generate_prediction_table(snapshot)
truth = pd.read_parquet(published / "holdout-labels.parquet")
selected = pd.read_parquet(prediction_table.table_path)
assert len(truth) >= 2 and truth["user_id"].is_unique
assert len(selected) == len(truth)
assert set(selected["ENTITY1"]) == set(truth["user_id"])
class_count = int(
    pd.read_parquet(prepared / "users.parquet")["preferred_cuisine"].nunique())
output.mkdir(parents=True, exist_ok=False)
destination = FileConnector(root_dir=str(output))
predictions = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    output_config=PredictionOutputConfig(
        connector=destination,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name="predictions.parquet",
        write_mode=WriteMode.OVERWRITE,
    ),
    num_classes_to_return=class_count,
).frame
print(predictions.head(8).to_string(index=False), flush=True)
embeddings = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    output_config=PredictionOutputConfig(
        connector=destination,
        artifact_type=PredictionArtifactType.EMBEDDINGS,
        table_name="embeddings.parquet",
        write_mode=WriteMode.OVERWRITE,
    ),
).frame
pd.testing.assert_frame_equal(predictions,
                              pd.read_parquet(output / "predictions.parquet"))
pd.testing.assert_frame_equal(embeddings,
                              pd.read_parquet(output / "embeddings.parquet"))
assert set(predictions["ENTITY"]) == set(truth["user_id"])
assert not predictions.duplicated(["ENTITY", "CLASS"]).any()
assert predictions.groupby("ENTITY").size().eq(class_count).all()
assert np.isfinite(predictions["SCORE"].to_numpy()).all()
assert predictions["SCORE"].between(0, 1).all()
assert np.allclose(predictions.groupby("ENTITY")["SCORE"].sum(), 1.0)
assert predictions.groupby("ENTITY")["PREDICTED"].sum().eq(1).all()
assert embeddings["ENTITY"].is_unique
assert set(embeddings["ENTITY"]) == set(truth["user_id"])
vectors = np.stack(embeddings["EMBEDDING"].to_numpy())
assert vectors.shape[0] == len(truth) and vectors.shape[1] > 0
assert np.isfinite(vectors).all()
print("Reloaded prediction rows:", len(predictions))
print("Reloaded embedding matrix shape:", vectors.shape)
print("New outputs:", output)
