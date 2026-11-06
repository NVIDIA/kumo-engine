"""Reload a trusted AdventureWorks model and score a compatible snapshot.

Retain or reconnect compatible prepared inputs and choose local scratch. Set
KUMO_PREDICTION_INPUT, KUMO_SCRATCH, KUMO_PREDICTION_OUTPUT, and a new
KUMO_RELOAD_OUTPUT directory. This script does not fit or update a model.

Install with `pip install kumo-engine`. Use Linux with GPU compute and local SSD scratch.
Prediction files retain entity IDs for alignment with the scoring population.
See https://docs.nvidia.com/sdgm/kumo-engine/ for setup and modeling guidance.
"""

from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from kumo.engine import FileConnector, PredictiveQuery, Trainer, configure
from kumo.engine.trainer.output_config import PredictionOutputConfig
from kumoapi.jobs import PredictionArtifactType, WriteMode

from local_kumo_engine_adventureworks_binary import build_graph
from local_kumo_engine_adventureworks_predict import QUERY


prepared, scratch, published, output = [
    Path(os.environ[name]).expanduser().resolve()
    for name in (
        "KUMO_PREDICTION_INPUT", "KUMO_SCRATCH",
        "KUMO_PREDICTION_OUTPUT", "KUMO_RELOAD_OUTPUT",
    )
]
for existing in (prepared, published):
    if not existing.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {existing}")
for first, second in combinations((prepared, scratch, published, output), 2):
    if first.is_relative_to(second) or second.is_relative_to(first):
        raise ValueError(
            "Prepared input, scratch, published output, and reload output "
            "paths must be separate."
        )
scratch.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError("Choose a new reload output directory for each run.")
configure(scratch_dir=scratch)
trainer = Trainer.load(published / "model")
graph = build_graph(prepared)
graph.validate()
snapshot = graph.snapshot()
query = PredictiveQuery(graph, QUERY)
query.validate()

prediction_time = pd.Timestamp(os.environ.get(
    "KUMO_PREDICTION_TIME", "2025-03-31T00:00:00"
)).to_pydatetime()
# Batch predictions
# Generate label-free scoring rows for the query population.
prediction_table = query.generate_prediction_table(
    snapshot, prediction_time=prediction_time,
    plan=trainer.training_table_generation_plan,
)
selected = pd.read_parquet(prediction_table.table_path)
output.mkdir(parents=True, exist_ok=False)
destination = FileConnector(root_dir=str(output))
predictions = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    binary_classification_threshold=0.5,
    output_config=PredictionOutputConfig(
        connector=destination,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name="predictions.parquet",
        write_mode=WriteMode.OVERWRITE,
    ),
).frame
print(predictions.head(8).to_string(index=False), flush=True)
embeddings = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    binary_classification_threshold=0.5,
    output_config=PredictionOutputConfig(
        connector=destination,
        artifact_type=PredictionArtifactType.EMBEDDINGS,
        table_name="embeddings.parquet",
        write_mode=WriteMode.OVERWRITE,
    ),
).frame
pd.testing.assert_frame_equal(
    predictions, pd.read_parquet(output / "predictions.parquet")
)
pd.testing.assert_frame_equal(
    embeddings, pd.read_parquet(output / "embeddings.parquet")
)
assert set(predictions.ENTITY) == set(selected.ENTITY1)
assert predictions.ENTITY.is_unique
assert np.isfinite(predictions.True_PROB.to_numpy()).all()
assert np.allclose(predictions.True_PROB + predictions.False_PROB, 1.0)
assert predictions.TARGET_PRED.eq(predictions.True_PROB.gt(0.5)).all()
assert embeddings.ENTITY.is_unique
assert set(embeddings.ENTITY) == set(selected.ENTITY1)
vectors = np.stack(embeddings.EMBEDDING.to_numpy())
assert vectors.shape[0] == len(selected) and vectors.shape[1] > 0
assert np.isfinite(vectors).all()
print("Reloaded prediction rows:", len(predictions))
print("Reloaded embedding matrix shape:", vectors.shape)
print("New outputs:", output)
