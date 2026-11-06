# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score a trusted local bundle on an explicit source graph for Airflow.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset mlops --output-dir ./example-data/mlops
Pass ./example-data/mlops as --source-root where applicable.
The generator writes local files only. For remote connectors, load those files
into your own source first, preserving file stems as table names and column types.
Fill the connection configuration with your own account, database and schema.
Synthetic evaluation metrics are demonstration only.


"""
from pathlib import Path

import pandas as pd
from mlops_companions.local_kumo_engine_mlops_settings import (
    QUERY,
    build_graph,
)

from kumo.engine import (
    FileConnector,
    PredictionOutputConfig,
    PredictiveQuery,
    Trainer,
    configure,
)


def predict(source_root, model_path, scratch_dir, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    configure(scratch_dir=Path(scratch_dir))
    graph = build_graph(source_root)
    snapshot = graph.snapshot()
    query = PredictiveQuery(graph, QUERY)
    trainer = Trainer.load(model_path)
    population = query.generate_prediction_table(
        snapshot, plan=trainer.training_table_generation_plan)
    expected = pd.read_parquet(population.table_path)['ENTITY1']
    if len(expected) < 2:
        raise ValueError('At least two scoring rows are required.')
    result = trainer.predict(
        population, snapshot=snapshot, binary_classification_threshold=0.5,
        output_config=PredictionOutputConfig(
            FileConnector(root_dir=str(output)), output_paths={
                'predictions': 'predictions',
                'embeddings': 'embeddings'
            }))
    frame = result.frame
    if len(frame) != len(expected) or set(frame['ENTITY']) != set(expected):
        raise ValueError('Scoring population mismatch.')
    print(frame.head(10).to_string(index=False))
    return str(output)
