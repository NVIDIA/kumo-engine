# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score current source data with the exact trusted bundle produced by the fit job.
Synthetic data setup, starting in the NVIDIA/kumo-engine checkout:
  cd examples/kumo-example-notebooks
  python synthetic-data/engine_other/generate.py --dataset mlops --output-dir ./example-data/mlops
Pass ./example-data/mlops as --source-root where applicable.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.
Install: pip install "kumo-engine[s3,text]"
Pass --model-path, --source-root, --scratch-dir on SSD and a new --output-dir.
"""
import pandas as pd
from mlops_companions.local_kumo_engine_mlops_settings import (
    QUERY,
    build_graph,
    parse_args,
)

from kumo.engine import PredictiveQuery, Trainer


def main():
    args = parse_args(score=True)
    graph = build_graph(args.source_root)
    graph.validate()
    snapshot = graph.snapshot()
    trainer = Trainer.load(args.model_path)
    query = PredictiveQuery(graph, QUERY)
    # Batch predictions
    population = query.generate_prediction_table(
        snapshot, plan=trainer.training_table_generation_plan)
    expected = pd.read_parquet(population.table_path)['ENTITY1']
    if len(expected) < 2:
        raise ValueError('Supply at least two users to score.')
    scores = trainer.predict(population, snapshot=snapshot,
                             binary_classification_threshold=0.5).frame
    if len(scores) != len(expected) or set(scores['ENTITY']) != set(expected):
        raise ValueError('Scores do not match the requested users.')
    print('True_PROB is seven-day repeat purchase probability.')
    print(scores.head(10).to_string(index=False))
    scores.to_parquet(args.output_dir / 'predictions.parquet', index=False)
    # MLOps snapshot/alias loading preview, requires that integration API:
    # trainer = Trainer.load_from_connector(mlops_config, location=registry_alias)
    # snapshot = GraphSnapshot.load_from_connector(mlops_config, id=snapshot_id, graph=graph)


if __name__ == '__main__':
    main()
