# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit a users repeat-purchase model and retain its exact bundle for a score job.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset mlops --output-dir ./example-data/mlops
Pass ./example-data/mlops as --source-root where applicable.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.
Install: pip install "kumo-engine[s3,text]"
Pass --source-root, --scratch-dir on SSD and a new --output-dir.
"""
from mlops_companions.local_kumo_engine_mlops_settings import QUERY, build_graph, parse_args
from kumo.engine import FileConnector, ModelOutputConfig, PredictiveQuery, Trainer


def main():
    args = parse_args()
    graph = build_graph(args.source_root)
    graph.validate()
    snapshot = graph.snapshot()
    query = PredictiveQuery(graph, QUERY)
    training = query.generate_training_table(snapshot)
    trainer = Trainer(query.suggest_model_plan(snapshot))
    result = trainer.fit(training, snapshot=snapshot,
        export_config=ModelOutputConfig(FileConnector(root_dir=str(args.output_dir)), path='model'))
    print('Metrics:', result.metrics)
    (args.output_dir / 'model_path.txt').write_text(str((args.output_dir / 'model').resolve()))
    # Tracking/alias automation preview, requires the MLOps integration API:
    # with kumo.mlops(kumo.MLOps(artifact_root, tracking=tracking, registry=registry)):
    #     result = trainer.fit(training, snapshot=snapshot)
    # Batch predictions
    population = query.generate_prediction_table(snapshot, plan=trainer.training_table_generation_plan)
    if population.row_count < 2:
        raise ValueError('At least two users are required for batch scoring.')
    import pandas as pd
    expected = pd.read_parquet(population.table_path)['ENTITY1']
    scores = trainer.predict(population, snapshot=snapshot, binary_classification_threshold=0.5).frame
    if len(scores) != len(expected) or set(scores['ENTITY']) != set(expected):
        raise ValueError('Scores must match the users in the scoring population.')
    print('True_PROB is the probability of repeat purchase within seven days.')
    print(scores.head(10).to_string(index=False))
    scores.to_parquet(args.output_dir / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
