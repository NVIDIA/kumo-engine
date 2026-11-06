# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict a movie's average rating during the next twelve months.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset movielens --output-dir ./example-data/movielens
Pass ./example-data/movielens as --source-root where applicable.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.
Install: pip install "kumo-engine[s3,text]"
Run this local Python script with --source-root, --scratch-dir on a local
Use --scratch-dir on local SSD and --output-dir at a new retained directory.
"""
import argparse
from pathlib import Path

import pandas as pd

from kumo.engine import (
    Edge,
    FileConnector,
    Graph,
    ModelOutputConfig,
    PredictiveQuery,
    S3Connector,
    Table,
    Trainer,
    configure,
)

QUERIES = {
    'average_rating':
    'PREDICT AVG(rating.rating, 0, 12, months) FOR EACH movie.movieId WHERE COUNT(rating.rating, -inf, 0) >= 3 AND COUNT(rating.rating, 0, 12, months) >= 3'
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES),
                        default='average_rating')
    parser.add_argument('--load-model', type=Path,
                        help='Trusted model bundle to score without fitting.')
    args = parser.parse_args()
    scratch = args.scratch_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        parser.error(
            'Scratch and retained outputs must be separate directories.')
    output.mkdir(parents=True, exist_ok=False)
    configure(scratch_dir=scratch)
    source = (S3Connector(root_dir=args.source_root)
              if args.source_root.startswith('s3://') else FileConnector(
                  root_dir=args.source_root))

    tables = {
        'user':
        Table.from_source_table(source['user'], primary_key='userId'),
        'movie':
        Table.from_source_table(source['movie'], primary_key='movieId'),
        'rating':
        Table.from_source_table(source['rating'], time_column='timestamp'),
    }
    graph = Graph(tables, [
        Edge(*edge) for edge in [('rating', 'userId',
                                  'user'), ('rating', 'movieId', 'movie')]
    ])

    graph.validate()
    # Graph visualization preview, requires a release with Graph.visualize.
    # graph.visualize(output / 'relationships.svg')
    query = PredictiveQuery(graph, QUERIES[args.problem])
    query.validate()
    snapshot = graph.snapshot()
    if args.load_model:
        trainer = Trainer.load(args.load_model)
    else:
        training = query.generate_training_table(snapshot, plan=None)
        plan = query.suggest_model_plan(snapshot)  # Supported FAST defaults.

        trainer = Trainer(plan)
        result = trainer.fit(
            training, snapshot=snapshot,
            export_config=ModelOutputConfig(FileConnector(root_dir=str(output)),
                                            path='model'))
        print('Evaluation metrics (synthetic, demonstration only):',
              result.metrics)
        print('Explanation artifacts:', trainer.scratch_dir)

    # Batch predictions
    # The generated population contains entity IDs and scoring times, not labels.
    # Future label conditions define training examples, not scoring eligibility.
    population_query = PredictiveQuery(
        graph, QUERIES[args.problem].split(
            ' AND COUNT(rating.rating, 0, 12, months)')[0])
    population = population_query.generate_prediction_table(
        snapshot, plan=trainer.training_table_generation_plan)
    scoring = pd.read_parquet(population.table_path)
    from kumo.engine import PredictionTable
    scoring.to_parquet(output / 'population.parquet', index=False)
    population = PredictionTable.from_source_table(
        FileConnector(root_dir=str(output))['population.parquet'], pq=query,
        snapshot=snapshot, plan=trainer.training_table_generation_plan)
    if len(scoring) < 2:
        raise ValueError('Choose a scoring population with at least two rows.')
    task_type = query.get_task_type()
    options = {}
    if task_type.value in {
            'binary_classification', 'multilabel_classification'
    }:
        options['binary_classification_threshold'] = 0.5
    if task_type.is_link_pred:
        # Match the query's requested recommendation count.
        import re
        top = re.search(r'RANK TOP (\d+)', query.query, re.IGNORECASE)
        options['num_classes_to_return'] = int(top.group(1)) if top else 10
    scores = trainer.predict(population, snapshot=snapshot, **options).frame
    if 'ENTITY' not in scores or 'ENTITY1' not in scoring:
        raise ValueError(
            'Expected entity columns are absent from scoring results.')
    if set(scores['ENTITY']) != set(scoring['ENTITY1']):
        raise ValueError('Prediction IDs do not match the scoring population.')
    if (task_type.is_link_pred or task_type.value in {
            'multiclass_classification', 'multilabel_classification',
            'multilabel_ranking'
    }):
        if not {'CLASS', 'SCORE'}.issubset(scores.columns):
            raise ValueError('Expected CLASS and SCORE for ranked predictions.')
        if scores.duplicated(['ENTITY', 'CLASS']).any():
            raise ValueError(
                'Duplicate ranked candidates for a scoring entity.')
        print(
            'CLASS identifies the candidate; SCORE orders candidates within each entity.'
        )
    elif task_type.value != 'forecasting':
        if len(scores) != len(scoring) or scores['ENTITY'].duplicated().any():
            raise ValueError('Expected one prediction per scoring entity.')
        print(
            'TARGET_PRED is the predicted outcome; probabilities appear for classification.'
        )
    else:
        print(
            'Forecast outputs retain a separate value for each forecast horizon.'
        )

    print(scores.head(10).to_string(index=False))
    scores.to_parquet(output / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
