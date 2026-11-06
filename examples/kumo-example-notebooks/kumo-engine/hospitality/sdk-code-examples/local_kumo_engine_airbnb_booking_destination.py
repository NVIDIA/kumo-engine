# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classify the booking destination of Airbnb new users using sessions and demographics.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_hospitality/generate.py --dataset booking --output-dir ./synthetic-booking
Use that directory as --source-root or --input-dir. Metrics are demonstration only.
Install: pip install "kumo-engine[s3,text]"
See https://github.com/NVIDIA/kumo-engine for Engine setup.
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
    'destination':
    'PREDICT users.country_destination FOR EACH users.hashed_user_id'
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES),
                        default='destination')
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
        'users':
        Table.from_source_table(
            source['users_relational'], column_names=[
                c.name for c in source['users_relational'].columns
                if c.name not in ['id', 'date_first_booking']
            ], primary_key='hashed_user_id'),
        'sessions':
        Table.from_source_table(
            source['sessions_relational'], column_names=[
                c.name for c in source['sessions_relational'].columns
                if c.name != 'user_id'
            ]),
        'buckets':
        Table.from_source_table(source['age_gender_bkts_relational'],
                                primary_key='age_gender_bkt'),
    }
    graph = Graph(tables, [
        Edge(*edge)
        for edge in [('users', 'age_gender_bkt',
                      'buckets'), ('sessions', 'hashed_user_id', 'users')]
    ])

    from kumoapi.typing import Stype
    tables['users'] = tables['users'].with_column(
        'timestamp_first_active', timestamp_format='yyyyMMddHHmmss')
    for column in ['action', 'action_detail']:
        tables['sessions'] = tables['sessions'].with_column(
            column, stype=Stype.categorical)
    graph = Graph(tables, graph.edges)
    graph.validate()
    # Graph visualization preview, requires a release with Graph.visualize.
    # graph.visualize(output / 'relationships.svg')
    query = PredictiveQuery(graph, QUERIES[args.problem])
    query.validate()
    snapshot = graph.snapshot()
    if args.load_model:
        trainer = Trainer.load(args.load_model)
        training = query.generate_training_table(snapshot)
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
    from kumo.engine import TrainingTableOutputConfig

    # Static classification normally selects only missing targets. Explicitly
    # score held-out IDs so a fully labeled public dataset remains useful.
    training.export(
        TrainingTableOutputConfig(FileConnector(root_dir=str(output)),
                                  'scoring_holdout'))
    heldout = pd.read_parquet(output / 'scoring_holdout')
    heldout = heldout.loc[heldout['SPLIT'] == 2]
    scoring = heldout[[
        c for c in ['ENTITY1', 'TIMESTAMP'] if c in heldout.columns
    ]].copy()
    from kumo.engine import PredictionTable
    scoring.to_parquet(output / 'scoring.parquet', index=False)
    population = PredictionTable.from_source_table(
        FileConnector(root_dir=str(output))['scoring.parquet'], pq=query,
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
    # Multiclass outputs contain one CLASS/SCORE row per returned destination.
    if not {'CLASS', 'SCORE'}.issubset(scores.columns):
        raise ValueError('Expected destination CLASS and SCORE columns.')
    if scores.duplicated(['ENTITY', 'CLASS']).any():
        raise ValueError('Duplicate destination scores for an entity.')

    print(scores.head(10).to_string(index=False))
    scores.to_parquet(output / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
