# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Match Walmart and Amazon products using the public labeled entity-matching pairs.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_retail/generate.py --dataset matching --output-dir ./synthetic-matching
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

QUERIES = {'pair': 'PREDICT pairs.label FOR EACH pairs.pair_id'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES), default='pair')
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
    if args.source_root.startswith('s3://'):
        parser.error('Stage the source files locally for pair preparation.')
    prepared = output / 'prepared'
    prepared.mkdir()
    parts = []
    for split, name in enumerate(['train', 'valid', 'test']):
        frame = pd.read_csv(Path(args.source_root) / f'{name}.csv')
        frame['SPLIT'] = split
        parts.append(frame)
    pairs = pd.concat(parts, ignore_index=True)
    pairs['label'] = pairs['label'].astype(bool)
    for name in ['tableA', 'tableB']:
        frame = pd.read_csv(Path(args.source_root) / f'{name}.csv').drop(
            columns=['modelno'], errors='ignore')
        frame.to_parquet(prepared / f'{name}.parquet', index=False)
    pairs['pair_id'] = range(len(pairs))
    pairs.to_parquet(prepared / 'pairs.parquet', index=False)
    source = FileConnector(root_dir=str(prepared))
    tables = {
        'left_items': Table.from_source_table(source['tableA'],
                                              primary_key='id'),
        'right_items': Table.from_source_table(source['tableB'],
                                               primary_key='id'),
        'pairs': Table.from_source_table(source['pairs'],
                                         primary_key='pair_id'),
    }
    graph = Graph(tables, [
        Edge(*edge)
        for edge in [('pairs', 'ltable_id',
                      'left_items'), ('pairs', 'rtable_id', 'right_items')]
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
        from kumoapi.train import TrainingTableSpec

        from kumo.engine import TrainingTable
        labels = pairs[['pair_id', 'label',
                        'SPLIT']].rename(columns={
                            'pair_id': 'ENTITY1',
                            'label': 'TARGET1',
                            'timestamp': 'TIMESTAMP'
                        })
        labels.to_parquet(output / 'labels.parquet', index=False)
        training = TrainingTable.from_source_table(
            FileConnector(root_dir=str(output))['labels.parquet'],
            TrainingTableSpec(split_col='SPLIT'), pq=query, snapshot=snapshot)
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
    scoring = pairs.loc[pairs['SPLIT'] != 0,
                        ['pair_id']].rename(columns={
                            'pair_id': 'ENTITY1',
                            'timestamp': 'TIMESTAMP'
                        }).copy()
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

    scores = scores.merge(pairs[['pair_id', 'ltable_id',
                                 'rtable_id']], left_on='ENTITY',
                          right_on='pair_id', validate='one_to_one')
    print(scores.head(10).to_string(index=False))
    scores.to_parquet(output / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
