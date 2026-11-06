# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a retrained challenger with a trusted champion on identical held-out rows.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_retail/generate.py --dataset fashion --output-dir ./example-data/fashion
Pass ./example-data/fashion as --source-root where applicable.
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
    PredictionTable,
    PredictiveQuery,
    S3Connector,
    Table,
    Trainer,
    configure,
)

QUERIES = {
    'purchase':
    'PREDICT COUNT(transactions.*, 0, 90, days) > 0 FOR EACH customers.customer_id WHERE COUNT(transactions.price, -60, 0, days) > 2'
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES), default='purchase')
    parser.add_argument('--champion-path', type=Path, required=True)
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
        'customers':
        Table.from_source_table(source['customers'], primary_key='customer_id'),
        'articles':
        Table.from_source_table(source['articles'], primary_key='article_id'),
        'transactions':
        Table.from_source_table(source['transactions'], time_column='t_dat'),
    }
    graph = Graph(tables, [
        Edge(*edge)
        for edge in [('transactions', 'customer_id',
                      'customers'), ('transactions', 'article_id', 'articles')]
    ])

    graph.validate()
    # Graph visualization preview, requires a release with Graph.visualize.
    # graph.visualize(output / 'relationships.svg')
    query = PredictiveQuery(graph, QUERIES[args.problem])
    query.validate()
    snapshot = graph.snapshot()
    from kumo.engine import TrainingTableOutputConfig
    files = FileConnector(root_dir=str(output))
    training = query.generate_training_table(snapshot)
    training.export(TrainingTableOutputConfig(files, 'labeled_rows'))
    rows = pd.read_parquet(output / 'labeled_rows')
    holdout = rows.loc[rows['SPLIT'] == 2].copy()
    if len(holdout) < 2:
        raise ValueError('At least two test rows are required.')
    labels = holdout[['ENTITY1', 'TIMESTAMP', 'TARGET1']].copy()
    population_rows = holdout[['ENTITY1', 'TIMESTAMP']].copy()
    population_rows.to_parquet(output / 'comparison.parquet', index=False)
    champion = Trainer.load(args.champion_path)
    challenger = Trainer(query.suggest_model_plan(snapshot))
    challenger.fit(training, snapshot=snapshot,
                   export_config=ModelOutputConfig(files, path='challenger'))
    comparison = PredictionTable.from_source_table(
        files['comparison.parquet'], pq=query, snapshot=snapshot,
        plan=challenger.training_table_generation_plan)
    from kumoapi.jobs import MetadataField

    from kumo.engine import PredictionOutputConfig
    errors = {}
    for name, candidate in [('champion', champion), ('challenger', challenger)]:
        result = candidate.predict(
            comparison, snapshot=snapshot, binary_classification_threshold=0.5,
            output_config=PredictionOutputConfig(
                files, output_paths={'predictions': name + '_comparison'},
                extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))
        scores = result.frame
        scores['ANCHOR_TIMESTAMP'] = pd.to_datetime(scores['ANCHOR_TIMESTAMP'])
        labels['TIMESTAMP'] = pd.to_datetime(labels['TIMESTAMP'])
        aligned = labels.merge(scores, left_on=['ENTITY1', 'TIMESTAMP'],
                               right_on=['ENTITY',
                                         'ANCHOR_TIMESTAMP'], how='outer',
                               validate='one_to_one', indicator=True)
        if not aligned['_merge'].eq('both').all():
            raise ValueError('Comparison rows are not aligned.')
        errors[name] = ((aligned['True_PROB'] -
                         aligned['TARGET1'].astype(float))**2).mean()
    print('Brier scores, lower is better:', errors)
    winner = 'challenger' if errors['challenger'] < errors[
        'champion'] else 'champion'
    trainer = challenger if winner == 'challenger' else champion
    (output / 'selected_model.txt').write_text(
        str((output / 'challenger').resolve() if winner ==
            'challenger' else args.champion_path.resolve()))
    # Batch predictions
    # The generated population contains entity IDs and scoring times, not labels.
    population = query.generate_prediction_table(
        snapshot, plan=trainer.training_table_generation_plan)
    scoring = pd.read_parquet(population.table_path)
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
