"""Regress hourly transformer oil temperature from the synthetic ETT series.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset transformer --output-dir ./example-data/transformer
Pass ./example-data/transformer as --source-root where applicable.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.
Install: pip install "kumo-engine[s3,text]"
Run this local Python script with --source-root, --scratch-dir on a local
Use --scratch-dir on local SSD and --output-dir at a new retained directory.
"""
import argparse
from pathlib import Path
import pandas as pd
from kumo.engine import (Edge, FileConnector, Graph, ModelOutputConfig,
                         PredictiveQuery, PredictionTable, S3Connector,
                         Table, Trainer, configure)

QUERIES = {'forecast': 'PREDICT AVG(target_table.OT, 0, 1, hours) FOR EACH entity_table.entity'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES), default='forecast')
    parser.add_argument('--horizon', type=int, default=24, help='ETT rolling-mean evaluation horizon in hours.')
    parser.add_argument('--load-model', type=Path, help='Trusted model bundle to score without fitting.')
    args = parser.parse_args()
    scratch = args.scratch_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        parser.error('Scratch and retained outputs must be separate directories.')
    output.mkdir(parents=True, exist_ok=False)
    configure(scratch_dir=scratch)
    source = (S3Connector(root_dir=args.source_root) if args.source_root.startswith('s3://')
              else FileConnector(root_dir=args.source_root))

    tables = {
        'entity_table': Table.from_source_table(source['entity_table'], primary_key='entity'),
        'target_table': Table.from_source_table(source['target_table'], time_column='date'),
    }
    graph = Graph(tables, [Edge(*edge) for edge in [('target_table', 'entity', 'entity_table')]])

    graph.validate()
    # Graph visualization preview, requires a release with Graph.visualize.
    # graph.visualize(output / 'relationships.svg')
    query = PredictiveQuery(graph, QUERIES[args.problem])
    query.validate()
    snapshot = graph.snapshot()
    if args.load_model:
        trainer = Trainer.load(args.load_model)
    else:
        from kumo.engine import TrainingTable
        from kumoapi.train import TrainingTableSpec
        if args.source_root.startswith('s3://'):
            parser.error('Stage this small time series locally for manual label preparation.')
        rows = pd.read_parquet(Path(args.source_root) / 'target_table.parquet').sort_values('date')
        labels = rows[['entity', 'date', 'OT']].rename(columns={'entity': 'ENTITY1', 'date': 'TIMESTAMP', 'OT': 'TARGET1'})
        # Anchor before the observation being predicted, excluding it from history.
        labels['TIMESTAMP'] = (pd.to_datetime(labels['TIMESTAMP']) - pd.Timedelta(hours=1)).astype('datetime64[ms]')
        labels['SPLIT'] = 0
        labels.loc[labels.index[int(len(labels)*0.8):], 'SPLIT'] = 1
        labels.loc[labels.index[int(len(labels)*0.9):], 'SPLIT'] = 2
        labels.to_parquet(output / 'labels.parquet', index=False)
        training = TrainingTable.from_source_table(FileConnector(root_dir=str(output))['labels.parquet'],
            TrainingTableSpec(split_col='SPLIT'), pq=query, snapshot=snapshot)
        plan = query.suggest_model_plan(snapshot)  # Supported FAST defaults.

        trainer = Trainer(plan)
        result = trainer.fit(training, snapshot=snapshot,
                             export_config=ModelOutputConfig(FileConnector(root_dir=str(output)), path='model'))
        print('Evaluation metrics (synthetic, demonstration only):', result.metrics)
        print('Explanation artifacts:', trainer.scratch_dir)

    # Batch predictions
    # The generated population contains entity IDs and scoring times, not labels.
    rows = pd.read_parquet(Path(args.source_root) / 'target_table.parquet').sort_values('date').reset_index(drop=True)
    heldout = rows.iloc[int(len(rows)*0.9):].copy()
    scoring = heldout[['entity', 'date']].rename(columns={'entity':'ENTITY1','date':'TIMESTAMP'})
    scoring['TIMESTAMP'] = (pd.to_datetime(scoring['TIMESTAMP']) - pd.Timedelta(hours=1)).astype('datetime64[ms]')
    if len(scoring) < 2:
        raise ValueError('At least two held-out timestamps are required.')
    scoring.to_parquet(output / 'scoring.parquet', index=False)
    population = PredictionTable.from_source_table(FileConnector(root_dir=str(output))['scoring.parquet'],
        pq=query, snapshot=snapshot, plan=trainer.training_table_generation_plan)
    from kumo.engine import PredictionOutputConfig
    from kumoapi.jobs import MetadataField
    result = trainer.predict(population, snapshot=snapshot,
        output_config=PredictionOutputConfig(FileConnector(root_dir=str(output)),
            output_paths={'predictions':'scores'}, extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))
    scores = result.frame
    expected = scoring.rename(columns={'ENTITY1':'ENTITY'})
    scores['ANCHOR_TIMESTAMP'] = pd.to_datetime(scores['ANCHOR_TIMESTAMP'])
    aligned = expected.merge(scores, left_on=['ENTITY','TIMESTAMP'],
        right_on=['ENTITY','ANCHOR_TIMESTAMP'], how='outer', validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all():
        raise ValueError('Predictions must match each entity and anchor timestamp.')
    actual = heldout[['entity','date','OT']].rename(columns={'entity':'ENTITY','date':'TIMESTAMP','OT':'ACTUAL'})
    actual['TIMESTAMP'] = pd.to_datetime(actual['TIMESTAMP']) - pd.Timedelta(hours=1)
    aligned = aligned.merge(actual, on=['ENTITY','TIMESTAMP'], validate='one_to_one').sort_values('TIMESTAMP')
    print('MAE:', (aligned['TARGET_PRED'] - aligned['ACTUAL']).abs().mean())
    if args.horizon < 1:
        parser.error('--horizon must be positive.')
    rolling = aligned[['TARGET_PRED', 'ACTUAL']].rolling(args.horizon).mean().dropna()
    print('Rolling mean MAE:', (rolling['TARGET_PRED'] - rolling['ACTUAL']).abs().mean())
    print(scores.head(10).to_string(index=False))
    scores.to_parquet(output / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
