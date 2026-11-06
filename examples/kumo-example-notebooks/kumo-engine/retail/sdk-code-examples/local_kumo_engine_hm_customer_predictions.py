"""Predict H&M churn, conditional lifetime value, recommendations or daily revenue.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_retail/generate.py --dataset fashion --output-dir ./synthetic-fashion
Use that directory as --source-root or --input-dir. Metrics are demonstration only.
Install: pip install "kumo-engine[s3,text]"
See https://github.com/NVIDIA/kumo-engine for Engine setup.
"""
import argparse
from pathlib import Path
import pandas as pd
from kumo.engine import (Edge, FileConnector, Graph, ModelOutputConfig,
                         PredictiveQuery, S3Connector,
                         Table, Trainer, configure)
from kumoapi.model_plan import TrainingTableGenerationPlan

QUERIES = {'churn': 'PREDICT COUNT(transactions.*, 0, 90, days) = 0 FOR EACH customers.customer_id WHERE COUNT(transactions.*, -90, 0, days) > 0', 'ltv': 'PREDICT SUM(transactions.price, 0, 90, days) FOR EACH customers.customer_id WHERE COUNT(transactions.*, 0, 90, days) > 0', 'recommendation': 'PREDICT LIST_DISTINCT(transactions.article_id, 0, 7, days) RANK TOP 12 FOR EACH customers.customer_id', 'daily_forecast': 'PREDICT SUM(transactions.price, 0, 1, days) FOR EACH customers.customer_id'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES), default='churn')
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
        'customers': Table.from_source_table(source['customers'], primary_key='customer_id'),
        'articles': Table.from_source_table(source['articles'], column_names=[c.name for c in source['articles'].columns if c.name not in ['product_code', 'detail_desc']], primary_key='article_id'),
        'transactions': Table.from_source_table(source['transactions_train'], time_column='t_dat'),
    }
    graph = Graph(tables, [Edge(*edge) for edge in [('transactions', 'customer_id', 'customers'), ('transactions', 'article_id', 'articles')]])

    graph.validate()
    # Graph visualization preview, requires a release with Graph.visualize.
    # graph.visualize(output / 'relationships.svg')
    query = PredictiveQuery(graph, QUERIES[args.problem])
    query.validate()
    snapshot = graph.snapshot()
    if args.load_model:
        trainer = Trainer.load(args.load_model)
    else:
        training = query.generate_training_table(snapshot, plan=TrainingTableGenerationPlan(forecast_length=3, lag_timesteps=1) if args.problem == 'daily_forecast' else None)
        plan = query.suggest_model_plan(snapshot)  # Supported FAST defaults.

        trainer = Trainer(plan)
        result = trainer.fit(training, snapshot=snapshot,
                             export_config=ModelOutputConfig(FileConnector(root_dir=str(output)), path='model'))
        print('Evaluation metrics (synthetic, demonstration only):', result.metrics)
        print('Explanation artifacts:', trainer.scratch_dir)

    # Batch predictions
    # The generated population contains entity IDs and scoring times, not labels.
    # Future label conditions define training examples, not scoring eligibility.
    population_query = PredictiveQuery(graph, QUERIES[args.problem].split(' WHERE ')[0] if args.problem == 'ltv' else QUERIES[args.problem])
    population = population_query.generate_prediction_table(snapshot, plan=trainer.training_table_generation_plan)
    scoring = pd.read_parquet(population.table_path)
    from kumo.engine import PredictionTable
    scoring.to_parquet(output / 'population.parquet', index=False)
    population = PredictionTable.from_source_table(FileConnector(root_dir=str(output))['population.parquet'],
        pq=query, snapshot=snapshot, plan=trainer.training_table_generation_plan)
    if len(scoring) < 2:
        raise ValueError('Choose a scoring population with at least two rows.')
    task_type = query.get_task_type()
    options = {}
    if task_type.value in {'binary_classification', 'multilabel_classification'}:
        options['binary_classification_threshold'] = 0.5
    if task_type.is_link_pred:
        # Match the query's requested recommendation count.
        import re
        top = re.search(r'RANK TOP (\d+)', query.query, re.IGNORECASE)
        options['num_classes_to_return'] = int(top.group(1)) if top else 10
    scores = trainer.predict(population, snapshot=snapshot, **options).frame
    if 'ENTITY' not in scores or 'ENTITY1' not in scoring:
        raise ValueError('Expected entity columns are absent from scoring results.')
    if set(scores['ENTITY']) != set(scoring['ENTITY1']):
        raise ValueError('Prediction IDs do not match the scoring population.')
    if (task_type.is_link_pred or task_type.value in {'multiclass_classification', 'multilabel_classification', 'multilabel_ranking'}):
        if not {'CLASS', 'SCORE'}.issubset(scores.columns):
            raise ValueError('Expected CLASS and SCORE for ranked predictions.')
        if scores.duplicated(['ENTITY', 'CLASS']).any():
            raise ValueError('Duplicate ranked candidates for a scoring entity.')
        print('CLASS identifies the candidate; SCORE orders candidates within each entity.')
    elif not (trainer.training_table_generation_plan and isinstance(trainer.training_table_generation_plan.forecast_length, int) and trainer.training_table_generation_plan.forecast_length > 1):
        if len(scores) != len(scoring) or scores['ENTITY'].duplicated().any():
            raise ValueError('Expected one prediction per scoring entity.')
        print('TARGET_PRED is the predicted outcome; probabilities appear for classification.')
    else:
        horizon = trainer.training_table_generation_plan.forecast_length
        counts = scores.groupby('ENTITY').size()
        if not counts.eq(horizon).all() or len(scores) != len(scoring) * horizon:
            raise ValueError('Expected one forecast value per entity and horizon step.')
        if 'ANCHOR_TIMESTAMP' not in scores or 'TIMESTAMP' not in scores:
            raise ValueError('Forecast outputs must preserve anchor and forecast times.')
        if scores.duplicated(['ENTITY', 'ANCHOR_TIMESTAMP', 'TIMESTAMP']).any():
            raise ValueError('Duplicate forecast horizons.')
        print('TARGET_PRED is the value at TIMESTAMP; ANCHOR_TIMESTAMP is the forecast origin.')

    print(scores.head(10).to_string(index=False))
    scores.to_parquet(output / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
