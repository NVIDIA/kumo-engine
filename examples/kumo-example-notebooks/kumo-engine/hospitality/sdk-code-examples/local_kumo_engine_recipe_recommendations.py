"""Recommend available recipes from authorized recipe/order tables.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_hospitality/generate.py --dataset recipes --output-dir ./synthetic-recipes
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

QUERIES = {'recipes': 'PREDICT LIST_DISTINCT(orders.rid, 0, 7, days) RANK TOP 10 FOR EACH customers.customer_id'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES), default='recipes')
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
        'orders': Table.from_source_table(source['orders'], time_column='order_date'),
        'recipes': Table.from_source_table(source['recipes'], primary_key='rid', time_column='start_date', end_time_column='end_date'),
        'customers': Table.from_source_table(source['customers'], primary_key='customer_id'),
        'menus': Table.from_source_table(source['menus'], primary_key='menu_id'),
    }
    graph = Graph(tables, [Edge(*edge) for edge in [('orders', 'customer_id', 'customers'), ('orders', 'menu_id', 'menus'), ('orders', 'rid', 'recipes')]])

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
        result = trainer.fit(training, snapshot=snapshot,
                             export_config=ModelOutputConfig(FileConnector(root_dir=str(output)), path='model'))
        print('Evaluation metrics (synthetic, demonstration only):', result.metrics)
        print('Explanation artifacts:', trainer.scratch_dir)

    # Batch predictions
    # The generated population contains entity IDs and scoring times, not labels.
    population = query.generate_prediction_table(snapshot, plan=trainer.training_table_generation_plan)
    scoring = pd.read_parquet(population.table_path)
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
    elif task_type.value != 'forecasting':
        if len(scores) != len(scoring) or scores['ENTITY'].duplicated().any():
            raise ValueError('Expected one prediction per scoring entity.')
        print('TARGET_PRED is the predicted outcome; probabilities appear for classification.')
    else:
        print('Forecast outputs retain a separate value for each forecast horizon.')

    print(scores.head(10).to_string(index=False))
    scores.to_parquet(output / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
