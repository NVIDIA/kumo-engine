"""Recommend future customers for each product in the public Olist dataset.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_retail/generate.py --dataset olist --output-dir ./synthetic-olist
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

QUERIES = {'customers': 'PREDICT LIST_DISTINCT(orders.customer_id, 0, 90, days) RANK TOP 10 FOR EACH products.product_id'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem', choices=list(QUERIES), default='customers')
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
        'orders': Table.from_source_table(source['olist_item_orders_dataset_unique'], primary_key='order_id', time_column='order_purchase_timestamp'),
        'customers': Table.from_source_table(source['olist_customers_dataset'], primary_key='customer_id'),
        'payments': Table.from_source_table(source['olist_order_payments_dataset']),
        'reviews': Table.from_source_table(source['olist_order_reviews_dataset'], column_names=[c.name for c in source['olist_order_reviews_dataset'].columns if c.name not in ['review_comment_title', 'review_answer_timestamp', 'review_creation_date']], primary_key='review_id'),
        'products': Table.from_source_table(source['olist_products_dataset'], primary_key='product_id'),
    }
    graph = Graph(tables, [Edge(*edge) for edge in [('orders', 'customer_id', 'customers'), ('orders', 'product_id', 'products'), ('payments', 'order_id', 'orders'), ('reviews', 'order_id', 'orders')]])
    from kumoapi.typing import Stype
    tables['customers'] = tables['customers'].with_column('customer_city', stype=Stype.categorical)
    tables['products'] = tables['products'].with_column('product_category_name', stype=Stype.text)
    tables['reviews'] = tables['reviews'].with_column('review_comment_message', stype=Stype.text)
    graph = Graph(tables, graph.edges)
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
