# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Commerce temporal link through the installed Engine wheel."""

from common import parse_args, run_case


def main():
    # Provisional free-SSD budgets. Qualify them on each scale before scheduling.
    args = parse_args('commerce_temporal_link', {
        "100M": 100,
        "1B": 300,
        "10B": 1600,
        "25B": 3200
    })
    with run_case(args) as run:
        from kumoapi.graph import Edge
        from kumoapi.jobs import MetadataField
        from kumoapi.model_plan import TrainingTableGenerationPlan
        from kumoapi.table import Column, TableDefinition

        import kumo.engine as kumo

        kumo.configure(scratch_dir=args.scratch)
        sources = kumo.S3Connector(root_dir=args.source, region=args.region)

        def table(name, key, clock, columns):
            return kumo.Table(
                sources, name, definition=TableDefinition(
                    source_table=sources[name].to_source_table(), pkey=key,
                    time_col=clock, cols=[
                        Column(
                            column, stype, dtype, timestamp_format="ms"
                            if stype == "timestamp" else None,
                            multicategorical_delimiter="|"
                            if stype == "multicategorical" else None)
                        for column, stype, dtype in columns
                    ]))

        graph = kumo.Graph(
            tables={
                'customers':
                table('customers', 'customer_id', 'signup_time', [
                    ('customer_id', 'ID', 'int64'),
                    ('signup_time', 'timestamp', 'time'),
                    ('region', 'categorical', 'string'),
                    ('city', 'categorical', 'string'),
                    ('membership', 'categorical', 'string'),
                    ('platform', 'categorical', 'string'),
                    ('marketing_opt_in', 'categorical', 'bool'),
                ]),
                'favorites':
                table('favorites', None, 'created_time', [
                    ('created_time', 'timestamp', 'time'),
                    ('customer_id', 'ID', 'int64'),
                    ('merchant_id', 'ID', 'int64'),
                ]),
                'merchant_views':
                table('merchant_views', None, 'view_time', [
                    ('view_time', 'timestamp', 'time'),
                    ('customer_id', 'ID', 'int64'),
                    ('merchant_id', 'ID', 'int64'),
                    ('source', 'categorical', 'string'),
                ]),
                'merchants':
                table('merchants', 'merchant_id', 'created_time', [
                    ('merchant_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('region', 'categorical', 'string'),
                    ('primary_category', 'categorical', 'string'),
                    ('tags', 'multicategorical', 'string'),
                    ('price_tier', 'categorical', 'string'),
                    ('is_partner', 'categorical', 'bool'),
                    ('avg_rating', 'numerical', 'float64'),
                ]),
                'purchases':
                table('purchases', 'purchase_id', 'purchase_time', [
                    ('purchase_id', 'ID', 'int64'),
                    ('purchase_time', 'timestamp', 'time'),
                    ('customer_id', 'ID', 'int64'),
                    ('merchant_id', 'ID', 'int64'),
                    ('channel', 'categorical', 'string'),
                    ('fulfillment', 'categorical', 'string'),
                    ('item_count', 'numerical', 'int64'),
                    ('subtotal', 'numerical', 'float64'),
                    ('fees', 'numerical', 'float64'),
                    ('tip', 'numerical', 'float64'),
                    ('is_first_purchase', 'categorical', 'bool'),
                    ('rating', 'numerical', 'int64'),
                ]),
                'reviews':
                table('reviews', None, 'review_time', [
                    ('review_time', 'timestamp', 'time'),
                    ('customer_id', 'ID', 'int64'),
                    ('merchant_id', 'ID', 'int64'),
                    ('stars', 'numerical', 'int64'),
                ]),
                'searches':
                table('searches', None, 'search_time', [
                    ('search_time', 'timestamp', 'time'),
                    ('customer_id', 'ID', 'int64'),
                    ('query_term', 'categorical', 'string'),
                    ('category', 'categorical', 'string'),
                    ('result_page', 'numerical', 'int64'),
                ]),
            }, edges=[
                Edge('purchases', 'customer_id', 'customers'),
                Edge('purchases', 'merchant_id', 'merchants'),
                Edge('merchant_views', 'customer_id', 'customers'),
                Edge('merchant_views', 'merchant_id', 'merchants'),
                Edge('searches', 'customer_id', 'customers'),
                Edge('favorites', 'customer_id', 'customers'),
                Edge('favorites', 'merchant_id', 'merchants'),
                Edge('reviews', 'customer_id', 'customers'),
                Edge('reviews', 'merchant_id', 'merchants'),
            ])
        graph.validate()
        query = kumo.PredictiveQuery(graph, (
            'PREDICT LIST_DISTINCT(purchases.merchant_id, 0, 7, days) RANK TOP 10 FOR EACH '
            'customers.customer_id '))
        query.validate()
        with run.phase("snapshot"):
            snapshot = graph.snapshot()
        run.record_graph(snapshot)

        generation = TrainingTableGenerationPlan(
            split=("TimeRangeSplit(["
                   "('2024-12-05T00:00:00', '2024-12-12T00:00:00'), "
                   "('2024-12-12T00:00:00', '2024-12-19T00:00:00'), "
                   "('2024-12-19T00:00:00', '2024-12-26T00:00:00')"
                   "], timeframe=7, flexible_boundaries=False)"),
            timeframe_step=7)
        with run.phase("training_table"):
            training_table = query.generate_training_table(
                snapshot, plan=generation)
        run.record_table(training_table, unordered_targets=True)

        plan = query.suggest_model_plan(snapshot)
        plan.training_job.num_experiments = 2
        plan.training_job.pruning = None
        plan.training_job.manual_seed = 0
        plan.training_job.refit_trainval = False
        plan.training_job.refit_full = False
        plan.training_job.disable_explain = True
        plan.training_job.metrics = ['recall@10']
        plan.training_job.tune_metric = 'recall@10'
        plan.optimization.max_epochs = 3
        plan.optimization.min_steps_per_epoch = 200
        plan.optimization.max_steps_per_epoch = 200
        plan.optimization.max_val_steps = 100
        plan.optimization.max_test_steps = 200
        trainer = kumo.Trainer(plan, accelerator="gpu")
        with run.phase("train"):
            fitted = trainer.fit(training_table, snapshot=snapshot)
        run.record_metrics(fitted.metrics)

        # Select the bounded population before asking the Engine to predict.
        prediction_table = run.prediction_table(training_table, query, snapshot,
                                                forecast=False)

        def predict(model, path):
            return model.predict(
                prediction_table, snapshot=snapshot, num_classes_to_return=11,
                output_config=kumo.PredictionOutputConfig(
                    connector=kumo.FileConnector(root_dir=str(args.scratch /
                                                              "predictions")),
                    output_paths={"predictions": path},
                    extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))

        with run.phase("predict"):
            first = predict(trainer, "first")
        repeated = predict(trainer, "repeat")
        del trainer
        restored = kumo.Trainer.load(fitted.bundle_dir, accelerator="gpu")
        reloaded = predict(restored, "reload")
        run.check_predictions(first, repeated, reloaded, task='temporal_link',
                              top_k=10)


if __name__ == "__main__":
    main()
