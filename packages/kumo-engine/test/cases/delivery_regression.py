# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delivery regression through the installed Engine wheel."""

from common import parse_args, run_case


def main():
    # Provisional free-SSD budgets. Qualify them on each scale before scheduling.
    args = parse_args('delivery_regression', {
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
                'order_items':
                table('order_items', 'order_item_id', 'item_time', [
                    ('order_item_id', 'ID', 'int64'),
                    ('item_time', 'timestamp', 'time'),
                    ('order_id', 'ID', 'int64'),
                    ('item_category', 'categorical', 'string'),
                    ('quantity', 'numerical', 'int64'),
                    ('unit_price', 'numerical', 'float64'),
                    ('modifiers', 'multicategorical', 'string'),
                ]),
                'orders':
                table('orders', 'order_id', 'order_time', [
                    ('order_id', 'ID', 'int64'),
                    ('order_time', 'timestamp', 'time'),
                    ('user_id', 'ID', 'int64'),
                    ('store_id', 'ID', 'int64'),
                    ('daypart', 'categorical', 'string'),
                    ('item_count', 'numerical', 'int64'),
                    ('subtotal', 'numerical', 'float64'),
                    ('tip', 'numerical', 'float64'),
                    ('delivery_minutes', 'numerical', 'int64'),
                    ('promo_applied', 'categorical', 'bool'),
                ]),
                'regions':
                table('regions', 'region_id', None, [
                    ('region_id', 'ID', 'int64'),
                    ('density_tier', 'categorical', 'string'),
                    ('climate_zone', 'categorical', 'string'),
                    ('base_delivery_fee', 'numerical', 'float64'),
                ]),
                'store_groups':
                table('store_groups', 'store_group_id', None, [
                    ('store_group_id', 'ID', 'int64'),
                    ('region_id', 'ID', 'int64'),
                    ('cuisine', 'categorical', 'string'),
                    ('price_tier', 'categorical', 'string'),
                    ('is_franchise', 'categorical', 'bool'),
                ]),
                'store_impressions':
                table('store_impressions', None, 'impression_time', [
                    ('impression_time', 'timestamp', 'time'),
                    ('user_id', 'ID', 'int64'),
                    ('store_id', 'ID', 'int64'),
                    ('position', 'numerical', 'int64'),
                    ('clicked', 'categorical', 'bool'),
                    ('surface', 'categorical', 'string'),
                ]),
                'stores':
                table('stores', 'store_id', 'opened_time', [
                    ('store_id', 'ID', 'int64'),
                    ('opened_time', 'timestamp', 'time'),
                    ('store_group_id', 'ID', 'int64'),
                    ('rating', 'numerical', 'float64'),
                    ('delivery_radius_km', 'numerical', 'float64'),
                    ('prep_minutes', 'numerical', 'int64'),
                    ('cuisine_tags', 'multicategorical', 'string'),
                ]),
                'users':
                table('users', 'user_id', 'signup_time', [
                    ('user_id', 'ID', 'int64'),
                    ('signup_time', 'timestamp', 'time'),
                    ('home_zone', 'categorical', 'string'),
                    ('platform', 'categorical', 'string'),
                    ('membership_tier', 'categorical', 'string'),
                    ('age_band', 'categorical', 'string'),
                    ('marketing_opt_in', 'categorical', 'bool'),
                ]),
            }, edges=[
                Edge('store_groups', 'region_id', 'regions'),
                Edge('stores', 'store_group_id', 'store_groups'),
                Edge('orders', 'user_id', 'users'),
                Edge('orders', 'store_id', 'stores'),
                Edge('order_items', 'order_id', 'orders'),
                Edge('store_impressions', 'user_id', 'users'),
                Edge('store_impressions', 'store_id', 'stores'),
            ])
        graph.validate()
        query = kumo.PredictiveQuery(graph, (
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE COUNT(orders.*, '
            '-90, 0, days) > 0 '))
        query.validate()
        with run.phase("snapshot"):
            snapshot = graph.snapshot()
        run.record_graph(snapshot)

        generation = TrainingTableGenerationPlan(
            split=("TimeRangeSplit(["
                   "('2025-10-02T00:00:00', '2025-11-01T00:00:00'), "
                   "('2025-11-01T00:00:00', '2025-12-01T00:00:00'), "
                   "('2025-12-01T00:00:00', '2025-12-31T00:00:00')"
                   "], timeframe=30, flexible_boundaries=False)"),
            timeframe_step=30)
        with run.phase("training_table"):
            training_table = query.generate_training_table(
                snapshot, plan=generation)
        run.record_table(training_table, unordered_targets=False)

        plan = query.suggest_model_plan(snapshot)
        plan.training_job.num_experiments = 2
        plan.training_job.pruning = None
        plan.training_job.manual_seed = 0
        plan.training_job.refit_trainval = False
        plan.training_job.refit_full = False
        plan.training_job.disable_explain = True
        plan.training_job.metrics = ['mae']
        plan.training_job.tune_metric = 'mae'
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
                prediction_table, snapshot=snapshot,
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
        run.check_predictions(first, repeated, reloaded, task='regression')


if __name__ == "__main__":
    main()
