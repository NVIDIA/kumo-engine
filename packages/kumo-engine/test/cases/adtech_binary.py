# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adtech binary through the installed Engine wheel."""

from common import parse_args, run_case


def main():
    # Provisional free-SSD budgets. Qualify them on each scale before scheduling.
    args = parse_args('adtech_binary', {
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
                'ads':
                table('ads', 'ad_id', 'created_time', [
                    ('ad_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('campaign_id', 'ID', 'int64'),
                    ('format', 'categorical', 'string'),
                    ('creative_size', 'categorical', 'string'),
                    ('call_to_action', 'categorical', 'string'),
                    ('segments', 'multicategorical', 'string'),
                    ('quality_score', 'numerical', 'float64'),
                ]),
                'advertisers':
                table('advertisers', 'advertiser_id', 'created_time', [
                    ('advertiser_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('vertical', 'categorical', 'string'),
                    ('country', 'categorical', 'string'),
                    ('account_tier', 'categorical', 'string'),
                    ('is_verified', 'categorical', 'bool'),
                ]),
                'campaigns':
                table('campaigns', 'campaign_id', 'start_time', [
                    ('campaign_id', 'ID', 'int64'),
                    ('start_time', 'timestamp', 'time'),
                    ('advertiser_id', 'ID', 'int64'),
                    ('objective', 'categorical', 'string'),
                    ('budget_tier', 'categorical', 'string'),
                    ('bid_strategy', 'categorical', 'string'),
                    ('daily_budget', 'numerical', 'float64'),
                ]),
                'clicks':
                table('clicks', 'click_id', 'click_time', [
                    ('click_id', 'ID', 'int64'),
                    ('click_time', 'timestamp', 'time'),
                    ('user_id', 'ID', 'int64'),
                    ('ad_id', 'ID', 'int64'),
                    ('dwell_seconds', 'numerical', 'float64'),
                ]),
                'conversions':
                table('conversions', 'conversion_id', 'conversion_time', [
                    ('conversion_id', 'ID', 'int64'),
                    ('conversion_time', 'timestamp', 'time'),
                    ('user_id', 'ID', 'int64'),
                    ('ad_id', 'ID', 'int64'),
                    ('conversion_type', 'categorical', 'string'),
                    ('value', 'numerical', 'float64'),
                ]),
                'impressions':
                table('impressions', None, 'impression_time', [
                    ('impression_time', 'timestamp', 'time'),
                    ('user_id', 'ID', 'int64'),
                    ('ad_id', 'ID', 'int64'),
                    ('placement', 'categorical', 'string'),
                    ('position', 'numerical', 'int64'),
                    ('viewability', 'numerical', 'float64'),
                ]),
                'interests':
                table('interests', 'interest_id', None, [
                    ('interest_id', 'ID', 'int64'),
                    ('topic', 'categorical', 'string'),
                    ('audience_size', 'numerical', 'int64'),
                ]),
                'user_interests':
                table('user_interests', None, 'created_time', [
                    ('created_time', 'timestamp', 'time'),
                    ('user_id', 'ID', 'int64'),
                    ('interest_id', 'ID', 'int64'),
                    ('affinity', 'numerical', 'float64'),
                ]),
                'users':
                table('users', 'user_id', 'signup_time', [
                    ('user_id', 'ID', 'int64'),
                    ('signup_time', 'timestamp', 'time'),
                    ('geo', 'categorical', 'string'),
                    ('os', 'categorical', 'string'),
                    ('age_bucket', 'categorical', 'string'),
                    ('language', 'categorical', 'string'),
                    ('opted_in', 'categorical', 'bool'),
                ]),
            }, edges=[
                Edge('user_interests', 'user_id', 'users'),
                Edge('user_interests', 'interest_id', 'interests'),
                Edge('campaigns', 'advertiser_id', 'advertisers'),
                Edge('ads', 'campaign_id', 'campaigns'),
                Edge('impressions', 'user_id', 'users'),
                Edge('impressions', 'ad_id', 'ads'),
                Edge('clicks', 'user_id', 'users'),
                Edge('clicks', 'ad_id', 'ads'),
                Edge('conversions', 'user_id', 'users'),
                Edge('conversions', 'ad_id', 'ads'),
            ])
        graph.validate()
        query = kumo.PredictiveQuery(graph, (
            'PREDICT COUNT(conversions.*, 0, 7, days) > 0 FOR EACH users.user_id WHERE '
            'COUNT(impressions.*, -14, 0, days) > 0 '))
        query.validate()
        with run.phase("snapshot"):
            snapshot = graph.snapshot()
        run.record_graph(snapshot)

        generation = TrainingTableGenerationPlan(
            split=("TimeRangeSplit(["
                   "('2025-12-10T00:00:00', '2025-12-17T00:00:00'), "
                   "('2025-12-17T00:00:00', '2025-12-24T00:00:00'), "
                   "('2025-12-24T00:00:00', '2025-12-31T00:00:00')"
                   "], timeframe=7, flexible_boundaries=False)"),
            timeframe_step=7)
        with run.phase("training_table"):
            training_table = query.generate_training_table(
                snapshot, plan=generation)
        run.record_table(training_table, unordered_targets=False)

        # Keep old-model compatibility in this one small nightly case.
        if args.compatibility:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as parquet

            if args.scale != '100M' or args.older_model is None or args.older_predictions is None:
                raise ValueError(
                    'Compatibility requires the 100M case, an older model and its approved predictions'
                )
            expected = pd.read_parquet(args.older_predictions)
            if not 1 <= len(expected) <= 20:
                raise ValueError(
                    'The compatibility fixture must contain 1 to 20 predictions'
                )
            population = pa.Table.from_pandas(
                expected[['ENTITY', 'ANCHOR_TIMESTAMP'
                          ]].rename(columns={
                              'ENTITY': 'ENTITY1',
                              'ANCHOR_TIMESTAMP': 'TIMESTAMP'
                          }), preserve_index=False)
            fixture_path = args.scratch / 'compatibility-input.parquet'
            parquet.write_table(population, fixture_path)
            fixture_table = kumo.PredictionTable.from_source_table(
                kumo.FileConnector(root_dir=args.scratch)[fixture_path.name],
                pq=query, snapshot=snapshot, plan=generation)
            older = kumo.Trainer.load(args.older_model, accelerator='gpu')
            with run.phase('compatibility'):
                observed = older.predict(
                    fixture_table, snapshot=snapshot,
                    binary_classification_threshold=0.5,
                    output_config=kumo.PredictionOutputConfig(
                        connector=kumo.FileConnector(root_dir=args.scratch /
                                                     'predictions'),
                        output_paths={'predictions': 'compatibility'},
                        extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))
            run.check_predictions(expected, observed, observed, task='binary',
                                  population=population)
            run.data['compatibility'] = dict(run.data['predictions'])
            del older

        plan = query.suggest_model_plan(snapshot)
        plan.training_job.num_experiments = 2
        plan.training_job.pruning = None
        plan.training_job.manual_seed = 0
        plan.training_job.refit_trainval = False
        plan.training_job.refit_full = False
        plan.training_job.disable_explain = True
        plan.training_job.metrics = ['auroc']
        plan.training_job.tune_metric = 'auroc'
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
                binary_classification_threshold=0.5,
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
        run.check_predictions(first, repeated, reloaded, task='binary')


if __name__ == "__main__":
    main()
