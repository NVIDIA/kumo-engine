# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Payments binary through the installed Engine wheel."""

from common import parse_args, run_case


def main():
    # Provisional free-SSD budgets. Qualify them on each scale before scheduling.
    args = parse_args('payments_binary', {
        "100M": 100,
        "1B": 300,
        "10B": 1600,
        "25B": 3200
    })
    with run_case(args) as run:
        from kumoapi.graph import Edge
        from kumoapi.jobs import MetadataField
        from kumoapi.model_plan import (
            NumNeighborsConfig,
            TrainingTableGenerationPlan,
        )
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
                'account_events':
                table('account_events', None, 'event_time', [
                    ('event_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('event_type', 'categorical', 'string'),
                    ('channel', 'categorical', 'string'),
                ]),
                'account_flags':
                table('account_flags', None, 'flag_time', [
                    ('flag_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('source', 'categorical', 'string'),
                ]),
                'accounts':
                table('accounts', 'account_id', 'created_time', [
                    ('account_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('institution_id', 'ID', 'int64'),
                    ('account_kind', 'categorical', 'string'),
                    ('segment', 'categorical', 'string'),
                    ('holder_age_band', 'categorical', 'string'),
                    ('merchant_category', 'categorical', 'string'),
                    ('region', 'categorical', 'string'),
                    ('onboarding_channel', 'categorical', 'string'),
                    ('stated_monthly_volume', 'numerical', 'float64'),
                ]),
                'beneficiary_registrations':
                table('beneficiary_registrations', None, 'registration_time', [
                    ('registration_time', 'timestamp', 'time'),
                    ('registrant_id', 'ID', 'int64'),
                    ('beneficiary_id', 'ID', 'int64'),
                    ('channel', 'categorical', 'string'),
                ]),
                'institutions':
                table('institutions', 'institution_id', None, [
                    ('institution_id', 'ID', 'int64'),
                    ('institution_type', 'categorical', 'string'),
                    ('region', 'categorical', 'string'),
                    ('size_tier', 'categorical', 'string'),
                    ('is_digital', 'categorical', 'bool'),
                ]),
                'pair_interactions':
                table('pair_interactions', None, 'snapshot_time', [
                    ('snapshot_time', 'timestamp', 'time'),
                    ('payer_id', 'ID', 'int64'),
                    ('payee_id', 'ID', 'int64'),
                    ('first_txn_time', 'timestamp', 'time'),
                    ('last_txn_time', 'timestamp', 'time'),
                    ('total_txns', 'numerical', 'int64'),
                    ('total_failed', 'numerical', 'int64'),
                    ('avg_amount', 'numerical', 'float64'),
                    ('period_txns', 'numerical', 'int64'),
                    ('period_median_amount', 'numerical', 'float64'),
                    ('period_max_amount', 'numerical', 'float64'),
                    ('period_active_days', 'numerical', 'int64'),
                ]),
                'transactions':
                table('transactions', 'transaction_id', 'transaction_time', [
                    ('transaction_id', 'ID', 'int64'),
                    ('transaction_time', 'timestamp', 'time'),
                    ('payer_id', 'ID', 'int64'),
                    ('payee_id', 'ID', 'int64'),
                    ('amount', 'numerical', 'float64'),
                    ('status', 'categorical', 'string'),
                    ('channel', 'categorical', 'string'),
                    ('purpose', 'categorical', 'string'),
                ]),
            }, edges=[
                Edge('accounts', 'institution_id', 'institutions'),
                Edge('transactions', 'payer_id', 'accounts'),
                Edge('transactions', 'payee_id', 'accounts'),
                Edge('pair_interactions', 'payer_id', 'accounts'),
                Edge('pair_interactions', 'payee_id', 'accounts'),
                Edge('beneficiary_registrations', 'registrant_id', 'accounts'),
                Edge('beneficiary_registrations', 'beneficiary_id', 'accounts'),
                Edge('account_events', 'account_id', 'accounts'),
                Edge('account_flags', 'account_id', 'accounts'),
            ])
        graph.validate()
        query = kumo.PredictiveQuery(graph, (
            'PREDICT COUNT(account_flags.*, 0, 14, days) > 0 FOR EACH accounts.account_id WHERE '
            'COUNT(account_flags.*, -INF, 0, days) = 0 '))
        query.validate()
        with run.phase("snapshot"):
            snapshot = graph.snapshot()
        run.record_graph(snapshot)

        generation = TrainingTableGenerationPlan(
            split=("TimeRangeSplit(["
                   "('2024-02-18T00:00:00', '2024-03-03T00:00:00'), "
                   "('2024-03-03T00:00:00', '2024-03-17T00:00:00'), "
                   "('2024-03-17T00:00:00', '2024-03-31T00:00:00')"
                   "], timeframe=14, flexible_boundaries=False)"),
            timeframe_step=14)
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
        plan.training_job.metrics = ['auroc']
        plan.training_job.tune_metric = 'auroc'
        plan.optimization.max_epochs = 3
        plan.optimization.min_steps_per_epoch = 200
        plan.optimization.max_steps_per_epoch = 200
        plan.optimization.max_val_steps = 100
        plan.optimization.max_test_steps = 200
        # Exercise the complete six-hop path with the capped training budget.
        plan.neighbor_sampling.num_neighbors = [
            NumNeighborsConfig(hop1={"default": 8}, hop2={"default": 8},
                               hop3={"default": 8}, hop4={"default": 8},
                               hop5={"default": 8}, hop6={"default": 8})
        ]
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
