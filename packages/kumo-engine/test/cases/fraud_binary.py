# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fraud binary through the installed Engine wheel."""

from common import parse_args, run_case


def main():
    # Provisional free-SSD budgets. Qualify them on each scale before scheduling.
    args = parse_args('fraud_binary', {
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
                'account_flags':
                table('account_flags', None, 'flag_time', [
                    ('flag_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                ]),
                'accounts':
                table('accounts', 'account_id', 'created_time', [
                    ('account_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('email_domain', 'categorical', 'string'),
                    ('country', 'categorical', 'string'),
                    ('account_type', 'categorical', 'string'),
                    ('loyalty_tier', 'categorical', 'string'),
                    ('is_verified', 'categorical', 'bool'),
                ]),
                'devices':
                table('devices', 'device_id', 'first_seen_time', [
                    ('device_id', 'ID', 'int64'),
                    ('first_seen_time', 'timestamp', 'time'),
                    ('os', 'categorical', 'string'),
                    ('browser', 'categorical', 'string'),
                    ('device_type', 'categorical', 'string'),
                    ('language', 'categorical', 'string'),
                    ('timezone', 'categorical', 'string'),
                ]),
                'ip_addresses':
                table('ip_addresses', 'ip_id', None, [
                    ('ip_id', 'ID', 'int64'),
                    ('isp', 'categorical', 'string'),
                    ('country', 'categorical', 'string'),
                    ('region', 'categorical', 'string'),
                    ('is_proxy', 'categorical', 'bool'),
                    ('is_datacenter', 'categorical', 'bool'),
                ]),
                'logins':
                table('logins', 'login_id', 'login_time', [
                    ('login_id', 'ID', 'int64'),
                    ('login_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('device_id', 'ID', 'int64'),
                    ('ip_id', 'ID', 'int64'),
                ]),
                'merchants':
                table('merchants', 'merchant_id', None, [
                    ('merchant_id', 'ID', 'int64'),
                    ('category', 'categorical', 'string'),
                    ('country', 'categorical', 'string'),
                    ('city', 'categorical', 'string'),
                    ('channel', 'categorical', 'string'),
                ]),
                'payment_methods':
                table('payment_methods', 'payment_method_id', None, [
                    ('payment_method_id', 'ID', 'int64'),
                    ('card_network', 'categorical', 'string'),
                    ('card_type', 'categorical', 'string'),
                    ('issuing_country', 'categorical', 'string'),
                    ('payer_name', 'multicategorical', 'string'),
                ]),
                'transactions':
                table('transactions', 'transaction_id', 'transaction_time', [
                    ('transaction_id', 'ID', 'int64'),
                    ('transaction_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('payment_method_id', 'ID', 'int64'),
                    ('merchant_id', 'ID', 'int64'),
                    ('amount_usd', 'numerical', 'float64'),
                    ('quantity', 'numerical', 'int64'),
                    ('currency', 'categorical', 'string'),
                    ('channel', 'categorical', 'string'),
                    ('billing_country', 'categorical', 'string'),
                    ('billing_postal_codes', 'multicategorical', 'string'),
                    ('risk_rule_hits', 'multicategorical', 'string'),
                    ('email_matches_payer', 'categorical', 'bool'),
                    ('installments', 'numerical', 'int64'),
                    ('traveler_name', 'categorical', 'string'),
                ]),
                'transfers':
                table('transfers', 'transfer_id', 'transfer_time', [
                    ('transfer_id', 'ID', 'int64'),
                    ('transfer_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('payee_id', 'ID', 'int64'),
                    ('login_id', 'ID', 'int64'),
                    ('amount_usd', 'numerical', 'float64'),
                ]),
            }, edges=[
                Edge('logins', 'account_id', 'accounts'),
                Edge('logins', 'device_id', 'devices'),
                Edge('logins', 'ip_id', 'ip_addresses'),
                Edge('transfers', 'account_id', 'accounts'),
                Edge('transfers', 'payee_id', 'accounts'),
                Edge('transfers', 'login_id', 'logins'),
                Edge('transactions', 'account_id', 'accounts'),
                Edge('transactions', 'payment_method_id', 'payment_methods'),
                Edge('transactions', 'merchant_id', 'merchants'),
                Edge('account_flags', 'account_id', 'accounts'),
            ])
        graph.validate()
        query = kumo.PredictiveQuery(graph, (
            'PREDICT COUNT(account_flags.*, 0, 28, days) > 0 FOR EACH accounts.account_id WHERE '
            'COUNT(account_flags.*, -INF, 0, days) = 0 AND COUNT(logins.*, -30, 0, days) > 0 '
        ))
        query.validate()
        with run.phase("snapshot"):
            snapshot = graph.snapshot()
        run.record_graph(snapshot)

        generation = TrainingTableGenerationPlan(
            split=("TimeRangeSplit(["
                   "('2025-05-21T00:00:00', '2025-06-18T00:00:00'), "
                   "('2025-06-18T00:00:00', '2025-07-16T00:00:00'), "
                   "('2025-07-16T00:00:00', '2025-08-13T00:00:00')"
                   "], timeframe=28, flexible_boundaries=False)"),
            timeframe_step=28)
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
