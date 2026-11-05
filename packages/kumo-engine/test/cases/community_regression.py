# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Community regression through the installed Engine wheel."""

from common import parse_args, run_case


def main():
    # Provisional free-SSD budgets. Qualify them on each scale before scheduling.
    args = parse_args('community_regression', {
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
                'accounts':
                table('accounts', 'account_id', 'created_time', [
                    ('account_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('email_domain_id', 'ID', 'int64'),
                    ('locale', 'categorical', 'string'),
                    ('client', 'categorical', 'string'),
                    ('age_bucket', 'categorical', 'string'),
                    ('interests', 'multicategorical', 'string'),
                    ('is_verified', 'categorical', 'bool'),
                ]),
                'communities':
                table('communities', 'community_id', 'created_time', [
                    ('community_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('category', 'categorical', 'string'),
                    ('size_tier', 'categorical', 'string'),
                    ('language', 'categorical', 'string'),
                    ('tags', 'multicategorical', 'string'),
                    ('is_public', 'categorical', 'bool'),
                    ('rules_count', 'numerical', 'int64'),
                ]),
                'devices':
                table('devices', 'device_id', 'first_seen_time', [
                    ('device_id', 'ID', 'int64'),
                    ('first_seen_time', 'timestamp', 'time'),
                    ('os', 'categorical', 'string'),
                    ('device_class', 'categorical', 'string'),
                    ('model', 'categorical', 'string'),
                ]),
                'email_domains':
                table('email_domains', 'email_domain_id', None, [
                    ('email_domain_id', 'ID', 'int64'),
                    ('tld', 'categorical', 'string'),
                    ('is_free_provider', 'categorical', 'bool'),
                    ('domain_age_years', 'numerical', 'float64'),
                ]),
                'logins':
                table('logins', None, 'login_time', [
                    ('login_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('network_id', 'ID', 'int64'),
                    ('device_id', 'ID', 'int64'),
                    ('platform', 'categorical', 'string'),
                    ('app_version', 'categorical', 'string'),
                    ('session_minutes', 'numerical', 'float64'),
                ]),
                'memberships':
                table('memberships', None, 'joined_time', [
                    ('joined_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('community_id', 'ID', 'int64'),
                    ('role', 'categorical', 'string'),
                    ('invite_source', 'categorical', 'string'),
                ]),
                'messages':
                table('messages', 'message_id', 'sent_time', [
                    ('message_id', 'ID', 'int64'),
                    ('sent_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('community_id', 'ID', 'int64'),
                    ('length_bucket', 'categorical', 'string'),
                    ('is_reply', 'categorical', 'bool'),
                    ('has_attachment', 'categorical', 'bool'),
                    ('mention_count', 'numerical', 'int64'),
                ]),
                'networks':
                table('networks', 'network_id', None, [
                    ('network_id', 'ID', 'int64'),
                    ('network_type', 'categorical', 'string'),
                    ('region', 'categorical', 'string'),
                    ('asn', 'categorical', 'string'),
                    ('is_hosting', 'categorical', 'bool'),
                ]),
                'reactions':
                table('reactions', None, 'reaction_time', [
                    ('reaction_time', 'timestamp', 'time'),
                    ('account_id', 'ID', 'int64'),
                    ('community_id', 'ID', 'int64'),
                    ('kind', 'categorical', 'string'),
                ]),
            }, edges=[
                Edge('accounts', 'email_domain_id', 'email_domains'),
                Edge('memberships', 'account_id', 'accounts'),
                Edge('memberships', 'community_id', 'communities'),
                Edge('messages', 'account_id', 'accounts'),
                Edge('messages', 'community_id', 'communities'),
                Edge('reactions', 'account_id', 'accounts'),
                Edge('reactions', 'community_id', 'communities'),
                Edge('logins', 'account_id', 'accounts'),
                Edge('logins', 'network_id', 'networks'),
                Edge('logins', 'device_id', 'devices'),
            ])
        graph.validate()
        query = kumo.PredictiveQuery(graph, (
            'PREDICT COUNT(messages.*, 0, 30, days) FOR EACH accounts.account_id WHERE '
            'COUNT(logins.*, -30, 0, days) > 0 '))
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
