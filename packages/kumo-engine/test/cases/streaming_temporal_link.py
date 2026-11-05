# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming temporal link through the installed Engine wheel."""

from common import parse_args, run_case


def main():
    # Provisional free-SSD budgets. Qualify them on each scale before scheduling.
    args = parse_args('streaming_temporal_link', {
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
                'genres':
                table('genres', 'genre_id', None, [
                    ('genre_id', 'ID', 'int64'),
                    ('genre_group', 'categorical', 'string'),
                    ('is_kids', 'categorical', 'bool'),
                ]),
                'ratings':
                table('ratings', None, 'rated_time', [
                    ('rated_time', 'timestamp', 'time'),
                    ('viewer_id', 'ID', 'int64'),
                    ('title_id', 'ID', 'int64'),
                    ('thumbs_up', 'categorical', 'bool'),
                ]),
                'series':
                table('series', 'series_id', 'created_time', [
                    ('series_id', 'ID', 'int64'),
                    ('created_time', 'timestamp', 'time'),
                    ('genre_id', 'ID', 'int64'),
                    ('language', 'categorical', 'string'),
                    ('maturity', 'categorical', 'string'),
                    ('format', 'categorical', 'string'),
                ]),
                'sessions':
                table('sessions', 'session_id', 'start_time', [
                    ('session_id', 'ID', 'int64'),
                    ('start_time', 'timestamp', 'time'),
                    ('viewer_id', 'ID', 'int64'),
                    ('title_id', 'ID', 'int64'),
                    ('watch_seconds', 'numerical', 'int64'),
                    ('completion_frac', 'numerical', 'float64'),
                    ('device_type', 'categorical', 'string'),
                    ('entry_point', 'categorical', 'string'),
                    ('resumed', 'categorical', 'bool'),
                ]),
                'title_genres':
                table('title_genres', None, None, [
                    ('title_id', 'ID', 'int64'),
                    ('genre_id', 'ID', 'int64'),
                    ('relevance', 'numerical', 'float64'),
                ]),
                'titles':
                table('titles', 'title_id', 'added_time', [
                    ('title_id', 'ID', 'int64'),
                    ('added_time', 'timestamp', 'time'),
                    ('series_id', 'ID', 'int64'),
                    ('season_number', 'numerical', 'int64'),
                    ('episode_number', 'numerical', 'int64'),
                    ('duration_minutes', 'numerical', 'float64'),
                    ('release_year', 'numerical', 'int64'),
                    ('language', 'categorical', 'string'),
                    ('maturity', 'categorical', 'string'),
                    ('tags', 'multicategorical', 'string'),
                    ('is_original', 'categorical', 'bool'),
                ]),
                'viewers':
                table('viewers', 'viewer_id', 'signup_time', [
                    ('viewer_id', 'ID', 'int64'),
                    ('signup_time', 'timestamp', 'time'),
                    ('device_type', 'categorical', 'string'),
                    ('region', 'categorical', 'string'),
                    ('plan', 'categorical', 'string'),
                    ('age_band', 'categorical', 'string'),
                    ('autoplay_enabled', 'categorical', 'bool'),
                ]),
            }, edges=[
                Edge('series', 'genre_id', 'genres'),
                Edge('titles', 'series_id', 'series'),
                Edge('title_genres', 'title_id', 'titles'),
                Edge('title_genres', 'genre_id', 'genres'),
                Edge('sessions', 'viewer_id', 'viewers'),
                Edge('sessions', 'title_id', 'titles'),
                Edge('ratings', 'viewer_id', 'viewers'),
                Edge('ratings', 'title_id', 'titles'),
            ])
        graph.validate()
        query = kumo.PredictiveQuery(graph, (
            'PREDICT LIST_DISTINCT(sessions.title_id, 0, 7, days) RANK TOP 50 FOR EACH '
            'viewers.viewer_id '))
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
        run.record_table(training_table, unordered_targets=True)

        plan = query.suggest_model_plan(snapshot)
        plan.training_job.num_experiments = 2
        plan.training_job.pruning = None
        plan.training_job.manual_seed = 0
        plan.training_job.refit_trainval = False
        plan.training_job.refit_full = False
        plan.training_job.disable_explain = True
        plan.training_job.metrics = ['recall@50']
        plan.training_job.tune_metric = 'recall@50'
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
                prediction_table, snapshot=snapshot, num_classes_to_return=51,
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
                              top_k=50)


if __name__ == "__main__":
    main()
