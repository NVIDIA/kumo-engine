# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming temporal link through the installed Engine wheel."""

from common import parse_args, run, sources, table

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
LIMITS = dict(prediction_rows=20, metric_tolerance=0.25,
              prediction_tolerance=0.005)
QUERY = ('PREDICT LIST_DISTINCT(sessions.title_id, 0, 7, days) RANK TOP 50 '
         'FOR EACH viewers.viewer_id ')
SPLIT = ("TimeRangeSplit([('2025-12-10T00:00:00', '2025-12-17T00:00:00'), "
         "('2025-12-17T00:00:00', '2025-12-24T00:00:00'), "
         "('2025-12-24T00:00:00', '2025-12-31T00:00:00')], timeframe=7, "
         'flexible_boundaries=False)')


def build_graph(connector):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables={
            'genres':
            table(connector, 'genres', 'genre_id', None,
                  [('genre_id', 'ID', 'int64'),
                   ('genre_group', 'categorical', 'string'),
                   ('is_kids', 'categorical', 'bool')]),
            'ratings':
            table(connector, 'ratings', None, 'rated_time',
                  [('rated_time', 'timestamp', 'time'),
                   ('viewer_id', 'ID', 'int64'), ('title_id', 'ID', 'int64'),
                   ('thumbs_up', 'categorical', 'bool')]),
            'series':
            table(connector, 'series', 'series_id', 'created_time',
                  [('series_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('genre_id', 'ID', 'int64'),
                   ('language', 'categorical', 'string'),
                   ('maturity', 'categorical', 'string'),
                   ('format', 'categorical', 'string')]),
            'sessions':
            table(connector, 'sessions', 'session_id', 'start_time',
                  [('session_id', 'ID', 'int64'),
                   ('start_time', 'timestamp', 'time'),
                   ('viewer_id', 'ID', 'int64'), ('title_id', 'ID', 'int64'),
                   ('watch_seconds', 'numerical', 'int64'),
                   ('completion_frac', 'numerical', 'float64'),
                   ('device_type', 'categorical', 'string'),
                   ('entry_point', 'categorical', 'string'),
                   ('resumed', 'categorical', 'bool')]),
            'title_genres':
            table(connector, 'title_genres', None, None,
                  [('title_id', 'ID', 'int64'), ('genre_id', 'ID', 'int64'),
                   ('relevance', 'numerical', 'float64')]),
            'titles':
            table(connector, 'titles', 'title_id', 'added_time',
                  [('title_id', 'ID', 'int64'),
                   ('added_time', 'timestamp', 'time'),
                   ('series_id', 'ID', 'int64'),
                   ('season_number', 'numerical', 'int64'),
                   ('episode_number', 'numerical', 'int64'),
                   ('duration_minutes', 'numerical', 'float64'),
                   ('release_year', 'numerical', 'int64'),
                   ('language', 'categorical', 'string'),
                   ('maturity', 'categorical', 'string'),
                   ('tags', 'multicategorical', 'string'),
                   ('is_original', 'categorical', 'bool')]),
            'viewers':
            table(connector, 'viewers', 'viewer_id', 'signup_time',
                  [('viewer_id', 'ID', 'int64'),
                   ('signup_time', 'timestamp', 'time'),
                   ('device_type', 'categorical', 'string'),
                   ('region', 'categorical', 'string'),
                   ('plan', 'categorical', 'string'),
                   ('age_band', 'categorical', 'string'),
                   ('autoplay_enabled', 'categorical', 'bool')])
        }, edges=[
            Edge('series', 'genre_id', 'genres'),
            Edge('titles', 'series_id', 'series'),
            Edge('title_genres', 'title_id', 'titles'),
            Edge('title_genres', 'genre_id', 'genres'),
            Edge('sessions', 'viewer_id', 'viewers'),
            Edge('sessions', 'title_id', 'titles'),
            Edge('ratings', 'viewer_id', 'viewers'),
            Edge('ratings', 'title_id', 'titles')
        ])


def main():
    args = parse_args('streaming_temporal_link', SCALES, family='streaming',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(
        build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=7),
        'temporal_link', {
            'training_job.metrics': ['recall@50'],
            'training_job.tune_metric': 'recall@50'
        }, args=args, prediction_kwargs={'num_classes_to_return': 50})


if __name__ == "__main__":
    main()
