# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from common import LIMITS, Scale, metric, parse_args, run, sources, table

SCALES = {
    "100M": Scale(100, "g7e.8xlarge"),
    "1B": Scale(300, "g7e.8xlarge"),
    "10B": Scale(1600, "g7e.12xlarge"),
    "25B": Scale(3200, "g7e.24xlarge")
}
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
        tables=[
            table(connector, 'genres', 'genre_id', None,
                  [('genre_id', 'ID', 'int64'),
                   ('genre_group', 'categorical', 'string'),
                   ('is_kids', 'categorical', 'bool')]),
            table(connector, 'ratings', None, 'rated_time',
                  [('rated_time', 'timestamp', 'time'),
                   ('viewer_id title_id', 'ID', 'int64'),
                   ('thumbs_up', 'categorical', 'bool')]),
            table(connector, 'series', 'series_id', 'created_time',
                  [('series_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('genre_id', 'ID', 'int64'),
                   ('language maturity format', 'categorical', 'string')]),
            table(connector, 'sessions', 'session_id', 'start_time',
                  [('session_id', 'ID', 'int64'),
                   ('start_time', 'timestamp', 'time'),
                   ('viewer_id title_id', 'ID', 'int64'),
                   ('watch_seconds', 'numerical', 'int64'),
                   ('completion_frac', 'numerical', 'float64'),
                   ('device_type entry_point', 'categorical', 'string'),
                   ('resumed', 'categorical', 'bool')]),
            table(connector, 'title_genres', None, None,
                  [('title_id genre_id', 'ID', 'int64'),
                   ('relevance', 'numerical', 'float64')]),
            table(connector, 'titles', 'title_id', 'added_time',
                  [('title_id', 'ID', 'int64'),
                   ('added_time', 'timestamp', 'time'),
                   ('series_id', 'ID', 'int64'),
                   ('season_number episode_number', 'numerical', 'int64'),
                   ('duration_minutes', 'numerical', 'float64'),
                   ('release_year', 'numerical', 'int64'),
                   ('language maturity', 'categorical', 'string'),
                   ('tags', 'multicategorical', 'string'),
                   ('is_original', 'categorical', 'bool')]),
            table(
                connector, 'viewers', 'viewer_id', 'signup_time',
                [('viewer_id', 'ID', 'int64'),
                 ('signup_time', 'timestamp', 'time'),
                 ('device_type region plan age_band', 'categorical', 'string'),
                 ('autoplay_enabled', 'categorical', 'bool')])
        ], edges=[
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

    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT,
                                    timeframe_step=7), 'temporal_link',
        metric('recall@50'), args=args,
        prediction_kwargs={'num_classes_to_return': 50})


if __name__ == "__main__":
    main()
