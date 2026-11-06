# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from common import LIMITS, metric, parse_args, run, sources, table

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
QUERY = ('PREDICT COUNT(conversions.*, 0, 7, days) > 0 FOR EACH users.user_id '
         'WHERE COUNT(impressions.*, -14, 0, days) > 0 ')
SPLIT = ("TimeRangeSplit([('2025-12-10T00:00:00', '2025-12-17T00:00:00'), "
         "('2025-12-17T00:00:00', '2025-12-24T00:00:00'), "
         "('2025-12-24T00:00:00', '2025-12-31T00:00:00')], timeframe=7, "
         'flexible_boundaries=False)')


def build_graph(connector):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables=[
            table(connector, 'ads', 'ad_id', 'created_time',
                  [('ad_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('campaign_id', 'ID', 'int64'),
                   ('format creative_size call_to_action', 'categorical',
                    'string'), ('segments', 'multicategorical', 'string'),
                   ('quality_score', 'numerical', 'float64')]),
            table(connector, 'advertisers', 'advertiser_id', 'created_time',
                  [('advertiser_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('vertical country account_tier', 'categorical', 'string'),
                   ('is_verified', 'categorical', 'bool')]),
            table(connector, 'campaigns', 'campaign_id', 'start_time', [
                ('campaign_id', 'ID', 'int64'),
                ('start_time', 'timestamp', 'time'),
                ('advertiser_id', 'ID', 'int64'),
                ('objective budget_tier bid_strategy', 'categorical', 'string'),
                ('daily_budget', 'numerical', 'float64')
            ]),
            table(connector, 'clicks', 'click_id', 'click_time',
                  [('click_id', 'ID', 'int64'),
                   ('click_time', 'timestamp', 'time'),
                   ('user_id ad_id', 'ID', 'int64'),
                   ('dwell_seconds', 'numerical', 'float64')]),
            table(connector, 'conversions', 'conversion_id', 'conversion_time',
                  [('conversion_id', 'ID', 'int64'),
                   ('conversion_time', 'timestamp', 'time'),
                   ('user_id ad_id', 'ID', 'int64'),
                   ('conversion_type', 'categorical', 'string'),
                   ('value', 'numerical', 'float64')]),
            table(connector, 'impressions', None, 'impression_time',
                  [('impression_time', 'timestamp', 'time'),
                   ('user_id ad_id', 'ID', 'int64'),
                   ('placement', 'categorical', 'string'),
                   ('position', 'numerical', 'int64'),
                   ('viewability', 'numerical', 'float64')]),
            table(connector, 'interests', 'interest_id', None,
                  [('interest_id', 'ID', 'int64'),
                   ('topic', 'categorical', 'string'),
                   ('audience_size', 'numerical', 'int64')]),
            table(connector, 'user_interests', None, 'created_time',
                  [('created_time', 'timestamp', 'time'),
                   ('user_id interest_id', 'ID', 'int64'),
                   ('affinity', 'numerical', 'float64')]),
            table(connector, 'users', 'user_id', 'signup_time',
                  [('user_id', 'ID', 'int64'),
                   ('signup_time', 'timestamp', 'time'),
                   ('geo os age_bucket language', 'categorical', 'string'),
                   ('opted_in', 'categorical', 'bool')])
        ], edges=[
            Edge('user_interests', 'user_id', 'users'),
            Edge('user_interests', 'interest_id', 'interests'),
            Edge('campaigns', 'advertiser_id', 'advertisers'),
            Edge('ads', 'campaign_id', 'campaigns'),
            Edge('impressions', 'user_id', 'users'),
            Edge('impressions', 'ad_id', 'ads'),
            Edge('clicks', 'user_id', 'users'),
            Edge('clicks', 'ad_id', 'ads'),
            Edge('conversions', 'user_id', 'users'),
            Edge('conversions', 'ad_id', 'ads')
        ])


def main():
    args = parse_args('adtech_binary', SCALES, family='adtech', **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=7), 'binary',
        metric('auroc'), args=args)


if __name__ == "__main__":
    main()
