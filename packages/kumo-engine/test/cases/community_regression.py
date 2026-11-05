# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Community regression through the installed Engine wheel."""

from common import parse_args, run, sources, table

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
LIMITS = dict(prediction_rows=20, metric_tolerance=0.25,
              prediction_tolerance=0.005)
QUERY = ('PREDICT COUNT(messages.*, 0, 30, days) FOR EACH accounts.account_id '
         'WHERE COUNT(logins.*, -30, 0, days) > 0 ')
SPLIT = ("TimeRangeSplit([('2025-10-02T00:00:00', '2025-11-01T00:00:00'), "
         "('2025-11-01T00:00:00', '2025-12-01T00:00:00'), "
         "('2025-12-01T00:00:00', '2025-12-31T00:00:00')], timeframe=30, "
         'flexible_boundaries=False)')


def build_graph(connector):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables={
            'accounts':
            table(connector, 'accounts', 'account_id', 'created_time',
                  [('account_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('email_domain_id', 'ID', 'int64'),
                   ('locale', 'categorical', 'string'),
                   ('client', 'categorical', 'string'),
                   ('age_bucket', 'categorical', 'string'),
                   ('interests', 'multicategorical', 'string'),
                   ('is_verified', 'categorical', 'bool')]),
            'communities':
            table(connector, 'communities', 'community_id', 'created_time',
                  [('community_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('category', 'categorical', 'string'),
                   ('size_tier', 'categorical', 'string'),
                   ('language', 'categorical', 'string'),
                   ('tags', 'multicategorical', 'string'),
                   ('is_public', 'categorical', 'bool'),
                   ('rules_count', 'numerical', 'int64')]),
            'devices':
            table(connector, 'devices', 'device_id', 'first_seen_time',
                  [('device_id', 'ID', 'int64'),
                   ('first_seen_time', 'timestamp', 'time'),
                   ('os', 'categorical', 'string'),
                   ('device_class', 'categorical', 'string'),
                   ('model', 'categorical', 'string')]),
            'email_domains':
            table(connector, 'email_domains', 'email_domain_id', None,
                  [('email_domain_id', 'ID', 'int64'),
                   ('tld', 'categorical', 'string'),
                   ('is_free_provider', 'categorical', 'bool'),
                   ('domain_age_years', 'numerical', 'float64')]),
            'logins':
            table(connector, 'logins', None, 'login_time',
                  [('login_time', 'timestamp', 'time'),
                   ('account_id', 'ID', 'int64'), ('network_id', 'ID', 'int64'),
                   ('device_id', 'ID', 'int64'),
                   ('platform', 'categorical', 'string'),
                   ('app_version', 'categorical', 'string'),
                   ('session_minutes', 'numerical', 'float64')]),
            'memberships':
            table(connector, 'memberships', None,
                  'joined_time', [('joined_time', 'timestamp', 'time'),
                                  ('account_id', 'ID', 'int64'),
                                  ('community_id', 'ID', 'int64'),
                                  ('role', 'categorical', 'string'),
                                  ('invite_source', 'categorical', 'string')]),
            'messages':
            table(connector, 'messages', 'message_id',
                  'sent_time', [('message_id', 'ID', 'int64'),
                                ('sent_time', 'timestamp', 'time'),
                                ('account_id', 'ID', 'int64'),
                                ('community_id', 'ID', 'int64'),
                                ('length_bucket', 'categorical', 'string'),
                                ('is_reply', 'categorical', 'bool'),
                                ('has_attachment', 'categorical', 'bool'),
                                ('mention_count', 'numerical', 'int64')]),
            'networks':
            table(connector, 'networks', 'network_id', None,
                  [('network_id', 'ID', 'int64'),
                   ('network_type', 'categorical', 'string'),
                   ('region', 'categorical', 'string'),
                   ('asn', 'categorical', 'string'),
                   ('is_hosting', 'categorical', 'bool')]),
            'reactions':
            table(connector, 'reactions', None,
                  'reaction_time', [('reaction_time', 'timestamp', 'time'),
                                    ('account_id', 'ID', 'int64'),
                                    ('community_id', 'ID', 'int64'),
                                    ('kind', 'categorical', 'string')])
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
            Edge('logins', 'device_id', 'devices')
        ])


def main():
    args = parse_args('community_regression', SCALES, family='community',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=30),
        'regression', {
            'training_job.metrics': ['mae'],
            'training_job.tune_metric': 'mae'
        }, args=args)


if __name__ == "__main__":
    main()
