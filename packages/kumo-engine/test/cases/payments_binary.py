# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from common import LIMITS, parse_args, run, sources, sparse_overrides, table

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
QUERY = ('PREDICT COUNT(account_flags.*, 0, 14, days) > 0 FOR EACH '
         'accounts.account_id WHERE COUNT(account_flags.*, -INF, 0, days) = 0 ')
SPLIT = ("TimeRangeSplit([('2024-02-18T00:00:00', '2024-03-03T00:00:00'), "
         "('2024-03-03T00:00:00', '2024-03-17T00:00:00'), "
         "('2024-03-17T00:00:00', '2024-03-31T00:00:00')], timeframe=14, "
         'flexible_boundaries=False)')


def build_graph(connector):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables=[
            table(connector, 'account_events', None, 'event_time',
                  [('event_time', 'timestamp', 'time'),
                   ('account_id', 'ID', 'int64'),
                   ('event_type channel', 'categorical', 'string')]),
            table(connector, 'account_flags', None, 'flag_time',
                  [('flag_time', 'timestamp', 'time'),
                   ('account_id', 'ID', 'int64'),
                   ('source', 'categorical', 'string')]),
            table(connector, 'accounts', 'account_id', 'created_time', [
                ('account_id', 'ID', 'int64'),
                ('created_time', 'timestamp', 'time'),
                ('institution_id', 'ID', 'int64'),
                ('account_kind segment holder_age_band merchant_category region onboarding_channel',
                 'categorical', 'string'),
                ('stated_monthly_volume', 'numerical', 'float64')
            ]),
            table(connector, 'beneficiary_registrations', None,
                  'registration_time',
                  [('registration_time', 'timestamp', 'time'),
                   ('registrant_id beneficiary_id', 'ID', 'int64'),
                   ('channel', 'categorical', 'string')]),
            table(
                connector, 'institutions', 'institution_id', None,
                [('institution_id', 'ID', 'int64'),
                 ('institution_type region size_tier', 'categorical', 'string'),
                 ('is_digital', 'categorical', 'bool')]),
            table(connector, 'pair_interactions', None, 'snapshot_time',
                  [('snapshot_time', 'timestamp', 'time'),
                   ('payer_id payee_id', 'ID', 'int64'),
                   ('first_txn_time last_txn_time', 'timestamp', 'time'),
                   ('total_txns total_failed', 'numerical', 'int64'),
                   ('avg_amount', 'numerical', 'float64'),
                   ('period_txns', 'numerical', 'int64'),
                   ('period_median_amount period_max_amount', 'numerical',
                    'float64'), ('period_active_days', 'numerical', 'int64')]),
            table(connector, 'transactions', 'transaction_id',
                  'transaction_time',
                  [('transaction_id', 'ID', 'int64'),
                   ('transaction_time', 'timestamp', 'time'),
                   ('payer_id payee_id', 'ID', 'int64'),
                   ('amount', 'numerical', 'float64'),
                   ('status channel purpose', 'categorical', 'string')])
        ], edges=[
            Edge('accounts', 'institution_id', 'institutions'),
            Edge('transactions', 'payer_id', 'accounts'),
            Edge('transactions', 'payee_id', 'accounts'),
            Edge('pair_interactions', 'payer_id', 'accounts'),
            Edge('pair_interactions', 'payee_id', 'accounts'),
            Edge('beneficiary_registrations', 'registrant_id', 'accounts'),
            Edge('beneficiary_registrations', 'beneficiary_id', 'accounts'),
            Edge('account_events', 'account_id', 'accounts'),
            Edge('account_flags', 'account_id', 'accounts')
        ])


def main():
    args = parse_args('payments_binary', SCALES, family='payments', **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    overrides = sparse_overrides(args.scale)
    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=14), 'binary',
        overrides, args=args)


if __name__ == "__main__":
    main()
