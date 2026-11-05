# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Payments binary through the installed Engine wheel."""

from common import parse_args, run, sources, table

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
LIMITS = dict(prediction_rows=20, metric_tolerance=0.25,
              prediction_tolerance=0.005)
QUERY = ('PREDICT COUNT(account_flags.*, 0, 14, days) > 0 FOR EACH '
         'accounts.account_id WHERE COUNT(account_flags.*, -INF, 0, days) = 0 '
         '')
SPLIT = ("TimeRangeSplit([('2024-02-18T00:00:00', '2024-03-03T00:00:00'), "
         "('2024-03-03T00:00:00', '2024-03-17T00:00:00'), "
         "('2024-03-17T00:00:00', '2024-03-31T00:00:00')], timeframe=14, "
         'flexible_boundaries=False)')


def build_graph(connector):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables={
            'account_events':
            table(connector, 'account_events', None, 'event_time',
                  [('event_time', 'timestamp', 'time'),
                   ('account_id', 'ID', 'int64'),
                   ('event_type', 'categorical', 'string'),
                   ('channel', 'categorical', 'string')]),
            'account_flags':
            table(connector, 'account_flags', None, 'flag_time',
                  [('flag_time', 'timestamp', 'time'),
                   ('account_id', 'ID', 'int64'),
                   ('source', 'categorical', 'string')]),
            'accounts':
            table(connector, 'accounts', 'account_id', 'created_time',
                  [('account_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('institution_id', 'ID', 'int64'),
                   ('account_kind', 'categorical', 'string'),
                   ('segment', 'categorical', 'string'),
                   ('holder_age_band', 'categorical', 'string'),
                   ('merchant_category', 'categorical', 'string'),
                   ('region', 'categorical', 'string'),
                   ('onboarding_channel', 'categorical', 'string'),
                   ('stated_monthly_volume', 'numerical', 'float64')]),
            'beneficiary_registrations':
            table(connector, 'beneficiary_registrations', None,
                  'registration_time',
                  [('registration_time', 'timestamp', 'time'),
                   ('registrant_id', 'ID', 'int64'),
                   ('beneficiary_id', 'ID', 'int64'),
                   ('channel', 'categorical', 'string')]),
            'institutions':
            table(connector, 'institutions', 'institution_id', None,
                  [('institution_id', 'ID', 'int64'),
                   ('institution_type', 'categorical', 'string'),
                   ('region', 'categorical', 'string'),
                   ('size_tier', 'categorical', 'string'),
                   ('is_digital', 'categorical', 'bool')]),
            'pair_interactions':
            table(connector, 'pair_interactions', None, 'snapshot_time',
                  [('snapshot_time', 'timestamp', 'time'),
                   ('payer_id', 'ID', 'int64'), ('payee_id', 'ID', 'int64'),
                   ('first_txn_time', 'timestamp', 'time'),
                   ('last_txn_time', 'timestamp', 'time'),
                   ('total_txns', 'numerical', 'int64'),
                   ('total_failed', 'numerical', 'int64'),
                   ('avg_amount', 'numerical', 'float64'),
                   ('period_txns', 'numerical', 'int64'),
                   ('period_median_amount', 'numerical', 'float64'),
                   ('period_max_amount', 'numerical', 'float64'),
                   ('period_active_days', 'numerical', 'int64')]),
            'transactions':
            table(connector, 'transactions', 'transaction_id',
                  'transaction_time',
                  [('transaction_id', 'ID', 'int64'),
                   ('transaction_time', 'timestamp', 'time'),
                   ('payer_id', 'ID', 'int64'), ('payee_id', 'ID', 'int64'),
                   ('amount', 'numerical', 'float64'),
                   ('status', 'categorical', 'string'),
                   ('channel', 'categorical', 'string'),
                   ('purpose', 'categorical', 'string')])
        }, edges=[
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
    from kumoapi.model_plan import (
        NumNeighborsConfig,
        TrainingTableGenerationPlan,
    )

    run(
        build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=14), 'binary', {
            'training_job.metrics': ['auroc'],
            'training_job.tune_metric':
            'auroc',
            'neighbor_sampling.num_neighbors': [
                NumNeighborsConfig(hop1={'default': 8}, hop2={'default': 8},
                                   hop3={'default': 8}, hop4={'default': 8},
                                   hop5={'default': 8}, hop6={'default': 8})
            ]
        }, args=args)


if __name__ == "__main__":
    main()
