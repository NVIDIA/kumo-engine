# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from common import LIMITS, parse_args, run, sources, sparse_overrides, table

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
QUERY = ('PREDICT COUNT(account_flags.*, 0, 28, days) > 0 FOR EACH '
         'accounts.account_id WHERE COUNT(account_flags.*, -INF, 0, days) = 0 '
         'AND COUNT(logins.*, -30, 0, days) > 0 ')
SPLIT = ("TimeRangeSplit([('2025-05-21T00:00:00', '2025-06-18T00:00:00'), "
         "('2025-06-18T00:00:00', '2025-07-16T00:00:00'), "
         "('2025-07-16T00:00:00', '2025-08-13T00:00:00')], timeframe=28, "
         'flexible_boundaries=False)')


def build_graph(connector):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables=[
            table(connector, 'account_flags', None, 'flag_time',
                  [('flag_time', 'timestamp', 'time'),
                   ('account_id', 'ID', 'int64')]),
            table(connector, 'accounts', 'account_id', 'created_time',
                  [('account_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('email_domain country account_type loyalty_tier',
                    'categorical', 'string'),
                   ('is_verified', 'categorical', 'bool')]),
            table(connector, 'devices', 'device_id', 'first_seen_time',
                  [('device_id', 'ID', 'int64'),
                   ('first_seen_time', 'timestamp', 'time'),
                   ('os browser device_type language timezone', 'categorical',
                    'string')]),
            table(connector, 'ip_addresses', 'ip_id', None,
                  [('ip_id', 'ID', 'int64'),
                   ('isp country region', 'categorical', 'string'),
                   ('is_proxy is_datacenter', 'categorical', 'bool')]),
            table(connector, 'logins', 'login_id', 'login_time',
                  [('login_id', 'ID', 'int64'),
                   ('login_time', 'timestamp', 'time'),
                   ('account_id device_id ip_id', 'ID', 'int64')]),
            table(connector, 'merchants', 'merchant_id', None,
                  [('merchant_id', 'ID', 'int64'),
                   ('category country city channel', 'categorical', 'string')]),
            table(connector, 'payment_methods', 'payment_method_id', None,
                  [('payment_method_id', 'ID', 'int64'),
                   ('card_network card_type issuing_country', 'categorical',
                    'string'), ('payer_name', 'multicategorical', 'string')]),
            table(
                connector, 'transactions', 'transaction_id', 'transaction_time',
                [('transaction_id', 'ID', 'int64'),
                 ('transaction_time', 'timestamp', 'time'),
                 ('account_id payment_method_id merchant_id', 'ID', 'int64'),
                 ('amount_usd', 'numerical', 'float64'),
                 ('quantity', 'numerical', 'int64'),
                 ('currency channel billing_country', 'categorical', 'string'),
                 ('billing_postal_codes risk_rule_hits', 'multicategorical',
                  'string'), ('email_matches_payer', 'categorical', 'bool'),
                 ('installments', 'numerical', 'int64'),
                 ('traveler_name', 'categorical', 'string')]),
            table(connector, 'transfers', 'transfer_id', 'transfer_time',
                  [('transfer_id', 'ID', 'int64'),
                   ('transfer_time', 'timestamp', 'time'),
                   ('account_id payee_id login_id', 'ID', 'int64'),
                   ('amount_usd', 'numerical', 'float64')])
        ], edges=[
            Edge('logins', 'account_id', 'accounts'),
            Edge('logins', 'device_id', 'devices'),
            Edge('logins', 'ip_id', 'ip_addresses'),
            Edge('transfers', 'account_id', 'accounts'),
            Edge('transfers', 'payee_id', 'accounts'),
            Edge('transfers', 'login_id', 'logins'),
            Edge('transactions', 'account_id', 'accounts'),
            Edge('transactions', 'payment_method_id', 'payment_methods'),
            Edge('transactions', 'merchant_id', 'merchants'),
            Edge('account_flags', 'account_id', 'accounts')
        ])


def main():
    args = parse_args('fraud_binary', SCALES, family='fraud', **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    overrides = sparse_overrides(args.scale)
    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=28), 'binary',
        overrides, args=args)


if __name__ == "__main__":
    main()
