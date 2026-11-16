# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""How much of the model's score is just "which capture is this?".

The `small` tier scores 0.9996 held-out AUROC on a target whose base rate is
90.5%, which is high enough to want an explanation before it is quoted.
``scenario_id`` is dropped from the graph, but a capture is one malware sample
run against one device, so its *hosts* are its own -- and both endpoint IPs are
edges into ``hosts``. The capture can therefore be recovered from graph
structure even though the column is gone.

This measures the ceiling that structure alone buys, without any model: score
every held-out flow by the malicious rate of its scenario, computed on the
*training* rows only, and take the AUROC of that. A model can beat this by
learning flow features; it cannot be credited for the part this already
explains.

Two further numbers put it in context: the majority-class accuracy, since 90.5%
positives makes accuracy meaningless on its own, and how far the scenario's
hosts are shared with any other scenario, since a host that appears in exactly
one capture is a direct label channel.
"""
import argparse

from snowflake.snowpark import Session

SCHEMA = 'HYU.IOT23'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default='small')
    parser.add_argument('--warehouse', default='KUMO_L')
    args = parser.parse_args()

    suffix = args.scale.upper()
    flows = f'{SCHEMA}.NETWORK_FLOWS_{suffix}'
    truth = f'{SCHEMA}.NETWORK_FLOWS_{suffix}_TRUTH'

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    def show(title: str, sql: str) -> None:
        print(f'\n--- {title} ---', flush=True)
        for row in session.sql(sql).collect():
            print(' ', row, flush=True)

    # `flow_id` is '<scenario_id>:<uid>', so the dropped column is recoverable
    # from the key. That is what makes this measurable at all.
    scenario = "SPLIT_PART(flow_id, ':', 1)"

    show(
        'held-out set: size and base rate', f"""
        SELECT COUNT(*) AS n,
               SUM(IFF(is_malicious, 1, 0)) AS malicious,
               AVG(IFF(is_malicious, 1, 0)) AS base_rate
        FROM {truth}
        """)

    # Scored by training-set scenario rate, so the baseline never sees a
    # held-out label. AUROC via the Mann-Whitney identity over midranks, which
    # is exact and needs no scikit-learn.
    show(
        'AUROC of "score = scenario malicious rate, fitted on train rows"',
        f"""
        WITH train_rate AS (
            SELECT {scenario} AS scenario,
                   AVG(IFF(is_malicious, 1, 0)) AS rate
            FROM {flows}
            WHERE is_malicious IS NOT NULL
            GROUP BY 1
        ), scored AS (
            SELECT t.is_malicious AS label,
                   COALESCE(r.rate, 0.5) AS score
            FROM {truth} t
            LEFT JOIN train_rate r ON r.scenario = {scenario}
        ), ranked AS (
            SELECT label, RANK() OVER (ORDER BY score) AS lo,
                   COUNT(*) OVER (PARTITION BY score) AS ties,
                   ROW_NUMBER() OVER (ORDER BY score) AS rn
            FROM scored
        )
        SELECT
            SUM(IFF(label, 1, 0)) AS n_pos,
            SUM(IFF(label, 0, 1)) AS n_neg,
            (SUM(IFF(label, rn, 0)) - SUM(IFF(label, 1, 0))
             * (SUM(IFF(label, 1, 0)) + 1) / 2)
            / (SUM(IFF(label, 1, 0)) * SUM(IFF(label, 0, 1))) AS auroc_approx
        FROM ranked
        """)

    show(
        'how many hosts appear in more than one scenario', f"""
        WITH endpoint AS (
            SELECT source_ip AS ip, {scenario} AS scenario FROM {flows}
            UNION
            SELECT destination_ip AS ip, {scenario} AS scenario FROM {flows}
        )
        SELECT COUNT(*) AS hosts,
               SUM(IFF(scenarios = 1, 1, 0)) AS in_one_scenario_only,
               AVG(scenarios) AS mean_scenarios
        FROM (SELECT ip, COUNT(DISTINCT scenario) AS scenarios
              FROM endpoint GROUP BY 1)
        """)


if __name__ == '__main__':
    main()
