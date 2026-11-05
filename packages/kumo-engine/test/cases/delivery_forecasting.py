# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delivery forecasting through the installed Engine wheel."""

from common import parse_args, run, sources
from delivery_graph import build_graph

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
LIMITS = dict(prediction_rows=20, metric_tolerance=0.25,
              prediction_tolerance=0.005)
QUERY = ('PREDICT COUNT(orders.*, 0, 7, days) FORECAST 12 TIMEFRAMES FOR EACH '
         'users.user_id ')
SPLIT = ("TimeRangeSplit([('2025-04-23T00:00:00', '2025-07-16T00:00:00'), "
         "('2025-07-16T00:00:00', '2025-10-08T00:00:00'), "
         "('2025-10-08T00:00:00', '2025-12-31T00:00:00')], timeframe=84, "
         'flexible_boundaries=False)')


def main():
    args = parse_args('delivery_forecasting', SCALES, family='delivery',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(
        build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=7,
                                    lag_timesteps=12, year_over_year=False),
        'forecasting', {
            'training_job.metrics': ['mae'],
            'training_job.tune_metric': 'mae'
        }, args=args)


if __name__ == "__main__":
    main()
