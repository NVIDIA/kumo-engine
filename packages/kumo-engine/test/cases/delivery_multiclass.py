# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delivery multiclass through the installed Engine wheel."""

from common import parse_args, run, sources
from delivery_graph import build_graph

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
LIMITS = dict(prediction_rows=20, metric_tolerance=0.25,
              prediction_tolerance=0.005)
QUERY = ('PREDICT FIRST(orders.daypart, 0, 14, days) FOR EACH users.user_id '
         'WHERE COUNT(orders.*, -90, 0, days) > 0 ')
SPLIT = ("TimeRangeSplit([('2025-11-19T00:00:00', '2025-12-03T00:00:00'), "
         "('2025-12-03T00:00:00', '2025-12-17T00:00:00'), "
         "('2025-12-17T00:00:00', '2025-12-31T00:00:00')], timeframe=14, "
         'flexible_boundaries=False)')


def main():
    args = parse_args('delivery_multiclass', SCALES, family='delivery',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=14),
        'multiclass', {
            'training_job.metrics': ['acc'],
            'training_job.tune_metric': 'acc'
        }, args=args)


if __name__ == "__main__":
    main()
