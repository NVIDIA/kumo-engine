# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from common import LIMITS, Scale, metric, parse_args, run, sources
from delivery_graph import build_graph

SCALES = {
    "100M": Scale(100, "g7e.8xlarge"),
    "1B": Scale(300, "g7e.8xlarge"),
    "10B": Scale(1600, "g7e.12xlarge"),
    "25B": Scale(3200, "g7e.24xlarge")
}
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
        'forecasting', metric('mae'), args=args)


if __name__ == "__main__":
    main()
