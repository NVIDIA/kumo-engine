# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from common import LIMITS, metric, parse_args, run, sources
from delivery_graph import build_graph

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
QUERY = ('PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE '
         'COUNT(orders.*, -90, 0, days) > 0 ')
SPLIT = ("TimeRangeSplit([('2025-10-02T00:00:00', '2025-11-01T00:00:00'), "
         "('2025-11-01T00:00:00', '2025-12-01T00:00:00'), "
         "('2025-12-01T00:00:00', '2025-12-31T00:00:00')], timeframe=30, "
         'flexible_boundaries=False)')


def main():
    args = parse_args('delivery_regression', SCALES, family='delivery',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=30),
        'regression', metric('mae'), args=args)


if __name__ == "__main__":
    main()
