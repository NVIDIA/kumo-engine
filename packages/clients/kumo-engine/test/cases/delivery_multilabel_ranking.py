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
QUERY = ('PREDICT LIST_DISTINCT(orders.daypart, 0, 14, days) RANK TOP 5 FOR '
         'EACH users.user_id WHERE COUNT(orders.*, -90, 0, days) > 0 ')
SPLIT = ("TimeRangeSplit([('2025-11-19T00:00:00', '2025-12-03T00:00:00'), "
         "('2025-12-03T00:00:00', '2025-12-17T00:00:00'), "
         "('2025-12-17T00:00:00', '2025-12-31T00:00:00')], timeframe=14, "
         'flexible_boundaries=False)')


def main():
    args = parse_args('delivery_multilabel_ranking', SCALES, family='delivery',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT, timeframe_step=14),
        'multilabel_ranking', metric('recall@5'), args=args)


if __name__ == "__main__":
    main()
