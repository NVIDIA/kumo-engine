# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from commerce_static_link import build_graph as commerce_graph
from common import LIMITS, Scale, metric, parse_args, run, sources

SCALES = {
    "100M": Scale(100, "g7e.8xlarge"),
    "1B": Scale(300, "g7e.8xlarge"),
    "10B": Scale(1600, "g7e.12xlarge"),
    "25B": Scale(3200, "g7e.24xlarge")
}
QUERY = ('PREDICT LIST_DISTINCT(purchases.merchant_id, 0, 7, days) RANK TOP '
         '10 FOR EACH customers.customer_id ')
SPLIT = ("TimeRangeSplit([('2024-12-05T00:00:00', '2024-12-12T00:00:00'), "
         "('2024-12-12T00:00:00', '2024-12-19T00:00:00'), "
         "('2024-12-19T00:00:00', '2024-12-26T00:00:00')], timeframe=7, "
         'flexible_boundaries=False)')


def build_graph(connector):
    return commerce_graph(connector, temporal=True)


def main():
    args = parse_args('commerce_temporal_link', SCALES, family='commerce',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT,
                                    timeframe_step=7), 'temporal_link',
        metric('recall@10'), args=args,
        prediction_kwargs={'num_classes_to_return': 10})


if __name__ == "__main__":
    main()
