# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Commerce static link through the installed Engine wheel."""

from common import parse_args, run, sources, table

SCALES = {"100M": 100, "1B": 300, "10B": 1600, "25B": 3200}
LIMITS = dict(prediction_rows=20, metric_tolerance=0.25,
              prediction_tolerance=0.005)
QUERY = ('PREDICT LIST_DISTINCT(purchases.merchant_id) RANK TOP 5 FOR EACH '
         'customers.customer_id ')
SPLIT = 'RandomLinkSplit([0.6, 0.2, 0.2])'


def build_graph(connector, temporal=False):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables={
            'customers':
            table(connector, 'customers', 'customer_id',
                  'signup_time' if temporal else None,
                  [('customer_id', 'ID', 'int64'),
                   ('signup_time', 'timestamp', 'time'),
                   ('region', 'categorical', 'string'),
                   ('city', 'categorical', 'string'),
                   ('membership', 'categorical', 'string'),
                   ('platform', 'categorical', 'string'),
                   ('marketing_opt_in', 'categorical', 'bool')]),
            'favorites':
            table(connector, 'favorites', None,
                  'created_time' if temporal else None,
                  [('created_time', 'timestamp', 'time'),
                   ('customer_id', 'ID', 'int64'),
                   ('merchant_id', 'ID', 'int64')]),
            'merchant_views':
            table(connector, 'merchant_views', None,
                  'view_time' if temporal else None,
                  [('view_time', 'timestamp', 'time'),
                   ('customer_id', 'ID', 'int64'),
                   ('merchant_id', 'ID', 'int64'),
                   ('source', 'categorical', 'string')]),
            'merchants':
            table(connector, 'merchants', 'merchant_id',
                  'created_time' if temporal else None,
                  [('merchant_id', 'ID', 'int64'),
                   ('created_time', 'timestamp', 'time'),
                   ('region', 'categorical', 'string'),
                   ('primary_category', 'categorical', 'string'),
                   ('tags', 'multicategorical', 'string'),
                   ('price_tier', 'categorical', 'string'),
                   ('is_partner', 'categorical', 'bool'),
                   ('avg_rating', 'numerical', 'float64')]),
            'purchases':
            table(connector, 'purchases', 'purchase_id',
                  'purchase_time' if temporal else None,
                  [('purchase_id', 'ID', 'int64'),
                   ('purchase_time', 'timestamp', 'time'),
                   ('customer_id', 'ID', 'int64'),
                   ('merchant_id', 'ID', 'int64'),
                   ('channel', 'categorical', 'string'),
                   ('fulfillment', 'categorical', 'string'),
                   ('item_count', 'numerical', 'int64'),
                   ('subtotal', 'numerical', 'float64'),
                   ('fees', 'numerical', 'float64'),
                   ('tip', 'numerical', 'float64'),
                   ('is_first_purchase', 'categorical', 'bool'),
                   ('rating', 'numerical', 'int64')]),
            'reviews':
            table(connector, 'reviews', None,
                  'review_time' if temporal else None,
                  [('review_time', 'timestamp', 'time'),
                   ('customer_id', 'ID', 'int64'),
                   ('merchant_id', 'ID', 'int64'),
                   ('stars', 'numerical', 'int64')]),
            'searches':
            table(connector, 'searches', None,
                  'search_time' if temporal else None,
                  [('search_time', 'timestamp', 'time'),
                   ('customer_id', 'ID', 'int64'),
                   ('query_term', 'categorical', 'string'),
                   ('category', 'categorical', 'string'),
                   ('result_page', 'numerical', 'int64')])
        }, edges=[
            Edge('purchases', 'customer_id', 'customers'),
            Edge('purchases', 'merchant_id', 'merchants'),
            Edge('merchant_views', 'customer_id', 'customers'),
            Edge('merchant_views', 'merchant_id', 'merchants'),
            Edge('searches', 'customer_id', 'customers'),
            Edge('favorites', 'customer_id', 'customers'),
            Edge('favorites', 'merchant_id', 'merchants'),
            Edge('reviews', 'customer_id', 'customers'),
            Edge('reviews', 'merchant_id', 'merchants')
        ])


def main():
    args = parse_args('commerce_static_link', SCALES, family='commerce',
                      **LIMITS)
    from kumoapi.model_plan import TrainingTableGenerationPlan

    run(
        build_graph(sources(args)), QUERY,
        TrainingTableGenerationPlan(split=SPLIT), 'static_link', {
            'training_job.metrics': ['recall@5'],
            'training_job.tune_metric': 'recall@5'
        }, args=args, prediction_kwargs={'num_classes_to_return': 5})


if __name__ == "__main__":
    main()
