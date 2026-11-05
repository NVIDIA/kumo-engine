# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The delivery graph shared by its five task cases."""

from common import table


def build_graph(connector):
    from kumoapi.graph import Edge

    import kumo.engine as kumo

    return kumo.Graph(
        tables={
            'order_items':
            table(connector, 'order_items', 'order_item_id', 'item_time',
                  [('order_item_id', 'ID', 'int64'),
                   ('item_time', 'timestamp', 'time'),
                   ('order_id', 'ID', 'int64'),
                   ('item_category', 'categorical', 'string'),
                   ('quantity', 'numerical', 'int64'),
                   ('unit_price', 'numerical', 'float64'),
                   ('modifiers', 'multicategorical', 'string')]),
            'orders':
            table(connector, 'orders', 'order_id', 'order_time',
                  [('order_id', 'ID', 'int64'),
                   ('order_time', 'timestamp', 'time'),
                   ('user_id', 'ID', 'int64'), ('store_id', 'ID', 'int64'),
                   ('daypart', 'categorical', 'string'),
                   ('item_count', 'numerical', 'int64'),
                   ('subtotal', 'numerical', 'float64'),
                   ('tip', 'numerical', 'float64'),
                   ('delivery_minutes', 'numerical', 'int64'),
                   ('promo_applied', 'categorical', 'bool')]),
            'regions':
            table(connector, 'regions', 'region_id', None,
                  [('region_id', 'ID', 'int64'),
                   ('density_tier', 'categorical', 'string'),
                   ('climate_zone', 'categorical', 'string'),
                   ('base_delivery_fee', 'numerical', 'float64')]),
            'store_groups':
            table(connector, 'store_groups', 'store_group_id', None,
                  [('store_group_id', 'ID', 'int64'),
                   ('region_id', 'ID', 'int64'),
                   ('cuisine', 'categorical', 'string'),
                   ('price_tier', 'categorical', 'string'),
                   ('is_franchise', 'categorical', 'bool')]),
            'store_impressions':
            table(connector, 'store_impressions', None, 'impression_time',
                  [('impression_time', 'timestamp', 'time'),
                   ('user_id', 'ID', 'int64'), ('store_id', 'ID', 'int64'),
                   ('position', 'numerical', 'int64'),
                   ('clicked', 'categorical', 'bool'),
                   ('surface', 'categorical', 'string')]),
            'stores':
            table(connector, 'stores', 'store_id', 'opened_time',
                  [('store_id', 'ID', 'int64'),
                   ('opened_time', 'timestamp', 'time'),
                   ('store_group_id', 'ID', 'int64'),
                   ('rating', 'numerical', 'float64'),
                   ('delivery_radius_km', 'numerical', 'float64'),
                   ('prep_minutes', 'numerical', 'int64'),
                   ('cuisine_tags', 'multicategorical', 'string')]),
            'users':
            table(connector, 'users', 'user_id', 'signup_time',
                  [('user_id', 'ID', 'int64'),
                   ('signup_time', 'timestamp', 'time'),
                   ('home_zone', 'categorical', 'string'),
                   ('platform', 'categorical', 'string'),
                   ('membership_tier', 'categorical', 'string'),
                   ('age_band', 'categorical', 'string'),
                   ('marketing_opt_in', 'categorical', 'bool')])
        }, edges=[
            Edge('store_groups', 'region_id', 'regions'),
            Edge('stores', 'store_group_id', 'store_groups'),
            Edge('orders', 'user_id', 'users'),
            Edge('orders', 'store_id', 'stores'),
            Edge('order_items', 'order_id', 'orders'),
            Edge('store_impressions', 'user_id', 'users'),
            Edge('store_impressions', 'store_id', 'stores')
        ])
