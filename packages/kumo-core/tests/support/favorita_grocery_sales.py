# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict

import pandas as pd
from kumoapi.encoder import Index, Numerical
from kumoapi.typing import Stype

from kumo_core.encoders.config import EncodersConfig, TargetEncoderConfig
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from tests.support.in_memory_graph import InMemoryGraphStore


def from_favorita_grocery_sales() -> tuple[
    InMemoryFeatureStore,
    InMemoryGraphStore,
    EncodersConfig,
]:
    t = time.perf_counter()
    root = ('s3://kumo-public-datasets/'
            'favorita-grocery-sales-forecasting/parquet')
    sales_df = pd.read_parquet(f'{root}/train.parquet')
    stores_df = pd.read_parquet(f'{root}/stores.parquet')
    items_df = pd.read_parquet(f'{root}/items.parquet')
    oil_df = pd.read_parquet(f'{root}/oil.parquet')
    transactions_df = pd.read_parquet(f'{root}/transactions.parquet')
    holidays_df = pd.read_parquet(f'{root}/holidays_events.parquet')

    # Create virtual composite (store_nbr, item_nbr) table:
    pair_df = sales_df[['store_nbr', 'item_nbr']].drop_duplicates()
    pair_df = pair_df.reset_index(drop=True)
    pair_df['pair_nbr'] = range(len(pair_df))
    sales_df = sales_df.merge(
        pair_df,
        on=['store_nbr', 'item_nbr'],
        how='left',
    )

    train_df = sales_df[['pair_nbr', 'date', 'unit_sales']]
    train_df['date'] -= pd.DateOffset(days=1)
    max_t = train_df['date'].max()
    train_df['test_mask'] = (  #
        train_df['date'] > max_t - pd.DateOffset(days=28))
    train_df['val_mask'] = (
        (train_df['date'] <= max_t - pd.DateOffset(days=28)) &
        (train_df['date'] > max_t - pd.DateOffset(days=2 * 28)))
    train_df['train_mask'] = (  #
        train_df['date'] <= max_t - pd.DateOffset(days=2 * 28))

    df_dict = {
        'pairs': pair_df,
        'sales': sales_df,
        'stores': stores_df,
        'items': items_df,
        'oil': oil_df,  # NOTE Not yet connected to any table.
        'transactions': transactions_df,
        'holidays': holidays_df,  # NOTE Not yet connected to any table.
        'train_table': train_df,
    }
    print(f'Downloaded raw tables in {time.perf_counter() - t:.2f} seconds')

    t = time.perf_counter()
    stypes: dict[str, dict[str, Stype]] = defaultdict(dict)
    stypes['pairs'] = {}
    stypes['sales']['date'] = Stype.timestamp
    stypes['sales']['unit_sales'] = Stype.numerical
    stypes['sales']['onpromotion'] = Stype.categorical
    stypes['stores']['city'] = Stype.categorical
    stypes['stores']['state'] = Stype.categorical
    stypes['stores']['type'] = Stype.categorical
    stypes['stores']['cluster'] = Stype.categorical
    stypes['items']['family'] = Stype.categorical
    stypes['items']['class'] = Stype.categorical
    stypes['items']['perishable'] = Stype.categorical
    stypes['oil']['date'] = Stype.timestamp
    stypes['oil']['dcoilwtico'] = Stype.numerical
    stypes['transactions']['date'] = Stype.timestamp
    stypes['transactions']['transactions'] = Stype.numerical
    stypes['holidays']['date'] = Stype.timestamp
    stypes['holidays']['type'] = Stype.categorical
    stypes['holidays']['locale'] = Stype.categorical
    stypes['holidays']['locale_name'] = Stype.categorical
    stypes['holidays']['transferred'] = Stype.categorical

    encoders = EncodersConfig.from_stypes(df_dict, stypes)
    encoders['items']['family'] = Index(min_occ=10)
    encoders['items']['class'] = Index(min_occ=10)
    encoders['holidays']['type'] = Index(min_occ=2)
    encoders['holidays']['locale_name'] = Index(min_occ=2)

    encoders.target = TargetEncoderConfig(
        'unit_sales',
        Numerical(scaler='standard'),
    )

    pkey_name_dict = {
        'pairs': 'pair_nbr',
        'stores': 'store_nbr',
        'items': 'item_nbr',
    }
    time_col_dict = {
        'sales': 'date',
        'oil': 'date',
        'transactions': 'date',
        'holidays': 'date',
    }
    fkeys = [
        ('sales.pair_nbr', 'pairs.pair_nbr'),
        ('sales.store_nbr', 'stores.store_nbr'),
        ('sales.item_nbr', 'items.item_nbr'),
        ('transactions.store_nbr', 'stores.store_nbr'),
        ('pairs.store_nbr', 'stores.store_nbr'),
        ('pairs.item_nbr', 'items.item_nbr'),
    ]
    print(f'Inferred metadata in {time.perf_counter() - t :.2f} seconds')

    t = time.perf_counter()
    feature_store = InMemoryFeatureStore.from_encoders(
        df_dict=df_dict,
        encoders=encoders,
        pkey_name_dict=pkey_name_dict,
        time_col_dict=time_col_dict | {'train_table': 'date'},
        end_time_col_dict={},
        train_table_name='train_table',
        train_fkeys=[('pair_nbr', 'pairs')],
        train_mask_names=('train_mask', 'val_mask', 'test_mask'),
    )
    print(f'Created feature store in {time.perf_counter() - t:.2f} seconds')

    t = time.perf_counter()
    graph_store = InMemoryGraphStore(
        df_dict=df_dict,
        fkeys=fkeys,
        time_col_dict=time_col_dict,
    )
    print(f'Created graph store in {time.perf_counter() - t:.2f} seconds')

    return feature_store, graph_store, encoders
