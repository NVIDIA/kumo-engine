# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

import pandas as pd
from kumoapi.typing import Stype

from kumo_core.encoders.config import EncodersConfig
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from tests.support.in_memory_graph import InMemoryGraphStore


def from_salt() -> tuple[
    InMemoryFeatureStore,
    InMemoryGraphStore,
    EncodersConfig,
]:
    root = 's3://kumo-public-datasets/SALT'
    doc_df = pd.read_parquet(f'{root}/salesdocuments.parquet')
    item_df = pd.read_parquet(f'{root}/salesdocument_items.parquet')
    customer_df = pd.read_parquet(f'{root}/customers.parquet')
    address_df = pd.read_parquet(f'{root}/addresses.parquet')

    item_df['ID'] = range(len(item_df))  # Custom primary key.

    df_dict = {
        'salesdocuments': doc_df,
        'salesdocument_items': item_df,
        'customers': customer_df,
        'addresses': address_df,
    }

    # Do not use training columns within the graph:
    # salesdocuments:
    # * SALESOFFICE
    # * SALESGROUP
    # * CUSTOMERPAYMENTTERMS
    # * SHIPPINGCONDITION
    # * INCOTERMSCLASSIFICATION
    # salesdocument_items:
    # * PLANT
    # * SHIPPINGPOINT
    # * INCOTERMSCLASSIFICATION

    stypes: dict[str, dict[str, Stype]] = defaultdict(dict)
    stypes['salesdocuments']['SALESDOCUMENTTYPE'] = Stype.categorical
    stypes['salesdocuments']['SALESORGANIZATION'] = Stype.categorical
    stypes['salesdocuments']['DISTRIBUTIONCHANNEL'] = Stype.categorical
    stypes['salesdocuments']['ORGANIZATIONDIVISION'] = Stype.categorical
    stypes['salesdocuments']['BILLINGCOMPANYCODE'] = Stype.categorical
    stypes['salesdocuments']['TRANSACTIONCURRENCY'] = Stype.categorical
    stypes['salesdocuments']['CREATIONDATETIME'] = Stype.timestamp
    stypes['salesdocument_items']['SALESDOCUMENTITEM'] = Stype.categorical
    stypes['salesdocument_items']['SALESDOCUMENTITEMCATEGORY'] = (
        Stype.categorical)
    stypes['salesdocument_items']['PRODUCT'] = Stype.categorical
    stypes['salesdocument_items']['CREATIONDATETIME'] = Stype.timestamp
    stypes['customers'] = {}
    stypes['addresses']['ADDRESSREPRESENTATIONCODE'] = Stype.categorical
    stypes['addresses']['COUNTRY'] = Stype.categorical
    stypes['addresses']['REGION'] = Stype.categorical
    encoders = EncodersConfig.from_stypes(df_dict, stypes)

    time_col_dict = {
        'salesdocuments': 'CREATIONDATETIME',
        'salesdocument_items': 'CREATIONDATETIME',
    }
    fkeys = [
        ('salesdocument_items.SALESDOCUMENT', 'salesdocuments.SALESDOCUMENT'),
        ('salesdocument_items.SOLDTOPARTY', 'customers.CUSTOMER'),
        ('salesdocument_items.SHIPTOPARTY', 'customers.CUSTOMER'),
        ('salesdocument_items.BILLTOPARTY', 'customers.CUSTOMER'),
        ('salesdocument_items.PAYERPARTY', 'customers.CUSTOMER'),
        ('customers.ADDRESSID', 'addresses.ADDRESSID'),
    ]
    pkey_name_dict = {
        'salesdocuments': 'SALESDOCUMENT',
        'salesdocument_items': 'ID',
        'customers': 'CUSTOMER',
        'addresses': 'ADDRESSID',
    }
    feature_store = InMemoryFeatureStore.from_encoders(
        df_dict=df_dict,
        encoders=encoders,
        pkey_name_dict=pkey_name_dict,
        time_col_dict=time_col_dict,
        end_time_col_dict={},
    )

    graph_store = InMemoryGraphStore(
        df_dict=df_dict,
        fkeys=fkeys,
        time_col_dict=time_col_dict,
    )

    return feature_store, graph_store, encoders
