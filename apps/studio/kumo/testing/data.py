# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from collections.abc import Callable

import pandas as pd

from kumo.connector import CSVConnector, ParquetConnector
from kumo.graph import Graph


def load_df(dataset: str, filename: str, **kwargs) -> pd.DataFrame:
    path = osp.dirname(__file__)
    path = osp.abspath(osp.join(path, '..', '..', 'test', 'csv_data'))
    path = osp.join(path, dataset, filename)

    kwargs.setdefault('sep', ';')
    df = pd.read_csv(path, **kwargs)

    return df


def withTestDataFrame(dataset: str, filename: str, key='df', **kwargs):
    def decorator(func: Callable):
        import pytest

        return pytest.mark.parametrize(
            key, [load_df(dataset, filename, **kwargs)])(func)

    return decorator


def financial_csv_dir() -> str:
    root_dir = osp.dirname(osp.realpath(__file__))
    return osp.join(root_dir, '..', '..', 'test', 'csv_data', 'FINANCIAL')


def financial_csv_connector() -> CSVConnector:
    root_dir = financial_csv_dir()
    return CSVConnector('csv', root_dir, na_values='?')


def connect_financial_csv(graph: Graph):
    connector = financial_csv_connector()
    graph.connect(connector)


def load_financial_csv() -> Graph:
    root_dir = financial_csv_dir()
    path = osp.join(root_dir, 'metadata.json')
    graph = Graph(name="FINANCIAL")
    connect_financial_csv(graph)
    graph.add_from_config(path, data_source='csv')
    return graph


def financial_parquet_dir() -> str:
    root_dir = osp.dirname(osp.realpath(__file__))
    return osp.join(root_dir, '..', '..', 'test', 'parquet_data', 'FINANCIAL')


def financial_parquet_connector() -> ParquetConnector:
    root_dir = financial_parquet_dir()
    return ParquetConnector('parquet', root_dir)
