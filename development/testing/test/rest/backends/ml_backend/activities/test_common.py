# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kumoapi.table as table_api
import pytest
from kumo_core.types.schema import FPID
from kumoapi.source_table import FileType
from kumoapi.typing import Dtype, Stype

from kumo.config import (
    ColumnConfig,
    ConnectorConfig,
    GraphConfig,
    PrimaryKeyConfig,
    TableConfig,
    TimeColumnConfig,
)
from rest.backends.ml_backend.activities import common as activity_common
from rest.config.graph_v2 import GraphDataComputeInput, GraphDefinition
from rest.config.table_v2 import Column, S3SourceTable, TableDefinition
from statedb.config.common import TableIDv2, TableName


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


def _make_graph_config() -> GraphConfig:
    return GraphConfig(
        name="graph",
        data_source_dict={'DUMMY': ConnectorConfig(name='DUMMY')},
        tables=[
            TableConfig(
                table_name='ACCOUNT',
                source_name='ACCOUNT',
                data_source='DUMMY',
                cols=[
                    ColumnConfig(
                        name="K_SYMBOL",
                        dtype=Dtype.string,
                        stype=Stype.categorical,
                    ),
                    ColumnConfig(
                        name="DATE",
                        dtype=Dtype.date,
                        stype=Stype.timestamp,
                    ),
                    ColumnConfig(
                        name="TIMESTAMP",
                        dtype=Dtype.time,
                        stype=Stype.timestamp,
                    ),
                    ColumnConfig(
                        name="TAGS",
                        dtype=Dtype.string,
                        stype=Stype.multicategorical,
                        multicategorical_delimiter=',',
                    ),
                ],
                time_col=TimeColumnConfig(
                    name="TIMESTAMP",
                    dtype=Dtype.time,
                    stype=Stype.timestamp,
                ),
                pkey=PrimaryKeyConfig(
                    name="ACCOUNT_ID",
                    dtype=Dtype.string,
                    stype=Stype.ID,
                ),
            ),
        ],
    )


def test_update_graph_config_from_graph_input():
    # Construct the v2 input by hand: `build_graph_data_compute_input`
    # requires a real connector to resolve s3 paths via `_get_path`,
    # which the synthetic `ConnectorConfig` in `_make_graph_config`
    # cannot provide. The function under test only consumes the resulting
    # `GraphDataComputeInput`, so building it directly exercises the
    # v2→v1 bridge contract without introducing real connector I/O.
    graph_config = _make_graph_config()
    table_def = TableDefinition(
        cols=[
            Column(name='ACCOUNT_ID', stype=Stype.ID, dtype=Dtype.string,
                   timestamp_format=None),
            Column(name='K_SYMBOL', stype=Stype.categorical, dtype=Dtype.string,
                   timestamp_format=None),
            Column(name='DATE', stype=Stype.timestamp, dtype=Dtype.date,
                   timestamp_format=table_api.TimestampUnit.SECOND),
            Column(name='TIMESTAMP', stype=Stype.timestamp, dtype=Dtype.time,
                   timestamp_format=table_api.TimestampUnit.SECOND),
            Column(name='TAGS', stype=Stype.multicategorical,
                   dtype=Dtype.string, multicategorical_delimiter='|'),
        ],
        source_table=S3SourceTable(
            file_type=FileType.PARQUET,
            connector_id=f'connector_{"0" * 32}',
            source={
                'mode': 'relative',
                'path': 'ACCOUNT'
            },
        ),
        pkey='ACCOUNT_ID',
        time_col='TIMESTAMP',
    )
    # Match the v2 id shape production builds in `GraphDriver` (which keys
    # `GraphDataComputeInput.tables` by `TableIDv2.compose(...)`).
    table_id = TableIDv2.compose(pquery_id=None,
                                 table_name=TableName('ACCOUNT'),
                                 fpid=FPID('a' * 32))
    graph_input = GraphDataComputeInput(
        graph_display_name='graph',
        graph_key='graph',
        tables={table_id: table_def},
        base_tables={},
        connectors={},
        graph_definition=GraphDefinition(
            tables={TableName('ACCOUNT'): table_def},
            col_groups=[],
        ),
        table_aliases={table_id: TableName('ACCOUNT')},
    )
    updated_graph_config = activity_common.update_graph_config_from_graph_input(
        graph_config, graph_input)
    assert graph_config['ACCOUNT'].time_col.timestamp_unit is None
    assert updated_graph_config['ACCOUNT'].time_col.timestamp_unit == 's'
    assert graph_config['ACCOUNT'].get_col('DATE').timestamp_unit is None
    assert updated_graph_config['ACCOUNT'].get_col('DATE').timestamp_unit == 's'
    assert graph_config['ACCOUNT'].get_col(
        'TAGS').multicategorical_delimiter == ','
    assert updated_graph_config['ACCOUNT'].get_col(
        'TAGS').multicategorical_delimiter == '|'
