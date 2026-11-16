# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fingerprinting scheme stability test to enforce stability of
fingerprint id of `TableDefinition` and `GraphDefinition` and to prevent
inadvertent change in either the schema or fingerprinting scheme `fpid()`.

Below are instructions when this test fails (most likely because of dataclass
schema change that affects `TableDefinition` or `GraphDefinition`):

1). Make sure to follow the general rules about schema change as documented
on `kumo_dataclass`.  In short words, always add new field with default value.

2). When new field is added, make sure to update `fpid()` scheme in the
closest surrounding dataclass, typically: a) skip if new field value is same as
default (for backward stability); b) hash the new field value in a "stable"
manner, "stable" generally means fpid should remain agnostic to irrelevant
information such as ordering of elements (that doesn't matter) or
debugging/inferred data.

3). After following 1) and 2), you should update this test: a) update the
checked-in schema json file in `test_data/`; b) add new test cases to verify
fingerprinting scheme for the newly added field, i.e., setting the new field
value different than default would change the fpid.
"""

import json
import os
import random
from typing import Any

import kumoapi.source_table as api
from kumoapi.typing import Dtype, Stype
from pydantic import TypeAdapter

from io_utils.json import load_json
from rest.config.graph_v2 import ColumnKeyGroup, GraphDefinition
from rest.config.predictive_query_v2 import PQueryResource
from rest.config.table_v2 import Column, TableDefinition
from statedb.config.common import ConnectorID, TableName

_FILE_CONNECTOR_ID = f'connector_{"1" * 32}'


def _file_source(
    source_table_name: str,
    file_type: api.FileType = api.FileType.CSV,
) -> api.S3SourceTable:
    return api.S3SourceTable(
        connector_id=_FILE_CONNECTOR_ID,
        source={'mode': 'relative', 'path': source_table_name},
        file_type=file_type,
    )


def delete_key_from_dict(obj: dict[str, Any], key: str) -> dict[str, Any]:
    for k, v in list(obj.items()):
        if k == key:
            del obj[k]
        elif isinstance(v, dict):
            obj[k] = delete_key_from_dict(v, key)
    return obj


def test_table_definition_schema():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    table_schema_json_file = os.path.join(dir, 'table_definition_schema.json')
    with open(table_schema_json_file, 'r') as f:
        saved_table_schema = json.loads(f.read())

    # Drop doc-strings from the schema:
    schema = delete_key_from_dict(
        TypeAdapter(TableDefinition).json_schema(), key='description')

    # Uncomment if you need to make changes:
    # with open(os.path.join(dir, 'table_definition_schema.json'), 'w') as f:
    #     json.dump(schema, f, indent=2)

    assert schema == saved_table_schema


def test_graph_definition_schema():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    graph_schema_json_file = os.path.join(dir, 'graph_definition_schema.json')
    with open(graph_schema_json_file, 'r') as f:
        saved_graph_schema = json.loads(f.read())

    # Drop doc-strings from the schema:
    schema = delete_key_from_dict(
        TypeAdapter(GraphDefinition).json_schema(), key='description')

    # Uncomment if you need to make changes:
    # with open(os.path.join(dir, 'graph_definition_schema.json'), 'w') as f:
    #    json.dump(schema, f, indent=2)
    assert schema == saved_graph_schema


def test_table_fpid_stability():
    cols = (
        Column('col_1', stype=Stype.ID, dtype=Dtype.int),
        Column('col_2', stype=Stype.timestamp, dtype=Dtype.date),
    )
    table_1a = TableDefinition(
        cols=cols,
        source_table=_file_source('db/table.csv'),
    )
    table_1b = TableDefinition(
        cols=tuple(reversed(cols)),  # fpid should be agnostic to col order
        source_table=_file_source('db/table.csv'),
    )
    table_1c = TableDefinition(
        cols=tuple(reversed(cols)),
        source_table=_file_source('db/table.csv', api.FileType.CSV),
    )
    assert table_1a.fpid == 'e6b7aa57358b7d5fbe1268d628f47c7a'
    assert table_1b.fpid == 'e6b7aa57358b7d5fbe1268d628f47c7a'
    assert table_1c.fpid == 'e6b7aa57358b7d5fbe1268d628f47c7a'

    table_2 = TableDefinition(
        cols=cols,
        source_table=_file_source('db/table', api.FileType.CSV),
    )
    assert table_2.fpid == '42e8f80ef376d13014422aa575963f9f'

    table_3 = TableDefinition(
        cols=cols,
        source_table=_file_source('db/table', api.FileType.CSV),
        pkey='col_1',
    )
    assert table_3.fpid == 'ded067e222c4e0f2c56062ea4b02cf99'

    table_4 = TableDefinition(
        cols=cols,
        source_table=_file_source('db/table', api.FileType.CSV),
        pkey='col_1',
        time_col='col_2',
    )
    assert table_4.fpid == 'edd4465760197d9b7e231ba8e8d7e4ec'

    table_5 = TableDefinition(
        cols=cols, source_table=api.SnowflakeSourceTable(
            snowflake_connector_id=ConnectorID(f'connector_{"2" * 32}'),
            database='db', schema_name='kumo', table=TableName('my_table')))
    assert table_5.fpid == '4303c5db452fde3f2ae6d71983f672b8'

    table_6 = TableDefinition(
        cols=cols, source_table=api.DatabricksSourceTable(
            databricks_connector_id=ConnectorID(f'connector_{"3" * 32}'),
            table=TableName('schema____my_table')))
    assert table_6.fpid == '64d1109235e801d8b69cc7c414b1762b'

    # BigQuery table:
    table_7 = TableDefinition(
        cols=cols, source_table=api.BigQuerySourceTable(
            bigquery_connector_id=ConnectorID(f'connector_{"4" * 32}'),
            table_name=TableName('my_table'),
            project_id='my-project',
            dataset_id='my_dataset',
        ))
    assert table_7.fpid == '66ba3083342a942c11f02fe5cfcb2710'


def test_multicategorical_delimiter_changes_all_table_identity_layers():
    def _table(delimiter: str) -> TableDefinition:
        return TableDefinition(
            cols=(
                Column('id', stype=Stype.ID, dtype=Dtype.int),
                Column(
                    'tags',
                    stype=Stype.multicategorical,
                    dtype=Dtype.string,
                    multicategorical_delimiter=delimiter,
                ),
            ),
            source_table=_file_source('db/items.csv'),
            pkey='id',
        )

    comma = _table(',')
    pipe = _table('|')

    assert comma.cols[1].fpid != pipe.cols[1].fpid
    assert comma.fpid != pipe.fpid
    assert comma.fingerprint().fpid() != pipe.fingerprint().fpid()
    assert comma._dedupe_hash() != pipe._dedupe_hash()
    assert comma.cache_key() != pipe.cache_key()
    assert comma.wf_key() != pipe.wf_key()


def test_fpid_stability_in_pquery_fixture():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    pquery_res_path = os.path.join(dir, 'pquery_resource.json')
    pquery_res = load_json(pquery_res_path, PQueryResource)

    table_fpids = {
        t_name: t_def.fpid
        for t_name, t_def in pquery_res.graph.tables.items()
    }
    assert table_fpids == {
        'ACCOUNT': '6054c01e614f9de3b2da6514d0f97e8d',
        'CARD': 'ecc460b657cc92807e731671ec0bf4ab',
        'CLIENT': '94d413f8e7e51c3493e7bc822d8e85ac',
        'DISP': '62824fb2ea5d21917465079b38406e45',
        'DISTRICT': '89ee0e9d162ae580d1130e724e54ba31',
        'LOAN': 'e9e9fe33ac182cf1f84d292d26bdee7e',
        'ORDERS': '31a54514b7777f70ec71c1cf3d6fedb4',
        'TRANS': 'd3ab66f4a3e61e0352eec9015f25a3ca',
    }

    # Testing Graph FPID.
    assert pquery_res.graph.fpid == '747db4cc38ed7327d887ea68c2a48eda'
    graph = pquery_res.graph
    graph2 = GraphDefinition(tables=dict(reversed(graph.tables.items())),
                             col_groups=list(reversed(graph.col_groups)))
    assert graph2.fpid == graph.fpid

    graph3 = GraphDefinition(
        tables=graph.tables,
        col_groups=[
            ColumnKeyGroup(
                # Randomize columns in each column group
                tuple(random.sample(col_group.columns, len(col_group.columns))))
            for col_group in graph.col_groups
        ])
    assert graph3.fpid == graph.fpid
