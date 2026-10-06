# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_snowflake.types import dtype_for, dtype_for_described
from kumoapi.typing import Dtype


@pytest.mark.parametrize('name,scale,expected', [
    ('FIXED', 0, Dtype.int64),
    ('FIXED', 2, Dtype.float64),
    ('REAL', None, Dtype.float64),
    ('TIMESTAMP', None, Dtype.time),
    ('MAP', None, Dtype.unsupported),
    ('VECTOR', None, Dtype.unsupported),
])
def test_described_aliases(name, scale, expected):
    assert dtype_for_described(name, scale) == expected


@pytest.mark.parametrize('name,scale,expected', [
    ('NUMBER', 0, Dtype.int64),
    ('NUMBER', 2, Dtype.float64),
    ('DECIMAL', '0', Dtype.int64),
    ('NUMERIC', '2', Dtype.float64),
    ('INTEGER', None, Dtype.int64),
    ('INT', None, Dtype.int64),
    ('BIGINT', None, Dtype.int64),
    ('FLOAT', None, Dtype.float64),
    ('DOUBLE', None, Dtype.float64),
    ('REAL', None, Dtype.float64),
    ('TEXT', None, Dtype.string),
    ('VARCHAR', None, Dtype.string),
    ('CHAR', None, Dtype.string),
    ('DATETIME', None, Dtype.time),
    ('VARBINARY', None, Dtype.binary),
    ('BINARY', None, Dtype.binary),
    ('DATE', None, Dtype.date),
    ('TIMESTAMP_NTZ', None, Dtype.time),
    ('TIMESTAMP_LTZ', None, Dtype.time),
    ('TIMESTAMP_TZ', None, Dtype.time),
    ('VARIANT', None, Dtype.string),
    ('ARRAY', None, Dtype.string),
    ('VECTOR', None, Dtype.unsupported),
    ('OBJECT', None, Dtype.unsupported),
    ('TIME', None, Dtype.unsupported),
    ('GEOGRAPHY', None, Dtype.unsupported),
    ('NOPE', None, Dtype.unsupported),
    ('text', None, Dtype.string),
    (' Boolean ', None, Dtype.bool),
])
def test_catalog_types(name, scale, expected):
    assert dtype_for(name, scale) == expected


def test_fractional_types_preserve_precision_through_native_ingestion(tmp_path):
    import json
    from decimal import Decimal

    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.diskgraph.ingest import (
        DiskgraphIngestSpec,
        execute_diskgraph_ingest,
    )
    from kumo_core.diskgraph.runner import IngestExecutionConfig

    pytest.importorskip('diskgraph')
    expected = {'FLOAT': 1.123456789012345, 'DECIMAL': 1.123456789}
    source = tmp_path / 'source'
    source.mkdir()
    pq.write_table(
        pa.table({
            'ID': [1],
            'FLOAT': [expected['FLOAT']],
            'DECIMAL':
            pa.array([Decimal('1.123456789')], type=pa.decimal128(18, 9)),
        }), source / 'part.parquet')
    graph = {
        'name':
        'precision',
        'tables': [{
            'table_name':
            'probe',
            'source_table_name':
            'probe',
            'connector_id':
            'snowflake',
            'pkey': {
                'name': 'ID',
                'dtype': 'int64',
                'stype': 'id'
            },
            'cols': [{
                'name': name,
                'dtype': dtype_for_described(kind, scale).value,
                'stype': 'numerical',
            } for name, kind, scale in [('FLOAT', 'REAL',
                                         None), ('DECIMAL', 'FIXED', 9)]],
            'fkeys': [],
        }],
    }
    result = execute_diskgraph_ingest(
        graph,
        DiskgraphIngestSpec(data_persist_dir=str(tmp_path / 'persist'),
                            table_data_versions={'probe': '1'},
                            source_paths={'probe': str(source)},
                            output_dir=str(tmp_path / 'index'),
                            execution=IngestExecutionConfig(initial_threads=2)))
    with open(result.col_stats_path) as handle:
        stats = json.load(handle)['probe']
    for name, value in expected.items():
        assert stats[name]['min'] == value
        assert stats[name]['max'] == value
