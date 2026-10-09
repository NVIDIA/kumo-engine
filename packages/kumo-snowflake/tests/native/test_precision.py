# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_snowflake.types import dtype_for_described


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
