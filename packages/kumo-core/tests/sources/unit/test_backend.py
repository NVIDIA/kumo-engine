# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.sources.backend import ingest_result, select_columns
from kumo_core.sources.errors import SourceNotFoundError
from kumo_core.sources.types import ReadMode, StagedTable
from kumoapi.typing import Dtype


@pytest.mark.parametrize('declared', [None, {}])
def test_unspecified_columns_use_catalog(declared):
    catalog = {'id': Dtype.int64, 'created_at': Dtype.time}
    selected = select_columns('events', catalog, declared)
    assert list(selected.items()) == list(catalog.items())
    assert selected is not catalog


def test_projection_keeps_catalog_order_and_declared_types():
    catalog = {
        'id': Dtype.int64,
        'created_at': Dtype.int64,
        'unused': Dtype.string,
    }
    declared = {'created_at': Dtype.time, 'id': Dtype.string}
    selected = select_columns('events', catalog, declared)
    assert list(selected.items()) == [('id', Dtype.string),
                                      ('created_at', Dtype.time)]
    assert catalog['created_at'] == Dtype.int64
    assert list(declared) == ['created_at', 'id']


def test_missing_declared_columns_are_reported_together():
    with pytest.raises(SourceNotFoundError) as error:
        select_columns('events', {'id': Dtype.int64}, {
            'missing_z': Dtype.string,
            'id': Dtype.int64,
            'missing_a': Dtype.time,
        })
    assert str(error.value) == (
        'Table events has no column missing_a, missing_z. The graph '
        'declares it, so it cannot be read; correct the table or drop '
        'the column from the graph.')


def test_empty_catalog():
    assert select_columns('events', {}) == {}
    with pytest.raises(SourceNotFoundError):
        select_columns('events', {}, {'id': Dtype.int64})


def test_ingest_result_preserves_local_paths_and_measured_size(tmp_path):
    path = tmp_path / 'part-000.parquet'
    staged = StagedTable(directory=str(tmp_path), paths=(str(path), ),
                         version='source-version', source_size=125,
                         mode_used=ReadMode.STAGED, num_rows=3)
    result = ingest_result(staged)
    assert result.source == {'type': 'parquet', 'paths': [str(path)]}
    assert result.source_size == 125
    assert result.version == 'source-version'
    assert result.mode_used == 'staged'


def test_schema_adapter_preserves_duplicate_and_quoted_names():
    from kumo_core.sources.backend import source_columns

    columns = source_columns([(' mixed ', Dtype.int64),
                              (' mixed ', Dtype.string)])
    assert [(col.name, col.dtype)
            for col in columns] == [(' mixed ', Dtype.int64),
                                    (' mixed ', Dtype.string)]
    assert all(col.is_primary is False for col in columns)


def test_sample_response_empty_rows_retains_column_order():
    from io import StringIO

    import pandas as pd
    from kumo_core.sources.backend import sampled_table, source_columns

    response = sampled_table(
        'events', source_columns([('z', Dtype.string), ('a', Dtype.int64)]), [])
    frame = pd.read_json(StringIO(response.sample_rows), orient='table')
    assert list(frame.columns) == ['z', 'a']
    assert len(frame) == 0
