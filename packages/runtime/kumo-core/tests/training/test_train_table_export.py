# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from kumo_core.training import export

FREQUENCY = '__kumo_internal_frequency_column__'


def write_bundle(root, stages, tt_id='tt-1'):
    spec = {'tt_id': tt_id, 'split_stages': []}
    for key, parts in stages.items():
        names = []
        for index, table in enumerate(parts):
            name = f'{key}/part-{index}.parquet'
            os.makedirs(root / key, exist_ok=True)
            pq.write_table(table, str(root / name))
            names.append(name)
        spec['split_stages'].append({
            'key': key,
            'parts': names,
            'row_count': sum(t.num_rows for t in parts),
        })
    (root / export.INGEST_SPEC).write_text(json.dumps(spec))
    return str(root)


def node_rows(entities, extra=True):
    columns = {
        'ENTITY1': pa.array(entities, pa.int64()),
        'TARGET1': pa.array([float(e) for e in entities], pa.float64()),
        'TIMESTAMP': pa.array([10 * e for e in entities], pa.int64()),
    }
    if extra:
        columns['ENTITY_INCLUDE_BIT'] = pa.array([True] * len(entities))
    return pa.table(columns)


def link_rows(entities):
    return pa.table({
        'ENTITY1': pa.array(entities, pa.int64()),
        'TARGET1': pa.array([[e, e + 1] for e in entities],
                            pa.list_(pa.int64())),
        'TIMESTAMP': pa.array([5] * len(entities), pa.int64()),
        'WEIGHT': pa.array([0.5] * len(entities), pa.float64()),
        FREQUENCY: pa.array([2] * len(entities), pa.int64()),
        'exclude_target_keys': pa.array([[1]] * len(entities),
                                        pa.list_(pa.int64())),
    })


def read_all(files):
    return pa.concat_tables([pq.read_table(f) for f in files])


def test_link_bundle_schema_and_split_values(tmp_path):
    bundle = write_bundle(
        tmp_path / 'b', {
            'train': [link_rows([1, 2]), link_rows([3])],
            'val': [link_rows([4])],
            'test': [link_rows([5])],
        })
    files = export.export_train_table(bundle, str(tmp_path / 'out'))
    table = read_all(files)
    assert [os.path.basename(f) for f in files] == [
        f'part-{i:05d}.parquet' for i in range(4)
    ]
    assert table.schema.equals(
        pa.schema([
            pa.field('ENTITY1', pa.int64()),
            pa.field('TARGET1', pa.list_(pa.int64())),
            pa.field('TIMESTAMP', pa.int64()),
            pa.field('WEIGHT', pa.float64()),
            pa.field(FREQUENCY, pa.int64()),
            pa.field('SPLIT', pa.int64(), nullable=False),
        ]))
    rows = sorted(zip(table['ENTITY1'].to_pylist(),
                      table['SPLIT'].to_pylist()))
    assert rows == [(1, 0), (2, 0), (3, 0), (4, 1), (5, 2)]
    assert table['TARGET1'].to_pylist()[0] == [1, 2]


def test_node_bundle_follows_stage_order_not_tt_parquet_order(tmp_path):
    root = tmp_path / 'b'
    bundle = write_bundle(root, {
        'test': [node_rows([9])],
        'train': [node_rows([3, 1])],
        'val': [node_rows([2])],
    })
    pq.write_table(node_rows([9, 2, 1, 3]), str(root / 'tt.parquet'))
    table = read_all(export.export_train_table(bundle, str(tmp_path / 'out')))
    assert table.column_names == ['ENTITY1', 'TARGET1', 'TIMESTAMP', 'SPLIT']
    assert table['ENTITY1'].to_pylist() == [3, 1, 2, 9]
    assert table['SPLIT'].to_pylist() == [0, 0, 1, 2]
    assert table['TARGET1'].to_pylist() == [3.0, 1.0, 2.0, 9.0]


def test_empty_stage_is_skipped(tmp_path):
    bundle = write_bundle(
        tmp_path / 'b', {
            'train': [node_rows([1, 2])],
            'val': [node_rows([])],
            'test': [node_rows([3])],
        })
    files = export.export_train_table(bundle, str(tmp_path / 'out'))
    table = read_all(files)
    assert len(files) == 2
    assert table['SPLIT'].to_pylist() == [0, 0, 2]


def test_all_empty_table_yields_one_zero_row_part(tmp_path):
    bundle = write_bundle(tmp_path / 'b', {
        'train': [node_rows([])],
        'val': [node_rows([])],
        'test': [node_rows([])],
    })
    files = export.export_train_table(bundle, str(tmp_path / 'out'))
    table = pq.read_table(files[0])
    assert len(files) == 1
    assert table.num_rows == 0
    assert table.column_names == ['ENTITY1', 'TARGET1', 'TIMESTAMP', 'SPLIT']


def test_stages_with_different_schemas_raise(tmp_path):
    other = node_rows([2]).set_column(1, 'TARGET1',
                                      pa.array([1], pa.int64()))
    bundle = write_bundle(tmp_path / 'b', {
        'train': [node_rows([1])],
        'val': [other],
    })
    with pytest.raises(ValueError, match='does not match the schema'):
        export.export_train_table(bundle, str(tmp_path / 'out'))


def test_multi_part_stage_with_mismatched_part_fails(tmp_path):
    other = node_rows([3]).set_column(1, 'TARGET1',
                                      pa.array(['x'], pa.string()))
    bundle = write_bundle(tmp_path / 'b',
                          {'train': [node_rows([1]), node_rows([2]), other]})
    parts = export.iter_export_parts(bundle, str(tmp_path / 'out'))
    assert len([next(parts), next(parts)]) == 2
    with pytest.raises(ValueError, match='train.*does not match the schema'):
        next(parts)


def test_directory_without_spec_raises(tmp_path):
    with pytest.raises(ValueError, match='not a generated training table'):
        export.export_train_table(str(tmp_path), str(tmp_path / 'out'))


def test_unexpected_stage_key_raises(tmp_path):
    bundle = write_bundle(tmp_path / 'b', {'holdout': [node_rows([1])]})
    with pytest.raises(ValueError, match='Unexpected training table stages'):
        export.export_train_table(bundle, str(tmp_path / 'out'))


def test_stages_listing_no_parts_raise(tmp_path):
    (tmp_path / 'b').mkdir()
    bundle = write_bundle(tmp_path / 'b', {'train': [], 'val': []})
    with pytest.raises(ValueError, match='Unexpected training table stages'):
        export.export_train_table(bundle, str(tmp_path / 'out'))


def test_export_through_filesystem_matches_local(tmp_path):
    root = tmp_path / 'root'
    write_bundle(root / 'bundles' / 'b', {
        'train': [link_rows([1, 2])],
        'val': [link_rows([3])],
    })
    local = read_all(
        export.export_train_table(str(root / 'bundles' / 'b'),
                                  str(tmp_path / 'local')))
    remote = read_all(
        export.iter_export_parts(
            'bundles/b', str(tmp_path / 'remote'),
            pafs.SubTreeFileSystem(str(root), pafs.LocalFileSystem())))
    assert remote.equals(local)
