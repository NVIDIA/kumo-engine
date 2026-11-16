# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom-row validation through the native builder and extension ingestion."""

import json
from dataclasses import replace
from pathlib import Path

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.training.table import (
    PreparedCustomTrainingTable,
    TrainTableRequest,
    generate_custom_train_table,
    ingest_train_table,
)


@pytest.fixture
def train_request(tmp_path):
    source = tmp_path / 'users.parquet'
    pq.write_table(pa.table({'id': list(range(6))}), source)
    index = tmp_path / 'index'
    diskgraph.ingest(
        config_json=json.dumps({
            'dataset_name':
            'custom_rows',
            'num_shards':
            1,
            'tables': [{
                'name': 'users',
                'primary_key': 'id',
                'columns': [{
                    'name': 'id',
                    'ingested_dtype': 'int64'
                }],
                'source': {
                    'type': 'parquet',
                    'paths': [str(source)]
                },
            }],
        }),
        output_dir=str(index),
        col_stats=False,
    )
    return TrainTableRequest(
        query=None,
        timestamps=(),
        task_type='regression',
        forecast_length=0,
        lag_timesteps=None,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(index),
        output_path=str(tmp_path / 'generated.parquet'),
        tt_id='custom',
        split_config={
            'split_column': 'SPLIT',
            'manual_train_values': ['0'],
            'manual_val_values': ['1'],
            'manual_test_values': ['2'],
        },
        split_family='ManualSplit',
        input_timestamp_unit='nanoseconds',
        target_stype='numerical',
        entity_node_type='users',
        entity_primary_key='id',
    )


def source_table(condition):
    columns = {
        'ENTITY1': pa.array(range(6), pa.int64()),
        'TARGET1': pa.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        'WEIGHT': pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        'SPLIT': pa.array([0, 0, 1, 1, 2, 2], pa.int64()),
    }
    column, bad = {
        'null_entity': ('ENTITY1', None),
        'null_target': ('TARGET1', None),
        'null_weight': ('WEIGHT', None),
        'non_finite_weight': ('WEIGHT', float('inf')),
        'negative_weight': ('WEIGHT', -1.0),
        'invalid_split': ('SPLIT', 7),
    }[condition]
    values = columns[column].to_pylist()
    values[1] = bad
    dtype = columns[column].type
    columns[column] = pa.array(values, dtype)
    return pa.table(columns)


@pytest.mark.parametrize('condition', [
    'null_entity',
    'null_target',
    'null_weight',
    'non_finite_weight',
    'negative_weight',
    'invalid_split',
])
@pytest.mark.parametrize('policy', ['abort', 'skip_row_on_error'])
def test_custom_row_policy_builds_and_ingests_native_bundle(
        tmp_path, train_request, condition, policy):
    source = tmp_path / 'custom.parquet'
    pq.write_table(source_table(condition), source)
    prepared = PreparedCustomTrainingTable(str(source), weight_column='WEIGHT',
                                           ingest_on_error=policy)

    if policy == 'abort':
        with pytest.raises(diskgraph.DiskgraphDataError, match='row 1'):
            generate_custom_train_table(train_request, prepared)
        assert not (tmp_path / 'index' / 'train_tables' / 'custom').exists()
        return

    generated = generate_custom_train_table(train_request, prepared)
    manifest = json.loads(Path(generated.manifest_path).read_text())
    failures = manifest['row_failures']
    assert failures['rows_read'] == 6
    assert failures['skipped_rows'] == 1
    assert failures['on_error'] == policy
    assert [(entry['condition'], entry['count'])
            for entry in failures['by_condition']] == [(condition, 1)]

    output = pq.read_table(generated.train_table_path)
    assert output.num_rows == 5
    assert output['ENTITY1'].to_pylist() == [0, 2, 3, 4, 5]
    assert output['WEIGHT'].to_pylist() == [1.0, 3.0, 4.0, 5.0, 6.0]
    assert output['TARGET1'].to_pylist() == [0.1, 0.3, 0.4, 0.5, 0.6]

    ingested = ingest_train_table(
        staging_dir=generated.staging_dir,
        index_path=train_request.index_path,
        tt_id=train_request.tt_id,
    )
    published = json.loads(Path(ingested.manifest_path).read_text())
    assert published['row_failures'] == failures


@pytest.mark.parametrize('policy', ['abort', 'skip_row_on_error'])
def test_source_with_no_valid_rows_fails_in_both_modes(tmp_path, train_request,
                                                       policy):
    source = tmp_path / 'invalid.parquet'
    table = source_table('null_target')
    table = table.set_column(1, 'TARGET1', pa.array([None] * 6, pa.float64()))
    pq.write_table(table, source)
    with pytest.raises(diskgraph.DiskgraphDataError):
        generate_custom_train_table(
            train_request,
            PreparedCustomTrainingTable(str(source), ingest_on_error=policy))
    assert not (tmp_path / 'index' / 'train_tables' / 'custom').exists()


def test_aborted_source_can_be_retried_with_skip_mode(tmp_path, train_request):
    source = tmp_path / 'retry.parquet'
    pq.write_table(source_table('null_weight'), source)
    with pytest.raises(diskgraph.DiskgraphDataError):
        generate_custom_train_table(
            train_request,
            PreparedCustomTrainingTable(str(source), weight_column='WEIGHT',
                                        ingest_on_error='abort'))
    assert source.exists()
    generated = generate_custom_train_table(
        train_request,
        PreparedCustomTrainingTable(str(source), weight_column='WEIGHT',
                                    ingest_on_error='skip_row_on_error'))
    assert pq.read_table(generated.train_table_path).num_rows == 5
    ingest_train_table(staging_dir=generated.staging_dir,
                       index_path=train_request.index_path,
                       tt_id=train_request.tt_id)


@pytest.mark.parametrize('policy', ['abort', 'skip_row_on_error'])
def test_temporal_link_weights_keep_negative_values(tmp_path, train_request,
                                                    policy):
    train_request = replace(train_request, task_type='temporal_link_prediction',
                            rhs_node_type='users', target_stype='ID',
                            list_distinct_target_is_array=True)
    source = tmp_path / 'links.parquet'
    table = source_table('negative_weight')
    table = table.set_column(
        1, 'TARGET1', pa.array([[i] for i in range(6)], pa.list_(pa.int64())))
    table = table.set_column(
        2, 'WEIGHT',
        pa.array([[1.0], [-1.0], [1.0], [1.0], [1.0], [1.0]],
                 pa.list_(pa.float64())))
    pq.write_table(table, source)
    generated = generate_custom_train_table(
        train_request,
        PreparedCustomTrainingTable(str(source), weight_column='WEIGHT',
                                    ingest_on_error=policy))
    output = pq.read_table(generated.train_table_path)
    assert output.num_rows == 6
    assert output['WEIGHT'].to_pylist() == [[1.0], [-1.0], [1.0], [1.0], [1.0],
                                            [1.0]]
    ingest_train_table(staging_dir=generated.staging_dir,
                       index_path=train_request.index_path,
                       tt_id=train_request.tt_id)


@pytest.mark.parametrize('unit,dtype,divisor', [
    ('nanoseconds', pa.int64(), 1),
    ('seconds', pa.int64(), 10**9),
    ('milliseconds', pa.int64(), 10**6),
    ('microseconds', pa.int64(), 10**3),
    ('nanoseconds', pa.timestamp('ms'), 10**6),
])
def test_correct_timestamp_units_build_and_publish(tmp_path, train_request,
                                                   unit, dtype, divisor):
    nanos = 1_700_000_000_000_000_000
    source = tmp_path / 'correct.parquet'
    table = source_table('null_target').set_column(1, 'TARGET1',
                                                   pa.array([0.1] * 6))
    table = table.append_column('TIMESTAMP',
                                pa.array([nanos // divisor] * 6, dtype))
    pq.write_table(table, source)
    generated = generate_custom_train_table(
        train_request,
        PreparedCustomTrainingTable(str(source), timestamp_unit=unit))
    assert pq.read_table(generated.train_table_path)['TIMESTAMP'].cast(
        pa.int64()).to_pylist() == [nanos] * 6
    ingest_train_table(staging_dir=generated.staging_dir,
                       index_path=train_request.index_path,
                       tt_id=train_request.tt_id)


@pytest.mark.parametrize('policy', ['abort', 'skip_row_on_error'])
@pytest.mark.parametrize('entity', [None, 99])
def test_all_fields_validated_before_entity_filter(tmp_path, train_request,
                                                   policy, entity):
    table = source_table('null_target')
    table = table.set_column(0, 'ENTITY1',
                             pa.array([0, entity, 2, 3, 4, 5], pa.int64()))
    table = table.set_column(2, 'WEIGHT',
                             pa.array([1.0, None, 3.0, 4.0, 5.0, 6.0]))
    source = tmp_path / 'multiple.parquet'
    pq.write_table(table, source)
    prepared = PreparedCustomTrainingTable(str(source), weight_column='WEIGHT',
                                           ingest_on_error=policy)
    if policy == 'abort':
        with pytest.raises(diskgraph.DiskgraphDataError) as error:
            generate_custom_train_table(train_request, prepared)
        assert 'target is null' in str(error.value)
        assert 'weight is null' in str(error.value)
        if entity is None:
            assert 'entity key is null' in str(error.value)
        assert not source.with_suffix('.artifacts').exists()
        return
    generated = generate_custom_train_table(train_request, prepared)
    failures = json.loads(Path(
        generated.manifest_path).read_text())['row_failures']
    expected = {'null_target', 'null_weight'}
    if entity is None:
        expected.add('null_entity')
    assert failures['skipped_rows'] == 1
    assert {entry['condition']
            for entry in failures['by_condition']} == expected
    assert all(entry['count'] == 1 for entry in failures['by_condition'])
    assert pq.read_table(generated.train_table_path).num_rows == 5
    ingested = ingest_train_table(staging_dir=generated.staging_dir,
                                  index_path=train_request.index_path,
                                  tt_id=train_request.tt_id)
    assert json.loads(Path(
        ingested.manifest_path).read_text())['row_failures'] == failures


@pytest.mark.parametrize('policy', ['abort', 'skip_row_on_error'])
@pytest.mark.parametrize('empty_target', [True, False])
def test_link_row_reports_all_failures(tmp_path, train_request, policy,
                                       empty_target):
    request = replace(train_request, task_type='multilabel_ranking',
                      rhs_node_type='users', target_stype='ID',
                      list_distinct_target_is_array=True)
    table = source_table('null_target').set_column(
        1, 'TARGET1',
        pa.array(
            [[0], [] if empty_target else [1, 2, 3, 4], [2], [3], [4], [5]],
            pa.list_(pa.int64())))
    table = table.set_column(
        2, 'WEIGHT',
        pa.array([[1.0], None if empty_target else
                  [None, None, float('inf'), -1.0], [1.0], [1.0], [1.0], [1.0]],
                 pa.list_(pa.float64())))
    source = tmp_path / 'links.parquet'
    pq.write_table(table, source)
    prepared = PreparedCustomTrainingTable(str(source), weight_column='WEIGHT',
                                           ingest_on_error=policy)
    if policy == 'abort':
        with pytest.raises(diskgraph.DiskgraphDataError) as error:
            generate_custom_train_table(request, prepared)
        assert 'weight is null' in str(error.value)
        assert ('target list is empty'
                if empty_target else 'weight is not finite') in str(error.value)
        if not empty_target:
            assert 'weight is negative' in str(error.value)
        return
    generated = generate_custom_train_table(request, prepared)
    failures = json.loads(Path(
        generated.manifest_path).read_text())['row_failures']
    expected = ({'empty_target_list', 'null_weight'} if empty_target else
                {'null_weight', 'non_finite_weight', 'negative_weight'})
    assert failures['skipped_rows'] == 1
    assert {entry['condition']
            for entry in failures['by_condition']} == expected
    assert all(entry['count'] == 1 for entry in failures['by_condition'])
    assert pq.read_table(generated.train_table_path).num_rows == 5
    ingested = ingest_train_table(staging_dir=generated.staging_dir,
                                  index_path=request.index_path,
                                  tt_id=request.tt_id)
    assert json.loads(Path(
        ingested.manifest_path).read_text())['row_failures'] == failures


def test_default_policy_skips_invalid_rows(tmp_path, train_request):
    source = tmp_path / 'default.parquet'
    pq.write_table(source_table('null_weight'), source)
    generated = generate_custom_train_table(
        train_request,
        PreparedCustomTrainingTable(str(source), weight_column='WEIGHT'))
    report = json.loads(Path(
        generated.manifest_path).read_text())['row_failures']
    assert report['on_error'] == 'skip_row_on_error'
    assert report['skipped_rows'] == 1
    assert pq.read_table(generated.train_table_path).num_rows == 5
