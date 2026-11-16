# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from google.cloud import bigquery
from kumo_bigquery.sink import _replace
from kumo_bigquery.write import (
    OutputSchemaMismatchError,
    _arrow_type,
    _keep_column_metadata,
    bq_field,
    bq_schema,
    conformed,
    conformed_batches,
    is_output_table_id,
    load_job_config,
    nonempty,
    plan,
    write_permissions,
)
from kumo_core.sources.errors import PermanentSourceError
from kumoapi.jobs import WriteMode

UTC = timezone.utc


def _write(path: Path, **columns) -> Path:
    pq.write_table(pa.table(columns), path)
    return path


REF = 'p.d.t'


def _target(files, extra=None):
    return plan(files, extra, replace=True, ref=REF).target


def _table(*fields: tuple[str, str, str]) -> bigquery.Table:
    table = bigquery.Table('p.d.t')
    table.schema = [bigquery.SchemaField(*field) for field in fields]
    return table


def test_plan_canonicalises_timestamps_and_nulls(tmp_path):
    stamps = pa.array([datetime(2026, 1, 1, tzinfo=UTC)],
                      pa.timestamp('ns', tz='UTC'))
    first = _write(tmp_path / 'a.parquet', ts=stamps, score=pa.array([None],
                                                                     pa.null()))
    second = _write(tmp_path / 'b.parquet', ts=stamps,
                    score=pa.array([1.5], pa.float64()))

    schema = _target([first, second])

    assert schema.field('ts').type == pa.timestamp('us', tz='UTC')
    assert schema.field('score').type == pa.float64()
    assert all(field.nullable for field in schema)


def test_plan_canonicalises_types(tmp_path):
    path = _write(
        tmp_path / 'a.parquet',
        i=pa.array([1], pa.int32()),
        f=pa.array([1.5], pa.float32()),
        s=pa.array(['x'], pa.large_string()),
        d=pa.array([datetime(2026, 1, 1).date()], pa.date32()),
        emb=pa.array([[0.5]], pa.list_(pa.float32())),
        ids=pa.array([[1]], pa.list_(pa.int8())),
    )

    schema = _target([path])

    assert [field.type for field in schema] == [
        pa.int64(),
        pa.float64(),
        pa.string(),
        pa.date32(),
        pa.list_(pa.float64()),
        pa.list_(pa.int64()),
    ]


def test_plan_keeps_unmappable_types(tmp_path):
    nested = pa.array([{'a': 1}], pa.struct([('a', pa.int64())]))
    path = _write(tmp_path / 'a.parquet', s=nested)

    assert _target([path]).field('s').type == nested.type


def test_bq_schema_maps_embeddings_to_repeated_float():
    target = pa.schema([('id', pa.int64()), ('emb', pa.list_(pa.float64()))])

    fields = bq_schema(target)

    assert [(f.name, f.field_type, f.mode) for f in fields] == [
        ('id', 'INTEGER', 'NULLABLE'),
        ('emb', 'FLOAT', 'REPEATED'),
    ]


def test_bq_schema_is_none_when_a_column_is_unmappable():
    target = pa.schema([('id', pa.int64()),
                        ('s', pa.struct([('a', pa.int64())]))])

    assert bq_schema(target) is None


def test_nonempty_drops_empty_shards(tmp_path):
    empty = _write(tmp_path / 'a.parquet', v=pa.array([], pa.int64()))
    full = _write(tmp_path / 'b.parquet', v=[1])

    assert nonempty([empty, full]) == [full]


def test_nonempty_keeps_one_shard_when_all_are_empty(tmp_path):
    first = _write(tmp_path / 'a.parquet', v=pa.array([], pa.int64()))
    second = _write(tmp_path / 'b.parquet', v=pa.array([], pa.int64()))

    assert nonempty([first, second]) == [first]


def test_conformed_batches_fills_missing_and_stamps_extra(tmp_path):
    path = _write(tmp_path / 'a.parquet', id=[1, 2])
    target = pa.schema([('id', pa.int64()), ('score', pa.float64()),
                        ('run', pa.string())])

    table = pa.Table.from_batches(conformed_batches(path, target,
                                                    {'run': 'r1'}))

    assert table.to_pydict() == {
        'id': [1, 2],
        'score': [None, None],
        'run': ['r1', 'r1'],
    }


def test_plan_extra_columns_append_and_override(tmp_path):
    path = _write(tmp_path / 'a.parquet', id=[1], run=['old'])

    schema = _target([path], {'run': 7, 'tag': 'x'})

    assert schema.names == ['id', 'run', 'tag']
    assert schema.field('run').type == pa.int64()
    assert schema.field('tag').type == pa.string()


def test_plan_rejects_incompatible_shards(tmp_path):
    ints = _write(tmp_path / 'a.parquet', v=[1])
    lists = _write(tmp_path / 'b.parquet', v=[[1.0]])

    with pytest.raises(PermanentSourceError, match='incompatible'):
        _target([ints, lists])


def test_conformed_passes_matching_file_through(tmp_path):
    path = _write(tmp_path / 'a.parquet', id=[1], score=[0.5])
    target = _target([path])

    with conformed(path, target, None, str(tmp_path)) as out:
        assert out == path
    assert path.exists()


def test_conformed_rewrites_to_target_and_cleans_up(tmp_path):
    when = datetime(2026, 1, 1, 0, 0, 1, 123456, tzinfo=UTC)
    nanos = pa.array([when], pa.timestamp('ns', tz='UTC'))
    path = _write(tmp_path / 'a.parquet', id=[1], ts=nanos)
    extra = {'run': 'r1'}
    target = _target([path], extra)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()

    with conformed(path, target, extra, str(scratch)) as out:
        assert out != path
        written = pq.read_table(out)
        assert written.schema.equals(target)
        assert written['ts'].to_pylist() == [when]
        assert written['run'].to_pylist() == ['r1']
    assert not out.exists()


def _append(files, existing, extra=None):
    return plan(files, extra, existing, replace=False, ref=REF)


def test_append_targets_the_table_columns(tmp_path):
    path = _write(tmp_path / 'a.parquet', id=[1], emb=[[0.5]])
    table = _table(('ID', 'INTEGER', 'NULLABLE'), ('emb', 'FLOAT', 'REPEATED'),
                   ('spare', 'STRING', 'NULLABLE'))

    layout = _append([path], table)

    assert layout.target.names == ['ID', 'emb']
    assert layout.target.field('emb').type == pa.list_(pa.float64())
    assert layout.schema == list(table.schema)


def test_append_names_every_mismatched_column(tmp_path):
    path = _write(tmp_path / 'a.parquet', ENTITY=['x'], CLASS=['y'], ok=[1])
    table = _table(('ENTITY', 'INTEGER', 'NULLABLE'),
                   ('ok', 'INTEGER', 'NULLABLE'))

    with pytest.raises(OutputSchemaMismatchError) as caught:
        _append([path], table)

    assert caught.value.problems == [
        'ENTITY is STRING NULLABLE but the table column is INTEGER NULLABLE',
        'CLASS (STRING NULLABLE) is not a table column',
    ]
    message = str(caught.value)
    assert 'Cannot APPEND to p.d.t' in message
    assert 'ENTITY is STRING NULLABLE but the table column is INTEGER' in message
    assert 'CLASS (STRING NULLABLE) is not a table column' in message


def test_append_rejects_float_into_integer_and_list_into_scalar(tmp_path):
    path = _write(tmp_path / 'a.parquet', a=[1.5], b=[[1]])
    table = _table(('a', 'INTEGER', 'NULLABLE'), ('b', 'INTEGER', 'NULLABLE'))

    with pytest.raises(OutputSchemaMismatchError) as caught:
        _append([path], table)

    assert len(caught.value.problems) == 2


@pytest.mark.parametrize('column, mode, expected', [
    ('FLOAT', 'NULLABLE', pa.float64()),
    ('NUMERIC', 'NULLABLE', pa.decimal128(38, 9)),
    ('BIGNUMERIC', 'NULLABLE', pa.decimal256(76, 38)),
    ('FLOAT', 'REPEATED', pa.list_(pa.float64())),
])
def test_append_widens_integers(tmp_path, column, mode, expected):
    path = _write(tmp_path / 'a.parquet',
                  v=[[1]] if mode == 'REPEATED' else [1])

    layout = _append([path], _table(('v', column, mode)))

    assert layout.target.field('v').type == expected


def test_append_drops_all_null_columns_that_fit_any_column(tmp_path):
    path = _write(tmp_path / 'a.parquet', id=[1], seen=pa.array([None],
                                                                pa.null()))
    table = _table(('id', 'INTEGER', 'NULLABLE'),
                   ('seen', 'DATETIME', 'NULLABLE'))

    layout = _append([path], table)

    assert layout.target.names == ['id']
    assert layout.schema == list(table.schema)


def test_append_checks_extra_columns(tmp_path):
    path = _write(tmp_path / 'a.parquet', id=[1])
    table = _table(('id', 'INTEGER', 'NULLABLE'),
                   ('run', 'INTEGER', 'NULLABLE'))

    assert _append([path], table, {'run': 7}).target.names == ['id', 'run']
    with pytest.raises(OutputSchemaMismatchError, match='run is STRING'):
        _append([path], table, {'run': 'r1'})


def test_overwrite_ignores_the_table_columns_but_keeps_their_metadata(tmp_path):
    path = _write(tmp_path / 'a.parquet', id=['x'], kept=[1])
    existing = _table(('id', 'INTEGER', 'NULLABLE'),
                      ('kept', 'INTEGER', 'NULLABLE'))
    existing.schema = [
        bigquery.SchemaField('id', 'INTEGER', description='old id'),
        bigquery.SchemaField('kept', 'INTEGER', description='kept'),
    ]

    layout = plan([path], None, existing, replace=True, ref=REF)

    assert [(f.name, f.field_type, f.description) for f in layout.schema] == [
        ('id', 'STRING', None),
        ('kept', 'INTEGER', 'kept'),
    ]


@pytest.mark.parametrize('type_name, mode, expected', [
    ('NUMERIC', 'NULLABLE', pa.decimal128(38, 9)),
    ('DECIMAL', 'NULLABLE', pa.decimal128(38, 9)),
    ('BIGNUMERIC', 'NULLABLE', pa.decimal256(76, 38)),
    ('DATETIME', 'NULLABLE', pa.timestamp('us')),
    ('TIME', 'NULLABLE', pa.time64('us')),
    ('BYTES', 'NULLABLE', pa.binary()),
    ('JSON', 'NULLABLE', pa.string()),
    ('GEOGRAPHY', 'NULLABLE', pa.string()),
    ('INT64', 'REPEATED', pa.list_(pa.int64())),
    ('RECORD', 'NULLABLE', None),
])
def test_arrow_type_of_table_columns(type_name, mode, expected):
    field = bigquery.SchemaField('c', type_name, mode=mode)

    assert _arrow_type(field) == expected


def test_keep_column_metadata_needs_the_same_name_and_type():
    tags = bigquery.PolicyTagList(['projects/p/taxonomies/1/policyTags/2'])
    current = [
        bigquery.SchemaField('A', 'INTEGER', description='a', policy_tags=tags),
        bigquery.SchemaField('b', 'STRING', description='b'),
    ]
    schema = [
        bigquery.SchemaField('a', 'INTEGER'),
        bigquery.SchemaField('b', 'INTEGER'),
    ]

    merged = _keep_column_metadata(schema, current)

    assert [f.description for f in merged] == ['a', None]
    assert merged[0].policy_tags == tags
    assert merged[1].policy_tags is None


def test_conformed_batches_turn_null_repeated_values_into_empty_lists(tmp_path):
    path = _write(tmp_path / 'a.parquet', emb=pa.array([None, [1.5]],
                                                       pa.list_(pa.float64())))
    target = pa.schema([('emb', pa.list_(pa.float64()))])

    table = pa.Table.from_batches(conformed_batches(path, target))

    assert table['emb'].to_pylist() == [[], [1.5]]


def test_bq_field_is_none_for_unmapped_types():
    assert bq_field(pa.field('a', pa.struct([('a', pa.int64())]))) is None
    assert bq_field(pa.field('a',
                             pa.list_(pa.struct([('a', pa.int64())])))) is None


def test_replace_follows_write_mode():
    assert _replace(WriteMode.OVERWRITE) is True
    assert _replace(WriteMode.APPEND) is False


@pytest.mark.parametrize('name, valid', [
    ('predictions_2026-01', True),
    ('a.b', False),
    ('has space', False),
    ('', False),
    ('name\n', False),
])
def test_is_output_table_id(name, valid):
    assert is_output_table_id(name) is valid


def test_write_permissions():
    assert write_permissions(replace=False) == [
        'bigquery.tables.get', 'bigquery.tables.updateData'
    ]
    assert write_permissions(replace=True) == [
        'bigquery.tables.get', 'bigquery.tables.update',
        'bigquery.tables.updateData'
    ]


def test_load_job_config_without_schema_autodetects():
    config = load_job_config(replace=True)

    assert config.source_format == bigquery.SourceFormat.PARQUET
    assert config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert config.autodetect is True
    assert config.parquet_options.enable_list_inference is True


def test_load_job_config_with_schema_pins_it():
    schema = [bigquery.SchemaField('id', 'INTEGER')]

    config = load_job_config(replace=False, schema=schema)

    assert config.write_disposition == bigquery.WriteDisposition.WRITE_APPEND
    assert config.autodetect is False
    assert config.schema == schema


def test_conformed_closes_and_removes_its_rewrites(tmp_path):
    shards = []
    for index in range(32):
        path = tmp_path / f'part-{index}.parquet'
        pq.write_table(pa.table({'x': pa.array([index], pa.int32())}), path)
        shards.append(path)
    target = plan(shards, None, replace=True, ref='p.d.t').target
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    before = len(os.listdir('/proc/self/fd'))

    for _ in range(4):
        for path in shards:
            with conformed(path, target, None, str(scratch)) as out:
                assert out != path
    with pytest.raises(RuntimeError):
        with conformed(shards[0], target, None, str(scratch)):
            raise RuntimeError

    assert len(os.listdir('/proc/self/fd')) <= before
    assert list(scratch.iterdir()) == []
