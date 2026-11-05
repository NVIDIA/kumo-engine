# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create reference controls from current rows with bounded disk-backed state."""

import itertools
import json
import math
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

BATCH_ROWS = 65536
ANSWER_FIELDS = {
    'TARGET1', 'future_labels', '__kumo_internal_frequency_column__'
}
TASKS = {
    'binary', 'multiclass', 'multilabel', 'multilabel_ranking', 'regression',
    'forecasting', 'static_link', 'temporal_link'
}


def validate_recipe(task, recipe):
    """Reject unsupported control recipes before reading source data."""
    method = ('entity-trajectory-permutation-v1'
              if task == 'static_link' else 'stage-permutation-v1')
    if (task not in TASKS or not isinstance(recipe, dict)
            or set(recipe) != {'method', 'seed'} or recipe['method'] != method
            or type(recipe['seed']) is not int or recipe['seed'] != 0):
        raise ValueError(
            'Use the explicit supported seed-zero control recipe for this task')


def _targets(rows):
    for row in rows:
        targets = row['TARGET1']
        if not isinstance(targets, list) or any(
                type(value) is not int for value in targets):
            raise ValueError(
                'Static control targets must be complete integer ID lists')
        for target in set(targets):
            yield row['ENTITY1'], json.dumps(target), row['SPLIT']


def _disjoint(connection, table):
    connection.execute(
        f'CREATE INDEX {table}_keys ON {table}(entity,target,stage)')
    if connection.execute(
            f'SELECT 1 FROM {table} a JOIN {table} b '
            'ON a.entity=b.entity AND a.target=b.target AND a.stage<b.stage LIMIT 1'
    ).fetchone():
        raise ValueError('Static control targets overlap across stages')


def _permuted(connection, query, parameters, count, work, rng):
    """Shuffle disk-backed IDs, then pair them with the same ordered recipients."""
    import numpy as np

    if shutil.disk_usage(work).free < count * 8 + 64 * 1024**2:
        raise ValueError('Control permutation exceeds free scratch capacity')
    path = work / 'permutation.bin'
    order = np.memmap(path, mode='w+', dtype=np.int64, shape=(count, ))
    try:
        offset = 0
        cursor = connection.execute(query, parameters)
        while batch := cursor.fetchmany(BATCH_ROWS):
            order[offset:offset + len(batch)] = [row[0] for row in batch]
            offset += len(batch)
        if offset != count:
            raise ValueError('Control group population changed')
        rng.shuffle(order)
        offset = 0
        cursor = connection.execute(query, parameters)
        while batch := cursor.fetchmany(BATCH_ROWS):
            yield [(row[0], int(donor))
                   for row, donor in zip(batch, order[offset:offset +
                                                      len(batch)], strict=True)]
            offset += len(batch)
    finally:
        del order
        path.unlink()


def _donors(connection, task, work, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    if task == 'static_link':
        connection.execute(
            'CREATE TABLE entities(entity INTEGER PRIMARY KEY,signature BLOB NOT NULL)'
        )
        cursor = connection.execute(
            'SELECT entity,stage,stamp FROM rows ORDER BY entity,stage,stamp')
        for entity, members in itertools.groupby(cursor,
                                                 key=lambda row: row[0]):
            signature = b''.join(
                bytes([stage]) + stamp for _, stage, stamp in members)
            connection.execute('INSERT INTO entities VALUES (?,?)',
                               (entity, signature))
        connection.execute(
            'CREATE INDEX entity_signature ON entities(signature,entity)')
        connection.execute(
            'CREATE TABLE entity_donors(recipient INTEGER PRIMARY KEY,donor INTEGER UNIQUE NOT NULL)'
        )
        groups = connection.execute(
            'SELECT signature,COUNT(*) FROM entities GROUP BY signature ORDER BY signature'
        )
        for signature, count in groups:
            query = 'SELECT entity FROM entities WHERE signature=? ORDER BY entity'
            for pairs in _permuted(connection, query, (signature, ), count,
                                   work, rng):
                connection.executemany('INSERT INTO entity_donors VALUES (?,?)',
                                       pairs)
        connection.execute(
            'INSERT INTO donors SELECT r.position,d.position FROM rows r '
            'JOIN entity_donors e ON e.recipient=r.entity '
            'JOIN rows d ON d.entity=e.donor AND d.stage=r.stage AND d.stamp=r.stamp'
        )
    else:
        columns = ('stage,stamp,position'
                   if task == 'temporal_link' else 'stage,position')
        connection.execute(f'CREATE INDEX row_groups ON rows({columns})')
        for stage in (0, 1, 2):
            groups = (connection.execute(
                'SELECT stamp,COUNT(*) FROM rows WHERE stage=? '
                'GROUP BY stamp ORDER BY stamp',
                (stage, )) if task == 'temporal_link' else
                      [(None,
                        connection.execute(
                            'SELECT COUNT(*) FROM rows WHERE stage=?',
                            (stage, )).fetchone()[0])])
            for stamp, count in groups:
                query = 'SELECT position FROM rows WHERE stage=?'
                parameters = (stage, )
                if task == 'temporal_link':
                    query += ' AND stamp=?'
                    parameters += (stamp, )
                for pairs in _permuted(connection, query + ' ORDER BY position',
                                       parameters, count, work, rng):
                    connection.executemany('INSERT INTO donors VALUES (?,?)',
                                           pairs)
    connection.commit()


def _batches(files, schema, work, helpers, ordered):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if ordered:
        for path in files:
            with pq.ParquetFile(path) as reader:
                if (not schema.equals(reader.schema_arrow, check_metadata=False)
                        or len(set(schema.names)) != len(schema.names)):
                    raise ValueError('Control row schemas differ')
                yield from reader.iter_batches(batch_size=BATCH_ROWS,
                                               use_threads=False)
        return
    keys = [('ENTITY1', 'ascending')]
    if 'TIMESTAMP' in schema.names:
        keys.append(('TIMESTAMP', 'ascending'))
    keys.append(('SPLIT', 'ascending'))
    for frame in helpers.ordered_batches(files, work / 'sort', ranked=False,
                                         sort_keys=keys):
        yield from pa.Table.from_pandas(
            frame, schema=schema,
            preserve_index=False).to_batches(max_chunksize=BATCH_ROWS)


def generate(files, task, directory, recipe, *, helpers, ordered=False):
    """Move complete answer tuples under one explicit, seed-zero control recipe."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    validate_recipe(task, recipe)
    if not files:
        raise ValueError('Control generation needs current training rows')
    schema = pq.read_schema(files[0]).remove_metadata()
    names = set(schema.names)
    if (not {'ENTITY1', 'SPLIT', 'TARGET1'} <= names
            or not pa.types.is_integer(schema.field('ENTITY1').type)
            or not pa.types.is_integer(schema.field('SPLIT').type)):
        raise ValueError(
            'Control rows need integer entity and split keys and target answers'
        )
    if 'TIMESTAMP' in names and not pa.types.is_timestamp(
            schema.field('TIMESTAMP').type):
        raise ValueError('Control timestamps must retain their timestamp type')
    if task == 'temporal_link' and 'TIMESTAMP' not in names:
        raise ValueError('Temporal controls need current timestamp anchors')
    if task == 'forecasting' and 'future_labels' not in names:
        raise ValueError('Forecast controls need complete future answers')
    if task == 'static_link' and 'exclude_target_keys' in names:
        raise ValueError(
            'Exported exclusions need a reviewed static control conversion')
    fields = sorted(names & ANSWER_FIELDS)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    outputs = []
    with TemporaryDirectory(prefix='.controls-', dir=directory) as temporary:
        work = Path(temporary)
        with closing(sqlite3.connect(work / 'answers.sqlite')) as connection:
            for pragma in ('journal_mode=OFF', 'synchronous=OFF',
                           'temp_store=FILE', 'cache_size=-32768'):
                connection.execute('PRAGMA ' + pragma)
            connection.executescript(
                'CREATE TABLE rows(position INTEGER PRIMARY KEY,entity INTEGER NOT NULL,'
                'stage INTEGER NOT NULL,stamp BLOB NOT NULL,answers TEXT NOT NULL,'
                'UNIQUE(entity,stage,stamp));'
                'CREATE TABLE donors(recipient INTEGER PRIMARY KEY,donor INTEGER UNIQUE NOT NULL);'
            )
            if task == 'static_link':
                connection.execute(
                    'CREATE TABLE normal_targets(entity INTEGER,target TEXT,stage INTEGER)'
                )
                connection.execute(
                    'CREATE TABLE control_targets(entity INTEGER,target TEXT,stage INTEGER)'
                )
            count, stages = 0, set()
            canonical = work / 'current.parquet'
            with pq.ParquetWriter(canonical, schema) as writer:
                for batch in _batches(files, schema, work, helpers, ordered):
                    block = pa.Table.from_batches([batch], schema=schema)
                    rows = block.to_pylist()
                    stamps = (pc.cast(
                        pc.cast(block['TIMESTAMP'], pa.timestamp('ns')),
                        pa.int64()).to_pylist()
                              if 'TIMESTAMP' in names else [None] * len(rows))
                    values = []
                    for row, stamp in zip(rows, stamps, strict=True):
                        entity, stage = row['ENTITY1'], row['SPLIT']
                        if (type(entity) is not int
                                or not -(1 << 63) <= entity < 1 << 63
                                or type(stage) is not int
                                or stage not in (0, 1, 2)):
                            raise ValueError(
                                'Control rows need valid integer entity and split keys'
                            )
                        if task == 'temporal_link' and stamp is None:
                            raise ValueError(
                                'Temporal control anchors must be nonnull')
                        if task == 'forecasting':
                            future = row['future_labels']
                            if (not isinstance(future, list)
                                    or len(future) != 12 or any(
                                        type(value) not in
                                        (int, float) or not math.isfinite(value)
                                        for value in [row['TARGET1'], *future])
                                ):
                                raise ValueError(
                                    'Forecast answers require twelve finite future values'
                                )
                        stamp_key = (b'\0' +
                                     bytes(8) if stamp is None else b'\1' +
                                     (stamp + (1 << 63)).to_bytes(8, 'big'))
                        answer = json.dumps(
                            {field: row[field]
                             for field in fields}, sort_keys=True,
                            separators=(',', ':'), allow_nan=False)
                        values.append((count, entity, stage, stamp_key, answer))
                        count += 1
                        stages.add(stage)
                    try:
                        connection.executemany(
                            'INSERT INTO rows VALUES (?,?,?,?,?)', values)
                    except sqlite3.IntegrityError as error:
                        raise ValueError(
                            'Duplicate current entity, time, or split keys'
                        ) from error
                    if task == 'static_link':
                        connection.executemany(
                            'INSERT INTO normal_targets VALUES (?,?,?)',
                            _targets(rows))
                    writer.write_table(block)
                    connection.commit()
            if stages != {0, 1, 2}:
                raise ValueError(
                    'Control generation needs all three nonempty stages')
            if task == 'static_link':
                _disjoint(connection, 'normal_targets')
            _donors(connection, task, work, recipe['seed'])
            if connection.execute(
                    'SELECT COUNT(*) FROM donors').fetchone()[0] != count:
                raise ValueError(
                    'Control donor population differs from current rows')
            offset = 0
            with pq.ParquetFile(canonical) as source:
                for batch in source.iter_batches(batch_size=BATCH_ROWS,
                                                 use_threads=False):
                    selected = connection.execute(
                        'SELECT a.answers FROM donors d JOIN rows a ON a.position=d.donor '
                        'WHERE d.recipient>=? AND d.recipient<? ORDER BY d.recipient',
                        (offset, offset + len(batch))).fetchall()
                    if len(selected) != len(batch):
                        raise ValueError(
                            'Control answer population differs from current rows'
                        )
                    answers = [json.loads(value) for value, in selected]
                    block = pa.Table.from_batches([batch], schema=schema)
                    for field in fields:
                        block = block.set_column(
                            schema.get_field_index(field), field,
                            pa.array([row[field] for row in answers],
                                     type=schema.field(field).type))
                    if task == 'static_link':
                        connection.executemany(
                            'INSERT INTO control_targets VALUES (?,?,?)',
                            _targets(block.to_pylist()))
                    path = directory / f'part-{len(outputs):07d}.parquet'
                    pq.write_table(block, path)
                    outputs.append(str(path))
                    offset += len(batch)
            if task == 'static_link':
                _disjoint(connection, 'control_targets')
    return outputs
