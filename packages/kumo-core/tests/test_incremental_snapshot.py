# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.diskgraph.incremental import detect_append, fingerprint_files


def test_content_edit_with_restored_size_and_mtime(tmp_path):
    source = tmp_path / 'events.csv'
    source.write_text('id,value\n1,100\n')
    before = fingerprint_files('events', source)
    stat = source.stat()
    source.write_text('id,value\n1,200\n')
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = fingerprint_files('events', source)
    assert before.schema_hash == after.schema_hash
    assert before.content_hash != after.content_hash
    assert after.row_count is None


def test_parquet_footer_and_partition_changes(tmp_path):
    pq.write_table(pa.table({'id': [1, 2]}), tmp_path / '0.parquet')
    before = fingerprint_files('events', tmp_path)
    assert before.row_count == 2
    assert before == fingerprint_files('events', tmp_path)
    pq.write_table(pa.table({'id': [3]}), tmp_path / '1.parquet')
    after = fingerprint_files('events', tmp_path)
    assert after.row_count == 3
    assert after.schema_hash == before.schema_hash
    assert after.content_hash != before.content_hash
    assert after.partition_fingerprints[
        '0.parquet'] == before.partition_fingerprints['0.parquet']
    (tmp_path / '0.parquet').unlink()
    assert fingerprint_files('events', tmp_path).row_count == 1


def test_schema_change(tmp_path):
    path = tmp_path / 'events.parquet'
    pq.write_table(pa.table({'id': [1]}), path)
    before = fingerprint_files('events', path)
    pq.write_table(pa.table({'id': [1], 'value': [2]}), path)
    assert fingerprint_files('events', path).schema_hash != before.schema_hash


def test_concurrent_edit_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'events.parquet'
    pq.write_table(pa.table({'id': [1]}), path)
    original = pq.read_metadata

    def replace_source(path):
        result = original(path)
        pq.write_table(pa.table({'id': [1, 2]}), path)
        return result

    monkeypatch.setattr(pq, 'read_metadata', replace_source)
    with pytest.raises(RuntimeError, match='Source changed'):
        fingerprint_files('events', path)


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='No CSV or Parquet'):
        fingerprint_files('events', tmp_path)


def test_renamed_partition_changes_content_identity(tmp_path):
    first = tmp_path / '0.parquet'
    pq.write_table(pa.table({'id': [1]}), first)
    before = fingerprint_files('events', tmp_path)
    first.rename(tmp_path / '1.parquet')
    after = fingerprint_files('events', tmp_path)
    assert before.schema_hash == after.schema_hash
    assert before.content_hash != after.content_hash


def test_csv_append_requires_unchanged_prefix(tmp_path):
    path = tmp_path / 'events.csv'
    path.write_text('id,time,value\n1,2026-01-02,100\n')
    before = fingerprint_files('events', path)
    with path.open('a') as stream:
        stream.write('2,2026-01-01,200\n')  # Late arrival is still an append.
    delta = detect_append(path, before, fingerprint_files('events', path))
    assert len(delta) == 1
    assert delta[0].byte_offset == before.partition_sizes[path.name]
    path.write_text(path.read_text().replace('100', '999'))
    assert detect_append(path, before, fingerprint_files('events',
                                                         path)) is None


def test_csv_unterminated_record_is_not_append(tmp_path):
    path = tmp_path / 'events.csv'
    path.write_text('id,value\n1,100')
    before = fingerprint_files('events', path)
    with path.open('a') as stream:
        stream.write('1\n2,200\n')
    assert detect_append(path, before, fingerprint_files('events',
                                                         path)) is None


def test_parquet_append_and_historical_edit(tmp_path):
    path = tmp_path / 'events.parquet'
    first = pa.table({'id': [1, 2], 'value': [100, 200]})
    pq.write_table(first, path, row_group_size=2)
    before = fingerprint_files('events', path)
    second = pa.table({'id': [3], 'value': [300]})
    pq.write_table(pa.concat_tables([first, second]), path, row_group_size=2)
    delta = detect_append(path, before, fingerprint_files('events', path))
    assert len(delta) == 1
    assert delta[0].row_group == 1
    edited = pa.table({'id': [1, 2, 3], 'value': [999, 200, 300]})
    pq.write_table(edited, path, row_group_size=2)
    assert detect_append(path, before, fingerprint_files('events',
                                                         path)) is None


def test_partition_append_removal_and_recompression(tmp_path):
    table = pa.table({'id': [1, 2]})
    path = tmp_path / '0.parquet'
    pq.write_table(table, path)
    before = fingerprint_files('events', tmp_path)
    assert detect_append(tmp_path, before, before) == []
    pq.write_table(pa.table({'id': [3]}), tmp_path / '1.parquet')
    delta = detect_append(tmp_path, before,
                          fingerprint_files('events', tmp_path))
    assert [part.path for part in delta] == [str(tmp_path / '1.parquet')]
    pq.write_table(table, path, compression='gzip')
    assert detect_append(tmp_path, before,
                         fingerprint_files('events', tmp_path)) is None
    path.unlink()
    assert detect_append(tmp_path, before,
                         fingerprint_files('events', tmp_path)) is None


def test_encoder_drift_warns_without_mutating_stats():
    from kumo_core.diskgraph.incremental import warn_encoder_drift
    old = {
        'price__kumo.numerical__MIN__False': {
            'value': 10.
        },
        'price__kumo.numerical__MAX__False': {
            'value': 20.
        },
        'kind__kumo.categorical__CATEGORY_COUNTS__False': {
            'value': [('old', 10)]
        }
    }
    new = {
        'price__kumo.numerical__MIN__False': {
            'value': 15.
        },
        'price__kumo.numerical__MAX__False': {
            'value': 100.
        },
        'kind__kumo.categorical__CATEGORY_COUNTS__False': {
            'value': [('new', 1)]
        }
    }
    with pytest.warns(UserWarning) as messages:
        warn_encoder_drift('events', old, new)
    assert len(messages) == 2
    assert old['price__kumo.numerical__MAX__False']['value'] == 20.


def test_hidden_files_and_directories_are_not_source_rows(tmp_path):
    pq.write_table(pa.table({'id': [1]}), tmp_path / 'part.parquet')
    before = fingerprint_files('events', tmp_path)
    for name in [
            '_uncommitted.parquet', '.partial.parquet', '_tmp/part.parquet',
            '.tmp/part.parquet'
    ]:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        pq.write_table(pa.table({'id': [999]}), path)
    after = fingerprint_files('events', tmp_path)
    assert after == before
    assert detect_append(tmp_path, before, after) == []
