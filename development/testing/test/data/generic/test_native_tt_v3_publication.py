# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import os.path as osp
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.data.generic.diskgraph_train_table as diskgraph_train_table
import kumo.data.generic.generate_train_table as generate
from io_utils import fsutil


def _write_source_bundle(root, tt_id='job', entities=(0, 1), part_rows=None):
    staging = root / f'.{tt_id}.staging'
    staging.mkdir()
    train_table = staging / 'tt.parquet'
    train_table.mkdir()
    row_count = len(entities)
    part_rows = part_rows or (row_count,)
    table = pa.table({
        'ENTITY1': pa.array(entities, type=pa.int64()),
        'TIMESTAMP': pa.array([0] * row_count, type=pa.int64()),
        'TARGET1': pa.array([1] * row_count, type=pa.int64()),
    })
    parts = []
    offset = 0
    for index, rows in enumerate(part_rows):
        part = train_table / f'part-{index:05}.parquet'
        pq.write_table(table.slice(offset, rows), part)
        parts.append({
            'path': f'tt.parquet/{part.name}',
            'row_count': rows,
            'size_bytes': part.stat().st_size,
        })
        offset += rows
    stats = staging / 'stats.json'
    stats.write_text(json.dumps({
        'version': 2,
        'kind': 'tt_stats',
        'num_total_rows': row_count,
        'num_train_rows': row_count,
        'num_entities': row_count,
        'num_targets': 1,
        'num_links': None,
    }))
    manifest = staging / 'manifest.json'
    manifest.write_text(json.dumps({
        'version': 3,
        'row_count': row_count,
        'train_table': {
            'path': 'tt.parquet',
            'format': 'tt_parquet_dataset_v1',
            'row_count': row_count,
            'size_bytes': sum(part['size_bytes'] for part in parts),
            'parts': parts,
        },
        'stats': {'path': 'stats.json', 'format': 'tt_stats_json'},
        'checksums': {
            'stats_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
        },
    }))
    (staging / 'tt_ingest_spec.json').write_text(json.dumps({
        'schema_version': 'diskgraph.tt_ingest_spec.v3',
        'tt_id': tt_id,
        'split_stages': None,
    }))
    return staging, train_table, manifest


def _configure_fake_generation(monkeypatch, train_table, manifest):
    class FakeGenerator:
        def __init__(self):
            self._diskgraph_last_output_path = str(train_table)
            self._diskgraph_last_manifest_path = str(manifest)
            self._diskgraph_staging_dir = None

        def generate_table(self, **_kwargs):
            return None

        def get_validate_train_table_response(self):
            from kumo.validation import ValidationResponse
            return ValidationResponse()

    generator = FakeGenerator()

    class FakePredictiveQuery:
        list_distinct_target_is_array = False

        def get_table_generator(self):
            return generator

        @classmethod
        def from_config(cls, _config):
            return cls()

    monkeypatch.setattr(generate, 'TrainingTableGeneratorV2', FakeGenerator)
    import kumo.pquery as pquery
    monkeypatch.setattr(pquery, 'PredictiveQueryV2', FakePredictiveQuery)
    return SimpleNamespace(graph=SimpleNamespace(tables=[]))


def _generate(config, output):
    return generate.generate_train_table_generic(
        config,
        str(output),
        destination='unused',
        job_id='job',
        diskgraph_index_path='unused',
    )


def test_generation_returns_selected_physical_train_table(tmp_path, monkeypatch):
    staging, train_table, manifest = _write_source_bundle(tmp_path)
    config = _configure_fake_generation(monkeypatch, train_table, manifest)

    physical = _generate(config, tmp_path / 'output')

    selected = osp.dirname(physical)
    assert selected.startswith(
        str(tmp_path / 'output.ttng_staging' / '_bundles'))
    assert osp.basename(physical) == 'tt.parquet'
    assert not (tmp_path / 'output.ttng_staging' / '_current.json').exists()
    diskgraph_train_table.assert_ttng_bundle_complete(selected)


def test_generation_copies_complete_multipart_dataset(tmp_path, monkeypatch):
    _staging, train_table, manifest = _write_source_bundle(
        tmp_path, entities=(0, 1, 2, 3, 4), part_rows=(2, 3))
    config = _configure_fake_generation(monkeypatch, train_table, manifest)
    original_copy = fsutil.copy_tree_threaded
    copies = []

    def copy_tree(source, destination, **kwargs):
        copies.append((source, destination))
        return original_copy(source, destination, **kwargs)

    monkeypatch.setattr(fsutil, 'copy_tree_threaded', copy_tree)

    physical = _generate(config, tmp_path / 'output')

    assert len(copies) == 1
    assert copies[0][0] == str(train_table.parent)
    assert sorted(os.listdir(physical)) == [
        'part-00000.parquet', 'part-00001.parquet']
    assert pq.read_table(physical).num_rows == 5
    diskgraph_train_table.assert_ttng_bundle_complete(
        osp.dirname(physical))


def test_source_bundle_is_validated_before_copy(tmp_path, monkeypatch):
    _staging, train_table, manifest = _write_source_bundle(tmp_path)
    config = _configure_fake_generation(monkeypatch, train_table, manifest)
    with open(train_table / 'part-00000.parquet', 'r+b') as handle:
        handle.seek(-4, os.SEEK_END)
        handle.write(b'NOPE')

    monkeypatch.setattr(
        fsutil,
        'copy_tree_threaded',
        lambda *_args, **_kwargs: pytest.fail(
            'copy ran before source validation'),
    )

    with pytest.raises(RuntimeError, match='footer/schema is invalid'):
        _generate(config, tmp_path / 'output')


def test_interrupted_copy_is_never_selected(tmp_path, monkeypatch):
    _staging, train_table, manifest = _write_source_bundle(tmp_path)
    config = _configure_fake_generation(monkeypatch, train_table, manifest)
    output = tmp_path / 'output'
    original_copy = fsutil.copy_tree_threaded

    def interrupted_copy(_source, destination, **_kwargs):
        os.makedirs(destination, exist_ok=True)
        os.makedirs(osp.join(destination, 'tt.parquet'))
        with open(osp.join(destination, 'tt.parquet',
                           'part-00000.parquet'), 'wb') as handle:
            handle.write(b'partial')
        raise RuntimeError('interrupted immutable copy')

    monkeypatch.setattr(fsutil, 'copy_tree_threaded', interrupted_copy)
    with pytest.raises(RuntimeError, match='interrupted immutable copy'):
        _generate(config, output)
    assert not (tmp_path / 'output.ttng_staging' / '_current.json').exists()

    monkeypatch.setattr(fsutil, 'copy_tree_threaded', original_copy)
    physical = _generate(config, output)
    assert pq.read_table(physical).num_rows == 2


def test_retry_selects_independent_immutable_attempts(tmp_path, monkeypatch):
    _staging, train_table, manifest = _write_source_bundle(tmp_path)
    config = _configure_fake_generation(monkeypatch, train_table, manifest)
    output = tmp_path / 'output'

    first = _generate(config, output)
    second = _generate(config, output)

    assert first != second
    assert pq.read_table(first).num_rows == 2
    assert pq.read_table(second).num_rows == 2


def test_publication_does_not_full_read_or_hash_parquet(tmp_path, monkeypatch):
    _staging, train_table, manifest = _write_source_bundle(tmp_path)
    config = _configure_fake_generation(monkeypatch, train_table, manifest)
    import kumo.diskgraph.diskgraph_native_tt_artifacts as artifacts

    original_read_binary = artifacts.fsutil.read_binary
    original_sha256_file = artifacts._sha256_file
    original_sha256_path = artifacts._sha256_path

    def guard(path):
        if f'{os.sep}tt.parquet{os.sep}' in str(path):
            pytest.fail('large train table reached a full-read/hash helper')

    def read_binary(path, *args, **kwargs):
        guard(path)
        return original_read_binary(path, *args, **kwargs)

    def sha256_file(path):
        guard(path)
        return original_sha256_file(path)

    def sha256_path(path):
        guard(path)
        return original_sha256_path(path)

    monkeypatch.setattr(artifacts.fsutil, 'read_binary', read_binary)
    monkeypatch.setattr(artifacts, '_sha256_file', sha256_file)
    monkeypatch.setattr(artifacts, '_sha256_path', sha256_path)

    physical = _generate(config, tmp_path / 'output')
    assert pq.read_table(physical).num_rows == 2


def test_native_tt_path_cannot_reach_legacy_full_parquet_hash_bridge():
    for module in (generate, diskgraph_train_table):
        with open(module.__file__) as source:
            live_source = source.read()
        assert '_diskgraph_legacy_graphstore_bridge' not in live_source
        assert 'train_table_sha256' not in live_source
