# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared TT-NG durable-bundle helpers that make train and
predict source the ``tt_id`` symmetrically and gate bundle completeness the
same way (PR: symmetric tt_id keying).

Covers:
  * ``read_tt_id_from_staging_bundle`` — reads the canonical id off the spec.
  * ``assert_ttng_bundle_complete`` — passes on complete (no-split + multi-part
    split) bundles; fails on a half-written bundle / missing declared part.
  * ``persist_ttng_bundle`` — immutable attempt copy for training.
  * ``persist_durable_ttng_bundle`` — stable-root publication for prediction.
  * ``restore_durable_ttng_bundle`` — interrupted local copies heal on retry.
"""
import hashlib
import json
import os
import os.path as osp
import shutil
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.data.generic.diskgraph_train_table as ttng
from io_utils.fsutil.api import PathDetail
from kumo.data.generic.diskgraph_train_table import (
    TTNGBundleError,
    assert_ttng_bundle_complete,
    persist_durable_ttng_bundle,
    read_tt_id_from_staging_bundle,
    resolve_durable_ttng_bundle,
    restore_durable_ttng_bundle,
    ttng_bundle_dir_for_train_table,
)

_INGEST_SPEC_V2 = 'diskgraph.tt_ingest_spec.v2'
_INGEST_SPEC_V3 = 'diskgraph.tt_ingest_spec.v3'


def _write(path, content=b'x'):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(content)


def _write_spec(d, spec):
    os.makedirs(d, exist_ok=True)
    with open(osp.join(d, 'tt_ingest_spec.json'), 'w') as f:
        json.dump(spec, f)


def _bundle(tmp_path, name, *, tt_id=None, content=None):
    del content
    tt_id = tt_id or name
    path = str(tmp_path / f'.{tt_id}.staging')
    os.makedirs(path, exist_ok=True)
    table = pa.table({
        'ENTITY1': pa.array([0, 1], type=pa.int64()),
        'TIMESTAMP': pa.array([0, 0], type=pa.int64()),
        'TARGET1': pa.array([1, 1], type=pa.int64()),
    })
    train_table = osp.join(path, 'tt.parquet')
    os.makedirs(train_table)
    part = osp.join(train_table, 'part-00000.parquet')
    pq.write_table(table, part)
    train_table_manifest = {
        'path': 'tt.parquet',
        'format': 'tt_parquet_dataset_v1',
        'row_count': 2,
        'size_bytes': os.stat(part).st_size,
        'parts': [{
            'path': 'tt.parquet/part-00000.parquet',
            'row_count': 2,
            'size_bytes': os.stat(part).st_size,
        }],
    }
    stats = {
        'version': 2,
        'kind': 'tt_stats',
        'num_total_rows': 2,
        'num_train_rows': 2,
        'num_entities': 2,
        'num_targets': 1,
        'num_links': None,
    }
    stats_path = osp.join(path, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    with open(stats_path, 'rb') as f:
        stats_sha256 = hashlib.sha256(f.read()).hexdigest()
    with open(osp.join(path, 'manifest.json'), 'w') as f:
        json.dump(
            {
                'version': 3,
                'row_count': 2,
                'train_table': train_table_manifest,
                'stats': {
                    'path': 'stats.json',
                    'format': 'tt_stats_json'
                },
                'checksums': {
                    'stats_sha256': stats_sha256
                },
            }, f)
    _write_spec(path, {
        'schema_version': _INGEST_SPEC_V3,
        'tt_id': tt_id,
        'split_stages': None,
    })
    return path


# --------------------------------------------------------------------------- #
# read_tt_id_from_staging_bundle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('tt_id', ['job-abc', 'pred_gen-job'])
def test_read_tt_id_from_spec(tmp_path, tt_id):
    d = str(tmp_path / 'bundle')
    _write_spec(d, {
        'schema_version': _INGEST_SPEC_V3,
        'tt_id': tt_id,
        'split_stages': None,
    })
    assert read_tt_id_from_staging_bundle(d) == tt_id


@pytest.mark.parametrize('spec, message', [(None, 'tt_ingest_spec.json'),
                                           ({
                                               'schema_version': _INGEST_SPEC_V3,
                                               'split_stages': None
                                           }, 'tt_id')])
def test_read_tt_id_rejects_incomplete_spec(tmp_path, spec, message):
    d = str(tmp_path / 'bundle')
    if spec is not None:
        _write_spec(d, spec)
    with pytest.raises(TTNGBundleError, match=message):
        read_tt_id_from_staging_bundle(d)


@pytest.mark.parametrize(
    'schema_version',
    [None, 'diskgraph.tt_ingest_spec.v1', _INGEST_SPEC_V2],
)
def test_read_tt_id_rejects_non_v3_spec(tmp_path, schema_version):
    d = str(tmp_path / 'bundle')
    spec = {'tt_id': 'job-abc', 'split_stages': None}
    if schema_version is not None:
        spec['schema_version'] = schema_version
    _write_spec(d, spec)

    with pytest.raises(TTNGBundleError, match='must use'):
        read_tt_id_from_staging_bundle(d)


# --------------------------------------------------------------------------- #
# assert_ttng_bundle_complete
# --------------------------------------------------------------------------- #
def test_complete_no_split_bundle_passes(tmp_path):
    """Predict / no-split shape: tt.parquet + spec (split_stages None)."""
    d = _bundle(tmp_path, 'pred_bp-job')
    assert_ttng_bundle_complete(d)  # no raise


def test_complete_multi_part_split_bundle_passes(tmp_path):
    """Split (train) shape: each stage declares one or more part files; a
    multi-part stage (future rollover) must pass when every part exists."""
    d = _bundle(tmp_path, 'job-xyz')
    parts = [
        'train_seeds/part-00000.parquet',
        'train_seeds/part-00001.parquet',
        'val_seeds/part-00000.parquet',
    ]
    for p in parts:
        path = osp.join(d, p)
        os.makedirs(osp.dirname(path), exist_ok=True)
        pq.write_table(pa.table({'ENTITY1': [1]}), path)

    def inventory(stage_parts):
        return {
            'schema_version': 'diskgraph.tt_stage_artifact_inventory.v1',
            'parts': [{
                'path': part,
                'size_bytes': os.stat(osp.join(d, part)).st_size,
                'row_count': 1,
            } for part in stage_parts],
        }
    _write_spec(
        d, {
            'schema_version': _INGEST_SPEC_V3,
            'tt_id':
            'job-xyz',
            'split_stages': [
                {
                    'key': 'train',
                    'table_name': 'train_seeds',
                    'parts': parts[:2],
                    'row_count': 2,
                    'artifact_inventory': inventory(parts[:2]),
                },
                {
                    'key': 'val',
                    'table_name': 'val_seeds',
                    'parts': parts[2:],
                    'row_count': 1,
                    'artifact_inventory': inventory(parts[2:]),
                },
            ],
        })
    assert_ttng_bundle_complete(d)  # no raise


@pytest.mark.parametrize('missing, message', [
    ('spec', 'tt_ingest_spec.json'),
    ('parquet', 'tt.parquet'),
])
def test_incomplete_bundle_fails(tmp_path, missing, message):
    d = _bundle(tmp_path, 'incomplete')
    missing_path = osp.join(
        d, 'tt.parquet' if missing == 'parquet' else 'tt_ingest_spec.json')
    if missing == 'parquet':
        shutil.rmtree(missing_path)
    else:
        os.unlink(missing_path)
    with pytest.raises(TTNGBundleError, match=message):
        assert_ttng_bundle_complete(d)


def test_missing_declared_stage_part_fails(tmp_path):
    """A half-written split bundle: the spec declares two parts but only one is
    on disk -> the gate must reject it (this is the A4 latent-bug fix)."""
    d = _bundle(tmp_path, 'job-xyz')
    part = osp.join(d, 'train_seeds/part-00000.parquet')
    os.makedirs(osp.dirname(part), exist_ok=True)
    pq.write_table(pa.table({'ENTITY1': [1]}), part)
    # part-00001 declared but NOT written.
    _write_spec(
        d, {
            'schema_version': _INGEST_SPEC_V3,
            'tt_id':
            'job-xyz',
            'split_stages': [{
                'key':
                'train',
                'table_name':
                'train_seeds',
                'parts': [
                    'train_seeds/part-00000.parquet',
                    'train_seeds/part-00001.parquet',
                ],
                'row_count':
                5,
                'artifact_inventory': {
                    'schema_version':
                    'diskgraph.tt_stage_artifact_inventory.v1',
                    'parts': [{
                        'path': 'train_seeds/part-00000.parquet',
                        'size_bytes': os.stat(part).st_size,
                        'row_count': 1,
                    }, {
                        'path': 'train_seeds/part-00001.parquet',
                        'size_bytes': 1,
                        'row_count': 4,
                    }],
                },
            }],
        })
    with pytest.raises(TTNGBundleError, match='part-00001'):
        assert_ttng_bundle_complete(d)


@pytest.mark.parametrize('spec', [[], {'split_stages': ['not-an-object']}])
def test_malformed_ingest_spec_shape_fails_closed(tmp_path, spec):
    d = _bundle(tmp_path, 'malformed')
    if isinstance(spec, dict):
        spec['schema_version'] = _INGEST_SPEC_V3
        spec['tt_id'] = 'malformed'
    _write_spec(d, spec)

    with pytest.raises(TTNGBundleError, match='JSON object|list of objects'):
        assert_ttng_bundle_complete(d)


def test_v3_bundle_rejects_readable_part_row_truncation(tmp_path):
    d = _bundle(tmp_path, 'bundle')
    truncated = pa.table({
        'ENTITY1': pa.array([0], type=pa.int64()),
        'TIMESTAMP': pa.array([0], type=pa.int64()),
        'TARGET1': pa.array([1], type=pa.int64()),
    })
    pq.write_table(truncated, osp.join(d, 'tt.parquet', 'part-00000.parquet'))

    with pytest.raises(TTNGBundleError, match='artifact manifest'):
        assert_ttng_bundle_complete(d)


@pytest.mark.parametrize('artifact', [
    'tt_ingest_spec.json',
    'tt.parquet',
    'manifest.json',
])
def test_bundle_rejects_symlink_top_level_artifact(tmp_path, artifact):
    d = _bundle(tmp_path, f'bundle-{artifact.replace(".", "-")}')
    artifact_path = tmp_path / 'external' / artifact
    artifact_path.parent.mkdir(exist_ok=True)
    os.replace(osp.join(d, artifact), artifact_path)
    os.symlink(
        artifact_path,
        osp.join(d, artifact),
        target_is_directory=artifact == 'tt.parquet',
    )

    with pytest.raises(TTNGBundleError, match='symlink'):
        assert_ttng_bundle_complete(d)


def test_bundle_rejects_symlink_staging_root(tmp_path):
    d = _bundle(tmp_path, 'bundle-root')
    real_root = tmp_path / 'real-bundle-root'
    os.replace(d, real_root)
    os.symlink(real_root, d, target_is_directory=True)

    with pytest.raises(TTNGBundleError, match='staging root.*non-symlink'):
        assert_ttng_bundle_complete(d)


def test_bundle_rejects_symlink_bundle_directory_escape(tmp_path):
    generated = _bundle(tmp_path, 'bundle-source')
    durable = str(tmp_path / 'durable')
    persist_durable_ttng_bundle(generated, durable)
    external_bundles = tmp_path / 'external-bundles'
    os.replace(osp.join(durable, '_bundles'), external_bundles)
    os.symlink(
        external_bundles,
        osp.join(durable, '_bundles'),
        target_is_directory=True,
    )

    with pytest.raises(TTNGBundleError, match='symlink'):
        assert_ttng_bundle_complete(durable)


def test_bundle_rejects_nonregular_top_level_artifact(tmp_path):
    d = _bundle(tmp_path, 'bundle-nonregular')
    shutil.rmtree(osp.join(d, 'tt.parquet'))
    _write(osp.join(d, 'tt.parquet'))

    with pytest.raises(TTNGBundleError, match='tt.parquet'):
        assert_ttng_bundle_complete(d)


@pytest.mark.parametrize(
    'schema_version',
    [None, 'diskgraph.tt_ingest_spec.v1', _INGEST_SPEC_V2],
)
def test_bundle_rejects_non_v3_ingest_spec(tmp_path, schema_version):
    d = _bundle(tmp_path, 'bundle')
    spec = {'tt_id': 'bundle', 'split_stages': None}
    if schema_version is not None:
        spec['schema_version'] = schema_version
    _write_spec(d, spec)

    with pytest.raises(TTNGBundleError, match='unsupported ingest spec'):
        assert_ttng_bundle_complete(d)


def _v3_stage_spec(bundle, *, declared_size_delta=0, declared_rows=1):
    part = 'train_seeds/part-00000.parquet'
    part_path = osp.join(bundle, part)
    os.makedirs(osp.dirname(part_path), exist_ok=True)
    pq.write_table(
        pa.table({
            'ENTITY1': pa.array([0], type=pa.int64()),
            'TIMESTAMP': pa.array([0], type=pa.int64()),
            'TARGET1': pa.array([1], type=pa.int64()),
        }),
        part_path,
    )
    _write_spec(bundle, {
        'schema_version': _INGEST_SPEC_V3,
        'tt_id': osp.basename(bundle).removeprefix('.').removesuffix('.staging'),
        'split_stages': [{
            'key': 'train',
            'table_name': 'train_seeds',
            'parts': [part],
            'row_count': 1,
            'seed_time': None,
            'artifact_inventory': {
                'schema_version':
                'diskgraph.tt_stage_artifact_inventory.v1',
                'parts': [{
                    'path': part,
                    'size_bytes':
                    os.stat(part_path).st_size + declared_size_delta,
                    'row_count': declared_rows,
                }],
            },
        }],
    })
    return part_path


def test_v3_split_stage_exact_artifact_metadata_passes(tmp_path):
    d = _bundle(tmp_path, 'bundle')
    _v3_stage_spec(d)
    assert_ttng_bundle_complete(d)


def test_v3_split_stage_rejects_symlink_part(tmp_path):
    d = _bundle(tmp_path, 'bundle-symlink-part')
    part_path = _v3_stage_spec(d)
    external = tmp_path / 'external-part.parquet'
    os.replace(part_path, external)
    os.symlink(external, part_path)

    with pytest.raises(TTNGBundleError, match='symlink'):
        assert_ttng_bundle_complete(d)


def test_v3_split_stage_rejects_symlink_directory_escape(tmp_path):
    d = _bundle(tmp_path, 'bundle-symlink-directory')
    _v3_stage_spec(d)
    stage_dir = osp.join(d, 'train_seeds')
    external = tmp_path / 'external-stage'
    os.replace(stage_dir, external)
    os.symlink(external, stage_dir, target_is_directory=True)

    with pytest.raises(TTNGBundleError, match='symlink'):
        assert_ttng_bundle_complete(d)


@pytest.mark.parametrize('damage', ['missing_inventory', 'size', 'rows'])
def test_v3_split_stage_rejects_invalid_artifact_metadata(tmp_path, damage):
    d = _bundle(tmp_path, 'bundle')
    _v3_stage_spec(
        d,
        declared_size_delta=1 if damage == 'size' else 0,
        declared_rows=2 if damage == 'rows' else 1,
    )
    if damage == 'missing_inventory':
        spec_path = osp.join(d, 'tt_ingest_spec.json')
        with open(spec_path) as source:
            spec = json.load(source)
        del spec['split_stages'][0]['artifact_inventory']
        with open(spec_path, 'w') as output:
            json.dump(spec, output)

    with pytest.raises(TTNGBundleError, match='inventory|size|row_count'):
        assert_ttng_bundle_complete(d)


def test_remote_bundle_validation_lists_directories_not_parts(monkeypatch):
    """Large remote bundles need bounded listings, not one probe per part."""
    root = 's3://bucket/bundle'
    seed_dir = f'{root}/train_seeds'
    train_table_dir = f'{root}/tt.parquet'
    parts = [f'train_seeds/part-{i:05}.parquet' for i in range(10_000)]
    spec = {
        'schema_version': _INGEST_SPEC_V3,
        'tt_id':
        'job-large',
        'split_stages': [{
            'key': 'train',
            'table_name': 'train_seeds',
            'parts': parts,
            'row_count': len(parts),
            'artifact_inventory': {
                'schema_version':
                'diskgraph.tt_stage_artifact_inventory.v1',
                'parts': [{
                    'path': part,
                    'size_bytes': 1,
                    'row_count': 1,
                } for part in parts],
            },
        }],
    }
    listings = {
        root: [
            PathDetail(path=f'{root}/tt_ingest_spec.json', is_directory=False),
            PathDetail(path=train_table_dir, is_directory=True),
            PathDetail(path=f'{root}/manifest.json', is_directory=False),
            PathDetail(path=seed_dir, is_directory=True),
        ],
        seed_dir: [
            PathDetail(path=f'{root}/{part}', is_directory=False)
            for part in parts
        ],
        train_table_dir: [
            PathDetail(
                path=f'{train_table_dir}/part-00000.parquet',
                is_directory=False,
            ),
        ],
    }
    list_calls = []

    def listdir(path, *, detail=False, **_kwargs):
        assert detail
        list_calls.append(path)
        return listings[path]

    monkeypatch.setattr(ttng, 'resolve_durable_ttng_bundle', lambda path: path)
    monkeypatch.setattr(ttng.fsutil, 'listdir', listdir)
    monkeypatch.setattr(
        ttng.fsutil,
        'isdir',
        lambda path: path == train_table_dir,
    )
    monkeypatch.setattr(
        ttng.fsutil,
        'read_binary',
        lambda path: json.dumps(spec).encode(),
    )

    def unexpected_exists(_path):
        raise AssertionError('remote validation must not probe each part')

    monkeypatch.setattr(ttng.fsutil, 'exists', unexpected_exists)
    monkeypatch.setattr(ttng, '_validate_ttng_bundle_for_publish',
                        lambda _path: {})
    monkeypatch.setattr(
        'kumo.diskgraph.diskgraph_native_tt_artifacts._inspect_parquet_artifact',
        lambda _path: (1, 1),
    )

    assert_ttng_bundle_complete(root)
    assert list_calls == [root, seed_dir, train_table_dir]


def test_remote_bundle_validation_falls_back_without_detailed_listing(
        monkeypatch):
    root = 's3://bucket/bundle'
    train_table_dir = f'{root}/tt.parquet'
    spec = {
        'schema_version': _INGEST_SPEC_V3,
        'tt_id': 'job',
        'split_stages': None,
    }
    root_paths = [
        f'{root}/tt_ingest_spec.json', train_table_dir,
        f'{root}/manifest.json'
    ]
    train_table_paths = [f'{train_table_dir}/part-00000.parquet']

    monkeypatch.setattr(ttng, 'resolve_durable_ttng_bundle', lambda path: path)
    monkeypatch.setattr(ttng.fsutil, 'is_local', lambda _path: False)

    def listdir(path, *, detail=False, **_kwargs):
        if detail:
            raise NotImplementedError
        if path == root:
            return root_paths
        assert path == train_table_dir
        return train_table_paths

    monkeypatch.setattr(ttng.fsutil, 'listdir', listdir)
    monkeypatch.setattr(
        ttng.fsutil,
        'isdir',
        lambda path: path == train_table_dir,
    )
    monkeypatch.setattr(
        ttng.fsutil,
        'read_binary',
        lambda _path: json.dumps(spec).encode(),
    )
    monkeypatch.setattr(ttng, '_validate_ttng_bundle_for_publish',
                        lambda _path: {})

    assert_ttng_bundle_complete(root)


# --------------------------------------------------------------------------- #
# publication and restore
# --------------------------------------------------------------------------- #
def test_selected_publication_returns_one_physical_bundle(tmp_path):
    source = _bundle(tmp_path, 'attempt')
    durable = str(tmp_path / 'durable')

    selected = ttng.persist_ttng_bundle(source, durable)
    physical = osp.join(selected, 'tt.parquet')

    assert selected.startswith(osp.join(durable, '_bundles'))
    assert not osp.exists(osp.join(durable, '_current.json'))
    assert_ttng_bundle_complete(selected)
    assert ttng_bundle_dir_for_train_table(physical) == selected


def test_selected_publication_is_attempt_isolated(tmp_path):
    source = _bundle(tmp_path, 'attempt')
    durable = str(tmp_path / 'durable')

    with ThreadPoolExecutor(max_workers=2) as executor:
        selected = list(
            executor.map(
                lambda _index: ttng.persist_ttng_bundle(
                    source, durable),
                range(2),
            ))

    assert selected[0] != selected[1]
    assert not osp.exists(osp.join(durable, '_current.json'))
    for bundle in selected:
        assert_ttng_bundle_complete(bundle)


def test_retry_with_changed_same_tt_id_selects_its_own_bundle(tmp_path):
    first = _bundle(tmp_path / 'first', 'attempt')
    second = _bundle(tmp_path / 'second', 'attempt')
    second_spec_path = osp.join(second, 'tt_ingest_spec.json')
    second_spec = json.loads(ttng.fsutil.read_text(second_spec_path))
    second_spec['producer_revision'] = 'changed'
    ttng.fsutil.write_text(second_spec_path, json.dumps(second_spec))
    durable = str(tmp_path / 'durable')

    first_selected = ttng.persist_ttng_bundle(first, durable)
    second_selected = ttng.persist_ttng_bundle(second, durable)

    assert first_selected != second_selected
    assert read_tt_id_from_staging_bundle(first_selected) == 'attempt'
    assert read_tt_id_from_staging_bundle(second_selected) == 'attempt'


def test_durable_publication_selects_bundle_from_stable_root(tmp_path):
    source = _bundle(tmp_path, 'attempt')
    durable = str(tmp_path / 'durable')

    selected = persist_durable_ttng_bundle(source, durable)

    assert resolve_durable_ttng_bundle(durable) == selected
    assert read_tt_id_from_staging_bundle(durable) == 'attempt'
    assert_ttng_bundle_complete(durable)


def test_publication_requires_attempt_scoped_source_name(tmp_path):
    source = _bundle(tmp_path, 'attempt')
    unscoped = str(tmp_path / 'producer-output')
    os.replace(source, unscoped)

    with pytest.raises(TTNGBundleError, match='not attempt-scoped'):
        ttng.persist_ttng_bundle(
            unscoped, str(tmp_path / 'durable'))


def test_publication_accepts_explicit_custom_tt_id(tmp_path):
    source = _bundle(tmp_path, 'attempt', tt_id='custom-tt')
    unscoped = str(tmp_path / 'custom_train_table.artifacts')
    os.replace(source, unscoped)

    selected = ttng.persist_ttng_bundle(
        unscoped,
        str(tmp_path / 'durable'),
        expected_tt_id='custom-tt',
    )

    assert read_tt_id_from_staging_bundle(selected) == 'custom-tt'


def test_publication_rejects_wrong_tt_id(tmp_path):
    source = _bundle(tmp_path, 'attempt', tt_id='custom-tt')

    with pytest.raises(TTNGBundleError, match='bundle attempt mismatch'):
        ttng.persist_ttng_bundle(
            source,
            str(tmp_path / 'durable'),
            expected_tt_id='different-attempt',
        )


def test_publication_validates_destination_before_selection(
        tmp_path, monkeypatch):
    source = _bundle(tmp_path, 'attempt')
    durable = str(tmp_path / 'durable')
    original_copy = ttng.fsutil.copy_tree_threaded

    def corrupt_copy(copy_source, destination):
        original_copy(copy_source, destination)
        train_table = osp.join(
            destination, 'tt.parquet', 'part-00000.parquet')
        with open(train_table, 'r+b') as handle:
            handle.seek(-4, os.SEEK_END)
            handle.write(b'NOPE')

    monkeypatch.setattr(ttng.fsutil, 'copy_tree_threaded', corrupt_copy)

    with pytest.raises(TTNGBundleError, match='artifact manifest'):
        persist_durable_ttng_bundle(source, durable)
    assert not osp.exists(osp.join(durable, '_current.json'))


@pytest.mark.parametrize('bundle_id', ['../escape', 'nested/bundle', '', '.'])
def test_pointer_bundle_id_cannot_escape_immutable_root(tmp_path, bundle_id):
    durable = str(tmp_path / 'durable')
    os.makedirs(durable)
    _write(
        osp.join(durable, '_current.json'),
        json.dumps({'bundle_id': bundle_id}).encode(),
    )

    with pytest.raises(TTNGBundleError, match='invalid bundle_id'):
        resolve_durable_ttng_bundle(durable)


def test_restore_heals_interrupted_local_copy_on_retry(monkeypatch, tmp_path):
    source = _bundle(tmp_path, 'attempt')
    durable = str(tmp_path / 'durable')
    local = str(tmp_path / 'local')
    persist_durable_ttng_bundle(source, durable)

    original_copy = ttng.fsutil.copy_tree_threaded
    attempts = 0

    def interrupted_copy(copy_source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            _write(osp.join(destination, 'tt.parquet'), b'partial')
            raise RuntimeError('copy interrupted')
        return original_copy(copy_source, destination)

    monkeypatch.setattr(
        ttng.fsutil, 'copy_tree_threaded', interrupted_copy)
    with pytest.raises(RuntimeError, match='copy interrupted'):
        restore_durable_ttng_bundle(durable, local)
    assert not osp.exists(local)

    assert restore_durable_ttng_bundle(durable, local) == local
    assert_ttng_bundle_complete(local)


def test_restore_rejects_incomplete_unselected_root(tmp_path):
    durable = str(tmp_path / 'durable')
    local = str(tmp_path / 'local')
    _write(osp.join(durable, 'tt.parquet'), b'unpublished')

    with pytest.raises(TTNGBundleError, match='missing'):
        restore_durable_ttng_bundle(durable, local)
