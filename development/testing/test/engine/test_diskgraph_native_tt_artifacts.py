# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import struct
from contextlib import contextmanager

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from io_utils import fsutil
from kumo.diskgraph.diskgraph_native_tt_artifacts import (
    _validate_train_table_parquet,
    get_validated_artifact_bundle,
    parse_split_assignment_artifact,
    parse_stats_artifact,
    resolve_manifest_artifact_paths,
    validate_artifact_bundle,
)


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


def _sha256(path: str) -> str:
    return hashlib.sha256(fsutil.read_binary(path)).hexdigest()


def _sha256_path(path: str) -> str:
    if not fsutil.isdir(path):
        return _sha256(path)

    files = []
    stack = [path]
    while stack:
        current = stack.pop()
        for name in fsutil.listdir(current):
            child = name
            if not str(child).startswith(current):
                child = f'{current}/{name}'
            if fsutil.isdir(child):
                stack.append(child)
            else:
                files.append(child)
    files.sort()

    hasher = hashlib.sha256()
    prefix = path.rstrip('/') + '/'
    for file_path in files:
        rel = file_path[len(prefix):] if file_path.startswith(prefix) else file_path
        hasher.update(rel.encode())
        hasher.update(b'\0')
        hasher.update(_sha256(file_path).encode())
        hasher.update(b'\0')
    return hasher.hexdigest()


def _stats_artifact(row_count: int) -> dict:
    return {
        'version': 2,
        'kind': 'tt_stats',
        'num_total_rows': row_count,
        'num_train_rows': row_count,
        'num_entities': 2,
        'num_targets': 3,
        'num_links': None,
    }


def _write_train_table(path, row_count: int) -> int:
    table = pa.table({
        'ENTITY1': pa.array(range(row_count), type=pa.int64()),
        'TIMESTAMP': pa.array(range(row_count), type=pa.int64()),
        'TARGET1': pa.array(range(row_count), type=pa.int64()),
    })
    pq.write_table(table, path, row_group_size=max(1, row_count // 2))
    return path.stat().st_size


def _manifest_payload(train_table, stats, row_count: int, **extra) -> dict:
    source_part = train_table.with_name(f'.{train_table.name}.part')
    train_table.replace(source_part)
    train_table.mkdir()
    part = train_table / 'part-00000.parquet'
    source_part.replace(part)
    payload = {
        'version': 3,
        'row_count': row_count,
        'train_table': {
            'path': train_table.name,
            'format': 'tt_parquet_dataset_v1',
            'row_count': row_count,
            'size_bytes': part.stat().st_size,
            'parts': [{
                'path': f'{train_table.name}/{part.name}',
                'row_count': row_count,
                'size_bytes': part.stat().st_size,
            }],
        },
        'stats': {'path': stats.name, 'format': 'tt_stats_json'},
        'checksums': {'stats_sha256': _sha256(str(stats))},
    }
    payload.update(extra)
    if 'train_table_edges' in extra:
        payload['checksums']['train_table_edges_sha256'] = 'present'
    return payload


def _write_train_table_dataset(path, part_rows: tuple[int, ...]) -> list[dict]:
    path.mkdir()
    parts = []
    for index, row_count in enumerate(part_rows):
        part = path / f'part-{index:05}.parquet'
        _write_train_table(part, row_count)
        parts.append({
            'path': f'{path.name}/{part.name}',
            'row_count': row_count,
            'size_bytes': part.stat().st_size,
        })
    return parts


def _manifest_payload_v3(train_table, stats, part_rows: tuple[int, ...]) -> dict:
    parts = _write_train_table_dataset(train_table, part_rows)
    row_count = sum(part_rows)
    return {
        'version': 3,
        'row_count': row_count,
        'train_table': {
            'path': train_table.name,
            'format': 'tt_parquet_dataset_v1',
            'row_count': row_count,
            'size_bytes': sum(part['size_bytes'] for part in parts),
            'parts': parts,
        },
        'stats': {'path': stats.name, 'format': 'tt_stats_json'},
        'checksums': {'stats_sha256': _sha256(str(stats))},
    }


def _build_train_table_edges_bundle(tmp_path):
    edge_dir = tmp_path / 'ACCOUNT__rev_ENTITY__train_table' / 'edge.parquet'
    edge_dir.mkdir(parents=True)
    (edge_dir / 'part-0000.parquet').write_bytes(b'edge-part')

    fkey_cols_dir = tmp_path / 'train_table_fkey_cols.parquet'
    fkey_cols_dir.mkdir(parents=True)
    (fkey_cols_dir / 'part-0000.parquet').write_bytes(b'fkey-part')

    bundle = {
        'version': 1,
        'kind': 'tt_train_table_edges',
        'format': 'tt_edge_bundle_v1',
        'reverse_edges': {
            'ACCOUNT__rev_ENTITY__train_table': {
                'path': edge_dir.relative_to(tmp_path).as_posix(),
                'format': 'edge_parquet',
            }
        },
        'fkey_cols': {
            'path': fkey_cols_dir.relative_to(tmp_path).as_posix(),
            'format': 'fkey_cols_parquet',
        },
        'mask': {
            'present': False,
            'reason': 'lp_array_target_no_mask',
        },
        'checksums': {
            'fkey_cols_sha256': _sha256_path(str(fkey_cols_dir)),
            'ACCOUNT__rev_ENTITY__train_table_sha256': _sha256_path(
                str(edge_dir)),
        },
    }
    return bundle


def test_stats_v2_accepts_bounded_optional_fields(tmp_path):
    path = tmp_path / 'stats.json'
    payload = _stats_artifact(4)
    payload['num_timeframes'] = 3
    path.write_text(json.dumps(payload))

    assert parse_stats_artifact(str(path))['num_timeframes'] == 3


def test_stats_v1_is_not_a_current_native_contract(tmp_path):
    path = tmp_path / 'stats.json'
    payload = _stats_artifact(4)
    payload['version'] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='version/kind mismatch'):
        parse_stats_artifact(str(path))


def test_validate_artifact_bundle_success(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'tt.manifest.json'

    _write_train_table(train_table, 4)
    stats.write_text(json.dumps(_stats_artifact(4)))

    train_table_edges = _build_train_table_edges_bundle(tmp_path)

    payload = _manifest_payload(
        train_table, stats, 4, train_table_edges=train_table_edges)
    manifest.write_text(json.dumps(payload))

    out = validate_artifact_bundle(str(manifest))
    assert out['version'] == 3
    assert 'parsed_split' not in out
    assert out['parsed_stats']['num_total_rows'] == 4


def test_validate_artifact_bundle_accepts_ordered_v3_dataset(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    stats.write_text(json.dumps(_stats_artifact(5)))
    payload = _manifest_payload_v3(train_table, stats, (2, 3))
    manifest.write_text(json.dumps(payload))

    out = validate_artifact_bundle(str(manifest))

    assert out['version'] == 3
    assert [part['row_count'] for part in out['train_table']['parts']] == [2, 3]


@pytest.mark.parametrize('section_format, header, message', [
    ('tt_fact_pseudo_time_v1', (1, 1, 3), None),
    ('tt_fact_pseudo_time_v2', (1, 1, 3), 'format unsupported'),
    ('tt_fact_pseudo_time_v1', (2, 1, 3), 'version mismatch'),
    ('tt_fact_pseudo_time_v1', (1, 1, 4), 'size mismatch'),
])
def test_validate_artifact_bundle_checks_fact_pseudo_time(
        tmp_path, section_format, header, message):
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps(_stats_artifact(5)))
    payload = _manifest_payload_v3(tmp_path / 'tt.parquet', stats, (5, ))
    payload['version'] = 4
    sidecar = tmp_path / 'fact_pseudo_time.bin'
    sidecar.write_bytes(b'DGPTIME\0' + struct.pack('<IIQ', *header) +
                        bytes([0, 1, 3]))
    payload['fact_pseudo_time'] = {
        'path': sidecar.name,
        'format': section_format,
        'node_type': 'orders',
    }
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(payload))

    if message is None:
        out = validate_artifact_bundle(str(manifest))
        assert out['fact_pseudo_time']['path'] == str(sidecar)
    else:
        with pytest.raises(RuntimeError, match=message):
            validate_artifact_bundle(str(manifest))


@pytest.mark.parametrize('mutation, message', [
    ('extra', 'inventory mismatch'),
    ('path', 'path is not canonical'),
])
def test_validate_artifact_bundle_rejects_invalid_v3_inventory(
    tmp_path,
    mutation,
    message,
):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    stats.write_text(json.dumps(_stats_artifact(3)))
    payload = _manifest_payload_v3(train_table, stats, (1, 2))
    if mutation == 'extra':
        _write_train_table(train_table / 'part-00002.parquet', 1)
    else:
        payload['train_table']['parts'][1]['path'] = \
            'tt.parquet/part-00002.parquet'
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=message):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_rejects_v3_schema_mismatch(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    stats.write_text(json.dumps(_stats_artifact(3)))
    payload = _manifest_payload_v3(train_table, stats, (2, 1))
    second_part = train_table / 'part-00001.parquet'
    pq.write_table(pa.table({'OTHER': pa.array([1])}), second_part)
    payload['train_table']['parts'][1]['size_bytes'] = second_part.stat().st_size
    payload['train_table']['size_bytes'] = sum(
        part['size_bytes'] for part in payload['train_table']['parts'])
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match='schema differs from part 0'):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_rejects_single_file_v3_shape(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))
    payload = {
        'version': 3,
        'row_count': 3,
        'train_table': {
            'path': train_table.name,
            'format': 'tt_parquet_dataset_v1',
            'row_count': 3,
            'size_bytes': train_table.stat().st_size,
            'parts': [],
        },
        'stats': {'path': stats.name, 'format': 'tt_stats_json'},
        'checksums': {'stats_sha256': _sha256(str(stats))},
    }
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match='non-symlink directory'):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_never_hashes_train_table(
    tmp_path,
    monkeypatch,
):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 100_000)
    stats.write_text(json.dumps(_stats_artifact(100_000)))
    payload = _manifest_payload(train_table, stats, 100_000)
    payload['checksums']['train_table_sha256'] = 'informational-only'
    manifest.write_text(json.dumps(payload))

    monkeypatch.setattr(
        'kumo.diskgraph.diskgraph_native_tt_artifacts._sha256_path',
        lambda _path: pytest.fail('tt.parquet must not be content-hashed'),
    )
    validate_artifact_bundle(str(manifest))


def test_validated_bundle_does_not_cache_mutated_train_table(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))
    manifest.write_text(json.dumps(_manifest_payload(train_table, stats, 3)))

    get_validated_artifact_bundle(str(manifest))
    with (train_table / 'part-00000.parquet').open('r+b') as handle:
        handle.seek(-4, io.SEEK_END)
        handle.write(b'NOPE')

    with pytest.raises(RuntimeError, match='footer/schema is invalid'):
        get_validated_artifact_bundle(str(manifest))


def test_train_table_metadata_validation_does_not_read_to_eof(
    tmp_path,
    monkeypatch,
):
    train_table = tmp_path / 'tt.parquet'
    _write_train_table(train_table, 200_000)
    real_size = train_table.stat().st_size
    bytes_read = 0

    class GuardedReader(io.BufferedReader):
        def read(self, size=-1):
            nonlocal bytes_read
            assert size != -1, 'metadata validation must not read to EOF'
            out = super().read(size)
            bytes_read += len(out)
            return out

    @contextmanager
    def guarded_handle(path, mode):
        assert path == str(train_table)
        assert mode == 'rb'
        with open(path, 'rb', buffering=0) as raw:
            with GuardedReader(raw) as reader:
                yield reader

    monkeypatch.setattr(fsutil, 'is_local', lambda _path: False)
    monkeypatch.setattr(fsutil, 'get_binary_handle', guarded_handle)
    _validate_train_table_parquet(
        str(train_table),
        expected_size_bytes=real_size,
        expected_row_count=200_000,
    )
    assert bytes_read < real_size


def test_adls_train_table_metadata_validation_uses_range_filesystem(
    tmp_path,
    monkeypatch,
):
    train_table = tmp_path / 'tt.parquet'
    _write_train_table(train_table, 10)
    real_size = train_table.stat().st_size
    uri = ('abfss://container@account.dfs.core.windows.net/'
           'tt.parquet')
    filesystem = pafs.SubTreeFileSystem(
        str(tmp_path),
        pafs.LocalFileSystem(),
    )

    monkeypatch.setattr(
        fsutil,
        'object_identity',
        lambda path: (fsutil.ObjectIdentity(
            size=real_size,
            etag='test-etag',
            version=None,
        ) if path == uri else None),
    )
    monkeypatch.setattr(fsutil, 'pyarrow_filesystem',
                        lambda path: filesystem if path == uri else None)
    monkeypatch.setattr(
        fsutil,
        'get_binary_handle',
        lambda *_args, **_kwargs: pytest.fail(
            'ADLS parquet validation must not use a sequential stream'),
    )

    _validate_train_table_parquet(
        uri,
        expected_size_bytes=real_size,
        expected_row_count=10,
    )


def test_adls_train_table_metadata_validation_preserves_provider_failure(
    monkeypatch,
):
    from azure.core.exceptions import ServiceRequestError

    uri = ('abfss://container@account.dfs.core.windows.net/'
           'tt.parquet')
    provider_error = ServiceRequestError('temporary Azure failure')

    def raise_provider_error(_path):
        raise provider_error

    monkeypatch.setattr(fsutil, 'object_identity',
                        raise_provider_error)
    monkeypatch.setattr(
        fsutil,
        'filesize',
        lambda _path: pytest.fail('validation must not use ADLS filesize'),
    )

    with pytest.raises(ServiceRequestError) as captured:
        _validate_train_table_parquet(
            uri,
            expected_size_bytes=1,
            expected_row_count=1,
        )
    assert captured.value is provider_error


def test_validate_artifact_bundle_rejects_size_mismatch(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))
    payload = _manifest_payload(train_table, stats, 3)
    payload['train_table']['size_bytes'] += 1
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match='size mismatch'):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_rejects_corrupt_footer(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    with train_table.open('r+b') as handle:
        handle.seek(-4, io.SEEK_END)
        handle.write(b'NOPE')
    stats.write_text(json.dumps(_stats_artifact(3)))
    manifest.write_text(json.dumps(_manifest_payload(train_table, stats, 3)))

    with pytest.raises(RuntimeError, match='footer/schema is invalid'):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_rejects_truncated_footer(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    with train_table.open('r+b') as handle:
        handle.truncate(train_table.stat().st_size - 3)
    stats.write_text(json.dumps(_stats_artifact(3)))
    manifest.write_text(json.dumps(_manifest_payload(train_table, stats, 3)))

    with pytest.raises(RuntimeError, match='footer/schema is invalid'):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_rejects_footer_row_mismatch(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(4)))
    manifest.write_text(json.dumps(_manifest_payload(train_table, stats, 4)))

    with pytest.raises(RuntimeError, match='row_count mismatch'):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_rejects_stats_row_mismatch(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(2)))
    manifest.write_text(json.dumps(_manifest_payload(train_table, stats, 3)))

    with pytest.raises(
            RuntimeError,
            match='stats num_total_rows mismatch with manifest row_count'):
        validate_artifact_bundle(str(manifest))


def test_validate_train_table_rejects_invalid_row_group_bounds(
    tmp_path,
    monkeypatch,
):
    train_table = tmp_path / 'tt.parquet'
    _write_train_table(train_table, 3)
    real_size = train_table.stat().st_size

    class FakeColumn:
        total_compressed_size = real_size
        data_page_offset = 4
        dictionary_page_offset = None

    class FakeRowGroup:
        num_rows = 3
        num_columns = 3

        @staticmethod
        def column(_index):
            return FakeColumn()

    class FakeMetadata:
        num_rows = 3
        num_columns = 3
        num_row_groups = 1
        serialized_size = 16

        @staticmethod
        def row_group(_index):
            return FakeRowGroup()

    class FakeSchema:
        names = ['ENTITY1', 'TIMESTAMP', 'TARGET1']

        def __len__(self):
            return 3

    class FakeParquet:
        metadata = FakeMetadata()
        schema_arrow = FakeSchema()

    monkeypatch.setattr(pq, 'ParquetFile', lambda _handle: FakeParquet())

    with pytest.raises(RuntimeError, match='exceeds data region'):
        _validate_train_table_parquet(
            str(train_table),
            expected_size_bytes=real_size,
            expected_row_count=3,
        )


def test_validate_artifact_bundle_requires_v3_size(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'manifest.json'
    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))
    payload = _manifest_payload(train_table, stats, 3)
    del payload['train_table']['size_bytes']
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match='size_bytes'):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_sidecar_checksum_mismatch(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'tt.manifest.json'

    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))

    train_table_edges = _build_train_table_edges_bundle(tmp_path)

    payload = _manifest_payload(
        train_table, stats, 3, train_table_edges=train_table_edges)
    payload['checksums']['stats_sha256'] = 'deadbeef'
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match='checksum mismatch'):
        validate_artifact_bundle(str(manifest))


@pytest.mark.parametrize('version, has_sidecar', [
    (3, True),
    (4, False),
    (5, True),
])
def test_manifest_rejects_version_visibility_mismatch(
    tmp_path,
    version,
    has_sidecar,
):
    manifest = tmp_path / 'manifest.json'
    payload = {'version': version}
    if has_sidecar:
        payload['fact_pseudo_time'] = {'path': 'fact_pseudo_time.bin'}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='manifest version mismatch'):
        validate_artifact_bundle(str(manifest))


@pytest.mark.parametrize('unsupported_version', [1, 2])
def test_validate_artifact_bundle_rejects_pre_v3(
    tmp_path,
    unsupported_version,
):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'tt.manifest.json'

    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))

    payload = _manifest_payload(train_table, stats, 3)
    payload['version'] = unsupported_version
    manifest.write_text(json.dumps(payload))

    with pytest.raises(
        RuntimeError,
        match=f'expected=3 got={unsupported_version}',
    ):
        validate_artifact_bundle(str(manifest))


def test_validate_artifact_bundle_without_train_table_edges(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'tt.manifest.json'
    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))
    manifest.write_text(json.dumps(_manifest_payload(train_table, stats, 3)))

    out = validate_artifact_bundle(str(manifest))
    assert out['parsed_train_table_edges']['enabled'] is False
    assert out['parsed_train_table_edges']['reverse_edges'] == {}


def test_validate_artifact_bundle_with_train_table_edges(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'tt.manifest.json'

    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))

    train_table_edges = _build_train_table_edges_bundle(tmp_path)

    payload = _manifest_payload(
        train_table, stats, 3, train_table_edges=train_table_edges)
    manifest.write_text(json.dumps(payload))

    out = validate_artifact_bundle(str(manifest))
    assert out['parsed_train_table_edges']['format'] == 'tt_edge_bundle_v1'


def test_validate_artifact_bundle_allows_disabled_empty_train_table_edges(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'tt.manifest.json'

    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))

    train_table_edges = {
        'version': 1,
        'kind': 'tt_train_table_edges',
        'format': 'tt_edge_bundle_v1',
        'enabled': False,
        'fkeys': [],
        'reverse_edges': {},
    }

    payload = _manifest_payload(
        train_table, stats, 3, train_table_edges=train_table_edges)
    manifest.write_text(json.dumps(payload))

    out = validate_artifact_bundle(str(manifest))
    assert out['parsed_train_table_edges']['enabled'] is False
    assert out['parsed_train_table_edges']['reverse_edges'] == {}


def test_validate_artifact_bundle_rejects_enabled_empty_train_table_edges(tmp_path):
    train_table = tmp_path / 'tt.parquet'
    stats = tmp_path / 'stats.json'
    manifest = tmp_path / 'tt.manifest.json'

    _write_train_table(train_table, 3)
    stats.write_text(json.dumps(_stats_artifact(3)))

    train_table_edges = {
        'version': 1,
        'kind': 'tt_train_table_edges',
        'format': 'tt_edge_bundle_v1',
        'enabled': True,
        'reverse_edges': {},
    }

    payload = _manifest_payload(
        train_table, stats, 3, train_table_edges=train_table_edges)
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match='missing reverse_edges map'):
        validate_artifact_bundle(str(manifest))


# ---------------------------------------------------------------------------
# Unit tests for schema-aware manifest path resolution.
# ---------------------------------------------------------------------------


class TestResolveManifestPathsToAbsolute:
    MANIFEST_DIR = '/tmp/dataplane/job-123/training_table'

    def test_relative_path_joined(self):
        manifest = {'train_table': {'path': 'tt.parquet'}}
        resolve_manifest_artifact_paths(manifest, self.MANIFEST_DIR)
        assert manifest['train_table']['path'] == (
            '/tmp/dataplane/job-123/training_table/tt.parquet')

    def test_absolute_local_path_rejected(self):
        manifest = {'train_table': {'path': '/legacy/absolute/path.parquet'}}
        with pytest.raises(RuntimeError, match='bundle-relative'):
            resolve_manifest_artifact_paths(manifest, self.MANIFEST_DIR)

    def test_external_uri_path_rejected(self):
        manifest = {
            'train_table': {'path': 's3://bucket/already-remote.parquet'}
        }
        with pytest.raises(RuntimeError, match='bundle-relative'):
            resolve_manifest_artifact_paths(manifest, self.MANIFEST_DIR)

    @pytest.mark.parametrize('path', [
        '../escape.parquet',
        'nested/../escape.parquet',
        './unnormalized.parquet',
        'nested//unnormalized.parquet',
        r'nested\windows-escape.parquet',
    ])
    def test_unsafe_relative_path_rejected(self, path):
        manifest = {'train_table': {'path': path}}
        with pytest.raises(RuntimeError, match='bundle-relative'):
            resolve_manifest_artifact_paths(manifest, self.MANIFEST_DIR)

    def test_nested_paths_resolved(self):
        manifest = {
            'train_table_edges': {
                'fkey_cols': {'path': 'train_table_fkey_cols.parquet'},
                'mask': {
                    'path': 'train_table_mask.parquet',
                    'present': True,
                },
                'reverse_edges': {
                    'k': {
                        'path': 'k/edge.parquet',
                        'format': 'edge_parquet',
                    },
                },
            },
        }
        resolve_manifest_artifact_paths(manifest, self.MANIFEST_DIR)
        edges = manifest['train_table_edges']
        assert edges['fkey_cols']['path'] == (
            f'{self.MANIFEST_DIR}/train_table_fkey_cols.parquet')
        assert edges['mask']['path'] == (
            f'{self.MANIFEST_DIR}/train_table_mask.parquet')
        # Non-path fields untouched.
        assert edges['mask']['present'] is True
        assert edges['reverse_edges']['k']['path'] == (
            f'{self.MANIFEST_DIR}/k/edge.parquet')
        assert edges['reverse_edges']['k']['format'] == 'edge_parquet'

    def test_unknown_and_invalid_sections_are_not_rewritten(self):
        manifest = {
            'train_table': {'path': None},
            'unknown': {'path': 'not-an-artifact'},
        }
        resolve_manifest_artifact_paths(manifest, self.MANIFEST_DIR)
        assert manifest['train_table']['path'] is None
        assert manifest['unknown']['path'] == 'not-an-artifact'


# parse_split_assignment_artifact (DG-main DGSPLT path). Binary format (LE):
#   [0:8] magic b'DGSPLT\x00\x00', [8:12] version u32, [12:20] row_count u64,
#   [20:] one stage-id byte per row (0=train, 1=val, 2=test, 255=unused).

_MAGIC = b'DGSPLT\x00\x00'


def _build_split_bin(assignments: list[int], version: int = 1) -> bytes:
    """Construct a valid DGSPLT artifact from a list of per-row stage ids."""
    row_count = len(assignments)
    header = _MAGIC + struct.pack('<IQ', version, row_count)
    return header + bytes(assignments)


class TestParseSplitAssignmentArtifact:
    def test_happy_path_train_val_test(self, tmp_path):
        # 6 rows: 2 train (0), 2 val (1), 2 test (2)
        assignments = [0, 0, 1, 1, 2, 2]
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(_build_split_bin(assignments))

        result = parse_split_assignment_artifact(str(artifact))

        assert result['version'] == 1
        assert result['row_count'] == 6
        assert result['counts'] == {'train': 2, 'val': 2, 'test': 2}
        assert result['unused_count'] == 0
        assert list(result['stage_ids']) == assignments

    def test_happy_path_with_unused_rows(self, tmp_path):
        # 5 rows: 1 train, 1 val, 1 test, 2 unused (255)
        assignments = [0, 1, 2, 255, 255]
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(_build_split_bin(assignments))

        result = parse_split_assignment_artifact(str(artifact))

        assert result['row_count'] == 5
        assert result['counts'] == {'train': 1, 'val': 1, 'test': 1}
        assert result['unused_count'] == 2

    def test_happy_path_train_only(self, tmp_path):
        # Edge case: all rows are train; val/test counts default to 0
        assignments = [0, 0, 0]
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(_build_split_bin(assignments))

        result = parse_split_assignment_artifact(str(artifact))

        assert result['counts']['train'] == 3
        assert result['counts']['val'] == 0
        assert result['counts']['test'] == 0
        assert result['unused_count'] == 0

    def test_rejects_unsupported_stage_id(self, tmp_path):
        # Stage id 3 is not in {0, 1, 2, 255} — must raise
        assignments = [0, 1, 3]
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(_build_split_bin(assignments))

        with pytest.raises(RuntimeError, match='unsupported stage ids'):
            parse_split_assignment_artifact(str(artifact))

    def test_rejects_multiple_unsupported_stage_ids(self, tmp_path):
        # Both 4 and 100 are unsupported; the error should mention them
        assignments = [0, 4, 100]
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(_build_split_bin(assignments))

        with pytest.raises(RuntimeError, match='unsupported stage ids'):
            parse_split_assignment_artifact(str(artifact))

    def test_rejects_wrong_magic(self, tmp_path):
        row_count = 2
        bad_magic = b'BADMAGIC'
        data = bad_magic + struct.pack('<IQ', 1, row_count) + bytes([0, 1])
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(data)

        with pytest.raises(RuntimeError, match='magic mismatch'):
            parse_split_assignment_artifact(str(artifact))

    def test_rejects_too_small_for_header(self, tmp_path):
        # Only 10 bytes — minimum valid header is 20 bytes
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(b'DGSPLT\x00\x00\x01\x00')

        with pytest.raises(RuntimeError, match='too small for header'):
            parse_split_assignment_artifact(str(artifact))

    def test_rejects_wrong_version(self, tmp_path):
        # version=2 is unsupported
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(_build_split_bin([0, 1], version=2))

        with pytest.raises(RuntimeError, match='version mismatch'):
            parse_split_assignment_artifact(str(artifact))

    def test_rejects_row_count_length_mismatch(self, tmp_path):
        # Declare row_count=5 but only supply 3 bytes of assignments
        header = _MAGIC + struct.pack('<IQ', 1, 5)
        artifact = tmp_path / 'split.bin'
        artifact.write_bytes(header + bytes([0, 1, 2]))  # only 3 rows

        with pytest.raises(RuntimeError, match='length mismatch'):
            parse_split_assignment_artifact(str(artifact))
