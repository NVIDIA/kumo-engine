# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json as pyjson
import os
import stat
import struct
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from io_utils import fsutil
from io_utils.fsutil.uri import ParseURI

_EPOCH_NS_THRESHOLD = 1_000_000_000_000_000_000
_EPOCH_US_THRESHOLD = 1_000_000_000_000_000
_EPOCH_MS_THRESHOLD = 1_000_000_000_000


def normalize_timestamp_like_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    def _from_epoch(raw: int):
        import pandas as pd
        from pandas.errors import OutOfBoundsDatetime

        # Temporal-regression split-stats payloads can carry int values
        # that overflow when interpreted at the magnitude-implied unit (e.g. a
        # huge int that pandas would treat as ns but that exceeds
        # `pd.Timestamp.max.value`, raising either `OverflowError: Python int
        # too large to convert to C long` or `OutOfBoundsDatetime`). Try units
        # in descending precision and fall back across the chain so a
        # mis-scaled input still produces a usable `pd.Timestamp` rather than
        # crashing the lazy-apply path in `_apply_native_split_stats`.
        mag = abs(raw)
        if mag >= _EPOCH_NS_THRESHOLD:
            unit_order = ('ns', 'us', 'ms', 's')
        elif mag >= _EPOCH_US_THRESHOLD:
            unit_order = ('us', 'ms', 's')
        elif mag >= _EPOCH_MS_THRESHOLD:
            unit_order = ('ms', 's')
        else:
            unit_order = ('s',)
        for unit in unit_order:
            try:
                return pd.Timestamp(raw, unit=unit)
            except (OverflowError, ValueError, OutOfBoundsDatetime):
                continue
        # All units exhausted: clamp to the representable max/min so downstream
        # `SplitTableStats(**payload)` still receives a `pd.Timestamp`. Better
        # to lose precision on an out-of-range bound than to crash split-stats.
        return pd.Timestamp.max if raw > 0 else pd.Timestamp.min

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _from_epoch(int(value))
    if isinstance(value, str):
        s = value.strip()
        if s.startswith('kumo_timestamp_'):
            s = s[len('kumo_timestamp_'):].strip()
        if s == '' or s.lower() in {'nat', 'none', 'null'}:
            return None
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return _from_epoch(int(s))
        return value
    return value


def get_validated_artifact_bundle(manifest_path: str) -> dict[str, Any]:
    # The manifest binds parquet size and footer metadata, not its mtime. A
    # manifest-only cache key could therefore hide a later same-size mutation.
    # Validation is bounded by parquet metadata and small sidecars, so
    # revalidate the bundle on every consumption boundary.
    return validate_artifact_bundle(manifest_path)


def _sha256_file(path: str) -> str:
    data = fsutil.read_binary(path)
    return hashlib.sha256(data).hexdigest()


def artifact_file_size(path: str) -> int:
    """Return one file's size without reading its contents."""
    if fsutil.is_local(path):
        return os.stat(path).st_size

    if fsutil.is_adls(path):
        return fsutil.object_identity(path).size

    with fsutil.get_binary_handle(path, 'rb') as handle:
        handle.seek(0, os.SEEK_END)
        return int(handle.tell())


def _inspect_parquet_artifact_details(path: str) -> tuple[int, int, Any]:
    """Return exact size/rows after bounded footer and schema validation.

    PyArrow seeks to the footer; this never hashes or scans parquet data pages.
    """
    import pyarrow.parquet as pq

    actual_size_bytes = artifact_file_size(path)

    try:
        if fsutil.is_adls(path):
            parquet = pq.ParquetFile(
                ParseURI(path).path,
                filesystem=fsutil.pyarrow_filesystem(path),
            )
            metadata = parquet.metadata
            schema = parquet.schema_arrow
        else:
            with fsutil.get_binary_handle(path, 'rb') as handle:
                parquet = pq.ParquetFile(handle)
                metadata = parquet.metadata
                schema = parquet.schema_arrow
    except Exception as error:
        raise RuntimeError('train-table parquet footer/schema is invalid') from error

    if metadata.num_rows < 0:
        raise RuntimeError('parquet footer has a negative row_count')
    if metadata.num_columns <= 0 or len(schema) <= 0:
        raise RuntimeError('train-table parquet schema must not be empty')
    if any(not name for name in schema.names):
        raise RuntimeError('train-table parquet schema has an empty field name')
    if len(set(schema.names)) != len(schema.names):
        raise RuntimeError('train-table parquet schema has duplicate field names')
    if metadata.num_rows > 0 and metadata.num_row_groups <= 0:
        raise RuntimeError('non-empty parquet has no row groups')

    footer_size = int(metadata.serialized_size)
    footer_start = actual_size_bytes - footer_size - 8
    if footer_size <= 0 or footer_start < 4:
        raise RuntimeError('train-table parquet footer bounds are invalid')

    row_group_rows = 0
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        if row_group.num_rows < 0:
            raise RuntimeError('train-table parquet row group has negative rows')
        if row_group.num_columns != metadata.num_columns:
            raise RuntimeError(
                'train-table parquet row-group/schema column mismatch')
        row_group_rows += row_group.num_rows

        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            compressed_size = int(column.total_compressed_size)
            data_offset = int(column.data_page_offset)
            dictionary_offset = column.dictionary_page_offset
            offsets = [data_offset]
            if dictionary_offset is not None:
                offsets.append(int(dictionary_offset))
            if compressed_size < 0 or any(
                    offset < 4 or offset >= footer_start for offset in offsets):
                raise RuntimeError(
                    'train-table parquet row-group column bounds are invalid')
            first_page_offset = min(offsets)
            if first_page_offset + compressed_size > footer_start:
                raise RuntimeError(
                    'train-table parquet row-group column exceeds data region')

    if row_group_rows != metadata.num_rows:
        raise RuntimeError(
            'parquet row-group rows do not match footer row_count')
    return actual_size_bytes, int(metadata.num_rows), schema


def _inspect_parquet_artifact(path: str) -> tuple[int, int]:
    actual_size_bytes, actual_row_count, _schema = \
        _inspect_parquet_artifact_details(path)
    return actual_size_bytes, actual_row_count


def _validate_train_table_parquet(
    path: str,
    *,
    expected_size_bytes: int,
    expected_row_count: int,
) -> None:
    """Validate a closed train table from bounded footer metadata only."""
    actual_size_bytes, actual_row_count = _inspect_parquet_artifact(path)
    if actual_size_bytes != expected_size_bytes:
        raise RuntimeError(
            'train-table size mismatch: '
            f'expected={expected_size_bytes} actual={actual_size_bytes}')
    if actual_row_count != expected_row_count:
        raise RuntimeError(
            'train-table parquet row_count mismatch: '
            f'expected={expected_row_count} actual={actual_row_count}')


def _dataset_entry_names(path: str) -> list[str]:
    """Return direct dataset entries after enforcing their concrete type."""
    if fsutil.is_local(path):
        try:
            root_stat = os.lstat(path)
        except OSError as error:
            raise RuntimeError(
                f'train-table dataset cannot be inspected: {path}') from error
        if (stat.S_ISLNK(root_stat.st_mode)
                or not stat.S_ISDIR(root_stat.st_mode)):
            raise RuntimeError(
                'train-table path must be a non-symlink directory')
        names = os.listdir(path)
        for name in names:
            entry_path = os.path.join(path, name)
            try:
                entry = os.lstat(entry_path)
            except OSError as error:
                raise RuntimeError(
                    f'train-table dataset entry cannot be inspected: '
                    f'{entry_path}') from error
            if (stat.S_ISLNK(entry.st_mode)
                    or not stat.S_ISREG(entry.st_mode)):
                raise RuntimeError(
                    'train-table dataset entries must be regular files: '
                    f'{entry_path}')
        return names

    if not fsutil.isdir(path):
        raise RuntimeError('train-table path must be a directory')
    try:
        entries = fsutil.listdir(path, detail=True)
    except NotImplementedError:
        entries = fsutil.listdir(path)
        names = [os.path.basename(str(entry).rstrip('/')) for entry in entries]
        for entry in entries:
            entry_path = str(entry)
            if not entry_path.startswith(path):
                entry_path = os.path.join(path, entry_path)
            if fsutil.isdir(entry_path):
                raise RuntimeError(
                    'train-table dataset entries must be files: '
                    f'{entry_path}')
        return names

    names = []
    for entry in entries:
        if entry.is_directory:
            raise RuntimeError(
                'train-table dataset entries must be files: '
                f'{entry.path}')
        names.append(os.path.basename(entry.path.rstrip('/')))
    return names


def _validate_train_table_dataset(train_table: dict[str, Any]) -> None:
    path = train_table['path']
    parts = train_table.get('parts')
    if not isinstance(parts, list):
        raise RuntimeError(
            'diskgraph manifest train_table.parts must be a list')

    expected_names = [f'part-{index:05}.parquet'
                      for index in range(len(parts))]
    actual_names = _dataset_entry_names(path)
    if len(actual_names) != len(set(actual_names)):
        raise RuntimeError('train-table dataset contains duplicate entries')
    if set(actual_names) != set(expected_names):
        raise RuntimeError(
            'train-table dataset inventory mismatch: '
            f'expected={expected_names!r} actual={sorted(actual_names)!r}')

    total_rows = 0
    total_bytes = 0
    dataset_schema = None
    for index, (part, expected_name) in enumerate(zip(parts, expected_names)):
        if not isinstance(part, dict) or set(part) != {
                'path', 'row_count', 'size_bytes'}:
            raise RuntimeError(
                f'train-table part {index} must contain exactly '
                'path/row_count/size_bytes')
        expected_path = os.path.join(path, expected_name)
        if part.get('path') != expected_path:
            raise RuntimeError(
                f'train-table part {index} path is not canonical: '
                f'{part.get("path")!r}')
        row_count = part.get('row_count')
        size_bytes = part.get('size_bytes')
        if (isinstance(row_count, bool) or not isinstance(row_count, int)
                or row_count < 0):
            raise RuntimeError(
                f'train-table part {index} row_count must be a '
                'non-negative int')
        if (isinstance(size_bytes, bool) or not isinstance(size_bytes, int)
                or size_bytes <= 0):
            raise RuntimeError(
                f'train-table part {index} size_bytes must be a positive int')

        actual_size, actual_rows, actual_schema = \
            _inspect_parquet_artifact_details(expected_path)
        if actual_size != size_bytes:
            raise RuntimeError(
                f'train-table part {index} size mismatch: '
                f'expected={size_bytes} actual={actual_size}')
        if actual_rows != row_count:
            raise RuntimeError(
                f'train-table part {index} row_count mismatch: '
                f'expected={row_count} actual={actual_rows}')
        if dataset_schema is None:
            dataset_schema = actual_schema
        elif not actual_schema.equals(dataset_schema, check_metadata=True):
            raise RuntimeError(
                f'train-table part {index} schema differs from part 0')
        total_rows += actual_rows
        total_bytes += actual_size

    if total_rows != train_table['row_count']:
        raise RuntimeError(
            'train-table aggregate row_count mismatch: '
            f'expected={train_table["row_count"]} actual={total_rows}')
    if total_bytes != train_table['size_bytes']:
        raise RuntimeError(
            'train-table aggregate size mismatch: '
            f'expected={train_table["size_bytes"]} actual={total_bytes}')


def _sha256_path(path: str) -> str:
    if not fsutil.isdir(path):
        return _sha256_file(path)

    files: list[str] = []
    stack = [path]
    while stack:
        current = stack.pop()
        for name in fsutil.listdir(current):
            child = name
            if not str(child).startswith(current):
                child = os.path.join(current, str(name))
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
        hasher.update(_sha256_file(file_path).encode())
        hasher.update(b'\0')
    return hasher.hexdigest()


def load_manifest(manifest_path: str) -> dict[str, Any]:
    if not fsutil.exists(manifest_path):
        raise RuntimeError(f'diskgraph manifest not found at: {manifest_path}')
    payload = pyjson.loads(fsutil.read_text(manifest_path))
    if not isinstance(payload, dict):
        raise RuntimeError('diskgraph manifest payload must be a JSON object')
    version = payload.get('version')
    expected = 4 if payload.get('fact_pseudo_time') is not None else 3
    if version != expected:
        raise RuntimeError(
            'diskgraph manifest version mismatch: '
            f'expected={expected} got={version}; '
            'version 4 requires fact visibility support')
    return payload


def parse_stats_artifact(path: str) -> dict[str, Any]:
    payload = pyjson.loads(fsutil.read_text(path))
    if not isinstance(payload, dict):
        raise RuntimeError('stats artifact must be a JSON object')

    version = payload.get('version')
    kind = payload.get('kind')
    if (version, kind) != (2, 'tt_stats'):
        raise RuntimeError(
            'stats artifact version/kind mismatch: '
            f'got version={version!r} kind={kind!r}')

    required_ints = ('num_total_rows', 'num_train_rows', 'num_entities', 'num_targets')
    for field in required_ints:
        if not isinstance(payload.get(field), int):
            raise RuntimeError(f'stats artifact missing integer field: {field}')

    num_links = payload.get('num_links')
    if num_links is not None and not isinstance(num_links, int):
        raise RuntimeError('stats artifact num_links must be int or null')

    return payload


def parse_split_stats_artifact(path: str) -> dict[str, Any]:
    payload = pyjson.loads(fsutil.read_text(path))
    if not isinstance(payload, dict):
        raise RuntimeError('split_stats artifact must be a JSON object')

    version = payload.get('version')
    kind = payload.get('kind')
    if (version, kind) != (1, 'tt_split_stats'):
        raise RuntimeError(
            'split_stats artifact version/kind mismatch: '
            f'got version={version!r} kind={kind!r}')

    train_stats = payload.get('train_table_stats')
    if not isinstance(train_stats, dict):
        raise RuntimeError('split_stats artifact missing train_table_stats')

    def _validate_stage(stage_name: str, stage_payload: Any) -> None:
        if stage_payload is None:
            return
        if not isinstance(stage_payload, dict):
            raise RuntimeError(f'split_stats {stage_name} must be dict or null')
        if not isinstance(stage_payload.get('num_train_rows'), int):
            raise RuntimeError(f'split_stats {stage_name} missing integer num_train_rows')
        if not isinstance(stage_payload.get('num_entities'), int):
            raise RuntimeError(f'split_stats {stage_name} missing integer num_entities')

    _validate_stage('train_table_stats', payload.get('train_table_stats'))
    _validate_stage('val_table_stats', payload.get('val_table_stats'))
    _validate_stage('test_table_stats', payload.get('test_table_stats'))
    return payload


def parse_train_table_col_stats_artifact(path: str) -> dict[str, Any]:
    payload = pyjson.loads(fsutil.read_text(path))
    if not isinstance(payload, dict):
        raise RuntimeError('train_table_col_stats artifact must be a JSON object')

    version = payload.get('version')
    kind = payload.get('kind')
    stats_format = payload.get('stats_format')
    if (version, kind) != (1, 'tt_train_table_col_stats'):
        raise RuntimeError(
            'train_table_col_stats artifact version/kind mismatch: '
            f'got version={version!r} kind={kind!r}')
    if stats_format != 'kumo_col_stats_v1':
        raise RuntimeError(
            'train_table_col_stats artifact format mismatch: '
            f'got stats_format={stats_format!r}')

    stats = payload.get('stats')
    if not isinstance(stats, dict):
        raise RuntimeError('train_table_col_stats artifact missing stats object')

    for key, value in stats.items():
        if not isinstance(key, str) or not key:
            raise RuntimeError('train_table_col_stats artifact contains invalid stat key')
        if not isinstance(value, dict):
            raise RuntimeError(
                'train_table_col_stats artifact stat payload must be an object')
        if 'value' not in value:
            raise RuntimeError(
                f'train_table_col_stats artifact stat {key!r} missing value')

    return payload


def parse_train_table_edges_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError('train_table_edges artifact must be a JSON object')

    version = payload.get('version')
    kind = payload.get('kind')
    edges_format = payload.get('format')
    if (version, kind, edges_format) != (1, 'tt_train_table_edges',
                                         'tt_edge_bundle_v1'):
        raise RuntimeError(
            'train_table_edges artifact version/kind/format mismatch: '
            f'got version={version!r} kind={kind!r} format={edges_format!r}')

    enabled = payload.get('enabled', True)
    if not isinstance(enabled, bool):
        raise RuntimeError('train_table_edges artifact enabled must be bool')

    reverse_edges = payload.get('reverse_edges')
    if not isinstance(reverse_edges, dict):
        raise RuntimeError('train_table_edges artifact missing reverse_edges map')

    declared_fkeys = payload.get('fkeys')
    has_declared_fkeys = isinstance(declared_fkeys, list) and len(declared_fkeys) > 0

    if enabled and not reverse_edges:
        raise RuntimeError('train_table_edges artifact missing reverse_edges map')
    if not enabled and reverse_edges and not has_declared_fkeys:
        raise RuntimeError(
            'train_table_edges artifact has reverse_edges even though enabled=false')

    for edge_key, edge_info in reverse_edges.items():
        if not isinstance(edge_key, str) or not edge_key:
            raise RuntimeError('train_table_edges artifact has invalid reverse edge key')
        if not isinstance(edge_info, dict):
            raise RuntimeError(
                f'train_table_edges artifact reverse edge {edge_key!r} must be an object')
        edge_path = edge_info.get('path')
        edge_format = edge_info.get('format')
        if not isinstance(edge_path, str) or not edge_path:
            raise RuntimeError(
                f'train_table_edges artifact reverse edge {edge_key!r} missing path')
        if edge_format != 'edge_parquet':
            raise RuntimeError(
                f'train_table_edges artifact reverse edge {edge_key!r} '
                f'format unsupported: {edge_format!r}')

    fkey_cols = payload.get('fkey_cols')
    if reverse_edges:
        if not isinstance(fkey_cols, dict):
            raise RuntimeError('train_table_edges artifact missing fkey_cols object')
        fkey_cols_path = fkey_cols.get('path')
        if not isinstance(fkey_cols_path, str) or not fkey_cols_path:
            raise RuntimeError('train_table_edges artifact has invalid fkey_cols.path')
        if fkey_cols.get('format') != 'fkey_cols_parquet':
            raise RuntimeError('train_table_edges artifact fkey_cols format unsupported')
    elif fkey_cols is not None and not isinstance(fkey_cols, dict):
        raise RuntimeError('train_table_edges artifact fkey_cols must be object when present')

    mask = payload.get('mask')
    if reverse_edges:
        if not isinstance(mask, dict):
            raise RuntimeError('train_table_edges artifact missing mask object')
        mask_present = mask.get('present')
        if not isinstance(mask_present, bool):
            raise RuntimeError('train_table_edges artifact mask.present must be bool')
        if mask_present:
            if not isinstance(mask.get('path'), str) or not mask.get('path'):
                raise RuntimeError('train_table_edges artifact mask.path missing')
            if mask.get('format') != 'mask_parquet':
                raise RuntimeError('train_table_edges artifact mask format unsupported')
    elif mask is not None and not isinstance(mask, dict):
        raise RuntimeError('train_table_edges artifact mask must be object when present')

    return payload


def _resolve_bundle_relative_path(path: str, manifest_dir: str) -> str:
    relative = PurePosixPath(path)
    if (relative.is_absolute() or '://' in path or '\\' in path
            or relative.as_posix() != path
            or any(part in {'.', '..'} for part in relative.parts)):
        raise RuntimeError(
            'native artifact path must be normalized and bundle-relative: '
            f'{path!r}')
    return os.path.join(manifest_dir, *relative.parts)


def _resolve_section_path(section: Any, manifest_dir: str) -> None:
    if not isinstance(section, dict):
        return
    path = section.get('path')
    if isinstance(path, str) and path:
        section['path'] = _resolve_bundle_relative_path(path, manifest_dir)


def resolve_manifest_artifact_paths(
    manifest: dict[str, Any],
    manifest_dir: str,
) -> None:
    """Resolve only path fields declared by the native manifest schema."""
    for section_name in (
            'train_table',
            'stats',
            'split_stats',
            'train_table_col_stats',
            'split',
            'fact_pseudo_time',
    ):
        _resolve_section_path(manifest.get(section_name), manifest_dir)

    train_table = manifest.get('train_table')
    if isinstance(train_table, dict):
        parts = train_table.get('parts')
        if isinstance(parts, list):
            for part in parts:
                _resolve_section_path(part, manifest_dir)

    edges = manifest.get('train_table_edges')
    if not isinstance(edges, dict):
        return
    for section_name in ('fkey_cols', 'mask', 'edge_attrs'):
        _resolve_section_path(edges.get(section_name), manifest_dir)
    reverse_edges = edges.get('reverse_edges')
    if isinstance(reverse_edges, dict):
        for edge in reverse_edges.values():
            _resolve_section_path(edge, manifest_dir)


def parse_split_assignment_artifact(path: str) -> dict[str, Any]:
    """Parse a legacy (non-TT-NG) DiskGraph ``split.bin`` (DGSPLT) artifact."""
    payload = fsutil.read_binary(path)
    if len(payload) < 20:
        raise RuntimeError('split assignment artifact too small for header')
    magic = payload[:8]
    if magic != b'DGSPLT\x00\x00':
        raise RuntimeError('split assignment artifact magic mismatch')
    version, row_count = struct.unpack('<IQ', payload[8:20])
    if version != 1:
        raise RuntimeError(
            f'split assignment artifact version mismatch: expected=1 got={version}')
    assignments = payload[20:]
    if len(assignments) != row_count:
        raise RuntimeError(
            'split assignment artifact length mismatch with declared row_count')
    counts = Counter(assignments)
    unsupported = sorted(v for v in counts if v not in (0, 1, 2, 255))
    if unsupported:
        raise RuntimeError(
            f'split assignment artifact has unsupported stage ids: {unsupported}')
    return {
        'version': version,
        'row_count': row_count,
        'counts': {
            'train': int(counts.get(0, 0)),
            'val': int(counts.get(1, 0)),
            'test': int(counts.get(2, 0)),
        },
        'unused_count': int(counts.get(255, 0)),
        'stage_ids': assignments,
    }


def _validate_fact_pseudo_time(section: Any) -> None:
    """Header-only check; the payload is one byte per fact row."""
    if (not isinstance(section, dict)
            or section.get('format') != 'tt_fact_pseudo_time_v1'):
        raise RuntimeError(
            'diskgraph manifest fact_pseudo_time format unsupported: '
            f'{section!r}')
    path = section.get('path')
    if not isinstance(path, str) or not fsutil.exists(path):
        raise RuntimeError(
            f'diskgraph manifest references missing artifact: {path}')
    with fsutil.get_binary_handle(path, 'rb') as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b'DGPTIME\x00':
        raise RuntimeError('fact_pseudo_time artifact magic mismatch')
    version, dtype, row_count = struct.unpack('<IIQ', header[8:])
    if (version, dtype) != (1, 1):
        raise RuntimeError(
            'fact_pseudo_time artifact version mismatch: '
            f'version={version} dtype={dtype}')
    if artifact_file_size(path) != 24 + row_count:
        raise RuntimeError('fact_pseudo_time artifact size mismatch')


def validate_artifact_bundle(manifest_path: str) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    resolve_manifest_artifact_paths(manifest, os.path.dirname(manifest_path))
    train_table = manifest.get('train_table')
    stats = manifest.get('stats')
    split_stats = manifest.get('split_stats')
    train_table_col_stats = manifest.get('train_table_col_stats')
    train_table_edges = manifest.get('train_table_edges')
    if not isinstance(train_table, dict) or not isinstance(stats, dict):
        raise RuntimeError('diskgraph manifest missing train_table/stats sections')
    if train_table_edges is None:
        # A node-task bundle can intentionally omit link-edge artifacts.
        # Consumers still enforce non-empty edge bundles when fkeys are present.
        train_table_edges = {
            'version': 1,
            'kind': 'tt_train_table_edges',
            'format': 'tt_edge_bundle_v1',
            'enabled': False,
            'fkeys': [],
            'reverse_edges': {},
        }
    elif not isinstance(train_table_edges, dict):
        raise RuntimeError('diskgraph manifest has invalid train_table_edges section')

    train_table_path = train_table.get('path')
    stats_path = stats.get('path')
    split_stats_path = split_stats.get('path') if isinstance(split_stats, dict) else None
    train_table_col_stats_path = (
        train_table_col_stats.get('path') if isinstance(train_table_col_stats, dict) else None)
    if not isinstance(train_table_path, str) or not train_table_path:
        raise RuntimeError('diskgraph manifest has invalid train_table.path')
    if train_table.get('format') != 'tt_parquet_dataset_v1':
        raise RuntimeError(
            'diskgraph manifest train_table format unsupported: '
            f'{train_table.get("format")!r}')
    train_table_size_bytes = train_table.get('size_bytes')
    if (isinstance(train_table_size_bytes, bool)
            or not isinstance(train_table_size_bytes, int)
            or train_table_size_bytes < 0):
        raise RuntimeError(
            'diskgraph manifest train_table.size_bytes must be a '
            'non-negative int')
    train_table_row_count = train_table.get('row_count')
    if (isinstance(train_table_row_count, bool)
            or not isinstance(train_table_row_count, int)
            or train_table_row_count < 0):
        raise RuntimeError(
            'diskgraph manifest train_table.row_count must be a '
            'non-negative int')
    if not isinstance(stats_path, str) or not stats_path:
        raise RuntimeError('diskgraph manifest has invalid stats.path')
    if split_stats is not None and (not isinstance(split_stats_path, str) or not split_stats_path):
        raise RuntimeError('diskgraph manifest has invalid split_stats.path')
    if train_table_col_stats is not None and (
            not isinstance(train_table_col_stats_path, str)
            or not train_table_col_stats_path):
        raise RuntimeError('diskgraph manifest has invalid train_table_col_stats.path')

    required_paths = [stats_path]
    if isinstance(split_stats_path, str):
        required_paths.append(split_stats_path)
    if isinstance(train_table_col_stats_path, str):
        required_paths.append(train_table_col_stats_path)
    parsed_train_table_edges = parse_train_table_edges_artifact(train_table_edges)
    fkey_cols = parsed_train_table_edges.get('fkey_cols', {})
    if isinstance(fkey_cols, dict):
        fkey_cols_path = fkey_cols.get('path')
        if isinstance(fkey_cols_path, str):
            required_paths.append(fkey_cols_path)
    mask = parsed_train_table_edges.get('mask', {})
    if isinstance(mask, dict) and bool(mask.get('present')):
        mask_path = mask.get('path')
        if isinstance(mask_path, str):
            required_paths.append(mask_path)
    reverse_edges = parsed_train_table_edges.get('reverse_edges', {})
    if isinstance(reverse_edges, dict):
        for edge_info in reverse_edges.values():
            if isinstance(edge_info, dict):
                edge_path = edge_info.get('path')
                if isinstance(edge_path, str):
                    required_paths.append(edge_path)
    for required_path in required_paths:
        if not fsutil.exists(required_path):
            raise RuntimeError(
                f'diskgraph manifest references missing artifact: {required_path}')

    stats_format = stats.get('format')
    if stats_format != 'tt_stats_json':
        raise RuntimeError(
            'diskgraph manifest stats format unsupported: '
            f'{stats_format}')
    if split_stats is not None:
        split_stats_format = split_stats.get('format')
        if split_stats_format != 'tt_split_stats_json':
            raise RuntimeError(
                'diskgraph manifest split_stats format unsupported: '
                f'{split_stats_format}')
    if train_table_col_stats is not None:
        train_table_col_stats_format = train_table_col_stats.get('format')
        if train_table_col_stats_format != 'tt_train_table_col_stats_json':
            raise RuntimeError(
                'diskgraph manifest train_table_col_stats format unsupported: '
                f'{train_table_col_stats_format}')

    parsed_stats = parse_stats_artifact(stats_path)
    parsed_split_stats = (
        parse_split_stats_artifact(split_stats_path)
        if isinstance(split_stats_path, str) else None
    )
    parsed_train_table_col_stats = (
        parse_train_table_col_stats_artifact(train_table_col_stats_path)
        if isinstance(train_table_col_stats_path, str) else None
    )
    manifest_row_count = manifest.get('row_count')
    if (isinstance(manifest_row_count, bool)
            or not isinstance(manifest_row_count, int)
            or manifest_row_count < 0):
        raise RuntimeError(
            'diskgraph manifest row_count must be a non-negative int')
    if parsed_stats['num_total_rows'] != manifest_row_count:
        raise RuntimeError(
            'stats num_total_rows mismatch with manifest row_count')

    if train_table_row_count != manifest_row_count:
        raise RuntimeError(
            'train_table row_count mismatch with manifest row_count')
    _validate_train_table_dataset(train_table)

    checksums = manifest.get('checksums')
    if not isinstance(checksums, dict):
        raise RuntimeError('diskgraph manifest missing checksums section')

    expected_stats = checksums.get('stats_sha256')
    expected_split_stats = checksums.get('split_stats_sha256')
    expected_train_table_col_stats = checksums.get('train_table_col_stats_sha256')
    expected_train_table_edges = checksums.get('train_table_edges_sha256')
    if not isinstance(expected_stats, str) or not expected_stats:
        raise RuntimeError('diskgraph manifest stats checksum missing')
    if split_stats is not None and not (isinstance(expected_split_stats, str) and expected_split_stats):
        raise RuntimeError('diskgraph manifest split_stats checksum missing')
    if train_table_col_stats is not None and not (
            isinstance(expected_train_table_col_stats, str)
            and expected_train_table_col_stats):
        raise RuntimeError('diskgraph manifest train_table_col_stats checksum missing')
    parsed_reverse_edges = parsed_train_table_edges.get('reverse_edges') if isinstance(
        parsed_train_table_edges, dict) else None
    has_reverse_edges = isinstance(parsed_reverse_edges, dict) and len(parsed_reverse_edges) > 0
    if has_reverse_edges and not (
            isinstance(expected_train_table_edges, str)
            and expected_train_table_edges):
        raise RuntimeError('diskgraph manifest train_table_edges checksum missing')

    actual_stats = _sha256_file(stats_path)
    actual_split_stats = _sha256_file(split_stats_path) if isinstance(split_stats_path, str) else None
    actual_train_table_col_stats = (
        _sha256_file(train_table_col_stats_path)
        if isinstance(train_table_col_stats_path, str) else None)

    if actual_stats != expected_stats:
        raise RuntimeError('diskgraph manifest stats checksum mismatch')
    if split_stats is not None and actual_split_stats != expected_split_stats:
        raise RuntimeError('diskgraph manifest split_stats checksum mismatch')
    if (train_table_col_stats is not None
            and actual_train_table_col_stats != expected_train_table_col_stats):
        raise RuntimeError('diskgraph manifest train_table_col_stats checksum mismatch')
    if isinstance(parsed_train_table_edges, dict):
        reverse_edges = parsed_train_table_edges.get('reverse_edges')
        has_reverse_edges = isinstance(reverse_edges, dict) and len(reverse_edges) > 0
        edge_checksums = parsed_train_table_edges.get('checksums')

        if has_reverse_edges:
            if not isinstance(edge_checksums, dict):
                raise RuntimeError('train_table_edges artifact missing checksums map')
            expected_fkey_cols = edge_checksums.get('fkey_cols_sha256')
            if not isinstance(expected_fkey_cols, str) or not expected_fkey_cols:
                raise RuntimeError('train_table_edges checksum missing fkey_cols_sha256')
            fkey_cols_path = parsed_train_table_edges['fkey_cols']['path']
            if _sha256_path(fkey_cols_path) != expected_fkey_cols:
                raise RuntimeError('train_table_edges fkey_cols checksum mismatch')

            mask_payload = parsed_train_table_edges.get('mask')
            if isinstance(mask_payload, dict) and bool(mask_payload.get('present')):
                expected_mask = edge_checksums.get('mask_sha256')
                if not isinstance(expected_mask, str) or not expected_mask:
                    raise RuntimeError('train_table_edges checksum missing mask_sha256')
                mask_path = mask_payload.get('path')
                if not isinstance(mask_path, str) or _sha256_path(mask_path) != expected_mask:
                    raise RuntimeError('train_table_edges mask checksum mismatch')

            for edge_key, edge_info in reverse_edges.items():
                if not isinstance(edge_info, dict):
                    continue
                edge_path = edge_info.get('path')
                checksum_key = f'{edge_key}_sha256'
                expected_edge = edge_checksums.get(checksum_key)
                if not isinstance(expected_edge, str) or not expected_edge:
                    raise RuntimeError(
                        f'train_table_edges checksum missing {checksum_key}')
                if not isinstance(edge_path, str) or _sha256_path(edge_path) != expected_edge:
                    raise RuntimeError(
                        f'train_table_edges reverse edge checksum mismatch for {edge_key}')

            edge_attrs = parsed_train_table_edges.get('edge_attrs')
            if isinstance(edge_attrs, dict):
                edge_attrs_path = edge_attrs.get('path')
                expected_edge_attrs = edge_checksums.get('edge_attrs_sha256')
                if not isinstance(expected_edge_attrs, str) or not expected_edge_attrs:
                    raise RuntimeError('train_table_edges checksum missing edge_attrs_sha256')
                if (not isinstance(edge_attrs_path, str)
                        or _sha256_path(edge_attrs_path) != expected_edge_attrs):
                    raise RuntimeError('train_table_edges edge_attrs checksum mismatch')

    # Only legacy DiskGraph manifests carry a `split` section; TT-NG has none
    # (stage membership lives in the per-stage seed dirs). Validate when present.
    split = manifest.get('split')
    if split is not None:
        if not isinstance(split, dict):
            raise RuntimeError('diskgraph manifest has invalid split section')
        split_path = split.get('path')
        if not isinstance(split_path, str) or not split_path:
            raise RuntimeError('diskgraph manifest has invalid split.path')
        if not fsutil.exists(split_path):
            raise RuntimeError(
                f'diskgraph manifest references missing artifact: {split_path}')
        split_format = split.get('format')
        if split_format != 'split_assignment_row_stage_bin':
            raise RuntimeError(
                f'diskgraph manifest split format unsupported: {split_format}')
        parsed_split = parse_split_assignment_artifact(split_path)
        if (isinstance(manifest_row_count, int)
                and parsed_split['row_count'] != manifest_row_count):
            raise RuntimeError(
                'split assignment row_count mismatch with manifest row_count')
        expected_split = checksums.get('split_sha256')
        if not isinstance(expected_split, str) or not expected_split:
            raise RuntimeError('diskgraph manifest split checksum missing')
        if _sha256_file(split_path) != expected_split:
            raise RuntimeError('diskgraph manifest split checksum mismatch')
        manifest['parsed_split'] = parsed_split

    fact_pseudo_time = manifest.get('fact_pseudo_time')
    if fact_pseudo_time is not None:
        _validate_fact_pseudo_time(fact_pseudo_time)

    manifest['parsed_stats'] = parsed_stats
    if parsed_split_stats is not None:
        manifest['parsed_split_stats'] = parsed_split_stats
    if parsed_train_table_col_stats is not None:
        manifest['parsed_train_table_col_stats'] = parsed_train_table_col_stats
    if parsed_train_table_edges is not None:
        manifest['parsed_train_table_edges'] = parsed_train_table_edges
    return manifest
