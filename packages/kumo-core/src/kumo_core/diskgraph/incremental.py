# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact source identities for immutable incremental snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def warn_encoder_drift(table: str, previous: Mapping, current: Mapping) -> None:
    """Compare native column statistics without changing fitted encoders."""
    for key, stat in previous.items():
        if '__kumo.categorical__CATEGORY_COUNTS__False' in key and key in current:
            known = {value for value, _ in stat.get('value', [])}
            added = {value
                     for value, _ in current[key].get('value', [])} - known
            if len(added) > len(known) * 0.05:
                warnings.warn(
                    f'{table}.{key.rsplit("__", 3)[0]} has more than 5% new categories; '
                    'encoder refitting may improve quality.', UserWarning,
                    stacklevel=2)
        if '__kumo.numerical__MIN__False' in key and key in current:
            maximum = key.replace('__MIN__False', '__MAX__False')
            values = [
                stat.get('value'),
                previous.get(maximum, {}).get('value'),
                current[key].get('value'),
                current.get(maximum, {}).get('value')
            ]
            if all(isinstance(value, (int, float)) for value in values):
                low, high, new_low, new_high = values
                if max(high, new_high) - min(low, new_low) > 2 * (high - low):
                    warnings.warn(
                        f'{table}.{key.rsplit("__", 3)[0]} numerical range expanded more than 2x; '
                        'encoder refitting may improve quality.', UserWarning,
                        stacklevel=2)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True,
                   separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class SourceFingerprint:
    table_name: str
    row_count: int | None
    schema_hash: str
    content_hash: str
    partition_fingerprints: Mapping[str, str] = field(default_factory=dict)
    partition_sizes: Mapping[str, int] = field(default_factory=dict)
    parquet_row_groups: Mapping[str, list[dict]] = field(default_factory=dict)


def _file_hashes(path, metadata=None):
    """Compute whole-file and compressed row-group identities in one scan."""
    groups, chunks = [], []
    if metadata is not None:
        for i in range(metadata.num_row_groups):
            group = metadata.row_group(i)
            columns = []
            for j in range(group.num_columns):
                column = group.column(j)
                start = min(offset for offset in (column.dictionary_page_offset,
                                                  column.data_page_offset)
                            if offset is not None and offset >= 0)
                digest = hashlib.sha256()
                chunks.append(
                    (start, start + column.total_compressed_size, digest))
                columns.append(
                    (digest,
                     (column.path_in_schema, column.physical_type,
                      column.compression, column.encodings, column.num_values)))
            groups.append((group.num_rows, columns))
    chunks.sort(key=lambda chunk: chunk[0])
    digest = hashlib.sha256()
    offset = cursor = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
            end = offset + len(block)
            index = cursor
            while index < len(chunks) and chunks[index][0] < end:
                start, stop, chunk_hash = chunks[index]
                if stop > offset:
                    chunk_hash.update(
                        memoryview(block)[max(start -
                                              offset, 0):min(stop, end) -
                                          offset])
                index += 1
            offset = end
            while cursor < len(chunks) and chunks[cursor][1] <= offset:
                cursor += 1
    if any(stop > offset for _, stop, _ in chunks):
        raise RuntimeError(f'Truncated Parquet source: {path}')
    return digest.hexdigest(), [{
        'rows':
        rows,
        'hash':
        _hash([(chunk.hexdigest(), decoding) for chunk, decoding in columns])
    } for rows, columns in groups]


def fingerprint_files(table_name: str, root: Path, *,
                      include_row_groups: bool = True) -> SourceFingerprint:
    """Hash exact CSV/Parquet bytes without decoding the table.

    Parquet row counts and schemas come from footers. CSV has no row-count
    metadata, so its count is unknown. Size/mtime alone are deliberately not
    content identities: same-size edits and restored mtimes must invalidate.
    A concurrent writer invalidates this observation instead of publishing a
    mixed fingerprint. Call again after staging to guard the copy interval.
    """
    import pyarrow.parquet as pq
    from kumo_core.sources.layout import is_hidden

    root = Path(root).resolve(strict=True)

    def files() -> list[Path]:
        if root.is_file():
            return [root]
        return sorted(
            p for p in root.rglob('*')
            if p.is_file() and not is_hidden(p.relative_to(root).as_posix())
            and p.suffix.lower() in ('.csv', '.parquet'))

    paths = files()
    if not paths:
        raise ValueError(f'No CSV or Parquet files in {root}')

    def signature(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    signatures = {p: signature(p) for p in paths}
    partitions, schemas, sizes, groups = {}, {}, {}, {}
    row_count: int | None = 0
    for path in paths:
        key = path.name if root.is_file() else path.relative_to(root).as_posix()
        metadata = pq.read_metadata(
            path) if path.suffix.lower() == '.parquet' else None
        partitions[key], row_groups = _file_hashes(
            path, metadata if include_row_groups else None)
        sizes[key] = signatures[path][0]
        if path.suffix.lower() == '.parquet':
            schema = metadata.schema.to_arrow_schema().remove_metadata()
            schemas[key] = [(f.name, str(f.type), f.nullable) for f in schema]
            groups[key] = row_groups
            if row_count is not None:
                row_count += metadata.num_rows
        elif path.suffix.lower() == '.csv':
            with path.open(newline='', encoding='utf-8-sig') as stream:
                schemas[key] = next(csv.reader(stream))
            row_count = None
        else:
            raise ValueError(f'Unsupported source file: {path}')
    if files() != paths or any(
            signature(p) != s for p, s in signatures.items()):
        raise RuntimeError(f'Source changed while fingerprinting {table_name}')
    # Schema identity does not depend on the number or names of partitions.
    schemas_seen = sorted({_hash(schema) for schema in schemas.values()})
    return SourceFingerprint(table_name, row_count, _hash(schemas_seen),
                             _hash(partitions), partitions, sizes, groups)


@dataclass(frozen=True)
class AppendPartition:
    """New source content; CSV starts at a byte offset, Parquet at a row group."""
    path: str
    byte_offset: int = 0
    row_group: int = 0


def detect_append(root: Path, previous: SourceFingerprint,
                  current: SourceFingerprint) -> list[AppendPartition] | None:
    """Return proven additions, or None when an update requires re-encoding.

    Exact historical hashes are required. Repartitioned/recompressed Parquet
    conservatively falls back even when its logical rows happen to be equal.
    Late arrivals are allowed: the runtime must apply temporal filtering.
    """
    if previous.schema_hash != current.schema_hash:
        return None
    old = previous.partition_fingerprints
    new = current.partition_fingerprints
    if not old.keys() <= new.keys():
        return None
    additions = []
    for name, content_hash in new.items():
        path = root if root.is_file() else root / name
        if name not in old:
            additions.append(AppendPartition(str(path)))
        elif content_hash == old[name]:
            continue
        elif path.suffix.lower() == '.csv':
            size = previous.partition_sizes.get(name, 0)
            if not size or size >= current.partition_sizes[name]:
                return None
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                remaining = size
                last = b''
                while remaining:
                    block = stream.read(min(remaining, 8 * 1024 * 1024))
                    if not block:
                        return None
                    digest.update(block)
                    remaining -= len(block)
                    last = block[-1:]
            if digest.hexdigest() != old[name] or last != b'\n':
                return None
            additions.append(AppendPartition(str(path), byte_offset=size))
        else:
            before = previous.parquet_row_groups.get(name, [])
            after = current.parquet_row_groups.get(name, [])
            if not before or len(after) <= len(
                    before) or after[:len(before)] != before:
                return None
            additions.append(AppendPartition(str(path), row_group=len(before)))
    return additions
