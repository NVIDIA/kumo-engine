# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from io_utils import fsutil


def _sha256_file(path: str) -> str:
    data = fsutil.read_binary(path)
    return hashlib.sha256(data).hexdigest()


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


def parse_prediction_table_edges_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError('prediction_table_edges artifact must be a JSON object')

    version = payload.get('version')
    kind = payload.get('kind')
    fmt = payload.get('format')
    if (version, kind, fmt) != (1, 'tt_prediction_table_edges', 'tt_pred_edge_bundle_v1'):
        raise RuntimeError(
            'prediction_table_edges artifact version/kind/format mismatch: '
            f'version={version!r}, kind={kind!r}, format={fmt!r}')

    reverse_edges = payload.get('reverse_edges')
    if not isinstance(reverse_edges, dict):
        raise RuntimeError('prediction_table_edges artifact missing reverse_edges map')

    for edge_key, edge_info in reverse_edges.items():
        if not isinstance(edge_key, str) or not edge_key:
            raise RuntimeError('prediction_table_edges artifact has invalid edge key')
        if not isinstance(edge_info, dict):
            raise RuntimeError(
                f'prediction_table_edges artifact reverse edge {edge_key!r} must be object')
        edge_path = edge_info.get('path')
        edge_format = edge_info.get('format')
        if not isinstance(edge_path, str) or not edge_path:
            raise RuntimeError(
                f'prediction_table_edges artifact reverse edge {edge_key!r} missing path')
        if edge_format != 'edge_parquet':
            raise RuntimeError(
                f'prediction_table_edges artifact reverse edge {edge_key!r} '
                f'format unsupported: {edge_format!r}')

    fkey_cols = payload.get('fkey_cols')
    if reverse_edges:
        if not isinstance(fkey_cols, dict):
            raise RuntimeError('prediction_table_edges artifact missing fkey_cols object')
        fkey_cols_path = fkey_cols.get('path')
        if not isinstance(fkey_cols_path, str) or not fkey_cols_path:
            raise RuntimeError('prediction_table_edges artifact has invalid fkey_cols.path')
        if fkey_cols.get('format') != 'fkey_cols_parquet':
            raise RuntimeError('prediction_table_edges artifact fkey_cols format unsupported')

    mask = payload.get('mask')
    if reverse_edges:
        if not isinstance(mask, dict):
            raise RuntimeError('prediction_table_edges artifact missing mask object')
        mask_present = mask.get('present')
        if not isinstance(mask_present, bool):
            raise RuntimeError('prediction_table_edges artifact mask.present must be bool')
        if mask_present:
            if not isinstance(mask.get('path'), str) or not mask.get('path'):
                raise RuntimeError('prediction_table_edges artifact mask.path missing')
            if mask.get('format') != 'mask_parquet':
                raise RuntimeError('prediction_table_edges artifact mask format unsupported')

    return payload


def get_validated_prediction_edge_bundle(bundle_path: str) -> dict[str, Any]:
    payload = json.loads(fsutil.read_text(bundle_path))
    parsed = parse_prediction_table_edges_artifact(payload)

    checksums = parsed.get('checksums')
    if not isinstance(checksums, dict):
        raise RuntimeError('prediction_table_edges artifact missing checksums map')

    fkey_cols = parsed.get('fkey_cols', {})
    if isinstance(fkey_cols, dict):
        fkey_cols_path = fkey_cols.get('path')
        expected = checksums.get('fkey_cols_sha256')
        if not isinstance(expected, str) or not expected:
            raise RuntimeError('prediction_table_edges checksum missing fkey_cols_sha256')
        if not isinstance(fkey_cols_path, str) or _sha256_path(fkey_cols_path) != expected:
            raise RuntimeError('prediction_table_edges fkey_cols checksum mismatch')

    mask = parsed.get('mask', {})
    if isinstance(mask, dict) and bool(mask.get('present')):
        expected = checksums.get('mask_sha256')
        mask_path = mask.get('path')
        if not isinstance(expected, str) or not expected:
            raise RuntimeError('prediction_table_edges checksum missing mask_sha256')
        if not isinstance(mask_path, str) or _sha256_path(mask_path) != expected:
            raise RuntimeError('prediction_table_edges mask checksum mismatch')

    reverse_edges = parsed.get('reverse_edges', {})
    if isinstance(reverse_edges, dict):
        for edge_key, edge_info in reverse_edges.items():
            if not isinstance(edge_info, dict):
                continue
            edge_path = edge_info.get('path')
            checksum_key = f'{edge_key}_sha256'
            expected = checksums.get(checksum_key)
            if not isinstance(expected, str) or not expected:
                raise RuntimeError(
                    f'prediction_table_edges checksum missing {checksum_key}')
            if not isinstance(edge_path, str) or _sha256_path(edge_path) != expected:
                raise RuntimeError(
                    'prediction_table_edges reverse edge checksum mismatch '
                    f'for {edge_key}')

    return parsed
