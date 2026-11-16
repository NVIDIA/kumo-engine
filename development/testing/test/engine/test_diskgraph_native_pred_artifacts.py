# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from kumo.diskgraph.diskgraph_native_pred_artifacts import (
    get_validated_prediction_edge_bundle,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def test_prediction_edge_bundle_validates(tmp_path):
    fkey_cols = tmp_path / 'fkey_cols.parquet'
    mask = tmp_path / 'mask.parquet'
    edge = tmp_path / 'edge.parquet'
    for p, content in (
        (fkey_cols, b'fkey-cols'),
        (mask, b'mask'),
        (edge, b'edge'),
    ):
        p.write_bytes(content)

    bundle = {
        'version': 1,
        'kind': 'tt_prediction_table_edges',
        'format': 'tt_pred_edge_bundle_v1',
        'reverse_edges': {
            'ACCOUNT__rev_ENTITY__pred_table': {
                'path': str(edge),
                'format': 'edge_parquet',
            },
        },
        'fkey_cols': {
            'path': str(fkey_cols),
            'format': 'fkey_cols_parquet',
        },
        'mask': {
            'present': True,
            'path': str(mask),
            'format': 'mask_parquet',
        },
        'checksums': {
            'fkey_cols_sha256': _sha256(fkey_cols),
            'mask_sha256': _sha256(mask),
            'ACCOUNT__rev_ENTITY__pred_table_sha256': _sha256(edge),
        },
    }
    bundle_path = tmp_path / 'pred_edges.json'
    bundle_path.write_text(json.dumps(bundle))

    parsed = get_validated_prediction_edge_bundle(str(bundle_path))
    assert parsed['kind'] == 'tt_prediction_table_edges'


def test_prediction_edge_bundle_requires_edge_checksum(tmp_path):
    fkey_cols = tmp_path / 'fkey_cols.parquet'
    edge = tmp_path / 'edge.parquet'
    fkey_cols.write_bytes(b'fkey-cols')
    edge.write_bytes(b'edge')

    bundle = {
        'version': 1,
        'kind': 'tt_prediction_table_edges',
        'format': 'tt_pred_edge_bundle_v1',
        'reverse_edges': {
            'ACCOUNT__rev_ENTITY__pred_table': {
                'path': str(edge),
                'format': 'edge_parquet',
            },
        },
        'fkey_cols': {
            'path': str(fkey_cols),
            'format': 'fkey_cols_parquet',
        },
        'mask': {
            'present': False,
        },
        'checksums': {
            'fkey_cols_sha256': _sha256(fkey_cols),
        },
    }
    bundle_path = tmp_path / 'pred_edges.json'
    bundle_path.write_text(json.dumps(bundle))

    with pytest.raises(RuntimeError, match='checksum missing'):
        get_validated_prediction_edge_bundle(str(bundle_path))
