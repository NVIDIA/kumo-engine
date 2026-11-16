# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the short-circuit gate
(``_bundle_destination_already_materialized``) used by the BP and training
edge-bundle materialization wrappers.

When the predict/training edge-bundle materialization wrappers run in the
predict subprocess, they may trigger a re-materialize of the edge bundle.
The destination edges were already populated by an upstream
materialize_*_task activity (and live on dataplane). The bundle
manifest's source paths point at the upstream pod's local SSD, which
is unreachable from the predict pod — re-copying would raise
FileNotFoundError. The short-circuit detects "destination already
populated" via the ``__SUCCESS_EDGES__`` marker and returns True
without touching the source files.
"""
from dataclasses import dataclass, field
from typing import List

from kumo.data.generic.edge_materialize import (
    _FKEY_COLS_FILE,
    _GRAPHSTORE_SUCCESS_FILE,
    _MASK_PARQUET_FILE,
    _bundle_destination_already_materialized,
)


@dataclass
class _FakeTable:
    name: str
    fkeys: List[object] = field(default_factory=lambda: [object()])


def _make_dst(graph_path, table_name: str, edge_keys: List[str],
              with_mask: bool = True) -> None:
    """Create a fully-materialized destination layout under graph_path."""
    (graph_path / f'{table_name}_{_FKEY_COLS_FILE}').touch()
    if with_mask:
        (graph_path / f'{table_name}_{_MASK_PARQUET_FILE}').touch()
    for ek in edge_keys:
        edge_dir = graph_path / ek
        edge_dir.mkdir(parents=True, exist_ok=True)
        (edge_dir / _GRAPHSTORE_SUCCESS_FILE).touch()


def _make_bundle(edge_keys: List[str], with_mask: bool = True) -> dict:
    """Build a minimal bundle dict pointing at /does/not/exist sources.

    The whole point of the short-circuit is that we never resolve these
    source paths when the destination is already populated.
    """
    return {
        'reverse_edges': {
            ek: {'path': '/does/not/exist/edge.parquet'}
            for ek in edge_keys
        },
        'fkey_cols': {'path': '/does/not/exist/fkey_cols.parquet'},
        'mask': {
            'present': with_mask,
            'path': ('/does/not/exist/mask.parquet' if with_mask else None),
        },
    }


def test_skip_when_all_destinations_present(tmp_path):
    table = _FakeTable(name='train_table')
    edge_keys = ['ACCOUNT__rev_ENTITY__train_table']
    _make_dst(tmp_path, table.name, edge_keys, with_mask=True)
    bundle = _make_bundle(edge_keys, with_mask=True)
    edge_persist_paths = {ek: str(tmp_path / ek) for ek in edge_keys}

    # Bundle source paths are bogus; short-circuit must succeed without
    # touching them.
    assert _bundle_destination_already_materialized(
        bundle, table, str(tmp_path), edge_persist_paths) is True


def test_no_skip_when_success_marker_missing(tmp_path):
    table = _FakeTable(name='train_table')
    edge_keys = ['ACCOUNT__rev_ENTITY__train_table']
    # Only sidecars; per-edge SUCCESS marker absent → not short-circuited.
    (tmp_path / f'{table.name}_{_FKEY_COLS_FILE}').touch()
    (tmp_path / f'{table.name}_{_MASK_PARQUET_FILE}').touch()
    bundle = _make_bundle(edge_keys, with_mask=True)
    edge_persist_paths = {ek: str(tmp_path / ek) for ek in edge_keys}

    assert _bundle_destination_already_materialized(
        bundle, table, str(tmp_path), edge_persist_paths) is False


def test_no_skip_when_fkey_cols_missing(tmp_path):
    table = _FakeTable(name='train_table')
    edge_keys = ['ACCOUNT__rev_ENTITY__train_table']
    # Per-edge SUCCESS + mask present, but fkey_cols sidecar missing.
    (tmp_path / f'{table.name}_{_MASK_PARQUET_FILE}').touch()
    for ek in edge_keys:
        (tmp_path / ek).mkdir(parents=True, exist_ok=True)
        (tmp_path / ek / _GRAPHSTORE_SUCCESS_FILE).touch()
    bundle = _make_bundle(edge_keys, with_mask=True)
    edge_persist_paths = {ek: str(tmp_path / ek) for ek in edge_keys}

    assert _bundle_destination_already_materialized(
        bundle, table, str(tmp_path), edge_persist_paths) is False


def test_skip_when_mask_absent_in_bundle_and_dst(tmp_path):
    table = _FakeTable(name='train_table')
    edge_keys = ['ACCOUNT__rev_ENTITY__train_table']
    _make_dst(tmp_path, table.name, edge_keys, with_mask=False)
    bundle = _make_bundle(edge_keys, with_mask=False)
    edge_persist_paths = {ek: str(tmp_path / ek) for ek in edge_keys}

    assert _bundle_destination_already_materialized(
        bundle, table, str(tmp_path), edge_persist_paths) is True


def test_no_skip_when_reverse_edges_empty(tmp_path):
    # Defensive: bundle with no reverse_edges short-circuits to False so
    # the wrapper falls through to the existing no-fkeys log path inside
    # _materialize_edges_from_bundle.
    table = _FakeTable(name='train_table')
    (tmp_path / f'{table.name}_{_FKEY_COLS_FILE}').touch()
    (tmp_path / f'{table.name}_{_MASK_PARQUET_FILE}').touch()
    bundle = {'reverse_edges': {}, 'fkey_cols': {}, 'mask': {'present': True}}
    assert _bundle_destination_already_materialized(
        bundle, table, str(tmp_path), {}) is False


def test_no_skip_when_mask_expected_but_dst_missing(tmp_path):
    table = _FakeTable(name='train_table')
    edge_keys = ['ACCOUNT__rev_ENTITY__train_table']
    # Bundle says mask is present, but mask sidecar isn't materialized.
    (tmp_path / f'{table.name}_{_FKEY_COLS_FILE}').touch()
    for ek in edge_keys:
        (tmp_path / ek).mkdir(parents=True, exist_ok=True)
        (tmp_path / ek / _GRAPHSTORE_SUCCESS_FILE).touch()
    bundle = _make_bundle(edge_keys, with_mask=True)
    edge_persist_paths = {ek: str(tmp_path / ek) for ek in edge_keys}

    assert _bundle_destination_already_materialized(
        bundle, table, str(tmp_path), edge_persist_paths) is False
