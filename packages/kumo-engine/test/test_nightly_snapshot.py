# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check exact nullable-FK accounting against real native snapshots."""

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.typing import Stype

import kumo.engine as kumo

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_snapshot', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)
CONTEXT = 'fkey:children.parent_id'


def native_snapshot(tmp_path, monkeypatch, references):
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    source = tmp_path / 'source'
    source.mkdir()
    pq.write_table(pa.table({'id': ['a', 'b']}), source / 'parents.parquet')
    pq.write_table(
        pa.table({
            'id': range(len(references)),
            'parent_id': references
        }), source / 'children.parquet')
    connector = kumo.FileConnector(root_dir=str(source))
    parents = kumo.Table.from_source_table(connector['parents.parquet'],
                                           primary_key='id')
    children = kumo.Table.from_source_table(connector['children.parquet'],
                                            primary_key='id').with_column(
                                                'parent_id', stype=Stype.ID)
    graph = kumo.Graph({
        'parents': parents,
        'children': children
    }, [Edge('children', 'parent_id', 'parents')])
    snapshot = graph.snapshot()
    schema = dict(
        tables={
            'parents':
            dict(pkey='id', cols=[dict(name='id')]),
            'children':
            dict(pkey='id', cols=[dict(name='id'),
                                  dict(name='parent_id')])
        }, edges=[
            dict(src_table='children', fkey='parent_id', dst_table='parents')
        ])
    nightly.save(tmp_path / 'schema.json', dict(optional=schema))
    spec = dict(input_root=str(tmp_path), schema='schema.json',
                profile='optional', row_counts={
                    'parents': 2,
                    'children': len(references)
                })
    stats = nightly.read(snapshot.index_path / 'ingest_stats.json')['stats']
    return snapshot, spec, stats


@pytest.mark.parametrize('references,count', [(['a', None, 'b', None], 2),
                                              (['a', 'b'], 0)])
def test_snapshot_accounts_only_frozen_nullable_fk_nulls(
        tmp_path, monkeypatch, references, count):
    snapshot, spec, stats = native_snapshot(tmp_path, monkeypatch, references)
    assert stats['key_parse_failures'] == count
    spec['nullable_foreign_key_null_counts'] = {CONTEXT: count}
    result = nightly.check_snapshot(snapshot, spec)
    assert result['node_counts'] == spec['row_counts']
    assert result['key_failures'] == dict.fromkeys(nightly.KEY_FAILURES, 0)
    assert result['native_key_failures'] == dict(key_parse_failures=count,
                                                 key_lookup_failures=0,
                                                 key_membership_failures=0,
                                                 pk_duplicate_total=0)
    assert result['native_key_failures_by_context'] == stats[
        'key_failures_by_context']
    assert result['nullable_foreign_key_null_counts'] == {CONTEXT: count}


def test_undeclared_nullable_fk_nulls_still_fail(tmp_path, monkeypatch):
    snapshot, spec, _ = native_snapshot(tmp_path, monkeypatch,
                                        ['a', None, 'b', None])
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)


@pytest.mark.parametrize('count', [0, 1, 3, -1, True, 2.0])
def test_nullable_counts_must_match_native_nulls_exactly(
        tmp_path, monkeypatch, count):
    snapshot, spec, _ = native_snapshot(tmp_path, monkeypatch,
                                        ['a', None, 'b', None])
    spec['nullable_foreign_key_null_counts'] = {CONTEXT: count}
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)


@pytest.mark.parametrize('context', [
    'fkey:children.id', 'pkey:children.id', 'fkey:children.unknown',
    'fkey:parents.id'
])
def test_nullable_allowance_requires_declared_non_primary_fk(
        tmp_path, monkeypatch, context):
    snapshot, spec, _ = native_snapshot(tmp_path, monkeypatch, ['a', 'b'])
    spec['nullable_foreign_key_null_counts'] = {context: 0}
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)


@pytest.mark.parametrize('allowance', [None, [], 0])
def test_nullable_allowance_requires_a_mapping(tmp_path, monkeypatch,
                                               allowance):
    snapshot, spec, _ = native_snapshot(tmp_path, monkeypatch, ['a', 'b'])
    spec['nullable_foreign_key_null_counts'] = allowance
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)


def test_non_null_parse_failure_cannot_spend_null_allowance(
        tmp_path, monkeypatch):
    snapshot, spec, stats = native_snapshot(tmp_path, monkeypatch,
                                            ['a', None, 'b', 'x' * 65536])
    assert stats['key_parse_failures'] == 2
    assert stats['key_failures_by_context'][CONTEXT][
        'non_null_parse_failures'] == 0
    assert stats['key_failures_by_context'][
        'dst_lookup:children_parent_id_to_parents_id'][
            'non_null_parse_failures'] == 1
    spec['nullable_foreign_key_null_counts'] = {CONTEXT: 2}
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)


def test_unknown_non_null_fk_still_fails(tmp_path, monkeypatch):
    snapshot, spec, stats = native_snapshot(tmp_path, monkeypatch,
                                            ['a', None, 'b', 'missing'])
    assert stats['key_lookup_failures'] + stats['key_membership_failures'] > 0
    spec['nullable_foreign_key_null_counts'] = {CONTEXT: 1}
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)


def test_primary_key_cannot_receive_a_nullable_fk_allowance(
        tmp_path, monkeypatch):
    snapshot, spec, _ = native_snapshot(tmp_path, monkeypatch, ['a', 'b'])
    schema = nightly.read(tmp_path / 'schema.json')
    schema['optional']['edges'].append(
        dict(src_table='children', fkey='id', dst_table='parents'))
    nightly.save(tmp_path / 'schema.json', schema)
    spec['nullable_foreign_key_null_counts'] = {'fkey:children.id': 0}
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)


@pytest.mark.parametrize('corruption', [
    'primary_context', 'unknown_context', 'missing_context', 'global_remainder',
    'boolean_counter', 'float_counter'
])
def test_snapshot_rejects_unaccounted_native_failure_evidence(
        tmp_path, monkeypatch, corruption):
    snapshot, spec, stats = native_snapshot(tmp_path, monkeypatch,
                                            ['a', None, 'b'])
    spec['nullable_foreign_key_null_counts'] = {CONTEXT: 1}
    contexts = stats['key_failures_by_context']
    if corruption == 'primary_context':
        contexts['pkey:children.id'] = contexts.pop(CONTEXT)
    elif corruption == 'unknown_context':
        contexts['unknown'] = dict(parse_failures=0, lookup_failures=0,
                                   membership_failures=0,
                                   non_null_parse_failures=0)
    elif corruption == 'missing_context':
        contexts.clear()
    else:
        stats['key_parse_failures'] = dict(global_remainder=2,
                                           boolean_counter=True,
                                           float_counter=1.0)[corruption]
    nightly.save(snapshot.index_path / 'ingest_stats.json', dict(stats=stats))
    with pytest.raises(ValueError):
        nightly.check_snapshot(snapshot, spec)
