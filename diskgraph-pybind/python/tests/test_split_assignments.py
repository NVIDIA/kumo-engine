from pathlib import Path
import importlib.util

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1] / 'diskgraph' / 'split_assignments.py'
    spec = importlib.util.spec_from_file_location('diskgraph_split_assignments', root)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_module()

def _write_tt(path: Path, cols: dict):
    table = pa.table({k: pa.array(v) for k, v in cols.items()})
    pq.write_table(table, path)


def test_fixed_split_dense_assignments(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': list(range(10)),
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 11)],
        },
    )

    assignments, meta = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config={
            '_target_': 'kumo.task.split.FixedSplit',
            'ratio': [0.8, 0.1, 0.1],
        },
    )
    assert assignments.shape[0] == 10
    assert meta['authoritative'] is True
    assert int((assignments == 0).sum()) == 8
    assert int((assignments == 1).sum()) == 1
    assert int((assignments == 2).sum()) == 1


def test_random_link_split_not_implemented(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': list(range(5)),
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 6)],
        },
    )

    with pytest.raises(NotImplementedError):
        M.compute_dense_split_assignments(
            train_table_path=str(p),
            split_config={
                '_target_': 'kumo.task.split.RandomLinkSplit',
                'ratio': [0.8, 0.1, 0.1],
                'seed': 1,
            },
        )


def test_column_split_is_stable_for_ties(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': [0, 1, 2, 3, 4, 5],
            'SPLIT_COL': [1, 1, 1, 2, 2, 2],
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 7)],
        },
    )

    assignments, meta = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config={
            '_target_': 'kumo.task.split.ColumnSplit',
            'col': {'name': 'SPLIT_COL'},
            'ratio': [0.5, 0.25, 0.25],
        },
    )
    assert meta['authoritative'] is True
    # With stable sort, ties keep original arange order. First 3 entries are stage 0,
    # next 2 entries stage 1, last entry stage 2.
    assert assignments.tolist() == [0, 0, 0, 1, 1, 2]


def test_ratio_assignment_uses_all_rows_no_unused(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': list(range(10)),
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 11)],
        },
    )

    assignments, _ = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config={
            '_target_': 'kumo.task.split.FixedSplit',
            'ratio': [0.34, 0.33, 0.33],
        },
    )
    assert assignments.shape[0] == 10
    assert int((assignments == 255).sum()) == 0
    assert int((assignments != 255).sum()) == 10


def test_split_family_hint_conflict_fails_fast(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': list(range(5)),
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 6)],
        },
    )

    with pytest.raises(RuntimeError, match='split contract mismatch'):
        M.compute_dense_split_assignments(
            train_table_path=str(p),
            split_config={
                '_target_': 'kumo.task.split.FixedSplit',
                'ratio': [0.8, 0.1, 0.1],
            },
            split_family_hint='TemporalSplit',
        )


def test_manual_split_requires_explicit_train_and_val_values(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': [0, 1, 2],
            'SPLIT_COL': ['a', 'b', 'c'],
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 4)],
        },
    )

    with pytest.raises(RuntimeError, match='ManualSplit requires explicit'):
        M.compute_dense_split_assignments(
            train_table_path=str(p),
            split_config={
                '_target_': 'kumo.task.split.ManualSplit',
                'col': {'name': 'SPLIT_COL'},
            },
        )


def test_entity_include_bit_masks_rows_with_canonical_column_name(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': [0, 1, 2, 3],
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 5)],
            'ENTITY_INCLUDE_BIT': [True, False, True, False],
        },
    )

    assignments, _ = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config={
            '_target_': 'kumo.task.split.FixedSplit',
            'ratio': [1.0, 0.0, 0.0],
        },
    )
    assert assignments.tolist() == [0, 255, 0, 255]
