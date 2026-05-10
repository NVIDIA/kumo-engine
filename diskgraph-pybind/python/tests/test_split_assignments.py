from pathlib import Path
import importlib.util

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from diskgraph._c import DiskgraphConfigError, DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError



def _load_module():
    root = Path(__file__).resolve().parents[1] / 'diskgraph' / 'split_assignments.py'
    spec = importlib.util.spec_from_file_location('diskgraph_split_assignments', root)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_split_contract():
    import sys
    root = Path(__file__).resolve().parents[1] / 'diskgraph' / 'split_contract.py'
    spec = importlib.util.spec_from_file_location('diskgraph_split_contract', root)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


M = _load_module()
SC = _load_split_contract()
SplitSpec = SC.SplitSpec

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
        split_config=SplitSpec(kind='FixedSplit', ratio=[0.8, 0.1, 0.1]),
    )
    assert assignments.shape[0] == 10
    assert meta['authoritative'] is True
    assert int((assignments == 0).sum()) == 8
    assert int((assignments == 1).sum()) == 1
    assert int((assignments == 2).sum()) == 1


def test_random_link_split_supported_and_seeded(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': list(range(20)),
            'TIMESTAMP': [f'2020-01-{(i % 28) + 1:02d}' for i in range(20)],
        },
    )

    a1, meta1 = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config=SplitSpec(kind='RandomLinkSplit', ratio=[0.8, 0.1, 0.1], rnd_seed=7),
    )
    a2, meta2 = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config=SplitSpec(kind='RandomLinkSplit', ratio=[0.8, 0.1, 0.1], rnd_seed=7),
    )

    assert meta1['authoritative'] is True
    assert meta1['family'] == 'RandomLinkSplit'
    assert meta2['family'] == 'RandomLinkSplit'
    assert a1.tolist() == a2.tolist()
    assert int((a1 == 0).sum()) == 16
    assert int((a1 == 1).sum()) == 2
    assert int((a1 == 2).sum()) == 2


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
        split_config=SplitSpec(kind='ColumnSplit', split_column='SPLIT_COL', ratio=[0.5, 0.25, 0.25]),
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
        split_config=SplitSpec(kind='FixedSplit', ratio=[0.34, 0.33, 0.33]),
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

    with pytest.raises(DiskgraphSystemError, match='split contract mismatch'):
        M.compute_dense_split_assignments(
            train_table_path=str(p),
            split_config=SplitSpec(kind='FixedSplit', ratio=[0.8, 0.1, 0.1]),
            split_family_hint='TemporalSplit',
        )


def test_rejects_dict_split_config(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': list(range(5)),
            'TIMESTAMP': [f'2020-01-{i:02d}' for i in range(1, 6)],
        },
    )

    with pytest.raises(DiskgraphDataError, match='requires split_config as SplitSpec'):
        M.compute_dense_split_assignments(
            train_table_path=str(p),
            split_config={
                '_target_': 'kumo.task.split.FixedSplit',
                'ratio': [0.8, 0.1, 0.1],
            },
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

    with pytest.raises(DiskgraphConfigError, match='ManualSplit requires explicit'):
        M.compute_dense_split_assignments(
            train_table_path=str(p),
            split_config=SplitSpec(kind='ManualSplit', split_column='SPLIT_COL'),
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
        split_config=SplitSpec(kind='FixedSplit', ratio=[1.0, 0.0, 0.0]),
    )
    assert assignments.tolist() == [0, 255, 0, 255]


def test_time_range_split_with_seconds_unit(tmp_path):
    """TimeRangeSplit with timestamp_unit='seconds' correctly converts
    range boundaries from seconds to nanoseconds before comparing against
    parquet timestamps (which are in nanoseconds)."""
    import pandas as pd

    # Create parquet with nanosecond timestamps (Arrow default)
    # 10 rows: 2020-01-01 through 2020-01-10
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': list(range(10)),
            'TIMESTAMP': [d.isoformat() for d in dates],
        },
    )

    # TimeRangeSplit with ranges in epoch seconds:
    #   train: (2020-01-01, 2020-01-07]  -> days 2-7
    #   val:   (2020-01-07, 2020-01-09]  -> days 8-9
    #   test:  (2020-01-09, 2020-01-11]  -> day 10
    # Convention is (start, end] — exclusive start, inclusive end.
    train_start = int(pd.Timestamp('2020-01-01').timestamp())
    train_end = int(pd.Timestamp('2020-01-07').timestamp())
    val_end = int(pd.Timestamp('2020-01-09').timestamp())
    test_end = int(pd.Timestamp('2020-01-11').timestamp())

    assignments, meta = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config=SplitSpec(
            kind='TimeRangeSplit',
            ranges=[(train_start, train_end), (train_end, val_end), (val_end, test_end)],
            timestamp_unit='seconds',
        ),
    )

    assert meta['authoritative'] is True
    assert meta['family'] == 'TimeRangeSplit'
    # Day 1 (2020-01-01) is on the open start boundary -> UNASSIGNED (255)
    # Days 2-7 -> train (0)  [> Jan 1 and <= Jan 7]
    # Days 8-9 -> val (1)    [> Jan 7 and <= Jan 9]
    # Day 10 -> test (2)     [> Jan 9 and <= Jan 11]
    assert assignments.tolist() == [255, 0, 0, 0, 0, 0, 0, 1, 1, 2]


def test_temporal_split_uses_configured_time_col(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': [0, 1, 2, 3],
            'PSEUDO_TIMESTAMP': [
                '2020-01-01',
                '2020-01-02',
                '2020-01-03',
                '2020-01-04',
            ],
        },
    )

    assignments, meta = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config=SplitSpec(kind='TemporalSplit', ratio=[0.5, 0.25, 0.25]),
        time_col='PSEUDO_TIMESTAMP',
    )

    assert meta['authoritative'] is True
    assert meta['family'] == 'TemporalSplit'
    assert assignments.tolist() == [0, 0, 1, 2]


def test_time_range_split_timestamp_unit_in_split_spec():
    """SplitSpec includes timestamp_unit in to_assignment_config()."""
    spec = SplitSpec(
        kind='TimeRangeSplit',
        ranges=[(1000, 2000), (2000, 3000)],
        timestamp_unit='seconds',
    )
    cfg = spec.to_assignment_config()
    assert cfg['timestamp_unit'] == 'seconds'

    spec2 = SplitSpec(kind='FixedSplit', ratio=[0.8, 0.1, 0.1])
    cfg2 = spec2.to_assignment_config()
    assert cfg2['timestamp_unit'] == 'seconds'  # default


def test_custom_split_allowlisted_by_contract():
    SplitSpec(
        kind='CustomSplit',
        split_column='users.bucket',
        split_output_column='__kumo_internal_split_column__',
        manual_train_values=['train'],
        manual_val_values=['val'],
        manual_test_values=['test'],
    ).validate_for_compiler()


def test_custom_link_split_allowlisted_by_contract():
    SplitSpec(
        kind='CustomLinkSplit',
        split_column='orders.bucket',
        split_output_column='__kumo_internal_split_column__',
        manual_train_values=['train'],
        manual_val_values=['val'],
        manual_test_values=['test'],
    ).validate_for_compiler()


def test_custom_split_rehydrates_via_manual_assigner(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(
        p,
        {
            'arange': [0, 1, 2, 3, 4, 5],
            '__kumo_internal_split_column__': ['train', 'train', 'val', 'test', 'train', 'val'],
        },
    )
    assignments, meta = M.compute_dense_split_assignments(
        train_table_path=str(p),
        split_config=SplitSpec(
            kind='CustomSplit',
            split_column='users.bucket',
            split_output_column='__kumo_internal_split_column__',
            manual_train_values=['train'],
            manual_val_values=['val'],
            manual_test_values=['test'],
        ),
    )
    assert meta['family'] == 'CustomSplit'
    assert meta['authoritative'] is True
    assert assignments.tolist() == [0, 0, 1, 2, 0, 1]


def test_custom_link_split_rehydration_is_unsupported(tmp_path):
    p = tmp_path / 'tt.parquet'
    _write_tt(p, {'arange': [0, 1], 'SPLIT_COL': ['train', 'val']})
    with pytest.raises(DiskgraphConfigError, match='CustomLinkSplit cannot be rehydrated'):
        M.compute_dense_split_assignments(
            train_table_path=str(p),
            split_config=SplitSpec(
                kind='CustomLinkSplit',
                split_column='orders.bucket',
                manual_train_values=['train'],
                manual_val_values=['val'],
                manual_test_values=['test'],
            ),
        )
