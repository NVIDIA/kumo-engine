from __future__ import annotations
try:
    from ._c import DiskgraphDataError, DiskgraphConfigError, DiskgraphSystemError
except ImportError:
    try:
        from diskgraph._c import DiskgraphDataError, DiskgraphConfigError, DiskgraphSystemError
    except ImportError:
        DiskgraphDataError = RuntimeError
        DiskgraphConfigError = RuntimeError
        DiskgraphSystemError = RuntimeError

from typing import Any, Optional

import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

_RUST_SPLIT_IMPORT_ERROR: Optional[Exception] = None

_rust_mod = None
try:
    from . import _c as _rust_mod
except Exception:
    try:
        from diskgraph import _c as _rust_mod
    except Exception as exc:  # pragma: no cover - stale/missing wheel should fail fast
        _RUST_SPLIT_IMPORT_ERROR = exc

apply_entity_include_mask = getattr(_rust_mod, 'apply_entity_include_mask', None)
assign_approx_date_offset_split = getattr(_rust_mod, 'assign_approx_date_offset_split', None)
assign_backtesting_split = getattr(_rust_mod, 'assign_backtesting_split', None)
assign_by_ratio_on_perm = getattr(_rust_mod, 'assign_by_ratio_on_perm', None)
assign_date_offset_split = getattr(_rust_mod, 'assign_date_offset_split', None)
assign_manual_split = getattr(_rust_mod, 'assign_manual_split', None)
assign_time_range_split = getattr(_rust_mod, 'assign_time_range_split', None)
compute_dense_split_assignments_core = getattr(_rust_mod, 'compute_dense_split_assignments_core', None)

_UNUSED_STAGE_ID = 255
_ENTITY_INCLUDE_BIT_COL = 'ENTITY_INCLUDE_BIT'


def _require_rust_split_core() -> None:
    if _RUST_SPLIT_IMPORT_ERROR is not None:
        raise DiskgraphSystemError(
            'diskgraph split assignment requires diskgraph._c Rust extension; '
            f'import failed: {_RUST_SPLIT_IMPORT_ERROR}'
        )


def _require_rust_fn(fn: Any, fn_name: str) -> Any:
    _require_rust_split_core()
    if fn is None:
        raise DiskgraphSystemError(
            f'diskgraph split assignment requires Rust function `{fn_name}` in diskgraph._c; '
            'rebuild/reinstall diskgraph wheel'
        )
    return fn


def _is_file_path(path: str) -> bool:
    return os.path.isfile(path)


def _read_parquet_columns(path: str, required: list[str]) -> pd.DataFrame:
    if _is_file_path(path):
        schema = pq.read_schema(path)
        missing = [c for c in required if c not in schema.names]
        if missing:
            raise DiskgraphDataError(
                f'diskgraph native split requires missing columns: {missing}')
        return pq.read_table(path, columns=required).to_pandas()

    dataset = ds.dataset(path, format='parquet')
    missing = [c for c in required if c not in dataset.schema.names]
    if missing:
        raise DiskgraphDataError(
            f'diskgraph native split requires missing columns: {missing}')
    return dataset.to_table(columns=required).to_pandas()


def _parquet_row_count(path: str) -> int:
    if _is_file_path(path):
        return int(pq.read_metadata(path).num_rows)
    return int(ds.dataset(path, format='parquet').count_rows())


def _parquet_schema_names(path: str) -> list[str]:
    if _is_file_path(path):
        return list(pq.read_schema(path).names)
    return list(ds.dataset(path, format='parquet').schema.names)


def _as_list(v: Any) -> list[Any]:
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return [v]


def _infer_datetime_series(ser: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(ser):
        return ser
    return pd.to_datetime(ser, errors='raise')


def _to_uint8_array(values: Any) -> np.ndarray:
    if isinstance(values, (bytes, bytearray, memoryview)):
        return np.frombuffer(values, dtype=np.uint8).copy()
    return np.asarray(values, dtype=np.uint8)


def _assign_for_temporal_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    ratio = [float(x) for x in split_cfg.get('ratio', [0.8, 0.1, 0.1])]
    df = _read_parquet_columns(train_table_path, ['arange', 'TIMESTAMP']).sort_values('arange', kind='mergesort')
    ts = _infer_datetime_series(df['TIMESTAMP'])
    # use ns epoch int for deterministic ordering in Rust core
    ts_ns = ts.astype('int64').to_numpy(dtype=np.int64)
    dense_core = _require_rust_fn(compute_dense_split_assignments_core, 'compute_dense_split_assignments_core')
    out = dense_core(
        family='TemporalSplit',
        row_count=len(df),
        ratio=ratio,
        timestamps=ts_ns.tolist(),
    )
    return _to_uint8_array(out)


def _assign_for_fixed_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    ratio = [float(x) for x in split_cfg.get('ratio', [0.8, 0.1, 0.1])]
    row_count = _parquet_row_count(train_table_path)
    dense_core = _require_rust_fn(compute_dense_split_assignments_core, 'compute_dense_split_assignments_core')
    out = dense_core(
        family='FixedSplit',
        row_count=row_count,
        ratio=ratio,
    )
    return _to_uint8_array(out)


def _assign_for_random_split(
    split_cfg: dict[str, Any],
    train_table_path: str,
    *,
    family: str = 'RandomSplit',
) -> np.ndarray:
    _require_rust_split_core()
    ratio = [float(x) for x in split_cfg.get('ratio', [0.8, 0.1, 0.1])]
    seed = int(split_cfg.get('rnd_seed', 0))
    row_count = _parquet_row_count(train_table_path)
    dense_core = _require_rust_fn(compute_dense_split_assignments_core, 'compute_dense_split_assignments_core')
    out = dense_core(
        family=family,
        row_count=row_count,
        ratio=ratio,
        seed=seed,
    )
    return _to_uint8_array(out)


def _assign_for_column_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    ratio = [float(x) for x in split_cfg.get('ratio', [0.8, 0.1, 0.1])]
    col_spec = split_cfg.get('col')
    col_name = col_spec.get('name') if isinstance(col_spec, dict) else col_spec
    if not isinstance(col_name, str) or not col_name:
        raise DiskgraphConfigError('ColumnSplit requires split.col name')
    df = _read_parquet_columns(train_table_path, ['arange', col_name]).sort_values('arange', kind='mergesort')
    perm = np.argsort(df[col_name].to_numpy(), kind='mergesort').astype(np.int64)
    assign_on_perm = _require_rust_fn(assign_by_ratio_on_perm, 'assign_by_ratio_on_perm')
    out = assign_on_perm(
        row_count=len(df),
        ratio=ratio,
        perm=perm.tolist(),
    )
    return _to_uint8_array(out)


def _assign_for_manual_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    col_spec = split_cfg.get('col')
    col_name = (
        split_cfg.get('split_output_column')
        or split_cfg.get('split_projection')
        or (col_spec.get('name') if isinstance(col_spec, dict) else col_spec)
    )
    if not isinstance(col_name, str) or not col_name:
        raise DiskgraphConfigError('ManualSplit requires split.col name')

    df = _read_parquet_columns(train_table_path, ['arange', col_name]).sort_values('arange', kind='mergesort')
    train_raw = split_cfg.get('manual_train_values')
    val_raw = split_cfg.get('manual_val_values')
    if train_raw is None or val_raw is None:
        raise DiskgraphConfigError(
            'ManualSplit requires explicit manual_train_values and manual_val_values'
        )
    train_vals = [str(v) for v in _as_list(train_raw)]
    val_vals = [str(v) for v in _as_list(val_raw)]
    if not train_vals or not val_vals:
        raise DiskgraphConfigError('ManualSplit requires non-empty train and val values')
    test_raw = split_cfg.get('manual_test_values')
    test_vals = [str(v) for v in (_as_list(test_raw) if test_raw is not None else [])]

    values = df[col_name].astype(str).to_numpy(dtype=object).tolist()
    manual_assign = _require_rust_fn(assign_manual_split, 'assign_manual_split')
    out = manual_assign(
        values=values,
        train_values=train_vals,
        val_values=val_vals,
        test_values=test_vals,
    )
    return _to_uint8_array(out)


def _assign_for_backtesting_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    ts_raw = split_cfg.get('backtesting_timestamp')
    if ts_raw is None:
        raise DiskgraphConfigError('BacktestingSplit requires backtesting_timestamp')
    df = _read_parquet_columns(train_table_path, ['arange', 'TIMESTAMP']).sort_values('arange', kind='mergesort')
    time_ser = _infer_datetime_series(df['TIMESTAMP'])
    ts_ns = time_ser.astype('int64').to_numpy(dtype=np.int64)
    timestamp_unit = str(split_cfg.get('timestamp_unit', 'seconds'))
    _UNIT_TO_NS = {
        'seconds': 1_000_000_000,
        'milliseconds': 1_000_000,
        'microseconds': 1_000,
        'nanoseconds': 1,
    }
    if isinstance(ts_raw, (int, float)) and timestamp_unit in _UNIT_TO_NS:
        target_ns = int(ts_raw) * _UNIT_TO_NS[timestamp_unit]
    else:
        target_ns = int(pd.Timestamp(ts_raw).value)
    backtesting_assign = _require_rust_fn(assign_backtesting_split, 'assign_backtesting_split')
    out = backtesting_assign(
        timestamps_ns=ts_ns.tolist(),
        target_timestamp_ns=target_ns,
    )
    return _to_uint8_array(out)


def _assign_for_time_range_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    ranges = split_cfg.get('ranges') or []
    if not isinstance(ranges, list) or len(ranges) < 2:
        raise DiskgraphConfigError('TimeRangeSplit requires at least two ranges')

    df = _read_parquet_columns(train_table_path, ['arange', 'TIMESTAMP']).sort_values('arange', kind='mergesort')
    time_ser = _infer_datetime_series(df['TIMESTAMP'])
    ts_ns = time_ser.astype('int64').to_numpy(dtype=np.int64)

    timestamp_unit = str(split_cfg.get('timestamp_unit', 'seconds'))
    _UNIT_TO_NS = {
        'seconds': 1_000_000_000,
        'milliseconds': 1_000_000,
        'microseconds': 1_000,
        'nanoseconds': 1,
    }

    starts: list[int] = []
    ends: list[int] = []
    for split_idx, pair in enumerate(ranges):
        if split_idx > 2:
            break
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise DiskgraphConfigError('TimeRangeSplit range entries must be [start, end]')
        raw_start, raw_end = pair
        if isinstance(raw_start, (int, float)) and timestamp_unit in _UNIT_TO_NS:
            starts.append(int(raw_start) * _UNIT_TO_NS[timestamp_unit])
            ends.append(int(raw_end) * _UNIT_TO_NS[timestamp_unit])
        else:
            starts.append(int(pd.Timestamp(raw_start).value))
            ends.append(int(pd.Timestamp(raw_end).value))

    timeframe = split_cfg.get('timeframe', None)
    unit = str(split_cfg.get('unit', 'days')).lower()
    anchor_start_offset = int(split_cfg.get('anchor_start_offset', 0) or 0)

    time_range_assign = _require_rust_fn(assign_time_range_split, 'assign_time_range_split')
    out = time_range_assign(
        timestamps_ns=ts_ns.tolist(),
        ranges_start_ns=starts,
        ranges_end_ns=ends,
        timeframe=(int(timeframe) if timeframe is not None else None),
        unit=unit,
        anchor_start_offset=anchor_start_offset,
    )
    return _to_uint8_array(out)


def _assign_for_date_offset_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    offsets_raw = split_cfg.get('offsets')
    if not isinstance(offsets_raw, list) or len(offsets_raw) < 1:
        raise DiskgraphConfigError('DateOffsetSplit requires offsets')
    offsets = [int(x) for x in offsets_raw]

    offset_unit = str(split_cfg.get('offset_unit', 'days')).lower()
    unit = str(split_cfg.get('unit', 'days')).lower()
    timeframe = split_cfg.get('timeframe', None)
    timeframe_i = int(timeframe) if timeframe is not None else None
    anchor_start_offset = int(split_cfg.get('anchor_start_offset', 0) or 0)

    df = _read_parquet_columns(train_table_path, ['arange', 'TIMESTAMP']).sort_values('arange', kind='mergesort')
    time_ser = _infer_datetime_series(df['TIMESTAMP'])
    ts_ns = time_ser.astype('int64').to_numpy(dtype=np.int64)

    date_offset_assign = _require_rust_fn(assign_date_offset_split, 'assign_date_offset_split')
    out = date_offset_assign(
        timestamps_ns=ts_ns.tolist(),
        offsets=offsets,
        offset_unit=offset_unit,
        unit=unit,
        timeframe=timeframe_i,
        anchor_start_offset=anchor_start_offset,
    )
    return _to_uint8_array(out)


def _assign_for_approx_date_offset_split(split_cfg: dict[str, Any], train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    ratio = [float(x) for x in split_cfg.get('ratio', [0.8, 0.1, 0.1])]
    if len(ratio) < 2 or len(ratio) > 3:
        raise DiskgraphConfigError('ApproxDateOffsetSplit requires 2 or 3 ratios')

    df = _read_parquet_columns(train_table_path, ['arange', 'TIMESTAMP']).sort_values('arange', kind='mergesort')
    ts = _infer_datetime_series(df['TIMESTAMP']).astype('int64').to_numpy(dtype=np.int64)
    approx_assign = _require_rust_fn(assign_approx_date_offset_split, 'assign_approx_date_offset_split')
    out = approx_assign(
        timestamps_ns=ts.tolist(),
        ratio=ratio,
    )
    return _to_uint8_array(out)


def _apply_entity_include_filter(assignments: np.ndarray, train_table_path: str) -> np.ndarray:
    _require_rust_split_core()
    schema_names = _parquet_schema_names(train_table_path)
    if _ENTITY_INCLUDE_BIT_COL not in schema_names:
        return assignments
    include_df = _read_parquet_columns(train_table_path, [_ENTITY_INCLUDE_BIT_COL])
    include_mask = include_df[_ENTITY_INCLUDE_BIT_COL].astype(bool).to_numpy()
    apply_mask = _require_rust_fn(apply_entity_include_mask, 'apply_entity_include_mask')
    out = apply_mask(
        assignments=np.asarray(assignments, dtype=np.uint8).tolist(),
        include_mask=np.asarray(include_mask, dtype=bool).tolist(),
    )
    return _to_uint8_array(out)


def compute_dense_split_assignments(
    train_table_path: str,
    split_config: 'SplitSpec',
    split_family_hint: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute dense row-stage split assignment vector aligned with TT arange order.

    Args:
        train_table_path: Path to the materialized train table parquet.
        split_config: A SplitSpec instance describing the split configuration.
            Must have ``to_assignment_config()`` method and ``kind`` attribute.
        split_family_hint: Optional override for the split family. Must match
            split_config.kind if both are set.

    Returns assignments uint8 vector and metadata dict.
    """
    if isinstance(split_config, dict):
        raise DiskgraphDataError(
            'compute_dense_split_assignments requires split_config as SplitSpec, '
            'got dict. Construct a SplitSpec first.'
        )
    if not (hasattr(split_config, 'to_assignment_config') and callable(split_config.to_assignment_config)
            and hasattr(split_config, 'kind')):
        raise DiskgraphDataError(
            f'compute_dense_split_assignments requires split_config as SplitSpec '
            f'(must have to_assignment_config() and kind), got {type(split_config).__name__}'
        )
    if split_family_hint and split_config.kind and split_family_hint != split_config.kind:
        raise DiskgraphSystemError(
            'diskgraph split contract mismatch: '
            f'split_family_hint={split_family_hint} conflicts with split_config.kind={split_config.kind}'
        )
    split_cfg = split_config.to_assignment_config()
    family = split_family_hint or split_config.kind
    if not family:
        raise DiskgraphConfigError('missing split family for diskgraph dense split assignment')

    if family == 'RandomSplit':
        assignments = _assign_for_random_split(split_cfg, train_table_path)
    elif family == 'FixedSplit':
        assignments = _assign_for_fixed_split(split_cfg, train_table_path)
    elif family == 'TemporalSplit':
        assignments = _assign_for_temporal_split(split_cfg, train_table_path)
    elif family == 'DateOffsetSplit':
        assignments = _assign_for_date_offset_split(split_cfg, train_table_path)
    elif family == 'ApproxDateOffsetSplit':
        assignments = _assign_for_approx_date_offset_split(split_cfg, train_table_path)
    elif family == 'TimeRangeSplit':
        assignments = _assign_for_time_range_split(split_cfg, train_table_path)
    elif family == 'BacktestingSplit':
        assignments = _assign_for_backtesting_split(split_cfg, train_table_path)
    elif family == 'ColumnSplit':
        assignments = _assign_for_column_split(split_cfg, train_table_path)
    elif family == 'ManualSplit':
        assignments = _assign_for_manual_split(split_cfg, train_table_path)
    elif family == 'CustomSplit':
        assignments = _assign_for_manual_split(split_cfg, train_table_path)
    elif family == 'CustomLinkSplit':
        raise DiskgraphConfigError(
            'CustomLinkSplit cannot be rehydrated from parquet alone; '
            're-run tt-gen to regenerate the split artifact'
        )
    elif family == 'RandomLinkSplit':
        assignments = _assign_for_random_split(
            split_cfg,
            train_table_path,
            family='RandomLinkSplit',
        )
    else:
        raise DiskgraphConfigError(f'unsupported split family for diskgraph dense split assignment: {family}')

    assignments = _apply_entity_include_filter(assignments.astype(np.uint8), train_table_path)
    return assignments, {
        'family': family,
        'authoritative': True,
        'reason': None,
    }
