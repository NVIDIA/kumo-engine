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

import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

_NATIVE_IMPORT_ERROR: Optional[Exception] = None
try:
    from . import _c as _native_c
except Exception as exc:  # pragma: no cover - exercised via capability checks
    _native_c = None
    _NATIVE_IMPORT_ERROR = exc

_execute_tt_plan_materialize_with_split = (
    getattr(_native_c, 'execute_tt_plan_materialize_with_split', None)
    if _native_c is not None else None
)
_persist_tt_artifact_bundle = (
    getattr(_native_c, 'persist_tt_artifact_bundle', None)
    if _native_c is not None else None
)
_build_tt_stats_artifact_native = (
    getattr(_native_c, 'build_tt_stats_artifact', None)
    if _native_c is not None else None
)
_build_split_stats_artifact_native = (
    getattr(_native_c, 'build_split_stats_artifact', None)
    if _native_c is not None else None
)
_build_train_table_edge_bundle_artifact_native = (
    getattr(_native_c, 'build_train_table_edge_bundle_artifact', None)
    if _native_c is not None else None
)
_build_train_table_col_stats_artifact_native = (
    getattr(_native_c, 'build_train_table_col_stats_artifact', None)
    if _native_c is not None else None
)
_export_tt_artifacts_native = (
    getattr(_native_c, 'export_tt_artifacts', None)
    if _native_c is not None else None
)
_export_config_cls = (
    getattr(_native_c, 'ExportConfig', None)
    if _native_c is not None else None
)
_fkey_config_cls = (
    getattr(_native_c, 'FkeyConfig', None)
    if _native_c is not None else None
)
from .split_assignments import compute_dense_split_assignments
from .split_contract import SplitSpec


@dataclass(frozen=True)
class DownsamplingConfig:
    enabled: bool
    max_rows: int
    preserve_pre_downsample_tt: bool
    is_link_prediction: bool
    list_distinct_target_is_array: bool
    uses_arl_labels: bool


@dataclass(frozen=True)
class SplitStatsConfig:
    entity_col: str
    target_col: str
    time_col: str
    task_type: Optional[str]
    list_distinct_target_is_array: bool


def _ensure_native_strict_artifact_capabilities() -> None:
    required = {
        'execute_tt_plan_materialize_with_split': _execute_tt_plan_materialize_with_split,
        'build_tt_stats_artifact': _build_tt_stats_artifact_native,
        'build_split_stats_artifact': _build_split_stats_artifact_native,
        'persist_tt_artifact_bundle': _persist_tt_artifact_bundle,
    }
    missing = [name for name, fn in required.items() if not callable(fn)]
    if not missing:
        return

    details = ''
    if _NATIVE_IMPORT_ERROR is not None:
        details = f' native_import_error={_NATIVE_IMPORT_ERROR!r}'
    raise DiskgraphSystemError(
        'diskgraph strict artifact path missing native helper(s): '
        + ', '.join(missing)
        + '. Install a compatible diskgraph runtime wheel.'
        + details
    )


def _ensure_native_existing_table_rehydration_capabilities() -> None:
    required = {
        'build_tt_stats_artifact': _build_tt_stats_artifact_native,
        'build_split_stats_artifact': _build_split_stats_artifact_native,
        'build_train_table_col_stats_artifact': _build_train_table_col_stats_artifact_native,
        'persist_tt_artifact_bundle': _persist_tt_artifact_bundle,
    }
    missing = [name for name, fn in required.items() if not callable(fn)]
    if not missing:
        return

    details = ''
    if _NATIVE_IMPORT_ERROR is not None:
        details = f' native_import_error={_NATIVE_IMPORT_ERROR!r}'
    raise DiskgraphSystemError(
        'diskgraph existing-table artifact rehydration missing native helper(s): '
        + ', '.join(missing)
        + '. Install a compatible diskgraph runtime wheel.'
        + details
    )


def _normalize_downsampling_config(raw: Optional[Any]) -> DownsamplingConfig:
    cfg = raw if isinstance(raw, dict) else {}
    enabled = bool(cfg.get('enabled', False))
    max_rows = int(cfg.get('max_rows', 0) or 0)
    if 'seed' in cfg:
        raise DiskgraphConfigError('diskgraph downsampling config no longer accepts seed')
    if 'fail_on_unsupported' in cfg:
        raise DiskgraphConfigError('diskgraph downsampling config no longer accepts fail_on_unsupported')
    preserve_pre = bool(cfg.get('preserve_pre_downsample_tt', True))
    is_link_prediction = bool(cfg.get('is_link_prediction', False))
    list_distinct_target_is_array = bool(cfg.get('list_distinct_target_is_array', False))
    uses_arl_labels = bool(cfg.get('uses_arl_labels', False))

    if max_rows <= 0:
        if enabled:
            raise DiskgraphConfigError(
                'diskgraph downsampling config requires positive max_rows')
        max_rows = 2_147_483_647

    return DownsamplingConfig(
        enabled=enabled,
        max_rows=max_rows,
        preserve_pre_downsample_tt=preserve_pre,
        is_link_prediction=is_link_prediction,
        list_distinct_target_is_array=list_distinct_target_is_array,
        uses_arl_labels=uses_arl_labels,
    )


def _normalize_split_stats_config(
    raw: Optional[Any],
    downsampling_cfg: DownsamplingConfig,
) -> SplitStatsConfig:
    cfg = raw if isinstance(raw, dict) else {}
    entity_col = str(cfg.get('entity_col', 'ENTITY'))
    target_col = str(cfg.get('target_col', 'TARGET'))
    time_col = str(cfg.get('time_col', 'TIMESTAMP'))
    task_type_raw = cfg.get('task_type')
    task_type = str(task_type_raw).lower() if isinstance(task_type_raw, str) else None
    list_distinct_target_is_array = bool(
        cfg.get('list_distinct_target_is_array', downsampling_cfg.list_distinct_target_is_array)
    )
    return SplitStatsConfig(
        entity_col=entity_col,
        target_col=target_col,
        time_col=time_col,
        task_type=task_type,
        list_distinct_target_is_array=list_distinct_target_is_array,
    )


def _build_split_assignment_artifact(assignments: np.ndarray) -> bytes:
    arr = np.asarray(assignments, dtype=np.uint8)
    row_count = int(arr.shape[0])
    magic = b'DGSPLT\x00\x00'
    header = struct.pack('<IQ', 1, row_count)
    return magic + header + arr.tobytes(order='C')


def _extract_payload_json_as_sorted_dict_json(
    built: Any,
    *,
    label: str,
) -> str:
    payload_json = built.get('payload_json') if isinstance(built, dict) else None
    if not isinstance(payload_json, str) or not payload_json:
        raise DiskgraphSystemError(
            f'diskgraph native {label} builder returned invalid payload_json')
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise DiskgraphSystemError(
            f'diskgraph native {label} builder returned non-dict payload')
    return json.dumps(payload, sort_keys=True)


def _prepare_existing_table_reader_path(output_path: str) -> tuple[str, Optional[str]]:
    """Return a parquet *file* path usable by native artifact readers.

    For directory-style parquet datasets, this builds a temporary merged parquet
    file for read-only artifact computation, leaving the source dataset intact.
    Returns (reader_path, temp_dir_to_cleanup).
    """
    if os.path.isfile(output_path):
        return output_path, None

    dataset = ds.dataset(output_path, format='parquet')
    tmp_dir = tempfile.mkdtemp(prefix='dg_existing_tt_')
    tmp_path = os.path.join(tmp_dir, 'train_table.parquet')
    writer = None
    try:
        for batch in dataset.to_batches():
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, batch.schema)
            writer.write_batch(batch)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, dataset.schema)
        writer.close()
        writer = None
        return tmp_path, tmp_dir
    except Exception:
        if writer is not None:
            writer.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _build_train_table_edges_artifact(
    *,
    output_path: str,
    index_path: str,
    cfg: dict[str, Any],
) -> Optional[str]:
    if not bool(cfg.get('enabled', False)):
        return None

    if not callable(_export_tt_artifacts_native) or _export_config_cls is None or _fkey_config_cls is None:
        raise DiskgraphSystemError(
            'diskgraph strict artifact path requires native export_tt_artifacts '
            'with ExportConfig/FkeyConfig when train_table_edges_config.enabled=true'
        )
    if not callable(_build_train_table_edge_bundle_artifact_native):
        raise DiskgraphSystemError(
            'diskgraph strict artifact path requires native helper '
            'build_train_table_edge_bundle_artifact when train_table_edges_config.enabled=true'
        )

    edge_output_dir = str(
        cfg.get('output_dir')
        or os.path.join(output_path + '.artifacts', 'train_table_edges')
    )
    train_table_name = str(cfg.get('train_table_name', 'custom_train_table'))
    row_idx_col = str(cfg.get('row_idx_col', 'arange'))
    fkeys_raw = cfg.get('fkeys')
    if not isinstance(fkeys_raw, list) or not fkeys_raw:
        raise DiskgraphConfigError('train_table_edges_config.fkeys must be a non-empty list')

    fkeys = []
    for idx, item in enumerate(fkeys_raw):
        if not isinstance(item, dict):
            raise DiskgraphConfigError(f'train_table_edges_config.fkeys[{idx}] must be dict')
        col = item.get('column_name')
        tgt = item.get('target_table')
        if not isinstance(col, str) or not col:
            raise DiskgraphConfigError(f'train_table_edges_config.fkeys[{idx}].column_name is required')
        if not isinstance(tgt, str) or not tgt:
            raise DiskgraphConfigError(f'train_table_edges_config.fkeys[{idx}].target_table is required')
        fkeys.append(_fkey_config_cls(column_name=col, target_table=tgt))

    os.makedirs(edge_output_dir, exist_ok=True)
    export_cfg = _export_config_cls(
        train_table_path=output_path,
        index_path=index_path,
        output_dir=edge_output_dir,
        fkeys=fkeys,
        row_idx_col=row_idx_col,
        train_table_name=train_table_name,
    )
    _export_tt_artifacts_native(export_cfg)

    built_edges = _build_train_table_edge_bundle_artifact_native(
        output_dir=edge_output_dir,
        train_table_name=train_table_name,
        mask_present=bool(cfg.get('mask_present', True)),
        mask_absent_reason=cfg.get('mask_absent_reason'),
        edge_attrs_path=cfg.get('edge_attrs_path'),
    )
    return _extract_payload_json_as_sorted_dict_json(
        built_edges,
        label='train-table edges',
    )


def build_train_table_artifacts_for_existing_table(
    *,
    output_path: str,
    split_config: SplitSpec,
    split_family_hint: Optional[str] = None,
    split_stats_config: Optional[dict[str, Any]] = None,
    train_table_col_stats_config: Optional[dict[str, Any]] = None,
    train_table_edges_config: Optional[dict[str, Any]] = None,
    index_path: Optional[str] = None,
    producer_summary_json: Optional[str] = None,
) -> dict[str, Any]:
    if not isinstance(split_config, SplitSpec):
        raise DiskgraphDataError(
            'diskgraph existing-table artifact rehydration requires split_config as SplitSpec '
            f'(output_path={output_path})')
    if split_family_hint and split_config.kind and split_family_hint != split_config.kind:
        raise DiskgraphConfigError(
            'diskgraph split contract mismatch: '
            f'split_family_hint={split_family_hint} conflicts with split_config.kind={split_config.kind}'
        )
    resolved_family = split_config.kind or split_family_hint
    if not resolved_family:
        raise DiskgraphConfigError(
            'diskgraph existing-table artifact rehydration requires split family in '
            f'split_config.kind or split_family_hint (output_path={output_path})')
    if resolved_family == 'RandomLinkSplit':
        raise DiskgraphConfigError(
            'diskgraph existing-table artifact rehydration does not support RandomLinkSplit yet '
            f'(output_path={output_path})')

    _ensure_native_existing_table_rehydration_capabilities()

    if producer_summary_json is not None and not isinstance(producer_summary_json, str):
        raise DiskgraphConfigError(
            'diskgraph existing-table artifact rehydration requires producer_summary_json as string when provided')

    if train_table_col_stats_config is not None and not isinstance(train_table_col_stats_config, dict):
        raise DiskgraphConfigError(
            'diskgraph existing-table artifact rehydration requires train_table_col_stats_config as dict')

    split_stats_cfg = _normalize_split_stats_config(
        split_stats_config,
        _normalize_downsampling_config(None),
    )

    reader_path, cleanup_dir = _prepare_existing_table_reader_path(output_path)
    try:
        assignments, split_meta = compute_dense_split_assignments(
            reader_path,
            split_config,
            split_family_hint=resolved_family,
        )
        row_count = int(assignments.shape[0])
        table_rows = int(pq.read_metadata(reader_path).num_rows)
        if row_count != table_rows:
            raise DiskgraphDataError(
                'diskgraph existing-table artifact rehydration row count mismatch '
                f'(output_path={output_path}, reader_path={reader_path}, split_family={resolved_family}, '
                f'table_rows={table_rows}, assignment_rows={row_count})'
            )

        split_bytes = _build_split_assignment_artifact(assignments)

        built = _build_tt_stats_artifact_native(
            row_count=row_count,
            summary_json=producer_summary_json,
            plan_json=None,
        )
        stats_json = _extract_payload_json_as_sorted_dict_json(
            built,
            label='tt stats',
        )

        split_stats_built = _build_split_stats_artifact_native(
            output_path=reader_path,
            split_assignments=assignments.tobytes(order='C'),
            split_stats_config_json=json.dumps({
                'entity_col': split_stats_cfg.entity_col,
                'target_col': split_stats_cfg.target_col,
                'time_col': split_stats_cfg.time_col,
                'task_type': split_stats_cfg.task_type,
                'list_distinct_target_is_array': split_stats_cfg.list_distinct_target_is_array,
            }, sort_keys=True),
        )
        split_stats_json = _extract_payload_json_as_sorted_dict_json(
            split_stats_built,
            label='split stats',
        )

        train_col_stats_built = _build_train_table_col_stats_artifact_native(
            output_path=reader_path,
            split_assignments=assignments.tobytes(order='C'),
            train_table_col_stats_config_json=(
                json.dumps(train_table_col_stats_config, sort_keys=True)
                if train_table_col_stats_config is not None else None
            ),
        )
        train_col_stats_json = _extract_payload_json_as_sorted_dict_json(
            train_col_stats_built,
            label='train table col stats',
        )

        edges_cfg = train_table_edges_config if isinstance(train_table_edges_config, dict) else {}
        if bool(edges_cfg.get('enabled', False)) and (not index_path or not isinstance(index_path, str)):
            raise DiskgraphConfigError(
                'diskgraph existing-table artifact rehydration requires index_path when '
                f'train_table_edges_config.enabled=true (output_path={output_path})')

        train_table_edges_json = None
        if bool(edges_cfg.get('enabled', False)):
            edge_cfg = dict(edges_cfg)
            edge_cfg.setdefault(
                'output_dir',
                os.path.join(output_path + '.artifacts', 'train_table_edges'),
            )
            train_table_edges_json = _build_train_table_edges_artifact(
                output_path=reader_path,
                index_path=str(index_path),
                cfg=edge_cfg,
            )

        artifacts_dir = output_path + '.artifacts'
        manifest_path = output_path + '.manifest.json'
        persisted = _persist_tt_artifact_bundle(
            output_path=output_path,
            row_count=row_count,
            stats_json=stats_json,
            split_stats_json=split_stats_json,
            split_bytes=split_bytes,
            split_family=resolved_family,
            split_authoritative=bool(split_meta.get('authoritative', False)),
            split_reason=split_meta.get('reason'),
            downsampling_json=None,
            producer_summary_json=producer_summary_json,
            train_table_col_stats_json=train_col_stats_json,
            train_table_edges_json=train_table_edges_json,
        )
        if isinstance(persisted, dict):
            manifest_path = str(persisted.get('manifest_path') or manifest_path)
            artifacts_dir = str(persisted.get('artifacts_dir') or artifacts_dir)

        split_counts = {
            int(stage): int(count)
            for stage, count in zip(*np.unique(assignments, return_counts=True))
        }
        return {
            'output_path': output_path,
            'manifest_path': manifest_path,
            'artifacts_dir': artifacts_dir,
            'row_count': row_count,
            'split_counts': split_counts,
            'split_meta': split_meta,
        }
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def execute_tt_plan_with_artifacts(
    *,
    plan_json: str,
    index_path: str,
    output_path: str,
    split_config: SplitSpec,
    split_family_hint: Optional[str] = None,
    downsampling: Optional[Any] = None,
    split_stats_config: Optional[Any] = None,
    train_table_edges_config: Optional[Any] = None,
) -> dict[str, Any]:
    """Execute TT plan and materialize producer artifacts.

    Writes:
    - output parquet (with appended contiguous arange)
    - split assignment artifact (binary row-stage mapping)
    - stats artifact
    - manifest with artifact references + checksums
    """
    if not isinstance(split_config, SplitSpec):
        raise DiskgraphDataError('diskgraph strict artifact path requires split_config as SplitSpec')
    if split_family_hint and split_config.kind and split_family_hint != split_config.kind:
        raise DiskgraphConfigError(
            'diskgraph split contract mismatch: '
            f'split_family_hint={split_family_hint} conflicts with split_config.kind={split_config.kind}'
        )
    resolved_family = split_config.kind or split_family_hint
    if not resolved_family:
        raise DiskgraphConfigError('diskgraph strict artifact path requires split family in split_config.kind or split_family_hint')

    _ensure_native_strict_artifact_capabilities()

    downsampling_cfg = _normalize_downsampling_config(downsampling)
    split_stats_cfg = _normalize_split_stats_config(
        split_stats_config,
        downsampling_cfg,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    split_cfg_dict = split_config.to_assignment_config()

    row_count = 0
    native_summary: Optional[dict[str, Any]] = None
    native_split_bytes: Optional[bytes] = None
    native_split_meta: Optional[dict[str, Any]] = None
    native_downsampling_handled = False
    native_downsampling_applied = False
    native_pre_downsample_row_count: Optional[int] = None

    materialize_with_split_fn = _execute_tt_plan_materialize_with_split

    split_cfg_json = json.dumps(split_cfg_dict)

    downsampling_cfg_json = json.dumps({
        'enabled': bool(downsampling_cfg.enabled),
        'max_rows': int(downsampling_cfg.max_rows),
        'preserve_pre_downsample_tt': bool(downsampling_cfg.preserve_pre_downsample_tt),
        'is_link_prediction': bool(downsampling_cfg.is_link_prediction),
        'list_distinct_target_is_array': bool(downsampling_cfg.list_distinct_target_is_array),
        'uses_arl_labels': bool(downsampling_cfg.uses_arl_labels),
    })

    mat = materialize_with_split_fn(
        plan_json=plan_json,
        index_path=index_path,
        output_path=output_path,
        split_config_json=split_cfg_json,
        split_family_hint=resolved_family,
        downsampling_config_json=downsampling_cfg_json,
    )
    if not isinstance(mat, dict):
        raise DiskgraphSystemError('diskgraph native materialize_with_split returned invalid payload type')
    row_count = int(mat.get('rows_written', 0) or 0)
    summary_json = mat.get('summary_json')
    if isinstance(summary_json, str) and summary_json:
        parsed = json.loads(summary_json)
        if isinstance(parsed, dict):
            native_summary = parsed
    split_blob = mat.get('split_assignments')
    if isinstance(split_blob, (bytes, bytearray)):
        native_split_bytes = bytes(split_blob)
    native_split_meta = {
        'family': mat.get('split_family') or resolved_family,
        'authoritative': bool(mat.get('split_authoritative', False)),
        'reason': mat.get('split_reason'),
    }
    native_downsampling_handled = bool(mat.get('downsampling_handled', False))
    native_downsampling_applied = bool(mat.get('downsampling_applied', False))
    pre_raw = mat.get('pre_downsample_row_count')
    if isinstance(pre_raw, int):
        native_pre_downsample_row_count = int(pre_raw)

    if row_count == 0:
        raise DiskgraphDataError('diskgraph execute_tt_plan returned zero rows')

    if downsampling_cfg.enabled and not native_downsampling_handled:
        raise DiskgraphSystemError(
            'diskgraph native materialize_with_split did not handle downsampling metadata'
        )

    pre_row_count = int(native_pre_downsample_row_count if native_pre_downsample_row_count is not None else row_count)
    downsample_applied = bool(native_downsampling_applied)
    pre_downsample_path = (
        output_path + '.pre_downsample.parquet'
        if (downsample_applied and downsampling_cfg.preserve_pre_downsample_tt)
        else None
    )
    downsample_info = {
        'path': 'native_materialize',
        'summary_source': 'native',
        'fallback_python_downsampling': False,
    }
    if downsampling_cfg.enabled:
        stats_obj = (native_summary or {}).get('stats') if isinstance(native_summary, dict) else None
        if not isinstance(stats_obj, dict):
            raise DiskgraphSystemError(
                'diskgraph native downsampling reported handled=true but summary.stats is missing'
            )
        native_rows = stats_obj.get('num_total_rows')
        if isinstance(native_rows, (int, float)) and int(native_rows) != int(row_count):
            raise DiskgraphDataError(
                'diskgraph native downsampling summary row-count mismatch '
                f'(summary_rows={int(native_rows)}, rows_written={int(row_count)})'
            )

    split_meta: dict[str, Any]
    split_bytes: bytes
    assignments: np.ndarray

    if native_split_bytes is None or native_split_meta is None:
        raise DiskgraphDataError('diskgraph native split assignment payload is missing')
    if not native_split_meta.get('authoritative', False):
        raise DiskgraphDataError(
            'diskgraph split assignment is non-authoritative '
            f"(family={native_split_meta.get('family')}, reason={native_split_meta.get('reason')})"
        )

    if len(native_split_bytes) != row_count:
        raise DiskgraphDataError(
            'diskgraph native split assignment row count mismatch '
            f'(rows={row_count}, assignment_rows={len(native_split_bytes)})'
        )
    assignments = np.frombuffer(native_split_bytes, dtype=np.uint8).copy()
    split_meta = native_split_meta
    split_bytes = _build_split_assignment_artifact(assignments)

    built = _build_tt_stats_artifact_native(
        row_count=int(row_count),
        summary_json=(json.dumps(native_summary, sort_keys=True) if native_summary is not None else None),
        plan_json=plan_json,
    )
    stats_json = _extract_payload_json_as_sorted_dict_json(
        built,
        label='tt stats',
    )

    split_stats_built = _build_split_stats_artifact_native(
        output_path=output_path,
        split_assignments=assignments.tobytes(order='C'),
        split_stats_config_json=json.dumps({
            'entity_col': split_stats_cfg.entity_col,
            'target_col': split_stats_cfg.target_col,
            'time_col': split_stats_cfg.time_col,
            'task_type': split_stats_cfg.task_type,
            'list_distinct_target_is_array': split_stats_cfg.list_distinct_target_is_array,
        }, sort_keys=True),
    )
    split_stats_json = _extract_payload_json_as_sorted_dict_json(
        split_stats_built,
        label='split stats',
    )

    train_col_stats_payload_json = mat.get('train_table_col_stats_json')
    if not isinstance(train_col_stats_payload_json, str) or not train_col_stats_payload_json:
        raise DiskgraphSystemError(
            'diskgraph native materialize_with_split did not provide train_table_col_stats_json')
    train_col_stats_payload = json.loads(train_col_stats_payload_json)
    if not isinstance(train_col_stats_payload, dict):
        raise DiskgraphSystemError(
            'diskgraph native train_table_col_stats_json payload must be dict')
    train_col_stats_json = json.dumps(train_col_stats_payload, sort_keys=True)

    downsampling_payload = {
        'config': {
            'enabled': downsampling_cfg.enabled,
            'max_rows': downsampling_cfg.max_rows,
            'preserve_pre_downsample_tt': downsampling_cfg.preserve_pre_downsample_tt,
            'is_link_prediction': downsampling_cfg.is_link_prediction,
            'list_distinct_target_is_array': downsampling_cfg.list_distinct_target_is_array,
            'uses_arl_labels': downsampling_cfg.uses_arl_labels,
        },
        'applied': downsample_applied,
        'pre_downsample_row_count': pre_row_count,
        'post_downsample_row_count': row_count,
        'pre_downsample_train_table_path': pre_downsample_path,
        'details': downsample_info,
    }

    train_table_edges_json = _build_train_table_edges_artifact(
        output_path=output_path,
        index_path=index_path,
        cfg=(train_table_edges_config if isinstance(train_table_edges_config, dict) else {}),
    )

    artifacts_dir = output_path + '.artifacts'
    manifest_path = output_path + '.manifest.json'
    persisted = _persist_tt_artifact_bundle(
        output_path=output_path,
        row_count=int(row_count),
        stats_json=stats_json,
        split_stats_json=split_stats_json,
        split_bytes=split_bytes,
        split_family=split_meta.get('family'),
        split_authoritative=bool(split_meta.get('authoritative', False)),
        split_reason=split_meta.get('reason'),
        downsampling_json=json.dumps(downsampling_payload, sort_keys=True),
        producer_summary_json=(json.dumps(native_summary, sort_keys=True) if native_summary is not None else None),
        train_table_col_stats_json=train_col_stats_json,
        train_table_edges_json=train_table_edges_json,
    )
    if isinstance(persisted, dict):
        manifest_path = str(persisted.get('manifest_path') or manifest_path)
        artifacts_dir = str(persisted.get('artifacts_dir') or artifacts_dir)

    split_counts = {
        int(stage): int(count)
        for stage, count in zip(*np.unique(assignments, return_counts=True))
    }
    return {
        'output_path': output_path,
        'manifest_path': manifest_path,
        'row_count': row_count,
        'split_counts': split_counts,
        'split_meta': split_meta,
        'downsampling': {
            'applied': downsample_applied,
            'pre_downsample_row_count': pre_row_count,
            'post_downsample_row_count': row_count,
            'pre_downsample_train_table_path': pre_downsample_path,
            'details': downsample_info,
        },
    }
