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
from typing import Any, Optional

import pyarrow.parquet as pq

_NATIVE_IMPORT_ERROR: Optional[Exception] = None
try:
    from . import _c as _native_c
except Exception as exc:  # pragma: no cover - exercised via capability checks
    _native_c = None
    _NATIVE_IMPORT_ERROR = exc

_export_tt_artifacts_native = (
    getattr(_native_c, 'export_tt_artifacts', None)
    if _native_c is not None else None
)
_build_prediction_table_edge_bundle_artifact_native = (
    getattr(_native_c, 'build_prediction_table_edge_bundle_artifact', None)
    if _native_c is not None else None
)
_build_prediction_table_col_stats_artifact_native = (
    getattr(_native_c, 'build_prediction_table_col_stats_artifact', None)
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
_persist_prediction_artifact_bundle_native = (
    getattr(_native_c, 'persist_prediction_artifact_bundle', None)
    if _native_c is not None else None
)


def _ensure_native_prediction_edge_capabilities() -> None:
    required = {
        'export_tt_artifacts': _export_tt_artifacts_native,
        'build_prediction_table_edge_bundle_artifact': _build_prediction_table_edge_bundle_artifact_native,
        'ExportConfig': _export_config_cls,
        'FkeyConfig': _fkey_config_cls,
    }
    missing = []
    for name, item in required.items():
        if name in ('export_tt_artifacts', 'build_prediction_table_edge_bundle_artifact'):
            if not callable(item):
                missing.append(name)
        elif item is None:
            missing.append(name)
    if not missing:
        return

    details = ''
    if _NATIVE_IMPORT_ERROR is not None:
        details = f' native_import_error={_NATIVE_IMPORT_ERROR!r}'
    raise DiskgraphSystemError(
        'diskgraph prediction artifact path missing native helper(s): '
        + ', '.join(missing)
        + '. Install a compatible diskgraph runtime wheel.'
        + details
    )


def _ensure_native_prediction_col_stats_capabilities() -> None:
    if callable(_build_prediction_table_col_stats_artifact_native):
        return

    details = ''
    if _NATIVE_IMPORT_ERROR is not None:
        details = f' native_import_error={_NATIVE_IMPORT_ERROR!r}'
    raise DiskgraphSystemError(
        'diskgraph prediction artifact path missing native helper(s): '
        'build_prediction_table_col_stats_artifact. '
        'Install a compatible diskgraph runtime wheel.'
        + details
    )


def _ensure_native_prediction_manifest_capabilities() -> None:
    if callable(_persist_prediction_artifact_bundle_native):
        return

    details = ''
    if _NATIVE_IMPORT_ERROR is not None:
        details = f' native_import_error={_NATIVE_IMPORT_ERROR!r}'
    raise DiskgraphSystemError(
        'diskgraph prediction artifact path missing native helper(s): '
        'persist_prediction_artifact_bundle. '
        'Install a compatible diskgraph runtime wheel.'
        + details
    )


def _resolve_prediction_row_count(prediction_table_path: str, cfg: dict[str, Any]) -> int:
    row_count = cfg.get('row_count')
    if isinstance(row_count, int) and row_count >= 0:
        return int(row_count)

    try:
        return int(pq.read_metadata(prediction_table_path).num_rows)
    except Exception as e:
        raise DiskgraphSystemError(
            'prediction artifact bundle requires row_count in cfg or readable parquet metadata; '
            f'failed to infer row_count from {prediction_table_path}: {e}'
        )


def _prediction_table_size_bytes(prediction_table_path: str) -> int:
    try:
        return int(os.stat(prediction_table_path).st_size)
    except Exception as e:
        raise DiskgraphSystemError(
            f'prediction_table_path missing or unreadable for size metadata: {e}'
        )


def validate_prediction_table_edges_artifact(payload: dict[str, Any]) -> None:
    if payload.get('kind') != 'tt_prediction_table_edges':
        raise DiskgraphDataError('prediction edge artifact kind must be tt_prediction_table_edges')
    if payload.get('format') != 'tt_pred_edge_bundle_v1':
        raise DiskgraphDataError('prediction edge artifact format must be tt_pred_edge_bundle_v1')

    reverse_edges = payload.get('reverse_edges')
    if not isinstance(reverse_edges, dict) or not reverse_edges:
        raise DiskgraphConfigError('prediction edge artifact requires non-empty reverse_edges')

    fkey_cols = payload.get('fkey_cols')
    if not isinstance(fkey_cols, dict):
        raise DiskgraphConfigError('prediction edge artifact requires fkey_cols object')
    if not isinstance(fkey_cols.get('path'), str) or not fkey_cols.get('path'):
        raise DiskgraphConfigError('prediction edge artifact fkey_cols.path is required')

    mask = payload.get('mask')
    if not isinstance(mask, dict):
        raise DiskgraphConfigError('prediction edge artifact requires mask object')
    if not isinstance(mask.get('present'), bool):
        raise DiskgraphConfigError('prediction edge artifact mask.present must be bool')
    if mask.get('present'):
        if not isinstance(mask.get('path'), str) or not mask.get('path'):
            raise DiskgraphConfigError('prediction edge artifact mask.path is required when mask.present=true')

    checksums = payload.get('checksums')
    if not isinstance(checksums, dict) or not checksums:
        raise DiskgraphConfigError('prediction edge artifact requires non-empty checksums')


def parse_prediction_table_edges_artifact(payload_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except Exception as e:
        raise DiskgraphDataError(f'invalid prediction edge artifact JSON: {e}')
    if not isinstance(payload, dict):
        raise DiskgraphDataError('prediction edge artifact payload must be a JSON object')
    validate_prediction_table_edges_artifact(payload)
    return payload


def validate_prediction_table_col_stats_artifact(payload: dict[str, Any]) -> None:
    if payload.get('kind') != 'tt_prediction_table_col_stats':
        raise DiskgraphDataError('prediction col-stats artifact kind must be tt_prediction_table_col_stats')
    if payload.get('stats_format') != 'kumo_col_stats_v1':
        raise DiskgraphDataError('prediction col-stats artifact stats_format must be kumo_col_stats_v1')

    stats = payload.get('stats')
    if not isinstance(stats, dict):
        raise DiskgraphConfigError('prediction col-stats artifact requires stats object')


def parse_prediction_table_col_stats_artifact(payload_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except Exception as e:
        raise DiskgraphDataError(f'invalid prediction col-stats artifact JSON: {e}')
    if not isinstance(payload, dict):
        raise DiskgraphDataError('prediction col-stats artifact payload must be a JSON object')
    validate_prediction_table_col_stats_artifact(payload)
    return payload


def build_prediction_table_col_stats_artifact(
    *,
    prediction_table_path: str,
    cfg: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    raw_cfg = cfg if isinstance(cfg, dict) else {}
    if not bool(raw_cfg.get('enabled', True)):
        return None

    _ensure_native_prediction_col_stats_capabilities()

    built = _build_prediction_table_col_stats_artifact_native(
        output_path=prediction_table_path,
        prediction_table_col_stats_config_json=json.dumps({
            'data_version': str(raw_cfg.get('data_version', 'v1')),
        }),
    )
    payload_json = built.get('payload_json') if isinstance(built, dict) else None
    if not isinstance(payload_json, str) or not payload_json:
        raise DiskgraphSystemError(
            'diskgraph native prediction-table col-stats builder returned invalid payload_json'
        )

    payload = parse_prediction_table_col_stats_artifact(payload_json)
    return json.dumps(payload, sort_keys=True)


def build_prediction_table_edges_artifact(
    *,
    prediction_table_path: str,
    index_path: str,
    cfg: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    raw_cfg = cfg if isinstance(cfg, dict) else {}
    if not bool(raw_cfg.get('enabled', False)):
        return None

    if str(raw_cfg.get('task_table_edge_kind', 'prediction')) != 'prediction':
        raise DiskgraphConfigError('prediction_table_edges_config.task_table_edge_kind must be "prediction"')

    _ensure_native_prediction_edge_capabilities()

    edge_output_dir = str(
        raw_cfg.get('output_dir')
        or os.path.join(prediction_table_path + '.artifacts', 'prediction_table_edges')
    )
    prediction_table_name = str(raw_cfg.get('prediction_table_name', 'custom_prediction_table'))
    row_idx_col = str(raw_cfg.get('row_idx_col', 'arange'))

    fkeys_raw = raw_cfg.get('fkeys')
    if not isinstance(fkeys_raw, list) or not fkeys_raw:
        raise DiskgraphConfigError('prediction_table_edges_config.fkeys must be a non-empty list')

    fkeys = []
    for idx, item in enumerate(fkeys_raw):
        if not isinstance(item, dict):
            raise DiskgraphConfigError(f'prediction_table_edges_config.fkeys[{idx}] must be dict')
        col = item.get('column_name')
        tgt = item.get('target_table')
        if not isinstance(col, str) or not col:
            raise DiskgraphConfigError(
                f'prediction_table_edges_config.fkeys[{idx}].column_name is required'
            )
        if not isinstance(tgt, str) or not tgt:
            raise DiskgraphConfigError(
                f'prediction_table_edges_config.fkeys[{idx}].target_table is required'
            )
        fkeys.append(_fkey_config_cls(column_name=col, target_table=tgt))

    os.makedirs(edge_output_dir, exist_ok=True)
    # Reuse existing native ExportConfig fields (train_table_* naming) for
    # prediction tables to avoid changing the shared Rust export interface.
    export_cfg = _export_config_cls(
        train_table_path=prediction_table_path,
        index_path=index_path,
        output_dir=edge_output_dir,
        fkeys=fkeys,
        row_idx_col=row_idx_col,
        train_table_name=prediction_table_name,
    )
    _export_tt_artifacts_native(export_cfg)

    built_edges = _build_prediction_table_edge_bundle_artifact_native(
        output_dir=edge_output_dir,
        prediction_table_name=prediction_table_name,
        mask_present=bool(raw_cfg.get('mask_present', True)),
        mask_absent_reason=raw_cfg.get('mask_absent_reason'),
        edge_attrs_path=raw_cfg.get('edge_attrs_path'),
    )
    payload_json = built_edges.get('payload_json') if isinstance(built_edges, dict) else None
    if not isinstance(payload_json, str) or not payload_json:
        raise DiskgraphSystemError(
            'diskgraph native prediction-table edges builder returned invalid payload_json'
        )

    payload = parse_prediction_table_edges_artifact(payload_json)
    return json.dumps(payload, sort_keys=True)


def build_prediction_table_artifact_bundle(
    *,
    prediction_table_path: str,
    index_path: str,
    cfg: Optional[dict[str, Any]] = None,
    producer_summary: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    raw_cfg = cfg if isinstance(cfg, dict) else {}
    edge_cfg = raw_cfg.get('prediction_table_edges_config')
    if not isinstance(edge_cfg, dict):
        # Backward-compatible flat config: only project edge-specific keys.
        edge_cfg = {
            k: raw_cfg[k]
            for k in (
                'enabled',
                'task_table_edge_kind',
                'output_dir',
                'prediction_table_name',
                'row_idx_col',
                'fkeys',
                'mask_present',
                'mask_absent_reason',
                'edge_attrs_path',
            )
            if k in raw_cfg
        }
    col_cfg = raw_cfg.get('prediction_table_col_stats_config')
    if not isinstance(col_cfg, dict):
        col_cfg = {}

    edges_json = build_prediction_table_edges_artifact(
        prediction_table_path=prediction_table_path,
        index_path=index_path,
        cfg=edge_cfg,
    )
    if edges_json is None:
        return None

    col_stats_json = build_prediction_table_col_stats_artifact(
        prediction_table_path=prediction_table_path,
        cfg=col_cfg,
    )

    _ensure_native_prediction_manifest_capabilities()

    row_count = _resolve_prediction_row_count(prediction_table_path, raw_cfg)

    persisted = _persist_prediction_artifact_bundle_native(
        output_path=prediction_table_path,
        row_count=row_count,
        prediction_table_edges_json=edges_json,
        prediction_table_col_stats_json=col_stats_json,
        producer_summary_json=(json.dumps(producer_summary) if isinstance(producer_summary, dict) else None),
    )
    if not isinstance(persisted, dict):
        raise DiskgraphSystemError('diskgraph native persist_prediction_artifact_bundle returned invalid payload')

    return {
        'manifest_path': persisted.get('manifest_path'),
        'artifacts_dir': persisted.get('artifacts_dir'),
        'prediction_table_edges_json': edges_json,
        'prediction_table_col_stats_json': col_stats_json,
        'prediction_table_size_bytes': _prediction_table_size_bytes(prediction_table_path),
    }
