from __future__ import annotations

import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._c import execute_tt_plan
try:
    from ._c import execute_tt_plan_materialize as _execute_tt_plan_materialize
except Exception:  # pragma: no cover - stale wheel fallback
    _execute_tt_plan_materialize = None
try:
    from ._c import execute_tt_plan_materialize_with_split as _execute_tt_plan_materialize_with_split
except Exception:  # pragma: no cover - stale wheel fallback
    _execute_tt_plan_materialize_with_split = None
from .split_assignments import compute_dense_split_assignments
from .split_contract import normalize_split_spec


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _to_epoch_seconds(v: Any) -> int:
    iv = int(v)
    av = abs(iv)
    # magnitude-based unit inference: s/ms/us/ns -> seconds
    if av >= 10**17:
        return iv // 1_000_000_000
    if av >= 10**14:
        return iv // 1_000_000
    if av >= 10**11:
        return iv // 1_000
    return iv


def _largest_remainder_counts(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        return []
    if total <= 0:
        return [0] * buckets
    ideal = [float(total) / float(buckets)] * buckets
    sizes = [int(x) for x in ideal]
    remaining = total - sum(sizes)
    order = list(range(buckets))
    order.sort(key=lambda i: (-(ideal[i] - sizes[i]), i))
    k = 0
    while remaining > 0 and order:
        sizes[order[k]] += 1
        remaining -= 1
        k += 1
        if k == len(order):
            k = 0
    return sizes


def _build_stats_artifact(
    native_summary: Optional[dict[str, Any]],
    row_count: int,
    plan_json: Optional[str] = None,
) -> dict[str, Any]:
    stats = (native_summary or {}).get('stats') if isinstance(native_summary, dict) else None
    if not isinstance(stats, dict):
        stats = {}

    num_total_rows = int(stats.get('num_total_rows', row_count))
    num_train_rows = int(stats.get('num_train_rows', row_count))
    num_entities = int(stats.get('num_entities', 0))
    num_targets = int(stats.get('num_targets', 0))
    num_links = stats.get('num_links')
    if num_links is not None:
        num_links = int(num_links)

    exact_stats = stats.get('exact_stats') if isinstance(stats.get('exact_stats'), dict) else None
    split_payload = stats.get('split_payload') if isinstance(stats.get('split_payload'), dict) else None

    # Temporal stats fallbacks required by strict native_full cache seeding.
    timestamp_unique = None
    timestamp_max = None
    if isinstance(exact_stats, dict):
        ts_stats = exact_stats.get('timestamp')
        if isinstance(ts_stats, dict):
            raw_unique = ts_stats.get('num_unique')
            if isinstance(raw_unique, int):
                timestamp_unique = raw_unique
            raw_max = ts_stats.get('max')
            if isinstance(raw_max, int):
                timestamp_max = raw_max

    temporal_timestamps: list[int] = []
    prediction_type_raw: Any = None
    if plan_json:
        try:
            plan_obj = json.loads(plan_json)
            if isinstance(plan_obj, dict):
                temporal_obj = plan_obj.get('temporal')
                if isinstance(temporal_obj, dict):
                    raw_ts = temporal_obj.get('timestamps')
                    if isinstance(raw_ts, list):
                        temporal_timestamps = [int(ts) for ts in raw_ts if isinstance(ts, (int, float))]
                prediction_type_raw = plan_obj.get('prediction_type')
        except Exception:
            # Non-fatal: fall back to exact-stats-only derivation.
            pass

    is_temporal = len(temporal_timestamps) > 0
    is_link_prediction = isinstance(prediction_type_raw, dict) and ('link' in prediction_type_raw)

    if is_temporal:
        num_timeframes = len(temporal_timestamps)
        num_timeframes_post_filter = len(temporal_timestamps)
    else:
        num_timeframes = int(timestamp_unique or 0)
        num_timeframes_post_filter = int(timestamp_unique or 0)

    avg_entities_per_timeframe = (
        int(round(num_train_rows / num_timeframes_post_filter))
        if num_timeframes_post_filter > 0 else 0
    )

    stats_per_timeframe: list[dict[str, Any]] = []
    if is_temporal:
        # Ensure at least two entries so legacy split-adjustment code that
        # indexes [0] and [1] remains safe.
        ts_values = temporal_timestamps if len(temporal_timestamps) >= 2 else (
            temporal_timestamps * 2 if temporal_timestamps else []
        )
        if not ts_values:
            fallback_ts = timestamp_max if timestamp_max is not None else 0
            ts_values = [fallback_ts, fallback_ts]

        per_rows = _largest_remainder_counts(num_train_rows, len(ts_values))
        for i, ts_raw in enumerate(ts_values):
            ts_sec = _to_epoch_seconds(ts_raw)
            dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()
            if is_link_prediction:
                stats_per_timeframe.append({
                    'min_date': dt,
                    'max_date': dt,
                    'anchor_date': dt,
                    'num_train_rows': int(per_rows[i]),
                    'num_entities': num_entities,
                    'num_targets': num_targets,
                    'num_links': num_links,
                    'index': i,
                })
            else:
                stats_per_timeframe.append({
                    'min_date': dt,
                    'max_date': dt,
                    'anchor_date': dt,
                    'num_train_rows': int(per_rows[i]),
                    'num_entities': num_entities,
                    'num_targets': num_targets,
                    'target_distribution': {},
                    'index': i,
                })

    payload: dict[str, Any] = {
        'version': 1,
        'kind': 'tt_stats',
        'num_total_rows': num_total_rows,
        'num_train_rows': num_train_rows,
        'num_entities': num_entities,
        'num_targets': num_targets,
        'num_links': num_links,
        'num_target_rows': num_train_rows,
        'num_timeframes': num_timeframes,
        'num_timeframes_post_filter': num_timeframes_post_filter,
        'avg_entities_per_timeframe': avg_entities_per_timeframe,
        # Keep temporal payload keys present for strict native_full. Detailed
        # structures are emitted as they become available from native stats.
        'node_degree_stats': None,
        'stats_per_timeframe': stats_per_timeframe,
        'target_distribution': {},
        'scoring_table_column_stats': None,
        # Optional DG-provided sections.
        'exact_stats': exact_stats,
        'split_payload': split_payload,
    }

    provided_fields = [
        'num_total_rows',
        'num_train_rows',
        'num_entities',
        'num_targets',
        'num_target_rows',
        'num_timeframes',
        'num_timeframes_post_filter',
        'avg_entities_per_timeframe',
        'node_degree_stats',
        'stats_per_timeframe',
        'target_distribution',
        'scoring_table_column_stats',
    ]
    if num_links is not None:
        provided_fields.append('num_links')
    if payload['exact_stats'] is not None:
        provided_fields.append('exact_stats')
    if payload['split_payload'] is not None:
        provided_fields.append('split_payload')

    payload['provided_fields'] = provided_fields
    return payload


def _build_split_assignment_artifact(assignments: np.ndarray) -> bytes:
    arr = np.asarray(assignments, dtype=np.uint8)
    row_count = int(arr.shape[0])
    magic = b'DGSPLT4\x00'
    header = struct.pack('<IQ', 1, row_count)
    return magic + header + arr.tobytes(order='C')



def _coerce_split_config_dict(split_config: Optional[Any]) -> Optional[dict[str, Any]]:
    if split_config is None:
        return None
    if isinstance(split_config, dict):
        return dict(split_config)
    if hasattr(split_config, 'config') and callable(split_config.config):
        payload = split_config.config()
        if isinstance(payload, dict):
            return payload
    if hasattr(split_config, 'model_dump') and callable(split_config.model_dump):
        payload = split_config.model_dump()
        if isinstance(payload, dict):
            return payload
    if hasattr(split_config, '__dict__'):
        return {k: v for k, v in split_config.__dict__.items() if not k.startswith('_')}
    return None


def execute_tt_plan_with_artifacts(
    *,
    plan_json: str,
    index_path: str,
    output_path: str,
    split_config: Optional[Any],
    split_family_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Execute TT plan and materialize producer artifacts.

    Writes:
    - output parquet (with appended contiguous arange)
    - split assignment artifact (binary row-stage mapping)
    - stats artifact
    - manifest with artifact references + checksums
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    split_spec = normalize_split_spec(
        split_config,
        split_family_hint=split_family_hint,
        require=True,
    )
    assert split_spec is not None
    split_cfg_dict = split_spec.to_assignment_config()

    row_count = 0
    native_summary: Optional[dict[str, Any]] = None
    native_split_bytes: Optional[bytes] = None
    native_split_meta: Optional[dict[str, Any]] = None

    materialize_with_split_fn = _execute_tt_plan_materialize_with_split
    materialize_fn = _execute_tt_plan_materialize

    if callable(materialize_with_split_fn):
        mat = materialize_with_split_fn(
            plan_json=plan_json,
            index_path=index_path,
            output_path=output_path,
            split_config_json=json.dumps(split_cfg_dict),
            split_family_hint=split_spec.kind,
        )
        if not isinstance(mat, dict):
            raise RuntimeError('diskgraph native materialize_with_split returned invalid payload type')
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
            'family': mat.get('split_family') or split_spec.kind,
            'authoritative': bool(mat.get('split_authoritative', False)),
            'reason': mat.get('split_reason'),
        }
    elif callable(materialize_fn):
        mat = materialize_fn(
            plan_json=plan_json,
            index_path=index_path,
            output_path=output_path,
        )
        if not isinstance(mat, dict):
            raise RuntimeError('diskgraph native materialize returned invalid payload type')
        row_count = int(mat.get('rows_written', 0) or 0)
        summary_json = mat.get('summary_json')
        if isinstance(summary_json, str) and summary_json:
            parsed = json.loads(summary_json)
            if isinstance(parsed, dict):
                native_summary = parsed
    else:
        # Backward-compat fallback for stale wheels.
        batch_iterator = execute_tt_plan(plan_json=plan_json, index_path=index_path)
        writer = None
        try:
            for raw_batch in batch_iterator:
                n = raw_batch.num_rows
                arange_array = pa.array(range(row_count, row_count + n), type=pa.int64())
                batch_with_idx = raw_batch.append_column('arange', arange_array)
                row_count += n

                if writer is None:
                    writer = pq.ParquetWriter(output_path, batch_with_idx.schema, compression='snappy')
                writer.write_batch(batch_with_idx)

            summary_fn = getattr(batch_iterator, 'execution_summary', None)
            if callable(summary_fn):
                raw_summary = summary_fn()
                if isinstance(raw_summary, dict):
                    native_summary = raw_summary
        finally:
            if writer is not None:
                writer.close()

    if row_count == 0:
        raise RuntimeError('diskgraph execute_tt_plan returned zero rows')

    artifacts_dir = output_path + '.artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)

    stats_payload = _build_stats_artifact(native_summary, row_count, plan_json)
    stats_bytes = json.dumps(stats_payload, sort_keys=True).encode('utf-8')
    stats_path = os.path.join(artifacts_dir, 'stats.tt.json')
    with open(stats_path, 'wb') as f:
        f.write(stats_bytes)

    split_meta: dict[str, Any]
    split_bytes: bytes
    assignments: np.ndarray

    if native_split_bytes is not None and native_split_meta is not None and native_split_meta.get('authoritative', False):
        if len(native_split_bytes) != row_count:
            raise RuntimeError(
                'diskgraph native split assignment row count mismatch '
                f'(rows={row_count}, assignment_rows={len(native_split_bytes)})'
            )
        assignments = np.frombuffer(native_split_bytes, dtype=np.uint8).copy()
        split_meta = native_split_meta
        split_bytes = _build_split_assignment_artifact(assignments)
    else:
        assignments, split_meta = compute_dense_split_assignments(
            train_table_path=output_path,
            split_config=split_spec,
            split_family_hint=split_spec.kind,
        )
        if len(assignments) != row_count:
            raise RuntimeError(
                'diskgraph dense split assignment row count mismatch '
                f'(rows={row_count}, assignment_rows={len(assignments)})'
            )
        split_bytes = _build_split_assignment_artifact(assignments)

    if not split_meta.get('authoritative', False):
        raise RuntimeError(
            'diskgraph split assignment is non-authoritative '
            f"(family={split_meta.get('family')}, reason={split_meta.get('reason')})"
        )
    split_path = os.path.join(artifacts_dir, 'split_assignments.row_stage.v1.bin')
    with open(split_path, 'wb') as f:
        f.write(split_bytes)

    manifest_path = output_path + '.manifest.json'
    manifest = {
        'version': 1,
        'row_count': row_count,
        'train_table': {'path': output_path, 'format': 'parquet_with_arange'},
        'split': {
            'path': split_path,
            'format': 'split_assignment_row_stage_bin',
            'authoritative': bool(split_meta.get('authoritative', False)),
            'family': split_meta.get('family'),
            'reason': split_meta.get('reason'),
        },
        'stats': {'path': stats_path, 'format': 'tt_stats_json'},
        'checksums': {
            'train_table_sha256': _sha256_file(output_path),
            'split_sha256': _sha256_bytes(split_bytes),
            'stats_sha256': _sha256_bytes(stats_bytes),
        },
        'producer_summary': native_summary,
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, sort_keys=True)

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
    }
