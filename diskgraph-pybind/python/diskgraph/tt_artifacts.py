from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._c import execute_tt_plan
try:
    from ._c import execute_tt_plan_materialize as _execute_tt_plan_materialize
except Exception:  # pragma: no cover - stale wheel fallback
    _execute_tt_plan_materialize = None
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


def _build_stats_artifact(native_summary: Optional[dict[str, Any]], row_count: int) -> dict[str, Any]:
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
    return {
        'version': 1,
        'kind': 'tt_stats_v1',
        'num_total_rows': num_total_rows,
        'num_train_rows': num_train_rows,
        'num_entities': num_entities,
        'num_targets': num_targets,
        'num_links': num_links,
    }


def _build_split_assignment_artifact(assignments: np.ndarray) -> bytes:
    arr = np.asarray(assignments, dtype=np.uint8)
    row_count = int(arr.shape[0])
    magic = b'DGSPLT4\x00'
    header = struct.pack('<IQ', 1, row_count)
    return magic + header + arr.tobytes(order='C')


# Backward-compatible aliases while downstream callers migrate.
_build_stats_artifact_v1 = _build_stats_artifact
_build_split_assignment_artifact_v1 = _build_split_assignment_artifact


def execute_tt_plan_with_artifacts(
    *,
    plan_json: str,
    index_path: str,
    output_path: str,
    split_config: Optional[Any],
    split_family_hint: Optional[str] = None,
    require_authoritative: bool = True,
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

    row_count = 0
    native_summary: Optional[dict[str, Any]] = None

    materialize_fn = _execute_tt_plan_materialize
    if callable(materialize_fn):
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

    artifacts_dir = output_path + '.artifacts_v4'
    os.makedirs(artifacts_dir, exist_ok=True)

    stats_payload = _build_stats_artifact(native_summary, row_count)
    stats_bytes = json.dumps(stats_payload, sort_keys=True).encode('utf-8')
    stats_path = os.path.join(artifacts_dir, 'stats.tt.v1.json')
    with open(stats_path, 'wb') as f:
        f.write(stats_bytes)

    split_spec = normalize_split_spec(
        split_config,
        split_family_hint=split_family_hint,
        require=True,
    )
    assert split_spec is not None
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
    if require_authoritative and not split_meta.get('authoritative', False):
        raise RuntimeError(
            'diskgraph split assignment is non-authoritative '
            f"(family={split_meta.get('family')}, reason={split_meta.get('reason')})"
        )

    split_bytes = _build_split_assignment_artifact(assignments)
    split_path = os.path.join(artifacts_dir, 'split_assignments.row_stage.v1.bin')
    with open(split_path, 'wb') as f:
        f.write(split_bytes)

    manifest_path = output_path + '.manifest.v4.json'
    manifest = {
        'version': 4,
        'row_count': row_count,
        'train_table': {'path': output_path, 'format': 'parquet_with_arange'},
        'split': {
            'path': split_path,
            'format': 'split_assignment_row_stage_v1_bin',
            'authoritative': bool(split_meta.get('authoritative', False)),
            'family': split_meta.get('family'),
            'reason': split_meta.get('reason'),
        },
        'stats': {'path': stats_path, 'format': 'tt_stats_v1_json'},
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
