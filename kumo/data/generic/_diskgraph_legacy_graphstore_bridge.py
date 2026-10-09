# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LEGACY-GRAPHSTORE BRIDGE — DELETE WHEN KUMO USES RUST-NATIVE SAMPLER.

Consolidates all transitional post-`execute_tt_plan` work needed to make
kumo's legacy GraphStore + Python sampler + PyG pipeline consume the new
TT-NG output:

  1. Inject arange row-index column into tt.parquet (legacy connector
     reads with pkey='arange'; Rust dataloader reads arange.bin sidecar
     directly so this step disappears in end-state).
  2. Update manifest's train_table_sha256 to reflect the rewritten parquet.
  3. Call diskgraph.export_tt_artifacts to produce reverse-edge dirs +
     fkey_cols.parquet + mask.parquet (consumed by legacy GraphStore;
     Rust sampler resolves these on-the-fly so they disappear in end-state).
  4. Write the train_table_edges manifest entry pointing at those files.

To delete this bridge:
  - rm this file
  - revert diskgraph_train_table.py's `apply_legacy_bridge` call back to
    `return tt_parquet_path, manifest_path` directly
  - remove fallback skip in kumo/task/task.py:551-562 (col-stats seeding)
  - drop train_table_edges manifest field validation in kumo/diskgraph/
    diskgraph_native_tt_artifacts.py if no longer needed

The bridge is gated by ``train_table_edges_config['enabled']`` and produces
NOTHING when the train table has no fkeys — in that case it only does the
arange + checksum work, plus emits a disabled-stub manifest entry so the
strict validator in ``diskgraph_native_tt_artifacts.parse_train_table_edges_artifact``
keeps passing.
"""
from __future__ import annotations

import hashlib
import json as pyjson
import os
import os.path as osp
from typing import Any, Optional

from kumo_core.types.edges import EdgeKey

from io_utils import logging
from kumo.diskgraph.diskgraph_native_tt_artifacts import _sha256_path

logger = logging.getLogger(__name__)


def apply_legacy_bridge(
    *,
    producer_result: dict,
    manifest_path: str,
    index_path: str,
    train_table_edges_config: Optional[dict[str, Any]],
) -> str:
    """Run all bridge work atomically; return the (possibly rewritten)
    tt.parquet path.

    Mutates ``manifest.json`` in place. Idempotent for retries (writes via
    ``*.tmp`` + ``os.replace``).

    Args:
        producer_result: dict returned by ``diskgraph.execute_tt_plan``;
            must contain ``tt_parquet_path``.
        manifest_path: path to the manifest.json that will be mutated.
        index_path: path to the diskgraph index dir (used by
            ``export_tt_artifacts`` for fkey resolution).
        train_table_edges_config: edge bundle config from kumo's
            ``train_table_generator``; ``None`` / ``enabled=False`` /
            empty fkeys means skip the export step entirely and just
            stamp the disabled stub.

    Returns:
        Absolute path to the (rewritten) tt.parquet file. Identical to
        ``producer_result['tt_parquet_path']`` since the rewrite is in-place.
    """
    src_parquet_path = producer_result.get('tt_parquet_path')
    if not src_parquet_path:
        raise RuntimeError(
            'legacy_graphstore_bridge requires producer_result["tt_parquet_path"]'
        )

    # Step 1 + 2: arange + weight injection + manifest checksum patch.
    # Combined into a single parquet rewrite pass (one read, append both
    # columns, one write). Weight injection is a no-op when manifest's
    # `weight_sidecar` field is null (PQ path or custom-TT plan with no
    # weight_col).
    _inject_arange_weight_and_patch_checksum(src_parquet_path, manifest_path)

    # Step 3 + 4: export edge artifacts (or stub) + manifest entry.
    edges_cfg = train_table_edges_config if isinstance(train_table_edges_config,
                                                       dict) else None
    fkeys = list(edges_cfg.get('fkeys') or []) if edges_cfg else []
    enabled = bool(edges_cfg and edges_cfg.get('enabled', False) and fkeys)

    if not enabled:
        _write_disabled_stub(manifest_path, fkeys)
        logger.info(
            'legacy_graphstore_bridge: skipped edge export '
            '(enabled=%s, fkeys=%d); wrote disabled-stub manifest entry',
            bool(edges_cfg and edges_cfg.get('enabled', False)),
            len(fkeys),
        )
        return src_parquet_path

    _export_train_table_edges_and_patch_manifest(
        src_parquet_path=src_parquet_path,
        manifest_path=manifest_path,
        index_path=index_path,
        edges_cfg=edges_cfg,
    )
    return src_parquet_path


# --------------------------------------------------------------------------- #
# Pre-bridge: normalize legacy column names to canonical multi-entity shape    #
# --------------------------------------------------------------------------- #


def normalize_legacy_parquet_for_rehydration(
    parquet_path: str,
    *,
    output_dir: str,
    normalize_timestamp: bool = False,
) -> str:
    """Cast a customer-supplied parquet's anchor to the canonical nanosecond
    unit and write it to ``<output_dir>/normalized_input.parquet``.

    Renames nothing. A supplied table names its entity column ``ENTITY1`` and
    its target ``TARGET1`` -- those are the names DiskGraph reads, and a table
    naming anything else is refused by
    :mod:`kumo_core.training.custom_table` rather than bridged into shape
    here. This used to map ``ENTITY``, ``TARGET`` and the graph's own
    primary-key column onto the canonical names, which made a misnamed table
    validate on one path and fail on every other.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    dataset = ds.dataset(parquet_path, format='parquet', partitioning='hive')
    os.makedirs(output_dir, exist_ok=True)
    out_path = osp.join(output_dir, 'normalized_input.parquet')
    writer = None
    try:
        for batch in dataset.to_batches():
            arrays = list(batch.columns)
            fields = []
            for index, field in enumerate(batch.schema):
                if (normalize_timestamp and field.name == 'TIMESTAMP'
                        and pa.types.is_timestamp(field.type)):
                    arrays[index] = arrays[index].cast(pa.timestamp('ns'))
                    field = pa.field(field.name, pa.timestamp('ns'),
                                     nullable=field.nullable)
                fields.append(field)
            normalized = pa.RecordBatch.from_arrays(arrays,
                                                    schema=pa.schema(fields))
            if writer is None:
                writer = pq.ParquetWriter(out_path, normalized.schema,
                                          compression='snappy')
            writer.write_batch(normalized)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(pa.Table.from_batches([], schema=dataset.schema),
                       out_path, compression='snappy')
    return out_path


# --------------------------------------------------------------------------- #
# Step 1 + 2: arange injection + checksum patch                                #
# --------------------------------------------------------------------------- #


def _inject_arange_weight_and_patch_checksum(src_parquet_path: str,
                                             manifest_path: str) -> None:
    """Append int64 arange row-index column AND optional float32 weight
    column to tt.parquet (single rewrite pass), rename DG physical
    multi-entity/multi-target column names to kumo's logical aliases
    (ENTITY1→ENTITY, TARGET1→TARGET — slot-0 only), and refresh
    manifest['checksums']['train_table_sha256'].

    Weight injection is a no-op when the manifest's `weight_sidecar`
    field is null (PQ path or custom-TT plan without `weight_col`). When
    non-null, reads the DGWGHT-formatted weight.bin once at the top,
    parses the f32 payload as a numpy array, and slices the array
    per-batch to append a `weight` (Float32) column to each batch.

    Both arange and weight injection are idempotent — if the column is
    already present in tt.parquet (e.g. from a retried run), the inject
    is skipped for that column."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from diskgraph import read_tt_parquet_batches

    # See _diskgraph_legacy_graphstore_bridge module docstring + the col-stats
    # alias map in kumo/task/task.py: kumo's split logic, materializer, and
    # training-time parquet read all key off logical names. End-state: kumo
    # adopts DG's multi-entity schema natively and this rename disappears.
    _DG_TO_KUMO_COL = {'ENTITY1': 'ENTITY', 'TARGET1': 'TARGET'}

    def _rename_columns(batch: pa.RecordBatch) -> pa.RecordBatch:
        new_names = [
            _DG_TO_KUMO_COL.get(name, name) for name in batch.schema.names
        ]
        if new_names == batch.schema.names:
            return batch
        new_schema = pa.schema([
            batch.schema.field(i).with_name(new_names[i])
            for i in range(len(new_names))
        ])
        return pa.RecordBatch.from_arrays(batch.columns, schema=new_schema)

    # Read manifest first to discover weight_sidecar (if any).
    with open(manifest_path) as mf:
        manifest_obj = pyjson.load(mf)
    weight_sidecar = manifest_obj.get('weight_sidecar')
    weight_payload: Optional[Any] = None  # numpy.ndarray of dtype=float32
    weight_path: Optional[str] = None
    if weight_sidecar is not None:
        weight_path = weight_sidecar.get('path')
        if not weight_path:
            raise RuntimeError(
                'legacy_graphstore_bridge: manifest.weight_sidecar present '
                'but missing path')
        weight_payload = _load_weight_bin(weight_path)

    tmp_parquet_path = src_parquet_path + '.with_arange.tmp'
    writer = None
    total_rows = 0
    weight_consumed = 0
    try:
        for batch in read_tt_parquet_batches(src_parquet_path):
            num_rows = batch.num_rows
            schema_names = list(batch.schema.names)

            # Arange injection (idempotent — skip if already present).
            if 'arange' not in schema_names:
                arange_array = pa.array(
                    range(total_rows, total_rows + num_rows), type=pa.int64())
                batch = batch.append_column('arange', arange_array)

            # Weight injection (idempotent — skip if `weight` column
            # already in batch). Only runs when weight_payload was loaded
            # from the manifest's `weight_sidecar`.
            if weight_payload is not None and 'weight' not in schema_names:
                start = weight_consumed
                end = start + num_rows
                if end > len(weight_payload):
                    raise RuntimeError(
                        f'weight.bin payload too short: total tt.parquet rows '
                        f'so far {end}, but weight payload has only '
                        f'{len(weight_payload)} entries')
                slice_arr = weight_payload[start:end]
                batch = batch.append_column(
                    'weight', pa.array(slice_arr, type=pa.float32()))
                weight_consumed = end

            batch = _rename_columns(batch)
            if writer is None:
                writer = pq.ParquetWriter(tmp_parquet_path, batch.schema,
                                          compression='snappy')
            writer.write_batch(batch)
            total_rows += num_rows
    finally:
        if writer is not None:
            writer.close()

    # Defensive: when weights were loaded, total emitted rows must equal
    # the weight payload length (both follow the writer's emitted count).
    if weight_payload is not None and weight_consumed != total_rows:
        # weight_consumed only advances when we actually injected; if
        # the existing parquet already had a 'weight' column, we skipped
        # injection and weight_consumed stays 0. In that idempotent case
        # we trust the existing column. Otherwise mismatch is fatal.
        if weight_consumed > 0:
            raise RuntimeError(
                f'weight injection mismatch: injected {weight_consumed} '
                f'weights but tt.parquet has {total_rows} rows')

    os.replace(tmp_parquet_path, src_parquet_path)

    with open(src_parquet_path, 'rb') as f:
        new_train_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Atomic manifest checksum update (tmp + replace).
    checksums = manifest_obj.setdefault('checksums', {})
    checksums['train_table_sha256'] = new_train_sha256
    tmp_manifest = manifest_path + '.tmp'
    with open(tmp_manifest, 'w') as mf:
        pyjson.dump(manifest_obj, mf, indent=2)
    os.replace(tmp_manifest, manifest_path)


def _load_weight_bin(path: str):
    """Parse a DGWGHT-formatted weight sidecar and return a numpy array
    of dtype=float32. Validates magic + version + dtype tag.

    File format (matches diskgraph-train-table/src/weight_bin.rs):
      - 8 bytes: magic b'DGWGHT\\0\\0'
      - 4 bytes: version u32 LE (must be 1)
      - 4 bytes: dtype tag u32 LE (must be 1 for f32)
      - 8 bytes: row_count u64 LE
      - row_count * 4 bytes: payload f32 LE
    """
    import numpy as np

    with open(path, 'rb') as f:
        bytes_data = f.read()
    if len(bytes_data) < 24:
        raise RuntimeError(
            f'weight.bin {path} too short: {len(bytes_data)} bytes')
    if bytes_data[0:8] != b'DGWGHT\x00\x00':
        raise RuntimeError(f'weight.bin {path} bad magic: {bytes_data[0:8]!r}')
    version = int.from_bytes(bytes_data[8:12], 'little')
    if version != 1:
        raise RuntimeError(f'weight.bin {path} unexpected version {version}')
    dtype_tag = int.from_bytes(bytes_data[12:16], 'little')
    if dtype_tag != 1:
        raise RuntimeError(
            f'weight.bin {path} unsupported dtype tag {dtype_tag} (only f32 supported)'
        )
    row_count = int.from_bytes(bytes_data[16:24], 'little')
    expected_payload_len = row_count * 4
    if len(bytes_data) - 24 != expected_payload_len:
        raise RuntimeError(
            f'weight.bin {path} payload size mismatch: header says {row_count} '
            f'rows ({expected_payload_len} bytes), got {len(bytes_data) - 24}')
    arr = np.frombuffer(bytes_data[24:], dtype=np.float32)
    if arr.shape[0] != row_count:
        raise RuntimeError(
            f'weight.bin {path} array length {arr.shape[0]} != header row_count {row_count}'
        )
    return arr


# --------------------------------------------------------------------------- #
# Step 3 + 4: export edge artifacts + manifest entry                           #
# --------------------------------------------------------------------------- #


def _export_train_table_edges_and_patch_manifest(
    *,
    src_parquet_path: str,
    manifest_path: str,
    index_path: str,
    edges_cfg: dict[str, Any],
) -> None:
    """Run ``diskgraph.export_tt_artifacts`` to produce per-fkey reverse-edge
    dirs + fkey_cols.parquet + mask.parquet next to the train table, rename
    the diskgraph-produced reverse-edge dirs to FQ EdgeKey-string names that
    kumo's GraphStore expects, and patch ``manifest['train_table_edges']``
    with paths the strict validator + materializer will accept.
    """
    try:
        from diskgraph import ExportConfig, FkeyConfig, export_tt_artifacts
    except Exception as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            'legacy_graphstore_bridge: failed to import diskgraph runtime '
            'for train-table edge export') from e

    fkeys = list(edges_cfg.get('fkeys') or [])
    train_table_name = edges_cfg.get('train_table_name', 'train_table')
    row_idx_col = edges_cfg.get('row_idx_col', 'arange')
    mask_present = bool(edges_cfg.get('mask_present', True))

    output_dir = osp.dirname(src_parquet_path) or src_parquet_path

    # Build ExportConfig. The fkey config dicts emitted by kumo's
    # train_table_generator use {'column_name', 'target_table'} keys
    # (target_table is already the un-prefixed/index-table name).
    #
    # kumo emits logical column names ('ENTITY', 'TARGET' — see the
    # constants in train_table_generator.py). After the arange step
    # (_inject_arange_and_patch_checksum) we also renamed DG physical
    # multi-entity/multi-target names (ENTITY1, TARGET1) to those
    # logical aliases, so tt.parquet now has the kumo-logical names
    # and we can pass column_name straight to FkeyConfig.
    fkey_configs = []
    fkey_logical_names: list[str] = []
    for fk in fkeys:
        column_name = fk.get('column_name') or fk.get('name')
        target_table = fk.get('target_table') or fk.get('dst_table')
        if not column_name or not target_table:
            raise RuntimeError(
                'legacy_graphstore_bridge: invalid fkey entry (need '
                f'column_name + target_table): {fk!r}')
        fkey_configs.append(FkeyConfig(column_name, target_table))
        fkey_logical_names.append(column_name)

    export_config = ExportConfig(
        train_table_path=src_parquet_path,
        index_path=index_path,
        output_dir=output_dir,
        fkeys=fkey_configs,
        row_idx_col=row_idx_col,
        train_table_name=train_table_name,
    )
    export_tt_artifacts(export_config)

    # Rename reverse-edge dirs to FQ EdgeKey-string names.
    # diskgraph emits: {plain_dst}__rev_{fkey}__{src}
    # GraphStore expects: EdgeKey(<src>, <fkey>, <dst>, is_reverse=True).to_str()
    # which produces: {dst}__rev_{fkey}__{src} where dst here equals what
    # the producer wrote. We rename so that downstream code keying off
    # EdgeKey strings (kumo/data/generic/edge_materialize.py:_resolve_edge_dst_dir
    # and graphstore_helper) finds the dirs by the canonical key.
    #
    # NOTE: since both diskgraph-produced and EdgeKey-produced names use
    # the same `target_table` string verbatim (the FkeyConfig.target_table
    # we just passed in), the rename is a no-op when callers pass FQ names.
    # We still walk the rename map so the manifest paths point at canonical
    # locations and the loop is robust to any future divergence.
    reverse_edges_manifest: dict[str, dict[str, str]] = {}
    for fk_cfg, logical_name in zip(fkey_configs, fkey_logical_names):
        physical_name = fk_cfg.column_name
        target_table = fk_cfg.target_table
        # DG writes the dir using the FkeyConfig.column_name (physical).
        dg_dir_name = (
            f'{target_table}__rev_{physical_name}__{train_table_name}')
        dg_dir = osp.join(output_dir, dg_dir_name)

        # GraphStore looks up by EdgeKey built from the LOGICAL fkey name.
        canonical_key = EdgeKey(target_table, logical_name, train_table_name,
                                is_reverse=True).to_str()
        canonical_dir = osp.join(output_dir, canonical_key)

        if dg_dir != canonical_dir:
            if osp.exists(dg_dir) and not osp.exists(canonical_dir):
                os.rename(dg_dir, canonical_dir)
            elif not osp.exists(canonical_dir):
                raise RuntimeError(
                    'legacy_graphstore_bridge: expected diskgraph reverse '
                    f'edge dir {dg_dir} (or canonical {canonical_dir}) to '
                    'exist after export_tt_artifacts')

        edge_parquet_dir = osp.join(canonical_dir, 'edge.parquet')
        if not osp.exists(edge_parquet_dir):
            raise RuntimeError(
                'legacy_graphstore_bridge: expected edge.parquet at '
                f'{edge_parquet_dir} after export_tt_artifacts')
        reverse_edges_manifest[canonical_key] = {
            'path': edge_parquet_dir,
            'format': 'edge_parquet',
        }

    fkey_cols_dir = osp.join(output_dir,
                             f'{train_table_name}_fkey_cols.parquet')
    if not osp.exists(fkey_cols_dir):
        raise RuntimeError(
            'legacy_graphstore_bridge: expected fkey_cols dir at '
            f'{fkey_cols_dir} after export_tt_artifacts')
    mask_dir = osp.join(output_dir, f'{train_table_name}_mask.parquet')
    if mask_present and not osp.exists(mask_dir):
        raise RuntimeError('legacy_graphstore_bridge: expected mask dir at '
                           f'{mask_dir} after export_tt_artifacts')

    # Compute per-artifact checksums for the bundle. kumo's
    # validate_artifact_bundle requires:
    #   - inner edge_checksums: fkey_cols_sha256, mask_sha256 (if mask
    #     present), <edge_key>_sha256 for each reverse edge dir
    #   - outer manifest checksums: train_table_edges_sha256 = sha256 of
    #     the bundle JSON (sort_keys)
    edge_checksums: dict[str, str] = {
        'fkey_cols_sha256': _sha256_path(fkey_cols_dir),
    }
    if mask_present:
        edge_checksums['mask_sha256'] = _sha256_path(mask_dir)
    for edge_key, edge_info in reverse_edges_manifest.items():
        edge_checksums[f'{edge_key}_sha256'] = _sha256_path(edge_info['path'])

    manifest_entry: dict[str, Any] = {
        'version':
        1,
        'kind':
        'tt_train_table_edges',
        'format':
        'tt_edge_bundle_v1',
        'enabled':
        True,
        'fkeys': [{
            'column_name': logical_name,
            'target_table': fk_cfg.target_table,
        } for fk_cfg, logical_name in zip(fkey_configs, fkey_logical_names)],
        'reverse_edges':
        reverse_edges_manifest,
        'fkey_cols': {
            'path': fkey_cols_dir,
            'format': 'fkey_cols_parquet',
        },
        'mask': {
            'present': mask_present,
            'path': mask_dir if mask_present else '',
            'format': 'mask_parquet',
        },
        'checksums':
        edge_checksums,
    }

    bundle_sha256 = hashlib.sha256(
        pyjson.dumps(manifest_entry, sort_keys=True).encode()).hexdigest()
    _write_train_table_edges_entry(manifest_path, manifest_entry,
                                   bundle_sha256=bundle_sha256)
    logger.info(
        'legacy_graphstore_bridge: exported %d reverse edges + fkey_cols '
        '(+ mask=%s) to %s',
        len(fkey_configs),
        mask_present,
        output_dir,
    )


def _write_disabled_stub(manifest_path: str, fkeys: list[dict[str,
                                                              Any]]) -> None:
    """Write the disabled-stub train_table_edges manifest entry that matches
    kumo's synthesized-default behavior in
    ``diskgraph_native_tt_artifacts.validate_artifact_bundle``."""
    declared_fkeys = []
    for fk in fkeys:
        column_name = fk.get('column_name') or fk.get('name')
        target_table = fk.get('target_table') or fk.get('dst_table')
        if column_name and target_table:
            declared_fkeys.append({
                'column_name': column_name,
                'target_table': target_table,
            })
    stub = {
        'version': 1,
        'kind': 'tt_train_table_edges',
        'format': 'tt_edge_bundle_v1',
        'enabled': False,
        'fkeys': declared_fkeys,
        'reverse_edges': {},
    }
    _write_train_table_edges_entry(manifest_path, stub)


def _write_train_table_edges_entry(manifest_path: str, entry: dict[str, Any],
                                   bundle_sha256: Optional[str] = None) -> None:
    with open(manifest_path) as mf:
        manifest_obj = pyjson.load(mf)
    manifest_obj['train_table_edges'] = entry
    if bundle_sha256 is not None:
        checksums = manifest_obj.setdefault('checksums', {})
        checksums['train_table_edges_sha256'] = bundle_sha256
    tmp_manifest = manifest_path + '.tmp'
    with open(tmp_manifest, 'w') as mf:
        pyjson.dump(manifest_obj, mf, indent=2)
    os.replace(tmp_manifest, manifest_path)
