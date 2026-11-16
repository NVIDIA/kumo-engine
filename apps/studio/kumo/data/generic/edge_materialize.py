# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import os.path as osp
import shutil
from typing import Any, Dict, Tuple

from kumo_core.types.edges import PathDict

from io_utils import fsutil, logging
from io_utils.json import load_json, save_json
from kumo.config.graph import TableMetadataConfig
from kumo.diskgraph.diskgraph_native_pred_artifacts import (
    get_validated_prediction_edge_bundle,
    parse_prediction_table_edges_artifact,
)
from kumo.diskgraph.diskgraph_native_tt_artifacts import (
    get_validated_artifact_bundle,
    parse_train_table_edges_artifact,
    resolve_manifest_artifact_paths,
)

_EDGE_PARQUET_FILE = 'edge.parquet'
_MASK_PARQUET_FILE = 'mask.parquet'

# NOTE must be aligned with graphstore_helper:_SUCCESS_FILE; we are not
# explicitly using that file to avoid a dependency between data and engine:
_GRAPHSTORE_SUCCESS_FILE = '__SUCCESS_EDGES__'

_FKEY_COLS_FILE = 'fkey_cols.parquet'
_PRED_TABLE_NAME = 'pred_table'

logger = logging.getLogger(__name__)


def get_materialized_num_nodes(table_dir: str) -> int:
    r"""Get the number of nodes in a table based on the persisted attributes
    when the table was materialized.

    Args:
        table_dir (str): directory table was persisted to
    """
    attr_path = osp.join(table_dir, 'attributes.json')
    attributes = load_json(attr_path, Dict, default=None,
                           raise_on_missing=False)
    if attributes is not None:
        return attributes['num_nodes']
    raise RuntimeError(f"Cannot determine num_nodes for table at {table_dir} — "
                       "attributes.json is missing.")


def _write_edge_attributes(edge_dir: str, count: int):
    attributes = {'edge_count': count}
    save_json(attributes, osp.join(edge_dir, 'attributes.json'))


def _parquet_row_count(path: str) -> int:
    """Get row count from parquet metadata without loading data."""
    import pyarrow.dataset as ds

    # PyArrow doesn't recognize s3a:// (Spark/Hadoop scheme); convert to s3://.
    path = fsutil.s3a_as_s3(path)
    dataset = ds.dataset(path, format='parquet')
    return dataset.count_rows()


def _consolidate_sharded_parquet(base_dir: str):
    """Merge sharded parquet directories (part-*.parquet) into single files.

    Rust export_tt_artifacts produces ~1400 tiny part files per artifact
    (one per DiskGraph index shard).  Uploading them individually to S3 is
    extremely slow (~60ms each → minutes for thousands of files).

    This walks the entire staging tree and rewrites every sharded
    ``*.parquet/`` directory as a single parquet file.  That includes
    top-level artifacts (``fkey_cols.parquet/``, ``mask.parquet/``) AND
    nested ones (``{edge_dir}/edge.parquet/``).
    """
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    # Walk bottom-up so we consolidate deepest dirs first (e.g.
    # edge.parquet/ inside a reverse-edge dir) before the parent is
    # processed.
    for dirpath, dirnames, _filenames in os.walk(base_dir, topdown=False):
        for dirname in dirnames:
            if not dirname.endswith('.parquet'):
                continue
            parquet_dir = osp.join(dirpath, dirname)
            parts = [
                f for f in os.listdir(parquet_dir) if f.startswith('part-')
            ]
            if len(parts) <= 1:
                continue

            # Read all parts → single file.
            dataset = ds.dataset(parquet_dir, format='parquet')
            tmp_file = parquet_dir + '.__consolidating__'
            pq.write_table(dataset.to_table(), tmp_file, compression='snappy')

            # Atomically replace directory with the single file.
            shutil.rmtree(parquet_dir)
            os.rename(tmp_file, parquet_dir)
            logger.info("Consolidated %d parquet parts into single file: %s",
                        len(parts), osp.relpath(parquet_dir, base_dir))


def _resolve_edge_dst_dir(
    bundle_edge_key: str,
    edge_persist_paths: PathDict,
    graph_path: str,
) -> str:
    """Resolve the destination directory for a bundle edge key."""
    direct = edge_persist_paths.get(bundle_edge_key)
    if isinstance(direct, str):
        return direct

    parts = bundle_edge_key.split('__')
    if len(parts) == 3:
        b_src, b_rel, b_dst = parts
        candidates: list[Tuple[str, str]] = []
        for expected_key, expected_path in edge_persist_paths.items():
            expected_parts = expected_key.split('__')
            if len(expected_parts) != 3:
                continue
            e_src, e_rel, e_dst = expected_parts
            if e_rel != b_rel:
                continue
            if e_src.split('.')[-1] != b_src.split('.')[-1]:
                continue
            # Strip "custom_" prefix for comparison: custom TT
            # ingestion renames train_table → custom_train_table but
            # the DG edge bundle still uses the original name.
            e_dst_name = e_dst.split('.')[-1]
            b_dst_name = b_dst.split('.')[-1]
            if e_dst_name.startswith('custom_'):
                e_dst_name = e_dst_name[len('custom_'):]
            if e_dst_name != b_dst_name:
                continue
            candidates.append((expected_key, expected_path))

        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            raise RuntimeError(
                'diskgraph strict path found ambiguous edge persist path '
                f'for bundle key {bundle_edge_key!r}: '
                f'{[key for key, _ in candidates]!r}')

    if edge_persist_paths:
        raise RuntimeError(
            'diskgraph strict path could not resolve edge persist path for '
            f'bundle key {bundle_edge_key!r}; expected one of '
            f'{list(edge_persist_paths.keys())!r}')

    return osp.join(graph_path, bundle_edge_key)


def _bundle_destination_already_materialized(
    parsed_bundle: Dict[str, Any],
    table: TableMetadataConfig,
    graph_path: str,
    edge_persist_paths: PathDict,
) -> bool:
    """True iff every per-edge destination plus the table-level
    fkey_cols / mask sidecars are already present.

    "Present" means the ``__SUCCESS_EDGES__`` marker exists in each
    edge destination dir and the sidecar paths exist. The marker is
    the last write in ``_materialize_edges_from_bundle``'s per-edge
    loop, so its presence implies the copy completed.
    """
    reverse_edges = parsed_bundle.get('reverse_edges')
    if not isinstance(reverse_edges, dict) or not reverse_edges:
        return False

    fkey_cols_dst = osp.join(graph_path, f'{table.name}_{_FKEY_COLS_FILE}')
    if not fsutil.exists(fkey_cols_dst):
        return False

    mask = parsed_bundle.get('mask') or {}
    mask_dst = osp.join(graph_path, f'{table.name}_{_MASK_PARQUET_FILE}')
    if bool(mask.get('present')) and not fsutil.exists(mask_dst):
        return False

    for edge_key in reverse_edges:
        if not isinstance(edge_key, str) or not edge_key:
            return False
        edge_dst_dir = _resolve_edge_dst_dir(edge_key, edge_persist_paths,
                                             graph_path)
        if not fsutil.exists(osp.join(edge_dst_dir, _GRAPHSTORE_SUCCESS_FILE)):
            return False
    return True


def _try_short_circuit_bundle_materialize(
    parsed_bundle: Dict[str, Any],
    table: TableMetadataConfig,
    graph_path: str,
    edge_persist_paths: PathDict,
    artifact_label: str,
) -> bool:
    """Log + return True if every destination edge + sidecar is already
    on disk for ``parsed_bundle``; False otherwise.

    Called from the BP / training wrappers BEFORE the bundle's source
    paths are touched (the validators that follow walk those paths to
    sha256-check them — unreachable cross-pod when the bundle's edge
    artifact files live on a different pod's local SSD).
    """
    if not _bundle_destination_already_materialized(
            parsed_bundle, table, graph_path, edge_persist_paths):
        return False
    logger.info('%s already materialized for %s; skipping bundle re-copy',
                artifact_label, table.name)
    return True


def _materialize_edges_from_bundle(
    edges_bundle: Dict[str, Any],
    table: TableMetadataConfig,
    graph_path: str,
    edge_persist_paths: PathDict,
    artifact_label: str,
) -> bool:
    """Shared logic for materializing edges from a diskgraph artifact bundle.

    Works for both prediction-table and train-table edge bundles.
    """
    reverse_edges = edges_bundle.get('reverse_edges')
    if not isinstance(reverse_edges, dict):
        raise RuntimeError(
            f'diskgraph strict path requires {artifact_label}.reverse_edges map'
        )

    if not reverse_edges:
        if len(getattr(table, 'fkeys', []) or []) > 0:
            raise RuntimeError(
                f'diskgraph strict path requires non-empty '
                f'{artifact_label}.reverse_edges bundle when '
                f'{artifact_label.split("_")[0]}-table fkeys are present')
        logger.info(
            'No %s edge artifacts declared for %s (no fkeys); '
            'skipping edge materialization',
            artifact_label,
            table.name,
        )
        return True

    fkey_cols = edges_bundle.get('fkey_cols')
    if not isinstance(fkey_cols, dict):
        raise RuntimeError(
            f'diskgraph strict path requires {artifact_label}.fkey_cols artifact'
        )
    fkey_cols_src = fkey_cols.get('path')
    if not isinstance(fkey_cols_src, str) or not fkey_cols_src:
        raise RuntimeError(
            f'diskgraph strict path requires {artifact_label}.fkey_cols.path')
    fkey_cols_dst = osp.join(graph_path, f'{table.name}_{_FKEY_COLS_FILE}')
    if fsutil.exists(fkey_cols_dst):
        fsutil.rm(fkey_cols_dst, ignore_errors=True)
    fsutil.copy(fkey_cols_src, fkey_cols_dst, overwrite=True)

    mask = edges_bundle.get('mask')
    if not isinstance(mask, dict):
        raise RuntimeError(
            f'diskgraph strict path requires {artifact_label}.mask object')
    mask_dst = osp.join(graph_path, f'{table.name}_{_MASK_PARQUET_FILE}')
    if bool(mask.get('present')):
        mask_src = mask.get('path')
        if not isinstance(mask_src, str) or not mask_src:
            raise RuntimeError(
                f'diskgraph strict path requires {artifact_label}.mask.path')
        if fsutil.exists(mask_dst):
            fsutil.rm(mask_dst, ignore_errors=True)
        fsutil.copy(mask_src, mask_dst, overwrite=True)
    elif fsutil.exists(mask_dst):
        fsutil.rm(mask_dst, ignore_errors=True)

    for edge_key, edge_info in reverse_edges.items():
        if not isinstance(edge_key, str) or not edge_key:
            raise RuntimeError(
                f'diskgraph strict path {artifact_label} contains invalid edge key'
            )
        if not isinstance(edge_info, dict):
            raise RuntimeError(
                f'diskgraph strict path {artifact_label}[{edge_key!r}] must be object'
            )
        edge_src = edge_info.get('path')
        if not isinstance(edge_src, str) or not edge_src:
            raise RuntimeError(
                f'diskgraph strict path {artifact_label}[{edge_key!r}].path missing'
            )

        edge_dst_dir = _resolve_edge_dst_dir(edge_key, edge_persist_paths,
                                             graph_path)
        fsutil.makedirs(edge_dst_dir, exist_ok=True)

        edge_dst = osp.join(edge_dst_dir, _EDGE_PARQUET_FILE)
        if fsutil.exists(edge_dst):
            fsutil.rm(edge_dst, ignore_errors=True)
        fsutil.copy(edge_src, edge_dst, overwrite=True)

        row_count = _parquet_row_count(edge_dst)
        _write_edge_attributes(edge_dst_dir, row_count)
        fsutil.touch(osp.join(edge_dst_dir, _GRAPHSTORE_SUCCESS_FILE))

    logger.info(
        'Materialized %s edge artifacts from diskgraph bundle for %s',
        artifact_label,
        table.name,
    )
    return True


def _materialize_prediction_table_edges_from_bundle(
    table: TableMetadataConfig,
    table_paths: Dict[str, str],
    graph_path: str,
    edge_persist_paths: PathDict,
) -> bool:
    pred_table_path = table_paths.get(table.name)
    if not isinstance(pred_table_path, str) or not pred_table_path:
        return False

    candidate_bundle_paths = [
        f'{pred_table_path}.prediction_edges.manifest.json',
        f'{pred_table_path}.prediction_edges.json',
    ]
    bundle_path = next(
        (path for path in candidate_bundle_paths if fsutil.exists(path)),
        None,
    )
    if not isinstance(bundle_path, str):
        return False

    # Parse the bundle JSON without sha256 validation so the
    # short-circuit can fire before any source-side read.
    # ``get_validated_prediction_edge_bundle`` walks the source paths
    # to checksum them; when the source artifacts live on a different
    # pod's local SSD (pool BP cross-pod), that walk raises
    # FileNotFoundError before we ever get to consult the destination.
    parsed = parse_prediction_table_edges_artifact(
        json.loads(fsutil.read_text(bundle_path)))
    if _try_short_circuit_bundle_materialize(parsed, table, graph_path,
                                             edge_persist_paths,
                                             'prediction_table_edges'):
        return True

    bundle = get_validated_prediction_edge_bundle(bundle_path)
    return _materialize_edges_from_bundle(
        bundle,
        table,
        graph_path,
        edge_persist_paths,
        artifact_label='prediction_table_edges',
    )


def _materialize_train_table_edges_from_manifest_bundle(
    table: TableMetadataConfig,
    table_paths: Dict[str, str],
    graph_path: str,
    edge_persist_paths: PathDict,
) -> bool:
    train_table_path = table_paths.get(table.name)
    if not isinstance(train_table_path, str) or not train_table_path:
        return False

    manifest_path = f'{train_table_path}.manifest.json'
    if not fsutil.exists(manifest_path):
        return False

    # Parse the manifest JSON without sha256 validation so the
    # short-circuit can fire before any source-side read; see the
    # equivalent comment in
    # ``_materialize_prediction_table_edges_from_bundle``. Manifest
    # paths post-#28723 are relative-to-manifest-dir; resolve them
    # before parsing the edges section.
    raw_manifest = json.loads(fsutil.read_text(manifest_path))
    resolve_manifest_artifact_paths(raw_manifest, osp.dirname(manifest_path))
    raw_edges = raw_manifest.get('train_table_edges')
    if isinstance(raw_edges, dict):
        try:
            parsed_peek = parse_train_table_edges_artifact(raw_edges)
        except Exception:
            parsed_peek = None
        if isinstance(parsed_peek, dict):
            if _try_short_circuit_bundle_materialize(parsed_peek, table,
                                                     graph_path,
                                                     edge_persist_paths,
                                                     'train_table_edges'):
                return True

    manifest = get_validated_artifact_bundle(manifest_path)
    edges_bundle = manifest.get('parsed_train_table_edges')
    if not isinstance(edges_bundle, dict):
        return False

    return _materialize_edges_from_bundle(
        edges_bundle,
        table,
        graph_path,
        edge_persist_paths,
        artifact_label='train_table_edges',
    )
