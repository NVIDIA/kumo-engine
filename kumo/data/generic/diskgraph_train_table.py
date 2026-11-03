# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import os.path as osp
import stat
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from diskgraph import TimeUnit
from kumo_core.training.table import (
    TrainTableRequest,
    apply_reference_resolution,
)
from kumo_core.training.table import \
    build_split_spec as build_native_split_spec
from kumo_core.training.table import \
    generate_train_table as generate_native_train_table
from kumoapi.pquery.validated_predictive_query import ValidatedPredictiveQuery

from io_utils import fsutil, logging
from io_utils.fsutil.api import PathDetail
from kumo.diskgraph._diskgraph_legacy import resolve_split_config
from kumo.graph import Graph
from kumo.pquery.utils import (
    get_lag_timestamps,
    get_rhs_lp_table,
    get_yoy_timestamps,
    table_name,
)

logger = logging.getLogger(__name__)

_LP_TASK_TYPES = {
    'static_link_prediction',
    'temporal_link_prediction',
    'link_prediction',
}


def ttng_persisted_staging_dir(train_table_path: str) -> str:
    """Stable gen→ingest handoff bundle persisted beside the train table."""
    return f'{train_table_path}.ttng_staging'


@contextmanager
def local_ttng_builder_input(path: str, leaf_name: str) -> Iterator[str]:
    """Give local-filesystem-only DG builders a local input path."""
    with fsutil.TemporaryDirectory() as local_scratch:
        if fsutil.is_local(path):
            yield path
        else:
            local_path = osp.join(local_scratch, leaf_name)
            fsutil.copy_tree_threaded(path, local_path)
            yield local_path


# The durable gen→ingest bundle's canonical files (DG contract; see
# ``tt_executor.rs``). ``tt_ingest_spec.json`` records the authoritative
# ``tt_id`` and the per-stage seed part files the ingest consumes.
TT_INGEST_SPEC_FILENAME = 'tt_ingest_spec.json'
TT_PARQUET_FILENAME = 'tt.parquet'
_BUNDLE_POINTER_FILENAME = '_current.json'
_IMMUTABLE_BUNDLES_DIRNAME = '_bundles'
_BUNDLE_RESTORE_SOURCE_FILENAME = '_restored_from.json'
_TT_INGEST_SPEC_V3 = 'diskgraph.tt_ingest_spec.v3'
_STAGE_ARTIFACT_INVENTORY_V1 = \
    'diskgraph.tt_stage_artifact_inventory.v1'


def ttng_bundle_dir_for_train_table(train_table_path: str) -> str:
    """Return the native bundle that owns a physical train-table path.

    Every native train-table consumer receives the selected ``tt.parquet``
    artifact path. Publication handles and customer source paths remain
    internal to their producers.
    """
    normalized = train_table_path.rstrip('/')
    if osp.basename(normalized) != TT_PARQUET_FILENAME:
        raise TTNGBundleError(
            'TT-NG: train_table_path must select the physical '
            f'{TT_PARQUET_FILENAME!r} artifact; got {train_table_path!r}.')
    return osp.dirname(normalized)


def _assert_local_bundle_root(staging_dir: str) -> None:
    """Reject local roots whose type could redirect final validation."""
    if not fsutil.is_local(staging_dir):
        return
    try:
        root_stat = os.lstat(staging_dir)
    except FileNotFoundError as error:
        raise TTNGBundleError(
            f'TT-NG: durable staging bundle is missing at '
            f'{staging_dir!r}.') from error
    except OSError as error:
        raise TTNGBundleError(
            f'TT-NG: staging root {staging_dir!r} cannot be inspected.') \
            from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise TTNGBundleError(
            f'TT-NG: staging root {staging_dir!r} must be a non-symlink '
            'directory.')


def _assert_local_bundle_path(staging_dir: str, relative_path: str,
                              *, leaf_is_file: bool) -> None:
    """Require a typed leaf beneath only real directory components."""
    if not fsutil.is_local(staging_dir):
        # Object-store entries have no symlink or POSIX special-file type. The
        # detailed bundle listing already distinguishes prefixes from objects.
        return
    current = staging_dir
    components = relative_path.split(osp.sep)
    for index, component in enumerate(components):
        current = osp.join(current, component)
        try:
            path_stat = os.lstat(current)
        except OSError as error:
            raise TTNGBundleError(
                f'TT-NG: bundle artifact {relative_path!r} cannot be '
                'inspected.') from error
        is_leaf = index == len(components) - 1
        if stat.S_ISLNK(path_stat.st_mode):
            raise TTNGBundleError(
                f'TT-NG: bundle artifact {relative_path!r} traverses a '
                'symlink.')
        expected_type = (
            stat.S_ISREG(path_stat.st_mode)
            if is_leaf and leaf_is_file else stat.S_ISDIR(path_stat.st_mode))
        if not expected_type:
            kind = 'regular file' if is_leaf and leaf_is_file else 'directory'
            raise TTNGBundleError(
                f'TT-NG: bundle artifact {relative_path!r} requires a '
                f'{kind} at {current!r}.')


def _assert_local_bundle_file(staging_dir: str, relative_path: str) -> None:
    _assert_local_bundle_path(
        staging_dir, relative_path, leaf_is_file=True)


def _assert_local_bundle_directory(staging_dir: str,
                                   relative_path: str) -> None:
    _assert_local_bundle_path(
        staging_dir, relative_path, leaf_is_file=False)


def _load_durable_ttng_pointer(staging_dir: str) -> Optional[str]:
    pointer_path = osp.join(staging_dir, _BUNDLE_POINTER_FILENAME)
    if not fsutil.exists(pointer_path):
        return None
    try:
        pointer = json.loads(fsutil.read_text(pointer_path))
    except Exception as error:
        raise TTNGBundleError(
            f'TT-NG: durable bundle pointer at {pointer_path!r} is invalid.'
        ) from error
    bundle_id = pointer.get('bundle_id') if isinstance(pointer, dict) else None
    if (not isinstance(bundle_id, str) or not bundle_id
            or bundle_id in {'.', '..'}
            or osp.basename(bundle_id) != bundle_id):
        raise TTNGBundleError(
            f'TT-NG: durable bundle pointer at {pointer_path!r} has an '
            'invalid bundle_id.')

    return bundle_id


def resolve_durable_ttng_bundle(staging_dir: str) -> str:
    """Resolve a publication root to its selected immutable bundle.

    Producer-local bundles are direct directories. Durable bundles publish
    immutable attempt-specific directories and switch a small pointer only
    after validating the copy.
    """
    bundle_id = _load_durable_ttng_pointer(staging_dir)
    if bundle_id is None:
        return staging_dir
    return osp.join(staging_dir, _IMMUTABLE_BUNDLES_DIRNAME, bundle_id)


def read_tt_id_from_staging_bundle(staging_dir: str) -> str:
    """Return the canonical ``tt_id`` recorded in a durable staging bundle's
    ``tt_ingest_spec.json``.

    This is the single, symmetric way train and predict source the ``tt_id``:
    the DG gen phase (``execute_tt_gen``) stamps the exact ``tt_id`` it was
    invoked with into the spec (train: the gen-traintable job id; predict:
    ``pred_<gen_job_id>``), so consumers read it back from the bundle instead of
    re-parsing a path or re-deriving it from a job id. The value is identical to
    the id gen published.
    """
    staging_dir = resolve_durable_ttng_bundle(staging_dir)
    spec_path = osp.join(staging_dir, TT_INGEST_SPEC_FILENAME)
    if not fsutil.exists(spec_path):
        raise TTNGBundleError(
            f'TT-NG: durable staging bundle at {staging_dir!r} is missing '
            f'{TT_INGEST_SPEC_FILENAME}; cannot resolve tt_id. The gen job '
            'must persist the bundle before consumers run.')
    spec = json.loads(fsutil.read_binary(spec_path))
    if not isinstance(spec, dict):
        raise TTNGBundleError(
            f'TT-NG: ingest spec at {spec_path!r} must be a JSON object.')
    if spec.get('schema_version') != _TT_INGEST_SPEC_V3:
        raise TTNGBundleError(
            f'TT-NG: ingest spec at {spec_path!r} must use '
            f'{_TT_INGEST_SPEC_V3!r}.')
    tt_id = spec.get('tt_id')
    if not isinstance(tt_id, str) or not tt_id:
        raise TTNGBundleError(
            f'TT-NG: {spec_path!r} has no usable "tt_id" field '
            f'(got {tt_id!r}).')
    return tt_id


def _bundle_relative_file_inventory(staging_dir: str) -> set[str]:
    """List bundle files with one backend listing per directory.

    In particular, do not call ``fsutil.exists`` once per manifest part for a
    remote bundle.  A training table can contain tens of thousands of seed
    parts, while those parts normally occupy only one directory per split.
    """
    if fsutil.is_local(staging_dir):
        return {
            osp.relpath(osp.join(directory, filename), staging_dir)
            for directory, _subdirs, filenames in os.walk(staging_dir)
            for filename in filenames
        }

    root = staging_dir.rstrip('/')
    root_prefix = f'{root}/'
    pending_dirs = [root]
    visited_dirs: set[str] = set()
    relative_files: set[str] = set()

    while pending_dirs:
        directory = pending_dirs.pop()
        if directory in visited_dirs:
            continue
        visited_dirs.add(directory)
        try:
            entries = fsutil.listdir(directory, detail=True)
        except NotImplementedError:
            # A backend whose listings do not expose entry types falls back
            # here. Others stay on the bounded detailed-listing path.
            entries = [
                PathDetail(path=path, is_directory=fsutil.isdir(path))
                for path in fsutil.listdir(directory)
            ]
        for entry in entries:
            if entry.is_directory:
                pending_dirs.append(entry.path.rstrip('/'))
                continue
            if not entry.path.startswith(root_prefix):
                raise TTNGBundleError(
                    f'TT-NG: bundle listing returned path {entry.path!r} '
                    f'outside {staging_dir!r}.')
            relative_files.add(entry.path[len(root_prefix):])

    return relative_files


def assert_ttng_bundle_complete(
    staging_dir: str,
) -> None:
    """Assert a durable TT-NG staging bundle is complete before it is consumed.

    Validation is bounded by directory entries plus Parquet footer/schema and
    row-group metadata; large Parquet data pages are never scanned or hashed.
    """
    _assert_ttng_bundle_complete(staging_dir, validate_manifest=True)


def _assert_ttng_bundle_complete(
    staging_dir: str,
    *,
    validate_manifest: bool,
) -> None:
    """Internal bundle gate used by publication to avoid duplicate checks.

    Requires ``tt_ingest_spec.json`` and the ``tt.parquet`` artifact, and --
    reading the per-stage part manifest recorded by gen in the spec's
    ``split_stages`` -- verifies every declared part file exists. This fails a
    half-written bundle AND stays correct if a future stage rolls over into
    ``part-*.parquet`` files (each stage's ``parts`` list is honored verbatim).

    Raises ``TTNGBundleError`` (caller wraps into the domain error) on any gap.
    """
    unresolved_staging_dir = staging_dir
    _assert_local_bundle_root(unresolved_staging_dir)
    staging_dir = resolve_durable_ttng_bundle(unresolved_staging_dir)
    if (staging_dir != unresolved_staging_dir
            and fsutil.is_local(unresolved_staging_dir)):
        _assert_local_bundle_directory(
            unresolved_staging_dir,
            osp.relpath(staging_dir, unresolved_staging_dir),
        )
    _assert_local_bundle_root(staging_dir)
    spec_path = osp.join(staging_dir, TT_INGEST_SPEC_FILENAME)
    relative_files = _bundle_relative_file_inventory(staging_dir)
    required_top_level = (TT_INGEST_SPEC_FILENAME, 'manifest.json')
    missing = [
        name for name in required_top_level if name not in relative_files
    ]
    if missing:
        raise TTNGBundleError(
            f'TT-NG: durable staging bundle is missing {missing} at '
            f'{staging_dir!r}.')
    _assert_local_bundle_file(staging_dir, TT_INGEST_SPEC_FILENAME)
    spec = json.loads(fsutil.read_binary(spec_path))
    if not isinstance(spec, dict):
        raise TTNGBundleError(
            f'TT-NG: ingest spec at {spec_path!r} must be a JSON object.')
    spec_version = spec.get('schema_version')
    if spec_version != _TT_INGEST_SPEC_V3:
        raise TTNGBundleError(
            f'TT-NG: durable staging bundle has unsupported ingest spec '
            f'version {spec_version!r}.')

    train_table_path = osp.join(staging_dir, TT_PARQUET_FILENAME)
    if not fsutil.isdir(train_table_path):
        raise TTNGBundleError(
            f'TT-NG: v3 durable staging bundle requires a '
            f'{TT_PARQUET_FILENAME!r} dataset directory at '
            f'{staging_dir!r}.')
    _assert_local_bundle_directory(staging_dir, TT_PARQUET_FILENAME)

    manifest_path = osp.join(staging_dir, 'manifest.json')
    if 'manifest.json' not in relative_files:
        raise TTNGBundleError(
            f'TT-NG: {spec_version} durable staging bundle at '
            f'{staging_dir!r} is '
            'missing manifest.json.')
    _assert_local_bundle_file(staging_dir, 'manifest.json')
    if validate_manifest:
        try:
            _validate_ttng_bundle_for_publish(staging_dir)
        except Exception as error:
            raise TTNGBundleError(
                f'TT-NG: artifact manifest at {manifest_path!r} is invalid.'
            ) from error
    # ``split_stages`` is None for the no-split (predict) path -- the
    # ``tt.parquet`` artifact (already checked) is the whole bundle. For the
    # split path each stage declares relative ``parts`` under the staging dir.
    stages = spec.get('split_stages') or []
    if not isinstance(stages, list) or any(
            not isinstance(stage, dict) for stage in stages):
        raise TTNGBundleError(
            f'TT-NG: split_stages in {spec_path!r} must be a list of '
            'objects.')
    declared_parts = []
    for stage in stages:
        parts = stage.get('parts') or []
        if not isinstance(parts, list):
            raise TTNGBundleError(
                'TT-NG: split stage parts must be a list.')
        for part in parts:
            normalized = osp.normpath(part) if isinstance(part, str) else None
            if (not isinstance(part, str) or not part or osp.isabs(part)
                    or normalized != part or part == '..'
                    or part.startswith(f'..{osp.sep}')):
                raise TTNGBundleError(
                    f'TT-NG: split stage has unsafe part path {part!r}.')
        declared_parts.extend(parts)
    if len(declared_parts) != len(set(declared_parts)):
        raise TTNGBundleError(
            'TT-NG: split stages contain duplicate part paths.')
    missing_parts = [part for part in declared_parts
                     if osp.normpath(part) not in relative_files]
    if missing_parts:
        raise TTNGBundleError(
            f'TT-NG: durable staging bundle at {staging_dir!r} is missing '
            f'declared stage part file(s) {missing_parts}; the gen→ingest '
            'handoff did not fully persist.')
    for part in declared_parts:
        _assert_local_bundle_file(staging_dir, part)

    from kumo.diskgraph.diskgraph_native_tt_artifacts import (
        _inspect_parquet_artifact,
    )

    for stage in stages:
        stage_key = stage.get('key')
        parts = stage.get('parts')
        row_count = stage.get('row_count')
        if (not isinstance(stage_key, str) or not stage_key
                or not isinstance(parts, list) or not parts
                or isinstance(row_count, bool)
                or not isinstance(row_count, int) or row_count < 0):
            raise TTNGBundleError(
                f'TT-NG: split stage declaration is invalid: {stage!r}.')
        inventory = stage.get('artifact_inventory')
        if not isinstance(inventory, dict):
            raise TTNGBundleError(
                f'TT-NG: split stage {stage_key!r} is missing its '
                'artifact inventory.')
        if (inventory.get('schema_version') !=
                _STAGE_ARTIFACT_INVENTORY_V1
                or not isinstance(inventory.get('parts'), list)
                or len(inventory['parts']) != len(parts)):
            raise TTNGBundleError(
                f'TT-NG: split stage {stage_key!r} has an invalid '
                'artifact inventory.')
        declared_artifacts = inventory['parts']

        observed_rows = 0
        for part_index, part in enumerate(parts):
            artifact_path = osp.join(staging_dir, part)
            try:
                actual_size, actual_rows = _inspect_parquet_artifact(
                    artifact_path)
            except Exception as error:
                raise TTNGBundleError(
                    f'TT-NG: split stage {stage_key!r} part {part!r} has '
                    'invalid parquet metadata.') from error
            declared = declared_artifacts[part_index]
            if (not isinstance(declared, dict)
                    or declared.get('path') != part
                    or isinstance(declared.get('size_bytes'), bool)
                    or not isinstance(declared.get('size_bytes'), int)
                    or declared['size_bytes'] <= 0
                    or isinstance(declared.get('row_count'), bool)
                    or not isinstance(declared.get('row_count'), int)
                    or declared['row_count'] < 0):
                raise TTNGBundleError(
                    f'TT-NG: split stage {stage_key!r} has invalid '
                    f'metadata for part {part!r}.')
            if actual_size != declared['size_bytes']:
                raise TTNGBundleError(
                    f'TT-NG: split stage {stage_key!r} part {part!r} '
                    'size mismatch.')
            if actual_rows != declared['row_count']:
                raise TTNGBundleError(
                    f'TT-NG: split stage {stage_key!r} part {part!r} '
                    'row_count mismatch.')
            observed_rows += actual_rows
        if observed_rows != row_count:
            raise TTNGBundleError(
                f'TT-NG: split stage {stage_key!r} row_count mismatch: '
                f'expected={row_count} actual={observed_rows}.')


def restore_durable_ttng_bundle(
    durable_staging_dir: str,
    local_staging_dir: str,
) -> str:
    """Restore a stable-root or already-selected immutable bundle."""
    durable_bundle = resolve_durable_ttng_bundle(durable_staging_dir)
    try:
        assert_ttng_bundle_complete(local_staging_dir)
        restored_from = json.loads(
            fsutil.read_text(
                osp.join(local_staging_dir, _BUNDLE_RESTORE_SOURCE_FILENAME)))
        if restored_from == {'source': durable_bundle}:
            return local_staging_dir
    except Exception:
        # Missing/malformed source marker means this cache predates the
        # versioned restore contract and must be refreshed.
        pass
    if osp.abspath(durable_bundle) == osp.abspath(local_staging_dir):
        assert_ttng_bundle_complete(durable_bundle)
        return durable_bundle
    if fsutil.exists(local_staging_dir):
        fsutil.rm(local_staging_dir, ignore_errors=True)

    assert_ttng_bundle_complete(durable_bundle)
    try:
        fsutil.copy_tree_threaded(durable_bundle, local_staging_dir)
        assert_ttng_bundle_complete(local_staging_dir)
        fsutil.write_text(
            osp.join(local_staging_dir, _BUNDLE_RESTORE_SOURCE_FILENAME),
            json.dumps({'source': durable_bundle}, sort_keys=True),
        )
    except BaseException:
        # A retry must never mistake a partial copy for a usable bundle.
        fsutil.rm(local_staging_dir, ignore_errors=True)
        raise
    return local_staging_dir


def _validate_ttng_bundle_for_publish(
    staging_dir: str,
    *,
    expected_tt_id: Optional[str] = None,
) -> tuple[str, int]:
    """Validate a supported artifact contract and bind it to one TT attempt."""
    from kumo.diskgraph.diskgraph_native_tt_artifacts import (
        artifact_file_size,
        validate_artifact_bundle,
    )

    staging_dir = resolve_durable_ttng_bundle(staging_dir)
    spec_path = osp.join(staging_dir, TT_INGEST_SPEC_FILENAME)
    manifest_path = osp.join(staging_dir, 'manifest.json')
    spec = json.loads(fsutil.read_text(spec_path))
    tt_id = spec.get('tt_id') if isinstance(spec, dict) else None
    if not isinstance(tt_id, str) or not tt_id:
        raise TTNGBundleError(
            f'TT-NG: {spec_path!r} has no usable tt_id for publication.')
    if expected_tt_id is not None and tt_id != expected_tt_id:
        raise TTNGBundleError(
            f'TT-NG: bundle attempt mismatch: expected {expected_tt_id!r}, '
            f'got {tt_id!r}.')

    try:
        manifest = validate_artifact_bundle(manifest_path)
    except Exception as error:
        raise TTNGBundleError(
            f'TT-NG: artifact manifest at {manifest_path!r} is invalid.'
        ) from error

    train_table = manifest['train_table']
    expected_path = osp.join(staging_dir, TT_PARQUET_FILENAME)
    if osp.normpath(train_table['path']) != osp.normpath(expected_path):
        raise TTNGBundleError(
            'TT-NG: manifest train_table.path is not bound to its bundle: '
            f'{train_table["path"]!r}.')
    size_bytes = int(train_table['size_bytes'])
    observed_size = sum(
        artifact_file_size(part['path'])
        for part in train_table['parts'])
    if observed_size != size_bytes:
        raise TTNGBundleError(
            'TT-NG: bundle train-table size changed after validation.')
    return tt_id, size_bytes


def _write_durable_ttng_pointer(durable_dst: str, bundle_id: str) -> None:
    pointer_path = osp.join(durable_dst, _BUNDLE_POINTER_FILENAME)
    pointer_json = json.dumps({'bundle_id': bundle_id}, sort_keys=True)
    if fsutil.is_local(pointer_path):
        fsutil.makedirs(durable_dst, exist_ok=True)
        temporary_pointer = osp.join(
            durable_dst, f'.{_BUNDLE_POINTER_FILENAME}.{bundle_id}.tmp')
        fsutil.write_text(temporary_pointer, pointer_json)
        os.replace(temporary_pointer, pointer_path)
    elif fsutil.is_adls(pointer_path):
        fsutil.atomic_write_text(pointer_path, pointer_json)
    else:
        # S3/GCS/UC object puts become visible as one complete object.
        fsutil.write_text(pointer_path, pointer_json)


def persist_ttng_bundle(
    ephemeral_src: str,
    durable_dst: str,
    *,
    expected_tt_id: Optional[str] = None,
) -> str:
    """Copy one attempt to immutable storage and return its selected bundle."""
    _assert_ttng_bundle_complete(ephemeral_src, validate_manifest=False)
    if expected_tt_id is None:
        source_basename = osp.basename(ephemeral_src.rstrip('/'))
        if not (source_basename.startswith('.')
                and source_basename.endswith('.staging')):
            raise TTNGBundleError(
                'TT-NG: producer bundle name is not attempt-scoped: '
                f'{source_basename!r}.')
        expected_tt_id = source_basename[1:-len('.staging')]
    elif not expected_tt_id:
        raise TTNGBundleError('TT-NG: expected bundle tt_id is empty.')
    source_tt_id, source_size = _validate_ttng_bundle_for_publish(
        ephemeral_src,
        expected_tt_id=expected_tt_id,
    )
    bundle_id = uuid.uuid4().hex
    bundle_dir = osp.join(durable_dst, _IMMUTABLE_BUNDLES_DIRNAME, bundle_id)
    fsutil.copy_tree_threaded(ephemeral_src, bundle_dir)
    _assert_ttng_bundle_complete(bundle_dir, validate_manifest=False)
    destination_tt_id, destination_size = _validate_ttng_bundle_for_publish(
        bundle_dir,
        expected_tt_id=source_tt_id,
    )
    if destination_tt_id != source_tt_id or destination_size != source_size:
        raise TTNGBundleError(
            'TT-NG: durable copy does not match the source bundle binding.')
    return bundle_dir


def persist_durable_ttng_bundle(
    ephemeral_src: str,
    durable_dst: str,
    *,
    expected_tt_id: Optional[str] = None,
) -> str:
    """Publish a selected bundle for callers that only retain a stable root."""
    bundle_dir = persist_ttng_bundle(
        ephemeral_src,
        durable_dst,
        expected_tt_id=expected_tt_id,
    )
    _write_durable_ttng_pointer(durable_dst, osp.basename(bundle_dir))
    return bundle_dir


class TTNGBundleError(RuntimeError):
    """A durable TT-NG staging bundle is missing or incomplete.

    Raised by the shared bundle helpers so each caller can wrap it in its own
    domain error (``TrainingDataMaterializationError`` /
    ``PredictionDataMaterializationError``) with the appropriate retry policy.
    """


def _apply_lp_resolution_plan(
    plan_json: str,
    *,
    list_distinct_target_is_array: bool,
    is_link_pred: bool,
    entity_table: Optional[str],
    lp_rhs_entity_table: Optional[str],
    is_static_lp: bool = False,
    lp_rhs_is_temporal: bool = False,
) -> str:
    """Declare the FK-free LP resolution plan (``plan['lp_resolution']``).

    A list of ``RefResolveSpec`` naming the PK columns tt-ingest resolves to
    node-ids as rowpack columns (no FK edge, no leak; positives are labels):
    ENTITY1 -> entity_node_id, TARGET1 -> target_node_ids. The dict must
    round-trip into the Rust ``RefResolveSpec`` (snake_case). Non-LP unchanged.

    For static LP on TT-NG, also resolve gen's ``exclude_target_keys`` key-list
    (prior-stage RHS keys on val/test seed rows) into ``exclude_node_ids`` so the
    engine can hydrate the past-visit exclude set. Resolved against the SAME RHS
    entity table as the positives.
    """
    if not list_distinct_target_is_array or not is_link_pred:
        return plan_json
    return apply_reference_resolution(
        plan_json,
        entity_node_type=entity_table,
        rhs_node_type=lp_rhs_entity_table,
        resolve_link_targets=True,
        is_static_link=is_static_lp,
        rhs_is_temporal=lp_rhs_is_temporal,
    )


def _time_col_name_from_plan_json(plan_json: str) -> str:
    try:
        plan = json.loads(plan_json)
    except Exception:
        return 'TIMESTAMP'
    if not isinstance(plan, dict):
        return 'TIMESTAMP'
    entity = plan.get('entity')
    if not isinstance(entity, dict):
        return 'TIMESTAMP'
    time_col_name = entity.get('output_time_col_name')
    if isinstance(time_col_name, str) and time_col_name:
        return time_col_name
    return 'TIMESTAMP'


def _query_year_over_year_enabled(query: ValidatedPredictiveQuery) -> bool:
    value = getattr(query, 'year_over_year', False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False


def _ttng_forecasting_offset_kwargs(
    query: ValidatedPredictiveQuery,
    forecast_length: int,
    lag_timesteps: Optional[int],
    year_over_year: bool,
) -> dict[str, Any]:
    pt = pd.Timestamp(0, unit='us')
    time_range = query.get_combined_date_offset_range()
    step = time_range.timeframe
    assert step is not None

    future_offsets_secs = [
        int(((pt + i * step) - pt).total_seconds())
        for i in range(forecast_length)
    ]
    lag_offsets_secs = None
    if lag_timesteps is not None:
        lag_ts = get_lag_timestamps(pt, lag_timesteps, step)
        lag_offsets_secs = [int((t - pt).total_seconds()) for t in lag_ts]

    yoy_offsets_secs = None
    if year_over_year:
        yoy_ts, _ = get_yoy_timestamps(pt, forecast_length, step)
        yoy_offsets_secs = [int((t - pt).total_seconds()) for t in yoy_ts]

    return {
        'future_offsets_secs': future_offsets_secs,
        'lag_offsets_secs': lag_offsets_secs,
        'yoy_offsets_secs': yoy_offsets_secs,
    }


def _build_split_spec(
    split_config: Any,
    split_family_hint: Optional[str],
    split_spec_cls: Any,
) -> Optional[Any]:
    split_config, split_family_hint = resolve_split_config(
        split_config, split_family_hint)
    return build_native_split_spec(
        split_config,
        split_family_hint,
        split_spec_cls,
    )


def generate_train_table_via_diskgraph(
    query: ValidatedPredictiveQuery,
    *,
    graph: Graph = None,
    timestamps: List[int] = None,
    task_type: str,
    forecast_length: int,
    lag_timesteps: Optional[int],
    list_distinct_target_is_array: bool,
    include_frequency: bool,
    entity_time_col: Optional[str],
    index_path: str,
    output_path: str,
    split_config: Any,
    split_family: Optional[str] = None,
    downsampling: Optional[dict[str, Any]] = None,
    train_table_edges_config: Optional[dict[str, Any]] = None,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
) -> Tuple[str, str]:
    """Compile and execute a diskgraph TT plan.

    Returns:
        Tuple[train_table_path, manifest_path]
    """
    try:
        from diskgraph import (
            DiskGraphPlanCompiler,
            SplitSpec,
            execute_tt_plan_with_artifacts,
        )
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            'Failed to import diskgraph runtime for train-table generation.'
        ) from e

    if split_config is None:
        raise RuntimeError(
            'diskgraph strict artifact path requires split_config')

    if not index_path:
        raise RuntimeError(
            'diskgraph train-table generation requires diskgraph index path')
    if not fsutil.exists(index_path):
        raise RuntimeError(f'diskgraph index path does not exist: {index_path}')

    compiler = DiskGraphPlanCompiler(index_path=index_path)
    split_spec = _build_split_spec(split_config, split_family, SplitSpec)
    if split_spec is None:
        raise RuntimeError(
            'diskgraph strict artifact path requires a valid split_config with split family'
        )

    # Resolve the RHS entity table so the executor can orphan/temporal-filter
    # ListDistinct targets (same helper as the BP flow).
    lp_rhs_entity_table: Optional[str] = None
    if graph is not None and task_type.lower() in _LP_TASK_TYPES:
        lp_rhs_entity_table = get_rhs_lp_table(query, graph, no_connector=True)
        logger.info("[DG] Orphan filter: lp_rhs_entity_table=%s",
                    lp_rhs_entity_table)

    # Forecasting needs explicit lag/yoy offsets passed to the compiler.
    lag_offsets_secs = None
    yoy_offsets_secs = None
    if task_type.lower() == 'forecasting':
        # Only relative offsets matter, so a dummy anchor suffices.
        pt = pd.Timestamp(0, unit='us')
        time_range = query.get_combined_date_offset_range()
        step = time_range.timeframe
        assert step is not None

        if lag_timesteps is not None:
            lag_ts = get_lag_timestamps(pt, lag_timesteps, step)
            lag_offsets_secs = [int((t - pt).total_seconds()) for t in lag_ts]

        if _query_year_over_year_enabled(query):
            yoy_ts, _ = get_yoy_timestamps(pt, forecast_length, step)
            yoy_offsets_secs = [int((t - pt).total_seconds()) for t in yoy_ts]

    plan_json = compiler.compile(
        query=query,
        timestamps=timestamps,
        task_type=task_type,
        forecast_length=forecast_length,
        lag_timesteps=lag_timesteps,
        list_distinct_target_is_array=list_distinct_target_is_array,
        include_frequency=include_frequency,
        entity_time_col=entity_time_col,
        input_timestamp_unit=input_timestamp_unit,
        target_stype=target_stype,
        split=split_spec,
        lp_rhs_entity_table=lp_rhs_entity_table,
        lag_offsets_secs=lag_offsets_secs,
        yoy_offsets_secs=yoy_offsets_secs,
    )
    time_col_name = _time_col_name_from_plan_json(plan_json)

    execute_kwargs = dict(
        plan_json=plan_json,
        index_path=index_path,
        output_path=output_path,
        split_config=split_spec,
        split_family_hint=split_spec.kind,
        downsampling=downsampling,
        split_stats_config={
            'entity_col': 'ENTITY',
            'target_col': 'TARGET',
            'time_col': time_col_name,
            'task_type': task_type,
            'list_distinct_target_is_array':
            bool(list_distinct_target_is_array),
        },
        train_table_edges_config=train_table_edges_config,
    )

    producer_result = execute_tt_plan_with_artifacts(**execute_kwargs)

    manifest_path = producer_result['manifest_path']

    logger.info(
        "Diskgraph producer materialized train-table bundle (rows=%s, output=%s, manifest=%s, split_family=%s, authoritative=%s, split_counts=%s)",
        producer_result.get('row_count'),
        output_path,
        manifest_path,
        (producer_result.get('split_meta') or {}).get('family'),
        (producer_result.get('split_meta') or {}).get('authoritative'),
        producer_result.get('split_counts'),
    )
    return output_path, manifest_path


def generate_train_table_via_diskgraph_ttng(
    query: ValidatedPredictiveQuery,
    *,
    graph: Graph = None,
    timestamps: List[int] = None,
    task_type: str,
    forecast_length: int,
    lag_timesteps: Optional[int],
    year_over_year: bool = False,
    list_distinct_target_is_array: bool,
    include_frequency: bool,
    entity_time_col: Optional[str],
    index_path: str,
    output_path: str,
    tt_id: str,
    split_config: Any,
    split_family: Optional[str] = None,
    downsampling: Optional[dict[str, Any]] = None,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
) -> Tuple[str, str]:
    """TT-NG sibling of :func:`generate_train_table_via_diskgraph`.

    Runs the ttgen-only phase via ``execute_tt_gen``: writes rows + split +
    col_stats to a staging dir, builds no edges and publishes no extension
    (``materialize`` later calls ``execute_tt_ingest`` on the staging parquet).
    TT-NG is the only backend.

    Returns:
        Tuple[staging_tt_parquet_path, staging_manifest_path]
    """
    is_link_pred = task_type.lower() in _LP_TASK_TYPES
    if not index_path:
        raise RuntimeError(
            'diskgraph train-table generation requires diskgraph index path')
    if not fsutil.exists(index_path):
        raise RuntimeError(f'diskgraph index path does not exist: {index_path}')
    lp_rhs_entity_table: Optional[str] = None
    lp_rhs_is_temporal = False
    if graph is not None and is_link_pred:
        lp_rhs_entity_table = get_rhs_lp_table(query, graph, no_connector=True)
        # Data-condition for single-timestamp grouping (mirrors legacy's
        # TimestampSampler RHS check): the RHS/target table has a create-time or
        # end-time column. Computed kumo-side here (single source of truth) and
        # passed into the plan as ``lp_rhs_is_temporal`` so the DG INGEST gate
        # (tt_executor.rs) only builds ``bucket_index`` when grouping can be
        # used. The LOAD gate (diskgraph_lightning_data.py ``_rhs_is_temporal``)
        # re-derives the same predicate from ``index_metadata.json``; see
        # SMELL-2 in TS_GROUPING_NOTES.md for why these two derivations stay
        # consistent and fail loud rather than silently mis-batch.
        try:
            rhs_table = graph[lp_rhs_entity_table]
            lp_rhs_is_temporal = bool(rhs_table.has_time_col()
                                      or rhs_table.has_end_time_col())
        except Exception:
            lp_rhs_is_temporal = False
        logger.info(
            "[DG] Orphan filter: lp_rhs_entity_table=%s rhs_is_temporal=%s",
            lp_rhs_entity_table, lp_rhs_is_temporal)
    resolved_split_config, resolved_split_family = resolve_split_config(
        split_config, split_family)
    entity_node_type = table_name(query.entity_column)
    if graph is None:
        entity_primary_key = ""
    else:
        entity_primary_key = graph[entity_node_type].pkey_name or ""
    result = generate_native_train_table(
        TrainTableRequest(
            query=query,
            timestamps=tuple(timestamps or ()),
            task_type=task_type,
            forecast_length=forecast_length,
            lag_timesteps=lag_timesteps,
            list_distinct_target_is_array=list_distinct_target_is_array,
            include_frequency=include_frequency,
            entity_time_col=entity_time_col,
            index_path=index_path,
            output_path=output_path,
            tt_id=tt_id,
            split_config=resolved_split_config,
            split_family=resolved_split_family,
            input_timestamp_unit=input_timestamp_unit,
            target_stype=target_stype,
            entity_node_type=entity_node_type,
            entity_primary_key=entity_primary_key,
            rhs_node_type=lp_rhs_entity_table,
            rhs_is_temporal=lp_rhs_is_temporal,
            downsampling=downsampling,
            compiler_options=(_ttng_forecasting_offset_kwargs(
                query, forecast_length, lag_timesteps, year_over_year)
                              if task_type.lower() == 'forecasting' else None),
            index_path_verified=True,
        ))
    return result.train_table_path, result.manifest_path


def refresh_split_artifact_from_initialized_split(
    train_table_path: str,
    split_obj: Any,
) -> bool:
    """No-op for DG producer-authoritative split pipeline."""
    logger.info(
        'Skipping split artifact refresh: producer-authoritative split payload mode '
        '(train_table_path=%s, split=%s)',
        train_table_path,
        split_obj.__class__.__name__ if split_obj is not None else None,
    )
    return False
