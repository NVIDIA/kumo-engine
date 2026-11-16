# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Optional

from common_utils.kumo_timing import kumo_timing
from io_utils import fsutil, logging

logger = logging.getLogger(__name__)

# Optional hard-fail threshold for orphan-FK skips at TT-NG stage ingest, in
# percent of seed rows (e.g. ``50``; ``0`` fails on ANY skip). Unset/empty →
# warn-only (the default). See ``check_ingest_orphan_skips``.
_ORPHAN_FAIL_PCT_ENV = 'KUMO_TTNG_ORPHAN_FAIL_PCT'

# Skipped fraction at/above which the warning escalates to "this train table
# most likely was not generated against this graph".
_ORPHAN_ESCALATE_PCT = 50.0


class TTNGOrphanSkipError(RuntimeError):
    """Orphan-FK skips at TT-NG stage ingest reached the hard-fail threshold
    (``KUMO_TTNG_ORPHAN_FAIL_PCT``). Callers wrap it with the skip message
    verbatim — unlike other ingest ``RuntimeError``s, this is not a
    missing-bundle problem."""


def predict_staging_tt_id(job_id: str) -> str:
    """Deterministic predict-staging ``tt_id`` shared by the generator and
    materialize activities."""
    return f'pred_{job_id}'


def ttng_staging_dir(index_path: str, tt_id: str) -> str:
    """Path to the TT-NG staging dir for ``tt_id``: ``<index>/train_tables/.<tt_id>.staging/``."""
    return osp.join(index_path, 'train_tables', f'.{tt_id}.staging')


def ttng_persisted_pred_staging_dir(pred_table_path: str) -> str:
    """Durable gen→ingest predict staging bundle, beside the pred table.

    Predict analog of ``diskgraph_train_table.ttng_persisted_staging_dir``; the
    location-independent bundle the gen job persists so any pod can re-ingest
    after delete_snapshot. ``pred_table_path`` == the gen job's ``output_root``.
    See ttng-predict-materialization-distributed-design.md."""
    return f'{pred_table_path.rstrip("/")}.ttng_staging'


def is_tt_extension_complete(extension_path: str) -> bool:
    """Whether a published ``train_tables/<tt_id>/`` extension is complete.

    ``execute_tt_ingest`` publishes the sidecars then writes
    ``diskgraph_manifest.json`` LAST (and it is the artifact consumers read), so
    its presence marks a fully-published extension for both the train and the
    predict stages. An interrupted publish leaves the dir without it.
    """
    return fsutil.exists(osp.join(extension_path, 'diskgraph_manifest.json'))


def check_ingest_orphan_skips(ingest_result: Any, kind: str) -> None:
    """Surface the orphan-FK skip counters of a TT-NG stage ingest.

    ``execute_tt_ingest`` membership-verifies every seed FK against the
    *training* graph and skips (never mis-resolves) references to entities
    that don't exist in it, reporting them under ``key_failures``
    (``membership_failures`` = MPH false positives, ``lookup_failures`` =
    genuine pkmap misses, ``parse_failures`` = values unparsable as the
    training graph's key type — the PK-type-mismatch case). Skips are
    expected to be zero when the seeds were generated against this very
    graph; a non-zero count means the seeds and the graph diverge
    (cross-graph train-table reuse, regenerated snapshot, or a custom train
    table with foreign keys) — the job would silently run on the surviving
    subset.

    ``ingest_result`` accepts either the live ``execute_tt_ingest`` result
    dict or the persisted ``diskgraph_manifest.json`` (same shape).

    Always warns when anything was skipped; escalates the wording at
    ``_ORPHAN_ESCALATE_PCT``. Set ``KUMO_TTNG_ORPHAN_FAIL_PCT`` to a percent
    threshold to raise :class:`TTNGOrphanSkipError` at/above it (the train
    activity layer wraps that into a non-retryable
    ``TrainingDataMaterializationError``). Diagnostics parsing is never
    fatal — only the intentional threshold raise escapes.
    """
    try:
        kf = (ingest_result or {}).get('key_failures') or {}
        skipped = int(kf.get('parse_failures') or 0) + \
            int(kf.get('lookup_failures') or 0) + \
            int(kf.get('membership_failures') or 0)
        if skipped == 0:
            return
        task_dataset = (ingest_result or {}).get('task_dataset') or {}
        # Denominator = PRE-drop seed total: orphans are dropped at encode, so
        # ``input_row_count`` (kept + dropped) keeps the fraction "% of original
        # seeds". Fall back to the stage row_count sum for pre-drop wheels.
        total_rows = int(task_dataset.get('input_row_count') or 0)
        if total_rows == 0:
            stages = task_dataset.get('stages') or {}
            total_rows = sum(
                int((stage or {}).get('row_count') or 0)
                for stage in stages.values())
        by_context = kf.get('by_context')
        samples = kf.get('samples')
    except Exception:  # malformed counters (wheel skew) must not kill the job
        logger.warning(
            'TT-NG %s: could not parse ingest key_failures %r; skipping the '
            'orphan-skip check', kind,
            (ingest_result or {}).get('key_failures'))
        return
    # Skips count FK *references*, not rows — an LP row carries an entity ref
    # plus one ref per target-list element — so the fraction is an upper-bound
    # signal; clamp it for display and thresholds.
    if total_rows:
        pct = min(100.0, 100.0 * skipped / total_rows)
        denom = f'{total_rows} ingested seed row(s)'
    else:
        pct = 100.0
        denom = 'an unknown number of seed rows (no stage row counts)'
    detail = (f'{skipped} seed FK reference(s) across {denom} '
              f'(~{pct:.1f}%) did not resolve to a node of the training '
              f'graph and were skipped (never trained on). '
              f'by_context={by_context} samples={samples}')
    if pct >= _ORPHAN_ESCALATE_PCT:
        message = (
            f'TT-NG {kind}: {detail} Most of the train table does not belong '
            f'to this graph — it was almost certainly generated against a '
            f'different graph or graph version. Re-generate the train table '
            f'against this graph, or bind the training job to the graph it '
            f'was generated from.')
    else:
        message = (
            f'TT-NG {kind}: {detail} These seeds reference entities that do '
            f'not exist in this graph (e.g. the graph was regenerated, or the '
            f'train table is reused across graphs) and are excluded from '
            f'training/prediction.')
    logger.warning('%s', message)
    fail_pct_raw = os.environ.get(_ORPHAN_FAIL_PCT_ENV, '')
    if fail_pct_raw:
        try:
            fail_pct = float(fail_pct_raw)
        except ValueError:
            logger.warning('Ignoring invalid %s=%r (expected a percent)',
                           _ORPHAN_FAIL_PCT_ENV, fail_pct_raw)
            return
        if pct >= fail_pct:
            raise TTNGOrphanSkipError(
                f'{message} (failing: ~{pct:.1f}% skipped >= '
                f'{_ORPHAN_FAIL_PCT_ENV}={fail_pct_raw})')


def check_extension_manifest_orphan_skips(extension_path: str,
                                          kind: str) -> None:
    """Re-run the orphan-skip check from the persisted manifest when a
    present-and-complete extension is REUSED without re-ingest.

    The live check runs only on the job that actually ingests, but the
    extension outlives it: a later job binding the same graph + tt_id (or a
    retry after a hard-fail — the extension publishes before the counters are
    checked) takes the skip-re-ingest path. The manifest persists the same
    ``key_failures`` + stage row counts, so every consumer warns/fails
    consistently, not just the first.
    """
    manifest_path = osp.join(extension_path, 'diskgraph_manifest.json')
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception:  # unreadable manifest → the consumer will fail loudly
        logger.warning(
            'TT-NG %s: could not read %s for the orphan-skip check', kind,
            manifest_path)
        return
    check_ingest_orphan_skips(manifest, f'{kind} (reused extension)')


def ensure_stage_ingested(
    *,
    index_path: str,
    target_tt_id: str,
    staging_dir: str,
    predict_stage_name: Optional[str] = None,
) -> str:
    """Idempotently re-ingest a TT-NG stage from ``staging_dir`` into
    ``<index>/train_tables/<target_tt_id>/`` (no-op if already present).

    Shared self-heal for the train and predict cross-pod hand-off: the per-job
    ``train_tables/<tt_id>/`` extension is NOT part of the graph snapshot, so a
    pod without it re-ingests from the durable staging bundle beside the table.
    ``predict_stage_name='predict'`` ingests the single predict seed stage;
    ``None`` ingests the split train/val/test stages.
    """
    kind = 'predict' if predict_stage_name is not None else 'train'
    extension_path = osp.join(index_path, 'train_tables', target_tt_id)
    if fsutil.exists(extension_path):
        # An interrupted publish can leave the dir without its final
        # ``diskgraph_manifest.json`` (the sentinel ingest writes LAST and the
        # consumer reads); a bare dir-existence check would skip re-ingest and
        # later fail on the incomplete extension. Validate, then drop-and-heal.
        if is_tt_extension_complete(extension_path):
            logger.info('TT-NG %s: extension already present at %s; skipping '
                        're-ingest', kind, extension_path)
            check_extension_manifest_orphan_skips(extension_path, kind)
            return extension_path
        logger.warning('TT-NG %s: extension at %s is incomplete (missing '
                       'diskgraph_manifest.json); deleting and re-ingesting',
                       kind, extension_path)
        fsutil.rm(extension_path, ignore_errors=True)

    from kumo.data.generic.diskgraph_train_table import (
        TTNGBundleError,
        assert_ttng_bundle_complete,
        restore_durable_ttng_bundle,
    )
    try:
        assert_ttng_bundle_complete(staging_dir)
    except TTNGBundleError as e:
        raise RuntimeError(
            f'{e} The ephemeral index was torn down / this pod has no '
            f'extension, and the durable {kind} staging beside the '
            f'{kind} table is missing or incomplete.') from e
    # Copy into this pod's index-local staging dir so the DG publish renames
    # within the index FS.
    local_staging = ttng_staging_dir(index_path, target_tt_id)
    local_staging = restore_durable_ttng_bundle(staging_dir, local_staging)

    from diskgraph import execute_tt_ingest
    with kumo_timing(f'{kind}_materialize.ttng_reingest_stage'):
        ingest_kwargs = dict(
            staging_dir=local_staging,
            index_path=index_path,
            tt_id=target_tt_id,
        )
        if predict_stage_name is not None:
            ingest_kwargs['predict_stage_name'] = predict_stage_name
        ingest_result = execute_tt_ingest(**ingest_kwargs)
    # Fail loud if the wheel publishes the extension somewhere other than the
    # path the manifest / consumer reads.
    published = (ingest_result or {}).get('tt_extension_path')
    if published is not None and (
            osp.abspath(published) != osp.abspath(extension_path)):
        raise RuntimeError(
            f'TT-NG {kind}: execute_tt_ingest published the extension at '
            f'{published!r} but the manifest/consumer reads {extension_path!r}; '
            'the wheel publish convention diverged from the train convention.')
    check_ingest_orphan_skips(ingest_result, kind)
    logger.info('TT-NG %s: re-ingested stage from %s into %s', kind,
                staging_dir, extension_path)
    return extension_path
