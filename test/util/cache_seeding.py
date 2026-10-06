# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-populate the DiskGraph col-stats cache from a pandas DataFrame.

``Table.compute_col_stats`` is a pure read of ``BackendCacheManager``,
and only DG ingest populates that cache in production. Tests that don't
run DG ingest therefore see an empty cache and
:class:`UnsupportedInDiskgraphError`.

:func:`seed_col_stats_cache` is an opt-in helper that mirrors what
``copy_ingest_stats_to_backend_cache`` writes in production, computing
the stats from a pandas DataFrame via the same
``compute_col_stats_python`` / ``compute_table_stats_native`` code paths
DG uses. It writes both the in-memory cache (via
:meth:`BackendCacheManager.add_col_stats` / ``add_table_stats``) and the
on-disk JSON files (via :meth:`save_col_stats_cache` /
:meth:`save_table_stats`) so cache-read paths in
``Table.compute_col_stats`` and ``Table.compute_table_stats`` resolve
without real ingest.

Usage from a test::

    def test_xyz(seed_col_stats):
        df = pd.DataFrame(...)
        seed_col_stats(df, table_name='ACCOUNT', data_version='1234567890',
                       primary_key='ACCOUNT_ID')
        # ... build Graph, call compute_col_stats, etc.

The companion fixture is :func:`test.conftest.seed_col_stats`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from kumo_core.types import ColStatType, TableStatType
from kumo_core.types.schema import (
    BaseColumnConfig,
    PartialTableFingerprint,
    TableFingerprint,
)
from kumoapi.typing import Stype

from kumo.data import get_active_cache, get_persist_dir
from kumo.data.backend_cache import BackendCacheManager
from kumo.stats import Stats, compute_stats_python


def _build_fingerprint(
    *,
    connector_name: str,
    table_name: str,
    df_columns: List[str],
    primary_key: Optional[str],
    time_col: Optional[str],
) -> TableFingerprint:
    """Build the same shape of fingerprint Table.compute_col_stats uses.

    The cache key derived in :meth:`BackendCacheManager._get_col_stats_cache_key`
    only depends on ``connector_name``, ``source_name``, ``pkey``, and
    ``time_col`` — NOT on ``cols`` — so the partial vs. ingested fingerprint
    distinction is irrelevant for cache lookups. We use a partial fingerprint
    here so callers don't have to thread dtypes through.
    """
    time_col_cfg: Optional[BaseColumnConfig] = None
    if time_col is not None:
        time_col_cfg = BaseColumnConfig(name=time_col)
    return PartialTableFingerprint(
        connector_name=connector_name,
        source_name=table_name,
        cols=list(df_columns),
        pkey=primary_key,
        time_col=time_col_cfg,
        end_time_col=None,
    )


def seed_col_stats_cache(
    df: pd.DataFrame,
    *,
    table_name: str,
    data_version: str,
    connector_name: str = 'parquet',
    persist_dir: Optional[str] = None,
    primary_key: Optional[str] = None,
    time_col: Optional[str] = None,
    include_heavy_nlp: bool = False,
) -> None:
    """Pre-populate the DG col-stats cache with synthetic stats from a DataFrame.

    Mirrors what ``copy_ingest_stats_to_backend_cache`` writes in production
    so survivor cache-read paths in :meth:`Table.compute_col_stats` and
    :meth:`Table.compute_table_stats` succeed in tests without running real
    ingest.

    Args:
        df: Source data; one ``Stats`` object is computed per (column,
            stat_type) pair using :func:`compute_col_stats_python`.
        table_name: The ``source_name`` half of the table fingerprint.
            Must match :attr:`kumo.graph.Table.source_name` for the test's
            target table.
        data_version: The data version string the test's connector will
            return from ``get_data_version(table_name)``. For
            ``ParquetConnector`` this is the parquet file's ``mtime`` (as a
            stringified epoch second; see ``FileConnector._get_data_version``).
        connector_name: The ``connector_name`` half of the fingerprint.
            Must match :attr:`kumo.graph.Table.data_source` (i.e. the name
            passed to the connector constructor). Defaults to ``'parquet'``
            because that's what most test fixtures in :mod:`test.conftest`
            use.
        persist_dir: Override the persist dir backing the cache. Defaults
            to the deploy_env-derived :func:`kumo.data.get_persist_dir`.
        primary_key: Name of the primary key column, if any. Must match the
            test's ``Table.pkey.name``.
        time_col: Name of the time column, if any. Must match the test's
            ``Table.time_col.name``.
        include_heavy_nlp: When ``True``, additionally seed the
            Python-computable text/token NLP stats (avg/max tokens, GloVe
            overlap, non-char fraction) that the default seed path omits by
            policy. Required by DG-native tests that assert on TEXT/GloVe
            encoder inference. ``SUMMARY_STATS_BY_CLASS`` is still NOT seeded
            (no per-series Python computation exists, and the DG Rust producer
            does not emit it). ``STRING_GLOVE_OVERLAP`` additionally honors the
            ``KUMO_ENABLE_GLOVE`` env gate.
    """
    if persist_dir is not None and persist_dir != get_persist_dir():
        # Caller wants a different cache dir than the autouse fixture set
        # up. Build a one-off BackendCacheManager rather than mutating
        # global state.
        cache: BackendCacheManager = BackendCacheManager(persist_dir)
    else:
        cache = get_active_cache()
        if cache is None:
            raise RuntimeError(
                "BackendCacheManager unavailable: persist_dir is not set. "
                "Either pass persist_dir=... explicitly, or ensure the "
                "autouse `run_before_and_after_tests` fixture in "
                "test/conftest.py has run.")

    fingerprint = _build_fingerprint(
        connector_name=connector_name,
        table_name=table_name,
        df_columns=list(df.columns),
        primary_key=primary_key,
        time_col=time_col,
    )

    # ------------------------------------------------------------------
    # Column stats: compute per-column via compute_col_stats_python
    # (same path the python data backend uses in production), then drop
    # them into BackendCacheManager and persist to disk.
    # ------------------------------------------------------------------
    col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {}
    for col_name in df.columns:
        ser = df[col_name]
        kumo_dtype, stats_for_col = (
            compute_stats_python.compute_supplemental_col_stats_from_series(
                ser,
                data_version=data_version,
                approx=False,
                include_heavy_nlp=include_heavy_nlp,
            ))
        if not stats_for_col:
            continue
        # The cache lookup key embeds the column's stype (see
        # ``col_stat_key`` in :mod:`kumo.data.backend_cache`), so a single
        # cache entry only matches lookups that use the exact same stype.
        # The test's :class:`Table` may query with stype=None (when stype
        # hasn't been inferred yet) OR with an inferred stype. Mirror DG
        # ingest's idempotent behavior by writing the same Stats payload
        # under each (stype=None, dtype) and every (stype, dtype) pair
        # whose ``Stype.supports_dtype`` accepts the column's dtype. The
        # extra keys are dead weight at most — they're never queried by
        # callers that don't ask for them.
        col_stats[BaseColumnConfig(name=col_name, stype=None,
                                   dtype=kumo_dtype)] = stats_for_col
        for stype in Stype:
            if stype == Stype.unsupported:
                continue
            try:
                supported = stype.supports_dtype(kumo_dtype)
            except Exception:
                supported = False
            if supported:
                col_stats[BaseColumnConfig(name=col_name, stype=stype,
                                           dtype=kumo_dtype)] = stats_for_col

    if col_stats:
        cache.add_col_stats(fingerprint, col_stats)
        cache.save_col_stats_cache(fingerprint, data_version)

    # ------------------------------------------------------------------
    # Table stats: NUM_ROWS is always safe; SOURCE_SIZE/MATERIALIZED_SIZE
    # require a real file on disk and are skipped here (the survivor
    # call site at kumo/graph/table.py compute_table_stats falls back to
    # `compute_table_stats_native` for any missing entries).
    # ------------------------------------------------------------------
    table_stats: Dict[TableStatType, Stats] = {
        TableStatType.NUM_ROWS:
        Stats(value=len(df), approx=False, data_version=data_version),
    }
    cache.add_table_stats(fingerprint, table_stats)
    cache.save_table_stats(fingerprint, data_version, table_stats)

    # Persist the table_version cache too — survivor connector code paths
    # call `get_data_version` which reads from `table_version_cache`. Test
    # connectors usually compute their own version (e.g. parquet mtime)
    # so this is a hint rather than a load-bearing write.
    cache.table_version_cache.save_in_memory(connector_name, table_name,
                                             data_version)
