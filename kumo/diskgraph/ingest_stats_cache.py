# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy Rust-produced DG stats from an ingest index into the Kumo data-backend cache.

The DiskGraph Rust ingestor writes kumo-format col stats under
``{index_path}/kumo_col_stats/`` and edge stats under
``{index_path}/edge_stats.json``. Downstream Kumo code (encoders,
link-health UI, etc.) consumes these via the data-backend cache layout,
so after ingest we read the JSON via pybind and write it into the
Kumo-owned backend cache.
"""

import json
from typing import Any, Dict, List, Optional, Protocol

from kumo_core.types import TableStatType

from io_utils import fsutil, logging
from kumo.data.backend_cache import get_active_cache
from kumo.data.cache.graph_cache import VersionedGraphKey
from kumo.stats import Stats

log = logging.getLogger(__name__)


def _pkey_duplicate_count_for_table(
    pkey_duplicate_counts: Dict[str, int],
    table_name: str,
) -> int:
    return sum(count for key, count in pkey_duplicate_counts.items()
               if key == table_name or key.startswith(f'{table_name}:shard-'))


class _Table(Protocol):
    """Structural view of v2 ``TableDefinition`` for col-stats caching.

    Declared locally so this module does not depend on ``rest.config``;
    ``TableDefinition`` instances satisfy the protocol by duck-typing.
    The per-graph alias is supplied as the dict key, not as a field on
    the table.
    """
    def get_connector_id(self) -> Any:
        ...

    def get_source_table_name(self) -> Any:
        ...

    def fingerprint(self) -> Any:
        """Table fingerprint used as the cache key."""


def _copy_col_stats(
    index_path: str,
    tables: Dict[str, _Table],
    table_data_versions: dict,
) -> None:
    """Copy per-table kumo-format col stats into the table cache.

    ``tables`` maps per-graph alias -> table. The alias is the lookup
    key DG uses to address a table's stats inside ``index_path``.
    """
    from diskgraph import read_col_stats as _read_col_stats

    for table_name, table in tables.items():
        version = str(table_data_versions.get(table_name, ''))
        if not version:
            raise RuntimeError(f'No table_data_version for table {table_name}')

        full_json = _read_col_stats(index_path, table_name, 'full')
        if full_json is None:
            raise RuntimeError(
                f'Diskgraph ingest did not produce col stats for table '
                f'{table_name}. Ensure the diskgraph package is up to date.')
        min_json = _read_col_stats(index_path, table_name, 'min')

        table_dir = get_active_cache().create_table_cache_dir(
            str(table.get_connector_id()),
            table.get_source_table_name(),
            version,
            table.fingerprint(),
        )
        fsutil.write_text(table_dir.col_stats, full_json)
        fsutil.write_text(table_dir.min_col_stats,
                          min_json if min_json is not None else full_json)


def _read_pkey_duplicate_counts(
        ingest_stats_path: str) -> Optional[Dict[str, int]]:
    try:
        report = json.loads(fsutil.read_text(ingest_stats_path))
        stats = report.get('stats')
        if not isinstance(stats, dict) or 'pk_duplicate_counts' not in stats:
            return None
        duplicate_counts = stats['pk_duplicate_counts']
        if not isinstance(duplicate_counts, dict):
            return None
        if any(not isinstance(key, str) or isinstance(count, bool)
               or not isinstance(count, int) or count < 0
               for key, count in duplicate_counts.items()):
            return None
        return duplicate_counts
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _save_num_rows_from_node_counts(
    ingest_stats_path: str,
    tables: Dict[str, _Table],
    table_data_versions: dict,
) -> None:
    """Backfill per-table ``NUM_ROWS`` from the DG ingest's ``node_counts``.

    The Rust ingestor already counts rows per node table while building the
    index and writes them to ``ingest_stats.json`` under
    ``['stats']['node_counts']`` (keyed by the per-graph alias). Reusing that
    count means no separate parquet footer scan is needed to populate
    ``NUM_ROWS`` — which downstream LP candidate/entity generation relies on.

    Merged (not saved) so it never clobbers ``SOURCE_SIZE`` and other stats
    already written for the same table; read via ``fsutil`` so it stays
    Temporal-sandbox safe.
    """
    try:
        node_counts = (((json.loads(fsutil.read_text(ingest_stats_path))
                         or {}).get('stats') or {}).get('node_counts') or {})
    except Exception:
        log.warning('Could not read node_counts from %s; NUM_ROWS left unset',
                    ingest_stats_path)
        return

    cache = get_active_cache()
    for alias, table in tables.items():
        count = node_counts.get(alias)
        if count is None:
            count = node_counts.get(str(table.get_source_table_name()))
        if count is None:
            continue
        version = str(table_data_versions.get(alias, ''))
        if not version:
            continue
        cache.add_table_stats(
            table.fingerprint(),
            {
                TableStatType.NUM_ROWS:
                Stats(value=int(count), approx=False, data_version=version)
            },
        )


def _copy_edge_stats(
    index_path: str,
    graph_key: str,
    graph_data_version: str,
    tables: Dict[str, _Table],
    has_edges: bool = True,
) -> None:
    """Copy edge stats JSON into the graph cache."""
    from diskgraph import read_edge_stats as _read_edge_stats

    edge_json = _read_edge_stats(index_path)
    if edge_json is None:
        if not has_edges:
            log.info(
                'Diskgraph ingest produced no edge stats for no-edge graph '
                '%s at %s', graph_key, index_path)
            return
        raise RuntimeError(
            f'Diskgraph ingest did not produce edge_stats.json at '
            f'{index_path}. Ensure the diskgraph package is up to date.')

    graph_cache = get_active_cache().graph_cache
    dest = graph_cache.edge_stats_path(
        VersionedGraphKey(graph_key, graph_data_version))
    log.info('Writing diskgraph edge stats to %s', dest)
    stats = json.loads(edge_json)
    source_names = {
        alias: str(table.get_source_table_name())
        for alias, table in tables.items()
    }
    known_source_names = set(source_names.values())
    normalized_stats = {}
    for key, value in stats.items():
        parts = key.split('__')
        if len(parts) != 7:
            raise ValueError(f'Invalid diskgraph edge stat key: {key}')
        for position in (0, 2):
            table_name = parts[position]
            if table_name in source_names:
                parts[position] = source_names[table_name]
            elif table_name not in known_source_names:
                raise ValueError(
                    f'Unknown table identity in diskgraph edge stat key: '
                    f'{key}')
        normalized_stats['__'.join(parts)] = value
    fsutil.write_text(dest, json.dumps(normalized_stats))


def copy_ingest_stats_to_backend_cache(
    index_path: str,
    ingest_stats_path: str,
    graph_key: str,
    graph_data_version: str,
    tables: Dict[str, _Table],
    table_data_versions: dict,
    has_edges: bool = True,
) -> Optional[Dict[str, int]]:
    """Copy both col stats and edge stats from a DG index into the cache.

    ``tables`` is alias-keyed (alias -> v2 ``TableDefinition``).
    Called by :meth:`DiskgraphService.copy_ingest_stats_to_cache` and by
    ``diskgraph_ingest_stats_activity`` via the service.
    """
    pkey_duplicate_counts = _read_pkey_duplicate_counts(ingest_stats_path)
    _copy_col_stats(
        index_path,
        tables,
        table_data_versions,
    )
    _save_num_rows_from_node_counts(ingest_stats_path, tables,
                                    table_data_versions)
    _copy_edge_stats(
        index_path,
        graph_key,
        graph_data_version,
        tables,
        has_edges,
    )
    if pkey_duplicate_counts is None:
        return None
    return {
        table_name:
        _pkey_duplicate_count_for_table(pkey_duplicate_counts, table_name)
        for table_name in tables
    }


class _GraphTable(Protocol):
    """Structural view of ``kumo.graph.Table`` for the seed path."""
    table_name: str
    name: str
    data_version: str

    def fingerprint(self) -> Any:
        """Table fingerprint with ``connector_name`` / ``source_name``."""


def seed_missing_col_stats_from_index(
    index_path: str,
    graph_tables: List[_GraphTable],
) -> None:
    """Copy DG col stats into the backend cache, filling any gaps.

    Used by cold-cache training paths (fresh activity subprocess /
    custom-TT jobs) where the per-table cache entries may not have
    been seeded by the main ingest-stats activity. Idempotent —
    always overwrites so the DG index remains authoritative.

    Tries ``fingerprint.source_name`` first, then ``table_name``,
    because DG uses unqualified names.
    """
    from diskgraph import read_col_stats as _read_col_stats

    cache = get_active_cache()
    for table in graph_tables:
        fp = table.fingerprint()
        table_cache = cache.create_table_cache_dir(fp.connector_name,
                                                   fp.source_name,
                                                   table.data_version, fp)

        for candidate_name in (fp.source_name, table.table_name):
            full_json = _read_col_stats(index_path, candidate_name, 'full')
            if full_json is None:
                continue
            log.info(
                'Seeding missing col stats for table %s (version=%s) '
                'from DG index', table.name, table.data_version)
            fsutil.write_text(table_cache.col_stats, full_json)
            # Preserve original: only seed min if cache doesn't already
            # have one.
            if not fsutil.exists(table_cache.min_col_stats):
                min_json = _read_col_stats(index_path, candidate_name, 'min')
                fsutil.write_text(
                    table_cache.min_col_stats,
                    min_json if min_json is not None else full_json)
            break
        else:
            log.warning(
                'Could not find col stats in DG index for table %s '
                '(tried %s, %s)', table.name, fp.source_name, table.table_name)
