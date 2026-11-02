# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskGraph ingest entry point + graph-config normalization helpers.

``run_diskgraph_ingest`` launches the standalone DiskGraph supervisor. The
normalization helpers reshape kumo's ``GraphConfig`` JSON into the
``connector_id`` / ``source_table_name`` shape that
``DiskgraphConfigConverter`` expects.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Callable, Optional

from kumo_core.diskgraph.converter import convert_diskgraph_config
from kumo_core.diskgraph.ingest import (
    normalize_legacy_column_aliases,
    set_timestamp_precision,
    validate_diskgraph_ingest_config,
)
from kumo_core.diskgraph.runner import (
    Diagnostic,
    DiskgraphIngestError,
    run_ingest,
)
from kumo_core.diskgraph.snapshot import DiskgraphIngestResult

from io_utils.json import to_json_dict
from kumo.config.graph import GraphConfig
from kumo.diskgraph._diskgraph_legacy import \
    infer_connector_id_from_ingest_dir as _infer_connector_id_from_ingest_dir
from kumo.diskgraph.errors import as_platform_diskgraph_exception
from statedb.utils.errors import DiskgraphError, rethrow_or_wrap

logger = logging.getLogger(__name__)


def _log_diagnostic(diagnostic: Diagnostic) -> None:
    chain = " -> ".join(f"{cause.kind}:{cause.summary}"
                        for cause in diagnostic.chain)
    frames = ",".join(f"{frame.module}@{frame.file}:{frame.line}"
                      for frame in diagnostic.frames)
    logger.error(
        "DiskGraph diagnostic: diagnostic_id=%s visibility=%s "
        "component=diskgraph-ingest operation=%s category=%s code=%s "
        "retryable=%s attempt=%d phase=%s exit_code=%s signal=%s "
        "summary=%s chain=%s frames=%s fingerprint=%s "
        "truncated_chain=%s truncated_frames=%s truncated_summary=%s",
        diagnostic.diagnostic_id,
        diagnostic.visibility,
        diagnostic.operation,
        diagnostic.category,
        diagnostic.code,
        diagnostic.retryable,
        diagnostic.fields.attempt,
        diagnostic.fields.phase,
        diagnostic.fields.exit_code,
        diagnostic.fields.signal,
        diagnostic.summary,
        chain,
        frames,
        diagnostic.fingerprint,
        diagnostic.chain_truncated,
        diagnostic.frames_truncated,
        diagnostic.summary_truncated,
    )


def normalize_diskgraph_source_fields(
    graph_config_json: dict,
    data_persist_dir: str,
    table_data_versions: dict,
) -> dict:
    """Normalize source table metadata for diskgraph config conversion.

    Different call paths represent source metadata differently:
    ``connector_id``/``source_table_name``,
    ``source_table.{connector_id, source_table_name}``,
    ``data_source``/``source_name`` (kumo GraphConfig). When
    ``connector_id`` is missing or default, attempts best-effort
    inference from existing ingest directories.
    """
    normalized = json.loads(json.dumps(graph_config_json))
    for table in normalized.get('tables', []):
        source_table = table.get('source_table', {}) or {}

        for fk in table.get('fkeys', []):
            dst_table = fk.get('dst_table')
            if isinstance(dst_table, str) and '.' in dst_table:
                # GraphConfig may use fully-qualified names
                # (<connector>.<table>); strip to bare table name.
                fk['dst_table'] = dst_table.split('.')[-1]

        source_table_name = (table.get('source_table_name')
                             or source_table.get('source_table_name')
                             or table.get('source_name')
                             or table.get('table_name'))
        connector_id = (table.get('connector_id')
                        or source_table.get('connector_id')
                        or table.get('data_source'))

        if (not connector_id
                or connector_id == 'default') and source_table_name:
            table_name = table.get('table_name') or source_table_name
            logger.warning(
                "connector_id missing or 'default' for table '%s' after "
                "upstream hydration — falling back to ingest dir inference. "
                "This indicates a bug in source metadata propagation.",
                table_name,
            )
            table_version = table_data_versions.get(table_name)
            if table_version:
                inferred = _infer_connector_id_from_ingest_dir(
                    data_persist_dir,
                    str(source_table_name),
                    str(table_version),
                )
                if inferred:
                    connector_id = inferred

        if connector_id:
            table['connector_id'] = connector_id
        if source_table_name:
            table['source_table_name'] = source_table_name

        source_table.setdefault('connector_id', connector_id)
        source_table.setdefault('source_table_name', source_table_name)
        table['source_table'] = source_table

    return normalized


def prepare_diskgraph_converter_config(graph_config_json: dict) -> dict:
    """Prepare Kumo GraphConfig JSON for DiskGraph conversion.

    DiskGraph parses customer columns declared as dtype=string + stype=sequence
    into float arrays at ingest time, so the converter must describe their
    storage dtype as floatlist until the type system can distinguish source
    representation from storage representation; `metadata.TS_FORMAT` is the
    same kind of source-side alias for the converter's `timestamp_format`.
    ``kumo_core`` owns both rewrites so that the hosted platform and
    ``LocalBackend`` accept exactly the same graph.
    """
    converter_config = copy.deepcopy(graph_config_json)
    normalize_legacy_column_aliases(converter_config)
    return converter_config


def run_diskgraph_ingest(
    graph_config: Optional[GraphConfig],
    data_persist_dir: str,
    table_data_versions: dict,
    output_dir: str,
    output_dataplane_dir: str,
    success_marker_content: str,
    source_paths: Optional[dict] = None,
    source_specs: Optional[dict] = None,
    subprocess_environment: Optional[dict[str, Optional[str]]] = None,
    heartbeat: Optional[Callable[[], None]] = None,
    timestamp_precision: str = 'seconds',
) -> DiskgraphIngestResult:
    """Convert the kumo graph config and launch the DiskGraph supervisor.

    Rust stream-mirrors each output file to ``output_dataplane_dir``
    while writing locally, finalizes ``_MANIFEST.json`` on both
    backends, then writes ``_SUCCESS`` (with ``success_marker_content``)
    atomically last on both — returning only once both backends are
    durable.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ingest_config = None
    if graph_config is not None:
        graph_config_json = normalize_diskgraph_source_fields(
            to_json_dict(graph_config),
            data_persist_dir,
            table_data_versions,
        )
        converter_config = prepare_diskgraph_converter_config(graph_config_json)
        ingest_config = convert_diskgraph_config(
            converter_config,
            data_persist_dir,
            {
                k: str(v)
                for k, v in table_data_versions.items()
            },
            source_paths=source_paths,
            source_specs=source_specs,
            subprocess_environment=subprocess_environment,
        )
        set_timestamp_precision(ingest_config, timestamp_precision)
        validate_diskgraph_ingest_config(ingest_config)

    # DiskGraph owns phase detection, concurrency reduction after OOM, and
    # resume from completed phase markers.
    #
    # Path layout (``diskgraph_snapshots/<key>/<ver>``) is owned by
    # ``DiskgraphService._dataplane_path``; the binary treats it as
    # opaque. The wrapper forwards `output_dataplane_dir` and
    # `success_marker_content` as `--output-dataplane-dir` /
    # `--success-marker-content` CLI flags so the binary streams every
    # output file to the dataplane and writes `_MANIFEST.json` /
    # `_SUCCESS` last.
    diskgraph_environment: dict[str,
                                Optional[str]] = dict(subprocess_environment
                                                      or {})
    # Nothing sets this any more, so unsetting it is defence rather than
    # cleanup: a value inherited from this process' own environment would
    # otherwise reach the binary, and the point of the retirement is that no
    # Databricks credential can. ``None`` removes the name from the child.
    diskgraph_environment['DISKGRAPH_EPHEMERAL_DATABRICKS_CONFIG'] = None

    try:
        run_result = run_ingest(
            ingest_config,
            output_path,
            col_stats=True,
            cat_count_strategy='hybrid',
            output_dataplane_dir=output_dataplane_dir,
            success_marker_content=success_marker_content,
            heartbeat=heartbeat,
            subprocess_environment=diskgraph_environment,
        )
        stats = run_result.stats.get('stats') or {}
        logger.info('diskgraph ingest supervisor completed successfully')
    except DiskgraphIngestError as e:
        logger.error('Diskgraph ingest failed: %s', e, exc_info=True)
        if e.diagnostic is not None:
            _log_diagnostic(e.diagnostic)
        if e.private_output_tail:
            logger.debug('Diskgraph ingest private output tail:\n%s',
                         e.private_output_tail)
        raise as_platform_diskgraph_exception(e) from None
    except Exception as e:
        logger.error('Diskgraph ingest failed: %s', e, exc_info=True)
        rethrow_or_wrap(
            e,
            DiskgraphError,
            'Kumo encountered an error during graph ingestion.',
        )
    logger.info(
        'diskgraph ingest completed: %d node tables, %d edge relations, '
        '%d nodes, %d edges',
        stats.get('node_tables', 0),
        stats.get('edge_relations', 0),
        stats.get('total_nodes', 0),
        stats.get('total_edges', 0),
    )

    col_stats_path = output_path / 'col_stats.json'
    ingest_stats_path = output_path / 'ingest_stats.json'
    index_metadata_path = output_path / 'index_metadata.json'
    kumo_stats_dir = output_path / 'kumo_col_stats'
    manifest_path = output_path / '_MANIFEST.json'
    success_path = output_path / '_SUCCESS'

    missing_files = [
        str(path) for path in (
            col_stats_path,
            ingest_stats_path,
            index_metadata_path,
            manifest_path,
            success_path,
        ) if not path.is_file()
    ]
    if not kumo_stats_dir.is_dir():
        missing_files.append(str(kumo_stats_dir))
    if missing_files:
        raise RuntimeError(
            'diskgraph ingest completed but missing expected output files: '
            f"{', '.join(missing_files)}")

    return DiskgraphIngestResult(
        index_path=str(output_path),
        col_stats_path=str(col_stats_path),
        ingest_stats_path=str(ingest_stats_path),
        kumo_stats_dir=str(kumo_stats_dir),
    )
