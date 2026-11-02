# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable snapshot updates and exact append staging."""
import csv
import json
import shutil
import tempfile
import time
import warnings
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as arrow_csv
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from kumo_core.diskgraph.incremental import (
    SourceFingerprint,
    detect_append,
    warn_encoder_drift,
)
from kumo_core.diskgraph.snapshot import (
    DiskgraphIngestResult,
    write_diskgraph_success_marker,
)


def _stage_additions(parts, table, destination):
    from kumo_core.sources import csv_format
    from kumo_core.sources.object_store import _fragment_batches, _projection

    destination.mkdir()
    files = []
    parquet_parts = [
        part for part in parts if Path(part.path).suffix.lower() == '.parquet'
    ]
    reader, name = table.connector.source_backend._reader(
        table.connector, table.table_name)
    columns = {column.name: column.dtype for column in table.definition.cols}
    if parquet_parts:
        dataset = reader.dataset(name)
        partitions = dataset.partitioning.schema.names if dataset.partitioning is not None else []
        projection = _projection(name, dataset.schema, columns, {},
                                 converts=reader.ingest_converts,
                                 partitions=partitions)
        fragments = {
            Path(fragment.path).resolve(): fragment
            for fragment in dataset.get_fragments()
        }
    for i, part in enumerate(parts):
        with ExitStack() as stack:
            target = destination / f'{i}.parquet'
            path = Path(part.path)
            if path.suffix.lower() == '.parquet':
                fragment = fragments[path.resolve()]
                fragment.ensure_complete_metadata()
                fragment = fragment.subset(row_group_ids=list(
                    range(part.row_group, fragment.num_row_groups)))
                batches = _fragment_batches(fragment, dataset.schema,
                                            projection, list(columns),
                                            own_types=reader.ingest_converts)
            else:
                options = reader.csv_options
                separator = (options.delimiter
                             if options else None) or csv_format.delimiter(
                                 reader.filesystem, str(path))
                with path.open(newline='', encoding='utf-8-sig') as header:
                    header_columns = next(
                        csv.reader(
                            header, delimiter=separator,
                            escapechar=options.escape_char or None
                            if options else None))
                stream = stack.enter_context(path.open('rb'))
                stream.seek(part.byte_offset)
                text_columns = {
                    column: pa.string()
                    for column in csv_format.pinned_types(columns)
                }
                batches = arrow_csv.open_csv(
                    stream, read_options=arrow_csv.ReadOptions(
                        column_names=header_columns if part.
                        byte_offset else None),
                    parse_options=csv_format.parse_options(separator, options),
                    convert_options=csv_format.convert_options(
                        text_columns, options))
                csv_projection = _projection(
                    name, batches.schema, columns, {}, normalize=True,
                    null_values=options.null_values if options is not None
                    and options.null_values is not None else
                    csv_format.ARROW_NA_TOKENS, converts=reader.ingest_converts)
                batches = (ds.InMemoryDataset(batch).to_table(
                    columns=csv_projection, use_threads=False)
                           for batch in batches)
            writer = None
            for batch in batches:
                if not batch.num_rows:
                    continue
                if writer is None:
                    writer = stack.enter_context(
                        pq.ParquetWriter(target, batch.schema))
                writer.write(batch)
        if writer is not None:
            files.append(str(target))
    return files


def _warn_append_drift(snapshot, name, paths):
    reference = json.loads(
        (snapshot.index_path / 'kumo_col_stats/col_stats_full' /
         f'{name}.json').read_text())
    observed, categories = {}, {}
    known = {
        key: {value
              for value, _ in stat.get('value', [])}
        for key, stat in reference.items()
        if '__kumo.categorical__CATEGORY_COUNTS__False' in key
    }
    columns = {
        key.rsplit('__', 3)[0]
        for key in reference if '__kumo.numerical__MIN__False' in key
        or '__kumo.categorical__CATEGORY_COUNTS__False' in key
    }
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=sorted(columns)):
            for column in batch.schema.names:
                values = batch[column]
                category_key = f'{column}__kumo.categorical__CATEGORY_COUNTS__False'
                if category_key in reference:
                    added = categories.setdefault(category_key, set())
                    if len(added) <= len(known[category_key]) * 0.05:
                        for value in pc.unique(values).drop_null().to_pylist():
                            if value not in known[category_key]:
                                added.add(value)
                            if len(added) > len(known[category_key]) * 0.05:
                                break
                else:
                    limits = pc.min_max(values).as_py()
                    for metric, value in [('MIN', limits['min']),
                                          ('MAX', limits['max'])]:
                        if value is None:
                            continue
                        key = f'{column}__kumo.numerical__{metric}__False'
                        prior = observed.get(key, {}).get('value', value)
                        observed[key] = {
                            'value':
                            min(prior, value) if metric == 'MIN' else max(
                                prior, value)
                        }
    observed.update({
        key: {
            'value': [(value, 1) for value in values]
        }
        for key, values in categories.items()
    })
    warn_encoder_drift(name, reference, observed)


def update_snapshot(snapshot, connector, *, mode, compact, reuse_encoders):
    started = time.perf_counter()
    from diskgraph import append_snapshot

    from kumo.engine.connector.file import FileConnector
    from kumo.engine.graph import Graph, GraphSnapshot

    if mode not in ('auto', 'append', 'full'):
        raise ValueError("mode must be 'auto', 'append', or 'full'")
    if snapshot._graph is None or not (snapshot.index_path /
                                       '_incremental/manifest.json').is_file():
        raise ValueError(
            'Snapshot is not update-enabled; create graph.snapshot(enable_updates=True) first.'
        )
    graph = snapshot._graph
    if connector is not None:
        graph = Graph(
            tables={
                name: replace(table, connector=connector)
                for name, table in graph._named_tables.items()
            }, edges=graph.edges)
    if mode == 'full':
        return graph.snapshot(previous_snapshot=snapshot,
                              force_tables=tuple(graph._named_tables),
                              reuse_encoders=reuse_encoders)

    def fallback(reason):
        if mode == 'append':
            raise ValueError(f'Cannot append: {reason}')
        result = graph.snapshot(previous_snapshot=snapshot,
                                reuse_encoders=reuse_encoders)
        if result.tables_rebuilt:
            warnings.warn(f'Using table rebuilds: {reason}', UserWarning, stacklevel=3)
        return result.compact() if compact else result

    if not reuse_encoders:
        return fallback('append requires frozen encoders')
    if not all(
            isinstance(table.connector, FileConnector)
            for table in graph._named_tables.values()):
        return fallback('append currently requires local file sources')
    previous = json.loads((snapshot.index_path /
                           '_incremental/source_fingerprints.json').read_text())
    current, additions, changed = {}, {}, []
    for name, table in graph._named_tables.items():
        backend = table.connector.source_backend
        reader, source_name = backend._reader(table.connector, table.table_name)
        root, _ = reader.resolve(source_name)
        current[name] = reader.fingerprint(source_name)
        parts = detect_append(Path(root), SourceFingerprint(**previous[name]),
                              current[name])
        if parts is None:
            changed.append(name)
        if parts:
            additions[name] = parts
    if changed:
        if mode == 'append' or not additions:
            return fallback(f'{", ".join(changed)} is not a proven append')
        cache = json.loads(
            (snapshot.index_path / '_incremental/manifest.json').read_text())
        if graph.id != snapshot._graph.id or any(
                cache['versions'].get(name) != previous[name]['content_hash']
                for name in additions):
            return fallback(
                'mixed update requires rebuilding an earlier append generation')
        snapshot = graph.snapshot(
            previous_snapshot=snapshot, _deferred_fingerprints={
                name: SourceFingerprint(**previous[name])
                for name in additions
            })
    if not additions:
        return snapshot.compact() if compact else snapshot
    parent_stats = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    if (parent_stats.get('key_lookup_failures', 0) or parent_stats.get(
            'key_membership_failures', 0)) and any(edge.dst_table in additions
                                                   for edge in graph.edges):
        return fallback('new entities may resolve historical orphan edges')
    output = snapshot.index_path.parent / uuid4().hex
    snapshot_id = f'{graph.id}@{output.name}'
    try:
        with tempfile.TemporaryDirectory(prefix='append-',
                                         dir=output.parent) as staging:
            paths = {
                name:
                _stage_additions(parts, graph._named_tables[name],
                                 Path(staging) / name)
                for name, parts in additions.items()
            }
            # Guard the proof-to-copy interval before native publication.
            appended_rows = {
                name: sum(pq.read_metadata(path).num_rows for path in files)
                for name, files in paths.items()
            }
            paths = {
                name: files
                for name, files in paths.items() if appended_rows[name]
            }
            for name, files in paths.items():
                _warn_append_drift(snapshot, name, files)
            for name, table in graph._named_tables.items():
                now = table.connector.source_backend.get_source_fingerprint(
                    table.connector, table.table_name)
                if now.content_hash != current[name].content_hash:
                    raise RuntimeError(f'Source changed while staging {name}')
            append_snapshot(
                json.dumps({
                    'parent': str(snapshot.index_path),
                    'output': str(output),
                    'tables': paths
                }))
            from kumo.engine.feature_updates import retain_sources

            for name, files in paths.items():
                if (snapshot.index_path / '_incremental/sources' / name).exists():
                    retain_sources(output, name, files, append=True)
        old = json.loads(
            (snapshot.index_path / '_incremental/update.json').read_text())
        metadata = json.loads((output / 'index_metadata.json').read_text())
        tables = {}
        for table in metadata['node_types']:
            name = table['name']
            rows = sum(
                max(0, shard['max_id'] - shard['min_id'] + 1)
                for shard in table['shards'])
            source_rows = current[name].row_count
            if source_rows is None:
                source_rows = previous[name]['row_count'] + appended_rows.get(
                    name, 0)
            current[name] = replace(current[name], table_name=name,
                                    row_count=source_rows)
            tables[name] = {
                'status':
                'appended' if name in additions else
                ('rebuilt' if name in changed else 'reused'),
                'rows':
                rows,
                'rows_added':
                rows - old['tables'][name]['rows'],
                'encoder_state':
                'inherited'
            }
        record = dict(snapshot_id=snapshot_id, parent_snapshot_id=snapshot.id,
                      parent=str(snapshot.index_path), created_at=datetime.now(
                          timezone.utc).isoformat(), creation_mode='append',
                      update_chain_length=old['update_chain_length'] + 1,
                      tables=tables,
                      duration_seconds=time.perf_counter() - started)
        stats_path = output / 'ingest_stats.json'
        report = json.loads(stats_path.read_text())
        stats = report['stats']
        append_stats = json.loads(
            (output / '_incremental/append_stats.json').read_text())
        stats['key_lookup_failures'] = stats.get(
            'key_lookup_failures', 0) + append_stats['key_lookup_failures']
        base_edges = stats.setdefault('base_edge_counts',
                                      dict(stats['edge_counts']))
        deltas = json.loads((output / '_incremental/deltas.json').read_text())
        counts = {
            edge: count
            for table in deltas['tables'].values()
            for edge, count in table['edge_counts'].items()
        }
        stats['edge_counts'] = {
            edge: count + counts.get(edge, 0)
            for edge, count in base_edges.items()
        }
        record['edges'] = {
            edge: {
                'status':
                'delta_appended' if count > parent_stats['edge_counts'].get(
                    edge, 0) else 'reused',
                'delta_entries':
                counts.get(edge, 0),
                'fragmentation':
                counts.get(edge, 0) /
                base_edges[edge] if base_edges[edge] else None
            }
            for edge, count in stats['edge_counts'].items()
        }
        record['total_fragmentation'] = sum(counts.values()) / max(
            1, sum(base_edges.values()))
        stats['node_counts'] = {
            name: table['rows']
            for name, table in tables.items()
        }
        stats['total_nodes'] = sum(stats['node_counts'].values())
        stats['total_edges'] = sum(stats['edge_counts'].values())
        stats['source_rows_by_table'] = {
            name: fp.row_count
            for name, fp in current.items()
        }
        stats_path.write_text(json.dumps(report))
        (output / '_incremental/update.json').write_text(json.dumps(record))
        (output / '_incremental/source_fingerprints.json').write_text(
            json.dumps({
                name: asdict(fp)
                for name, fp in current.items()
            }))
        write_diskgraph_success_marker(str(output),
                                       DiskgraphIngestResult.for_index(output))
    except Exception as error:
        if output.exists():
            shutil.rmtree(output)
        if mode == 'auto' and any(
                reason in str(error)
                for reason in ('existing or duplicate primary key',
                               'requires full table encoding',
                               'frozen sort-key range',
                               'append has a null timestamp')):
            return fallback(str(error))
        raise
    result = GraphSnapshot(snapshot_id, output, graph._graph_config, graph)
    if record['update_chain_length'] >= 30:
        warnings.warn(
            'Snapshot has at least 30 updates; compact or rebuild it.',
            UserWarning, stacklevel=2)
    ratios = result.fragmentation
    if compact or record['update_chain_length'] >= 30 or any(
            ratio > 0.25 for ratio in ratios.values()):
        result = result.compact()
        path = result.index_path / '_incremental/update.json'
        record = json.loads(path.read_text())
        record['duration_seconds'] = time.perf_counter() - started
        path.write_text(json.dumps(record))
        return result
    if any(ratio > 0.10 for ratio in ratios.values()):
        warnings.warn(
            'Edge deltas exceed 10% of the base; compaction is recommended.',
            UserWarning, stacklevel=2)
    return result


def fragmentation(snapshot):
    path = snapshot.index_path / '_incremental/deltas.json'
    stats = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    base = stats.get('base_edge_counts', stats['edge_counts'])
    counts = {}
    if path.exists():
        manifest = json.loads(path.read_text())
        counts = {
            edge: count
            for table in manifest['tables'].values()
            for edge, count in table['edge_counts'].items()
        }
    return {
        edge:
        counts.get(edge, 0) / count if count else
        (float('inf') if counts.get(edge, 0) else 0.0)
        for edge, count in base.items()
    }


def compact_snapshot(snapshot):
    from diskgraph import compact_snapshot as native_compact

    from kumo.engine.graph import GraphSnapshot

    if not (snapshot.index_path / '_incremental/manifest.json').is_file():
        raise ValueError('Legacy snapshots cannot be compacted.')
    deltas_path = snapshot.index_path / '_incremental/deltas.json'
    deltas = json.loads(
        deltas_path.read_text())['tables'] if deltas_path.exists() else {}
    pending_keys = any(
        len(table['primary_key_logs']) > 1 or any(
            not (snapshot.index_path / path).with_suffix('.idx').is_file()
            for path in table['primary_key_logs']) for table in deltas.values())
    pending_features = any(table.get('feature_file') for table in deltas.values())
    pending_edges = any(table['edge_logs'] for table in deltas.values())
    if not pending_edges and not pending_keys and not pending_features:
        return snapshot
    output = snapshot.index_path.parent / uuid4().hex
    started = time.perf_counter()
    snapshot_id = f'{snapshot.id.split("@")[0]}@{output.name}'
    try:
        native_compact(str(snapshot.index_path), str(output))
        record_path = output / '_incremental/update.json'
        record = json.loads(record_path.read_text())
        record.update(snapshot_id=snapshot_id, parent_snapshot_id=snapshot.id,
                      parent=str(snapshot.index_path),
                      creation_mode='compaction',
                      created_at=datetime.now(timezone.utc).isoformat(),
                      update_chain_length=0,
                      duration_seconds=time.perf_counter() - started)
        for edge in record.get('edges', {}).values():
            edge.update(status='compacted', delta_entries=0, fragmentation=0.0)
        record['total_fragmentation'] = 0.0
        record_path.write_text(json.dumps(record))
        stats_path = output / 'ingest_stats.json'
        report = json.loads(stats_path.read_text())
        report['stats']['base_edge_counts'] = dict(
            report['stats']['edge_counts'])
        stats_path.write_text(json.dumps(report))
        write_diskgraph_success_marker(str(output),
                                       DiskgraphIngestResult.for_index(output))
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise
    return GraphSnapshot(snapshot_id, output, snapshot.graph_config,
                         snapshot._graph)
