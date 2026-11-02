# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable additions of tables and their relationships."""
import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from kumo.engine.feature_updates import retain_sources


def try_expansion(graph, parent, fingerprints):
    from diskgraph import DiskgraphConfigConverter, append_snapshot
    from kumo_core.diskgraph.ingest import prepare_diskgraph_converter_config
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        write_diskgraph_success_marker,
    )

    from kumo.engine.graph import GraphSnapshot, _warn_ingest_data_loss

    started = time.perf_counter()
    previous = json.loads((parent.index_path / '_incremental/source_fingerprints.json').read_text())
    old_names = set(previous)
    new_names = set(fingerprints) - old_names
    if not new_names or not old_names <= fingerprints.keys():
        return None
    if any(fingerprints[name].content_hash != previous[name]['content_hash'] for name in old_names):
        return None
    report = json.loads((parent.index_path / 'ingest_stats.json').read_text())
    stats = report['stats']
    if (stats.get('pk_duplicate_total', 0) or stats.get('key_parse_failures', 0)
            or any(stats.get('dropped_rows_counts', {}).values())
            or any(stats.get('pk_null_timestamp_drops', {}).values())):
        return None
    sources = {name: sorted((parent.index_path / '_incremental/sources' / name).glob('*.parquet'))
               for name in old_names}
    if not all(sources.values()):
        return None
    output = parent.index_path.parent / uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix='expand-', dir=output.parent) as staging:
            for name in sorted(new_names):
                table = graph._named_tables[name]
                prepared = table.connector.source_backend.prepare_ingest(
                    table.connector, table.table_name, str(Path(staging) / name),
                    dtypes={c.name: c.dtype for c in table.definition.cols},
                    data_version=fingerprints[name].content_hash)
                sources[name] = [Path(p) for p in prepared.source['paths']]
            config = DiskgraphConfigConverter(
                prepare_diskgraph_converter_config(graph._graph_config), staging,
                {name: fp.content_hash for name, fp in fingerprints.items()},
                source_specs={name: {'source': {'type': 'parquet', 'paths': [str(p) for p in paths]},
                                     'source_size': sum(p.stat().st_size for p in paths)}
                              for name, paths in sources.items()}).convert()
            original = json.loads((parent.index_path / 'diskgraph_config.json').read_text())
            tables = {t['name']: t for t in config['tables']}
            for old in original['tables']:
                current = tables[old['name']]
                # Source locations and versions may differ; declarations,
                # key semantics, and existing links may not.
                ignored = {'source', 'source_size', 'data_version', 'fkeys'}

                def schema(table):
                    return {k: v for k, v in table.items() if k not in ignored}
                if schema(old) != schema(current):
                    return None
                if any(fk not in current.get('fkeys', []) for fk in old.get('fkeys', [])):
                    return None
            append_snapshot(json.dumps({'parent': str(parent.index_path), 'output': str(output),
                                        'tables': {}, 'expansion': config}))
            deltas = json.loads((output / '_incremental/deltas.json').read_text())
            if deltas['version'] != 3:
                raise RuntimeError('Native diskgraph does not support table expansion')
            for name in new_names:
                retain_sources(output, name, sources[name])
            for name, table in graph._named_tables.items():
                now = table.connector.source_backend.get_source_fingerprint(table.connector, table.table_name)
                if now.content_hash != fingerprints[name].content_hash:
                    raise RuntimeError(f'Source changed while expanding {name}')
        native = json.loads((output / '_incremental/expansion_stats.json').read_text())
        added = native['added']
        for key in ('node_counts', 'source_rows_by_table', 'source_bytes_by_table'):
            stats.setdefault(key, {}).update(added.get(key, {}))
        base = stats.setdefault('base_edge_counts', dict(stats['edge_counts']))
        metadata = json.loads((output / 'index_metadata.json').read_text())
        for rel in metadata['edge_relations']:
            # Initial edges of a new relation are its baseline, not append fragmentation.
            base.setdefault(rel['name'], native['edge_counts'].get(rel['name'], 0))
        counts = {edge: count for table in deltas['tables'].values()
                  for edge, count in table['edge_counts'].items()}
        stats['edge_counts'] = {edge: value + counts.get(edge, 0) for edge, value in base.items()}
        stats['total_nodes'] = sum(stats['node_counts'].values())
        stats['total_edges'] = sum(stats['edge_counts'].values())
        stats['node_tables'] = len(stats['node_counts'])
        stats['edge_relations'] = len(stats['edge_counts'])
        stats['key_lookup_failures'] += native['key_lookup_failures']
        stats['output'] = str(output)
        stats['perf'] = added['perf']
        stats['perf']['wall_time_ms'] = round(native['seconds'] * 1000)
        if 'source_config' in report:
            report['source_config']['sha256'] = hashlib.sha256((output / 'diskgraph_config.json').read_bytes()).hexdigest()
        else:
            report['source'] = config
        (output / 'ingest_stats.json').write_text(json.dumps(report))
        old_record = parent._latest_update
        snapshot_id = f'{graph.id}@{output.name}'
        record = dict(snapshot_id=snapshot_id, parent_snapshot_id=parent.id, parent=str(parent.index_path),
                      created_at=datetime.now(timezone.utc).isoformat(), creation_mode='expansion',
                      update_chain_length=old_record['update_chain_length'] + 1,
                      added_tables=sorted(new_names), rebuilt_tables=[], reused_tables=sorted(old_names),
                      reused_edges=sorted(old_record['edges']), duration_seconds=time.perf_counter() - started)
        record['tables'] = {
            name: {'status': 'added' if name in new_names else 'reused', 'rows': rows,
                   'encoder_state': 'fitted' if name in new_names else 'inherited'}
            for name, rows in stats['node_counts'].items()}
        record['edges'] = {
            name: {'status': 'reused' if name in old_record['edges'] else 'added',
                   'delta_entries': counts.get(name, 0),
                   'fragmentation': counts.get(name, 0) / value if value else None}
            for name, value in base.items()}
        record['total_fragmentation'] = sum(counts.values()) / max(1, sum(base.values()))
        (output / '_incremental/update.json').write_text(json.dumps(record))
        (output / '_incremental/source_fingerprints.json').write_text(json.dumps({
            name: asdict(replace(fp, row_count=stats['source_rows_by_table'][name]))
            for name, fp in fingerprints.items()}))
        result = DiskgraphIngestResult.for_index(output)
        write_diskgraph_success_marker(str(output), result)
        _warn_ingest_data_loss(result, graph._graph_config)
        snapshot = GraphSnapshot(snapshot_id, output, graph._graph_config, graph)
        return snapshot.compact() if record['update_chain_length'] >= 30 else snapshot
    except Exception as error:
        if output.exists():
            shutil.rmtree(output)
        if 'requires full table encoding' in str(error):
            return None
        raise
