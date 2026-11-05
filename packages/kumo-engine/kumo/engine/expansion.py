# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable additions of tables and their relationships."""
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4


def source_fingerprints(graph):
    """Exact local-source identities; reject files changed during the scan."""
    from kumo.engine.connector.file import FileConnector

    fingerprints = {}
    for alias, table in graph._named_tables.items():
        if not isinstance(table.connector, FileConnector):
            raise ValueError('Table expansion requires local FileConnector sources')
        reader, name = table.connector.source_backend._reader(table.connector, table.table_name)
        paths = sorted(Path(p).resolve() for p in reader.object_paths(name))
        signatures = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths}
        root = Path(table.connector.config.root_dir).resolve()
        contents = []
        for path in paths:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            contents.append((path.relative_to(root).as_posix(), digest))
        if (paths != sorted(Path(p).resolve() for p in reader.object_paths(name))
                or any((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) != stat
                       for p, stat in signatures.items())):
            raise RuntimeError(f'Source changed while fingerprinting {alias}')
        fingerprints[alias] = hashlib.sha256(json.dumps(contents).encode()).hexdigest()
    return fingerprints


def retain_sources(output, name, paths):
    destination = output / '_expansion/sources' / name
    destination.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        target = destination / f'{index:08d}.parquet'
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)


def try_expansion(graph, parent, fingerprints):
    from diskgraph import DiskgraphConfigConverter, expand_snapshot
    from kumo_core.diskgraph.ingest import prepare_diskgraph_converter_config
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        write_diskgraph_success_marker,
    )

    from kumo.engine.graph import GraphSnapshot, _warn_ingest_data_loss

    started = time.perf_counter()
    previous = json.loads((parent.index_path / '_expansion/source_fingerprints.json').read_text())
    old_names = set(previous)
    new_names = set(fingerprints) - old_names
    if not new_names or not old_names <= fingerprints.keys():
        return None
    if any(fingerprints[name] != previous[name] for name in old_names):
        return None
    report = json.loads((parent.index_path / 'ingest_stats.json').read_text())
    stats = report['stats']
    if (stats.get('pk_duplicate_total', 0) or stats.get('key_parse_failures', 0)
            or any(stats.get('dropped_rows_counts', {}).values())
            or any(stats.get('pk_null_timestamp_drops', {}).values())):
        return None
    sources = {name: sorted((parent.index_path / '_expansion/sources' / name).glob('*.parquet'))
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
                    data_version=fingerprints[name])
                sources[name] = [Path(p) for p in prepared.source['paths']]
            config = DiskgraphConfigConverter(
                prepare_diskgraph_converter_config(graph._graph_config), staging,
                fingerprints,
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
            expand_snapshot(json.dumps({'parent': str(parent.index_path), 'output': str(output),
                                        'expansion': config}))
            for name in new_names:
                retain_sources(output, name, sources[name])
            if source_fingerprints(graph) != fingerprints:
                raise RuntimeError('Source changed while expanding graph')
        native = json.loads((output / '_expansion/stats.json').read_text())
        added = native['added']
        for key in ('node_counts', 'source_rows_by_table', 'source_bytes_by_table'):
            stats.setdefault(key, {}).update(added.get(key, {}))
        stats['edge_counts'].update(native['edge_counts'])
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
        (output / '_expansion/source_fingerprints.json').write_text(json.dumps(fingerprints))
        (output / '_expansion/lineage.json').write_text(json.dumps({
            'parent_snapshot_id': parent.id, 'added_tables': sorted(new_names),
            'reused_tables': sorted(old_names), 'duration_seconds': time.perf_counter() - started,
        }))
        result = DiskgraphIngestResult.for_index(output)
        write_diskgraph_success_marker(str(output), result)
        _warn_ingest_data_loss(result, graph._graph_config)
        return GraphSnapshot(f'{graph.id}@{output.name}', output, graph._graph_config)
    except Exception as error:
        if output.exists():
            shutil.rmtree(output)
        if 'requires full table encoding' in str(error):
            return None
        raise
