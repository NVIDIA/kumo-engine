# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact source comparison and immutable replacements for feature-only edits."""
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def retain_sources(output, name, paths, *, append=False):
    destination = output / '_incremental/sources' / name
    start = len(list(destination.glob('*.parquet'))) if append else 0
    if destination.exists() and not append:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start):
        target = destination / f'{index:08d}.parquet'
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)


def _changed_rows(before, after, structural, target):
    """Compare aligned batches, independent of file/row-group boundaries."""
    old, new = [ds.dataset(paths, format='parquet') for paths in (before, after)]
    if old.schema != new.schema or old.count_rows() != new.count_rows():
        return None
    previous = iter(old.to_batches(batch_size=65536))
    current = iter(new.to_batches(batch_size=65536))
    left = right = None
    count = 0
    with pq.ParquetWriter(target, new.schema) as writer:
        while True:
            if left is None or not left.num_rows:
                left = next(previous, None)
            if right is None or not right.num_rows:
                right = next(current, None)
            if left is None or right is None:
                return count if left is None and right is None else None
            size = min(left.num_rows, right.num_rows)
            a, b = left.slice(0, size), right.slice(0, size)
            if not a.select(structural).equals(b.select(structural)):
                return None
            changed = pa.array([False] * size)
            for name in new.schema.names:
                if name in structural:
                    continue
                x, y = a.column(name), b.column(name)
                try:
                    same = pc.or_(pc.fill_null(pc.equal(x, y), False),
                                  pc.and_(pc.is_null(x), pc.is_null(y)))
                except pa.ArrowNotImplementedError:
                    same = pa.array([u.equals(v) for u, v in zip(x, y)])
                changed = pc.or_(changed, pc.invert(same))
            rows = b.filter(changed)
            if rows.num_rows:
                writer.write_batch(rows)
                count += rows.num_rows
            left, right = left.slice(size), right.slice(size)


def try_feature_update(graph, parent, fingerprints):
    from diskgraph import append_snapshot
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult, write_diskgraph_success_marker,
    )
    from kumo.engine.graph import GraphSnapshot

    started = time.perf_counter()
    previous = json.loads((parent.index_path / '_incremental/source_fingerprints.json').read_text())
    changed = [name for name, fp in fingerprints.items()
               if fp.content_hash != previous[name]['content_hash']]
    if not changed:
        return None
    retained = {name: sorted((parent.index_path / '_incremental/sources' / name).glob('*.parquet'))
                for name in changed}
    if not all(retained.values()):
        return None
    stats = json.loads((parent.index_path / 'ingest_stats.json').read_text())['stats']
    if stats.get('pk_duplicate_total', 0) or any(stats.get('dropped_rows_counts', {}).values()):
        return None
    config = json.loads((parent.index_path / 'diskgraph_config.json').read_text())
    cache = json.loads((parent.index_path / '_incremental/manifest.json').read_text())
    if cache['num_shards'] != config['num_shards']:
        return None
    specs = {table['name']: table for table in config['tables']}
    structural = {}
    for name in changed:
        spec = specs[name]
        if not spec.get('primary_key') or spec.get('edge_table'):
            return None
        structural[name] = {spec['primary_key']}
        structural[name].update(fk['src_column'] for fk in spec.get('fkeys', []))
        structural[name].update(c['name'] for c in spec['columns']
                                if c['ingested_dtype'] in ('timestamp', 'date'))
        structural[name].update(spec[k] for k in ('create_time', 'weight') if spec.get(k))
    for table in specs.values():
        for fk in table.get('fkeys', []):
            for direction in ('forward', 'reverse'):
                for role in ('time', 'weight'):
                    ref = (fk.get(direction) or {}).get(role)
                    if ref and ref['table'] in structural:
                        structural[ref['table']].add(ref['column'])
    output = parent.index_path.parent / uuid4().hex
    replacements, sources, counts = {}, {}, {}
    try:
        with tempfile.TemporaryDirectory(prefix='features-', dir=output.parent) as staging:
            for name in changed:
                table = graph._named_tables[name]
                prepared = table.connector.source_backend.prepare_ingest(
                    table.connector, table.table_name, str(Path(staging) / name),
                    dtypes={c.name: c.dtype for c in table.definition.cols},
                    data_version=fingerprints[name].content_hash)
                sources[name] = prepared.source['paths']
                target = Path(staging) / f'{name}.parquet'
                counts[name] = _changed_rows(retained[name], sources[name], sorted(structural[name]), target)
                if counts[name] is None:
                    return None
                if counts[name]:
                    replacements[name] = [str(target)]
            for name, table in graph._named_tables.items():
                now = table.connector.source_backend.get_source_fingerprint(table.connector, table.table_name)
                if now.content_hash != fingerprints[name].content_hash:
                    raise RuntimeError(f'Source changed while staging {name}')
            from kumo.engine.incremental import _warn_append_drift

            for name, paths in replacements.items():
                _warn_append_drift(parent, name, paths)
            append_snapshot(json.dumps({'parent': str(parent.index_path), 'output': str(output),
                                        'tables': {}, 'replacements': replacements}))
            if replacements and json.loads((output / '_incremental/deltas.json').read_text())['version'] not in (2, 3):
                raise RuntimeError('Native diskgraph does not support feature replacements')
            for name, paths in sources.items():
                retain_sources(output, name, paths)
        record_path = output / '_incremental/update.json'
        record = json.loads(record_path.read_text())
        snapshot_id = f'{graph.id}@{output.name}'
        record.update(snapshot_id=snapshot_id, parent_snapshot_id=parent.id,
                      parent=str(parent.index_path), creation_mode='feature_update',
                      created_at=datetime.now(timezone.utc).isoformat(),
                      update_chain_length=record['update_chain_length'] + 1,
                      rebuilt_tables=[], reused_tables=sorted(set(specs) - set(changed)),
                      replaced_rows=counts, duration_seconds=time.perf_counter() - started)
        record['reused_edges'] = sorted(stats['edge_counts'])
        for name, table in record['tables'].items():
            table.update(status='features_updated' if name in changed else 'reused',
                         rows_added=0, rows_replaced=counts.get(name, 0))
        for edge in record.get('edges', {}).values():
            edge['status'] = 'reused'
        record_path.write_text(json.dumps(record))
        (output / '_incremental/source_fingerprints.json').write_text(json.dumps(
            {name: asdict(replace(fp, row_count=previous[name]['row_count']))
             for name, fp in fingerprints.items()}))
        write_diskgraph_success_marker(str(output), DiskgraphIngestResult.for_index(output))
        result = GraphSnapshot(snapshot_id, output, graph._graph_config, graph)
        return result.compact() if record['update_chain_length'] >= 30 else result
    except Exception as error:
        if output.exists():
            shutil.rmtree(output)
        if any(reason in str(error) for reason in (
                'replacement key missing', 'duplicate replacement key',
                'requires full table encoding')):
            return None
        raise
