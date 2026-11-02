# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore held-out transaction rows at 0.1%, 1%, 5%, and 10% of the base."""
import argparse
import json
import shutil
import time
from pathlib import Path

import pyarrow.csv as csv
import pyarrow.parquet as pq
from bench_incremental_snapshot import KEYS, graph, link

from kumo.engine import configure

BASE_ROWS = 31_500_000


def prepare(source, output):
    data = output / 'data'
    data.mkdir(parents=True)
    for name in KEYS:
        if name != 'transactions':
            link(source / f'{name}.csv', data / f'{name}.csv')
    (data / 'transactions').mkdir()
    reader = csv.open_csv(
        source / 'transactions.csv',
        read_options=csv.ReadOptions(block_size=16 * 1024 * 1024))
    writers = [
        pq.ParquetWriter(data / 'transactions/base.parquet', reader.schema),
        pq.ParquetWriter(output / 'withheld.parquet', reader.schema)
    ]
    rows = 0
    try:
        for batch in reader:
            before = max(0, min(batch.num_rows, BASE_ROWS - rows))
            if before:
                writers[0].write_batch(batch.slice(0, before))
            if before < batch.num_rows:
                writers[1].write_batch(batch.slice(before))
            rows += batch.num_rows
    finally:
        for writer in writers:
            writer.close()
    assert rows == 35_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results_path = args.output / 'results.json'
    args.output.mkdir(parents=True, exist_ok=False)
    prepare(args.source, args.output)
    configure(scratch_dir=args.output / 'scratch')
    data = args.output / 'data'
    addition = data / 'transactions/new.parquet'
    current_graph = graph(data)
    results = {'base_transaction_rows': BASE_ROWS, 'runs': []}

    def measure(name, operation):
        record = {'scenario': name}
        start = time.perf_counter()
        try:
            snapshot = operation()
            stats = json.loads((snapshot.index_path /
                                'ingest_stats.json').read_text())['stats']
            record.update(seconds=time.perf_counter() - start, error=None,
                          id=snapshot.id, path=str(snapshot.index_path),
                          fragmentation=snapshot.fragmentation,
                          update=snapshot.update_history[-1],
                          node_counts=stats['node_counts'],
                          edge_counts=stats['edge_counts'])
            return snapshot
        except Exception as error:
            record.update(seconds=time.perf_counter() - start,
                          error=f'{type(error).__name__}: {error}')
            raise
        finally:
            results['runs'].append(record)
            results_path.write_text(json.dumps(results, indent=2))
            print(json.dumps(record), flush=True)

    base = measure('baseline',
                   lambda: current_graph.snapshot(enable_updates=True))
    for fraction in [0.001, 0.01, 0.05, 0.10]:
        remaining = int(BASE_ROWS * fraction)
        source = pq.ParquetFile(args.output / 'withheld.parquet')
        with pq.ParquetWriter(addition, source.schema_arrow) as writer:
            for batch in source.iter_batches():
                take = min(remaining, batch.num_rows)
                writer.write_batch(batch.slice(0, take))
                remaining -= take
                if remaining == 0:
                    break
        assert remaining == 0
        measure(f'append_{fraction:g}', lambda: base.update(mode='append'))
        full = measure(
            f'full_{fraction:g}', lambda: current_graph.snapshot(
                force_full_rebuild=True, enable_updates=True))
        paired = {
            r['scenario']: r
            for r in results['runs'] if r['error'] is None
        }
        a, b = paired[f'append_{fraction:g}'], paired[f'full_{fraction:g}']
        assert a['node_counts'] == b['node_counts']
        assert a['edge_counts'] == b['edge_counts']
        a['speedup'] = b['seconds'] / a['seconds']
        if full.index_path.exists():
            shutil.rmtree(full.index_path)
        b['index_removed_after_count_comparison'] = True
        results_path.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
