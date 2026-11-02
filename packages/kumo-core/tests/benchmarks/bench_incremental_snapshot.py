# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare real snapshot builds using the supplied fraud_dataset_50m CSVs."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
from kumoapi.typing import Stype

from kumo.engine import Edge, FileConnector, Graph, Table, configure

KEYS = dict(customers='customer_id', accounts='account_id', cards='card_id',
            merchants='merchant_id', devices='device_id',
            transactions='transaction_id', login_events='login_event_id')
TIMES = dict(customers='joined_date', accounts='opened_date',
             cards='issued_date', merchants='onboarded_date',
             devices='first_seen_at', transactions='transaction_at',
             login_events='event_at')
RELATIONS = {
    'accounts': [('customer_id', 'customers')],
    'cards': [('account_id', 'accounts')],
    'devices': [('registered_customer_id', 'customers')],
    'transactions': [('card_id', 'cards'), ('account_id', 'accounts'),
                     ('merchant_id', 'merchants'), ('device_id', 'devices')],
    'login_events': [('account_id', 'accounts'), ('device_id', 'devices')],
}


def link(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def graph(data):
    connector = FileConnector(root_dir=str(data))
    tables = {}
    for name in connector.table_names():
        table = Table.from_source_table(connector[name], primary_key=KEYS[name],
                                        time_column=TIMES[name])
        for column, _ in RELATIONS.get(name, []):
            table = table.with_column(column, stype=Stype.ID)
        tables[name] = table
    edges = [
        Edge(src_table=src, fkey=key, dst_table=dst)
        for src, links in RELATIONS.items() for key, dst in links
        if src in tables and dst in tables
    ]
    return Graph(tables=tables, edges=edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--storage', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / 'data'
    data.mkdir()
    configure(scratch_dir=args.output / 'scratch')
    for name in KEYS:
        if name not in ('devices', 'merchants'):
            link(args.source / f'{name}.csv', data / f'{name}.csv')
    merchants = csv.read_csv(args.source / 'merchants.csv')
    values = merchants['business_size'].to_pylist()
    values[:5] = [None] * 5
    hidden = merchants.set_column(
        merchants.schema.get_field_index('business_size'), 'business_size',
        pa.array(values, type=pa.string()))
    pq.write_table(hidden, data / 'merchants.parquet')
    results = {'storage': args.storage, 'source': str(args.source), 'runs': []}

    def measure(name, current_graph, **kwargs):
        started = time.perf_counter()
        record = {'scenario': name}
        try:
            snapshot = current_graph.snapshot(enable_updates=True, **kwargs)
            record.update(seconds=time.perf_counter() - started, id=snapshot.id,
                          path=str(snapshot.index_path), error=None)
            record['update'] = json.loads(
                (snapshot.index_path / '_incremental/update.json').read_text())
            record['stats'] = json.loads(
                (snapshot.index_path /
                 'ingest_stats.json').read_text())['stats']
            return snapshot
        except Exception as error:
            record.update(seconds=time.perf_counter() - started,
                          error=f'{type(error).__name__}: {error}')
            raise
        finally:
            results['runs'].append(record)
            (args.output / 'results.json').write_text(
                json.dumps(results, indent=2))
            print(json.dumps({
                k: v
                for k, v in record.items() if k != 'stats'
            }), flush=True)

    current_graph = graph(data)
    base = measure('baseline_devices_and_five_values_withheld', current_graph)
    unchanged = measure('unchanged', current_graph)
    assert base.index_path == unchanged.index_path
    pq.write_table(merchants, data / 'merchants.parquet')
    changed = measure('restore_five_values_incremental', current_graph,
                      previous_snapshot=base)
    assert results['runs'][-1]['update']['rebuilt_tables'] == ['merchants']
    for name in [
            'customers', 'accounts', 'cards', 'transactions', 'login_events'
    ]:
        relative = Path('_incremental/encode/_tmp/nodes') / name
        for path in (base.index_path / relative).rglob('*.bin'):
            copied = changed.index_path / path.relative_to(base.index_path)
            assert path.stat().st_ino == copied.stat().st_ino, path
    measure('same_values_full_rebuild', current_graph, force_full_rebuild=True)
    link(args.source / 'devices.csv', data / 'devices.csv')
    current_graph = graph(data)
    measure('restore_devices_table_incremental', current_graph,
            previous_snapshot=changed)


if __name__ == '__main__':
    main()
