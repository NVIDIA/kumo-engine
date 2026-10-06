# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read local CSV data, validate a query, and build a graph snapshot."""
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    expected = (Path(__file__).resolve().parents[2] /
                'packages/kumo-engine/kumo/engine/__init__.py')
    try:
        spec = find_spec('kumo.engine')
    except ModuleNotFoundError:
        spec = None
    if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
        raise SystemExit(
            'Use the in-tree local SDK as described in examples/sdk-local/README.md.')

    from diskgraph import verify_snapshot_integrity

    from kumo.engine import (
        FileConnector,
        Graph,
        PredictiveQuery,
        Table,
        configure,
    )

    with TemporaryDirectory(prefix='kumo-local-example-') as directory:
        root = Path(directory)
        source = root / 'source'
        source.mkdir()
        (source / 'users.csv').write_text(
            'id,age,churned\n'
            '1,21,true\n2,35,false\n3,48,true\n4,57,false\n',
            encoding='utf-8')
        configure(scratch_dir=str(root / 'scratch'))
        files = FileConnector(root_dir=str(source))
        users = Table.from_source_table(files['users'], primary_key='id')
        graph = Graph(tables=[users])
        graph.validate()
        query = PredictiveQuery(graph, 'PREDICT users.churned FOR EACH users.id')
        query.validate()
        if query.get_task_type() != 'binary_classification':
            raise RuntimeError('Expected a binary classification query')
        snapshot = graph.snapshot()
        verify_snapshot_integrity(str(snapshot.index_path))

    print('Local SDK: CSV ingest, query validation, and snapshot verification passed.')


if __name__ == '__main__':
    main()
