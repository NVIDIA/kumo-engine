# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import kumo_core.diskgraph.stores as stores
import pyarrow as pa
import pytest


class _PrimaryKeyEngine:
    def __init__(self, values: list[object], canonical_ids: set[int]) -> None:
        self.values = values
        self.canonical_ids = canonical_ids
        self.requests: list[dict] = []
        self.canonical_types: list[pa.DataType] = []

    def fetch_features(self, request: list[dict]):
        spec = request[0]
        self.requests.append(spec)
        values = [self.values[node_id] for node_id in spec['node_ids']]
        batch = pa.RecordBatch.from_arrays([pa.array(values)],
                                           names=spec['feature_names'])
        return [{'type': 'feature', 'batch': batch}]

    def canonical_node_mask(self, _table_name: str, node_ids: list[int],
                            primary_keys: pa.Array) -> list[bool]:
        assert isinstance(primary_keys, pa.Array)
        self.canonical_types.append(primary_keys.type)
        return [node_id in self.canonical_ids for node_id in node_ids]


def test_iter_primary_key_batches_is_ordered_and_bounded(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stores,
        'load_diskgraph_schema',
        lambda _: SimpleNamespace(pkey_dict={'items': 'ITEM_ID'}),
    )
    engine = _PrimaryKeyEngine(['a', 'b', 'b', 'd', 'e'], {0, 1, 3, 4})

    batches = list(
        stores.iter_primary_key_batches(
            '/index',
            'items',
            start=1,
            stop=5,
            batch_size=2,
            engine=engine,
        ))

    assert [(ids.tolist(), keys.tolist()) for ids, keys in batches] == [
        ([1], ['b']),
        ([3, 4], ['d', 'e']),
    ]
    assert [request['node_ids'] for request in engine.requests] == [[1, 2],
                                                                    [3, 4]]
    assert all(request['feature_names'] == ['ITEM_ID']
               for request in engine.requests)


@pytest.mark.parametrize('values', [[10, 20, 20, 30], ['a', 'b', 'b', 'c']])
def test_iter_primary_key_batches_preserves_sparse_canonical_ids(
        monkeypatch: pytest.MonkeyPatch, values: list[object]) -> None:
    monkeypatch.setattr(
        stores,
        'load_diskgraph_schema',
        lambda _: SimpleNamespace(pkey_dict={'items': 'ITEM_ID'}),
    )
    engine = _PrimaryKeyEngine(values, {0, 1, 3})

    batches = list(
        stores.iter_primary_key_batches('/index', 'items', stop=4, batch_size=4,
                                        engine=engine))

    assert len(batches) == 1
    node_ids, primary_keys = batches[0]
    assert node_ids.tolist() == [0, 1, 3]
    assert primary_keys.tolist() == [values[0], values[1], values[3]]


@pytest.mark.parametrize(
    ('ingested_dtype', 'precision', 'scale', 'arrow_type'),
    [
        ('decimal128', 38, 2, pa.decimal128(38, 2)),
        ('decimal256', 76, 3, pa.decimal256(76, 3)),
    ],
)
def test_iter_primary_key_batches_preserves_decimal_primary_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ingested_dtype: str,
    precision: int,
    scale: int,
    arrow_type: pa.DataType,
) -> None:
    monkeypatch.setattr(
        stores,
        'load_diskgraph_schema',
        lambda _: SimpleNamespace(pkey_dict={'items': 'ITEM_ID'}),
    )
    (tmp_path / 'diskgraph_config.json').write_text(
        json.dumps({
            'tables': [{
                'name': 'items',
                'columns': [{
                    'name': 'ITEM_ID',
                    'ingested_dtype': ingested_dtype,
                    'precision': precision,
                    'scale': scale,
                }],
            }],
        }),
        encoding='ascii',
    )
    values = ['1.00', '-2.50', '3.75']
    engine = _PrimaryKeyEngine(values, {0, 2})

    [(node_ids, primary_keys)] = list(
        stores.iter_primary_key_batches(str(tmp_path), 'items', stop=3,
                                        batch_size=3, engine=engine))

    assert node_ids.tolist() == [0, 2]
    assert primary_keys.dtype == object
    assert primary_keys.tolist() == [Decimal('1.00'), Decimal('3.75')]
    assert engine.canonical_types == [arrow_type]
