# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import os.path as osp
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq


def write_v3_bundle(
    parent: str,
    tt_id: str,
    *,
    split_stages=None,
    declared_parts=(),
    rows: int = 3,
    ingest_spec_schema_version: str | None = 'diskgraph.tt_ingest_spec.v3',
) -> str:
    staging = osp.join(parent, f'.{tt_id}.staging')
    os.makedirs(staging, exist_ok=True)
    table = pa.table({
        'ENTITY1': pa.array(range(rows), type=pa.int64()),
        'TIMESTAMP': pa.array([0] * rows, type=pa.int64()),
        'TARGET1': pa.array([1] * rows, type=pa.int64()),
    })
    train_table = osp.join(staging, 'tt.parquet')
    os.makedirs(train_table, exist_ok=True)
    train_table_part = osp.join(train_table, 'part-00000.parquet')
    pq.write_table(table, train_table_part)
    for part in declared_parts:
        path = osp.join(staging, part)
        os.makedirs(osp.dirname(path), exist_ok=True)
        pq.write_table(table.slice(0, min(1, rows)), path)

    stats_path = osp.join(staging, 'stats.json')
    with open(stats_path, 'w') as output:
        json.dump({
            'version': 2,
            'kind': 'tt_stats',
            'num_total_rows': rows,
            'num_train_rows': rows,
            'num_entities': rows,
            'num_targets': 1,
            'num_links': None,
        }, output)
    with open(stats_path, 'rb') as source:
        stats_sha = hashlib.sha256(source.read()).hexdigest()
    train_table_size = os.stat(train_table_part).st_size
    with open(osp.join(staging, 'manifest.json'), 'w') as output:
        json.dump({
            'version': 3,
            'row_count': rows,
            'source_mode': 'pquery',
            'split_mode': 'none',
            'train_table': {
                'path': 'tt.parquet',
                'format': 'tt_parquet_dataset_v1',
                'row_count': rows,
                'size_bytes': train_table_size,
                'parts': [{
                    'path': 'tt.parquet/part-00000.parquet',
                    'row_count': rows,
                    'size_bytes': train_table_size,
                }],
            },
            'stats': {'path': 'stats.json', 'format': 'tt_stats_json'},
            'checksums': {'stats_sha256': stats_sha},
        }, output)
    normalized_stages = deepcopy(split_stages)
    for stage in normalized_stages or []:
        if 'key' not in stage and 'stage' in stage:
            stage['key'] = stage.pop('stage')
        parts = stage.get('parts', [])
        if 'row_count' not in stage:
            stage['row_count'] = sum(
                pq.ParquetFile(osp.join(staging, part)).metadata.num_rows
                for part in parts)
        if 'artifact_inventory' not in stage:
            stage['artifact_inventory'] = {
                'schema_version':
                    'diskgraph.tt_stage_artifact_inventory.v1',
                'parts': [{
                    'path': part,
                    'size_bytes': os.stat(osp.join(staging, part)).st_size,
                    'row_count': pq.ParquetFile(
                        osp.join(staging, part)).metadata.num_rows,
                } for part in parts],
            }
    ingest_spec = {'tt_id': tt_id, 'split_stages': normalized_stages}
    if ingest_spec_schema_version is not None:
        ingest_spec['schema_version'] = ingest_spec_schema_version
    with open(osp.join(staging, 'tt_ingest_spec.json'), 'w') as output:
        json.dump(ingest_spec, output)
    return staging
