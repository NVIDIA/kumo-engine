# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.graph import GraphSnapshot


@pytest.fixture
def staged_seed_dir(tmp_path):
    """Supply seed metadata for adapter tests that mock materialization."""
    from kumo_core.training.export import INGEST_SPEC

    staging_dir = tmp_path / 'staging'
    staging_dir.mkdir()
    seed_part = 'seed.parquet'
    # Preflight needs the normalized schema; these tests mock row ingestion.
    schema = pa.schema([('ENTITY', pa.int64()), ('TARGET', pa.float64())])
    pq.write_table(schema.empty_table(), staging_dir / seed_part)
    (staging_dir / INGEST_SPEC).write_text(
        json.dumps({
            'split_stages': [{
                'key': 'train',
                'parts': [seed_part]
            }],
        }))
    return staging_dir


@pytest.fixture
def local_s3_object_identity(monkeypatch):
    """Give local-filesystem S3 doubles content-sensitive provider tokens.

    Local NativeFiles have no S3 metadata. Real Arrow metadata is covered by
    kumo-s3's HTTP endpoint tests; these engine tests exercise the callers.
    """
    probes = []

    def identity(filesystem, path):
        with filesystem.open_input_file(path) as stream:
            contents = stream.read()
        probes.append(path)
        return (hashlib.sha256(contents).hexdigest(), '', str(len(contents)))

    monkeypatch.setattr('kumo_s3.access._object_identity', identity)
    return probes


@pytest.fixture
def snapshot(tmp_path):
    config = {
        'tables': [{
            'table_name':
            'users',
            'source_table_name':
            'db.schema.users',
            'pkey': {
                'name': 'id',
                'dtype': 'int64',
                'stype': 'ID'
            },
            'cols': [
                {
                    'name': 'label',
                    'dtype': 'bool',
                    'stype': 'categorical'
                },
                {
                    'name': 'age',
                    'dtype': 'int64',
                    'stype': 'numerical'
                },
            ],
            'fkeys': [],
        }]
    }
    return GraphSnapshot('snapshot', tmp_path / 'index', config)


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('preflight')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    try:
        rng = np.random.default_rng(0)
        data = root / 'data'
        data.mkdir()
        pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
        pd.DataFrame({
            'EID':
            np.arange(3000),
            'ENT_ID':
            rng.integers(0, 50, 3000),
            'TS':
            pd.Timestamp('2024-01-01') +
            pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
        }).to_parquet(data / 'events.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        events = kumo.Table.from_source_table(
            connector['events.parquet'], primary_key='EID',
            time_column='TS').infer_metadata().with_column(
                'ENT_ID', stype=Stype.ID)
        graph = kumo.Graph(
            tables={
                'ents':
                kumo.Table.from_source_table(connector['ents.parquet'],
                                             primary_key='ID').infer_metadata(),
                'events':
                events,
            },
            edges=[Edge(src_table='events', fkey='ENT_ID', dst_table='ents')])
        snapshot = graph.snapshot()
        pquery = kumo.PredictiveQuery(
            graph=graph, query='PREDICT COUNT(events.*, 0, 7, days) > 0 '
            'FOR EACH ents.ID')
        plan = ModelPlan()
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        trainer = kumo.Trainer(model_plan=plan)
        trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
        yield trainer, pquery.generate_prediction_table(snapshot), snapshot
    finally:
        kumo.configure(scratch_dir=previous)
