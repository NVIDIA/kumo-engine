# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from runpy import run_path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.model_plan import ModelPlan

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
        settings = Path(__file__).with_name('wheel_workflow.json')
        fixture = json.loads(settings.read_text())['fixture']
        graph = run_path(str(settings.with_suffix('.py')))['create_graph'](
            root / 'data', fixture)
        snapshot = graph.snapshot()
        pquery = kumo.PredictiveQuery(graph=graph, query=fixture['query'])
        plan = ModelPlan()
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = fixture['epochs']
        plan.optimization.max_steps_per_epoch = fixture['steps_per_epoch']
        trainer = kumo.Trainer(model_plan=plan)
        trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
        yield trainer, pquery.generate_prediction_table(snapshot), snapshot
    finally:
        kumo.configure(scratch_dir=previous)
