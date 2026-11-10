# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real training and scoring across expansion and process restarts."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype
from test_expansion_recovery import record, reload_snapshot
from test_table_expansion import hashes

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

QUERY = 'PREDICT COUNT(transactions.*, 0, 7, days) FOR EACH merchants.merchant_id'


def make_graph(data, expanded=False, linked=True):
    connector = kumo.FileConnector(root_dir=str(data))
    merchants = kumo.Table.from_source_table(
        connector['merchants'], primary_key='merchant_id').infer_metadata()
    transactions = kumo.Table.from_source_table(
        connector['transactions'], primary_key='transaction_id',
        time_column='transaction_at').infer_metadata()
    for name in ('merchant_id', 'device_id'):
        transactions = transactions.with_column(name, stype=Stype.ID)
    tables = dict(merchants=merchants, transactions=transactions)
    edges = [
        Edge(src_table='transactions', fkey='merchant_id',
             dst_table='merchants')
    ]
    if expanded:
        tables['devices'] = kumo.Table.from_source_table(
            connector['devices'], primary_key='device_id').infer_metadata()
        if linked:
            edges.append(
                Edge(src_table='transactions', fkey='device_id',
                     dst_table='devices'))
    return kumo.Graph(tables=tables, edges=edges)


@pytest.fixture(scope='module')
def trained_expansion(tmp_path_factory, request):
    previous = get_scratch_dir()
    request.addfinalizer(lambda: kumo.configure(scratch_dir=previous))
    root = tmp_path_factory.mktemp('expansion-model')
    data = Path(os.environ.get('KUMO_EXPANSION_TEST_DATA', root / 'data'))
    if 'KUMO_EXPANSION_TEST_DATA' not in os.environ:
        data.mkdir()
        rng = np.random.default_rng(0)
        pd.DataFrame({
            'merchant_id': np.arange(50)
        }).to_parquet(data / 'merchants.parquet')
        pd.DataFrame({
            'transaction_id':
            np.arange(3000),
            'merchant_id':
            rng.integers(0, 50, 3000),
            'device_id':
            rng.integers(0, 100, 3000),
            'transaction_at':
            pd.Timestamp('2026-01-01') +
            pd.to_timedelta(rng.integers(0, 90, 3000), unit='D')
        }).to_parquet(data / 'transactions.parquet')
        pd.DataFrame({
            'device_id': np.arange(100),
            'value': rng.random(100)
        }).to_parquet(data / 'devices.parquet')
    kumo.configure(scratch_dir=root / 'scratch')
    graph = make_graph(data)
    ordinary = graph.snapshot()
    assert not (ordinary.index_path / '_expansion').exists()
    query = kumo.PredictiveQuery(graph=graph, query=QUERY)
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    training = trainer.fit(query.generate_training_table(ordinary),
                           snapshot=ordinary)
    prediction_table = query.generate_prediction_table(ordinary)
    return root, data, trainer, training, prediction_table, ordinary


@pytest.mark.parametrize('linked', [False, True],
                         ids=['isolated-table', 'linked-table'])
@pytest.mark.parametrize('predict_first', [False, True],
                         ids=['train-then-reload', 'predict-then-reload'])
def test_model_reloads_expanded_snapshot(trained_expansion, predict_first,
                                         linked):
    root, data, trainer, training, population, ordinary = trained_expansion
    before = {ordinary.id: hashes(ordinary.index_path)}
    old_files = {}
    if predict_first:
        original = trainer.predict(population, snapshot=ordinary)
        assert original.num_rows == population.row_count > 0
        old_files = {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in original.files
        }
    parent = make_graph(data).snapshot(enable_expansion=True)
    before[parent.id] = hashes(parent.index_path)
    graph = make_graph(data, expanded=True, linked=linked)
    expanded = graph.snapshot(previous_snapshot=parent)
    assert (expanded.index_path / '_expansion/lineage.json').is_file()
    before[expanded.id] = hashes(expanded.index_path)
    directory = root / f'{predict_first}-{linked}'
    directory.mkdir()
    # Only locally generated, trusted test data is deserialized.
    with (directory / 'population.pkl').open('wb') as stream:
        pickle.dump(population, stream)
    outcomes = {}
    for kind, snapshot in [('expanded', expanded), ('full', graph.snapshot())]:
        destination = directory / kind
        destination.mkdir()
        payload = dict(snapshot=record(snapshot),
                       bundle=str(training.bundle_dir),
                       directory=str(destination),
                       population=str(directory / 'population.pkl'))
        result = subprocess.run([
            sys.executable, '-c',
            'import json,sys; from test_expansion_model_reload import reload_and_predict; reload_and_predict(json.loads(sys.argv[1]))',
            json.dumps(payload)
        ], env={
            **os.environ, 'PYTHONPATH':
            os.pathsep.join(map(str, sys.path))
        }, capture_output=True, text=True, timeout=600)
        (destination / 'reload.log').write_text(result.stdout + result.stderr)
        outcomes[kind] = dict(returncode=result.returncode,
                              error=result.stderr[-2000:])
    (directory / 'outcomes.json').write_text(json.dumps(outcomes, indent=2))
    for path, digest in old_files.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for snapshot in (ordinary, parent, expanded):
        after = hashes(snapshot.index_path)
        assert {
            name: after[name]
            for name in before[snapshot.id]
        } == before[snapshot.id]
        # Prediction adds task indexes, but must not alter any existing files.
        assert all(
            name.startswith('train_tables/')
            for name in after.keys() - before[snapshot.id].keys())
    assert all(o['returncode'] == 0 for o in outcomes.values()), outcomes
    expected = trainer.predict(population, snapshot=expanded).frame
    actual = pd.read_parquet(directory / 'expanded/predictions.parquet')
    pd.testing.assert_frame_equal(expected, actual, check_exact=False,
                                  rtol=1e-5, atol=1e-6)
    assert len(actual) == population.row_count > 0
    assert np.isfinite(actual.select_dtypes(include='number').to_numpy()).all()


def reload_and_predict(payload):
    directory = Path(payload['directory'])
    kumo.configure(scratch_dir=directory / 'scratch')
    trainer = kumo.Trainer.load(payload['bundle'])
    with Path(payload['population']).open('rb') as stream:
        population = pickle.load(stream)
    snapshot = reload_snapshot(payload['snapshot'])
    result = trainer.predict(population, snapshot=snapshot)
    assert result.num_rows == population.row_count > 0
    result.frame.to_parquet(directory / 'predictions.parquet')
