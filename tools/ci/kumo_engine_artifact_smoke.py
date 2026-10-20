#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed release wheel; no source checkout or mocked engine."""
import argparse
import gc
import importlib.metadata
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--version', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
root = Path.cwd()
evidence = args.output_dir
evidence.mkdir(parents=True, exist_ok=True)
result = {
    'status': 'RUNNING',
    'wheel_version': args.version,
    'seed': 0,
    'entities': 50,
    'events': 3000,
    'epochs': 1,
    'steps_per_epoch': 2,
    'reload_atol': 1e-6,
    'reload_rtol': 1e-5,
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'checks': {},
    'stage_seconds': {}
}
started = time.monotonic()
last_stage = started


def stage(name):
    global last_stage
    now = time.monotonic()
    if 'stage' in result:
        result['stage_seconds'][result['stage']] = round(now - last_stage, 3)
    last_stage = now
    result['stage'] = name
    (evidence / 'workflow-result.json').write_text(
        json.dumps(result, indent=2, default=str))
    print(name, flush=True)


try:
    stage('import')
    import numpy as np
    import pandas as pd
    import torch
    from kumoapi.graph import Edge
    from kumoapi.model_plan import ModelPlan
    from kumoapi.typing import Stype

    import kumo.engine as kumo
    assert importlib.metadata.version(
        'kumo-engine') == kumo.__version__ == args.version
    assert Path(kumo.__file__).resolve().is_relative_to(Path(
        sys.prefix)), kumo.__file__
    assert not torch.cuda.is_available(), 'This test is specifically CPU-only'
    torch.set_num_threads(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    result['import_path'] = kumo.__file__
    result['torch_version'] = torch.__version__
    result['device'] = 'cpu'
    result['checks']['wheel_import'] = 'PASS'
    result['python'] = sys.version
    kumo.configure(scratch_dir=root / 'scratch')

    stage('create_synthetic_data')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
    pd.DataFrame({
        'EID':
        np.arange(3000),
        'ENT_ID':
        rng.integers(0, 50, 3000),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 3000), unit='D')
    }).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    events = kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='EID',
        time_column='TS').infer_metadata().with_column('ENT_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'ents':
            kumo.Table.from_source_table(connector['ents.parquet'],
                                         primary_key='ID').infer_metadata(),
            'events':
            events
        }, edges=[Edge(src_table='events', fkey='ENT_ID', dst_table='ents')])

    stage('graph_snapshot')
    snapshot = graph.snapshot()
    stats_path = snapshot.index_path / 'ingest_stats.json'
    stats = json.loads(stats_path.read_text())['stats']
    (evidence / 'ingest_stats.json').write_bytes(stats_path.read_bytes())
    assert stats['total_nodes'] == 3050, stats
    assert stats['total_edges'] == 6000, stats
    assert stats['source_rows_by_table'] == {'ents': 50, 'events': 3000}, stats
    result['checks']['graph_counts_and_relationships'] = 'PASS'
    query = 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH ents.ID'
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    result['query'] = query

    stage('training_table')
    training_table = pquery.generate_training_table(snapshot)
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.training_job.manual_seed = 0
    # These limits apply to each trial and refit, not to the whole search.
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)

    stage('training')
    fitted = trainer.fit(training_table, snapshot=snapshot)
    result['training_metrics'] = fitted.metrics
    assert fitted.bundle_dir.is_dir()
    result['checks']['training'] = 'PASS'

    stage('save_model')
    saved = root / 'saved-model'
    shutil.copytree(fitted.bundle_dir, saved)
    assert (saved / 'training-spec.json').is_file()
    assert (saved / 'publication-manifest.json').is_file()
    result['checks']['saved_bundle'] = 'PASS'

    stage('prediction_table')
    prediction_table = pquery.generate_prediction_table(snapshot)
    assert prediction_table.row_count == 50, prediction_table.row_count

    stage('prediction')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    predicted = trainer.predict(prediction_table, snapshot=snapshot,
                                binary_classification_threshold=0.5)
    assert predicted.num_rows == 50
    before = predicted.frame.sort_values('ENTITY').reset_index(drop=True)
    before.to_parquet(evidence / 'predictions-before.parquet', index=False)
    result['checks']['prediction_execution'] = 'PASS'
    del trainer
    gc.collect()

    stage('reload_and_predict')
    restored = kumo.Trainer.load(saved)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    after = restored.predict(
        prediction_table, snapshot=snapshot,
        binary_classification_threshold=0.5).frame.sort_values(
            'ENTITY').reset_index(drop=True)
    after.to_parquet(evidence / 'predictions-after.parquet', index=False)
    for frame in (before, after):
        assert list(frame.columns) == [
            'ENTITY', 'TARGET_PRED', 'False_PROB', 'True_PROB'
        ], list(frame.columns)
        assert len(frame) == 50 and frame['ENTITY'].is_unique
        assert frame['ENTITY'].tolist() == list(
            range(50)), frame['ENTITY'].tolist()
        probs = frame[['False_PROB', 'True_PROB']].to_numpy()
        assert np.isfinite(probs).all()
        assert ((probs >= 0) & (probs <= 1)).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(frame['TARGET_PRED'], frame['True_PROB']
                                      > 0.5)
    result['checks']['prediction_rows_and_values'] = 'PASS'
    np.testing.assert_allclose(before[['False_PROB', 'True_PROB']],
                               after[['False_PROB',
                                      'True_PROB']], atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(before['TARGET_PRED'], after['TARGET_PRED'])
    result['max_reload_probability_difference'] = float(
        np.abs(before[['False_PROB', 'True_PROB']].to_numpy() -
               after[['False_PROB', 'True_PROB']].to_numpy()).max())
    result['checks']['reload_predictions_match'] = 'PASS'
    result['status'] = 'PASS'
    stage('complete')
except Exception as exc:
    result['status'] = 'FAIL'
    result['error'] = type(exc).__name__ + ': ' + str(exc)
    traceback.print_exc()
finally:
    result['finished_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                              time.gmtime())
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    (evidence / 'workflow-result.json').write_text(
        json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)
raise SystemExit(0 if result['status'] == 'PASS' else 1)
