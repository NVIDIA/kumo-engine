"""Tiny native saved/reloaded-model gate for both walk execution paths."""
import argparse
import hashlib
import importlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import kumo.engine as kumo
from kumo.engine.evidence import EvidenceConfig, WalkAttributionConfig
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype


def run(root):
    torch.set_num_threads(2)
    kumo.configure(scratch_dir=root / 'scratch')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(72)
    values = rng.uniform(1, 20, 64)
    pd.DataFrame({
        'id': np.arange(64),
        'quantity': values,
        'target': 3 * values
    }).to_parquet(data / 'entities.parquet')
    pd.DataFrame({
        'id': np.arange(256),
        'entity_id': np.tile(np.arange(64), 4),
        'units': rng.uniform(1, 20, 256)
    }).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    entity = kumo.Table.from_source_table(connector['entities.parquet'],
                                          primary_key='id').infer_metadata()
    entity = entity.with_column('quantity', stype=Stype.numerical).with_column(
        'target', stype=Stype.numerical)
    event = kumo.Table.from_source_table(connector['events.parquet'],
                                         primary_key='id').infer_metadata()
    event = event.with_column('entity_id', stype=Stype.ID).with_column(
        'units', stype=Stype.numerical)
    graph = kumo.Graph(
        tables={
            'entities': entity,
            'events': event
        }, edges=[
            kumo.Edge(src_table='events', fkey='entity_id',
                      dst_table='entities')
        ])
    snapshot = graph.snapshot()
    query = kumo.PredictiveQuery(
        graph=graph, query='PREDICT entities.target FOR EACH entities.id')
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.training_job.refit_trainval = False
    plan.training_job.refit_full = False
    plan.training_job.manual_seed = 72
    plan.optimization.loss = ['mse']
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    plan.optimization.batch_size = [16]
    plan.neighbor_sampling.num_neighbors = [[8]]
    plan.model_architecture.channels = [16]
    plan.model_architecture.activation = ['relu']
    plan.model_architecture.normalization = [None]
    plan.model_architecture.aggregation = [['sum']]
    trained = kumo.Trainer(plan, accelerator='cpu').fit(
        query.generate_training_table(snapshot), snapshot=snapshot)
    saved = root / 'model'
    shutil.copytree(trained.bundle_dir, saved)
    restored = kumo.Trainer.load(saved, accelerator='cpu')
    pd.DataFrame({'ENTITY1': [0, 1]}).to_parquet(data / 'score.parquet')
    ordinary = restored.predict(snapshot=snapshot,
                                entities=connector['score.parquet'],
                                batch_size=1)
    config = WalkAttributionConfig(top_k=10000, max_walks=100000)
    explained = restored.predict(
        snapshot=snapshot, entities=connector['score.parquet'], batch_size=1,
        evidence=EvidenceConfig(numerical_sensitivity=None,
                                walk_attribution=config))
    pd.testing.assert_frame_equal(ordinary.frame,
                                  explained.frame[ordinary.frame.columns],
                                  check_exact=False, rtol=1e-6, atol=1e-6)
    bundle = explained.evidence.export(root / 'evidence')
    originals = list(bundle.iter_numerical_evidence())
    assert len(originals) == 2
    offline = bundle.attribute_walks(model=restored, config=config,
                                     output=root / 'offline')
    derived = list(offline.iter_rows())
    assert len(derived) == 2
    for original, replay in zip(originals, derived):
        live = original['walk_attribution']
        assert live['status'] in ('available', 'zero_signal'), live
        assert replay['status'] == live['status'], replay
        assert replay['prediction_id'] == original['prediction_id']
        assert replay['origin'] == 'retained_graph_recomputation'
        assert live['origin'] == 'original_forward_attribution'
        assert replay['walks'] == live['walks']
        assert abs(live['accounting_error']) < 1e-5
    return {
        'rows': len(originals),
        'both_paths_match': True,
        'saved_reloaded': True,
        'prediction_parity': True,
        'scope': 'CPU scalar MSE; one worker and one seed per batch'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.development:
        if args.wheel is None:
            parser.error('--wheel required for installed acceptance')
        with zipfile.ZipFile(args.wheel) as archive:
            for name in ('kumo.engine.evidence.walks',
                         'kumo_core.explain.walk_attribution'):
                path = Path(importlib.import_module(name).__file__).resolve()
                assert path.is_relative_to(Path(sys.prefix).resolve())
                assert path.read_bytes() == archive.read(
                    name.replace('.', '/') + '.py')
    report = run(args.output)
    report.update(
        installed_wheel_acceptance=not args.development,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
