#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed release wheel; no source checkout or mocked engine."""
import argparse
import gc
import importlib.metadata
import inspect
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path


def create_graph(data, fixture):
    """Build the same generated graph for package and installed-wheel tests."""
    import numpy as np
    import pandas as pd
    from kumoapi.graph import Edge
    from kumoapi.typing import Stype

    import kumo.engine as kumo

    num_entities = fixture['entities']
    num_events = fixture['events']
    data.mkdir()
    rng = np.random.default_rng(fixture['seed'])
    pd.DataFrame({
        'ID': np.arange(num_entities)
    }).to_parquet(data / 'ents.parquet')
    pd.DataFrame({
        'EID':
        np.arange(num_events),
        'ENT_ID':
        rng.integers(0, num_entities, num_events),
        'TS':
        pd.Timestamp(fixture['start_time']) +
        pd.to_timedelta(rng.integers(0, fixture['days'], num_events), unit='D')
    }).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    events = kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='EID',
        time_column='TS').infer_metadata().with_column('ENT_ID', stype=Stype.ID)
    return kumo.Graph(
        tables={
            'ents':
            kumo.Table.from_source_table(connector['ents.parquet'],
                                         primary_key='ID').infer_metadata(),
            'events':
            events
        }, edges=[Edge(src_table='events', fkey='ENT_ID', dst_table='ents')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--measure-variation', action='store_true')
    args = parser.parse_args()
    root = Path.cwd()
    evidence = args.output_dir
    evidence.mkdir(parents=True, exist_ok=True)
    result = {'status': 'RUNNING', 'wheel_version': args.version, 'checks': {}}
    started = time.monotonic()

    def stage(name):
        result['stage'] = name
        print(name, flush=True)

    try:
        stage('import')
        requirements = json.loads(
            Path(__file__).with_suffix('.json').read_text())
        fixture = requirements['fixture']
        result.update(fixture)
        seed = fixture['seed']
        num_entities = fixture['entities']
        num_events = fixture['events']
        prob_columns = ['False_PROB', 'True_PROB']
        tolerance = {
            'atol': fixture['reload_atol'],
            'rtol': fixture['reload_rtol']
        }
        import numpy as np
        import torch
        from kumoapi.model_plan import ModelPlan

        import kumo.engine as kumo
        assert importlib.metadata.version(
            'kumo-engine') == kumo.__version__ == args.version
        assert Path(kumo.__file__).resolve().is_relative_to(Path(
            sys.prefix)), kumo.__file__
        if args.device == 'cuda':
            assert torch.cuda.is_available(
            ), 'CUDA was requested but is unavailable'
            result['gpu'] = torch.cuda.get_device_name(0)
            result['cuda_version'] = torch.version.cuda
        placement = 'gpu' if args.device == 'cuda' else 'cpu'
        # The approved 1.0.0 wheel uses the environment. Newer wheels accept
        # an instance argument. Observe Lightning's actual device in both cases.
        os.environ['KUMO_ACCELERATOR'] = placement
        trainer_options = ({
            'accelerator': placement
        } if 'accelerator' in inspect.signature(kumo.Trainer).parameters else
                           {})
        from kumo_core.train.trainer import Trainer as CoreTrainer
        original_create = CoreTrainer.create_pl_trainer
        result['observed_devices'] = []

        def observe_device(trainer):
            original_create(trainer)
            device = trainer.pl_trainer.strategy.root_device
            result['observed_devices'].append({
                'stage': result['stage'],
                'device': str(device),
            })
            assert device.type == args.device, (
                f'Requested {args.device}, observed {device}')

        CoreTrainer.create_pl_trainer = observe_device
        torch.set_num_threads(requirements['configuration']['threads'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        result['import_path'] = kumo.__file__
        result['torch_version'] = torch.__version__
        result['device'] = args.device
        result['checks']['wheel_import'] = 'PASS'
        result['python'] = sys.version
        kumo.configure(scratch_dir=root / 'scratch')

        stage('graph_snapshot')
        graph = create_graph(root / 'data', fixture)
        snapshot = graph.snapshot()
        stats_path = snapshot.index_path / 'ingest_stats.json'
        stats = json.loads(stats_path.read_text())['stats']
        (evidence / 'ingest_stats.json').write_bytes(stats_path.read_bytes())
        assert stats['total_nodes'] == num_entities + num_events, stats
        assert stats['total_edges'] == 2 * num_events, stats
        assert stats['source_rows_by_table'] == {
            'ents': num_entities,
            'events': num_events
        }, stats
        result['checks']['graph_counts_and_relationships'] = 'PASS'
        query = fixture['query']
        pquery = kumo.PredictiveQuery(graph=graph, query=query)
        result['query'] = query

        stage('training')
        training_table = pquery.generate_training_table(snapshot)
        plan = ModelPlan()
        plan.training_job.disable_explain = True
        plan.training_job.manual_seed = seed
        # These limits apply to each trial and refit, not to the whole search.
        plan.optimization.max_epochs = fixture['epochs']
        plan.optimization.max_steps_per_epoch = fixture['steps_per_epoch']
        (evidence / 'model-plan.json').write_text(
            json.dumps(plan.as_json(), indent=2))
        trainer = kumo.Trainer(model_plan=plan, **trainer_options)

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

        stage('prediction')
        prediction_table = pquery.generate_prediction_table(snapshot)
        assert prediction_table.row_count == num_entities, prediction_table.row_count

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        predicted = trainer.predict(
            prediction_table, snapshot=snapshot,
            binary_classification_threshold=fixture['threshold'])
        assert predicted.num_rows == num_entities
        before = predicted.frame.sort_values('ENTITY').reset_index(drop=True)
        before.to_parquet(evidence / 'predictions-before.parquet', index=False)
        result['checks']['prediction_execution'] = 'PASS'
        frames = {'before': before}
        if args.measure_variation:
            stage('repeat_prediction')
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            repeated = trainer.predict(
                prediction_table, snapshot=snapshot,
                binary_classification_threshold=fixture['threshold']
            ).frame.sort_values('ENTITY').reset_index(drop=True)
            repeated.to_parquet(evidence / 'predictions-repeat.parquet',
                                index=False)
            frames['repeat'] = repeated
        del trainer
        gc.collect()

        stage('reload_and_predict')
        restored = kumo.Trainer.load(saved, **trainer_options)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        after = restored.predict(
            prediction_table, snapshot=snapshot,
            binary_classification_threshold=fixture['threshold']
        ).frame.sort_values('ENTITY').reset_index(drop=True)
        after.to_parquet(evidence / 'predictions-after.parquet', index=False)
        frames['after'] = after
        for frame in frames.values():
            assert list(
                frame.columns) == ['ENTITY', 'TARGET_PRED',
                                   *prob_columns], list(frame.columns)
            assert len(frame) == num_entities and frame['ENTITY'].is_unique
            assert frame['ENTITY'].tolist() == list(
                range(num_entities)), frame['ENTITY'].tolist()
            probs = frame[prob_columns].to_numpy()
            assert np.isfinite(probs).all()
            assert ((probs >= 0) & (probs <= 1)).all()
            np.testing.assert_allclose(probs.sum(axis=1), 1, **tolerance)
            np.testing.assert_array_equal(
                frame['TARGET_PRED'], frame['True_PROB'] > fixture['threshold'])
        result['checks']['prediction_rows_and_values'] = 'PASS'
        before_probs = before[prob_columns].to_numpy()
        after_probs = after[prob_columns].to_numpy()
        result['max_reload_probability_difference'] = float(
            np.abs(before_probs - after_probs).max())
        if args.measure_variation:
            result['predictions'] = {
                name: {
                    'entity_ids': frame['ENTITY'].tolist(),
                    'probabilities': frame[prob_columns].to_numpy().tolist(),
                    'labels': frame['TARGET_PRED'].tolist(),
                }
                for name, frame in frames.items()
            }
            result['checks']['reload_predictions_match'] = 'NOT_EVALUATED'
            result['status'] = 'MEASURED'
        else:
            np.testing.assert_allclose(before_probs, after_probs, **tolerance)
            np.testing.assert_array_equal(before['TARGET_PRED'],
                                          after['TARGET_PRED'])
            result['checks']['reload_predictions_match'] = 'PASS'
            result['status'] = 'PASS'
        stage('complete')
    except Exception as exc:
        result['status'] = 'FAIL'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        traceback.print_exc()
    finally:
        if 'original_create' in locals():
            CoreTrainer.create_pl_trainer = original_create
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (evidence / 'workflow-result.json').write_text(
            json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result['status'] in ('PASS', 'MEASURED') else 1


if __name__ == '__main__':
    sys.exit(main())
