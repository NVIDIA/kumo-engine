#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute repository-owned weekly work against one installed wheel."""

import argparse
import gc
import importlib.metadata
import importlib.util
import inspect
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    'weekly',
    Path(__file__).with_name('weekly_check.py'))
weekly = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(weekly)


def check_gpu_class(expected, actual):
    if (actual['gpu_count'] != 1
            or actual['compute_capability'] != expected['compute_capability']
            or actual['memory_bytes'] < expected['minimum_memory_bytes']):
        raise ValueError(
            'GPU class or memory differs from the approved workload')


def runtime(args, case):
    import torch

    import kumo.engine as kumo
    if (not sys.flags.isolated or sys.prefix == sys.base_prefix
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or kumo.__version__ != args.version
            or importlib.metadata.version('kumo-engine') != args.version
            or not Path(kumo.__file__).resolve().is_relative_to(
                Path(sys.prefix).resolve())
            or weekly.digest(args.wheel) != args.wheel_sha256):
        raise ValueError(
            'Use an isolated installed wheel and one visible CUDA GPU')
    helper = Path(__file__).resolve().parents[1] / 'test/wheel_gpu_smoke.py'
    spec = importlib.util.spec_from_file_location('weekly_install', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    installed = module.verify_installed_files(
        args.wheel, importlib.metadata.distribution('kumo-engine'))
    subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', 'check'],
                   check=True)
    torch.set_num_threads(4)
    device = torch.cuda.get_device_properties(0)
    actual = dict(
        gpu_count=1, model=device.name, memory_bytes=device.total_memory,
        compute_capability=[device.major, device.minor], uuid=str(device.uuid),
        torch=torch.__version__, cuda=torch.version.cuda,
        driver=subprocess.check_output([
            'nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'
        ], text=True).strip())
    check_gpu_class(case['gpu_class'], actual)
    return dict(
        version=args.version, wheel_sha256=args.wheel_sha256,
        installed_files=installed, device=actual, dependencies={
            d.metadata['Name']: d.version
            for d in importlib.metadata.distributions()
        })


@contextmanager
def observe_work(state):
    """Observe Lightning's public callbacks without changing engine methods."""
    import torch
    from pytorch_lightning import Callback, Trainer

    initialize, fit = Trainer.__init__, Trainer.fit

    class Proof(Callback):
        def cuda(self, values):
            if isinstance(values, dict):
                for value in values.values():
                    self.cuda(value)
            elif isinstance(values, (tuple, list)):
                for value in values:
                    self.cuda(value)
            elif isinstance(values, torch.Tensor):
                if not values.is_cuda or values.device.index != 0 or not values.numel(
                ):
                    raise ValueError('Model work used a CPU or empty tensor')
                state['cuda_tensors'] += 1

        def on_train_batch_end(self, trainer, module, outputs, batch,
                               batch_idx):
            before = state['cuda_tensors']
            self.cuda(outputs)
            if state['cuda_tensors'] == before:
                raise ValueError('Training produced no CUDA tensor')
            state['training_batches'] += 1

        def on_fit_end(self, trainer, module):
            state['completed_fits'] += 1
            state['optimizer_steps'] += int(trainer.global_step)

        def on_predict_batch_end(self, trainer, module, outputs, batch,
                                 batch_idx, dataloader_idx=0):
            before = state['cuda_tensors']
            self.cuda(outputs)
            if state['cuda_tensors'] == before:
                raise ValueError('Prediction produced no CUDA tensor')
            stage = state['stage']
            state['prediction_batches'][
                stage] = state['prediction_batches'].get(stage, 0) + 1

    def initialize_with_proof(trainer, *args, **kwargs):
        kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), Proof()]
        initialize(trainer, *args, **kwargs)
        if str(trainer.strategy.root_device) != 'cuda:0':
            raise ValueError('Lightning fell back from CUDA')

    def count_fit(trainer, *args, **kwargs):
        state['physical_fits'] += 1
        weekly.save(state['ledger'], state)
        if state['stage'] != 'training' or state['physical_fits'] != 1:
            raise ValueError(
                'Unexpected fit, retry, or fit during reference replay')
        return fit(trainer, *args, **kwargs)

    with patch.object(Trainer, '__init__', initialize_with_proof), patch.object(
            Trainer, 'fit', count_fit):
        yield


def data_directory(root, name, files):
    path = root / name
    declared = {key for key in files if key.startswith(name.rstrip('/') + '/')}
    actual = {str(p.relative_to(root)) for p in path.rglob('*') if p.is_file()}
    if not declared or actual != declared:
        raise ValueError('Frozen directory has missing or unlisted data')
    for key in declared:
        weekly.data_file(root, key, files)
    return path


def prepared(root, files, case):
    import kumo.engine as kumo

    snapshot = weekly.read(weekly.data_file(root, case['snapshot'], files))
    index = data_directory(root, snapshot['index_path'], files)
    snapshot = kumo.GraphSnapshot(id=snapshot['id'], index_path=index,
                                  graph_config=snapshot['graph_config'])
    descriptor = weekly.read(
        weekly.data_file(root, case['prediction_table'], files))
    staging = data_directory(root, descriptor['staging_dir'], files)
    table = kumo.PredictionTable(
        query=case['query'], staging_dir=staging,
        table_path=weekly.data_file(root, descriptor['table_path'],
                                    files), row_count=descriptor['row_count'],
        tt_id=descriptor['tt_id'], timestamps=tuple(descriptor['timestamps']),
        required_stages=tuple(descriptor['required_stages']))
    return snapshot, table


def prediction_values(files, population, case, scratch, input_root):
    """Sort raw Parquet once with bounded spill, then read the complete population."""
    import duckdb
    import numpy as np
    import pandas as pd

    ids = population.ENTITY1.tolist()
    if not ids or not population.ENTITY1.is_unique or population.ENTITY1.isna(
    ).any():
        raise ValueError('Stored population is empty or repeated')
    positions = {v: i for i, v in enumerate(ids)}
    values, ranked, decisions = [None] * len(ids), [None] * len(ids), [
        None
    ] * len(ids)
    link = case['metric'].startswith('recall@')
    k = int(case['metric'].split('@')[1]) if link else None
    catalog = set(pd.read_parquet(
        input_root / case['catalog']).CLASS.tolist()) if link else None
    group, pairs = None, []
    seen = set()

    def finish(entity):
        if entity not in positions or entity in seen:
            raise ValueError('Prediction has foreign or repeated entities')
        seen.add(entity)
        index = positions[entity]
        if link:
            if (len(pairs) != case['raw_limit']
                    or len({p[0]
                            for p in pairs}) != len(pairs)
                    or any(p[0] not in catalog or not weekly.finite(p[1])
                           for p in pairs) or len(pairs) < k):
                raise ValueError(
                    'Incomplete ranked output or invalid catalog keys')
            if len(pairs) < len(catalog) and pairs[k - 1][1] == pairs[-1][1]:
                raise ValueError(
                    'Capped output cannot establish the top-K cutoff')
            ranked[index] = [p[0] for p in pairs[:k]]
            values[index] = [p[1] for p in pairs[:k]]
        else:
            if len(pairs) != 1 or not weekly.finite(pairs[0][0]):
                raise ValueError('Scalar prediction is repeated or nonfinite')
            values[index], decisions[index] = [pairs[0][0]], pairs[0][1]

    scratch.mkdir(parents=True, exist_ok=False)
    try:
        with duckdb.connect(config={
                'threads': 1,
                'memory_limit': '2GB',
                'temp_directory': str(scratch)
        }) as db:
            relation = db.read_parquet([str(p) for p in files])
            order = 'ENTITY, SCORE DESC, CLASS ASC' if link else 'ENTITY'
            for batch in relation.order(order).fetch_record_batch(65536):
                frame = batch.to_pandas()
                if not pd.api.types.is_integer_dtype(frame.ENTITY):
                    raise ValueError(
                        'Prediction changed the integer entity key type')
                for row in frame.to_dict('records'):
                    entity = row['ENTITY']
                    if entity not in positions:
                        raise ValueError('Foreign prediction entity')
                    if 'TIMESTAMP' in population:
                        anchor = population.TIMESTAMP.iloc[positions[entity]]
                        if pd.Timestamp(row.get(
                                'ANCHOR_TIMESTAMP')) != pd.Timestamp(anchor):
                            raise ValueError(
                                'Prediction anchor differs from the stored population'
                            )
                    if group is not None and entity != group:
                        finish(group)
                        pairs = []
                    group = entity
                    if link:
                        pairs.append((row['CLASS'], row['SCORE']))
                    elif case['metric'] == 'auroc':
                        probability = row['True_PROB']
                        if (not weekly.finite(probability)
                                or not 0 <= probability <= 1
                                or not weekly.finite(row['False_PROB']) or
                                abs(probability + row['False_PROB'] - 1) > 1e-6
                                or row['TARGET_PRED'] != (probability > .5)):
                            raise ValueError(
                                'Invalid binary probabilities or label')
                        pairs.append((probability, bool(row['TARGET_PRED'])))
                    else:
                        pairs.append((row['TARGET_PRED'], None))
            if group is not None:
                finish(group)
        if seen != set(ids):
            raise ValueError('Prediction omitted part of the stored population')
        result = dict(entity_ids=ids, values=values)
        result['ranked_ids'
               if link else 'decisions'] = ranked if link else decisions
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError('Prediction output contains nonfinite values')
        return result
    finally:
        shutil.rmtree(scratch)


def score(case, values, root):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    if case['id'] == 'P2-fixed':
        return None
    truth = pd.read_parquet(root / case['answers'])
    if (not truth.ENTITY1.is_unique
            or set(truth.ENTITY1) != set(values['entity_ids'])):
        raise ValueError(
            'Reserved answers differ from the full prediction population')
    targets = truth.set_index('ENTITY1').loc[
        values['entity_ids']].TARGET1.tolist()
    if case['metric'].startswith('recall@'):
        if any(not len(v) for v in targets):
            raise ValueError('Recall needs nonempty reserved targets')
        return float(
            np.mean([
                len(set(t) & set(p)) / len(set(t))
                for t, p in zip(targets, values['ranked_ids'], strict=True)
            ]))
    numbers = [v[0] for v in values['values']]
    if case['metric'] == 'mae':
        result = float(np.mean(np.abs(np.asarray(targets) - numbers)))
    else:
        result = float(roc_auc_score(targets, numbers))
    if not weekly.finite(result):
        raise ValueError('Quality score is nonfinite')
    return result


def run(args):
    import numpy as np
    import pandas as pd
    import torch
    from kumo_core.io.json import from_json_dict
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumoapi.model_plan import ModelPlan

    import kumo.engine as kumo

    started = time.monotonic()
    state = dict(physical_fits=0, completed_fits=0, optimizer_steps=0,
                 training_batches=0, prediction_batches={}, cuda_tensors=0,
                 stage='reference_replay',
                 ledger=str(args.output / 'fits.json'))
    report = dict(status='RUNNING', case_id=args.case, work=state)
    try:
        manifest = weekly.load_manifest(args.input_root, args.manifest_sha256,
                                        verify_files=False)
        [case] = [c for c in manifest['cases'] if c['id'] == args.case]
        files = {key: manifest['files'][key] for key in case['files']}
        report['runtime'] = runtime(args, case)
        args.scratch.mkdir(parents=True, exist_ok=False)
        kumo.configure(scratch_dir=args.scratch)
        snapshot, table = prepared(args.input_root, files, case)
        population = pd.read_parquet(
            weekly.data_file(args.input_root, case['population'], files))
        plan_json = weekly.read(
            weekly.data_file(args.input_root, case['model_plan'], files))
        plan = from_json_dict(plan_json, ModelPlan)
        job = plan.training_job
        if (job.num_experiments != 1 or job.refit_trainval or job.refit_full
                or job.manual_seed != 0):
            raise ValueError(
                'The frozen seed-zero plan must run one experiment without refits'
            )
        weekly.save(args.output / 'requested-plan.json', plan_json)
        descriptor = weekly.read(
            weekly.data_file(args.input_root, case['training_table'], files))
        training = kumo.TrainingTable(
            query=case['query'],
            staging_dir=data_directory(args.input_root,
                                       descriptor['staging_dir'], files),
            train_table_path=weekly.data_file(args.input_root,
                                              descriptor['train_table_path'],
                                              files),
            manifest_path=weekly.data_file(args.input_root,
                                           descriptor['manifest_path'], files),
            required_stages=tuple(descriptor['required_stages']),
            generation_plan=from_json_dict(descriptor['generation_plan'],
                                           TrainingTableGenerationPlan))

        def seed():
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)

        def load(path):
            options = {
                'accelerator': 'gpu'
            } if 'accelerator' in inspect.signature(
                kumo.Trainer.load).parameters else {}
            return kumo.Trainer.load(path, **options)

        def predict(model, stage):
            from kumoapi.jobs import MetadataField
            seed()
            state['stage'] = stage
            output = args.output / 'raw' / stage
            config = kumo.PredictionOutputConfig(
                kumo.FileConnector(root_dir=str(output)),
                output_paths={'predictions': 'predictions'},
                extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
            options = dict(output_config=config)
            if case['metric'].startswith('recall@'):
                options['num_classes_to_return'] = case['raw_limit']
            elif case['metric'] == 'auroc':
                options['binary_classification_threshold'] = .5
            before = state['prediction_batches'].get(stage, 0)
            result = model.predict(table, snapshot=snapshot, **options)
            torch.cuda.synchronize()
            if state['prediction_batches'].get(stage, 0) <= before:
                raise ValueError('Prediction completed no CUDA batches')
            return prediction_values(result.files, population, case,
                                     args.scratch / ('sort-' + stage),
                                     args.input_root)

        with observe_work(state):
            if args.reference_root:
                previous = args.reference_root / 'cases' / args.case
                report['reference_replay'] = weekly.replay_reference(
                    previous / 'model',
                    weekly.read(previous / 'predictions.json'), load,
                    lambda model: predict(model, 'reference_replay'),
                    **case['tolerance'])
                if state['physical_fits']:
                    raise ValueError('Reference replay performed training')
                gc.collect()
            seed()
            state['stage'] = 'training'
            options = {
                'accelerator': 'gpu'
            } if 'accelerator' in inspect.signature(
                kumo.Trainer).parameters else {}
            trainer = kumo.Trainer(plan, **options)
            torch.cuda.reset_peak_memory_stats()
            training_started = time.monotonic()
            fitted = trainer.fit(training, snapshot=snapshot)
            torch.cuda.synchronize()
            report['training_seconds'] = time.monotonic() - training_started
            if (state['physical_fits'] != 1 or state['completed_fits'] != 1
                    or state['training_batches'] <= 0
                    or state['optimizer_steps'] <= 0):
                raise ValueError(
                    'Training did not complete exactly one actual CUDA fit')
            shutil.copytree(fitted.bundle_dir, args.output / 'model')
            model_files = weekly.inventory(args.output / 'model')
            report['checkpoint_sha256'] = model_files['model/checkpoint']
            predictions = predict(trainer, 'candidate')
            weekly.save(args.output / 'predictions.json', predictions)
            report['score'] = score(case, predictions, args.input_root)
            del trainer
            gc.collect()
            restored = load(args.output / 'model')
            report['reload'] = weekly.compare_predictions(
                predictions, predict(restored, 'reload'), **case['tolerance'])
            if weekly.inventory(args.output / 'model') != model_files:
                raise ValueError('Reload changed the published model')
            report.update(
                status='PASS', fit_id=str(uuid.uuid4()),
                physical_fits=state['physical_fits'],
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    finally:
        report.update(
            physical_fits=state['physical_fits'],
            seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            1024)
        weekly.save(args.output / 'fits.json', state)
        weekly.save(args.output / 'worker-result.json', report)
        # The parent retains model evidence even if this worker fails.
        if args.scratch.exists():
            shutil.rmtree(args.scratch)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'output', 'scratch', 'wheel', 'reference-root'):
        parser.add_argument('--' + name, type=Path, required=name
                            != 'reference-root')
    for name in ('manifest-sha256', 'case', 'wheel-sha256', 'version'):
        parser.add_argument('--' + name, required=True)
    return 0 if run(parser.parse_args())['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
