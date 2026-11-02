#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the retained P2 model and compare one-worker and two-worker predictions."""

import argparse
import fcntl
import importlib.metadata
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    'weekly_worker',
    Path(__file__).with_name('weekly_worker.py'))
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)
weekly = worker.weekly


def prediction_drift(reference, observed):
    """Measure score drift by class ID without approving changed rankings."""
    ids = reference['entity_ids']
    if not ids or len(set(ids)) != len(ids) or ids != observed['entity_ids']:
        raise ValueError('Cannot measure drift for different populations')
    report = dict(entities=len(ids), rank_changed_entities=0,
                  membership_changed_entities=0, removed_recommendations=0,
                  added_recommendations=0, common_score_coordinates=0,
                  max_abs_common_score_difference=0., membership_examples=[],
                  rank_examples=[])
    for entity, left, right, left_scores, right_scores in zip(
            ids, reference['ranked_ids'], observed['ranked_ids'],
            reference['values'], observed['values'], strict=True):
        if (not left or len(left) != len(right) or len(left) != len(left_scores)
                or len(right) != len(right_scores)
                or len(set(left)) != len(left) or len(set(right)) != len(right)
                or not all(
                    weekly.finite(v) for v in (*left_scores, *right_scores))):
            raise ValueError('Cannot measure incomplete or nonfinite rankings')
        before, after = dict(zip(left,
                                 left_scores)), dict(zip(right, right_scores))
        if left != right:
            report['rank_changed_entities'] += 1
            if len(report['rank_examples']) < 5:
                report['rank_examples'].append(
                    dict(entity_id=entity, reference_ids=left,
                         observed_ids=right, reference_scores=left_scores,
                         observed_scores=right_scores))
        removed, added = before.keys() - after.keys(), after.keys(
        ) - before.keys()
        if removed or added:
            report['membership_changed_entities'] += 1
            report['removed_recommendations'] += len(removed)
            report['added_recommendations'] += len(added)
            if len(report['membership_examples']) < 5:
                report['membership_examples'].append(
                    dict(
                        entity_id=entity, removed=[[key, before[key]]
                                                   for key in sorted(removed)],
                        added=[[key, after[key]] for key in sorted(added)],
                        reference_boundary_score=min(left_scores),
                        observed_boundary_score=min(right_scores)))
        common = before.keys() & after.keys()
        report['common_score_coordinates'] += len(common)
        report['max_abs_common_score_difference'] = max(
            report['max_abs_common_score_difference'],
            max((abs(before[key] - after[key]) for key in common), default=0.))
    return report


def check_predictions(reference, one_gpu, two_gpu, *, atol, rtol):
    return dict(
        one_gpu=weekly.compare_predictions(reference, one_gpu, atol=atol,
                                           rtol=rtol),
        two_gpu=weekly.compare_predictions(reference, two_gpu, atol=atol,
                                           rtol=rtol),
        worker_counts=weekly.compare_predictions(one_gpu, two_gpu, atol=atol,
                                                 rtol=rtol))


def check_devices(records, count, checkpoint):
    if (len(records) != count or {r['device']
                                  for r in records} != set(range(count))
            or len({r['uuid']
                    for r in records}) != count
            or len({r['pid']
                    for r in records}) != count
            or len({r['model']
                    for r in records}) != 1
            or any(r['status'] != 'PASS' or r['checkpoint_sha256'] != checkpoint
                   or r['restored_tensors'] <= 0 or r['batches'] <= 0
                   or r['elements'] <= 0 or r['fit_attempts'] != 0
                   for r in records)):
        raise ValueError(
            'B9 requires actual prediction and retained weights on every distinct matching GPU'
        )


def installation_helper():
    return worker.installation_helper()


@contextmanager
def observe_predictions():
    """Attach public Lightning callbacks in the parent or a spawned scorer."""
    import torch
    from pytorch_lightning import Callback, Trainer

    output = Path(os.environ['KUMO_B9_RECEIPTS'])
    checkpoint = os.environ['KUMO_B9_CHECKPOINT']
    destination = output / f'{os.getpid()}.json'
    proof = dict(status='RUNNING', pid=os.getpid(), batches=0, elements=0,
                 fit_attempts=0, restored_tensors=0,
                 checkpoint_sha256=checkpoint)
    initialize = Trainer.__init__

    class Observation(Callback):
        def on_predict_start(self, trainer, module):
            device = trainer.strategy.root_device
            if device.type != 'cuda':
                raise ValueError('B9 prediction fell back to CPU')
            properties = torch.cuda.get_device_properties(device)
            proof.update(device=device.index, uuid=str(properties.uuid),
                         model=properties.name)
            proof['restored_tensors'] = installation_helper().verify_checkpoint(
                module, trainer.ckpt_path, checkpoint)
            weekly.save(destination, proof)

        def on_predict_batch_end(self, trainer, module, outputs, batch,
                                 batch_idx, dataloader_idx=0):
            scores = outputs.get('score') if isinstance(outputs, dict) else None
            if (not isinstance(scores, torch.Tensor)
                    or scores.device != trainer.strategy.root_device
                    or not scores.is_cuda or not scores.numel()
                    or not torch.isfinite(scores).all()):
                raise ValueError(
                    'B9 scorer did not return finite CUDA ranking tensors')
            torch.cuda.synchronize(scores.device)
            proof['batches'] += 1
            proof['elements'] += scores.numel()

        def on_predict_end(self, trainer, module):
            proof['status'] = 'PASS'
            weekly.save(destination, proof)

        def on_exception(self, trainer, module, exception):
            proof.update(status='FAIL',
                         error=f'{type(exception).__name__}: {exception}')
            weekly.save(destination, proof)

    def initialize_with_observer(trainer, *args, **kwargs):
        kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), Observation()]
        initialize(trainer, *args, **kwargs)

    def reject_fit(*args, **kwargs):
        proof['fit_attempts'] += 1
        proof.update(status='FAIL',
                     error='Training is forbidden during retained prediction')
        weekly.save(destination, proof)
        raise ValueError(proof['error'])

    with patch.object(Trainer, '__init__',
                      initialize_with_observer), patch.object(
                          Trainer, 'fit', reject_fit):
        yield


def run(args):
    with args.host_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'
        ], text=True).strip()
        if active:
            raise ValueError(
                'B9 cannot start while the host has active GPU work')
        return run_locked(args)


def run_locked(args):
    import numpy as np
    import pandas as pd
    import torch
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo

    manifest = weekly.load_manifest(args.input_root, args.manifest_sha256,
                                    profile='scale', retained_p2=True)
    reference = weekly.load_reference(args.reference_root,
                                      args.reference_sha256, manifest,
                                      args.manifest_sha256, profile='scale',
                                      retained_p2=True)
    [case] = [c for c in manifest['cases'] if c['id'] == 'P2-fixed']
    files = {key: manifest['files'][key] for key in case['files']}
    retained = reference['cases']['P2-fixed']
    model = args.reference_root / 'cases/P2-fixed/model'
    if args.output.exists() or args.scratch.exists():
        raise ValueError('Use fresh B9 evidence and scratch directories')
    if (args.output.resolve().is_relative_to(args.scratch.resolve())
            or args.scratch.resolve().is_relative_to(args.output.resolve())
            or any(path.resolve().is_relative_to(root.resolve())
                   for root in (args.input_root, args.reference_root)
                   for path in (args.output, args.scratch))):
        raise ValueError(
            'Keep scratch and evidence outside immutable inputs and models')
    args.output.mkdir(parents=True)
    report = dict(format='kumo-engine-weekly-b9-result-v1', status='RUNNING',
                  manifest_sha256=args.manifest_sha256,
                  reference_sha256=args.reference_sha256, physical_fits=0,
                  retained_fit_id=retained['fit_id'],
                  checkpoint_sha256=retained['checkpoint_sha256'], calls=[])
    started = time.monotonic()
    try:
        if not sys.flags.isolated or sys.prefix == sys.base_prefix:
            raise ValueError('Use the isolated selected wheel environment')
        distribution = importlib.metadata.distribution('kumo-engine')
        if (weekly.digest(args.wheel) != args.wheel_sha256
                or distribution.version != args.version
                or kumo.__version__ != args.version):
            raise ValueError('B9 installed wheel identity differs')
        installation_helper().verify_installed_files(args.wheel, distribution)
        subprocess.run([sys.executable, '-I', '-m', 'pip', 'check'], check=True)
        report['runtime'] = dict(
            version=args.version, wheel_sha256=args.wheel_sha256, dependencies={
                d.metadata['Name']: d.version
                for d in importlib.metadata.distributions()
            })
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise ValueError('B9 needs exactly two visible CUDA GPUs')
        devices = [torch.cuda.get_device_properties(i) for i in range(2)]
        if devices[0].name != devices[1].name or str(devices[0].uuid) == str(
                devices[1].uuid):
            raise ValueError('B9 needs two distinct matching GPUs')
        for device in devices:
            worker.check_gpu_class(
                case['gpu_class'],
                dict(gpu_count=1, memory_bytes=device.total_memory,
                     compute_capability=[device.major, device.minor]))
        args.scratch.mkdir(parents=True)
        weekly.check_capacity([case], weekly.capacity(args.scratch.parent))
        kumo.configure(scratch_dir=args.scratch)
        snapshot, table = worker.prepared(args.input_root, files, case,
                                          args.scratch)
        report['graph_population'] = worker.check_graph_population(
            snapshot, case,
            weekly.read(
                weekly.data_file(args.input_root, case['dataset_manifest'],
                                 files)))
        population = pd.read_parquet(
            weekly.data_file(args.input_root, case['population'], files))
        load_options = {
            'accelerator': 'gpu'
        } if 'accelerator' in inspect.signature(
            kumo.Trainer.load).parameters else {}
        trainer = None
        os.environ['KUMO_B9_CHECKPOINT'] = retained['checkpoint_sha256']
        predictions = []
        for count in (1, 2):
            receipts = args.output / f'devices-{count}'
            receipts.mkdir()
            os.environ['KUMO_B9_RECEIPTS'] = str(receipts.resolve())
            torch.manual_seed(0)
            np.random.seed(0)
            output = args.output / f'raw-{count}'
            config = kumo.PredictionOutputConfig(
                kumo.FileConnector(root_dir=str(output)),
                output_paths={'predictions': 'predictions'},
                extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
            before = time.monotonic()
            with observe_predictions():
                if trainer is None:
                    trainer = kumo.Trainer.load(model, **load_options)
                result = trainer.predict(
                    table, snapshot=snapshot, num_workers=count,
                    num_classes_to_return=case['raw_limit'],
                    output_config=config)
                for index in range(2):
                    torch.cuda.synchronize(index)
            proof = [weekly.read(p) for p in sorted(receipts.glob('*.json'))]
            check_devices(proof, count, retained['checkpoint_sha256'])
            values = worker.prediction_values(result.files, population, case,
                                              args.scratch / f'sort-{count}',
                                              args.input_root)
            weekly.save(args.output / f'predictions-{count}.json', values)
            predictions.append(values)
            report['calls'].append(
                dict(workers=count, seconds=time.monotonic() - before,
                     devices=proof))
            weekly.save(args.output / 'b9-result.json', report)
        stored = weekly.read(args.reference_root /
                             'cases/P2-fixed/predictions.json')
        report['drift'] = dict(one_gpu=prediction_drift(stored, predictions[0]),
                               two_gpu=prediction_drift(stored, predictions[1]),
                               worker_counts=prediction_drift(*predictions))
        weekly.save(args.output / 'b9-result.json', report)
        report['comparison'] = check_predictions(stored, *predictions,
                                                 **case['tolerance'])
        report.update(status='PASS', manifest_sha256=args.manifest_sha256,
                      reference_sha256=args.reference_sha256)
        weekly.load_reference(args.reference_root, args.reference_sha256,
                              manifest, args.manifest_sha256, profile='scale',
                              retained_p2=True)
        weekly.load_manifest(args.input_root, args.manifest_sha256,
                             profile='scale', retained_p2=True)
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.monotonic() - started
        weekly.save(args.output / 'b9-result.json', report)
        os.environ.pop('KUMO_B9_RECEIPTS', None)
        os.environ.pop('KUMO_B9_CHECKPOINT', None)
        if args.scratch.exists():
            shutil.rmtree(args.scratch)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'reference-root', 'output', 'scratch',
                 'host-lock', 'wheel'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'reference-sha256', 'wheel-sha256',
                 'version'):
        parser.add_argument('--' + name, required=True)
    return 0 if run(parser.parse_args())['status'] == 'PASS' else 1


# Python's spawn method imports this main module before starting a scorer.
# Install the same public callbacks there, without changing product worker code.
if __name__ == '__mp_main__' and os.environ.get('KUMO_B9_RECEIPTS'):
    _SPAWN_OBSERVER = observe_predictions()
    _SPAWN_OBSERVER.__enter__()

if __name__ == '__main__':
    sys.exit(main())
