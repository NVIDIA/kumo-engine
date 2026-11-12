# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate desktop producer/consumer processes using retained public exports."""
import argparse
import importlib.util
import shutil
import traceback
from pathlib import Path


def helpers():
    spec = importlib.util.spec_from_file_location(
        'desktop_helpers',
        Path(__file__).with_name('wheel_mlops_lifecycle.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprints(directory):
    h = helpers()
    return {
        str(p.relative_to(directory)): h.digest(p)
        for p in sorted(directory.rglob('*')) if p.is_file()
    }


def produce(root, device):
    import kumo.engine as kumo
    h = helpers()
    producer = root / 'producer'
    producer.mkdir()
    durable = root / 'durable'
    durable.mkdir()
    kumo.configure(scratch_dir=producer / 'scratch')
    graph = h.graph(producer, '2024-03-31')
    snapshot = graph.snapshot()
    query = kumo.PredictiveQuery(graph=graph, query=h.QUERY)
    plan = h.gpu_helpers().make_plan('binary')
    trainer = kumo.Trainer(plan, accelerator=device)
    fitted = trainer.fit(query.generate_training_table(snapshot),
                         snapshot=snapshot)
    h.require(fitted.bundle_id and snapshot.id,
              'Durable model and snapshot identities are required')
    connector = kumo.FileConnector(root_dir=str(durable))
    fitted.export(kumo.ModelOutputConfig(connector, path='model'))
    snapshot.export(kumo.SnapshotOutputConfig(connector, path='snapshot'))
    table = query.generate_prediction_table(snapshot,
                                            prediction_time='2024-03-30')
    prediction = trainer.predict(table, snapshot=snapshot,
                                 binary_classification_threshold=0.5)
    h.population(prediction).to_parquet(durable / 'expected.parquet')
    h.write(durable / 'identity.json',
            dict(bundle=fitted.bundle_id, snapshot=snapshot.id))
    h.write(root / 'fingerprints.json', fingerprints(durable))
    return dict(bundle=fitted.bundle_id, rows=prediction.scored_rows,
                files=fingerprints(durable))


def consume(root, device):
    import numpy as np
    import pandas as pd

    import kumo.engine as kumo
    h = helpers()
    h.require(not (root / 'producer').exists(),
              'Producer data and scratch must be absent')
    durable = root / 'durable'
    before = fingerprints(durable)
    identity = h.read(durable / 'identity.json')
    expected = pd.read_parquet(durable / 'expected.parquet')
    kumo.configure(scratch_dir=root / 'consumer-scratch')
    connector = kumo.FileConnector(root_dir=str(durable))
    snapshot = kumo.GraphSnapshot.load_from_connector(connector,
                                                      path='snapshot')
    trainer = kumo.Trainer.load_from_connector(connector, path='model',
                                               accelerator=device)
    h.require(
        trainer.result.bundle_id == identity['bundle']
        and snapshot.id == identity['snapshot'],
        'Loaded a different model or snapshot')
    query = kumo.PredictiveQuery(graph=snapshot, query=h.QUERY)
    table = query.generate_prediction_table(snapshot,
                                            prediction_time='2024-03-30')
    frames = []
    for _ in range(2):
        result = trainer.predict(table, snapshot=snapshot,
                                 binary_classification_threshold=0.5)
        frame = h.population(result)
        np.testing.assert_array_equal(frame.ENTITY, expected.ENTITY)
        np.testing.assert_array_equal(frame.TARGET_PRED, expected.TARGET_PRED)
        np.testing.assert_allclose(frame[['False_PROB', 'True_PROB']],
                                   expected[['False_PROB',
                                             'True_PROB']], atol=1e-4, rtol=0)
        frames.append(frame)
    h.require(
        before == h.read(root / 'fingerprints.json') == fingerprints(durable),
        'Prediction changed the retained model or graph')
    return dict(bundle=identity['bundle'], snapshot=snapshot.id,
                predictions=len(frames), rows=len(frames[0]),
                producer_absent=True, durable_unchanged=True, score_atol=1e-4,
                score_rtol=0)


def incomplete(root, _device):
    import kumo.engine as kumo
    h = helpers()
    broken = root / 'incomplete-model'
    shutil.copytree(root / 'durable/model', broken)
    manifest = h.read(broken / 'publication-manifest.json')
    entry = next(a for a in manifest['artifacts']
                 if a['role'] == 'training_spec')
    missing = broken / entry['relative_path']
    missing.unlink()
    try:
        kumo.Trainer.load(broken, accelerator='cpu')
    except (ValueError, FileNotFoundError) as error:
        h.require(
            missing.name in str(error) or 'missing' in str(error).lower(),
            'Incomplete artifact diagnostic does not identify the failure')
        return dict(rejected=True, diagnostic=str(error))
    raise AssertionError('Incomplete retained model was accepted')


def no_faiss(_root, _device):
    import torch
    from kumo_core.modeling.retrieval.mips import (
        FaissUnavailableError,
        MIPSFaiss,
    )
    assert importlib.util.find_spec(
        'faiss') is None, 'FAISS was not actually removed'
    try:
        MIPSFaiss(torch.zeros(2, 3))
    except FaissUnavailableError as error:
        assert 'kumo-engine[cpu]' in str(
            error), 'Missing actionable installation guidance'
        return dict(rejected=True, diagnostic=str(error), faiss_absent=True)
    raise AssertionError('Retrieval without its dependency was accepted')


def main():
    from unittest.mock import patch

    import torch
    from kumo_core.train.trainer import Trainer as CoreTrainer
    from torch.optim.optimizer import register_optimizer_step_post_hook
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--accelerator', choices=('cpu', 'mps'), required=True)
    parser.add_argument(
        'phase', choices=('produce', 'consume', 'incomplete', 'no-faiss'))
    args = parser.parse_args()
    h = helpers()
    report = dict(status='failed', requested=args.accelerator, devices=[],
                  optimizer_steps=0)
    original = CoreTrainer.create_pl_trainer

    def observe(trainer):
        original(trainer)
        actual = trainer.pl_trainer.strategy.root_device.type
        report['devices'].append(actual)
        h.require(actual == args.accelerator,
                  'Unexpected device or CPU fallback')

    def optimizer_step(optimizer, _args, _kwargs):
        gradients = [
            p.grad for group in optimizer.param_groups for p in group['params']
            if p.grad is not None
        ]
        h.require(
            gradients
            and all(g.device.type == args.accelerator for g in gradients),
            'Optimizer gradients differ from the selected device')
        h.require(all(torch.isfinite(g).all().item() for g in gradients),
                  'Nonfinite gradients')
        report['optimizer_steps'] += 1

    hook = register_optimizer_step_post_hook(optimizer_step)
    torch.set_num_threads(2)
    try:
        with patch.object(CoreTrainer, 'create_pl_trainer', observe):
            report['result'] = globals()[args.phase.replace('-', '_')](
                args.root, args.accelerator)
        if args.phase in ('produce', 'consume'):
            h.require(report['devices'], 'No model execution observed')
        if args.phase == 'produce':
            h.require(report['optimizer_steps'] > 0,
                      'No optimizer work observed')
        else:
            h.require(report['optimizer_steps'] == 0,
                      'Inference unexpectedly trained')
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        hook.remove()
        h.write(args.root / (args.phase + '.json'), report)


if __name__ == '__main__':
    main()
