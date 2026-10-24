# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit smoke: pytest -q -s packages/kumo-engine/test/smoke_platform.py.

Set KUMO_SMOKE_DEVICE=cpu for CPU training; the default is mandatory CUDA.
The filename keeps this integration check out of ordinary test discovery.
"""

import json
import math
import os
import platform
import sys

import numpy as np
import pandas as pd
import torch
from kumo_core.diskgraph.runner import DiskgraphIngestError
from kumoapi.model_plan import ModelPlan
from torch.optim.optimizer import register_optimizer_step_post_hook

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


def test_platform_train_predict(tmp_path, tmp_path_factory, monkeypatch):
    device = os.environ.get('KUMO_SMOKE_DEVICE', 'cuda')
    assert device in {'cpu', 'cuda'}, 'KUMO_SMOKE_DEVICE must be cpu or cuda'
    diagnostics = {
        'requested_device': device,
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'python': sys.version,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
    }
    print(json.dumps(diagnostics, sort_keys=True), flush=True)
    if device == 'cuda':
        assert torch.cuda.is_available(), 'This smoke test requires a CUDA GPU'
        print(f'CUDA device: {torch.cuda.get_device_name(0)}', flush=True)
    monkeypatch.setenv('KUMO_REQUIRE_GPU', '1' if device == 'cuda' else '0')
    monkeypatch.setenv('KUMO_ACCELERATOR', device)

    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        # A single numerical table exercises real ingestion and training
        # without external datasets, credentials, or text-model downloads.
        rng = np.random.default_rng(0)
        row_count = 400
        labeled_count = 300
        targets = rng.random(row_count)
        targets[labeled_count:] = np.nan
        source_dir = tmp_path_factory.mktemp('smoke-source')
        pd.DataFrame({
            'ID': np.arange(row_count),
            'Y': targets,
            'F1': rng.normal(size=row_count),
        }).to_parquet(source_dir / 'entities.parquet')
        connector = kumo.FileConnector(root_dir=str(source_dir))
        entities = kumo.Table.from_source_table(
            connector['entities.parquet'], primary_key='ID').infer_metadata()
        graph = kumo.Graph(tables={'entities': entities}, edges=[])
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(
            graph=graph, query='PREDICT entities.Y FOR EACH entities.ID')
        plan = ModelPlan()
        plan.training_job.num_experiments = 1
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        trainer = kumo.Trainer(model_plan=plan)
        optimizer_steps = []

        def record_optimizer_step(optimizer, _args, _kwargs):
            gradients = [parameter.grad for group in optimizer.param_groups
                         for parameter in group['params']
                         if parameter.grad is not None]
            assert gradients, 'Optimizer step did not produce gradients'
            assert all(gradient.device.type == device for gradient in gradients)
            assert all(torch.isfinite(gradient).all().item()
                       for gradient in gradients)
            optimizer_steps.append(1)

        # Observe real optimizer work; availability alone cannot establish
        # that training actually ran on the requested device.
        hook = register_optimizer_step_post_hook(record_optimizer_step)
        try:
            fitted = trainer.fit(query.generate_training_table(snapshot),
                                 snapshot=snapshot)
        finally:
            hook.remove()
        assert optimizer_steps, f'No {device} optimizer step was observed'
        assert fitted.metrics
        assert all(math.isfinite(value) for value in fitted.metrics.values())
        assert any(name.startswith('train_') for name in fitted.metrics)

        predictions = trainer.predict(
            query.generate_prediction_table(snapshot), snapshot=snapshot)
        frame = predictions.frame
        prediction_count = row_count - labeled_count
        assert predictions.num_rows == prediction_count
        assert len(frame) == prediction_count
        assert set(frame['ENTITY']) == set(range(labeled_count, row_count))
        assert np.isfinite(frame['TARGET_PRED'].to_numpy()).all()
        if device == 'cuda':
            torch.cuda.synchronize()
        print(f'Platform smoke completed: {len(optimizer_steps)} '
              f'{device} optimizer steps, '
              f'{len(frame)} finite predictions',
              flush=True)
    except DiskgraphIngestError as error:
        # This test uses only generated fixture data; expose its subprocess
        # diagnostics without changing production error privacy or retrying.
        print(error.private_output_tail, file=sys.stderr, flush=True)
        raise
    finally:
        kumo.configure(scratch_dir=previous)
