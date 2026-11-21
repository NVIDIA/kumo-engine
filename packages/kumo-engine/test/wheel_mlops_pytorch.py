#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customer regression workload using kumo-pytorch and native MLflow models.

Pass this file to wheel_mlops_lifecycle.py --workload. Requires kumo-pytorch
and MLflow's PyTorch dependencies in the candidate environment. The in-memory
fixture exercises the batch/artifact API, not DiskGraph scale or performance.
"""

import json
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path

PHASES = ('train', 'reload')
METRICS = ('train/mse', 'validation/rmse')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inputs():
    from kumo.pytorch import api

    schema = api.DataSchema(tables=(api.TableSpec(
        'users', 'id', {
            'id': api.DType('int64'),
            'value': api.DType('float64'),
            'created': api.DType('timestamp', unit='s', timezone='UTC')
        }, time_column='created'), ))
    sample = api.SampleSpec(
        entity=api.EntitySpec('users', entity_type='user'), entity_values={
            'raw_value': api.Field(column='value'),
            'value': api.Derived('log1p', ('raw_value', )),
        }, targets={'label': api.ExternalTarget(dtype=api.DType('float64'))})
    batch = api.BatchSpec(
        inputs={
            'user':
            api.Input(api.EntitySource(),
                      api.Fields((api.Encode('value', api.ZScore()), )),
                      api.Separate())
        }, targets={'label': api.LabelTarget('label', api.Raw())},
        batching=api.Batching(max_samples=16))
    contract = api.DataLoaderSpec(schema, sample, batch)
    dataset = api.Dataset({
        'users':
        api.InMemoryRows([{
            'id': i,
            'value': float(i),
            'created': 0
        } for i in range(32)])
    })
    requests = [
        api.SampleRequest(
            str(i), i, 1, split='train' if i < 24 else 'validation',
            targets=(api.TargetValue('label', 2.0 * math.log1p(i) + 1), ))
        for i in range(32)
    ]
    return contract, dataset, requests


def features(batch, device='cpu'):
    return batch.inputs['user']['value']['values'].float().reshape(-1, 1).to(device)


def devices(profile):
    import torch

    if profile is None or profile.accelerator == 'cpu':
        return ('cpu',)
    require(profile.accelerator == 'gpu', 'Unknown workload accelerator')
    require(torch.cuda.is_available(), 'GPU workload requires an available CUDA device')
    return ('cuda',)


@contextmanager
def tracking(root, profile):
    import mlflow
    from mlflow import MlflowClient

    if profile is None:
        uri = f'sqlite:///{root / "customer-tracking.db"}'
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        experiment, model = 'customer-regression', 'customer-regression'
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(
                experiment,
                artifact_location=(root / 'registry-artifacts').as_uri())
        clients = nullcontext(MlflowClient())
    else:
        if profile.config['platform'] == 'snowflake':
            raise ImportError(
                'This example uses native MLflow registration; supply a workload '
                'using the Snowflake native model registry for Snowflake')
        devices(profile)
        cfg = profile.settings(False)
        clients = cfg.tracking.client()
        experiment, model = profile.experiment, profile.model
    with clients as client:
        mlflow.set_experiment(experiment)
        yield client, model


def predictions(task, model):
    import torch

    result = {}
    with torch.no_grad():
        for batch in task.batches():
            require(not batch.targets, 'Inference emitted training targets')
            device = next(model.parameters()).device
            values = model(features(batch, device)).flatten().cpu().tolist()
            for key, value in zip(batch.metadata['sample_ids'], values,
                                  strict=True):
                require(key not in result and math.isfinite(value),
                        'Duplicate ID or nonfinite prediction')
                result[key] = value
    require(
        set(result) == {str(i)
                        for i in range(32)},
        'Predictions lost or invented a requested ID')
    return result


def train(root, profile):
    import mlflow
    import mlflow.pytorch
    import numpy as np
    import torch

    from kumo.pytorch import api

    with tracking(root, profile) as (client, name):
        contract, dataset, requests = inputs()
        versions = []
        selected_devices = devices(profile)
        with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
            with plan.materialize(requests, fit_split='train') as task:
                centered = torch.cat(
                    [features(batch) for batch in task.batches(split='train')])
                require(
                    abs(centered.mean().item()) < 1e-6,
                    'Derived input normalization was not fitted on the training split'
                )
                variants = [(device, seed) for device in selected_devices for seed in (0, 1)]
                for index, (device, seed) in enumerate(variants):
                    torch.manual_seed(seed)
                    model = torch.nn.Linear(1, 1).to(device)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    directory = root / 'scratch-train' / str(index)
                    directory.mkdir(parents=True)
                    task.to_artifact().save(directory / 'preprocessing')
                    with mlflow.start_run() as run:
                        for epoch in range(2):
                            for batch in task.batches(split='train',
                                                      epoch=epoch):
                                labels = batch.targets['label']['values'].float(
                                ).reshape(-1, 1).to(device)
                                loss = torch.nn.functional.mse_loss(
                                    model(features(batch, device)), labels)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                            with torch.no_grad():
                                squared = [
                                    (model(features(batch, device)).flatten() -
                                     batch.targets['label']['values'].float().to(device)
                                     ).square() for batch in task.batches(
                                         split='validation')
                                ]
                                rmse = torch.cat(squared).mean().sqrt().item()
                            mlflow.log_metrics(
                                dict(zip(METRICS, (loss.item(), rmse))),
                                step=epoch)
                        model.eval()
                        api.save_weights(directory / 'weights.pt',
                                         model.state_dict(), task.fitted_state)
                        with plan.materialize(requests,
                                              fitted=task.fitted_state,
                                              inference=True) as scoring:
                            expected = predictions(scoring, model)
                        write(directory / 'expected.json', expected)
                        mlflow.log_artifacts(str(directory),
                                             artifact_path='customer')
                        example = np.zeros((2, 1), dtype=np.float32)
                        with torch.no_grad():
                            output = model(torch.from_numpy(example).to(device)).cpu().numpy()
                        logged = mlflow.pytorch.log_model(
                            model, name='model', input_example=example,
                            signature=mlflow.models.infer_signature(example, output))
                        registered = mlflow.register_model(
                            logged.model_uri, name)
                        mlflow.set_tag('customer.preprocessing',
                                       task.fitted_state.fingerprint)
                    versions.append(
                        dict(version=str(registered.version), device=device,
                             run=run.info.run_id,
                             fingerprint=task.fitted_state.fingerprint))
        require(len({v['version'] for v in versions}) == len(versions),
                'Registry reused a version')
        client.set_registered_model_alias(name, 'champion',
                                          versions[0]['version'])
        state = dict(model=name, versions=versions, devices=list(selected_devices))
        write(root / 'customer-state.json', state)
        return state


def reload(root, profile):
    import mlflow.pytorch
    import torch

    from kumo.pytorch import api

    with tracking(root, profile) as (client, name):
        state = json.loads((root / 'customer-state.json').read_text())
        selected_devices = devices(profile)
        require(state['model'] == name and state['devices'] == list(selected_devices)
                and len(state['versions']) == 2 * len(selected_devices),
                'Recovered another or incomplete registry model lifecycle')
        _, dataset, requests = inputs()
        checks = []
        for expected in state['versions']:
            version, device = expected['version'], expected['device']
            require(device in selected_devices, 'Recovered an unselected model device')
            registered = client.get_model_version(name, version)
            require(registered.run_id == expected['run'],
                    'Registry refers to another training run')
            run = client.get_run(expected['run'])
            require(run.info.status == 'FINISHED',
                    'Training tracking did not finish')
            require(
                run.data.tags.get(
                    'customer.preprocessing') == expected['fingerprint'],
                'Tracking refers to another fitted preprocessing state')
            history = {}
            for metric in METRICS:
                points = client.get_metric_history(expected['run'], metric)
                require({p.step
                         for p in points} == {0, 1}
                        and all(math.isfinite(p.value) for p in points),
                        f'Missing finite multi-epoch history: {metric}')
                history[metric] = [(p.step, p.value) for p in points]
            destination = root / 'scratch-reload' / version
            destination.mkdir(parents=True)
            directory = Path(
                client.download_artifacts(expected['run'], 'customer',
                                          str(destination)))
            artifact = api.Artifact.load(directory / 'preprocessing')
            require(
                artifact.fitted_state.fingerprint == expected['fingerprint'],
                'Recovered different preprocessing state')
            model = torch.nn.Linear(1, 1)
            model.load_state_dict(
                api.load_weights(directory / 'weights.pt',
                                 artifact.fitted_state))
            model.to(device).eval()
            saved = json.loads((directory / 'expected.json').read_text())
            client.set_registered_model_alias(name, 'champion', version)
            require(
                str(
                    client.get_model_version_by_alias(
                        name, 'champion').version) == version,
                'Alias promotion resolved another version')
            # The registry loads a native PyTorch model; Engine Trainer is never involved.
            published = mlflow.pytorch.load_model(f'models:/{name}@champion',
                                                  map_location=device)
            require(next(published.parameters()).device.type == device,
                    'MLflow loaded the native model onto another device')
            with api.prepare(artifact.contract, dataset, api.InMemoryEngine(),
                             inference=True) as plan:
                with plan.materialize(requests, fitted=artifact.fitted_state,
                                      inference=True) as task:
                    for actual in (predictions(task, model),
                                   predictions(task, published)):
                        require(
                            actual.keys() == saved.keys() and all(
                                math.isclose(actual[key], saved[key], rel_tol=0,
                                             abs_tol=1e-6) for key in saved),
                            'Fresh-process predictions changed')
            checks.append(
                dict(version=version, run=expected['run'], device=device, rows=len(saved),
                     history=history))
        require({device: sum(c['device'] == device for c in checks)
                 for device in selected_devices} == dict.fromkeys(selected_devices, 2),
                'Both model versions must reload on every selected device')
        return dict(versions=checks, devices=list(selected_devices))


def run_phase(name, root, profile):
    import torch

    torch.set_num_threads(2)
    return {'train': train, 'reload': reload}[name](root, profile)
