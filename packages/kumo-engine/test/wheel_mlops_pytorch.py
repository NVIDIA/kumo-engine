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


def features(batch):
    return batch.inputs['user']['value']['values'].float().reshape(-1, 1)


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
    else:
        if profile.config['platform'] == 'snowflake':
            raise ImportError(
                'This example uses native MLflow registration; supply a workload '
                'using the Snowflake native model registry for Snowflake')
        require(profile.accelerator == 'cpu',
                'This example requires a CPU profile')
        # Configure the selected platform's URI and credential target, without
        # using Engine Trainer or its Kumo model serialization.
        profile.settings(False)
        experiment, model = profile.experiment, profile.model
    mlflow.set_experiment(experiment)
    return MlflowClient(), model


def predictions(task, model):
    import torch

    result = {}
    with torch.no_grad():
        for batch in task.batches():
            require(not batch.targets, 'Inference emitted training targets')
            values = model(features(batch)).flatten().tolist()
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

    client, name = tracking(root, profile)
    contract, dataset, requests = inputs()
    versions = []
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests, fit_split='train') as task:
            centered = torch.cat(
                [features(batch) for batch in task.batches(split='train')])
            require(
                abs(centered.mean().item()) < 1e-6,
                'Derived input normalization was not fitted on the training split'
            )
            for seed in (0, 1):
                torch.manual_seed(seed)
                model = torch.nn.Linear(1, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                directory = root / 'scratch-train' / str(seed)
                directory.mkdir(parents=True)
                task.to_artifact().save(directory / 'preprocessing')
                with mlflow.start_run() as run:
                    for epoch in range(2):
                        for batch in task.batches(split='train', epoch=epoch):
                            labels = batch.targets['label']['values'].float(
                            ).reshape(-1, 1)
                            loss = torch.nn.functional.mse_loss(
                                model(features(batch)), labels)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        with torch.no_grad():
                            squared = [
                                (model(features(batch)).flatten() -
                                 batch.targets['label']['values'].float()
                                 ).square()
                                for batch in task.batches(split='validation')
                            ]
                            rmse = torch.cat(squared).mean().sqrt().item()
                        mlflow.log_metrics(
                            dict(zip(METRICS, (loss.item(), rmse))), step=epoch)
                    model.eval()
                    api.save_weights(directory / 'weights.pt',
                                     model.state_dict(), task.fitted_state)
                    with plan.materialize(requests, fitted=task.fitted_state,
                                          inference=True) as scoring:
                        expected = predictions(scoring, model)
                    write(directory / 'expected.json', expected)
                    mlflow.log_artifacts(str(directory),
                                         artifact_path='customer')
                    logged = mlflow.pytorch.log_model(
                        model, name='model', input_example=np.zeros(
                            (2, 1), dtype=np.float32))
                    registered = mlflow.register_model(logged.model_uri, name)
                    mlflow.set_tag('customer.preprocessing',
                                   task.fitted_state.fingerprint)
                versions.append(
                    dict(version=str(registered.version), run=run.info.run_id,
                         fingerprint=task.fitted_state.fingerprint))
    require(versions[0]['version'] != versions[1]['version'],
            'Registry reused a version')
    client.set_registered_model_alias(name, 'champion', versions[0]['version'])
    state = dict(model=name, versions=versions)
    write(root / 'customer-state.json', state)
    return state


def reload(root, profile):
    import mlflow.pytorch
    import torch

    from kumo.pytorch import api

    client, name = tracking(root, profile)
    state = json.loads((root / 'customer-state.json').read_text())
    require(state['model'] == name, 'Recovered another registry model')
    _, dataset, requests = inputs()
    checks = []
    for expected in state['versions']:
        version = expected['version']
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
        require(artifact.fitted_state.fingerprint == expected['fingerprint'],
                'Recovered different preprocessing state')
        model = torch.nn.Linear(1, 1)
        model.load_state_dict(
            api.load_weights(directory / 'weights.pt', artifact.fitted_state))
        model.eval()
        saved = json.loads((directory / 'expected.json').read_text())
        client.set_registered_model_alias(name, 'champion', version)
        require(
            str(client.get_model_version_by_alias(
                name, 'champion').version) == version,
            'Alias promotion resolved another version')
        # The registry loads a native PyTorch model; Engine Trainer is never involved.
        published = mlflow.pytorch.load_model(f'models:/{name}@champion',
                                              map_location='cpu')
        with api.prepare(artifact.contract, dataset, api.InMemoryEngine(),
                         inference=True) as plan:
            with plan.materialize(requests, fitted=artifact.fitted_state,
                                  inference=True) as task:
                for actual in (predictions(task,
                                           model), predictions(task,
                                                               published)):
                    require(
                        actual.keys() == saved.keys() and all(
                            math.isclose(actual[key], saved[key], rel_tol=0,
                                         abs_tol=1e-6) for key in saved),
                        'Fresh-process predictions changed')
        checks.append(
            dict(version=version, run=expected['run'], rows=len(saved),
                 history=history))
    return dict(versions=checks)


def run_phase(name, root, profile):
    import torch

    torch.set_num_threads(2)
    return {'train': train, 'reload': reload}[name](root, profile)
