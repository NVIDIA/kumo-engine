#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customer PyTorch lifecycle using Snowflake's native experiment and registry APIs.

Stage this module beside wheel_mlops_pytorch.py and select it with --workload.
The registered model is an ordinary torch.nn.Linear; Engine Trainer is not used.
"""

import hashlib
import importlib.util
import json
import math
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

PHASES = ('train', 'reload')
FILES = ('preprocessing/contract.json', 'preprocessing/fitted-state.json',
         'preprocessing/task.json', 'preprocessing/manifest.json', 'weights.pt',
         'expected.json')


@cache
def recipe():
    path = Path(__file__).with_name('wheel_mlops_pytorch.py')
    spec = importlib.util.spec_from_file_location('customer_pytorch_recipe',
                                                  path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the customer model recipe at {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def upload(session, directory, stage):
    """Commit this fixture's preprocessing, weights and predictions to a stage."""
    checksums = {name: digest(directory / name) for name in FILES}
    write(directory / 'integrity.json', checksums)
    for name in (*FILES, 'integrity.json'):
        path = Path(name)
        target = stage.rstrip('/') + ('/' + path.parent.as_posix()
                                      if path.parent != Path('.') else '')
        results = session.file.put(str(directory / name), target,
                                   auto_compress=False, overwrite=False)
        require(results and all(r.status == 'UPLOADED' for r in results),
                f'Customer artifact was not uploaded: {name}')


def download(session, directory, stage):
    """Reject missing, mixed or changed artifact files before loading weights."""
    def get(name):
        destination = (directory / name).parent
        destination.mkdir(parents=True, exist_ok=True)
        results = session.file.get(
            stage.rstrip('/') + '/' + name, str(destination))
        require(results and all(r.status == 'DOWNLOADED' for r in results),
                f'Customer artifact was not downloaded: {name}')

    get('integrity.json')
    checksums = json.loads((directory / 'integrity.json').read_text())
    require(set(checksums) == set(FILES), 'Customer artifact inventory changed')
    for name in FILES:
        get(name)
        require(
            digest(directory / name) == checksums[name],
            f'Customer artifact checksum changed: {name}')
    return directory


def native(profile):
    require(profile is not None and profile.config['platform'] == 'snowflake',
            'This workload requires a Snowflake platform profile')
    require(profile.accelerator == 'cpu',
            'This workload requires a CPU profile')
    from snowflake.ml.registry import Registry

    cfg = profile.settings(False)
    session = cfg.tracking.session
    location = dict(database_name=profile.config['database'],
                    schema_name=profile.config['schema'])
    registry = Registry(session, **location)
    tracking = profile.history_api()(session, model_registry=registry,
                                     **location)
    tracking.set_experiment(profile.experiment, **location)
    return cfg, session, registry, tracking


def promote(registry, name, version):
    model = registry.get_model(name)
    previous = model.version('CHAMPION')
    if str(previous.version_name) != version:
        previous.unset_alias('CHAMPION')
        try:
            model.version(version).set_alias('CHAMPION')
        except Exception:
            previous.set_alias('CHAMPION')
            raise
    require(
        str(model.version('CHAMPION').version_name) == version,
        'Snowflake alias resolved another customer model version')


def train(root, profile):
    import torch

    from kumo.pytorch import api

    shared = recipe()
    cfg, session, registry, tracking = native(profile)
    contract, dataset, requests = shared.inputs()
    versions = []
    python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests, fit_split='train') as task:
            centered = torch.cat(
                [shared.features(b) for b in task.batches(split='train')])
            require(
                abs(centered.mean().item()) < 1e-6,
                'Input normalization did not fit only the training split')
            for seed in (0, 1):
                torch.manual_seed(seed)
                model = torch.nn.Linear(1, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                version, run_name = f'V{seed + 1}', f'CUSTOMER_FIT_{seed}'
                directory = root / 'scratch-train' / version
                directory.mkdir(parents=True)
                task.to_artifact().save(directory / 'preprocessing')
                tracking.start_run(run_name)
                try:
                    for epoch in range(2):
                        for batch in task.batches(split='train', epoch=epoch):
                            labels = batch.targets['label']['values'].float(
                            ).reshape(-1, 1)
                            loss = torch.nn.functional.mse_loss(
                                model(shared.features(batch)), labels)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        with torch.no_grad():
                            squared = [
                                (model(shared.features(batch)).flatten() -
                                 batch.targets['label']['values'].float()
                                 ).square()
                                for batch in task.batches(split='validation')
                            ]
                            rmse = torch.cat(squared).mean().sqrt().item()
                        tracking.log_metrics(
                            dict(zip(shared.METRICS, (loss.item(), rmse))),
                            step=epoch)
                    model.eval()
                    identity = dict(
                        customer_run=run_name,
                        customer_preprocessing=task.fitted_state.fingerprint)
                    tracking.log_params(
                        dict(identity, customer_model=profile.model,
                             customer_version=version))
                    api.save_weights(directory / 'weights.pt',
                                     model.state_dict(), task.fitted_state)
                    with plan.materialize(requests, fitted=task.fitted_state,
                                          inference=True) as scoring:
                        write(directory / 'expected.json',
                              shared.predictions(scoring, model))
                    upload(session, directory,
                           profile.artifact_root + '/customer/' + version)
                    logged = tracking.log_model(
                        model, model_name=profile.model, version_name=version,
                        metrics=identity, sample_input_data=torch.zeros((2, 1)),
                        target_platforms=['SNOWPARK_CONTAINER_SERVICES'],
                        python_version=python_version, pip_requirements=[
                            f'torch=={torch.__version__.split("+")[0]}'
                        ],
                        options={'cuda_version': torch.version.cuda or '12.6'})
                    require(
                        str(logged.version_name) == version,
                        'Snowflake registered another version')
                except BaseException:
                    tracking.end_run(status='FAILED')
                    raise
                else:
                    tracking.end_run(status='FINISHED')
                versions.append(
                    dict(version=version, run=run_name,
                         fingerprint=task.fitted_state.fingerprint))
    registry.get_model(profile.model).version(
        versions[0]['version']).set_alias('CHAMPION')
    state = dict(model=profile.model, versions=versions)
    write(root / 'customer-state.json', state)
    result = session.file.put(str(root / 'customer-state.json'),
                              profile.artifact_root, auto_compress=False,
                              overwrite=False)
    require(result and all(r.status == 'UPLOADED' for r in result),
            'Customer lifecycle state was not persisted')
    return state


def reload(root, profile):
    import torch

    from kumo.pytorch import api

    shared = recipe()
    cfg, session, registry, tracking = native(profile)
    control = root / 'scratch-reload' / 'control'
    control.mkdir(parents=True)
    result = session.file.get(profile.artifact_root + '/customer-state.json',
                              str(control))
    require(result and all(r.status == 'DOWNLOADED' for r in result),
            'Customer lifecycle state was not recovered from the stage')
    state = json.loads((control / 'customer-state.json').read_text())
    require(state['model'] == profile.model and len(state['versions']) == 2,
            'Recovered another or incomplete customer model lifecycle')
    _, dataset, requests = shared.inputs()
    checks = []
    for expected in state['versions']:
        version = expected['version']
        record = profile.record(cfg, SimpleNamespace(run_id=expected['run']))
        require(
            record['params'].get('customer_preprocessing')
            == expected['fingerprint']
            and record['params'].get('customer_version') == version,
            'Native tracking describes another model/preprocessing state')
        history = {
            name: record['history'].get(name, [])
            for name in shared.METRICS
        }
        require(
            all({step
                 for step, value in points} == {0, 1} and all(
                     math.isfinite(value) for step, value in points)
                for points in history.values()),
            'Missing native multi-epoch metric history')
        registered = registry.get_model(profile.model).version(version)
        metrics = registered.show_metrics()
        require(
            metrics.get('customer_run') == expected['run'] and
            metrics.get('customer_preprocessing') == expected['fingerprint'],
            'Native registry identifies another customer training run')
        directory = download(session, root / 'scratch-reload' / version,
                             profile.artifact_root + '/customer/' + version)
        artifact = api.Artifact.load(directory / 'preprocessing')
        require(artifact.fitted_state.fingerprint == expected['fingerprint'],
                'Recovered another fitted preprocessing state')
        model = torch.nn.Linear(1, 1)
        model.load_state_dict(
            api.load_weights(directory / 'weights.pt', artifact.fitted_state))
        model.eval()
        saved = json.loads((directory / 'expected.json').read_text())
        promote(registry, profile.model, version)
        published = registry.get_model(
            profile.model).version('CHAMPION').load(options={'use_gpu': False})
        require(isinstance(published, torch.nn.Module),
                'Native registry did not load a PyTorch model')
        with api.prepare(artifact.contract, dataset, api.InMemoryEngine(),
                         inference=True) as plan:
            with plan.materialize(requests, fitted=artifact.fitted_state,
                                  inference=True) as task:
                for actual in (shared.predictions(task, model),
                               shared.predictions(task, published.eval())):
                    require(
                        actual.keys() == saved.keys() and all(
                            math.isclose(actual[key], saved[key], rel_tol=0,
                                         abs_tol=1e-6) for key in saved),
                        'Fresh-process Snowflake predictions changed')
        checks.append(
            dict(version=version, run=expected['run'], rows=len(saved),
                 history=history))
    return dict(versions=checks)


def run_phase(name, root, profile):
    import torch

    torch.set_num_threads(2)
    return {'train': train, 'reload': reload}[name](root, profile)
