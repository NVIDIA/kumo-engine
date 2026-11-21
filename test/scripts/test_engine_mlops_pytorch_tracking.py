# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customer MLflow calls retain the selected platform's credential scope."""

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test' /
          'wheel_mlops_pytorch.py')
SPEC = importlib.util.spec_from_file_location('customer_tracking', SCRIPT)
workload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workload)


@pytest.mark.parametrize('platform', ['databricks', 'sagemaker'])
@pytest.mark.parametrize('raises', [False, True])
def test_customer_calls_use_platform_identity_until_scope_exit(
        monkeypatch, tmp_path, platform, raises):
    monkeypatch.setenv('KUMO_BYOM_TEST_IDENTITY', 'ambient')
    client = object()
    observed = []

    @contextmanager
    def clients():
        previous = os.environ['KUMO_BYOM_TEST_IDENTITY']
        os.environ['KUMO_BYOM_TEST_IDENTITY'] = 'configured'
        try:
            yield client
        finally:
            os.environ['KUMO_BYOM_TEST_IDENTITY'] = previous

    def set_experiment(name):
        assert name == 'customer-experiment'
        assert os.environ['KUMO_BYOM_TEST_IDENTITY'] == 'configured'
        observed.append(name)

    mlflow = ModuleType('mlflow')
    mlflow.set_experiment = set_experiment
    mlflow.MlflowClient = lambda: pytest.fail('Do not replace the scoped client'
                                              )
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    cfg = SimpleNamespace(tracking=SimpleNamespace(client=clients))
    profile = SimpleNamespace(config=dict(platform=platform), accelerator='cpu',
                              experiment='customer-experiment',
                              model='customer-model',
                              settings=lambda register: cfg)

    def call():
        with workload.tracking(tmp_path, profile) as (actual, model):
            assert actual is client and model == 'customer-model'
            assert os.environ['KUMO_BYOM_TEST_IDENTITY'] == 'configured'
            if raises:
                raise RuntimeError('customer training failed')

    if raises:
        with pytest.raises(RuntimeError, match='customer training failed'):
            call()
    else:
        call()
    assert observed == ['customer-experiment']
    assert os.environ['KUMO_BYOM_TEST_IDENTITY'] == 'ambient'


def test_local_customer_tracking_uses_the_attempt_database(
        monkeypatch, tmp_path):
    client = object()
    calls = []
    mlflow = ModuleType('mlflow')
    mlflow.set_tracking_uri = lambda uri: calls.append(('tracking', uri))
    mlflow.set_registry_uri = lambda uri: calls.append(('registry', uri))
    mlflow.get_experiment_by_name = lambda name: object()
    mlflow.set_experiment = lambda name: calls.append(('experiment', name))
    mlflow.MlflowClient = lambda: client
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    with workload.tracking(tmp_path, None) as (actual, model):
        assert actual is client and model == 'customer-regression'
    uri = f'sqlite:///{tmp_path / "customer-tracking.db"}'
    assert calls == [('tracking', uri), ('registry', uri),
                     ('experiment', 'customer-regression')]
