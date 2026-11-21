# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native customer-model stage integrity and registry alias guards."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test' /
          'wheel_mlops_pytorch_snowflake.py')
SPEC = importlib.util.spec_from_file_location('customer_snowflake', SCRIPT)
workload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workload)


class Stage:
    def __init__(self):
        self.files = {}

    def put(self, source, destination, *, auto_compress, overwrite):
        assert auto_compress is False and overwrite is False
        key = destination.rstrip('/') + '/' + Path(source).name
        if key in self.files:
            return [SimpleNamespace(status='SKIPPED')]
        self.files[key] = Path(source).read_bytes()
        return [SimpleNamespace(status='UPLOADED')]

    def get(self, source, destination):
        Path(destination,
             source.rsplit('/', 1)[-1]).write_bytes(self.files[source])
        return [SimpleNamespace(status='DOWNLOADED')]


@pytest.mark.parametrize('corruption',
                         [None, 'weights.pt', 'missing', 'inventory'])
def test_preprocessing_and_weights_recover_only_with_complete_matching_files(
        tmp_path, corruption):
    source = tmp_path / 'source'
    source.mkdir()
    for name in workload.FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    stage = Stage()
    session = SimpleNamespace(file=stage)
    workload.upload(session, source, '@DB.SCHEMA.STAGE/attempt/V1')
    if corruption == 'weights.pt':
        stage.files['@DB.SCHEMA.STAGE/attempt/V1/weights.pt'] = b'other weights'
    elif corruption == 'missing':
        del stage.files[
            '@DB.SCHEMA.STAGE/attempt/V1/preprocessing/contract.json']
    elif corruption == 'inventory':
        stage.files['@DB.SCHEMA.STAGE/attempt/V1/integrity.json'] = b'{}'
    restored = tmp_path / 'restored'
    if corruption is not None:
        with pytest.raises((AssertionError, KeyError)):
            workload.download(session, restored, '@DB.SCHEMA.STAGE/attempt/V1')
    else:
        workload.download(session, restored, '@DB.SCHEMA.STAGE/attempt/V1')
        assert all(
            (restored / name).read_bytes() == (source / name).read_bytes()
            for name in workload.FILES)
    with pytest.raises(AssertionError, match='not uploaded'):
        workload.upload(session, source, '@DB.SCHEMA.STAGE/attempt/V1')


@pytest.mark.parametrize('failure', [False, True])
def test_native_alias_promotion_restores_previous_version_on_failure(failure):
    state = {'alias': 'V1'}

    class Version:
        def __init__(self, name):
            self.version_name = name

        def set_alias(self, alias):
            assert alias == 'CHAMPION'
            if failure and self.version_name == 'V2':
                raise RuntimeError('native alias update failed')
            state['alias'] = self.version_name

        def unset_alias(self, alias):
            assert alias == 'CHAMPION' and state['alias'] == self.version_name
            state['alias'] = None

    def version(name):
        return Version(state['alias'] if name == 'CHAMPION' else name)

    registry = SimpleNamespace(
        get_model=lambda name: SimpleNamespace(version=version))
    if failure:
        with pytest.raises(RuntimeError, match='native alias update failed'):
            workload.promote(registry, 'CUSTOMER_MODEL', 'V2')
        assert state['alias'] == 'V1'
    else:
        workload.promote(registry, 'CUSTOMER_MODEL', 'V2')
        assert state['alias'] == 'V2'


def test_snowflake_module_loads_the_same_customer_recipe():
    assert workload.recipe().PHASES == workload.PHASES
    assert workload.recipe().METRICS == ('train/mse', 'validation/rmse')


def test_cpu_training_does_not_publish_cuda_only_registry_requirements(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('kumo.pytorch.api')
    # A CUDA-enabled installation does not make this CPU workload a GPU model.
    monkeypatch.setattr(torch.version, 'cuda', '13.2')
    calls = []

    class Tracking:
        def start_run(self, name):
            pass

        def log_metrics(self, metrics, *, step):
            assert set(metrics) == {'train/mse', 'validation/rmse'}

        def log_params(self, params):
            pass

        def log_model(self, model, **kwargs):
            assert next(model.parameters()).device.type == 'cpu'
            calls.append(kwargs)
            return SimpleNamespace(version_name=kwargs['version_name'])

        def end_run(self, *, status):
            assert status == 'FINISHED'

    version = SimpleNamespace(set_alias=lambda name: None)
    registry = SimpleNamespace(get_model=lambda name: SimpleNamespace(version=lambda name: version))
    stage = Stage()
    profile = SimpleNamespace(accelerator='cpu', model='CUSTOMER_MODEL', artifact_root='@DB.SCHEMA.STAGE/attempt')
    monkeypatch.setattr(workload, 'native', lambda p: (None, SimpleNamespace(file=stage), registry, Tracking()))
    result = workload.run_phase('train', tmp_path, profile)
    assert len(result['versions']) == len(calls) == 2
    for call in calls:
        assert call['options'] == {'cuda_version': None}
        assert call['pip_requirements'] == [f'torch=={torch.__version__.split("+")[0]}']
        assert call['target_platforms'] == ['SNOWPARK_CONTAINER_SERVICES']


@pytest.mark.parametrize('available', [False, True])
def test_gpu_registration_requires_real_cuda_and_matching_distribution(monkeypatch, available):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.version, 'cuda', '12.8')
    monkeypatch.setattr(workload, 'package_version', lambda name: '2.9.1')
    profile = SimpleNamespace(accelerator='gpu')
    with pytest.raises(AssertionError, match='CUDA device|CUDA-tagged'):
        workload.devices(profile)
    if available:
        monkeypatch.setattr(workload, 'package_version', lambda name: '2.9.1+cu128')
        assert workload.devices(profile) == ('cpu', 'cuda')
