# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-local `BiasOptions`: API, bundle records, load and warm start."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from kumo_core.training.orchestration import BiasOptions
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from walk_binary_fixtures import (
    BINARY_QUERY,
    binary_graph,
    binary_plan,
    write_binary_tables,
)

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer._spec import (
    MANIFEST_FILE,
    SPEC_FILE,
    read_training_spec,
    write_training_spec,
)

GNN = 'kumo_core.modeling.gnn.hetero.HeteroGNN'
HEAD = 'kumo_core.modeling.heads.node_prediction.NodePredHead'


def test_bias_options_is_exported_and_type_checked():
    assert kumo.BiasOptions is BiasOptions
    assert 'BiasOptions' in kumo.__all__
    assert kumo.Trainer(ModelPlan()).bias_options is None
    options = BiasOptions(gnn=False, head=False)
    assert kumo.Trainer(ModelPlan(), bias_options=options).bias_options is (
        options)
    with pytest.raises(TypeError):
        kumo.Trainer(ModelPlan(), bias_options={'gnn': False})


def _bundle(root, *, gnn, head, recorded):
    bundle = root / 'published-model'
    (bundle / 'model').mkdir(parents=True)
    config = json.dumps({'model': {'_target_': GNN, 'bias': gnn},
                         'head': {'_target_': HEAD, 'bias': head}})
    (bundle / 'model' / 'config.json').write_text(config)
    manifest = {'artifacts': [{
        'role': 'trainer_config', 'relative_path': 'model/config.json',
        'size_bytes': len(config),
        'sha256': hashlib.sha256(config.encode()).hexdigest()}]}
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest))
    write_training_spec(bundle, predictive_query=BINARY_QUERY,
                        model_plan=ModelPlan(), snapshot_id='snap',
                        training_table_generation=TrainingTableGenerationPlan(),
                        bias_options=recorded)
    return bundle


def test_spec_records_only_non_default_options(tmp_path):
    default = _bundle(tmp_path / 'a', gnn=True, head=True, recorded=None)
    assert 'bias_options' not in json.loads((default / SPEC_FILE).read_text())
    assert read_training_spec(default).bias_options is None
    free = _bundle(tmp_path / 'b', gnn=False, head=False,
                   recorded={'gnn': False, 'head': False})
    assert json.loads((free / SPEC_FILE).read_text())['bias_options'] == {
        'gnn': False, 'head': False}
    assert read_training_spec(free).bias_options == {'gnn': False,
                                                     'head': False}


def test_load_restores_recorded_options(tmp_path):
    free = _bundle(tmp_path / 'free', gnn=False, head=True,
                   recorded={'gnn': False, 'head': True})
    assert kumo.Trainer.load(free).bias_options == BiasOptions(gnn=False)
    default = _bundle(tmp_path / 'default', gnn=True, head=True,
                      recorded=None)
    assert kumo.Trainer.load(default).bias_options is None


@pytest.mark.parametrize('config,recorded', [
    ((False, False), None),
    ((True, True), {'gnn': False, 'head': False}),
    ((False, True), {'gnn': False, 'head': False}),
])
def test_load_rejects_spec_that_disagrees_with_trained_config(
        tmp_path, config, recorded):
    bundle = _bundle(tmp_path, gnn=config[0], head=config[1],
                     recorded=recorded)
    with pytest.raises(ValueError, match='bias options'):
        kumo.Trainer.load(bundle)


# End-to-end fits ##############################################################


@pytest.fixture(scope='module')
def binary_snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp('bias-options')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    try:
        data = root / 'data'
        data.mkdir()
        write_binary_tables(data)
        connector = kumo.FileConnector(root_dir=str(data))
        graph = binary_graph(kumo, connector)
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=BINARY_QUERY)
        yield snapshot, query.generate_training_table(snapshot), root
    finally:
        kumo.configure(scratch_dir=previous)


def _state_keys(bundle_dir):
    path = Path(bundle_dir) / 'model' / 'checkpoint'
    state = torch.load(path, map_location='cpu',
                       weights_only=False)['state_dict']
    return set(state)


def _fit(snapshot_table, *, bias_options=None, warm_start=None, epochs=1):
    snapshot, table, _ = snapshot_table
    trainer = kumo.Trainer(binary_plan(epochs=epochs, steps=2),
                           accelerator='cpu', bias_options=bias_options)
    trainer.fit(table, snapshot=snapshot, warm_start=warm_start)
    return trainer


def _bundle_dir(trainer):
    result = trainer.result
    assert result is not None
    return result.bundle_dir


def test_bias_free_fit_records_and_reloads(binary_snapshot):
    trainer = _fit(binary_snapshot, bias_options=BiasOptions(gnn=False,
                                                             head=False))
    bundle = _bundle_dir(trainer)
    config = json.loads((bundle / 'model' / 'config.json').read_text())
    assert config['model']['bias'] is False and config['head']['bias'] is False
    keys = _state_keys(bundle)
    assert not [k for k in keys if k.startswith(('model.', 'head.'))
                and k.endswith('.bias')]
    assert [k for k in keys if k.startswith('encoder.')]
    assert json.loads((bundle / SPEC_FILE).read_text())['bias_options'] == {
        'gnn': False, 'head': False}
    copy = binary_snapshot[2] / 'bias-free-copy'
    shutil.copytree(bundle, copy)
    assert kumo.Trainer.load(copy).bias_options == BiasOptions(gnn=False,
                                                               head=False)


def test_default_fit_keeps_spec_bytes_and_biases(binary_snapshot):
    trainer = _fit(binary_snapshot)
    bundle = _bundle_dir(trainer)
    config = json.loads((bundle / 'model' / 'config.json').read_text())
    assert config['model']['bias'] is True and config['head']['bias'] is True
    assert 'bias_options' not in json.loads((bundle / SPEC_FILE).read_text())
    keys = _state_keys(bundle)
    assert 'head.mlp.lins.0.bias' in keys
    assert kumo.Trainer.load(bundle).bias_options is None


def test_warm_start_inherits_flags_and_rejects_explicit_conflict(
        binary_snapshot):
    source = _fit(binary_snapshot, bias_options=BiasOptions(gnn=False,
                                                            head=False))
    inherited = _fit(binary_snapshot, warm_start=source)
    config = json.loads((_bundle_dir(inherited) / 'model' / 'config.json').read_text())
    assert config['model']['bias'] is False and config['head']['bias'] is False
    assert json.loads((_bundle_dir(inherited) / SPEC_FILE).read_text())['bias_options'] == {
                           'gnn': False, 'head': False}
    with pytest.raises(ValueError, match='conflict with the warm-start'):
        _fit(binary_snapshot, bias_options=BiasOptions(), warm_start=source)
