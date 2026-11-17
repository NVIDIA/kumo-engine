# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-local bias switches: head signature, compatibility and plumbing."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.nn import MLP

from kumo_core.automl.auto_trainer import AutoTrainer, checkpoint_bias_options
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.training.orchestration import (
    BiasOptions,
    CallableTrainingOptions,
    validate_bias_options,
)
from kumo_core.types import TaskType

RELATION = ('events', 'entity_id', 'entities')


def config_dict(module):
    return {k: v for k, v in asdict(module.config()).items()}


def test_default_head_equals_base_construction_under_one_seed():
    # Base 1909c5b9 built the MLP without a `bias` argument (PyG default True).
    torch.manual_seed(3)
    head = NodePredHead(8, 1, num_layers=2, norm=None, param_keys=[])
    after_head = torch.rand(1)
    torch.manual_seed(3)
    mlp = MLP(in_channels=8, hidden_channels=8, out_channels=1, num_layers=2,
              dropout=0.0, act='relu', act_first=False, norm=None,
              norm_kwargs=None)
    after_mlp = torch.rand(1)
    assert torch.equal(after_head, after_mlp)
    state = head.mlp.state_dict()
    assert list(state) == list(mlp.state_dict())
    assert all(torch.equal(state[k], v) for k, v in mlp.state_dict().items())
    assert head.bias is True


def test_old_config_without_bias_reloads_as_biased():
    head = NodePredHead(8, 1, num_layers=2, norm=None, param_keys=[])
    old = {k: v for k, v in config_dict(head).items()
           if k not in ('_target_', 'bias')}
    reloaded = NodePredHead.from_config(old)
    assert reloaded.bias is True
    assert list(reloaded.state_dict()) == list(head.state_dict())
    reloaded.load_state_dict(head.state_dict(), strict=True)
    x = torch.randn(3, 8)
    assert torch.equal(reloaded(x), head(x))


def test_positional_arguments_map_by_signature_order():
    old_style = NodePredHead(4, 1, None, 1, 0.0, 'relu', False, None, [], False)
    assert old_style.bias is True and old_style.prediction_time_encodings is False
    bias_free = NodePredHead(4, 1, None, 1, 0.0, 'relu', False, None, [], False,
                             False)
    assert bias_free.bias is False
    assert all(lin.bias is None for lin in bias_free.mlp.lins)


def test_bias_free_head_has_no_bias_tensors_and_strict_loads_fail():
    biased = NodePredHead(4, 1, num_layers=2, norm=None, param_keys=[])
    free = NodePredHead(4, 1, num_layers=2, norm=None, param_keys=[],
                        bias=False)
    assert sorted(free.state_dict()) == ['mlp.lins.0.weight',
                                         'mlp.lins.1.weight']
    assert config_dict(free)['bias'] is False
    x = torch.rand(2, 4)
    # Bias-free ReLU MLP: positively homogeneous of degree one.
    torch.testing.assert_close(free(2 * x), 2 * free(x), rtol=0, atol=1e-6)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        free.load_state_dict(biased.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match='Missing key'):
        biased.load_state_dict(free.state_dict(), strict=True)


def test_bias_free_gnn_with_preprocessing_has_no_bias_tensors():
    gnn = HeteroGNN(4, 2, metadata=(['events', 'entities'], [RELATION]),
                    num_pre_mp_layers=1, aggr='mean', norm=None, bias=False)
    assert not [name for name, _ in gnn.named_parameters()
                if name.endswith('bias')]


def test_bias_options_values_and_serialization():
    assert BiasOptions().is_default
    assert not BiasOptions(head=False).is_default
    assert BiasOptions(gnn=False).to_dict() == {'gnn': False, 'head': True}
    assert BiasOptions.from_dict({'gnn': False}) == BiasOptions(gnn=False)
    with pytest.raises(TypeError):
        BiasOptions(gnn=0)
    with pytest.raises(ValueError, match='Unknown'):
        BiasOptions.from_dict({'gnn': False, 'encoder': False})
    with pytest.raises(TypeError):
        CallableTrainingOptions(bias_options={'gnn': False})
    assert CallableTrainingOptions().bias_options is None


def _plan(models):
    return SimpleNamespace(model_architecture=SimpleNamespace(model=models))


def test_switches_that_cannot_be_honored_raise():
    from kumoapi.model_plan import GraphTransformerModelPlan
    transformer = _plan([GraphTransformerModelPlan()])
    with pytest.raises(ValueError, match='GraphTransformer'):
        validate_bias_options(BiasOptions(gnn=False), transformer,
                              TaskType.REGRESSION)
    validate_bias_options(BiasOptions(head=False), transformer,
                          TaskType.BINARY_CLASSIFICATION)
    for task in (TaskType.FORECASTING, TaskType.STATIC_LINK_PREDICTION,
                 TaskType.TEMPORAL_LINK_PREDICTION):
        with pytest.raises(ValueError, match='NodePredHead'):
            validate_bias_options(BiasOptions(head=False), _plan([]), task)
    validate_bias_options(BiasOptions(gnn=False, head=False), _plan([]),
                          TaskType.BINARY_CLASSIFICATION)


def _stub_auto_trainer(task_type, options, checkpoint_path=None, models=()):
    trainer = AutoTrainer.__new__(AutoTrainer)
    trainer.task = SimpleNamespace(task_type=task_type)
    trainer.space = _plan(list(models))
    trainer.checkpoint_path = checkpoint_path
    trainer.bias_options = trainer._resolve_bias_options(options)
    return trainer


def _cfg():
    architecture = SimpleNamespace(
        model=[SimpleNamespace(channels=[4], aggregation=[['mean']],
                               dropout=[0.0])],
        activation=['relu'], normalization=[None],
        num_pre_message_passing_layers=[1],
        num_post_message_passing_layers=[2],
        prediction_time_encodings=[False])
    return SimpleNamespace(model_architecture=architecture,
                           neighbor_sampling=SimpleNamespace(
                               num_neighbors=[[2, 2]]))


@pytest.mark.parametrize('options', [None, BiasOptions(), BiasOptions(
    gnn=False), BiasOptions(head=False), BiasOptions(gnn=False, head=False)])
def test_auto_trainer_builds_requested_flags(options):
    trainer = _stub_auto_trainer(TaskType.BINARY_CLASSIFICATION, options)
    expected = options or BiasOptions()
    model = trainer._get_model(_cfg())
    head = trainer._get_head(_cfg())
    assert model.bias is expected.gnn and head.bias is expected.head
    # Materialize the lazy modules as the trainer does after graph loading.
    model.metadata = (['events', 'entities'], [RELATION])
    head.in_channels, head.out_channels, head.param_keys = 4, 1, []
    model_biases = [n for n, _ in model.named_parameters() if 'bias' in n]
    head_biases = [n for n, _ in head.named_parameters() if 'bias' in n]
    assert bool(model_biases) is expected.gnn
    assert bool(head_biases) is expected.head


def test_default_auto_trainer_modules_identical_to_base_under_one_seed():
    # Base 1909c5b9 passed no bias argument to either constructor.
    trainer = _stub_auto_trainer(TaskType.BINARY_CLASSIFICATION, None)
    torch.manual_seed(11)
    model = trainer._get_model(_cfg())
    model.metadata = (['events', 'entities'], [RELATION])
    torch.manual_seed(11)
    base = HeteroGNN(channels=4, num_mp_layers=2, num_pre_mp_layers=1,
                     aggr=['mean'], dropout=0.0, act='relu', norm=None,
                     metadata=(['events', 'entities'], [RELATION]))
    state, expected = model.state_dict(), base.state_dict()
    assert list(state) == list(expected)
    assert all(torch.equal(state[k], v) for k, v in expected.items())
    assert config_dict(model) == config_dict(base)


def test_auto_trainer_rejects_unsupported_switches_before_training():
    from kumoapi.model_plan import GraphTransformerModelPlan
    with pytest.raises(ValueError, match='GraphTransformer'):
        _stub_auto_trainer(TaskType.REGRESSION, BiasOptions(gnn=False),
                           models=[GraphTransformerModelPlan()])
    with pytest.raises(ValueError, match='NodePredHead'):
        _stub_auto_trainer(TaskType.FORECASTING, BiasOptions(head=False))
    with pytest.raises(TypeError):
        _stub_auto_trainer(TaskType.REGRESSION, {'gnn': False})


def _checkpoint(path, *, gnn, head, mixed=False):
    state = {
        'encoder.encoders.users.lin.bias': torch.zeros(1),
        'calibrator.weight_and_bias': torch.zeros(2, 1),
        'model.hetero_convs.0.<events___entity_id___entities>.lin.weight':
        torch.zeros(2, 2),
        'model.root_skips.0.lin_dict.events.weight': torch.zeros(2, 2),
        'model.post_convs.0.layer_dict.events.0.module.bias': torch.zeros(2),
        'head.mlp.lins.0.weight': torch.zeros(1, 2),
    }
    if gnn:
        state['model.hetero_convs.0.<events___entity_id___entities>.lin.bias'] = (
            torch.zeros(2))
        if not mixed:
            state['model.root_skips.0.lin_dict.events.bias'] = torch.zeros(2)
    if head:
        state['head.mlp.lins.0.bias'] = torch.zeros(1)
    torch.save({'state_dict': state}, path)
    return str(path)


@pytest.mark.parametrize('gnn,head', [(True, True), (False, False),
                                      (False, True), (True, False)])
def test_checkpoint_flags_read_only_model_and_head_linears(tmp_path, gnn, head):
    # Encoder, normalization and calibrator keys end in `bias` too; they
    # must not count (contract 8, re-review nit).
    path = _checkpoint(tmp_path / 'ckpt', gnn=gnn, head=head)
    assert checkpoint_bias_options(path) == BiasOptions(gnn=gnn, head=head)


def test_checkpoint_with_mixed_bias_parameters_is_rejected(tmp_path):
    path = _checkpoint(tmp_path / 'ckpt', gnn=True, head=True, mixed=True)
    with pytest.raises(ValueError, match='mixes'):
        checkpoint_bias_options(path)


def test_warm_start_inherits_and_rejects_explicit_conflicts(tmp_path):
    path = _checkpoint(tmp_path / 'ckpt', gnn=False, head=False)
    inherited = _stub_auto_trainer(TaskType.BINARY_CLASSIFICATION, None,
                                   checkpoint_path=path)
    assert inherited.bias_options == BiasOptions(gnn=False, head=False)
    same = _stub_auto_trainer(TaskType.BINARY_CLASSIFICATION,
                              BiasOptions(gnn=False, head=False),
                              checkpoint_path=path)
    assert same.bias_options == BiasOptions(gnn=False, head=False)
    with pytest.raises(ValueError, match='conflict with the warm-start'):
        _stub_auto_trainer(TaskType.BINARY_CLASSIFICATION, BiasOptions(),
                           checkpoint_path=path)
    biased = _checkpoint(tmp_path / 'biased', gnn=True, head=True)
    assert _stub_auto_trainer(TaskType.REGRESSION, None,
                              checkpoint_path=biased).bias_options == (
                                  BiasOptions())
