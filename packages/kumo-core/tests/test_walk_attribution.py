# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kumo_core.explain.walk_attribution import (
    WalkAttributionConfig, WalkCapture, UnsupportedWalkAttribution,
)
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead


def fixture():
    module = torch.nn.Module()
    module.entity_table_name = 'product'
    module.model = HeteroGNN(1, 1, metadata=(['product'], []), aggr='sum',
                            learn_skip=True, norm=None, bias=True)
    module.head = NodePredHead(1, 1, num_layers=1, norm=None, param_keys=[])
    module.double().eval()
    with torch.no_grad():
        module.model.root_skips[0].lin_dict['product'].weight.fill_(2)
        module.model.root_skips[0].lin_dict['product'].bias.fill_(1)
        module.head.mlp.lins[0].weight.fill_(3)
        module.head.mlp.lins[0].bias.fill_(4)
    return module


def forward(module):
    return module.head(module.model({'product': torch.tensor([[2.]], dtype=torch.float64)}, {})['product'])


def test_capture_bias_ledger_and_no_mutation():
    module = fixture()
    expected = forward(module).detach().clone()
    with WalkCapture(module) as capture:
        actual = forward(module)
        result = capture.result()
    assert torch.equal(actual, expected)
    # y = 3*(2*2 + 1) + 4 = 19, source=12, biases=7.
    assert result['score'] == 19
    assert sum(w['relevance'] for w in result['walks']) == pytest.approx(12)
    assert result['bias_relevance'] == pytest.approx(7)
    assert result['accounting_error'] == pytest.approx(0, abs=1e-12)
    assert not capture.handles
    assert all(not m._forward_hooks for m in module.modules())


def test_budget_reports_omitted_signed_mass():
    module = fixture()
    with WalkCapture(module, WalkAttributionConfig(max_walks=1)) as capture:
        forward(module)
        result = capture.result()
    assert result['truncated'] and not result['complete']
    assert result['omitted_signed_relevance'] == pytest.approx(15)
    assert result['accounting_error'] == pytest.approx(0, abs=1e-12)


def test_unsupported_before_hooks():
    module = fixture()
    module.model.act = 'gelu'
    with pytest.raises(UnsupportedWalkAttribution):
        with WalkCapture(module):
            pass
    assert all(not m._forward_hooks for m in module.modules())


@pytest.mark.parametrize('kwargs', [{'gamma': float('nan')}, {'epsilon': 0},
                                    {'max_walks': True}, {'output_index': -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        WalkAttributionConfig(**kwargs)


def test_signed_readout_and_bias_cancellation():
    module = fixture()
    with torch.no_grad():
        module.head.mlp.lins[0].weight.fill_(-3)
    with WalkCapture(module) as capture:
        forward(module)
        result = capture.result()
    assert result['score'] == -11
    assert sum(w['relevance'] for w in result['walks']) == pytest.approx(-12)
    assert result['bias_relevance'] == pytest.approx(1)
    assert result['accounting_error'] == pytest.approx(0, abs=1e-12)


def test_capture_failure_preserves_second_prediction():
    module = fixture()
    with WalkCapture(module) as capture:
        first = forward(module)
        second = forward(module)
        assert torch.equal(first, second)
        with pytest.raises(ValueError, match='capture failed'):
            capture.result()
    assert all(not m._forward_hooks for m in module.modules())


def test_inspection_has_no_hooks():
    from kumo_core.explain.walk_attribution import inspect_walk_capability
    module = fixture()
    result = inspect_walk_capability(module)
    assert result['supported'] and result['device_qualified']
    assert all(not m._forward_hooks for m in module.modules())


def test_selected_logit_in_multiseed_forward():
    module = fixture()
    module.head = NodePredHead(1, 2, num_layers=1, norm=None, param_keys=[]).double().eval()
    with torch.no_grad():
        module.head.mlp.lins[0].weight.copy_(torch.tensor([[3.], [-3.]]))
        module.head.mlp.lins[0].bias.fill_(4)
    with WalkCapture(module, WalkAttributionConfig(output_index=1)) as capture:
        module.head(module.model({'product': torch.tensor([[2.], [5.]], dtype=torch.float64)}, {})['product'])
        result = capture.result(seed_index=1)
    assert result['score'] == -29
    assert {w['source_occurrence'] for w in result['walks']} == {1}
    assert sum(w['relevance'] for w in result['walks']) == pytest.approx(-30)
    assert result['bias_relevance'] == pytest.approx(1)
