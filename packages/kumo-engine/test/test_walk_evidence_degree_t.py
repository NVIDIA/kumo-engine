# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GNN-GI degree-T rows through native capture, bundles and offline replay.

Built on P2's frozen fixture helper. Fixture B1 hand values: eight walks
summing to `f = 9`; `s = -1` negates every walk exactly.
"""
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.explain.walk_attribution import (
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
    inspect_walk_capability,
)
from kumo_core.explain.walk_graph import (
    float32_matmul_precision,
    mkldnn_matmul_fp32_precision,
)
from kumo_core.explain.walk_schema import validate_v2_row
from walk_binary_fixtures import (
    b1_sample,
    binary_scoring_module,
    capture_record,
    comparable_row,
    derive_offline,
    emitted_frame,
    native_offline_parity,
)

from kumo.engine.evidence import PredictionEvidenceBundle
from kumo.engine.evidence.bundle import PredictionEvidenceSource

METHOD = 'hetero_gnn_gi_degree_t_v1'
B1_SCORES = [2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 0.5, 0.5]  # ranked by |relevance|


def config(**options):
    return WalkAttributionConfig(**{'method': METHOD, 'feature_vectors': True, 'bags': True,
                                    'top_k': 10000, **options})


def no_hooks(module):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


@pytest.mark.parametrize('selected,sign', [(None, 1), (True, 1), (False, -1)])
def test_native_binary_rows_claim_the_b1_decomposition(tmp_path, selected, sign):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    captured = capture_record(module, tmp_path, config(selected_class=selected))
    record, row = captured.record, captured.record['walk_attribution']
    assert row['schema'] == 'walk_attribution/v2' and row['status'] == 'available', row
    assert row['method'] == METHOD and row['coordinate'] == 'selected_class_logit'
    binary = row['output']['binary']
    assert binary['logit_sign'] == sign and row['score'] == sign * 9.0
    assert [w['relevance'] for w in row['walks']] == [sign * v for v in B1_SCORES]
    exactness = row['exactness']
    assert exactness['decomposition'] == 'exact_degree_t' and exactness['degree_t_claim']
    assert exactness['piece'] == 'interior' and exactness['min_abs_gated_preactivation'] == 3.5
    assert 'bias_free_gnn' in exactness['qualification']
    assert row['ledger']['precision_residual'] == 0.0
    assert row['ledger']['accounting_error'] == 0.0
    assert row['coverage']['bags'] == 'complete' and len(row['bags']) == 8
    assert row['diagnostics'][METHOD]['f_piece'] == sign * 9.0
    # The record still binds the emitted probability of encoded class 1.
    emitted = emitted_frame(module, captured.logits.detach())
    assert record['prediction'] == float(emitted.true_prob[0])
    selected_probability = emitted.true_prob if sign == 1 else emitted.false_prob
    assert binary['emitted_probability_selected'] == float(selected_probability[0])
    assert captured.manifest['complete'] is True and no_hooks(module)
    validate_v2_row(row, b1_sample())


@pytest.mark.parametrize('selected', [True, False])
def test_offline_replay_matches_the_native_row(tmp_path, selected):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    native, offline, bundle = native_offline_parity(module, config(selected_class=selected),
                                                    tmp_path)
    assert native['status'] == 'available' and native['exactness']['degree_t_claim']
    assert comparable_row(native) == comparable_row(offline)  # bitwise, bags included
    assert offline['replay_logit_abs_diff'] == 0.0
    assert offline['replay_probability_abs_diff'] == 0.0
    assert bundle.validate()['status_counts'] == {'available': 1}


def test_save_and_reload_parity(tmp_path):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    selected = config(selected_class=False)
    captured = capture_record(module, tmp_path / 'native', selected)
    torch.save(module.state_dict(), tmp_path / 'module.pt')
    reloaded = binary_scoring_module(calibration=(0.1, 0.2), head_weight=3.0)
    reloaded.load_state_dict(torch.load(tmp_path / 'module.pt', weights_only=True))
    offline = next(derive_offline(captured, reloaded, selected, tmp_path / 'offline').iter_rows())
    native = captured.record['walk_attribution']
    assert comparable_row(offline) == comparable_row(native)
    round_trip = json.loads(json.dumps(native, allow_nan=False))
    validate_v2_row(round_trip, b1_sample())
    assert round_trip == native


@pytest.mark.parametrize('options,reason', [({'gnn_bias': True}, 'gnn_bias_present'),
                                            ({'head_bias': True}, 'head_bias_present')])
def test_biased_models_are_refused_before_hooks(tmp_path, options, reason):
    reference = binary_scoring_module(**options)
    expected = float(emitted_frame(reference, reference.common_step().detach()).true_prob[0])
    module = binary_scoring_module(**options)
    with pytest.raises(UnsupportedWalkAttribution, match=f'^{reason}: '):
        capture_record(module, tmp_path / 'raise', config())
    assert no_hooks(module)
    captured = capture_record(module, tmp_path / 'record', config(), on_unsupported='record')
    row = captured.record['walk_attribution']
    assert row['status'] == 'unsupported' and row['reason'].startswith(f'{reason}: ')
    assert captured.record['prediction'] == expected  # predictions continue unchanged
    assert captured.manifest['complete'] is False and no_hooks(module)
    report = inspect_walk_capability(module, config=config())
    assert not report['supported'] and report['reason'].startswith(f'{reason}: ')


def test_budget_exhaustion_is_incomplete_without_claim_or_bags(tmp_path):
    captured = capture_record(binary_scoring_module(), tmp_path, config(max_walks=6, top_k=2),
                              on_unsupported='record')
    row = captured.record['walk_attribution']
    assert row['status'] == 'incomplete' and row['coverage']['status'] == 'budget_exhausted'
    exactness = row['exactness']
    assert exactness['reasons'] == ['enumeration_incomplete']
    assert not exactness['degree_t_claim'] and exactness['decomposition'] == 'piecewise_gnn_gi'
    assert row['bags'] is None and row['coverage']['bags'] == 'omitted_incomplete_enumeration'
    ledger = row['ledger']
    shown = ledger['displayed_walk_relevance'] + ledger['omitted_display_relevance']
    # Exact signed prefix mass: the omitted walks of the complete decomposition.
    assert ledger['omitted_search_relevance'] == 9.0 - shown
    assert captured.manifest['complete'] is False
    validate_v2_row(row, b1_sample())


def _snapshot(module):
    """State dict tensors cloned; non-tensor extra state (the calibrator's
    fitted flag) copied as JSON."""
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else json.dumps(v)
            for k, v in module.state_dict().items()}


def _same(a, b):
    return set(a) == set(b) and all(
        torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k] for k in a)


def test_capture_leaves_predictions_parameters_and_calibrator_unchanged(tmp_path):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    plain = module.common_step().detach().clone()
    state = _snapshot(module)
    captured = capture_record(module, tmp_path, config())
    assert torch.equal(captured.logits.detach(), plain)
    assert captured.record['prediction'] == float(emitted_frame(module, plain).true_prob[0])
    assert _same(_snapshot(module), state)
    assert module.calibrator.is_fitted and no_hooks(module)


def test_bundles_with_degree_t_rows_validate(tmp_path):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    run_root = tmp_path / 'run'
    captured = capture_record(module, run_root / 'xai' / 'local', config())
    emitted = emitted_frame(module, captured.logits.detach())
    frame = pd.DataFrame({
        'PREDICTION_ID': [captured.record['prediction_id']], 'ENTITY': [17],
        'TASK_ROW_ID': [8], 'TARGET_PRED': pd.array([None], dtype='boolean'),
        'False_PROB': np.array([float(emitted.false_prob[0])], dtype=np.float32),
        'True_PROB': np.array([float(emitted.true_prob[0])], dtype=np.float32)})
    frame.to_parquet(tmp_path / 'predictions.parquet')
    bundle = PredictionEvidenceSource((str(tmp_path / 'predictions.parquet'), ), run_root,
                                      {}).export(tmp_path / 'bundle')
    report = PredictionEvidenceBundle.load(bundle.root).validate()
    assert report['valid'] and report['evidence_rows'] == 1


# Replay precision #############################################################

EVENT = ('events', 'entity_id', 'entities')
REV_EVENT = ('entities', 'rev_entity_id', 'events')


def bias_free_precision_module(*, events=12, channels=32, seed=3):
    """A bias-free regression scorer whose 32-channel linears see 12 rows, so
    bf16 internals under `medium` change its forward on hosts that have them.
    Rebuilt identically in any process from `seed`."""
    from kumo_core.encoders.target.numerical import NumericalTargetEncoder
    from kumo_core.modeling.gnn.hetero import HeteroGNN
    from kumo_core.modeling.heads.node_prediction import NodePredHead
    from kumo_core.train.loss import MSELoss
    from kumo_core.types import TaskType
    generator = torch.Generator().manual_seed(seed)
    module = cast(Any, torch.nn.Module())
    module.entity_table_name = 'entities'
    module.train_table_name = 'task'
    module.model = HeteroGNN(channels, 2, metadata=(['entities', 'events'], [EVENT, REV_EVENT]),
                             aggr=['mean'], norm=None, bias=False)
    module.head = NodePredHead(channels, 1, num_layers=2, norm=None, param_keys=[], bias=False)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) / channels ** 0.5)
    module.task_type = TaskType.REGRESSION
    module.loss = MSELoss(TaskType.REGRESSION)
    module.target_encoder = NumericalTargetEncoder(0, 1)
    module.encoder = torch.nn.Identity()
    module.device = torch.device('cpu')
    module.eval()
    x = {'entities': torch.rand(1, channels, generator=generator),
         'events': torch.rand(events, channels, generator=generator)}
    edges = {EVENT: torch.stack([torch.arange(events), torch.zeros(events, dtype=torch.long)]),
             REV_EVENT: torch.stack([torch.zeros(events, dtype=torch.long),
                                     torch.arange(events)])}
    nodes = {'entities': [1, 0, 0], 'events': [0, events, 0]}
    counts = {EVENT: [events, 0], REV_EVENT: [0, events]}

    def common_step(_=None):
        out = module.model(dict(x), dict(edges), {k: list(v) for k, v in nodes.items()},
                           {k: list(v) for k, v in counts.items()})
        return module.head(out['entities'][:1])

    module.common_step = common_step
    return module


def precision_sample(events=12):
    return {'origin': 'original_forward', 'seed_table': 'entities',
            'train_table_name': 'task', 'entity_table_name': 'entities',
            'nodes': {'entities': {'pkeys': [1], 'node_ids': [1], 'num_nodes': 1},
                      'events': {'pkeys': list(range(100, 100 + events)),
                                 'node_ids': list(range(events)), 'num_nodes': events}},
            'edges': []}


REPLAY_SCRIPT = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import torch
from walk_binary_fixtures import derive_offline, seed_batch
from test_walk_evidence_degree_t import bias_free_precision_module, precision_sample
from kumo_core.explain.walk_attribution import WalkAttributionConfig
payload = json.loads(Path(sys.argv[2]).read_text())
captured = SimpleNamespace(record=payload['record'], manifest=payload['manifest'],
                           sample=precision_sample(), batch=seed_batch('entities'))
before = torch.get_float32_matmul_precision()
out = derive_offline(captured, bias_free_precision_module(),
                     WalkAttributionConfig(**payload['config']), Path(sys.argv[3]))
print(json.dumps({'before': before, 'after': torch.get_float32_matmul_precision(),
                  'rows': list(out.iter_rows())}))
'''


def _medium_bf16():
    with float32_matmul_precision('medium'):
        return mkldnn_matmul_fp32_precision() == 'bf16'


def test_fresh_process_replay_of_a_medium_capture_is_bitwise(tmp_path):
    """The native row saved to disk and replayed in a fresh process (which
    starts at `highest` and rebuilds the module) under the recorded `medium`
    mode reproduces walks, exactness and the precision residual bitwise."""
    selected = config(top_k=1000)
    module = bias_free_precision_module()
    with float32_matmul_precision('medium'):
        captured = capture_record(module, tmp_path / 'native', selected,
                                  sample=precision_sample(), pkey=1)
    native = captured.record['walk_attribution']
    assert native['status'] == 'available', native
    assert native['numerics']['precision_mode'] == 'medium'
    diagnostics = native['diagnostics'][METHOD]
    assert native['ledger']['precision_residual'] == native['score'] - diagnostics['f_piece']
    if _medium_bf16():  # bf16 internals: the float32 score is not the float64 piece
        assert native['ledger']['precision_residual'] != 0.0
    payload = tmp_path / 'payload.json'
    payload.write_text(json.dumps({'record': captured.record, 'manifest': captured.manifest,
                                   'config': selected.to_dict()}))
    process = subprocess.run([sys.executable, '-c', REPLAY_SCRIPT, str(Path(__file__).parent),
                              str(payload), str(tmp_path / 'offline')],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr[-3000:]
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result['before'] == result['after'] == 'highest'
    offline, = result['rows']
    assert offline['numerics']['replay_precision_mode'] == 'medium'
    assert comparable_row(offline) == comparable_row(native)
    assert math.isfinite(offline['ledger']['accounting_error'])
