# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The numerical coordinate, failure semantics, and streaming lifecycle."""

import json
from types import SimpleNamespace

import kumo_core.modeling.context as context
import numpy as np
import pytest
import torch
from kumo_core.encoders.fused.numerical import FusedNumericalEncoder
from kumo_core.encoders.target.numerical import NumericalTargetEncoder
from kumo_core.explain.local_sensitivity import LocalSensitivity
from kumo_core.train.loss import MSELoss
from kumoapi.encoder import NAStrategy, Numerical
from kumoapi.task import TaskType
from torch_geometric.data import HeteroData


def fixture(tmp_path, *, weight=2.0, raw_value=7.0):
    encoder = FusedNumericalEncoder(
        {'quantity': Numerical(na_strategy=NAStrategy.ZERO)}, [], 1, 'fuse')
    with torch.no_grad():
        encoder.offsets.fill_(3)
        encoder.scales.fill_(4)
        encoder.lin.weight.fill_(weight)
        encoder.lin.bias.fill_(0)
    block = torch.nn.Module()
    block.stype_encoder_dict = torch.nn.ModuleDict({'numerical': encoder})
    whole = torch.nn.Module()
    whole.enc_dict = torch.nn.ModuleDict({'events': block})
    module = SimpleNamespace(
        encoder=whole, loss=MSELoss(TaskType.REGRESSION),
        task_type=TaskType.REGRESSION,
        target_encoder=NumericalTargetEncoder(5, 7, log_transform=True),
        train_table_name='task', device=torch.device('cpu'))
    batch = HeteroData()
    batch.input_type = 'events'
    batch['events'].num_nodes = 1
    batch['events'].batch_size = 1
    batch['events'].pkey = np.array(['source-17'])
    batch['events'].input_id = torch.tensor([19])
    raw = torch.tensor([[raw_value]])
    batch['events'].tf = SimpleNamespace(
        col_names_dict={'numerical': ['quantity']},
        feat_dict={'numerical': raw})
    collector = LocalSensitivity(str(tmp_path), {'run_id': 'one'})
    collector.explain_start(module,
                            SimpleNamespace(base_seed=42, sampling_plan={}),
                            None)
    collector.explain_pre_step(batch, module, None, None, 0)
    with context.explaining():
        logits = encoder(raw)
    output = {'logits': logits, 'pkey': np.array(['source-17']), 'time': None}
    return collector, module, batch, output, encoder


def test_transformed_coordinate_decoded_output_and_source_mapping(tmp_path):
    collector, module, batch, output, encoder = fixture(tmp_path)
    collector.explain_post_step(batch, module, None, output, None, 0)
    collector.explain_end(module, None, {}, None)
    index = json.loads((tmp_path / 'index.jsonl').read_text())
    row = json.loads(
        (tmp_path / 'rows' / f'{index["prediction_id"]}.json').read_text())
    assert row['status'] == 'available'
    cell = row['cells'][0]
    assert cell['raw_value'] == 7 and cell['transformed_value'] == 1
    assert cell['pkey'] == 'source-17'
    assert cell['abs_gradient'] == pytest.approx(14 * np.exp(2))
    # Raw-coordinate differentiation would incorrectly be four times smaller.
    assert cell['transform']['scale'] == 4
    h = .001
    evaluate = lambda z: float(
        module.target_encoder.decode_tensor(
            encoder.forward_learnable(torch.tensor([[z]]))).detach())
    finite_difference = abs((evaluate(1 + h) - evaluate(1 - h)) / (2 * h))
    assert cell['abs_gradient'] == pytest.approx(finite_difference, rel=1e-3)
    assert json.loads((tmp_path / 'manifest.json').read_text())['complete']
    assert not hasattr(encoder, '_saved_tensors')


def test_zero_is_distinct_from_failed_source_mapping(tmp_path):
    collector, module, batch, output, _ = fixture(tmp_path, weight=0)
    collector.explain_post_step(batch, module, None, output, None, 0)
    assert json.loads(
        (tmp_path / 'index.jsonl').read_text())['status'] == 'zero_signal'
    collector, module, batch, output, _ = fixture(tmp_path / 'missing')
    del batch['events'].pkey
    collector.explain_post_step(batch, module, None, output, None, 0)
    row = json.loads((tmp_path / 'missing/index.jsonl').read_text())
    assert row['status'] == 'failed'
    assert row['sample_sha256'] is None
    assert row['reason'] == 'ValueError: local sensitivity capture failed'


def test_streaming_finalizer_never_reads_retained_predictions(tmp_path):
    from kumo_core.train.base_module import BaseLightningModule
    collector, module, batch, output, _ = fixture(tmp_path)
    collector.explain_post_step(batch, module, None, output, None, 0)
    dummy = SimpleNamespace(explainers=[collector],
                            trainer=SimpleNamespace(datamodule=None),
                            _embedding_step_outputs=None)
    BaseLightningModule.on_predict_end(dummy)
    assert json.loads((tmp_path / 'manifest.json').read_text())['rows'] == 1


def test_unsupported_task_rejected_before_first_step(tmp_path):
    collector, module, _, _, _ = fixture(tmp_path)
    module.task_type = TaskType.BINARY_CLASSIFICATION
    with pytest.raises(ValueError, match='scalar MSE regression'):
        collector.explain_start(module, None, None)


def test_sink_failure_preserves_scoring_and_records_partial(
        tmp_path, monkeypatch):
    collector, module, batch, output, _ = fixture(tmp_path)

    def fail(*args):
        raise OSError('internal path must not be published')

    monkeypatch.setattr(collector, '_record_step', fail)
    collector.explain_post_step(batch, module, None, output, None, 0)
    collector.explain_end(module, None, {}, None)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert not manifest['complete']
    assert manifest['failure'] == 'local_artifact_capture_failed'
    assert 'internal path' not in json.dumps(manifest)
    assert 'logits' in output


def test_signed_derivative_and_identity_ignore_batch_ordinal(tmp_path):
    first, module, batch, output, _ = fixture(tmp_path / 'a', weight=-2)
    first.explain_post_step(batch, module, None, output, None, 11)
    record = json.loads((tmp_path / 'a/index.jsonl').read_text())
    row = json.loads(
        (tmp_path / 'a/rows' / f'{record["prediction_id"]}.json').read_text())
    assert row['cells'][0]['signed_gradient'] < 0
    assert row['cells'][0][
        'abs_gradient'] == -row['cells'][0]['signed_gradient']
    assert row['cells'][0]['missing'] is False
    assert row['cells'][0]['source_value_status'] == 'not_provided'
    second, module, batch, output, _ = fixture(tmp_path / 'b', weight=-2)
    second.explain_post_step(batch, module, None, output, None, 999)
    other = json.loads((tmp_path / 'b/index.jsonl').read_text())
    assert record['prediction_id'] == other['prediction_id']
    assert output['prediction_id'] == [record['prediction_id']]
    batch['events'].input_id = torch.tensor([20])
    assert second._identity(batch, output) != first._identity(
        fixture(tmp_path / 'c')[2], output)


def test_gradient_failure_preserves_graph_but_is_not_complete(
        tmp_path, monkeypatch):
    collector, module, batch, output, _ = fixture(tmp_path)

    def fail(*args):
        raise RuntimeError('sensitive details')

    monkeypatch.setattr(collector, '_cells', fail)
    collector.explain_post_step(batch, module, None, output, None, 0)
    collector.explain_end(module, None, {}, None)
    row = json.loads((tmp_path / 'index.jsonl').read_text())
    assert row['graph_status'] == 'available'
    assert row['status'] == 'failed'
    assert (tmp_path / 'samples' / f'{row["prediction_id"]}.json').exists()
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['generation_closed'] and not manifest['complete']
    assert 'sensitive details' not in str(manifest)


def test_graph_only_does_not_differentiate(tmp_path, monkeypatch):
    collector, module, batch, output, _ = fixture(tmp_path)
    collector.numerical_sensitivity = False

    def fail(*args):
        raise AssertionError('Gradient must not be computed')

    monkeypatch.setattr(collector, '_cells', fail)
    collector.explain_post_step(batch, module, None, output, None, 0)
    collector.explain_end(module, None, {}, None)
    row = json.loads((tmp_path / 'index.jsonl').read_text())
    assert row['status'] == 'disabled'
    assert row['graph_status'] == 'available'
    assert json.loads((tmp_path / 'manifest.json').read_text())['complete']


@pytest.mark.parametrize('raw_value,missing', [(0.0, False),
                                               (float('nan'), True),
                                               (float('inf'), True)])
def test_true_zero_and_imputation_are_distinct(tmp_path, raw_value, missing):
    collector, module, batch, output, _ = fixture(tmp_path, raw_value=raw_value)
    collector.explain_post_step(batch, module, None, output, None, 0)
    record = json.loads((tmp_path / 'index.jsonl').read_text())
    row = json.loads(
        (tmp_path / 'rows' / f'{record["prediction_id"]}.json').read_text())
    cell = row['cells'][0]
    assert cell['missing'] is missing
    assert cell['mapper_value'] == (None if missing else 0.0)
    assert cell['transformed_value'] == -.75
    assert cell['signed_gradient'] > 0


def test_existing_sink_is_never_overwritten(tmp_path):
    collector, module, batch, output, _ = fixture(tmp_path)
    collector.explain_post_step(batch, module, None, output, None, 0)
    before = {
        str(p): p.read_bytes()
        for p in tmp_path.rglob('*') if p.is_file()
    }
    second = LocalSensitivity(str(tmp_path), {'run_id': 'another'})
    second.explain_start(module, SimpleNamespace(), None)
    second.explain_end(module, None, {}, None)
    assert second.failure == 'local_artifact_write_failed'
    assert before == {
        str(p): p.read_bytes()
        for p in tmp_path.rglob('*') if p.is_file()
    }


def test_inclusive_sampler_graph_retained_with_strict_temporal_violation(
        tmp_path):
    collector, module, batch, output, _ = fixture(tmp_path)
    collector.provenance['model_reconstruction'] = {
        'metadata': {
            'is_temporal': True
        }
    }
    collector.provenance['temporal_boundary']['source_time_columns'] = {
        'events': 'event_time'
    }
    batch['events'].time = torch.tensor([100])
    output['time'] = torch.tensor([100])
    collector.explain_post_step(batch, module, None, output, None, 0)
    collector.explain_end(module, None, {}, None)
    row = json.loads((tmp_path / 'index.jsonl').read_text())
    assert row['graph_status'] == 'available'
    assert row['temporal_validation']['status'] == 'violated'
    assert row['temporal_validation']['violations'][0]['pkey'] == 'source-17'
    assert row['temporal_validation']['violations'][0][
        'source_time_column'] == 'event_time'
    assert not json.loads((tmp_path / 'manifest.json').read_text())['complete']
