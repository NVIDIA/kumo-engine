# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import kumo_core.train.forecasting_gnn_module as forecasting_gnn_module
import torch
from kumo_core.train.autoregression import (
    AUTOREGRESSION_COL,
    FUTURE_LABELS_COL,
)
from torch_frame import TensorFrame
from torch_frame import stype as torch_stype
from torch_frame.data import MultiEmbeddingTensor
from torch_geometric.data import HeteroData


class _IdentityTargetEncoder:
    def encode_tensor(self, value):
        return value


class _FailingTrainTable:
    def query_future(self, **_kwargs):
        raise AssertionError('legacy train-table query should not run')

    def query_past(self, **_kwargs):
        raise AssertionError('legacy train-table query should not run')


def _module(*, arl_length=2, forecast_length=3):
    module = object.__new__(
        forecasting_gnn_module.ForecastingGNNLightningModule)
    module.train_table_name = 'TRAIN_TABLE'
    module.entity_table_name = 'User'
    module.arl_length = arl_length
    module.forecast_length = forecast_length
    module.target_encoder = _IdentityTargetEncoder()
    module.use_precomputed_forecasting_sequences = True
    module._fc_train_table = _FailingTrainTable()
    return module


def _batch(column_name: str, values: torch.Tensor) -> HeteroData:
    batch = HeteroData()
    batch['User'].batch_size = values.size(0)
    batch['User'].n_id = torch.arange(values.size(0))
    batch['User'].seed_time = torch.zeros(values.size(0), dtype=torch.long)
    batch['TRAIN_TABLE'].tf = TensorFrame(
        feat_dict={
            torch_stype.sequence_numerical:
            MultiEmbeddingTensor(
                num_rows=values.size(0),
                num_cols=1,
                values=values,
                offset=torch.tensor([0, values.size(1)]),
            )
        },
        col_names_dict={torch_stype.sequence_numerical: [column_name]},
    )
    return batch


def test_get_future_targets_reads_ttng_future_label_sequence():
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, float('nan'), 6.0]])
    module = _module(forecast_length=3)

    future = module.get_future_targets(_batch(FUTURE_LABELS_COL, values))

    assert torch.equal(future, values.nan_to_num())


def test_get_past_targets_reads_ttng_autoregression_sequence():
    values = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    module = _module(arl_length=2)

    past = module.get_past_targets(_batch(AUTOREGRESSION_COL, values))

    assert torch.equal(past, values)


def test_batch_sequence_reads_are_explicitly_gated():
    class _TrainTable:
        def query_future(self, **_kwargs):
            return torch.tensor([[7.0, 8.0, 9.0]])

    values = torch.tensor([[1.0, 2.0, 3.0]])
    module = _module(forecast_length=3)
    module.use_precomputed_forecasting_sequences = False
    module._fc_train_table = _TrainTable()

    future = module.get_future_targets(_batch(FUTURE_LABELS_COL, values))

    assert torch.equal(future, torch.tensor([[7.0, 8.0, 9.0]]))


def test_predict_forward_reads_ttng_ar_sequence_numerical():
    values = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    module = _module(arl_length=2)
    captured = {}

    module.year_over_year_encoder = None
    module.past_encoder = None
    module.encoder = lambda *_args, **_kwargs: {}
    module.model = lambda *_args, **_kwargs: {
        'User': torch.ones(values.size(0), 4)
    }

    def _head(_x, *, length, past, **_kwargs):
        captured['length'] = length
        captured['past'] = past
        return torch.zeros(values.size(0), length)

    module.head = _head
    tf = _batch(AUTOREGRESSION_COL, values)['TRAIN_TABLE'].tf

    out = module.forward(
        tf_dict={'TRAIN_TABLE': tf},
        edge_index_dict={},
        time_dict={},
        batch_dict={},
        node_id=torch.arange(values.size(0)),
        seed_time=torch.zeros(values.size(0), dtype=torch.long),
    )

    assert out.shape == (values.size(0), module.forecast_length)
    assert captured['length'] == module.forecast_length
    assert torch.equal(captured['past'], values)


def test_predict_step_returns_requested_embedding_per_batch():
    module = _module(forecast_length=3)
    module._kumo_return_predict_embeddings = True
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state: dict[str, Any] = {'hook': None, 'removed': False}

    class _Handle:
        def remove(self):
            state['removed'] = True

    def register_hook(hook):
        state['hook'] = hook
        return _Handle()

    def common_step(_batch, *, predict):
        assert predict is True
        state['hook'](None, (embedding, ))
        return torch.zeros(2, 3, 1)

    module._get_register_embedding_hook_fn = lambda: register_hook
    module.common_step = common_step
    batch = HeteroData()
    batch['User'].batch_size = 2
    batch['User'].pkey = torch.tensor([10, 11])
    batch['TRAIN_TABLE'].time = torch.tensor([100, 200])

    output = module.predict_step(batch, batch_idx=0)

    assert state['removed'] is True
    assert torch.equal(output['embedding'], embedding)
    assert output['logits'].shape == (2, 3)
