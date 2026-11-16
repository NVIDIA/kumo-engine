# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple
from dataclasses import asdict, dataclass

import torch
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.base_module import _config_hparams
from kumo_core.train.node_gnn_module import NodeGNNLightningModule
from kumo_core.types import ColStatType, TaskType
from kumoapi.encoder import Index, Numerical


def _model():
    city = Index()
    city._stats = {ColStatType.CATEGORY_COUNTS: [['a', 3], ['b', 1]]}
    amount = Numerical()
    amount._stats = {ColStatType.MEAN: 1.5}
    return NodeGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'users': {
                'city': city,
                'amount': amount
            }},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=2, num_mp_layers=0,
                        metadata=(['users'], [])),
        head=NodePredHead(in_channels=2, out_channels=1),
        task_type=TaskType.BINARY_CLASSIFICATION,
        train_table_name='train_table',
        entity_table_name='users',
    )


def _without_stats(value):
    if isinstance(value, dict):
        return {k: _without_stats(v) for k, v in value.items() if k != '_stats'}
    return value


def test_hparams_are_asdict_without_encoder_stats():
    model = _model()
    expected = asdict(model.encoder.config())
    assert expected['table_encoder_dict']['users']['city']['_stats']
    assert model.hparams['encoder'] == _without_stats(expected)

    Pair = namedtuple('Pair', ['left', 'right'])

    @dataclass
    class Config:
        pair: Pair
        nested: dict

    config = Config(Pair(1, [2]), {('k', 1): Pair(3, 4)})
    assert _config_hparams(config) == asdict(config)


def test_checkpoints_with_encoder_stats_in_hparams_still_load(tmp_path):
    model = _model()
    path = tmp_path / 'old.ckpt'
    torch.save(
        {
            'state_dict': model.state_dict(),
            'hyper_parameters': {
                **model.hparams, 'encoder': asdict(model.encoder.config())
            },
        }, path)

    loaded = _model().load_checkpoint(str(path)).state_dict()

    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(loaded[key], value)
