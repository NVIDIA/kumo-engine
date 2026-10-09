# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for ``GradientExplainer`` against a hand-built
``HeteroData`` batch and a real (non-DiskGraph, non-legacy-datamodule)
forward pass.

The legacy version of this test drove ``GradientExplainer`` through
``trainer.explain()`` end to end (checkpoint save/load, PL trainer
orchestration, the deleted legacy datamodule). ``GradientExplainer``
itself only needs: a real forward pass run inside
``kumo_core.modeling.context.explaining()`` (which makes each stype encoder
save its input for `.backward()`), the resulting ``step_output`` dict
(model logits), and ``batch.tf_dict``/``batch[table].batch`` -- none of
that is specific to how the batch was produced. This builds the batch
the same way as ``test_column_analysis.py`` (real ``TensorFrame`` via
``kumo_core``'s own ``TensorMapper`` + ``get_tf``) and drives a real
``NodeGNNLightningModule`` forward pass directly (bypassing ``Trainer
.init()``'s DiskGraph/legacy-datamodule split, which nothing here needs).
"""

import json
import os.path as osp

import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.explain.gradient_explainer import GradientExplainer
from kumo_core.modeling.context import explaining
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumo_core.train.trainer import Trainer
from kumo_core.training.task_config import TaskConfig
from kumo_core.types import TaskType
from kumoapi.encoder import Stype
from kumoapi.train import TrainingStage
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData

from kumo.testing.fixtures.task import TestTableSchema, create_task_data

USER = 'USER'
TRAIN_TABLE = 'TRAIN_TABLE'


def _build_trainer_and_batch(root_dir: str) -> tuple[Trainer, HeteroData]:
    age_mapper = TensorMapper.from_stype(Stype.numerical,
                                         pd.Series([18.0, 25.0, 30.0, 50.0]))
    gender_mapper = TensorMapper.from_stype(Stype.categorical,
                                            pd.Series(['m', 'f', 'f', 'm']))
    user_tf = get_tf(
        pa.RecordBatch.from_arrays(
            [
                pa.array([20.0, 20.0, 30.0, 30.0], type=pa.float64()),
                pa.array(['f', 'm', 'f', 'f']),
            ],
            names=['AGE', 'GENDER'],
        ),
        {'AGE': age_mapper, 'GENDER': gender_mapper},
    )

    hetero = HeteroData()
    hetero[USER].tf = user_tf
    hetero[USER].batch_size = 4
    hetero[USER].batch = torch.arange(4)
    hetero[USER].n_id = torch.arange(4)
    hetero[USER].pkey = [f'user_{i}' for i in range(4)]
    hetero[TRAIN_TABLE].tf = TensorFrame({}, {}, num_rows=4)
    hetero[TRAIN_TABLE].batch = torch.arange(4)
    hetero[TRAIN_TABLE].batch_size = 4
    hetero[TRAIN_TABLE].y = torch.tensor([[True], [False], [True], [False]])

    df_dict = {
        USER:
        pd.DataFrame({
            'USER_ID': [0, 1, 2, 3],
            'AGE': [20, 30, 40, 50],
            'GENDER': ['f', 'm', 'f', 'f'],
        }),
    }
    schema_dict = {
        USER:
        TestTableSchema(
            stypes={
                'AGE': Stype.numerical,
                'GENDER': Stype.categorical
            },
            pkey_name='USER_ID',
        ),
    }
    train_table_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'TARGET': [True, False, True, False],
    })
    _, _, encoders = create_task_data(
        df_dict=df_dict,
        schema_dict=schema_dict,
        task_type=TaskType.BINARY_CLASSIFICATION,
        train_table_df=train_table_df,
    )

    trainer = Trainer(
        task=TaskConfig(TaskType.BINARY_CLASSIFICATION,
                        name='TEST_GRADIENT_EXPLAIN'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=0),
        head=NodePredHead(),
        encoders=encoders,
        root_dir=root_dir,
        max_epochs=1,
    )
    trainer.initialize_model(
        node_types=[USER],
        edge_types=[],
        train_table_name=TRAIN_TABLE,
        has_train_table_time_col=False,
        time_col_dict={},
        entity_table_names=[USER],
    )
    rhs_index_map = pd.DataFrame(index=range(4))
    rhs_index_map[NODE_INDEX_COLUMN] = range(4)
    trainer.create_pl_model(
        entity_table_names=[USER],
        train_table_name=TRAIN_TABLE,
        rhs_index_map=rhs_index_map,
        ckpt_path=None,
    )
    trainer.pl_model.eval()
    return trainer, hetero


def test_gradient_explainer_scores_every_reachable_column(
        tmp_path: object) -> None:
    root_dir = str(tmp_path)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        trainer, hetero = _build_trainer_and_batch(root_dir)

    explainer = GradientExplainer(task_type=TaskType.BINARY_CLASSIFICATION,
                                  root_dir=root_dir)
    explainer.explain_start(module=trainer.pl_model, datamodule=None,
                            stage=TrainingStage.TEST)
    explainer.explain_pre_step(batch=hetero, module=trainer.pl_model,
                               datamodule=None, stage=TrainingStage.TEST,
                               batch_idx=0)

    # A real forward pass, run inside `explaining()` so each stype encoder
    # saves its input tensor for `GradientExplainer.get_grad()`'s
    # `.backward()` call to produce gradients against.
    with explaining():
        logits = trainer.pl_model.common_step(hetero)

    explainer.explain_post_step(
        batch=hetero,
        module=trainer.pl_model,
        datamodule=None,
        step_output={'logits': logits},
        stage=TrainingStage.TEST,
        batch_idx=0,
    )
    explainer.explain_end(module=trainer.pl_model, datamodule=None,
                          outputs={}, stage=TrainingStage.TEST)

    scores = explainer._table_score_dict
    counts = explainer._table_count_dict
    assert set(scores[USER].keys()) == {'AGE', 'GENDER'}
    for col_name in ('AGE', 'GENDER'):
        assert counts[USER][col_name] == 4  # One gradient row per seed.

    with open(osp.join(root_dir, 'scores.json')) as f:
        out = json.load(f)
    user_columns = out['tables']['USER']['columns']
    assert set(user_columns) == {'AGE', 'GENDER'}
    # Un-normalized per-column scores are non-negative and normalize to 1.
    assert all(score >= 0 for score in user_columns.values())
    assert sum(user_columns.values()) == pytest.approx(1.0, abs=1e-4)
