# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for ``Trainer.explain()``'s explainer-interface
orchestration (start/pre_step/post_step/end call counts, limit_xxx_batches
save/restore) against a hand-built, non-DiskGraph, non-legacy-datamodule
setup.

The legacy version of this test drove ``Trainer.explain()`` through
``create_inmemory_trainer`` (the deleted legacy ``LightningData`` path).
``Trainer.explain()`` itself just needs a working ``Trainer`` -- real
``pl_model``/``pl_trainer``, and a ``datamodule`` exposing
``val_dataloader``/``test_dataloader`` and ``eval_loader_kwargs`` -- none
of that is DiskGraph-specific either. This builds the model the same way
as ``test_gradient.py`` (bypassing ``Trainer.init()``'s DiskGraph/legacy-
datamodule split via ``initialize_model()``/``create_pl_model()``
directly) and wires a minimal fake ``LightningDataModule`` serving a
couple of hand-built ``HeteroData`` batches, then runs a real (tiny)
``Trainer.fit()`` so ``Trainer.explain()``'s checkpoint load has
something real to load. Prediction-stage explain is out of scope here:
``Trainer.predict()`` needs a much larger ``FeatureStore`` surface for
its own output-writing logic that has nothing to do with the explainer
interface being tested.
"""

import os.path as osp

import pandas as pd
import pyarrow as pa
import pytorch_lightning as pl
import torch
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.explain.base import Explainer
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumo_core.train.trainer import Trainer
from kumo_core.training.task_config import TaskConfig
from kumo_core.types import TaskType
from kumoapi.encoder import Stype
from torch.utils.data import DataLoader
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData

from kumo.explain.runtime import MonolithExplainRuntime
from kumo.testing.fixtures.task import TestTableSchema, create_task_data

USER = 'USER'
TRAIN_TABLE = 'TRAIN_TABLE'


class _DummyExplainer(Explainer):
    supported_task_types = {TaskType.BINARY_CLASSIFICATION}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_count = 0
        self.pre_step_count = 0
        self.post_step_count = 0
        self.end_count = 0

    def explain_start(self, *args, **kwargs) -> None:
        self.start_count += 1

    def explain_pre_step(self, *args, **kwargs) -> None:
        self.pre_step_count += 1

    def explain_post_step(self, *args, **kwargs) -> None:
        self.post_step_count += 1

    def explain_end(self, *args, **kwargs) -> None:
        self.end_count += 1


def _make_batch(ages: list[float]) -> HeteroData:
    n = len(ages)
    age_mapper = TensorMapper.from_stype(Stype.numerical,
                                         pd.Series([18.0, 25.0, 30.0, 50.0]))
    tf = get_tf(
        pa.RecordBatch.from_arrays([pa.array(ages, type=pa.float64())],
                                   names=['AGE']),
        {'AGE': age_mapper},
    )
    hetero = HeteroData()
    hetero[USER].tf = tf
    hetero[USER].batch_size = n
    hetero[USER].batch = torch.arange(n)
    hetero[USER].n_id = torch.arange(n)
    hetero[USER].input_id = torch.arange(n)  # Marks USER as `batch.input_type`.
    hetero[USER].pkey = [f'user_{i}' for i in range(n)]
    hetero[TRAIN_TABLE].tf = TensorFrame({}, {}, num_rows=n)
    hetero[TRAIN_TABLE].batch = torch.arange(n)
    hetero[TRAIN_TABLE].batch_size = n
    hetero[TRAIN_TABLE].y = torch.tensor([[bool(i % 2)] for i in range(n)])
    return hetero


class _FakeDataModule(pl.LightningDataModule):
    """Two batches per stage, so the explain-interface call counts below
    are actually exercising a multi-batch loop, not a degenerate case."""
    def __init__(self):
        super().__init__()
        self.eval_loader_kwargs = {'batch_size': 4}
        self._batches = [
            _make_batch([20.0, 30.0, 40.0, 50.0]),
            _make_batch([21.0, 31.0, 41.0, 51.0]),
        ]

    def _loader(self) -> DataLoader:
        return DataLoader(self._batches, batch_size=None,
                          collate_fn=lambda batch: batch)

    def train_dataloader(self) -> DataLoader:
        return self._loader()

    def val_dataloader(self) -> DataLoader:
        return self._loader()

    def test_dataloader(self) -> DataLoader:
        return self._loader()


class _FakeFeatureStore:
    """Stand-in for the bits of ``FeatureStore`` ``Trainer._validate_mask``
    (called from ``fit()``) reads."""
    train_table_name = TRAIN_TABLE
    has_train_table_time_col = False
    has_train_table_test_mask = True

    def __getitem__(self, key: object) -> torch.Tensor:
        return torch.ones(4, dtype=torch.bool)

    def get_time_col_name(self, table_name: str) -> None:
        return None


def _build_trainer(root_dir: str) -> Trainer:
    df_dict = {
        USER: pd.DataFrame({
            'USER_ID': [0, 1, 2, 3],
            'AGE': [20, 30, 40, 50]
        }),
    }
    schema_dict = {
        USER: TestTableSchema(stypes={'AGE': Stype.numerical},
                              pkey_name='USER_ID'),
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
                        name='TEST_EXPLAIN_INTERFACE'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=0),
        head=NodePredHead(),
        encoders=encoders,
        feature_store=_FakeFeatureStore(),
        root_dir=root_dir,
        max_epochs=1,
        max_steps_per_epoch=1,
        max_val_steps=2,
        max_test_steps=2,
        early_stopping=False,
        explain_runtime=MonolithExplainRuntime(),
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
    trainer.create_pl_trainer()
    trainer.datamodule = _FakeDataModule()
    trainer._fully_initialized = True
    trainer.fit()
    return trainer


def test_explain_interface(tmp_path: object) -> None:
    trainer = _build_trainer(str(tmp_path))

    explainer = _DummyExplainer(TaskType.BINARY_CLASSIFICATION,
                                root_dir=str(tmp_path))
    trainer.pl_model.explainers.append(explainer)

    trainer.explain(explain_methods=[], stage='val')
    assert explainer.start_count == 1
    assert explainer.pre_step_count == 2  # 2 batches in the val dataloader.
    assert explainer.post_step_count == 2
    assert explainer.end_count == 1
    # `explain()` empties `explainers` at the end of each call:
    assert trainer.pl_model.explainers == []

    trainer.pl_model.explainers.append(explainer)
    assert trainer.pl_trainer.limit_test_batches == 1.0
    trainer.explain(explain_methods=[], stage='test')
    # `explain()` must restore `limit_test_batches` after running:
    assert trainer.pl_trainer.limit_test_batches == 1.0
    assert explainer.start_count == 2
    assert explainer.pre_step_count == 4
    assert explainer.post_step_count == 4
    assert explainer.end_count == 2

    assert osp.exists(trainer.log_dir)
