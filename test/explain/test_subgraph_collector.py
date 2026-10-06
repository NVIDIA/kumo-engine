# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for ``SubgraphCollector``'s ``csv`` output format
against a hand-built ``HeteroData`` batch.

The legacy version of this suite drove ``SubgraphCollector`` through
``create_inmemory_trainer`` + ``trainer.explain()`` (the deleted legacy
``LightningData`` path); by the time it was deleted, 4 of its 6 tests
were already ``@pytest.mark.skip``'d (``create_remote_trainer ->
create_clients is a raise-on-call stub with no DG-native equivalent``),
so only ``test_subgraph_collector_small_batches`` and ``test_residuals``
were still exercising real code.

``SubgraphCollector.explain_post_step`` only needs
``datamodule.data -> (feature_store, graph_store)`` exposing
``table_names``/``get_pkey_name``/``get_time_col_name``/
``get_tensor_mapper`` (the ``feature_store`` side; same contract
``ColumnAnalysis`` uses in ``test_column_analysis.py``) and
``get_foreign_keys`` (the ``graph_store`` side, only consulted for tables
that declare a foreign key) -- none of that is DiskGraph-specific. This
exercises the ``csv`` format path directly (per-entity CSV slicing +
``TensorMapper.backward()`` value decoding), which is the mechanism every
other format (``nested_json``/``output``/``rfm``) builds on top of.
Multi-hop foreign-key reconstruction (``graph_store.get_foreign_keys``
returning non-empty, walking ``batch.edge_index_dict``) is out of scope
here -- it needs a second linked table with real edges, which is a
materially bigger fixture than what a single-entity-type collector run
needs to exercise the core per-entity output contract.
"""

import glob
import os.path as osp

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.explain.subgraph_collector import SubgraphCollector
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


class _FakeFeatureStore:
    """Stand-in for the bits of ``FeatureStore``
    ``SubgraphCollector.explain_post_step``'s ``csv`` path reads."""
    train_table_name = TRAIN_TABLE
    table_names = [USER]

    def __init__(self, mapper_dict: dict[str, dict[str, TensorMapper]]):
        self._mapper_dict = mapper_dict

    def get_tensor_mapper(self, table_name: str, col_name: str) -> TensorMapper:
        return self._mapper_dict[table_name][col_name]

    def get_pkey_name(self, table_name: str) -> str | None:
        return 'USER_ID' if table_name == USER else None

    def get_time_col_name(self, table_name: str) -> None:
        return None


class _FakeGraphStore:
    """No foreign keys: exercises the single-entity-type path, not the
    multi-hop foreign-key reconstruction (see module docstring)."""
    def get_foreign_keys(self, table_name: str) -> list[tuple[str, str]]:
        return []


class _FakeDataModule:
    def __init__(
        self,
        feature_store: _FakeFeatureStore,
        graph_store: _FakeGraphStore,
    ):
        self.data = (feature_store, graph_store)


def test_subgraph_collector_writes_per_entity_csv(tmp_path: object) -> None:
    root_dir = str(tmp_path)
    ages = [20.0, 30.0, 40.0, 50.0]

    age_mapper = TensorMapper.from_stype(Stype.numerical,
                                         pd.Series([18.0, 25.0, 30.0, 50.0]))
    tf = get_tf(
        pa.RecordBatch.from_arrays([pa.array(ages, type=pa.float64())],
                                   names=['AGE']),
        {'AGE': age_mapper},
    )

    hetero = HeteroData()
    hetero[USER].tf = tf
    hetero[USER].batch_size = 4
    hetero[USER].n_id = torch.arange(4)
    hetero[USER].batch = torch.arange(4)
    hetero[USER].pkey = np.array([f'user_{i}' for i in range(4)])
    hetero[USER].input_id = torch.arange(4)
    hetero[TRAIN_TABLE].tf = TensorFrame({}, {}, num_rows=4)
    hetero[TRAIN_TABLE].batch = torch.arange(4)
    hetero[TRAIN_TABLE].batch_size = 4
    hetero[TRAIN_TABLE].y = torch.tensor([[True], [False], [True], [False]])

    df_dict = {
        USER:
        pd.DataFrame({
            'USER_ID': [0, 1, 2, 3],
            'AGE': ages
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
                        name='TEST_SUBGRAPH_COLLECTOR'),
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
    trainer.pl_model.init_metrics()

    feature_store = _FakeFeatureStore({USER: {'AGE': age_mapper}})
    datamodule = _FakeDataModule(feature_store, _FakeGraphStore())

    collector = SubgraphCollector(
        task_type=TaskType.BINARY_CLASSIFICATION,
        root_dir=root_dir,
        formats=['csv'],
        target_to_entity_paths=[],
    )
    collector.explain_pre_step(batch=hetero, module=trainer.pl_model,
                               datamodule=datamodule, stage=TrainingStage.TEST,
                               batch_idx=0)
    logits = trainer.pl_model.common_step(hetero)
    collector.explain_post_step(
        batch=hetero,
        module=trainer.pl_model,
        datamodule=datamodule,
        step_output={
            'logits': logits,
            'target': hetero[TRAIN_TABLE].y
        },
        stage=TrainingStage.TEST,
        batch_idx=0,
    )

    csv_paths = sorted(
        glob.glob(osp.join(root_dir, 'test', 'csv', '*', 'USER.csv')))
    assert len(csv_paths) == 4

    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        assert df['USER_ID'].tolist() == [f'user_{i}']
        # The AGE value round-trips through TensorMapper.backward() back to
        # the original (unscaled) value fed into the tensor frame above.
        assert df['AGE'].tolist() == [ages[i]]
