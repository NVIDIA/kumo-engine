# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for ``ColumnAnalysis`` against a hand-built
``HeteroData`` batch.

The legacy version of this test drove ``ColumnAnalysis`` through a real
``Trainer.init()`` + in-memory ``FeatureStore``/``GraphStore`` (the
``LightningData`` path this repo no longer ships). ``ColumnAnalysis``
itself only ever consumes an already-materialized ``HeteroData`` batch
(``TensorFrame``s + a ``batch`` index per node type + per-table
``num_sampled_nodes`` hop boundaries) plus a ``datamodule.data ->
(feature_store, graph_store)`` pair exposing ``train_table_name`` and
``get_tensor_mapper``. None of that is DiskGraph- or legacy-datamodule
-specific, so this builds the batch directly: a real ``TensorFrame`` per
node type via ``kumo_core``'s own ``TensorMapper`` + ``get_tf`` (the same
primitives ``DiskgraphLightningDataModule`` uses internally), and a
minimal stand-in for ``datamodule.data`` exposing just what
``ColumnAnalysis`` reads. This exercises the real cohort-computation code
(``collect_count``/``collect_numerical``/``collect_categorical``/
``aggregate_range``/``aggregate_category``), independent of which
datamodule produced the batch.
"""

import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.types import TaskType
from kumoapi.encoder import Stype
from kumoapi.train import TrainingStage
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData

USER = 'USER'
ORDER = 'ORDER'
TRAIN_TABLE = 'TRAIN_TABLE'


class _FakeFeatureStore:
    """Stand-in for the bits of ``FeatureStore`` ``ColumnAnalysis`` reads:
    ``train_table_name`` and per-(table, column) ``TensorMapper`` lookup."""
    train_table_name = TRAIN_TABLE

    def __init__(self, mapper_dict: dict[str, dict[str, TensorMapper]]):
        self._mapper_dict = mapper_dict

    def get_tensor_mapper(self, table_name: str, col_name: str) -> TensorMapper:
        return self._mapper_dict[table_name][col_name]


class _FakeDataModule:
    """Stand-in for ``datamodule``: ``ColumnAnalysis`` only reads ``.data``."""
    def __init__(self, feature_store: _FakeFeatureStore):
        self.data = (feature_store, None)


class _FakeModule:
    """Stand-in for the ``pl_model``: ``ColumnAnalysis`` only reads
    ``.encoder.exclude_cols_dict`` and ``.target_encoder``."""
    class encoder:
        exclude_cols_dict: dict = {}

    target_encoder = None


def _build_batch() -> tuple[HeteroData, _FakeDataModule]:
    """Two-hop batch: 4 USER seeds (hop 0, AGE numerical + GENDER
    categorical) each optionally linked to ORDER rows (hop 1, AMOUNT
    numerical) -- user0 has 2 orders, user1 has 1, user2 has 0, user3 has 1.
    """
    age_mapper = TensorMapper.from_stype(Stype.numerical,
                                         pd.Series([18.0, 25.0, 30.0, 50.0]))
    gender_mapper = TensorMapper.from_stype(Stype.categorical,
                                            pd.Series(['f', 'm', 'f', 'f']))
    amount_mapper = TensorMapper.from_stype(
        Stype.numerical, pd.Series([10.0, 20.0, 30.0, 40.0]))

    user_tf = get_tf(
        pa.RecordBatch.from_arrays(
            [
                pa.array([20.0, 30.0, 40.0, 50.0], type=pa.float64()),
                pa.array(['f', 'm', 'f', 'm']),
            ],
            names=['AGE', 'GENDER'],
        ),
        {'AGE': age_mapper, 'GENDER': gender_mapper},
    )
    order_tf = get_tf(
        pa.RecordBatch.from_arrays(
            [pa.array([100.0, 200.0, 300.0, 400.0], type=pa.float64())],
            names=['AMOUNT'],
        ),
        {'AMOUNT': amount_mapper},
    )

    hetero = HeteroData()
    hetero[USER].tf = user_tf
    hetero[USER].batch_size = 4
    hetero[USER].batch = torch.tensor([0, 1, 2, 3])
    hetero[USER].num_sampled_nodes = [4]

    hetero[ORDER].tf = order_tf
    # user0 -> orders 0,1; user1 -> order 2; user2 -> none; user3 -> order 3.
    hetero[ORDER].batch = torch.tensor([0, 0, 1, 3])
    hetero[ORDER].num_sampled_nodes = [4]

    hetero[TRAIN_TABLE].tf = TensorFrame({}, {}, num_rows=4)
    hetero[TRAIN_TABLE].batch = torch.arange(4)
    hetero[TRAIN_TABLE].batch_size = 4
    hetero[TRAIN_TABLE].y = torch.tensor([[True], [False], [True], [False]])

    feature_store = _FakeFeatureStore({
        USER: {
            'AGE': age_mapper,
            'GENDER': gender_mapper
        },
        ORDER: {
            'AMOUNT': amount_mapper
        },
    })
    return hetero, _FakeDataModule(feature_store)


def test_column_analysis_correctness() -> None:
    hetero, datamodule = _build_batch()
    module = _FakeModule()

    analyzer = ColumnAnalysis(task_type=TaskType.BINARY_CLASSIFICATION,
                              root_dir=None, normalize=False)
    analyzer.explain_start(module=module, datamodule=datamodule,
                           stage=TrainingStage.VAL)
    analyzer.explain_post_step(
        batch=hetero,
        module=module,
        datamodule=datamodule,
        step_output={},
        stage=TrainingStage.VAL,
        batch_idx=0,
    )

    # Per-entity intermediate output: one row per USER, scatter-aggregated
    # from the linked ORDER rows via the `batch` index.
    df = analyzer._dfs_dict[('USER', 0)][-1]
    assert df['COUNT(*)'].tolist() == [1, 1, 1, 1]
    assert df['AGE'].tolist() == [20.0, 30.0, 40.0, 50.0]
    assert df['GENDER:f'].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert df['GENDER:m'].tolist() == [0.0, 1.0, 0.0, 1.0]

    df = analyzer._dfs_dict[('ORDER', 0)][-1]
    assert df['COUNT(*)'].tolist() == [2, 1, 0, 1]
    # user0's two orders (100, 200) average to 150; user2 has none -> NaN.
    assert df['AMOUNT'].fillna(-1).tolist() == [150.0, 300.0, -1.0, 400.0]

    prob = torch.tensor([0.9, 0.1, 0.8, 0.2])
    out = analyzer.explain_end(
        module=module,
        datamodule=datamodule,
        outputs={
            'prob': torch.stack([1 - prob, prob], dim=1),
            'target': hetero[TRAIN_TABLE].y.view(-1),
        },
        stage=TrainingStage.VAL,
    )

    assert out is not None
    assert out.task_type == TaskType.BINARY_CLASSIFICATION
    assert {(c.table_name, c.hop) for c in out.cohorts} == {
        ('USER', 0),
        ('ORDER', 0),
    }

    cohort = out['USER', 0, 'AGE']
    assert cohort.stype == Stype.numerical
    assert cohort.populations == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert cohort.targets == pytest.approx([1.0, 0.0, 1.0, 0.0])

    cohort = out['USER', 0, 'GENDER']
    assert cohort.stype == Stype.categorical
    assert cohort.cohorts == ['f', 'm']
    assert cohort.populations == pytest.approx([0.5, 0.5])
    assert cohort.targets == pytest.approx([1.0, 0.0])

    cohort = out['ORDER', 0, 'AMOUNT']
    assert cohort.stype == Stype.numerical
    # Only the 3 non-NaN averaged AMOUNT values (150, 300, 400) form cohorts;
    # user2 (no orders) is excluded from this table's per-entity rows.
    assert len(cohort.cohorts) == 3
    assert sum(cohort.populations) == pytest.approx(1.0)
