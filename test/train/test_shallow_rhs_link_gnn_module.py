# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import kumo_core.modeling.context as modeling_context
import pandas as pd
import pytest
import torch
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from kumo_core.modeling.retrieval.mips import MIPSFaiss
from kumo_core.train.base_link_module import BaseLinkLightningModule
from kumo_core.train.mapping import (
    NODE_INDEX_COLUMN,
    TTNG_RHS_PRIMARY_KEY_INDEX_NAME,
)
from kumo_core.train.shallow_rhs_link_gnn_module import (
    MAX_RHS_SAMPLE_SIZE,
    ShallowRHSLinkGNNLightningModule,
)
from kumo_core.types import ColStatType, TaskType
from kumo_core.types.features import TIMESTAMP_PREFIX
from kumoapi.encoder import (
    Datetime,
    Encoder,
    Hash,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    Numerical,
    NumericalList,
)
from kumoapi.model_plan import RHSEmbeddingMode
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor
from torch_geometric.data import HeteroData


def get_tf_and_col_encoder_dict() -> tuple[TensorFrame, dict[str, Encoder]]:
    num_rows = 4
    numerical = Stype.numerical
    categorical = Stype.categorical
    timestamp = Stype.timestamp
    sequence = Stype.sequence
    feat_dict = {
        numerical: torch.randn(num_rows, 3),
        categorical: torch.arange(num_rows).unsqueeze(1).repeat(1, 2),
        timestamp: torch.stack([
            torch.randint(0, 60, (num_rows, 1)),
            torch.randint(0, 24, (num_rows, 1)),
            torch.randint(1990, 2023, (num_rows, 1)),
            torch.randint(0, 7, (num_rows, 1)),
            torch.randint(0, 28, (num_rows, 1)),
            torch.randint(0, 365, (num_rows, 1)),
            torch.randint(28, 31, (num_rows, 1)),
            torch.randint(365, 366, (num_rows, 1)),
            torch.randint(52, 53, (num_rows, 1)),
            torch.randint(0, 12, (num_rows, 1)),
            torch.randint(0, 52, (num_rows, 1)),
            torch.randint(0, 10, (num_rows, 1)),
            torch.randint(0, 10, (num_rows, 1)),
            torch.randint(0, 10, (num_rows, 1)),
            torch.randint(0, 10, (num_rows, 1)),
            torch.randint(0, 1000000000, (num_rows, 1)),
        ], dim=-1),
        sequence: MultiEmbeddingTensor.from_tensor_list(
            [torch.randn(num_rows, 11)]),
    }
    tf = TensorFrame(
        feat_dict=feat_dict,
        col_names_dict={
            numerical: ['num0', 'num1', 'num2'],
            categorical: ['cat0', 'cat1'],
            timestamp: ['time0'],
            sequence: ['seq0'],
        },
    )
    category_counts = [['A', 4], ['B', 3], ['C', 2], ['D', 1]]
    num0 = Numerical()
    num0._stats = {
        ColStatType.MEAN: float(torch.mean(tf.feat_dict[numerical][:, 0])),
    }
    num1 = MaxLogNumerical()
    num1._stats = {
        ColStatType.MIN: float(torch.min(tf.feat_dict[numerical][:, 1])),
        ColStatType.MEAN: float(torch.mean(tf.feat_dict[numerical][:, 1])),
    }
    num2 = MinLogNumerical()
    num2._stats = {
        ColStatType.MAX: float(torch.max(tf.feat_dict[numerical][:, 2])),
        ColStatType.MEAN: float(torch.mean(tf.feat_dict[numerical][:, 2])),
    }
    cat0 = Index()
    cat0._stats = {ColStatType.CATEGORY_COUNTS: category_counts}
    cat1 = Hash(num_components=num_rows)
    cat1._stats = {ColStatType.CATEGORY_COUNTS: category_counts}
    time0 = Datetime(num_year_periods=1)
    time0._stats = {
        ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
        ColStatType.MAX: f'{TIMESTAMP_PREFIX}2022-12-31',
    }
    seq0 = NumericalList()
    seq0._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: tf.feat_dict[sequence].values.size(1),
        ColStatType.SEQUENCE_MIN_LENGTH: tf.feat_dict[sequence].values.size(1),
    }
    return tf, {
        'num0': num0,
        'num1': num1,
        'num2': num2,
        'cat0': cat0,
        'cat1': cat1,
        'time0': time0,
        'seq0': seq0,
    }


def _rhs_lookup_model(index_map: pd.DataFrame):
    num_rhs_nodes = (int(index_map[NODE_INDEX_COLUMN].max()) +
                     1 if len(index_map) else 0)
    return ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(
            channels=2,
            num_mp_layers=0,
            metadata=(['LHS', 'RHS'], []),
        ),
        head=LinkPredHead(in_channels=2, embedding_dim=2,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=num_rhs_nodes,
        rhs_emb_mode='lookup',
        rhs_index_map=index_map,
    )


def test_ttng_rhs_sparse_canonical_ids_filter_static_mips() -> None:
    index_map = pd.DataFrame(
        {NODE_INDEX_COLUMN: [0, 2, 4]},
        index=pd.Index(['a', 'b', 'c'], name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME),
    )
    model = _rhs_lookup_model(index_map)
    model.task_type = TaskType.STATIC_LINK_PREDICTION
    model.register_buffer('canonical_rhs_node_idx', torch.tensor([0, 2, 4]),
                          persistent=False)

    model.on_evaluation_start(TrainingStage.VAL)

    assert model.valid_rhs_node_idx.tolist() == [0, 2, 4]
    assert model._canonical_rhs_node_ids(
        model.device).data_ptr() == model.canonical_rhs_node_idx.data_ptr()
    assert model.rhs_embedding().shape == (3, 2)
    assert model._mips_index.num_keys == 3


def test_ttng_rhs_temporal_validity_intersects_canonical_ids() -> None:
    model = _rhs_lookup_model(_ttng_rhs_index(['a', 'b', 'c', 'd', 'e']))
    model.register_buffer('canonical_rhs_node_idx', torch.tensor([0, 2, 4]),
                          persistent=False)
    lp_train_table = SimpleNamespace(
        valid_rhs_idx_dict={1: torch.tensor([1, 2, 4])},
        get_valid_rhs_idx=lambda _seed_time: torch.tensor([1, 2, 4]),
    )
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(
        lp_train_table=lp_train_table))

    with patch.object(torch, 'isin', side_effect=AssertionError):
        valid = model._get_valid_rhs_idx_for_seed_time(1)

    assert valid is not None
    assert valid.tolist() == [2, 4]


def test_evaluation_end_restores_canonical_rhs_features_for_training() -> None:
    rhs_feats, rhs_encoders = get_tf_and_col_encoder_dict()
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': rhs_encoders},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=2, num_mp_layers=0),
        head=LinkPredHead(in_channels=2, embedding_dim=2,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.STATIC_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=4,
        rhs_emb_mode=RHSEmbeddingMode.FEATURE,
    )
    canonical = torch.tensor([0, 2])
    model.register_buffer('canonical_rhs_node_idx', canonical,
                          persistent=False)

    class _FeatureStore:
        def __getitem__(self, key):
            table_name, attr, index = key
            assert table_name == 'RHS'
            assert attr == 'tf'
            return rhs_feats[index]

    feature_store = _FeatureStore()
    model.rhs_embedding.filter(canonical)
    model.rhs_embedding.fetch_feats(feature_store, 'RHS')
    assert model.rhs_embedding._filtered_feats

    with (
        patch.object(BaseLinkLightningModule, 'evaluation_epoch_end'),
        patch.object(type(model), 'datamodule', new_callable=PropertyMock)
        as datamodule,
    ):
        datamodule.return_value = SimpleNamespace(feature_store=feature_store)
        model.evaluation_epoch_end(TrainingStage.VAL, [])

    assert model.rhs_embedding._filtered_index is not None
    assert model.rhs_embedding._filtered_index.tolist() == [0, 2]
    assert model.rhs_embedding._filtered_feats
    assert model.rhs_embedding(canonical).shape == (2, 2)


def _ttng_rhs_index(keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {NODE_INDEX_COLUMN: range(len(keys))},
        index=pd.Index(keys, name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME),
    )


def test_ttng_rhs_checkpoint_reorders_by_raw_primary_key() -> None:
    old_model = _rhs_lookup_model(_ttng_rhs_index(['a', 'b']))
    old_model.rhs_embedding.lookup_embedding.weight.data.copy_(
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]))
    state = old_model.state_dict()

    new_index = _ttng_rhs_index(['b', 'c', 'a'])
    new_model = _rhs_lookup_model(new_index)
    new_model.set_rhs_index_map(new_index)
    new_model.load_state_dict(state)

    weight = new_model.rhs_embedding.lookup_embedding.weight.detach()
    assert torch.equal(weight[0], torch.tensor([2.0, 20.0]))
    assert torch.equal(weight[2], torch.tensor([1.0, 10.0]))
    assert torch.equal(weight[1], weight[[0, 2]].mean(dim=0))


def test_ttng_rhs_checkpoint_reorders_decimal_primary_keys_and_sparse_ids(
) -> None:
    old_index = pd.DataFrame(
        {NODE_INDEX_COLUMN: [0, 2, 5]},
        index=pd.Index(
            [Decimal('1.00'), Decimal('-2.50'), Decimal('9999999999.125')],
            name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME,
        ),
    )
    old_model = _rhs_lookup_model(old_index)
    old_weight = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    old_model.rhs_embedding.lookup_embedding.weight.data.copy_(old_weight)
    state = old_model.state_dict()

    new_index = pd.DataFrame(
        {NODE_INDEX_COLUMN: [1, 4, 7]},
        index=pd.Index(
            [Decimal('-2.500'), Decimal('3.75'), Decimal('1.0')],
            name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME,
        ),
    )
    new_model = _rhs_lookup_model(new_index)
    new_model.set_rhs_index_map(new_index)
    new_model.load_state_dict(state)

    weight = new_model.rhs_embedding.lookup_embedding.weight.detach()
    expected_mean = old_weight[[0, 2, 5]].mean(dim=0)
    assert weight.shape == (8, 2)
    assert torch.equal(weight[1], old_weight[2])
    assert torch.equal(weight[7], old_weight[0])
    assert torch.equal(weight[4], expected_mean)
    assert new_model.rhs_embedding.lookup_embedding.mask.tolist() == [
        True, False, True, True, True, True, True, False
    ]


@pytest.mark.parametrize(
    ('old_keys', 'new_keys'),
    [
        (['a', 'b', 'c'], ['c', 'a', 'new']),
        (
            [Decimal('1.00'), Decimal('-2.50'), Decimal('9999999999.125')],
            [Decimal('9999999999.1250'), Decimal('1.0'), Decimal('3.75')],
        ),
    ],
)
def test_ttng_rhs_checkpoint_reorders_sparse_optimizer_state(
        tmp_path: Path, old_keys: list[object],
        new_keys: list[object]) -> None:
    old_index = pd.DataFrame(
        {NODE_INDEX_COLUMN: [0, 2, 4]},
        index=pd.Index(old_keys, name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME),
    )
    old_model = _rhs_lookup_model(old_index)
    old_model.lr_scheduler = None
    old_configs = old_model.configure_optimizers()
    assert old_configs is not None
    rhs_optimizer = old_configs[1]['optimizer']
    assert isinstance(rhs_optimizer, torch.optim.Optimizer)
    lookup_embedding = old_model.rhs_embedding.lookup_embedding
    assert lookup_embedding is not None
    rhs_parameter = lookup_embedding.weight
    old_exp_avg = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    old_exp_avg_sq = old_exp_avg + 100
    rhs_optimizer.state[rhs_parameter] = {
        'step': torch.tensor(1.0),
        'exp_avg': old_exp_avg,
        'exp_avg_sq': old_exp_avg_sq,
    }
    checkpoint_path = tmp_path / 'sparse-rhs.ckpt'
    torch.save(
        {
            'state_dict':
            old_model.state_dict(),
            'optimizer_states':
            [config['optimizer'].state_dict() for config in old_configs],
        },
        checkpoint_path,
    )

    new_index = pd.DataFrame(
        {NODE_INDEX_COLUMN: [1, 3, 5]},
        index=pd.Index(new_keys, name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME),
    )
    new_model = _rhs_lookup_model(new_index)
    new_model.lr_scheduler = None
    new_model.set_rhs_index_map(new_index)
    new_model.load_checkpoint(
        str(checkpoint_path),
        map_location=torch.device('cpu'),
        load_optimizer_state=True,
        _new_rhs_order=new_index,
    )

    new_configs = new_model.configure_optimizers()
    assert new_configs is not None
    new_optimizer = new_configs[1]['optimizer']
    assert isinstance(new_optimizer, torch.optim.Optimizer)
    new_state = new_optimizer.state_dict()['state'][0]
    expected_avg = torch.zeros(6, 2)
    expected_avg[1] = old_exp_avg[4]
    expected_avg[3] = old_exp_avg[0]
    expected_avg_sq = torch.zeros(6, 2)
    expected_avg_sq[1] = old_exp_avg_sq[4]
    expected_avg_sq[3] = old_exp_avg_sq[0]
    assert torch.equal(new_state['exp_avg'], expected_avg)
    assert torch.equal(new_state['exp_avg_sq'], expected_avg_sq)


def test_ttng_rhs_checkpoint_rejects_unmarked_identity_map() -> None:
    old_index = pd.DataFrame({NODE_INDEX_COLUMN: range(2)}, index=range(2))
    old_model = _rhs_lookup_model(old_index)
    state = old_model.state_dict()

    new_index = _ttng_rhs_index(['b', 'c', 'a'])
    new_model = _rhs_lookup_model(new_index)
    new_model.set_rhs_index_map(new_index)

    with pytest.raises(RuntimeError, match='predates raw-primary-key'):
        new_model.load_state_dict(state)


@pytest.mark.parametrize(
    'rhs_sample_size,expected_rhs_sample_size',
    [
        (None, None),
        pytest.param(5, 5, id='rhs_sample_size<num_rhs_nodes'),
        pytest.param(15, None, id='rhs_sample_size==num_rhs_nodes'),
        pytest.param(30, None, id='rhs_sample_size>num_rhs_nodes'),
    ],
)
def test_sampled_postprocess_step(
    rhs_sample_size: int,
    expected_rhs_sample_size: int,
):
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_sample_size=rhs_sample_size,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
    )

    lhs_emb = torch.randn(3, 8)
    lhs_y_batch = torch.tensor([0, 0, 1, 1, 2])
    rhs_y_index = torch.tensor([10, 11, 4, 2, 10])

    assert model.rhs_sample_size == expected_rhs_sample_size
    if model.rhs_sample_size is not None:
        assert isinstance(model.rhs_embedding.lookup_embedding,
                          torch.nn.Embedding)
        assert model.rhs_embedding.lookup_embedding.sparse is True
        for edge_weights in [None, torch.rand(len(lhs_y_batch))]:
            (_, rhs_y_index_sampled, lhs_y_batch_sampled, rhs_emb_sampled,
             edge_weights_sampled,
             _) = model.sampled_postprocess_step(lhs_emb, rhs_y_index,
                                                 lhs_y_batch, edge_weights)
            assert (rhs_y_index_sampled.min() == 0
                    and rhs_y_index_sampled.max() == 3)
            assert lhs_y_batch_sampled.tolist() == [0, 0, 1, 1, 2]
            assert rhs_emb_sampled.size() == (5, 8)
            if edge_weights is None:
                assert edge_weights_sampled is None


@pytest.mark.parametrize('rhs_sample_size', [None, 10, 5_000_000])
@pytest.mark.parametrize('num_rhs_nodes', [100, 500_001, 1_000_001])
@pytest.mark.parametrize('rhs_emb_mode', [
    RHSEmbeddingMode.FEATURE,
    RHSEmbeddingMode.SHALLOW_FEATURE,
    RHSEmbeddingMode.FUSION,
])
def test_enabling_sampled_for_many_rhs_nodes(
    rhs_sample_size: int | None,
    num_rhs_nodes: int,
    rhs_emb_mode: RHSEmbeddingMode,
):
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=2, num_mp_layers=0),
        head=LinkPredHead(in_channels=2, embedding_dim=2,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=num_rhs_nodes,
        rhs_sample_size=rhs_sample_size,
        rhs_emb_mode=rhs_emb_mode,
    )
    assert model.num_rhs_nodes == num_rhs_nodes
    if rhs_emb_mode == RHSEmbeddingMode.FEATURE:
        if rhs_sample_size is None:
            if num_rhs_nodes < MAX_RHS_SAMPLE_SIZE:
                assert model.rhs_sample_size is None
            else:
                assert (model.rhs_sample_size == MAX_RHS_SAMPLE_SIZE)
        else:
            rhs_sample_size = min(rhs_sample_size, MAX_RHS_SAMPLE_SIZE)
            if num_rhs_nodes <= rhs_sample_size:
                assert model.rhs_sample_size is None
            else:
                assert model.rhs_sample_size == rhs_sample_size
    else:
        if rhs_sample_size is None:
            if num_rhs_nodes < 1_000_000:
                assert model.rhs_sample_size is None
            else:
                assert (model.rhs_sample_size == MAX_RHS_SAMPLE_SIZE)
        else:
            rhs_sample_size = min(rhs_sample_size, MAX_RHS_SAMPLE_SIZE)
            if num_rhs_nodes <= rhs_sample_size:
                assert model.rhs_sample_size is None
            else:
                assert model.rhs_sample_size == rhs_sample_size


def test_sampled_and_full_equality():
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_sample_size=None,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
    )

    # Manually patch `rhs_sample_size` to 15 to test `sampled_postprocess_step`
    # Patching this won't have any effect on full softmax code path.
    model.rhs_sample_size = 15

    lhs_emb = torch.randn(3, 8)
    lhs_y_batch = torch.tensor([0, 0, 1, 1, 2])
    rhs_y_index = torch.tensor([10, 11, 4, 2, 10])

    for edge_weights in [None, torch.rand(len(lhs_y_batch))]:
        # Full softmax
        full_logits = lhs_emb @ model.rhs_embedding().t()
        full_loss = model.compute_loss(lhs_emb, model.rhs_embedding(),
                                       full_logits, rhs_y_index, lhs_y_batch,
                                       edge_weights)

        # Sampled softmax
        out = model.sampled_postprocess_step(lhs_emb, rhs_y_index, lhs_y_batch,
                                             edge_weights)
        sampled_logits, rhs_y_index, lhs_y_batch, rhs_emb, weight, _ = out
        sampled_loss = model.compute_loss(lhs_emb, rhs_emb, sampled_logits,
                                          rhs_y_index, lhs_y_batch, weight)

        assert torch.allclose(full_loss, sampled_loss)


@pytest.mark.parametrize('rhs_sample_size', [2, 14])
def test_temporal_valid_rhs_node_idx(rhs_sample_size: int):
    rhs_sample_size = 2
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_sample_size=rhs_sample_size,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
    )

    lhs_emb = torch.randn(3, 8)
    lhs_y_batch = torch.tensor([0, 0, 1, 1, 2])
    rhs_y_index = torch.tensor([10, 11, 4, 2, 10])
    valid_rhs_node_idx = torch.tensor([10, 11, 4, 2, 1])
    model.rhs_embedding.filter(valid_rhs_node_idx)

    out = lhs_emb @ model.rhs_embedding().t()

    assert out.shape == (3, len(valid_rhs_node_idx))

    out = model.sampled_postprocess_step(lhs_emb, rhs_y_index, lhs_y_batch,
                                         None, valid_rhs_node_idx)
    assert out[0].shape == (3, min(rhs_sample_size, len(valid_rhs_node_idx)))


def test_cosine_rhs():
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_sample_size=2,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
        distance_measure='cosine',
    )

    torch.randn(3, 8)
    torch.tensor([0, 0, 1, 1, 2])
    torch.tensor([10, 11, 4, 2, 10])
    valid_rhs_node_idx = torch.tensor([10, 11, 4, 2, 1])
    model.rhs_embedding.filter(valid_rhs_node_idx)
    out = model.rhs_embedding()
    norm = torch.linalg.vector_norm(out, 2, dim=1)
    assert norm.allclose(torch.ones(5).to())


def test_reset_embedding_hook():
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
    )

    assert not model.has_embedding_hook()
    model.register_embedding_hook(TrainingStage.PRED)
    assert model.has_embedding_hook()

    # Attempting to register another hook should raise RuntimeError
    with pytest.raises(RuntimeError,
                       match="Cannot register a new embedding hook"):
        model.register_embedding_hook(TrainingStage.VAL)

    assert model.has_embedding_hook()
    model.reset_embedding_hook()
    assert not model.has_embedding_hook()

    # Should be able to register a new hook after reset
    model.register_embedding_hook(TrainingStage.TEST)
    assert model.has_embedding_hook()

    # Multiple reset calls should be safe (idempotent)
    model.reset_embedding_hook()
    assert not model.has_embedding_hook()
    model.reset_embedding_hook()
    assert not model.has_embedding_hook()


def test_predict_step_xai_with_valid_rhs_node_idx():
    """Test that predict_step works correctly with XAI enabled and
    valid_rhs_node_idx set.
    """

    num_rhs_nodes = 15
    embedding_dim = 8
    batch_size = 3
    pred_top_k = 2

    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=embedding_dim,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=num_rhs_nodes,
        rhs_sample_size=None,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
    )

    model.pred_top_k = pred_top_k

    # Set valid_rhs_node_idx to filter to only 5 of the 15 RHS nodes
    valid_rhs_node_idx = torch.tensor([10, 11, 12, 13, 14])
    model.valid_rhs_node_idx = valid_rhs_node_idx

    # Filter the embeddings and build MIPS index from filtered embeddings
    model.rhs_embedding.filter(valid_rhs_node_idx)
    filtered_rhs_emb = model.rhs_embedding()
    assert filtered_rhs_emb.shape == (len(valid_rhs_node_idx), embedding_dim)

    mips_index = MIPSFaiss(filtered_rhs_emb)
    model._mips_index = mips_index

    assert mips_index._key.shape == (len(valid_rhs_node_idx), embedding_dim)
    assert mips_index.num_keys == len(valid_rhs_node_idx)

    batch = HeteroData()
    batch['LHS'].batch_size = batch_size
    batch['LHS'].pkey = torch.tensor([100, 101, 102])
    batch['LHS'].seed_time = torch.tensor([1000, 1001, 1002])
    batch['LHS'].n_id = torch.tensor([0, 1, 2])

    lhs_emb = torch.randn(batch_size, embedding_dim)
    with patch.object(model, 'common_step', return_value=lhs_emb):
        with modeling_context.explaining(True):
            assert modeling_context.is_explaining()
            result = model.predict_step(batch, batch_idx=0)

            # Verify shape
            assert result['target'].shape == (batch_size, pred_top_k)
            assert result['target'].min() >= valid_rhs_node_idx.min()
            assert result['target'].max() <= valid_rhs_node_idx.max()
            assert result['score'].shape == (batch_size, pred_top_k)


def test_get_embedding_lhs_parity_with_predict_step():
    """get_embedding(side='lhs') must return the same tensor that
    predict_step produces for the LHS side"""
    channels = 16
    embedding_dim = 8
    batch_size = 2
    num_rhs_nodes = 15
    num_extra = 1
    n = batch_size + num_extra

    tf, col_encoder_dict = get_tf_and_col_encoder_dict()

    metadata = (['LHS', 'RHS'], [])
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={
                'LHS': col_encoder_dict,
                'RHS': col_encoder_dict,
            },
            out_channels=channels,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=channels, num_mp_layers=0, metadata=metadata),
        head=LinkPredHead(
            in_channels=channels,
            embedding_dim=embedding_dim,
            use_rhs_projector=False,
            use_mlp_head=False,
        ),
        task_type=TaskType.STATIC_LINK_PREDICTION,
        train_table_name='TRAIN_TABLE',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=num_rhs_nodes,
        rhs_emb_mode='lookup',
    )
    model.eval()
    model.pred_top_k = 0
    model._kumo_return_predict_embeddings = True

    batch = HeteroData()
    batch['LHS'].batch_size = batch_size
    batch['LHS'].n_id = torch.arange(n)
    batch['LHS'].pkey = torch.arange(n)
    batch['LHS'].tf = tf[:n]
    batch['LHS'].batch = torch.zeros(n, dtype=torch.long)

    batch['RHS'].batch_size = batch_size
    batch['RHS'].n_id = torch.arange(n)
    batch['RHS'].tf = tf[:n]
    batch['RHS'].batch = torch.zeros(n, dtype=torch.long)

    captured = {}
    handle = model.lhs_projector.register_forward_hook(
        lambda m, inp, out: captured.update(lhs_emb=out.detach().clone()),
    )
    with torch.no_grad():
        result = model.predict_step(batch, batch_idx=0)
    handle.remove()

    assert set(result) == {'pkey', 'id', 'time', 'lhs_embedding'}
    assert torch.equal(result['lhs_embedding'], captured['lhs_emb'])

    with torch.no_grad():
        emb = model.get_embedding(batch, 'LHS', side='lhs')

    assert torch.allclose(captured['lhs_emb'], emb)


def test_get_embedding_rhs_parity():
    """get_embedding(side='rhs') must match rhs_embedding(node_ids),
    """
    num_rhs_nodes = 15
    embedding_dim = 8
    batch_size = 3

    metadata = (['LHS', 'RHS'], [])
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0, metadata=metadata),
        head=LinkPredHead(
            in_channels=16,
            embedding_dim=embedding_dim,
            use_rhs_projector=False,
            use_mlp_head=False,
        ),
        task_type=TaskType.STATIC_LINK_PREDICTION,
        train_table_name='TRAIN_TABLE',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=num_rhs_nodes,
        rhs_emb_mode='lookup',
    )
    model.eval()

    rhs_node_ids = torch.arange(batch_size)
    batch = HeteroData()
    batch['RHS'].batch_size = batch_size
    batch['RHS'].n_id = rhs_node_ids

    with torch.no_grad():
        expected = model.rhs_embedding(rhs_node_ids)
        actual = model.get_embedding(batch, 'RHS', side='rhs')

    assert torch.allclose(expected, actual)


def test_static_eval_skips_past_visit_ground_truth():
    """Static-LP eval must skip get_past_visit_ground_truth (the
    DiskgraphValidRHS provider has no query_stage_until); it reads the
    pre-resolved exclude set off the batch. Regression for the static
    shallow_rhs crash."""
    channels = 16
    num_rhs_nodes = 15
    _, col_encoder_dict = get_tf_and_col_encoder_dict()
    metadata = (['LHS', 'RHS'], [])
    model = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={
                'LHS': col_encoder_dict,
                'RHS': col_encoder_dict,
            },
            out_channels=channels,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=channels, num_mp_layers=0, metadata=metadata),
        head=LinkPredHead(in_channels=channels, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.STATIC_LINK_PREDICTION,
        train_table_name='TRAIN_TABLE',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=num_rhs_nodes,
        rhs_emb_mode='lookup',
    )
    model.eval()
    model.disable_eval_metrics = True
    model.eval_top_k = 0
    model.eval_new_visit = False
    model.valid_rhs_node_idx = None

    rhs_y_index = torch.tensor([0, 1])
    lhs_y_batch = torch.tensor([0, 1])
    score = torch.randn(2, num_rhs_nodes)
    index = torch.zeros(2, num_rhs_nodes, dtype=torch.long)

    # Adapter-attached exclude set; TT-NG reads it off the batch (valid_rhs is
    # asserted None for static shallow_rhs, so no remap).
    batch = HeteroData()
    batch['LHS'].exclude_rhs_index = torch.tensor([5, 6])
    batch['LHS'].exclude_lhs_batch = torch.tensor([0, 1])

    captured = {}

    def spy_pred(_q, _emb, ex_rhs, ex_lhs):
        captured['exclude_rhs_index'] = ex_rhs
        captured['exclude_lhs_batch'] = ex_lhs
        return score, index

    with patch.object(model, 'common_step', return_value=torch.randn(2, 8)), \
         patch.object(model, 'get_ground_truth',
                      return_value=(rhs_y_index, lhs_y_batch, None)), \
         patch.object(model, 'get_rhs_pred_index_mat', side_effect=spy_pred), \
         patch.object(model, 'get_past_visit_ground_truth',
                      return_value=(None, None)) as past_visit:
        out = model.evaluation_step(batch, stage=TrainingStage.VAL, batch_idx=0)

    assert out == {}
    assert past_visit.called is False
    # The batch exclude is consumed verbatim (no remap path).
    assert torch.equal(captured['exclude_rhs_index'], torch.tensor([5, 6]))
    assert torch.equal(captured['exclude_lhs_batch'], torch.tensor([0, 1]))
