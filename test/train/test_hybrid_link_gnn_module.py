# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from kumo_core.train.hybrid_link_gnn_module import HybridLinkGNNLightningModule
from torch_geometric.data import HeteroData


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
    model = HybridLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=True),
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_sample_size=rhs_sample_size,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
    )

    lhs_emb = torch.randn(3, 16)  # unprojected
    lhs_y_batch = torch.tensor([0, 0, 1, 1, 2])
    rhs_y_index = torch.tensor([10, 11, 4, 2, 10])
    lhs_idgnn_batch = lhs_y_batch
    rhs_idgnn_index = rhs_y_index
    rhs_gnn_emb = torch.randn(len(lhs_idgnn_batch), 16)

    assert model.rhs_sample_size == expected_rhs_sample_size
    if model.rhs_sample_size is not None:
        assert isinstance(model.rhs_embedding.lookup_embedding,
                          torch.nn.Embedding)
        assert model.rhs_embedding.lookup_embedding.sparse is True
        (_, rhs_y_index, lhs_y_batch, _, rhs_emb, lhs_emb_proj,
         _) = model.sampled_postprocess_step(lhs_emb, rhs_gnn_emb,
                                             rhs_idgnn_index, lhs_idgnn_batch,
                                             rhs_y_index, lhs_y_batch, None)

        assert rhs_y_index is not None
        assert lhs_y_batch is not None
        assert rhs_y_index.min() == 0 and rhs_y_index.max() == 3
        assert lhs_y_batch.tolist() == [0, 0, 1, 1, 2]
        assert rhs_emb.size() == (5, 8)
        assert lhs_emb_proj.size() == (3, 8)


def test_sampled_and_full_equality():
    model = HybridLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=True),
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

    lhs_emb = torch.randn(3, 16)  # unprojected
    lhs_y_batch = torch.tensor([0, 0, 1, 1, 2])
    rhs_y_index = torch.tensor([10, 11, 4, 2, 10])
    lhs_idgnn_batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    rhs_idgnn_index = torch.tensor([0, 1, 8, 12, 13, 2, 6, 7])
    rhs_gnn_emb = torch.randn(len(rhs_idgnn_index), 16)

    full_logits, lhs_emb_projected = model.postprocess_step(
        lhs_emb, rhs_gnn_emb, rhs_idgnn_index, lhs_idgnn_batch,
        model.rhs_embedding())
    for edge_weights in [None, torch.rand(len(lhs_y_batch))]:
        full_loss = model.compute_loss(lhs_emb_projected, model.rhs_embedding(),
                                       full_logits, rhs_y_index, lhs_y_batch,
                                       edge_weights)

        out = model.sampled_postprocess_step(lhs_emb, rhs_gnn_emb,
                                             rhs_idgnn_index, lhs_idgnn_batch,
                                             rhs_y_index, lhs_y_batch, None)
        logits, rhs_y_index, lhs_y_batch, _, rhs_emb, lhs_emb_proj, _ = out
        assert rhs_y_index is not None
        assert lhs_y_batch is not None
        sampled_loss = model.compute_loss(lhs_emb_proj, rhs_emb, logits,
                                          rhs_y_index, lhs_y_batch,
                                          edge_weights)
        assert torch.allclose(full_loss, sampled_loss)


@pytest.mark.parametrize('rhs_sample_size', [2])
def test_temporal_valid_rhs_node_idx(rhs_sample_size: int):
    model = HybridLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=True),
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_sample_size=rhs_sample_size,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
    )

    lhs_emb = torch.randn(3, 16)  # unprojected
    lhs_y_batch = torch.tensor([0, 0, 1, 1, 2])
    rhs_y_index = torch.tensor([10, 11, 4, 2, 10])
    lhs_idgnn_batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    rhs_idgnn_index = torch.tensor([2, 1, 4, 1, 1, 10, 4, 2])
    rhs_gnn_emb = torch.randn(len(rhs_idgnn_index), 16)
    valid_rhs_node_idx = torch.tensor([10, 11, 4, 2, 1])
    model.rhs_embedding.filter(valid_rhs_node_idx)

    # All rhs_idgnn_index are valid
    out = model.postprocess_step(lhs_emb, rhs_gnn_emb,
                                 rhs_idgnn_index, lhs_idgnn_batch,
                                 model.rhs_embedding(), valid_rhs_node_idx)
    assert out[0].shape == (3, len(valid_rhs_node_idx))

    out = model.sampled_postprocess_step(lhs_emb, rhs_gnn_emb, rhs_idgnn_index,
                                         lhs_idgnn_batch, rhs_y_index,
                                         lhs_y_batch, None, valid_rhs_node_idx)
    assert out[0].shape == (3, min(rhs_sample_size, len(valid_rhs_node_idx)))
    assert out[4].shape == (min(rhs_sample_size, len(valid_rhs_node_idx)), 8)

    rhs_idgnn_index = torch.tensor([0, 1, 8, 12, 13, 2, 6, 7])


def test_cosine_rhs():
    model = HybridLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=True),
        train_table_name='train_table_name',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_sample_size=2,
        rhs_emb_mode='lookup',
        emb_l2_reg_coeff=0.1,
        distance_measure='cosine',
    )

    torch.randn(3, 16)
    torch.tensor([0, 0, 1, 1, 2])
    torch.tensor([10, 11, 4, 2, 10])
    valid_rhs_node_idx = torch.tensor([10, 11, 4, 2, 1])
    model.rhs_embedding.filter(valid_rhs_node_idx)
    out = model.rhs_embedding()
    norm = torch.linalg.vector_norm(out, 2, dim=1)
    assert norm.allclose(torch.ones(5).to())


def test_predict_step_returns_requested_lhs_embedding():
    model = HybridLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={'RHS': {}},
            out_channels=2,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=16, num_mp_layers=0),
        head=LinkPredHead(in_channels=16, embedding_dim=8,
                          use_rhs_projector=False, use_mlp_head=True),
        train_table_name='TRAIN_TABLE',
        entity_table_names=('LHS', 'RHS'),
        num_rhs_nodes=15,
        rhs_emb_mode='lookup',
    )
    lhs_embedding = torch.randn(2, 16)
    model.common_step = lambda batch: (
        lhs_embedding,
        torch.empty(0, 16),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    projected = None

    def postprocess(lhs, *args, **kwargs):
        nonlocal projected
        projected = model.lhs_projector(lhs)
        return torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long)

    model.postprocess_eval_step = postprocess
    model.pred_top_k = 1
    model._kumo_return_predict_embeddings = True
    batch = HeteroData()
    batch['LHS'].batch_size = 2
    batch['LHS'].pkey = torch.tensor([10, 11])
    batch['LHS'].seed_time = torch.tensor([100, 200])
    batch['LHS'].n_id = torch.tensor([0, 1])

    output = model.predict_step(batch, batch_idx=0)

    assert projected is not None
    assert torch.equal(output['lhs_embedding'], projected)
