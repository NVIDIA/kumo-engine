# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for two-tower ``LinkGNNLightningModule`` negative-mask
handling in the contrastive loss.

The DG two-tower per-anchor negative path emits a ``[B, M]`` bool ``neg_mask``
(``True`` == real negative, ``False`` == budget-exhausted placeholder). These
tests assert that a masked placeholder negative does NOT change the loss
relative to a batch that omits it, for each loss variant, and that the
no-mask path is unchanged.
"""
from unittest.mock import patch

import torch
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.link_prediction import (
    LinkPredHead,
    LinkPredHeadOutput,
)
from kumo_core.train.link_gnn_module import LinkGNNLightningModule
from kumo_core.types import TaskType
from kumoapi.train import TrainingStage
from torch_geometric.data import HeteroData


def _make_module(loss_fn: str) -> LinkGNNLightningModule:
    return LinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={
                'LHS': {},
                'RHS': {},
                'TRAIN_TABLE': {}
            },
            out_channels=8,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=8, num_mp_layers=0),
        head=LinkPredHead(in_channels=8, embedding_dim=4,
                          use_rhs_projector=True, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='TRAIN_TABLE',
        entity_table_names=('LHS', 'RHS'),
        loss_fn=loss_fn,
        # Avoid the L2 reg term contributing in these checks: keep coeff 0.
        emb_l2_reg_coeff=0.0,
    )


def _make_output(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
) -> LinkPredHeadOutput:
    # `lhs_x` / `rhs_pos_x` are only used by the (disabled) L2 reg term; the
    # loss reductions under test consume only `pos_score` / `neg_score`.
    return LinkPredHeadOutput(
        pos_score=pos_score,
        neg_score=neg_score,
        lhs_x=torch.zeros(pos_score.size(0), 4),
        rhs_pos_x=torch.zeros(pos_score.size(0), 4),
        rhs_neg_x=None,
    )


def test_no_mask_unchanged() -> None:
    """With ``neg_mask=None`` the loss is identical to the legacy
    ``.mean()`` / unmasked behavior."""
    torch.manual_seed(0)
    pos = torch.randn(3, 1)
    neg = torch.randn(3, 4)

    for loss_fn in ('margin', 'bpr', 'sampled_softmax'):
        model = _make_module(loss_fn)
        out = _make_output(pos, neg)
        with_none = model.loss(out, neg_mask=None)
        default = model.loss(out)
        assert torch.allclose(with_none, default)


def test_masked_negative_equals_omitted() -> None:
    """A masked (placeholder) negative must not affect the loss: the masked
    loss over ``M`` negatives equals the unmasked loss over only the real
    ``M-k`` negatives."""
    torch.manual_seed(1)
    B = 4
    pos = torch.randn(B, 1)
    # 5 negatives; mask out the last 2 for every anchor.
    neg_full = torch.randn(B, 5)
    mask = torch.ones(B, 5, dtype=torch.bool)
    mask[:, 3:] = False
    # Replace the masked columns with garbage values to prove they are
    # ignored regardless of magnitude.
    neg_garbage = neg_full.clone()
    neg_garbage[:, 3:] = 1e6
    # Reference: loss over ONLY the real negatives (columns 0..2).
    neg_real_only = neg_full[:, :3]

    for loss_fn in ('margin', 'bpr', 'sampled_softmax'):
        model = _make_module(loss_fn)
        masked_loss = model.loss(_make_output(pos, neg_garbage), neg_mask=mask)
        ref_loss = model.loss(_make_output(pos, neg_real_only), neg_mask=None)
        assert torch.allclose(
            masked_loss, ref_loss,
            atol=1e-5), (f'{loss_fn}: masked={masked_loss} ref={ref_loss}')


def test_all_masked_row_no_nan() -> None:
    """An anchor with zero real negatives (all masked) must contribute a
    finite value (0) and not produce NaN/inf."""
    torch.manual_seed(2)
    pos = torch.randn(3, 1)
    neg = torch.randn(3, 4)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[1] = False  # anchor 1 has no real negatives.

    for loss_fn in ('margin', 'bpr', 'sampled_softmax'):
        model = _make_module(loss_fn)
        loss = model.loss(_make_output(pos, neg), neg_mask=mask)
        assert torch.isfinite(loss).all(), f'{loss_fn} produced {loss}'


def test_train_step_threads_mask_from_batch() -> None:
    """``train_step`` must read ``batch[rhs_name].dst_neg_mask`` and pass it
    into ``loss``."""
    torch.manual_seed(3)
    model = _make_module('sampled_softmax')

    pos = torch.randn(2, 1)
    neg = torch.randn(2, 3)
    out = _make_output(pos, neg)
    mask = torch.tensor([[True, False, True], [True, True, False]])

    batch = HeteroData()
    batch['RHS'].dst_neg_mask = mask

    captured = {}
    real_loss = model.loss

    def spy_loss(o, neg_mask=None):
        captured['neg_mask'] = neg_mask
        return real_loss(o, neg_mask=neg_mask)

    with patch.object(model, 'common_step', return_value=out), \
            patch.object(model, 'log_learning_rate'), \
            patch.object(model, 'log_epoch'), \
            patch.object(model, 'loss', side_effect=spy_loss):
        loss = model.train_step(batch, batch_idx=0)

    assert captured['neg_mask'] is not None
    assert torch.equal(captured['neg_mask'], mask)
    # And the returned loss matches the explicit masked computation.
    assert torch.allclose(loss, real_loss(out, neg_mask=mask))


def test_train_step_no_mask_when_absent() -> None:
    """When the RHS block has no ``dst_neg_mask`` (legacy / shared-negative /
    no-negative path), ``train_step`` passes ``neg_mask=None``."""
    torch.manual_seed(4)
    model = _make_module('margin')

    pos = torch.randn(2, 1)
    neg = torch.randn(2, 3)
    out = _make_output(pos, neg)

    batch = HeteroData()
    batch['RHS'].x = torch.randn(2, 4)  # no dst_neg_mask attribute

    captured = {}
    real_loss = model.loss

    def spy_loss(o, neg_mask=None):
        captured['neg_mask'] = neg_mask
        return real_loss(o, neg_mask=neg_mask)

    with patch.object(model, 'common_step', return_value=out), \
            patch.object(model, 'log_learning_rate'), \
            patch.object(model, 'log_epoch'), \
            patch.object(model, 'loss', side_effect=spy_loss):
        model.train_step(batch, batch_idx=0)

    assert captured['neg_mask'] is None


def _make_static_module() -> LinkGNNLightningModule:
    model = LinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(
            table_encoder_dict={
                'LHS': {},
                'RHS': {},
                'TRAIN_TABLE': {}
            },
            out_channels=8,
            time_col_dict={},
            exclude_cols_dict={},
        ),
        model=HeteroGNN(channels=8, num_mp_layers=0, metadata=(['LHS',
                                                                'RHS'], [])),
        head=LinkPredHead(in_channels=8, embedding_dim=4,
                          use_rhs_projector=True, use_mlp_head=False),
        task_type=TaskType.STATIC_LINK_PREDICTION,
        train_table_name='TRAIN_TABLE',
        entity_table_names=('LHS', 'RHS'),
    )
    model.eval()
    model.num_rhs_nodes = 10
    model.disable_eval_metrics = True
    model.eval_top_k = 0
    model.valid_rhs_node_idx = None
    return model


def _static_eval_batch(batch_size: int = 2) -> HeteroData:
    batch = HeteroData()
    batch['LHS'].n_id = torch.arange(batch_size, dtype=torch.long)
    batch['LHS'].batch_size = batch_size
    return batch


def test_static_two_tower_uses_canonical_rhs_catalog() -> None:
    model = _make_static_module()
    canonical = torch.tensor([0, 2, 4])
    model.register_buffer('canonical_rhs_node_idx', canonical, persistent=False)

    valid = model.tt_ng_valid_rhs_for_stage(TrainingStage.PRED)

    assert valid is not None
    assert torch.equal(valid, canonical)


def test_static_eval_reads_exclude_off_batch_and_skips_past_visit():
    """Two-tower static eval: reads the exclude off the batch and skips
    ``get_past_visit_ground_truth`` (the adapter pre-resolves the exclude set)."""
    model = _make_static_module()
    rhs_y_index = torch.tensor([0, 1])
    lhs_y_batch = torch.tensor([0, 1])

    batch = _static_eval_batch()
    # Adapter-attached exclude set (global RHS space); TT-NG should consume it.
    batch['LHS'].exclude_rhs_index = torch.tensor([3, 4])
    batch['LHS'].exclude_lhs_batch = torch.tensor([0, 1])

    captured = {}

    def spy_pred(_q, _emb, ex_rhs, ex_lhs):
        captured['exclude_rhs_index'] = ex_rhs
        captured['exclude_lhs_batch'] = ex_lhs
        return None, torch.zeros(2, 1, dtype=torch.long)

    with patch.object(model, 'common_step',
                      return_value={'LHS': torch.randn(2, 8)}), \
         patch.object(model.head, 'lhs_forward',
                      side_effect=lambda e: e), \
         patch.object(model, 'get_ground_truth',
                      return_value=(rhs_y_index, lhs_y_batch, None)), \
         patch.object(model, 'get_rhs_pred_index_mat', side_effect=spy_pred), \
         patch.object(model, 'update_metrics'), \
         patch.object(model, 'get_past_visit_ground_truth',
                      return_value=(None, None)) as past_visit:
        model.disable_eval_metrics = False
        model.evaluation_step(batch, stage=TrainingStage.VAL, batch_idx=0)

    assert past_visit.called is False
    # The batch exclude flowed into the MIPS exclusion (no remap, valid_rhs
    # is None here).
    assert torch.equal(captured['exclude_rhs_index'], torch.tensor([3, 4]))
    assert torch.equal(captured['exclude_lhs_batch'], torch.tensor([0, 1]))


def test_static_eval_tt_ng_remaps_exclude_to_valid_rhs():
    """With ``valid_rhs_node_idx`` set, the global exclude index is remapped to
    valid-set space and excludes outside the valid set are dropped."""
    model = _make_static_module()
    # MIPS covers valid RHS global ids [3, 7, 9] -> valid positions [0, 1, 2].
    model.valid_rhs_node_idx = torch.tensor([3, 7, 9])
    rhs_y_index = torch.tensor([3])  # global -> valid pos 0
    lhs_y_batch = torch.tensor([0])

    batch = _static_eval_batch()
    # global excludes: 7 (in valid -> pos 1), 5 (not in valid -> dropped).
    batch['LHS'].exclude_rhs_index = torch.tensor([7, 5])
    batch['LHS'].exclude_lhs_batch = torch.tensor([0, 1])

    captured = {}

    def spy_pred(_q, _emb, ex_rhs, ex_lhs):
        captured['exclude_rhs_index'] = ex_rhs
        captured['exclude_lhs_batch'] = ex_lhs
        return None, torch.zeros(2, 1, dtype=torch.long)

    with patch.object(model, 'common_step',
                      return_value={'LHS': torch.randn(2, 8)}), \
         patch.object(model.head, 'lhs_forward',
                      side_effect=lambda e: e), \
         patch.object(model, 'get_ground_truth',
                      return_value=(rhs_y_index, lhs_y_batch, None)), \
         patch.object(model, 'get_rhs_pred_index_mat', side_effect=spy_pred), \
         patch.object(model, 'update_metrics'):
        model.disable_eval_metrics = False
        model.evaluation_step(batch, stage=TrainingStage.VAL, batch_idx=0)

    # 7 -> valid pos 1 kept; 5 dropped.
    assert torch.equal(captured['exclude_rhs_index'], torch.tensor([1]))
    assert torch.equal(captured['exclude_lhs_batch'], torch.tensor([0]))
