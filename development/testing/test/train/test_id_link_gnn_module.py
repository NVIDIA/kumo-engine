# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import torch
from kumo_core.train.id_link_gnn_module import IDLinkGNNLightningModule
from torch_geometric.metrics import LinkPredRecall


def _module_stub(num_rhs_nodes: int) -> IDLinkGNNLightningModule:
    # ``get_rhs_pred_index_mat`` only reads ``num_rhs_nodes`` and
    # ``metric_store.get_metric_collection(stage).max_k``; stub just those.
    return cast(
        IDLinkGNNLightningModule,
        SimpleNamespace(
            num_rhs_nodes=num_rhs_nodes,
            metric_store=SimpleNamespace(
                get_metric_collection=lambda stage: SimpleNamespace(max_k=100)),
        ),
    )


def test_get_rhs_pred_index_mat_pads_with_out_of_range_sentinel():
    r"""``get_rhs_pred_index_mat`` must pad short rows with the out-of-range
    sentinel ``num_rhs_nodes`` rather than ``-1``.

    id_gnn ranks only the RHS nodes reachable in each LHS anchor's sampled
    subgraph, so per-anchor candidate pools are typically far smaller than the
    metric's ``k`` and the dense top-k matrix is mostly padding. The matrix is
    consumed by torch_geometric's ``LinkPredMetricData``, which encodes a
    prediction ``(row, rhs_id)`` as ``row * max_index + rhs_id`` and matches it
    against ground-truth edges encoded the same way. A ``-1`` slot encodes to
    ``(row - 1) * max_index + (max_index - 1)`` -- the code of the real edge
    ``(row - 1, max_index - 1)`` -- so whenever ``max(ground_truth_rhs) >=
    max(predicted_rhs)`` every padding slot is scored as a correct prediction
    and recall/MAP are silently inflated (observed end-to-end: id_gnn
    recall@100 jumping from the honest 0.0 to 0.98). A non-negative sentinel
    ``>= num_rhs_nodes`` can never collide with a real ``(row, rhs_id)`` code.
    """
    num_rhs_nodes = 1000
    module = _module_stub(num_rhs_nodes)

    # Two LHS anchors. Anchor 0 has a single reachable candidate (RHS id 5);
    # anchor 1 has none -> its whole row is padding.
    logits = torch.tensor([2.0])
    rhs_pred_index = torch.tensor([5])
    lhs_pred_batch = torch.tensor([0])
    # Ground truth: both anchors' true RHS is id 999 (the largest id in play),
    # and neither anchor has it as a candidate -> honest recall is 0.0.
    # 999 >= max(predicted)=5 is exactly the condition that triggers the bug.
    lhs_y_batch = torch.tensor([0, 1])
    rhs_y_index = torch.tensor([999, 999])

    mat = IDLinkGNNLightningModule.get_rhs_pred_index_mat(
        module, logits, rhs_pred_index, lhs_pred_batch, lhs_y_batch)

    # Padding must be the out-of-range sentinel, never a negative index.
    assert (mat < 0).sum().item() == 0
    assert int((mat == num_rhs_nodes).sum().item()) > 0
    # The real candidate is preserved.
    assert int(mat[0, 0].item()) == 5

    # The metric over this matrix must report the honest 0.0. With ``-1``
    # padding this returned ~1.0 (the padding aliased the id-999 ground truth).
    recall = LinkPredRecall(k=100)
    recall.update(mat, torch.stack([lhs_y_batch, rhs_y_index]))
    assert float(recall.compute()) == 0.0


def test_get_rhs_pred_index_mat_counts_true_candidate_hit():
    r"""Sanity check the inverse: when an anchor's true RHS *is* a candidate it
    must still be scored as a hit after the sentinel-padding change.
    """
    module = _module_stub(num_rhs_nodes=1000)
    # Anchor 0 has candidate 7 which IS its true RHS; anchor 1 has only a
    # non-matching candidate.
    logits = torch.tensor([3.0, 1.0])
    rhs_pred_index = torch.tensor([7, 3])
    lhs_pred_batch = torch.tensor([0, 1])
    lhs_y_batch = torch.tensor([0, 1])
    rhs_y_index = torch.tensor([7, 999])

    mat = IDLinkGNNLightningModule.get_rhs_pred_index_mat(
        module, logits, rhs_pred_index, lhs_pred_batch, lhs_y_batch)

    recall = LinkPredRecall(k=100)
    recall.update(mat, torch.stack([lhs_y_batch, rhs_y_index]))
    # Anchor 0 retrieves its positive, anchor 1 does not -> recall = 0.5.
    assert float(recall.compute()) == 0.5
