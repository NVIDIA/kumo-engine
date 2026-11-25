# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feed model-format DiskGraph tensors into the released Pinterest OmniSage embedder."""

import importlib.util
from pathlib import Path

import torch

from tests.e2e.omnisage_fixture import CANDIDATE_NAMES, KEEP


def released_model(model_file):
    """Load an atg-research checkout and use its unmodified Pin–board transformer."""
    path = Path(model_file).resolve(strict=True)
    module_spec = importlib.util.spec_from_file_location("pinterest_omnisage_model", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.GraphSageV6AlphaTransformerAggregator(
        graph_to_num_neighbors={"pin_board": {0: KEEP["pins"], 1: KEEP["boards"]}},
        entity_to_feature_names={
            0: ["sig_annotations_v5_k50_terms"],
            1: ["board_annotations_v5_terms"],
        },
        self_feature_dims={},
        neighbor_feature_dims={},
        hidden_size=32,
        mlp_size=128,
        heads=4,
        num_layers=1,
        emb_size=32,
        dropout=0.0,
    )


def _ranked(graph, table, field, roots):
    edge = graph["edges"][f"walk.{table}"]
    values = graph["nodes"][table]["features"][field]["values"]
    source, dest = edge["index"].long()
    rank = edge["rank"].long()
    features = values.new_zeros(roots, KEEP[table], values.shape[-1])
    weight = edge["score"].new_zeros(roots, KEEP[table])
    features[source, rank] = values[dest]
    weight[source, rank] = edge["score"]
    return features, weight


def _features(graph, roots):
    pins = graph["nodes"]["pins"]["features"]
    images = pins["image"]["values"][:roots]
    if images.shape[1] != 1152:
        raise ValueError("the released model requires 1,024 visual and 128 annotation features")
    neighbors, scores = _ranked(graph, "pins", "image", roots)
    boards, board_scores = _ranked(graph, "boards", "annotation", roots)
    # The released model slices the same ranked Pin visual tensor for both entity
    # types. Boards supply their own annotation tensor and walk weights.
    return {
        "ue_v4": images[:, :1024],
        "sig_annotations_v5_k50_terms": images[:, 1024:],
        "self_degree": pins["degree"]["values"][:roots].float().reshape(-1, 1),
        "neigh_pin_board_ue_v4": neighbors[:, :, :1024],
        "neigh_pin_board_sig_annotations_v5_k50_terms": neighbors[:, :, 1024:],
        "neigh_pin_board_board_annotations_v5_terms": boards,
        "neighbor_weight_pin_board_signature": scores,
        "neighbor_weight_pin_board_board": board_scores,
    }


def train_step(batch, model, optimizer):
    """Use all three positive sources and exclude each anchor's known positives from negatives."""
    size = len(batch.metadata["sample_ids"])
    query_graph = batch.inputs["query"]
    query_ids = query_graph["nodes"]["pins"]["ids"][:size]
    features = [_features(query_graph, size)]
    owners, ids = [], []
    for name in CANDIDATE_NAMES:
        graph = batch.inputs[name]
        counts = graph["candidate_offsets"].long().diff()
        owner = torch.repeat_interleave(torch.arange(size), counts)
        if len(owner):
            features.append(_features(graph, len(owner)))
            owners.append(owner)
            ids.append(graph["nodes"]["pins"]["ids"][: len(owner)])
    if not owners:
        raise ValueError("no training pairs in this batch")
    owner, candidate_ids = torch.cat(owners), torch.cat(ids)
    optimizer.zero_grad(set_to_none=True)
    inputs = {key: torch.cat([data[key] for data in features]) for key in features[0]}
    embeddings, _ = model(inputs)
    scores = embeddings[:size] @ embeddings[size:].T / 0.1
    unique, inverse = torch.unique(candidate_ids, return_inverse=True)
    positives = torch.zeros(size, len(unique), dtype=torch.bool)
    positives[owner, inverse] = True
    allowed = ~(positives[:, inverse] | (query_ids[:, None] == candidate_ids))
    usable = allowed.any(dim=1)[owner]
    if not usable.any():
        raise ValueError("no eligible in-batch negatives for this batch")
    # Only include rows with valid negatives in logsumexp and its backward pass.
    active = allowed.any(dim=1)
    negative_sum = scores.new_zeros(size)
    negative_sum[active] = scores[active].masked_fill(~allowed[active], -torch.inf).logsumexp(1)
    positive = scores[owner, torch.arange(len(owner))]
    loss = torch.nn.functional.softplus(negative_sum[owner[usable]] - positive[usable]).mean()
    if not torch.isfinite(loss):
        raise AssertionError("non-finite OmniSage loss")
    loss.backward()
    optimizer.step()
    return loss.detach().item(), int(usable.sum())
