# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pandas as pd
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.modeling.embedding import Embedding


def test_embedding_basic() -> None:
    embedding = Embedding(10, 16)
    assert str(embedding) == 'Embedding(10, 16)'

    out = embedding()
    assert out.size() == (10, 16)
    assert torch.equal(out, embedding.weight[:10])

    out = embedding(torch.arange(10))
    assert out.size() == (10, 16)
    assert torch.equal(out, embedding.weight[:10])

    out = embedding(torch.arange(3, 5))
    assert out.size() == (2, 16)
    assert torch.equal(out, embedding.weight[3:5])


@withCUDA
def test_embedding_reorder(device: torch.device) -> None:
    index_map = pd.DataFrame({
        'pkey': ['a', 'b', 'c', 'd', 'e', 'z'],
        'arange': [0, 1, 2, 3, 4, 5]
    })
    index_map.set_index('pkey', inplace=True)
    embedding = Embedding(6, 10, index_map=index_map).to(device)
    embedding.reset_parameters()

    new_index_map = pd.DataFrame({
        'pkey': ['a', 'c', 'd', 'f', 'g', 'h', 'i', 'j'],
        'arange': [6, 2, 1, 0, 4, 3, 5, 7]
    })
    new_index_map.set_index('pkey', inplace=True)

    old_embedding = copy.deepcopy(embedding)
    embedding.order_by(new_index_map, num_embeddings=8)
    mean_tensor = old_embedding.weight.data.mean(dim=0).unsqueeze(0)

    # Shape:
    assert embedding.weight.size() == (8, 10)

    # Valid indices ('d' and 'c'):
    old_idx = torch.tensor([3, 2], device=device)
    new_idx = torch.tensor([1, 2], device=device)

    assert torch.equal(old_embedding(old_idx), embedding(new_idx))

    # Invalid index: present in old, not in new ('e'):
    with pytest.warns(UserWarning,
                      match="Requested embeddings of invalid rows."):
        idx = torch.tensor([4], device=device)
        assert torch.equal(embedding(idx), mean_tensor)

    # Invalid index: present in new, not in old ('i'):
    idx = torch.tensor([5], device=device)
    assert torch.equal(embedding(idx), mean_tensor)

    assert torch.equal(old_embedding(torch.tensor([0], device=device)),
                       embedding(torch.tensor([6], device=device)))


def test_sparse_constructor_normalizes_without_mutating_and_masks_holes(
) -> None:
    index_map = pd.DataFrame({'node_id': [0, 2, 4]}, index=['a', 'b', 'c'])

    embedding = Embedding(5, 2, index_map=index_map)

    assert list(index_map.columns) == ['node_id']
    assert embedding.index_map is not None
    assert list(embedding.index_map.columns) == ['arange']
    assert embedding.mask.tolist() == [False, True, False, True, False]


@pytest.mark.parametrize(
    'index_map,error',
    [
        (pd.DataFrame({'id': [0, 1]}, index=['a', 'a'
                                             ]), 'non-unique primary-key'),
        (pd.DataFrame({'id': [0, 0]}, index=['a', 'b'
                                             ]), 'duplicate physical node'),
        (pd.DataFrame({'id': [0.0, 1.0]}, index=['a', 'b']), 'integral'),
        (pd.DataFrame({'id': [0, None]}, index=['a', 'b'
                                                ]), 'null physical node'),
        (pd.DataFrame({'id': [-1, 0]}, index=['a', 'b']), 'negative'),
        (pd.DataFrame({'id': [0, 2]}, index=['a', 'b']), 'capacity'),
    ],
)
def test_embedding_rejects_invalid_index_maps(index_map: pd.DataFrame,
                                              error: str) -> None:
    with pytest.raises(ValueError, match=error):
        Embedding(2, 2, index_map=index_map)


def test_embedding_rejects_duplicate_checkpoint_primary_keys() -> None:
    source = Embedding(
        2,
        2,
        index_map=pd.DataFrame({'id': [0, 1]}, index=['a', 'b']),
    )
    state = source.state_dict()
    state['_extra_state'] = {
        'index_map': pd.DataFrame({'id': [0, 1]}, index=['a', 'a'])
    }

    with pytest.raises(ValueError, match='non-unique primary-key'):
        Embedding(2, 2).load_state_dict(state)


def test_embedding_rejects_duplicate_destination_primary_keys() -> None:
    embedding = Embedding(
        2,
        2,
        index_map=pd.DataFrame({'id': [0, 1]}, index=['a', 'b']),
    )
    duplicate_map = pd.DataFrame({'id': [0, 1]}, index=['a', 'a'])

    with pytest.raises(ValueError, match='non-unique primary-key'):
        embedding.order_by(duplicate_map, num_embeddings=2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_sparse_low_precision_mean_accumulates_in_float32(
        dtype: torch.dtype) -> None:
    count = 65_536
    old_map = pd.DataFrame({'id': range(count)}, index=range(count))
    embedding = Embedding(count + 1, 1, index_map=old_map).to(dtype)
    with torch.no_grad():
        embedding.weight.fill_(1)
    new_map = pd.DataFrame({'id': range(count + 1)}, index=range(count + 1))

    embedding.order_by(new_map, num_embeddings=count + 2)

    assert embedding.weight.dtype == dtype
    assert torch.isfinite(embedding.weight).all()
    assert embedding.weight[count].item() == 1.0
