# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import pandas as pd
import pytest
import torch
from kumoapi.encoder import Encoder, Hash, Index
from kumoapi.typing import ColStatType
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.categorical import FusedCategoricalEncoder


@withCUDA
def test_forward(device: torch.device) -> None:
    data = pd.DataFrame(
        data=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        columns=[
            'hash0', 'hash1', 'hash2', 'index0', 'index1', 'index2', 'index3',
            'exclude'
        ],
    )
    num_rows, num_cols = data.shape
    col_encoder_dict = {
        'hash0': Hash(na_strategy='separate', num_components=2),
        'hash1': Hash(na_strategy='separate', num_components=3),
        'hash2': Hash(na_strategy='most_frequent', num_components=2),
        'index0': Index(na_strategy='zero'),
        'index1': Index(na_strategy='separate'),
        'index2': Index(na_strategy='most_frequent'),
        'index3': Index(na_strategy='separate', min_occ=100),
        'exclude': Index(na_strategy='most_frequent'),
    }
    for cfg in col_encoder_dict.values():
        # TODO(akihironitta): Test unordered category counts
        cfg._stats[ColStatType.CATEGORY_COUNTS] = [
            ['A', 5],
            ['B', 4],
            ['C', 3],
            ['D', 2],
            ['E', 1],
        ]
    out_channels = 3
    feat = torch.from_numpy(data.values).to(device=device, dtype=torch.long)
    encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=['exclude'],
        out_channels=out_channels,
        enc_mode='fuse',
        device=device,
    )
    assert encoder.should_run_hash is True
    assert torch.allclose(
        encoder.na_thresholds,
        torch.tensor([5, 5, 5, 5, 5, 5, 0, 5], device=device),
    )
    assert torch.allclose(
        encoder.modulo_values,
        torch.tensor([2, 3, 2, 5, 5, 5, 1, 5], device=device),
    )
    assert torch.allclose(
        encoder.exclude_cols_mask,
        torch.tensor([False, False, False, False, False, False, False, True],
                     device=device),
    )
    num_embeddings = (
        # hash0(separate): num_components + 1
        2 + 1
        # hash1(separate): num_components + 1
        + 3 + 1
        # hash2(most_frequent): num_components
        + 2
        # index0(zero): num_categories
        + 5
        # index1(separate): num_categories + 1
        + 5 + 1
        # index2(most_frequent): num_categories
        + 5
        # index3(separate, min_occ=100): num_categories + 1
        + 0 + 1
        # exclude(most_frequent): num_categories
        + 5
        # a global zero vector for exclude_cols and zero
        + 1)
    assert encoder.emb.weight.size() == (num_embeddings, out_channels)
    # Only the last index of embeddings should always be a zero vector
    assert encoder.zero_vector_index == num_embeddings - 1
    assert torch.allclose(
        encoder.emb.weight[-1],
        torch.zeros(out_channels, device=device),
    )
    for i in range(num_embeddings - 1):
        assert not torch.allclose(
            encoder.emb.weight[i],
            torch.zeros(out_channels, device=device),
        )
    assert torch.allclose(
        encoder.impute_values,
        torch.tensor(
            [
                # Hash(num_components=2, na_strategy='separate')
                2,
                # Hash(num_components=3, na_strategy='separate')
                3,
                # Hash(num_components=2, na_strategy='most_frequent')
                0,
                # Index(na_strategy='zero')
                encoder.zero_vector_index - encoder.cumulative_offset[3],
                # Index(na_strategy='separate')
                5,
                # Index(na_strategy='most_frequent')
                0,
                # Index(na_strategy='separate', min_occ=100)
                0,
                # Index(na_strategy='most_frequent')
                0,
            ],
            device=device,
        ),
    )
    out = encoder(feat)
    assert out.size() == (num_rows, out_channels)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_empty_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    col_encoder_dict: dict[str, Encoder] = {
        'col0': Index(),
        'col1': Index(),
        'col2': Index(),
    }
    for cfg in col_encoder_dict.values():
        cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 10], ['B', 10]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=5,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(torch.empty(0, 3, dtype=torch.long, device=device))
    assert out.size() == (0, 5) if enc_mode == 'fuse' else (0, 3, 5)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_all_na_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    col_encoder_dict: dict[str, Encoder] = {
        'col0': Index(),
        'col1': Index(),
    }
    for cfg in col_encoder_dict.values():
        cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 10]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=3,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(torch.tensor([[-1, -1]], device=device))
    assert out.size() == (1, 3) if enc_mode == 'fuse' else (1, 2, 3)


@withCUDA
def test_na_strategy_without_min_occ(device: torch.device) -> None:
    col_encoder_dict: dict[str, Encoder] = {
        'col0': Index(na_strategy='most_frequent'),
        'col1': Index(na_strategy='separate'),
        'col2': Index(na_strategy='zero'),
    }
    col_encoder_dict['col0']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    col_encoder_dict['col1']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    col_encoder_dict['col2']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=7,
        enc_mode='column',
        device=device,
    )

    WHATEVER = 0  # Input value that doesn't matter in the test
    # Test most_frequent maps -1 to the most frequent category 0:
    assert torch.allclose(
        encoder(torch.tensor([[0, WHATEVER, WHATEVER]],
                             device=device))[:, 0, :],  # col0
        encoder(torch.tensor([[-1, WHATEVER, WHATEVER]],
                             device=device))[:, 0, :],  # col0
    )

    # Test separate maps -1 to none of existing categories:
    assert not torch.allclose(  # Test that it doesn't map -1 to category 0:
        encoder(torch.tensor([[WHATEVER, 0, WHATEVER]],
                             device=device))[:, 1, :],  # col1
        encoder(torch.tensor([[WHATEVER, -1, WHATEVER]],
                             device=device))[:, 1, :],  # col1
    )
    assert not torch.allclose(  # Test that it doesn't map -1 to category 1:
        encoder(torch.tensor([[WHATEVER, 1, WHATEVER]],
                             device=device))[:, 1, :],  # col1
        encoder(torch.tensor([[WHATEVER, -1, WHATEVER]],
                             device=device))[:, 1, :],  # col1
    )

    # Test zero maps -1 to a zero vector:
    assert torch.allclose(
        torch.zeros(1, 7, device=device),
        encoder(torch.tensor([[WHATEVER, WHATEVER, -1]],
                             device=device))[:, 2, :],  # col2
    )


@withCUDA
def test_na_strategy_with_min_occ(device: torch.device) -> None:
    col_encoder_dict: dict[str, Encoder] = {
        'col0': Index(na_strategy='most_frequent', min_occ=2),
        'col1': Index(na_strategy='separate', min_occ=2),
        'col2': Index(na_strategy='zero', min_occ=2),
    }
    col_encoder_dict['col0']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    col_encoder_dict['col1']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    col_encoder_dict['col2']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=7,
        enc_mode='column',
        device=device,
    )

    WHATEVER = 0  # Input value that doesn't matter in the test
    # Test most_frequent maps -1 and 1 to the most frequent category 0 as the
    # category 1 doesn't satisfy the min_occ=2 requirement:
    assert torch.allclose(
        encoder(torch.tensor([[0, WHATEVER, WHATEVER]],
                             device=device))[:, 0, :],  # col0
        encoder(torch.tensor([[1, WHATEVER, WHATEVER]],
                             device=device))[:, 0, :],  # col0
    )
    assert torch.allclose(
        encoder(torch.tensor([[-1, WHATEVER, WHATEVER]],
                             device=device))[:, 0, :],  # col0
        encoder(torch.tensor([[1, WHATEVER, WHATEVER]],
                             device=device))[:, 0, :],  # col0
    )

    # Test separate maps -1 and 1 to none of existing categories as the
    # category 1 doesn't satisfy the min_occ=2 requirement:
    assert not torch.allclose(  # Test it doesn't map 1 to category 0:
        encoder(torch.tensor([[WHATEVER, 0, WHATEVER]],
                             device=device))[:, 1, :],  # col1
        encoder(torch.tensor([[WHATEVER, 1, WHATEVER]],
                             device=device))[:, 1, :],  # col1
    )
    assert torch.allclose(  # Test that -1 and 1 are mapped to the same vector:
        encoder(torch.tensor([[WHATEVER, -1, WHATEVER]],
                             device=device))[:, 1, :],  # col1
        encoder(torch.tensor([[WHATEVER, 1, WHATEVER]],
                             device=device))[:, 1, :],  # col1
    )

    # Test zero maps -1 and 1 to a zero vector as the category 1 doesn't
    # satisfy the min_occ=2 requirement:
    assert torch.allclose(
        torch.zeros(1, 7, device=device),
        encoder(torch.tensor([[WHATEVER, WHATEVER, -1]],
                             device=device))[:, 2, :],  # col2
    )
    assert torch.allclose(
        torch.zeros(1, 7, device=device),
        encoder(torch.tensor([[WHATEVER, WHATEVER, 1]],
                             device=device))[:, 2, :],  # col2
    )


@withCUDA
def test_enc_mode(device: torch.device) -> None:
    num_rows, num_cols = 5, 3
    feat = torch.randint(2, (num_rows, num_cols), device=device)
    col_encoder_dict: dict[str, Encoder] = {
        'col0': Index(na_strategy='most_frequent'),
        'col1': Index(na_strategy='separate'),
        'col2': Index(na_strategy='zero'),
    }
    col_encoder_dict['col0']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    col_encoder_dict['col1']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    col_encoder_dict['col2']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    out_channels = 7

    # ===============
    # enc_mode='fuse'
    # ===============
    fused_encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='fuse',
        device=device,
    )

    out = fused_encoder(feat)
    assert out.size() == (num_rows, out_channels)
    # =================
    # enc_mode='column'
    # =================
    fused_encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='column',
        device=device,
    )
    out = fused_encoder(feat)
    assert out.size() == (num_rows, num_cols, out_channels)
    # ================
    # invalid enc_mode
    # ================
    with pytest.raises(ValueError, match="Unsupported encoder mode 'invalid'"):
        FusedCategoricalEncoder(
            col_encoder_dict=col_encoder_dict,
            exclude_cols=[],
            out_channels=out_channels,
            enc_mode='invalid',
        )


def test_inconsistent_col_order() -> None:
    col_encoder_dict: dict[str, Encoder] = {'col0': Index(), 'col1': Index()}
    col_encoder_dict['col0']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    col_encoder_dict['col1']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3],
                                                                    ['B', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=3,
        enc_mode='column',
    )
    # Test raise no error:
    encoder(torch.zeros(1, 2, dtype=torch.long), ['col0', 'col1'])
    with pytest.raises(ValueError, match='The order of columns'):
        encoder(torch.zeros(1, 2, dtype=torch.long), ['col1', 'col0'])


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_gradient(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    col_encoder_dict: dict[str, Encoder] = {'col0': Index(), 'col1': Index()}
    col_encoder_dict['col0']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 2],
                                                                    ['B', 1]]
    col_encoder_dict['col1']._stats[ColStatType.CATEGORY_COUNTS] = [['A', 2],
                                                                    ['B', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=['col1'],
        out_channels=8,
        enc_mode=enc_mode,
        device=device,
    )
    with modeling_context.explaining():
        out = encoder(torch.randint(-1, 2, (5, 2), device=device))

    out.backward(torch.randn_like(out))
    grad = encoder.get_grad_wrt_input()
    assert isinstance(grad, torch.Tensor)
    assert grad.size() == (5, 2)
    assert not torch.allclose(grad[:, 0], torch.zeros(5, device=device))
    assert torch.allclose(grad[:, 1], torch.zeros(5, device=device))
    assert not hasattr(encoder, 'saved_tensors')


def test_index_zero_num_min_occ_categories_eq_num_categories() -> None:
    cfg = Index(na_strategy='zero')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 4  # ['A', 'B', 'C', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1], [2]])),
        encoder.emb.weight[[[[0], [1], [2]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[99]])),
        encoder.emb.weight[[[[3]]]],
    )

    # Test N/A is mapped to a zero vector
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        torch.zeros(1, 1, 2),
    )


def test_index_zero_num_min_occ_categories_lt_num_categories() -> None:
    cfg = Index(min_occ=2, na_strategy='zero')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 3  # ['A', 'B', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1]])),
        encoder.emb.weight[[[[0], [1]]]],
    )

    # Test categories that don't fulfil min_occ constrant are treated as N/A
    assert torch.allclose(
        encoder(torch.tensor([[2]])),
        torch.zeros(1, 1, 2),
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[99]])),
        torch.zeros(1, 1, 2),
    )

    # Test N/A is mapped to a zero vector
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        torch.zeros(1, 1, 2),
    )


def test_index_zero_num_min_occ_categories_eq_zero() -> None:
    cfg = Index(min_occ=4, na_strategy='zero')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 1  # [zero]
    assert torch.allclose(encoder.emb.weight, torch.zeros(1, 2))

    # Test everything is mapped to zeros
    assert torch.allclose(
        encoder(torch.tensor([[-1], [0], [1], [2], [99]])),
        torch.zeros(5, 1, 2),
    )


def test_index_separate_num_min_occ_categories_eq_num_categories() -> None:
    cfg = Index(na_strategy='separate')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 5  # ['A', 'B', 'C', separate, zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1], [2]])),
        encoder.emb.weight[[[[0], [1], [2]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[99]])),
        encoder.emb.weight[[[[3]]]],
    )

    # Test N/A is mapped to a separate category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[3]]]],
    )


def test_index_separate_num_min_occ_categories_lt_num_categories() -> None:
    cfg = Index(min_occ=3, na_strategy='separate')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 3  # ['A', separate, zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0]])),
        encoder.emb.weight[[[[0]]]],
    )

    # Test categories that don't fulfil min_occ constrant are treated as N/A
    assert torch.allclose(
        encoder(torch.tensor([[1], [2]])),
        encoder.emb.weight[[[[1], [1]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[99]])),
        encoder.emb.weight[[[[1]]]],
    )

    # Test N/A is mapped to a separate category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[1]]]],
    )


def test_index_separate_num_min_occ_categories_eq_zero() -> None:
    cfg = Index(min_occ=4, na_strategy='zero')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 1  # [zero]
    assert torch.allclose(encoder.emb.weight, torch.zeros(1, 2))

    # Test everything is mapped to zeros
    assert torch.allclose(
        encoder(torch.tensor([[-1], [0], [1], [2], [99]])),
        torch.zeros(5, 1, 2),
    )


def test_index_most_frequent_num_min_occ_categories_eq_num_categories(
) -> None:
    cfg = Index(na_strategy='most_frequent')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 4  # ['A', 'B', 'C', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1], [2]])),
        encoder.emb.weight[[[[0], [1], [2]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[3], [99]])),
        encoder.emb.weight[[[[0], [0]]]],
    )

    # Test N/A is mapped to the most frequent category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[0]]]],
    )


def test_index_most_frequent_num_min_occ_categories_lt_num_categories(
) -> None:
    cfg = Index(min_occ=3, na_strategy='most_frequent')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 2  # ['A', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0]])),
        encoder.emb.weight[[[[0]]]],
    )

    # Test categories that don't fulfil min_occ constrant are treated as N/A
    assert torch.allclose(
        encoder(torch.tensor([[1], [2]])),
        encoder.emb.weight[[[[0], [0]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[99]])),
        encoder.emb.weight[[[0]]],
    )

    # Test N/A is mapped to the most frequent category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[0]]]],
    )


def test_index_most_frequent_num_min_occ_categories_eq_zero() -> None:
    cfg = Index(min_occ=4, na_strategy='most_frequent')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 1  # [zero]
    assert torch.allclose(encoder.emb.weight, torch.zeros(1, 2))

    # Test everything is mapped to zeros
    assert torch.allclose(
        encoder(torch.tensor([[-1], [0], [1], [2], [99]])),
        torch.zeros(5, 1, 2),
    )


def test_hash_separate_num_components_lt_num_categories() -> None:
    cfg = Hash(num_components=2, na_strategy='separate')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [
        ['A', 5],
        ['B', 4],
        ['C', 3],
        ['D', 2],
        ['E', 1],
    ]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 4  # ['A', 'B', separate, zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1], [2], [3], [4]])),
        encoder.emb.weight[[[[0], [1], [0], [1], [0]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[5], [99]])),
        encoder.emb.weight[[[[2], [2]]]],
    )

    # Test N/A is mapped to a separate category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[2]]]],
    )


def test_hash_separate_num_components_eq_num_categories() -> None:
    # num_components==num_categories case is rare and is equivalent to
    # `Index(min_occ=1, na_strategy='separate')` as all categories are within
    # the range of num_components.
    cfg = Hash(num_components=5, na_strategy='separate')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [
        ['A', 5],
        ['B', 4],
        ['C', 3],
        ['D', 2],
        ['E', 1],
    ]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 7  # ['A', 'B', 'C', 'D', 'E', separate, zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1], [2], [3], [4]])),
        encoder.emb.weight[[[[0], [1], [2], [3], [4]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[5], [6], [7]])),
        encoder.emb.weight[[[[5], [5], [5]]]],
    )

    # Test N/A is mapped to a separate category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[5]]]],
    )


def test_hash_most_frequent_num_components_lt_num_categories() -> None:
    cfg = Hash(num_components=2, na_strategy='most_frequent')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [
        ['A', 5],
        ['B', 4],
        ['C', 3],
        ['D', 2],
        ['E', 1],
    ]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 3  # ['A', 'B', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1], [2], [3], [4]])),
        encoder.emb.weight[[[[0], [1], [0], [1], [0]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[5], [99]])),
        encoder.emb.weight[[[[0], [0]]]],
    )

    # Test N/A is mapped to a separate category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[0]]]],
    )


def test_hash_most_frequent_num_components_eq_num_categories() -> None:
    # num_components==num_categories case is rare and is equivalent to
    # `Index(min_occ=1, na_strategy='most_frequent')` as all categories are
    # within the range of num_components.
    cfg = Hash(num_components=5, na_strategy='most_frequent')
    cfg._stats[ColStatType.CATEGORY_COUNTS] = [
        ['A', 5],
        ['B', 4],
        ['C', 3],
        ['D', 2],
        ['E', 1],
    ]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={'col0': cfg},
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 6  # ['A', 'B', 'C', 'D', 'E', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0], [1], [2], [3], [4]])),
        encoder.emb.weight[[[[0], [1], [2], [3], [4]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[5], [99]])),
        encoder.emb.weight[[[[0], [0]]]],
    )

    # Test N/A is mapped to a separate category
    assert torch.allclose(
        encoder(torch.tensor([[-1]])),
        encoder.emb.weight[[[[0]]]],
    )


def test_index_zero_multiple_cols() -> None:
    cfg0 = Index(na_strategy='zero')
    cfg0._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    cfg1 = Index(na_strategy='zero')
    cfg1._stats[ColStatType.CATEGORY_COUNTS] = [['D', 3], ['E', 2]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={
            'col0': cfg0,
            'col1': cfg1
        },
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 6  # ['A', 'B', 'C', 'D', 'E', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0, 0], [1, 1], [2, 0]])),
        encoder.emb.weight[[[[0, 3], [1, 4], [2, 3]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[3, 2], [99, 99]])),
        torch.zeros(2, 2, 2),
    )

    # Test N/A is mapped to a zero vector
    assert torch.allclose(
        encoder(torch.tensor([[-1, -1]])),
        torch.zeros(1, 2, 2),
    )


def test_index_separate_multiple_cols() -> None:
    cfg0 = Index(na_strategy='separate')
    cfg0._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    cfg1 = Index(na_strategy='separate')
    cfg1._stats[ColStatType.CATEGORY_COUNTS] = [['D', 3], ['E', 2]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={
            'col0': cfg0,
            'col1': cfg1
        },
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 8  # ['A', 'B', 'C', sep0, 'D', 'E', sep1, zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0, 0], [1, 1], [2, 0]])),
        encoder.emb.weight[[[[0, 4], [1, 5], [2, 4]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[3, 2], [99, 99]])),
        encoder.emb.weight[[[[3, 6], [3, 6]]]],
    )

    # Test N/A is mapped to a zero vector
    assert torch.allclose(
        encoder(torch.tensor([[-1, -1]])),
        encoder.emb.weight[[[[3, 6]]]],
    )


def test_index_most_frequent_multiple_cols() -> None:
    cfg0 = Index(na_strategy='most_frequent')
    cfg0._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    cfg1 = Index(na_strategy='most_frequent')
    cfg1._stats[ColStatType.CATEGORY_COUNTS] = [['D', 3], ['E', 2]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={
            'col0': cfg0,
            'col1': cfg1
        },
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 6  # ['A', 'B', 'C', 'D', 'E', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0, 0], [1, 1], [2, 0]])),
        encoder.emb.weight[[[[0, 3], [1, 4], [2, 3]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[3, 2], [99, 99]])),
        encoder.emb.weight[[[[0, 3], [0, 3]]]],
    )

    # Test N/A is mapped to the most frequent category
    assert torch.allclose(
        encoder(torch.tensor([[-1, -1]])),
        encoder.emb.weight[[[[0, 3]]]],
    )


def test_hash_separate_multiple_cols() -> None:
    cfg0 = Hash(num_components=2, na_strategy='separate')
    cfg0._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    cfg1 = Hash(num_components=2, na_strategy='separate')
    cfg1._stats[ColStatType.CATEGORY_COUNTS] = [['D', 3], ['E', 2], ['F', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={
            'col0': cfg0,
            'col1': cfg1
        },
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 7  # ['A|C', 'B', sep0, 'D|F', 'E', sep1, zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0, 0], [1, 1], [2, 0]])),
        encoder.emb.weight[[[[0, 3], [1, 4], [0, 3]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[3, 3], [99, 99]])),
        encoder.emb.weight[[[[2, 5], [2, 5]]]],
    )

    # Test N/A is mapped to a separate category
    assert torch.allclose(
        encoder(torch.tensor([[-1, -1]])),
        encoder.emb.weight[[[[2, 5]]]],
    )


def test_hash_most_frequent_multiple_cols() -> None:
    cfg0 = Hash(num_components=2, na_strategy='most_frequent')
    cfg0._stats[ColStatType.CATEGORY_COUNTS] = [['A', 3], ['B', 2], ['C', 1]]
    cfg1 = Hash(num_components=2, na_strategy='most_frequent')
    cfg1._stats[ColStatType.CATEGORY_COUNTS] = [['D', 3], ['E', 2], ['F', 1]]
    encoder = FusedCategoricalEncoder(
        col_encoder_dict={
            'col0': cfg0,
            'col1': cfg1
        },
        exclude_cols=[],
        out_channels=2,
        enc_mode='column',
    )

    # Test embedding matrix
    num_embeddings = encoder.emb.weight.size(0)
    assert num_embeddings == 5  # ['A|C', 'B', 'D|F', 'E', zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    # Test non-N/A are mapped to the corresponding vectors
    assert torch.allclose(
        encoder(torch.tensor([[0, 0], [1, 1], [2, 3]])),
        encoder.emb.weight[[[[0, 2], [1, 3], [0, 2]]]],
    )

    # Test unknown categories are treated as N/A gracefully
    assert torch.allclose(
        encoder(torch.tensor([[3, 3], [99, 99]])),
        encoder.emb.weight[[[[0, 2], [0, 2]]]],
    )

    # Test N/A is mapped to the most frequent category
    assert torch.allclose(
        encoder(torch.tensor([[-1, -1]])),
        encoder.emb.weight[[[[0, 2]]]],
    )
