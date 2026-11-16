# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import pytest
import torch
from kumoapi.encoder import Encoder, MultiCategorical
from kumoapi.typing import ColStatType
from torch_frame.data import MultiNestedTensor
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.multicategorical import FusedMultiCategoricalEncoder
from kumo_core.encoders.fused.multicategorical import \
    _get_num_min_occ_categories  # isort: skip


def get_fake_multi_nested_tensor(
        device: torch.device) -> tuple[MultiNestedTensor, dict[str, Encoder]]:
    # 3 categories:
    feat_multicat0 = [
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1, 2]),
    ]
    # 4 categories:
    feat_multicat1 = [
        torch.tensor([0]),
        torch.tensor([1, 2]),
        torch.tensor([2, 3]),
        torch.tensor([3]),
    ]
    # 2 categories:
    feat_multicat2 = [
        torch.tensor([0]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([2]),  # nan
    ]
    # 1 category:
    feat_exclude = [
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
    ]
    mnt = MultiNestedTensor.from_tensor_mat([[
        feat_multicat0[row],
        feat_multicat1[row],
        feat_multicat2[row],
        feat_exclude[row],
    ] for row in range(len(feat_multicat0))])
    # Test ZERO with different min_occ
    multicat0 = MultiCategorical(na_strategy='zero', min_occ=1)
    multicat0._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [
        ['A', 6],
        ['B', 3],
        ['C', 1],
    ]
    # Test MOST_FREQUENT with different min_occ
    multicat1 = MultiCategorical(na_strategy='most_frequent', min_occ=2)
    # 'C' and 'D' will be treated as na:
    multicat1._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [
        ['A', 4],
        ['B', 4],
        ['C', 1],
        ['D', 1],
    ]
    multicat2 = MultiCategorical(na_strategy='most_frequent', min_occ=3)
    # 'B' will be treated as na:
    multicat2._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 8], ['B', 2]]
    exclude = MultiCategorical(na_strategy='zero')
    exclude._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 10]]
    col_encoder_dict: dict[str, Encoder] = {
        'multicat0': multicat0,  # 3 cat
        'multicat1': multicat1,  # 2 cat (2 others don't satisfy min_occ)
        'multicat2': multicat2,  # 1 cat (another doesn't satisfy min_occ)
        'exclude': exclude,  # 1 cat
    }
    return mnt.to(device), col_encoder_dict


@withCUDA
def test_forward(device: torch.device) -> None:
    out_channels = 3
    feat, col_encoder_dict = get_fake_multi_nested_tensor(device)
    fused_encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=['exclude'],
        out_channels=out_channels,
        enc_mode='fuse',
        device=device,
    )
    assert torch.equal(
        fused_encoder.exclude_cols_mask,
        torch.tensor(
            [
                False,
                False,
                False,
                True,  # exclude_cols
            ],
            device=device,
        ),
    )
    assert torch.equal(
        fused_encoder.impute_values,
        torch.tensor(
            [
                7,  # NAStrategy.ZERO
                0,  # NAStrategy.MOST_FREQUENT
                0,  # NAStrategy.MOST_FREQUENT
                1,  # exclude_cols
            ],
            device=device,
        ))
    assert torch.equal(
        fused_encoder.na_thresholds,
        torch.tensor([
            _get_num_min_occ_categories(col_encoder_dict['multicat0']),
            _get_num_min_occ_categories(col_encoder_dict['multicat1']),
            _get_num_min_occ_categories(col_encoder_dict['multicat2']),
            _get_num_min_occ_categories(col_encoder_dict['exclude']),
        ], device=device),
    )
    num_embeddings = 7 + 1  # num min_occ categories + 1 for zero vector
    assert fused_encoder.emb.weight.size() == (num_embeddings, out_channels)
    out = fused_encoder(feat)
    assert out.size() == (feat.num_rows, out_channels)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_empty_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    feat = MultiNestedTensor(
        num_rows=0,
        num_cols=3,
        values=torch.tensor([], dtype=torch.long),
        offset=torch.tensor([0]),
    ).to(device)
    cfg = MultiCategorical(min_occ=1)
    cfg._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 6], ['B', 3]]
    encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict={
            'col0': cfg,
            'col1': cfg,
            'col2': cfg,
        },
        exclude_cols=[],
        out_channels=5,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(feat)
    assert out.size() == (0, 5) if enc_mode == 'fuse' else (0, 3, 5)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_all_na_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    # The value `1` means N/A. See MultiCategoricalTensorMapper for details.
    feat = MultiNestedTensor.from_tensor_mat([
        [torch.tensor([1]), torch.tensor([1])],
    ]).to(device)
    cfg = MultiCategorical(min_occ=1)
    cfg._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 6]]
    encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict={
            'col0': cfg,
            'col1': cfg,
        },
        exclude_cols=[],
        out_channels=3,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(feat)
    assert out.size() == (1, 3) if enc_mode == 'fuse' else (1, 2, 3)


@withCUDA
def test_enc_mode(device: torch.device) -> None:
    feat, col_encoder_dict = get_fake_multi_nested_tensor(device)
    out_channels = 3
    # ===============
    # enc_mode='fuse'
    # ===============
    fused_encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='fuse',
        device=device,
    )
    out = fused_encoder(feat)
    assert out.size() == (feat.num_rows, out_channels)
    # =================
    # enc_mode='column'
    # =================
    fused_encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='column',
        device=device,
    )
    out = fused_encoder(feat)
    assert out.size() == (feat.num_rows, feat.num_cols, out_channels)
    # ================
    # invalid enc_mode
    # ================
    with pytest.raises(ValueError, match="Unsupported encoder mode 'invalid'"):
        FusedMultiCategoricalEncoder(
            col_encoder_dict=col_encoder_dict,
            exclude_cols=[],
            out_channels=out_channels,
            enc_mode='invalid',
        )


def test_inconsistent_col_order() -> None:
    """Test FusedMultiCategoricalEncoder raises error if the column order is
    different between in TF and in the fused encoder.
    """
    out_channels = 3
    feat, col_encoder_dict = get_fake_multi_nested_tensor(torch.device('cpu'))
    fused_encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='fuse',
    )
    valid_col_names = ['multicat0', 'multicat1', 'multicat2', 'exclude']
    fused_encoder(feat, valid_col_names)  # raises no error
    invalid_col_names = ['exclude', 'multicat0', 'multicat1', 'multicat2']
    with pytest.raises(ValueError, match='The order of columns'):
        fused_encoder(feat, invalid_col_names)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_gradient(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    feat, col_encoder_dict = get_fake_multi_nested_tensor(device)
    encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=['exclude'],
        out_channels=16,
        enc_mode=enc_mode,
        device=device,
    )

    with modeling_context.explaining():
        out = encoder(feat)

    out.backward(torch.randn_like(out))

    grad = encoder.get_grad_wrt_input()
    assert isinstance(grad, torch.Tensor)
    assert grad.size() == (4, 4)
    assert torch.allclose(grad[:, 3], torch.zeros(4, device=device))
    assert not hasattr(encoder, 'saved_tensors')


@withCUDA
def test_gradient_with_empty_entries(device: torch.device) -> None:
    values = torch.tensor([0, 1, 1, 0])
    offset = torch.tensor([0, 2, 2, 3, 4, 4])
    feat = MultiNestedTensor(5, 1, values, offset).to(device)
    cfg = MultiCategorical(na_strategy='zero', min_occ=1)
    cfg._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [[0, 2], [1, 2]]
    encoder = FusedMultiCategoricalEncoder(
        col_encoder_dict={'multicat0': cfg},
        exclude_cols=[],
        out_channels=16,
        enc_mode='fuse',
        device=device,
    )

    with modeling_context.explaining():
        out = encoder(feat)

    out.backward(torch.randn_like(out))

    grad = encoder.get_grad_wrt_input()
    assert isinstance(grad, torch.Tensor)
    assert grad.size() == (5, 1)
    assert torch.allclose(grad[1], torch.zeros(1, device=device))
    assert torch.allclose(grad[4], torch.zeros(1, device=device))
    assert not hasattr(encoder, 'saved_tensors')


def test_zero_multiple_cols() -> None:
    cfg0 = MultiCategorical(na_strategy='zero')
    cfg0._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 3], ['B', 2],
                                                      ['C', 1]]
    cfg1 = MultiCategorical(na_strategy='zero')
    cfg1._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['D', 3], ['E', 2]]
    encoder = FusedMultiCategoricalEncoder(
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

    weight = encoder.emb.weight

    # Test non-N/A are mapped to the corresponding vectors
    tensor_mat = [[torch.tensor([0, 1, 2]), torch.tensor([0, 1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[0, 1, 2]].sum(dim=0))
    assert out[:, 1, :].allclose(weight[[3, 4]].sum(dim=0))

    # Test unknown categories are treated as N/A gracefully
    tensor_mat = [
        [torch.tensor([3]), torch.tensor([2])],
        [torch.tensor([99]), torch.tensor([99])],
    ]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out.allclose(torch.zeros(2, 2, 2))

    # Test N/A is mapped to a zero vector
    tensor_mat = [[torch.tensor([-1]), torch.tensor([-1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out.allclose(torch.zeros(2, 2, 2))


def test_most_frequent_multiple_cols() -> None:
    cfg0 = MultiCategorical(na_strategy='most_frequent')
    cfg0._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 3], ['B', 2],
                                                      ['C', 1]]
    cfg1 = MultiCategorical(na_strategy='most_frequent')
    cfg1._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['D', 3], ['E', 2]]
    encoder = FusedMultiCategoricalEncoder(
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

    weight = encoder.emb.weight

    # Test non-N/A are mapped to the corresponding vectors
    tensor_mat = [[torch.tensor([0, 1, 2]), torch.tensor([0, 1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[0, 1, 2]].sum(dim=0))
    assert out[:, 1, :].allclose(weight[[3, 4]].sum(dim=0))

    # Test unknown categories are treated as N/A gracefully
    tensor_mat = [
        [torch.tensor([3]), torch.tensor([2])],
        [torch.tensor([99]), torch.tensor([99])],
    ]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[0, 0]])
    assert out[:, 1, :].allclose(weight[[3, 3]])

    # Test N/A is mapped to the most frequent category
    tensor_mat = [[torch.tensor([-1]), torch.tensor([-1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[0]])
    assert out[:, 1, :].allclose(weight[[3]])


def test_separate_multiple_cols() -> None:
    cfg0 = MultiCategorical(na_strategy='separate')
    cfg0._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 3], ['B', 2],
                                                      ['C', 1]]
    cfg1 = MultiCategorical(na_strategy='separate')
    cfg1._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['D', 3], ['E', 2]]
    encoder = FusedMultiCategoricalEncoder(
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

    weight = encoder.emb.weight

    # Test non-N/A are mapped to the corresponding vectors
    tensor_mat = [[torch.tensor([0, 1, 2]), torch.tensor([0, 1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[0, 1, 2]].sum(dim=0))
    assert out[:, 1, :].allclose(weight[[4, 5]].sum(dim=0))

    # Test unknown categories are treated as N/A gracefully
    tensor_mat = [
        [torch.tensor([3]), torch.tensor([2])],
        [torch.tensor([99]), torch.tensor([99])],
    ]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[3, 3]])
    assert out[:, 1, :].allclose(weight[[6, 6]])

    # Test N/A is mapped to a new category
    tensor_mat = [[torch.tensor([-1]), torch.tensor([-1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[3])
    assert out[:, 1, :].allclose(weight[6])


def test_min_occ_multiple_cols() -> None:
    cfg0 = MultiCategorical(na_strategy='separate', min_occ=2)
    cfg0._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['A', 3], ['B', 2],
                                                      ['C', 1]]
    cfg1 = MultiCategorical(na_strategy='separate', min_occ=3)
    cfg1._stats[ColStatType.MULTI_CATEGORY_COUNTS] = [['D', 3], ['E', 2]]
    encoder = FusedMultiCategoricalEncoder(
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
    assert num_embeddings == 6  # ['A', 'B', sep0, 'D', sep1, zero]
    for i in range(num_embeddings - 1):
        assert not torch.allclose(encoder.emb.weight[i], torch.zeros(2))
    assert torch.allclose(encoder.emb.weight[-1], torch.zeros(2))

    weight = encoder.emb.weight

    # Test non-N/A are mapped to the corresponding vectors
    tensor_mat = [[torch.tensor([0, 1, 2]), torch.tensor([0, 1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[0, 1, 2]].sum(dim=0))
    assert out[:, 1, :].allclose(weight[[3, 4]].sum(dim=0))

    # Test unknown categories are treated as N/A gracefully
    tensor_mat = [
        [torch.tensor([2]), torch.tensor([1])],
        [torch.tensor([99]), torch.tensor([99])],
    ]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[[2, 2]])
    assert out[:, 1, :].allclose(weight[[4, 4]])

    # Test N/A is mapped to a new category
    tensor_mat = [[torch.tensor([-1]), torch.tensor([-1])]]
    out = encoder(MultiNestedTensor.from_tensor_mat(tensor_mat))
    assert out[:, 0, :].allclose(weight[2])
    assert out[:, 1, :].allclose(weight[4])
