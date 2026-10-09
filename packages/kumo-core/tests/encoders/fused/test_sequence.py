# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import pytest
import torch
from kumoapi.encoder import Encoder, NumericalList
from kumoapi.typing import ColStatType
from torch_frame.data import MultiEmbeddingTensor
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.sequence import FusedSequenceEncoder


@withCUDA
def test_forward(device: torch.device) -> None:
    tensor_list = [
        torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]),  # 'seq0'
        torch.tensor([[0.6, float("nan")], [0.7, float("inf")]]),  # 'seq1'
        torch.tensor([[1.0], [1.1]]),  # 'exclude'
    ]
    feat = MultiEmbeddingTensor.from_tensor_list(tensor_list).to(device)
    col_encoder_dict: dict[str, Encoder] = {
        # Test different column dimensions:
        'seq0': NumericalList(),
        'seq1': NumericalList(),
        # Test exclude_cols:
        'exclude': NumericalList(),
    }
    col_encoder_dict['seq0']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 3,
        ColStatType.SEQUENCE_MIN_LENGTH: 3,
    }
    col_encoder_dict['seq1']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 2,
        ColStatType.SEQUENCE_MIN_LENGTH: 2,
    }
    col_encoder_dict['exclude']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 1,
        ColStatType.SEQUENCE_MIN_LENGTH: 1,
    }
    out_channels = 5
    fused_encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        out_channels=out_channels,
        enc_mode="fuse",
        exclude_cols=['exclude'],
        device=device,
    )
    assert fused_encoder.num_cols == feat.num_cols  # including exclude_cols
    assert fused_encoder.col_dims == [3, 2, 1]
    assert torch.allclose(
        fused_encoder.impute_values,
        # currently we only support 'zero' na strategy
        torch.zeros(3 + 2 + 1, device=device),
    )
    assert torch.allclose(
        fused_encoder.exclude_cols_mask,
        torch.tensor([False, False, False, False, False, True], device=device),
    )
    out = fused_encoder.forward(feat)
    assert out.size() == (feat.num_rows, out_channels)
    assert bool(out.isnan().any()) is False


@withCUDA
def test_forward_with_scaler(device: torch.device) -> None:
    tensor_list = [
        torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]),  # 'seq0'
        torch.tensor([[0.6, float("nan")], [0.1, float("inf")]]),  # 'seq1'
        torch.tensor([[0.3, 0.5], [0.2, 0.4]]),  # 'seq2'
        torch.tensor([[1.0], [1.1]]),  # 'exclude'
    ]
    feat = MultiEmbeddingTensor.from_tensor_list(tensor_list).to(device)
    col_encoder_dict: dict[str, Encoder] = {
        # Test different column dimensions:
        'seq0': NumericalList(),
        'seq1': NumericalList(scaler="standard"),
        'seq2': NumericalList(scaler="cosine"),
        # Test exclude_cols:
        'exclude': NumericalList(),
    }
    col_encoder_dict['seq0']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 3,
        ColStatType.SEQUENCE_MIN_LENGTH: 3,
    }
    seq1 = torch.nan_to_num(tensor_list[1], nan=0.0, posinf=0.0)
    mean_seq1 = seq1.mean().item()
    std_seq1 = seq1.std().item()
    col_encoder_dict['seq1']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 2,
        ColStatType.SEQUENCE_MIN_LENGTH: 2,
        ColStatType.SEQUENCE_MEAN: mean_seq1,
        ColStatType.SEQUENCE_STD: std_seq1,
    }
    col_encoder_dict['seq2']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 2,
        ColStatType.SEQUENCE_MIN_LENGTH: 2,
    }
    col_encoder_dict['exclude']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 1,
        ColStatType.SEQUENCE_MIN_LENGTH: 1,
    }
    out_channels = 5
    fused_encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        out_channels=out_channels,
        enc_mode="fuse",
        exclude_cols=['exclude'],
        device=device,
    )
    assert fused_encoder.num_cols == feat.num_cols  # including exclude_cols
    assert fused_encoder.col_dims == [3, 2, 2, 1]
    assert torch.allclose(
        fused_encoder.impute_values,
        # currently we only support 'zero' na strategy
        torch.zeros(3 + 2 + 2 + 1, device=device),
    )
    assert torch.allclose(
        fused_encoder.exclude_cols_mask,
        torch.tensor([False] * (3 + 2 + 2) + [True], device=device),
    )
    out = fused_encoder.forward(feat)
    assert out.size() == (feat.num_rows, out_channels)
    assert bool(out.isnan().any()) is False
    assert torch.allclose(
        fused_encoder.offsets,
        torch.tensor(
            [
                0.,  # no scaler
                0.,  # no scaler
                0.,  # no scaler
                mean_seq1,  # mean for standard scaler
                mean_seq1,  # mean for standard scaler
                0.,  # cosine scaler
                0.,  # cosine scaler
                0.,  # no scaler
            ],
            device=device),
    )
    assert torch.allclose(
        fused_encoder.scales,
        torch.tensor(
            [
                1.,  # no scaler
                1.,  # no scaler
                1.,  # no scaler
                std_seq1,  # std for standard scaler
                std_seq1,  # std for standard scaler
                1.,  # unused norm for cosine scaler
                1.,  # unused norm for cosine scaler
                1.,  # no scaler
            ],
            device=device),
    )


@withCUDA
@pytest.mark.parametrize("enc_mode", ["fuse", "column"])
def test_empty_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    feat = MultiEmbeddingTensor(
        num_rows=0,
        num_cols=3,
        values=torch.tensor([]),
        offset=torch.tensor([0, 1, 2, 3]),
    ).to(device)
    col_encoder_dict: dict[str, Encoder] = {
        'seq0': NumericalList(),
        'seq1': NumericalList(),
        'seq2': NumericalList(),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.SEQUENCE_MAX_LENGTH: 1,
            ColStatType.SEQUENCE_MIN_LENGTH: 1,
        }
    encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=5,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(feat)
    assert out.size() == (0, 5) if enc_mode == "fuse" else (0, 3, 5)


@withCUDA
@pytest.mark.parametrize("enc_mode", ["fuse", "column"])
def test_all_na_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    feat = MultiEmbeddingTensor(
        num_rows=1,
        num_cols=2,
        values=torch.tensor([[float("nan"), float("nan")]]),
        offset=torch.tensor([0, 1, 2]),
    ).to(device)
    col_encoder_dict: dict[str, Encoder] = {
        'col0': NumericalList(),
        'col1': NumericalList(),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.SEQUENCE_MAX_LENGTH: 1,
            ColStatType.SEQUENCE_MIN_LENGTH: 1,
        }
    encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=3,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(feat)
    assert out.size() == (1, 3) if enc_mode == "fuse" else (1, 2, 3)


@withCUDA
def test_enc_mode(device: torch.device) -> None:
    num_rows = 5
    col_dims = [3, 2, 1]
    num_cols = len(col_dims)
    values = torch.randn(num_rows, sum(col_dims), device=device)
    values_nan_mask = torch.randint(0, 2, size=(num_rows, sum(col_dims)),
                                    dtype=torch.bool, device=device)
    values[values_nan_mask] = float('nan')
    offset = torch.tensor([0, 3, 5, 6], device=device)
    feat = MultiEmbeddingTensor(num_rows, num_cols, values, offset)
    col_encoder_dict: dict[str, Encoder] = {
        'seq0': NumericalList(),
        'seq1': NumericalList(),
        'seq2': NumericalList(),
    }
    col_encoder_dict['seq0']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 3,
        ColStatType.SEQUENCE_MIN_LENGTH: 3,
    }
    col_encoder_dict['seq1']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 2,
        ColStatType.SEQUENCE_MIN_LENGTH: 2,
    }
    col_encoder_dict['seq2']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 1,
        ColStatType.SEQUENCE_MIN_LENGTH: 1,
    }
    out_channels = 3
    # ===============
    # enc_mode="fuse"
    # ===============
    fused_encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode="fuse",
        device=device,
    )
    out = fused_encoder.forward(feat)
    assert out.size() == (num_rows, out_channels)
    # =================
    # enc_mode="column"
    # =================
    fused_encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode="column",
        device=device,
    )
    out = fused_encoder.forward(feat)
    assert out.size() == (num_rows, num_cols, out_channels)
    # ================
    # invalid enc_mode
    # ================
    with pytest.raises(ValueError, match="Unsupported encoder mode 'invalid'"):
        FusedSequenceEncoder(
            col_encoder_dict=col_encoder_dict,
            exclude_cols=[],
            out_channels=out_channels,
            enc_mode="invalid",
        )


def test_inconsistent_col_order() -> None:
    num_rows = 7
    col_dims = [5, 3]
    num_cols = len(col_dims)
    values = torch.randn(num_rows, sum(col_dims))
    values_nan_mask = torch.randint(0, 2, size=(num_rows, sum(col_dims)),
                                    dtype=torch.bool)
    values[values_nan_mask] = float('nan')
    offset = torch.tensor([0, 5, 8])
    feat = MultiEmbeddingTensor(num_rows, num_cols, values, offset)
    col_encoder_dict: dict[str, Encoder] = {
        'seq0': NumericalList(),
        'seq1': NumericalList(),
    }
    col_encoder_dict['seq0']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 5,
        ColStatType.SEQUENCE_MIN_LENGTH: 5,
    }
    col_encoder_dict['seq1']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 3,
        ColStatType.SEQUENCE_MIN_LENGTH: 3,
    }
    fused_encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=11,
        enc_mode="fuse",
    )
    valid_col_names = ["seq0", "seq1"]
    fused_encoder(feat, valid_col_names)
    invalid_col_names = ["seq1", "seq0"]
    with pytest.raises(ValueError, match="The order of columns"):
        fused_encoder(feat, invalid_col_names)


@withCUDA
def test_gradient(device: torch.device) -> None:
    col_encoder_dict: dict[str, Encoder] = {
        'seq0': NumericalList(),
        'seq1': NumericalList(),
    }
    col_encoder_dict['seq0']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 5,
        ColStatType.SEQUENCE_MIN_LENGTH: 5,
    }
    col_encoder_dict['seq1']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 3,
        ColStatType.SEQUENCE_MIN_LENGTH: 3,
    }
    encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=['seq1'],
        out_channels=16,
        enc_mode="fuse",
        device=device,
    )

    with modeling_context.explaining():
        feat = MultiEmbeddingTensor(
            num_rows=10,
            num_cols=2,
            values=torch.randn(10, 8),
            offset=torch.tensor([0, 5, 8]),
        ).to(device)
        out = encoder(feat)

    out.backward(torch.randn_like(out))

    grad = encoder.get_grad_wrt_input()
    assert isinstance(grad, torch.Tensor)
    assert grad.size() == (10, 2)
    assert not torch.allclose(grad[:, 0], torch.zeros(10, device=device))
    assert torch.allclose(grad[:, 1], torch.zeros(10, device=device))
    assert not hasattr(encoder, 'saved_tensors')


@withCUDA
def test_cosine(device: torch.device) -> None:
    col_encoder_dict: dict[str, Encoder] = {
        'seq0': NumericalList(scaler='cosine'),
        'seq1': NumericalList(scaler='cosine'),
    }
    col_encoder_dict['seq0']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 5,
        ColStatType.SEQUENCE_MIN_LENGTH: 5,
    }
    col_encoder_dict['seq1']._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: 3,
        ColStatType.SEQUENCE_MIN_LENGTH: 3,
    }
    encoder = FusedSequenceEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=['seq1'],
        out_channels=16,
        enc_mode="fuse",
        device=device,
    )

    feat = MultiEmbeddingTensor(
        num_rows=10,
        num_cols=2,
        values=torch.randn(10, 8),
        offset=torch.tensor([0, 5, 8]),
    ).to(device)

    out = encoder(feat)
    assert out.size() == (10, 16)
