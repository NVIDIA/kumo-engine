# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import numpy as np
import pytest
import torch
from kumoapi.encoder import (
    Encoder,
    MaxLogNumerical,
    MinLogNumerical,
    Numerical,
)
from kumoapi.typing import ColStatType
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.numerical import FusedNumericalEncoder


@withCUDA
def test_forward(device: torch.device) -> None:
    num_rows = 100
    num_cols = 13
    # Use the same values across cols so that we can use the same stats
    # for all cols to make this test case look simpler
    feat = torch.randn(num_rows, 1).repeat(1, num_cols)
    std, mean = map(float, torch.std_mean(feat[:, 0]))
    min_ = float(torch.min(feat[:, 0]))
    max_ = float(torch.max(feat[:, 0]))
    quantiles = list(map(float, np.quantile(feat[:, 0], [0.25, 0.5, 0.75])))
    nan_mask = torch.randint(0, 2, size=(num_rows, num_cols), dtype=torch.bool)
    feat[nan_mask] = float('nan')
    feat = feat.to(device)
    col_encoder_dict = {
        # Test Numerical with different na strategies:
        'num_na_zero': Numerical(na_strategy="zero"),
        'num_na_mean': Numerical(na_strategy="mean"),
        # Test Numerical with different scalers:
        'num_scaler_None': Numerical(scaler=None, na_strategy="zero"),
        'num_scaler_standard': Numerical(scaler="standard",
                                         na_strategy="zero"),
        'num_scaler_minmax': Numerical(scaler="minmax", na_strategy="zero"),
        'num_scaler_robust': Numerical(scaler="robust", na_strategy="zero"),
        # Test MaxLogNumerical with different na strategies:
        'maxlog_na_zero': MaxLogNumerical(na_strategy="zero"),
        'maxlog_na_mean': MaxLogNumerical(na_strategy="mean"),
        # Test MinLogNumerical with different na strategies:
        'minlog_na_zero': MinLogNumerical(na_strategy="zero"),
        'minlog_na_mean': MinLogNumerical(na_strategy="mean"),
        'separate': Numerical(scaler="standard", na_strategy="separate"),
        # Test exclude_cols:
        'exclude1': Numerical(na_strategy="zero"),
        'exclude2': Numerical(na_strategy="separate"),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.MEAN: mean,
            ColStatType.STD: std,
            ColStatType.MIN: min_,
            ColStatType.MAX: max_,
            ColStatType.QUANTILES: quantiles,
        }

    fused_encoder = FusedNumericalEncoder(
        col_encoder_dict=col_encoder_dict,
        out_channels=5,
        enc_mode="fuse",  # unused for this test case
        exclude_cols=['exclude1', 'exclude2'],
        device=device,
    )
    assert fused_encoder.num_cols == num_cols  # including exclude_cols
    assert fused_encoder.has_log_encoder is True
    assert torch.allclose(
        fused_encoder.offsets,
        torch.tensor(
            [
                0.,  # no scaler
                0.,  # no scaler
                0.,  # no scaler
                mean,  # standard scaler
                min_,  # minmax scaler
                quantiles[1],  # robust scaler
                min_ - 1.,  # maxlog should always have min_ - 1
                min_ - 1.,  # maxlog should always have min_ - 1
                max_ + 1.,  # minlog should always have max_ + 1
                max_ + 1.,  # minlog should always have max_ + 1
                mean,  # standard scaler
                0.,  # no scaler
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
                std,  # standard scaler
                max_ - min_,  # minmax scaler
                quantiles[2] - quantiles[0],  # robust scaler
                1.,  # maxlog should always have 1
                1.,  # maxlog should always have 1
                -1.,  # minlog should always have -1
                -1.,  # minlog should always have -1
                std,  # standard scaler
                1.,  # no scaler
                1.,  # no scaler
            ],
            device=device),
    )
    assert torch.allclose(
        fused_encoder.impute_values,
        torch.tensor(
            [
                0.,  # NAStrategy.ZERO
                mean,  # NAStrategy.MEAN
                0.,  # NAStrategy.ZERO
                0.,  # NAStrategy.ZERO
                0.,  # NAStrategy.ZERO
                0.,  # NAStrategy.ZERO
                0.,  # NAStrategy.ZERO
                mean,  # NAStrategy.MEAN
                0.,  # NAStrategy.ZERO
                mean,  # NAStrategy.MEAN
                float('NaN'),  # NAStrategy.SEPARATE
                0.,  # NAStrategy.ZERO
                float('NaN'),  # NAStrategy.SEPARATE
            ],
            device=device,
        ),
        equal_nan=True,
    )
    assert torch.allclose(
        fused_encoder.log_mask,
        torch.tensor(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                True,  # maxlog
                True,  # maxlog
                True,  # minlog
                True,  # minlog
                False,
                False,
                False,
            ],
            device=device),
    )
    assert torch.allclose(
        fused_encoder.exclude_cols_mask,
        torch.tensor(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,  # exclude_cols
                True,  # exclude_cols
            ],
            device=device),
    )
    assert torch.allclose(
        fused_encoder.separate_na_mask,
        torch.tensor([
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ], device=device),
    )


@withCUDA
@pytest.mark.parametrize("enc_mode", ["fuse", "column"])
def test_empty_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    encoder = FusedNumericalEncoder(
        col_encoder_dict={
            'num0': Numerical(na_strategy="zero"),
            'num1': Numerical(na_strategy="zero"),
            'num2': Numerical(na_strategy="zero"),
        },
        exclude_cols=[],
        out_channels=5,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(torch.empty(0, 3, device=device))
    assert out.size() == (0, 5) if enc_mode == "fuse" else (0, 3, 5)


@withCUDA
@pytest.mark.parametrize("enc_mode", ["fuse", "column"])
def test_all_na_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    cfg = Numerical()
    cfg._stats = {ColStatType.MEAN: 0.0}
    encoder = FusedNumericalEncoder(
        col_encoder_dict={
            'col0': cfg,
            'col1': cfg,
        },
        exclude_cols=[],
        out_channels=3,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(torch.tensor([[float("nan"), float("nan")]], device=device))
    assert out.size() == (1, 3) if enc_mode == "fuse" else (1, 2, 3)


@withCUDA
@pytest.mark.parametrize("enc_mode", ["fuse", "column"])
def test_all_same_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    cfg = Numerical(scaler='standard')
    cfg._stats = {ColStatType.MEAN: 0.0, ColStatType.STD: None}
    encoder = FusedNumericalEncoder(
        col_encoder_dict={
            'col0': cfg,
            'col1': cfg,
        },
        exclude_cols=[],
        out_channels=3,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(torch.tensor([[1.0, 1.0]], device=device))
    assert out.size() == (1, 3) if enc_mode == "fuse" else (1, 2, 3)


@withCUDA
def test_enc_mode(device: torch.device) -> None:
    """Test the head linear layer of FusedNumericalEncoder."""
    num_rows = 5
    num_cols = 3
    feat = torch.randn(num_rows, num_cols, device=device)
    nan_mask = torch.randint(0, 2, feat.size(), dtype=torch.bool,
                             device=device)
    feat[nan_mask] = float('nan')
    col_encoder_dict: dict[str, Encoder] = {
        'num0': Numerical(na_strategy="zero"),
        'num1': Numerical(na_strategy="zero"),
        'num2': Numerical(na_strategy="separate"),
    }
    out_channels = 3
    # ===============
    # enc_mode="fuse"
    # ===============
    fused_encoder = FusedNumericalEncoder(
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
    fused_encoder = FusedNumericalEncoder(
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
        FusedNumericalEncoder(
            col_encoder_dict=col_encoder_dict,
            exclude_cols=[],
            out_channels=out_channels,
            enc_mode="invalid",
        )


def test_inconsistent_col_order() -> None:
    """Test FusedNumericalEncoder raises error if the column order is different
    between in TF and in the fused encoder.
    """
    num_rows = 5
    num_cols = 2
    feat = torch.randn(num_rows, num_cols)
    nan_mask = torch.randint(0, 2, size=(num_rows, num_cols), dtype=torch.bool)
    feat[nan_mask] = float('nan')
    col_encoder_dict: dict[str, Encoder] = {
        'num0': Numerical(na_strategy="zero"),
        'num1': Numerical(na_strategy="zero"),
    }
    fused_encoder = FusedNumericalEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=1,
        enc_mode="fuse",
    )
    valid_col_names = ["num0", "num1"]
    fused_encoder(feat, valid_col_names)
    invalid_col_names = ["num1", "num0"]
    with pytest.raises(ValueError, match="The order of columns"):
        fused_encoder(feat, invalid_col_names)


@withCUDA
def test_gradient(device: torch.device) -> None:
    encoder = FusedNumericalEncoder(
        col_encoder_dict={
            '0': Numerical(na_strategy='zero'),
            '1': Numerical(na_strategy='zero'),
            '2': Numerical(na_strategy='separate'),
            '3': Numerical(na_strategy='separate'),
        },
        exclude_cols=['1', '3'],
        out_channels=16,
        enc_mode="fuse",
        device=device,
    )

    feat = torch.randn(10, 4, device=device)
    nan_mask = torch.randint(0, 2, feat.size(), dtype=torch.bool,
                             device=device)
    feat[nan_mask] = float('nan')

    with modeling_context.explaining():
        out = encoder(feat)

    with pytest.raises(ValueError, match="Gradients not yet computed"):
        encoder.get_grad_wrt_input()

    out.backward(torch.randn_like(out))

    grad = encoder.get_grad_wrt_input()
    assert isinstance(grad, torch.Tensor)
    assert grad.size() == (10, 4)
    assert not torch.allclose(grad[:, 0], torch.zeros(10, device=device))
    assert torch.allclose(grad[:, 1], torch.zeros(10, device=device))
    assert not torch.allclose(grad[:, 2], torch.zeros(10, device=device))
    assert torch.allclose(grad[:, 3], torch.zeros(10, device=device))
    assert not hasattr(encoder, 'saved_tensors')

    with pytest.raises(ValueError, match="Did not store required data"):
        encoder.get_grad_wrt_input()


@withCUDA
def test_clamp(device: torch.device) -> None:
    value = torch.randn(10, 1, device=device)

    encoder = FusedNumericalEncoder(
        col_encoder_dict={
            '0': Numerical(na_strategy='zero'),
        },
        exclude_cols=[],
        out_channels=16,
        enc_mode="fuse",
        device=device,
    )

    value[0] = 1000.0
    value[1] = -1000.0
    out1 = encoder(value)

    value[0] = 1e28
    value[1] = -1e28
    out2 = encoder(value)

    assert torch.allclose(out1, out2)
