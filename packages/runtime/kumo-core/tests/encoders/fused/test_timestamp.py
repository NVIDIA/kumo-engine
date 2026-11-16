# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Literal

import pytest
import torch
from kumoapi.encoder import Datetime, Encoder
from kumoapi.typing import ColStatType
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumo_core.encoders.fused.timestamp import FusedTimestampEncoder
from kumo_core.types.features import TIMESTAMP_PREFIX


def get_fake_timestamp_feat(
    num_rows: int,
    num_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feat = torch.stack(
        [
            torch.randint(0, 60, (num_rows, num_cols)),  # minute
            torch.randint(0, 24, (num_rows, num_cols)),  # hour
            torch.randint(1990, 2023, (num_rows, num_cols)),  # year
            torch.randint(0, 7, (num_rows, num_cols)),  # day_of_week
            torch.randint(0, 31, (num_rows, num_cols)),  # day_of_month
            torch.randint(0, 365, (num_rows, num_cols)),  # day_of_year
            torch.randint(28, 31, (num_rows, num_cols)),  # days_per_month
            torch.randint(365, 366, (num_rows, num_cols)),  # days_per_year
            torch.randint(52, 53, (num_rows, num_cols)),  # weeks_per_year
            torch.randint(0, 12, (num_rows, num_cols)),  # month
            torch.randint(0, 53, (num_rows, num_cols)),  # week_of_year
        ],
        dim=-1)  # [num_rows, num_cols, 11]

    # 10% of values are NaN
    num_na_values = math.ceil(0.1 * num_rows * num_cols)
    na_rows = torch.randperm(num_rows)[:num_na_values]
    na_cols = torch.randperm(num_cols)[:num_na_values]
    feat[na_rows, na_cols, :] = -1
    return feat, na_rows, na_cols


@withCUDA
def test_forward(device: torch.device) -> None:
    num_rows = 10
    num_cols = 3
    feat, _, _ = get_fake_timestamp_feat(num_rows, num_cols)
    feat = feat.to(device)
    col_encoder_dict: dict[str, Encoder] = {
        # Test different include_* flags are respected:
        'time0':
        Datetime(
            include_minute=True,
            include_hour=True,
            include_year=True,
            include_day_of_week=True,
            include_day_of_month=True,
            include_day_of_year=True,
            num_year_periods=3,
        ),
        'time1':
        Datetime(
            include_minute=False,
            include_hour=False,
            include_year=False,
            include_day_of_week=False,
            include_day_of_month=False,
            include_day_of_year=False,
            num_year_periods=2,
        ),
        # Test exclude_cols:
        'exclude':
        Datetime(num_year_periods=1),
    }
    for col_name, encoder in col_encoder_dict.items():
        if col_name != 'time1':
            encoder._stats = {
                ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
                ColStatType.MAX: f'{TIMESTAMP_PREFIX}2025-01-01',
            }

    out_channels = 5
    fused_encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        out_channels=out_channels,
        enc_mode="fuse",  # not used for this test
        exclude_cols=['exclude'],
        device=device,
    )
    assert fused_encoder.num_cols == num_cols  # including exclude_cols
    assert fused_encoder.max_num_year_periods == 3
    assert torch.equal(
        fused_encoder.exclude_cols_mask,
        torch.tensor(
            [
                False,
                False,
                True,  # exclude_cols
            ],
            device=device),
    )
    assert torch.equal(
        fused_encoder.year_repeats,
        torch.tensor([1, 1, 3, 1, 1, 1], device=device),
    )
    assert torch.allclose(
        fused_encoder.year_bases,
        torch.tensor([1., 4., 9.], device=device),
    )
    assert torch.equal(
        # [:2] because we're not interested in testing the 'exclude' column
        fused_encoder.exclude_feat_mask[:2],
        torch.tensor([
            fused_encoder.exclude_feat_mask.size(1) * [False],
            fused_encoder.exclude_feat_mask.size(1) * [True],
        ], device=device),
    )
    out = fused_encoder.forward(feat)
    assert out.size() == (num_rows, out_channels)


def test_na_forward() -> None:
    num_rows = 10
    num_cols = 2
    feat, na_rows, na_cols = get_fake_timestamp_feat(num_rows, num_cols)
    col_encoder_dict: dict[str, Encoder] = {
        # Test different include_* flags are respected:
        'time0':
        Datetime(
            include_minute=True,
            include_hour=True,
            include_year=True,
            include_day_of_week=True,
            include_day_of_month=True,
            include_day_of_year=True,
            num_year_periods=2,
        ),
        'time1':
        Datetime(
            include_minute=True,
            include_hour=True,
            include_year=True,
            include_day_of_week=True,
            include_day_of_month=True,
            include_day_of_year=True,
            num_year_periods=3,
        ),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
            ColStatType.MAX: f'{TIMESTAMP_PREFIX}2025-01-01',
        }

    out_channels = 5
    fused_encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        out_channels=out_channels,
        enc_mode="fuse",  # not used for this test
        exclude_cols=[],
    )

    # Test N/A handling:
    out = fused_encoder.forward_fixed(feat)
    num_expanded_feats = fused_encoder.num_expanded_feats
    assert out.size() == (num_rows, num_cols, num_expanded_feats, 2)
    na_out = out[na_rows, na_cols, :, :]
    assert torch.allclose(na_out, torch.zeros_like(na_out))


@withCUDA
def test_forward_no_year(device: torch.device) -> None:
    num_rows = 10
    num_cols = 1
    feat, _, _ = get_fake_timestamp_feat(num_rows, num_cols)
    feat = feat.to(device)
    col_encoder_dict: dict[str, Encoder] = {
        'no_year':
        Datetime(
            include_minute=True,
            include_hour=True,
            include_year=False,
            include_day_of_week=True,
            include_day_of_month=True,
            include_day_of_year=True,
            # TODO: specifying num_year_periods with include_year=False should
            # fail early as it doesn't have any effect.
            num_year_periods=999,
        ),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
            ColStatType.MAX: f'{TIMESTAMP_PREFIX}2025-01-01',
        }

    out_channels = 5
    fused_encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        out_channels=out_channels,
        enc_mode="fuse",
        exclude_cols=[],
        device=device,
    )
    assert fused_encoder.num_cols == num_cols
    assert fused_encoder.max_num_year_periods == 1
    assert torch.equal(
        fused_encoder.exclude_cols_mask,
        torch.tensor([False], device=device),
    )
    assert torch.equal(
        fused_encoder.year_repeats,
        torch.tensor([1, 1, 1, 1, 1, 1], device=device),
    )
    assert torch.allclose(
        fused_encoder.year_bases,
        torch.tensor([1.], device=device),
    )
    assert torch.equal(
        fused_encoder.exclude_feat_mask,
        torch.tensor(
            # only year dimension is excluded:
            [[False, False, True, False, False, False]],
            device=device,
        ),
    )
    out = fused_encoder.forward(feat)
    assert out.size() == (num_rows, out_channels)

    # Test different years result in the same output because they're excluded:
    year_index = TimestampTensorMapper.index['year']
    feat[:, :, year_index] = torch.randint(1990, 2023, (num_rows, 1))
    out2 = fused_encoder.forward(feat)
    assert torch.allclose(out, out2)


@withCUDA
@pytest.mark.parametrize("enc_mode", ["fuse", "column"])
def test_empty_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    feat = torch.empty(0, 3, len(TimestampTensorMapper.index), device=device)

    col_encoder_dict = {
        'col0': Datetime(num_year_periods=1),
        'col1': Datetime(num_year_periods=1),
        'col2': Datetime(num_year_periods=1),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
            ColStatType.MAX: f'{TIMESTAMP_PREFIX}2025-01-01',
        }

    encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=5,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(feat)
    assert out.size() == (0, 5) if enc_mode == "fuse" else (0, 3, 5)


@withCUDA
def test_enc_mode(device: torch.device) -> None:
    """Test the head linear layer of FusedNumericalEncoder."""
    feat = torch.tensor(
        [
            # minute, hour, year, day_of_week, day_of_month, day_of_year, days_per_month, days_per_year, weeks_per_year  # noqa: E501
            [[0, 0, 1990, 0, 0, 31, 0, 365, 53],
             [0, 0, 2023, 1, 1, 31, 1, 365, 53]],
            [[59, 23, 1990, 6, 0, 31, 0, 365, 53],
             [0, 0, 2023, 1, 1, 31, 1, 365, 53]],
            [[0, 0, 1990, 0, 0, 31, 0, 365, 53],
             [0, 0, 2023, 1, 1, 31, 1, 365, 53]],
        ],
        device=device,
    )
    num_rows, num_cols, _ = feat.size()
    col_encoder_dict: dict[str, Encoder] = {
        'time0': Datetime(num_year_periods=1),
        'time1': Datetime(num_year_periods=1),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
            ColStatType.MAX: f'{TIMESTAMP_PREFIX}2025-01-01',
        }

    out_channels = 3
    # ===============
    # enc_mode="fuse"
    # ===============
    fused_encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode="fuse",
        device=device,
    )
    out = fused_encoder(feat)
    assert out.size() == (num_rows, out_channels)
    # =================
    # enc_mode="column"
    # =================
    fused_encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode="column",
        device=device,
    )
    out = fused_encoder(feat)
    assert out.size() == (num_rows, num_cols, out_channels)
    # ================
    # invalid enc_mode
    # ================
    with pytest.raises(ValueError, match="Unsupported encoder mode 'invalid'"):
        FusedTimestampEncoder(
            col_encoder_dict=col_encoder_dict,
            exclude_cols=[],
            out_channels=out_channels,
            enc_mode="invalid",
        )


def test_inconsistent_col_order() -> None:
    """Test FusedTimestampEncoder raises error if the column order is different
    between in TF and in the fused encoder.
    """
    num_rows = 5
    num_cols = 2
    feat = torch.randn(num_rows, num_cols, len(TimestampTensorMapper.index))
    col_encoder_dict: dict[str, Encoder] = {
        'time0': Datetime(num_year_periods=1),
        'time1': Datetime(num_year_periods=1),
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
            ColStatType.MAX: f'{TIMESTAMP_PREFIX}2025-01-01',
        }

    fused_encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=1,
        enc_mode="fuse",
    )
    valid_col_names = ["time0", "time1"]
    fused_encoder(feat, valid_col_names)
    invalid_col_names = ["time1", "time0"]
    with pytest.raises(ValueError, match="The order of columns"):
        fused_encoder(feat, invalid_col_names)


@withCUDA
def test_gradient(device: torch.device) -> None:
    col_encoder_dict = {
        '0': Datetime(num_year_periods=1),
        '1': Datetime(num_year_periods=1)
    }
    for encoder in col_encoder_dict.values():
        encoder._stats = {
            ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
            ColStatType.MAX: f'{TIMESTAMP_PREFIX}2025-01-01',
        }
    encoder = FusedTimestampEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=['1'],
        out_channels=16,
        enc_mode="fuse",
        device=device,
    )

    with modeling_context.explaining():
        feat = torch.randn(10, 2, len(TimestampTensorMapper.index),
                           device=device)
        out = encoder(feat)

    out.backward(torch.randn_like(out))

    grad = encoder.get_grad_wrt_input()
    assert isinstance(grad, torch.Tensor)
    assert grad.size() == (10, 2)
    assert not torch.allclose(grad[:, 0], torch.zeros(10, device=device))
    assert torch.allclose(grad[:, 1], torch.zeros(10, device=device))
    assert not hasattr(encoder, 'saved_tensors')
