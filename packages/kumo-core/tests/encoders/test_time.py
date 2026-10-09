# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor
from torch_geometric.testing import withCUDA
from torch_geometric.utils import one_hot

from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumo_core.encoders.time import TimeEncoder
from kumo_core.types.features import NANOSECONDS_IN_DAY


@withCUDA
@pytest.mark.parametrize('unix_step_size', [None, NANOSECONDS_IN_DAY])
@pytest.mark.parametrize('length', [1, 2])
@pytest.mark.parametrize('padding_size', [0, 1])
def test_time_encoder(
    unix_step_size: int | None,
    length: int,
    padding_size: int,
    device: torch.device,
) -> None:
    model = TimeEncoder(unix_step_size)
    assert TimeEncoder.out_channels == 23
    assert str(model) == 'TimeEncoder()'

    datetime = np.array(['2007-07-13', '2007-01-01'], dtype='datetime64[ns]')
    time = torch.from_numpy(datetime.astype(int)).to(device)

    if unix_step_size is None and (length != 1 or padding_size != 0):
        with pytest.raises(ValueError, match="'unix_step_size' needs to be"):
            model(time, length, padding_size)
        return

    out = model(time, length, padding_size)
    assert out.device == device
    assert out.size() == (
        2,
        length + 2 * padding_size,
        model.out_channels,
    )

    sin = out.view(-1, model.out_channels)[:, 0:5]
    cos = out.view(-1, model.out_channels)[:, 5:10]
    recon = torch.atan2(sin, cos)
    recon = recon[:, :5]  # Only test on cyclic features.
    recon = (recon + math.pi) / (2 * math.pi)
    recon[recon > 1 - 1e-6] = 0.0  # Map values close to 1 back to 0.

    # Denormalize:
    recon[:, 0] *= 60
    recon[:, 1] *= 24
    recon[:, 2] *= 7
    recon[:, 3] *= 31
    recon[:, 4] *= 365

    expected_np = datetime.repeat(length + 2 * padding_size)
    if unix_step_size is not None:
        for i in range(0, length + 2 * padding_size):
            j = i - padding_size
            expected_np[i] += j * np.timedelta64(unix_step_size, 'ns')
            expected_np[length + 2 * padding_size +
                        i] += j * np.timedelta64(unix_step_size, 'ns')

    expected = TimestampTensorMapper()(pd.Series(expected_np), device=device)
    assert isinstance(expected, Tensor)
    index = TimestampTensorMapper.index
    expected = expected[:, [
        index['minute'],
        index['hour'],
        index['day_of_week'],
        index['day_of_month'],
        index['day_of_year'],
    ]]
    assert torch.allclose(recon, expected.to(recon.dtype))

    holiday = out.view(-1, model.out_channels)[:, 10:12]
    holiday_name = out.view(-1, model.out_channels)[:, 12:23]
    holiday_name = holiday_name.nonzero(as_tuple=True)[1]
    is_holiday = model.is_holiday(time, length, padding_size)
    assert holiday_name.equal(torch.tensor([7], device=device))
    assert holiday.equal(one_hot((~is_holiday).view(-1).long()))
