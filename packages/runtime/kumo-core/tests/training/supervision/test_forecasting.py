# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA
from torch_geometric.utils import lexsort

from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.training.supervision.forecasting import ForecastingTrainTable


@withCUDA
def test_forecasting_train_table(device: torch.device) -> None:
    fkey = torch.tensor([0, 0, 0, 0, 1, 1, 3, 4, 4])
    time = torch.tensor([0, 2, 4, 6, 2, 4, 2, 4, 6])
    target = torch.arange(fkey.numel()) + 1

    perm = torch.randperm(fkey.numel())  # Shuffle randomly.
    fkey, time, target = fkey[perm], time[perm], target[perm]

    train_table = ForecastingTrainTable(fkey, time, target)
    assert str(train_table) == 'ForecastingTrainTable([5, 4])'

    assert train_table.mat.tolist() == [
        [1, 2, 3, 4],
        [0, 5, 6, 0],
        [0, 0, 0, 0],
        [0, 7, 0, 0],
        [0, 0, 8, 9],
    ]

    node_id = torch.tensor([1, 2, 3], device=device)
    seed_time = torch.tensor([2, 2, 4], device=device)

    with pytest.raises(ValueError, match="Cannot query past the timestamp"):
        train_table.query(node_id, seed_time, start=0, length=3)

    out = train_table.query(node_id, seed_time, start=0, length=0)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.size() == (3, 0)

    out = train_table.query(node_id, seed_time, start=-1, length=3)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.tolist() == [
        [0, 5, 6],
        [0, 0, 0],
        [7, 0, 0],
    ]

    node_id = torch.tensor([2, 1, 0], device=device)
    seed_time = torch.tensor([4, 4, 6], device=device)
    out = train_table.query_past(node_id, seed_time, length=2)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.tolist() == [
        [0, 0],
        [0, 5],
        [2, 3],
    ]

    node_id = torch.tensor([2, 1, 0], device=device)
    seed_time = torch.tensor([2, 2, 4], device=device)
    out = train_table.query_future(node_id, seed_time, length=2)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.tolist() == [
        [0, 0],
        [5, 6],
        [3, 4],
    ]

    node_id = torch.tensor([2, 1, 0], device=device)
    seed_time = torch.tensor([2, 2, 4], device=device)
    out, time = train_table.query_past_year(
        node_id,
        seed_time,
        length=2,
        padding_size=4,
    )
    assert out.equal(torch.zeros_like(out))
    assert time.equal(seed_time - NANOSECONDS_IN_DAY * 364)


def test_incorrect_forecasting_train_table() -> None:
    fkey = torch.tensor([0, 1])
    time = torch.tensor([0, 0])
    target = torch.randn(fkey.numel())

    with pytest.raises(ValueError, match="more than a single timestamp"):
        ForecastingTrainTable(fkey, time, target)

    fkey = torch.tensor([0, 0, 0, 1, 1, 1])
    time = torch.tensor([0, 2, 6, 0, 2, 6])
    target = torch.randn(fkey.numel())

    with pytest.raises(ValueError, match="series with a single step size"):
        ForecastingTrainTable(fkey, time, target)

    fkey = torch.tensor([0, 0, 0, 1, 1, 1])
    time = torch.tensor([0, 2, 4, 0, 2, 6])
    target = torch.randn(fkey.numel())

    with pytest.raises(ValueError, match="series with a single step size"):
        ForecastingTrainTable(fkey, time, target)


def test_query_past_year_overflows_when_padding_exceeds_year_offset() -> None:
    # Quarterly step: Y = round(364 / 91) = 4.
    # With padding_size=5 the YoY window extends 1 step *past* seed_time, so
    # querying the last timestamp in the table goes out of bounds.
    step = 91 * NANOSECONDS_IN_DAY
    num_timestamps = 10
    times = torch.arange(num_timestamps, dtype=torch.long) * step
    fkey = torch.cat([torch.zeros(num_timestamps),
                      torch.ones(num_timestamps)]).long()
    time = times.repeat(2)
    target = torch.ones(2 * num_timestamps)
    train_table = ForecastingTrainTable(fkey, time, target)

    last_time = torch.tensor([(num_timestamps - 1) * step], dtype=torch.long)
    node_id = torch.tensor([0])

    out, _ = train_table.query_past_year(
        node_id=node_id,
        seed_time=last_time,
        length=2,
        padding_size=5,  # 5 > Y=4, so window overshoots end of data
    )
    # Overflow positions are zero-padded rather than raising an exception:
    assert out.shape == (1, 2 + 2 * 5)
    assert out[0, -1] == 0.0


def test_adjust_train_mask() -> None:
    fkey = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 4])
    time = torch.tensor([0, 2, 4, 6, 8, 2, 4, 6, 2, 4, 6, 8, 2])
    mask = torch.tensor([1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1]).bool()
    target = torch.arange(fkey.numel()) + 1
    perm = torch.randperm(fkey.numel())  # Shuffle randomly.
    fkey, time, target, mask = fkey[perm], time[perm], target[perm], mask[perm]
    train_table = ForecastingTrainTable(fkey, time, target)

    assert train_table.adjust_train_mask(mask, length=1).equal(mask)

    out = train_table.adjust_train_mask(mask, length=2)
    masked_fkey, masked_time = fkey[out], time[out]
    perm = lexsort([masked_time, masked_fkey])
    assert masked_fkey[perm].tolist() == [0, 0, 0, 1, 1, 3, 3, 4]
    assert masked_time[perm].tolist() == [0, 2, 4, 2, 4, 2, 4, 2]

    out = train_table.adjust_train_mask(mask, length=3)
    masked_fkey, masked_time = fkey[out], time[out]
    perm = lexsort([masked_time, masked_fkey])
    assert masked_fkey[perm].tolist() == [0, 0, 1, 3, 4]
    assert masked_time[perm].tolist() == [0, 2, 2, 2, 2]

    out = train_table.adjust_train_mask(mask, length=4)
    masked_fkey, masked_time = fkey[out], time[out]
    perm = lexsort([masked_time, masked_fkey])
    assert masked_fkey[perm].tolist() == [0]
    assert masked_time[perm].tolist() == [0]

    with pytest.raises(ValueError, match="No valid training examples"):
        train_table.adjust_train_mask(mask, length=5)


def test_adjust_eval_mask() -> None:
    fkey = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 4])
    time = torch.tensor([0, 2, 4, 6, 8, 2, 4, 6, 2, 4, 6, 8, 2])
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]).bool()
    target = torch.arange(fkey.numel()) + 1
    perm = torch.randperm(fkey.numel())  # Shuffle randomly.
    fkey, time, target, mask = fkey[perm], time[perm], target[perm], mask[perm]
    train_table = ForecastingTrainTable(fkey, time, target)

    assert train_table.adjust_eval_mask(mask, length=1).equal(mask)

    out = train_table.adjust_eval_mask(mask, length=2)
    masked_fkey, masked_time = fkey[out], time[out]
    perm = lexsort([masked_time, masked_fkey])
    assert masked_fkey[perm].tolist() == [0, 0, 1, 3, 3, 4]
    assert masked_time[perm].tolist() == [2, 6, 6, 2, 6, 2]

    out = train_table.adjust_eval_mask(mask, length=3)
    masked_fkey, masked_time = fkey[out], time[out]
    perm = lexsort([masked_time, masked_fkey])
    assert masked_fkey[perm].tolist() == [0, 1, 3]
    assert masked_time[perm].tolist() == [4, 4, 4]

    out = train_table.adjust_eval_mask(mask, length=4)
    masked_fkey, masked_time = fkey[out], time[out]
    perm = lexsort([masked_time, masked_fkey])
    assert masked_fkey[perm].tolist() == [0, 3, 4]
    assert masked_time[perm].tolist() == [2, 2, 2]

    out = train_table.adjust_eval_mask(mask, length=5)
    masked_fkey, masked_time = fkey[out], time[out]
    perm = lexsort([masked_time, masked_fkey])
    assert masked_fkey[perm].tolist() == [0]
    assert masked_time[perm].tolist() == [0]

    with pytest.raises(ValueError, match="No valid evaluation examples"):
        train_table.adjust_eval_mask(mask, length=6)
