# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor
from torch_geometric.utils import lexsort

from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.encoders.index_mapping import Mapper


class ForecastingTrainTable:
    r"""An efficient data-structure of a forecasting training table to
    efficiently query autoregressive labels and target sequences, given by
    entity indices and seed timestamps.

    Internally, :class:`ForecastingTrainTable` expects a node regression
    training table and reshapes it into format
    ``[num_entity, num_timestamps]``.

    Args:
        fkey: The foreign keys of the training table.
        time: The timestamp of the training table.
        target: The target of the training table.
    """
    def __init__(
        self,
        fkey: Tensor,
        time: Tensor,
        target: Tensor,
    ):
        assert fkey.dim() == 1 and time.dim() == 1 and target.dim() == 1

        self.fkey = fkey
        self.time = time

        # Ensure that we are always operating on floating-point data, even if
        # targets are integers, e.g., in `COUNT(*)` queries:
        target = target.to(torch.get_default_dtype())

        # Sort everything by `fkey` (entity ID) first and then `time`:
        perm = lexsort([time, fkey])
        fkey, time, target = fkey[perm], time[perm], target[perm]

        # Check that the time series is continuously given, i.e., we only allow
        # missing entries at the start and the end of the sequence, but do not
        # allow missing values in between:
        unique_time = time.unique(sorted=True)

        if unique_time.numel() <= 1:
            raise ValueError(f"'{self.__class__.__name__}' expects more than "
                             f"a single timestamp")

        self.step_size = int(unique_time[1] - unique_time[0])

        # Check equal step sizes on unique timestamps:
        if (unique_time.diff() != self.step_size).any():
            raise ValueError(f"'{self.__class__.__name__}' expects a time "
                             f"series with a single step size of "
                             f"{self.step_size}")

        # Check equal step sizes per entity:
        if (time.diff()[fkey.diff() == 0] != self.step_size).any():
            raise ValueError(f"'{self.__class__.__name__}' expects a time "
                             f"series with a single step size of "
                             f"{self.step_size}")

        # Mapping time into consecutive time window indices:
        self.time_map = Mapper(unique_time.cpu().numpy())

        # Bring targets into dense format:
        num_entities = int(fkey.max()) + 1
        num_timestamps = unique_time.numel()
        self.mat = torch.zeros(num_entities * num_timestamps)
        index = self.time_map.map(time) + fkey * num_timestamps
        self.mat[index] = target
        self.mat = self.mat.view(num_entities, num_timestamps)

    def query(
        self,
        node_id: Tensor,
        seed_time: Tensor,
        start: int,
        length: int,
        zero_pad_on_overflow: bool = False,
    ) -> Tensor:

        assert node_id.dim() == 1 and seed_time.dim() == 1

        if length == 0:
            return self.mat.new_empty(node_id.size(0), 0,
                                      device=node_id.device)

        mapped_time = self.time_map.map(seed_time.cpu().numpy())
        assert isinstance(mapped_time, Tensor)
        index = torch.arange(start, start + length)
        index = index.view(1, -1) + mapped_time.view(-1, 1)
        mask = index < 0  # Filter invalid entries.

        if index.max() >= self.mat.size(1):
            if not zero_pad_on_overflow:
                raise ValueError(
                    f"Cannot query past the timestamp horizon. The "
                    f"training table contains {self.mat.size(1)} "
                    f"timestamps, but requested timestamp "
                    f"{int(index.max())}")
            mask = mask | (index >= self.mat.size(1))

        # One-dimensional index to parallelize in both dimensions:
        index = index + (node_id.cpu() * self.mat.size(1)).view(-1, 1)
        out = self.mat.view(-1)[index.clamp(min=0).view(-1)].view(index.size())
        out = out.to(node_id.device)
        out[mask] = 0.0  # Reset invalid entries.

        return out

    def query_past(
        self,
        node_id: Tensor,
        seed_time: Tensor,
        length: int,
    ) -> Tensor:
        r"""Queries the past from a given timestamp per entity.

        Args:
            node_id: The entity node indices.
            seed_time: The timestamp vector of the query.
            length: The number targets from the past.
        """
        return self.query(node_id, seed_time, start=-length, length=length)

    def query_future(
        self,
        node_id: Tensor,
        seed_time: Tensor,
        length: int,
    ) -> Tensor:
        r"""Queries the future from a given timestamp per entity.

        Args:
            node_id: The entity node indices.
            seed_time: The timestamp vector of the query.
            length: The number targets from the future.
        """
        return self.query(node_id, seed_time, start=0, length=length)

    def query_past_year(
        self,
        node_id: Tensor,
        seed_time: Tensor,
        length: int,
        padding_size: int = 0,
    ) -> tuple[Tensor, Tensor]:
        r"""Queries the one-year ago past sequence and timestamp for a given
        seed timestamp per entity.

        Args:
            node_id: The entity node indices.
            seed_time: The timestamp vector of the query.
            length: The number targets from the past.
            padding_size: The padding size added to the start and end of the
                sequence.
        """
        # Find the closest 364 day offset in the training table:
        # NOTE We use an offset of 364 days to define the Year-Over-Year
        # sequence, which guarantees that we compare same week days.
        # Information about holidays, leap years, etc is captured and
        # aggregated through the 1D-CNN on the sequence.
        start = round(364 * NANOSECONDS_IN_DAY / self.step_size)

        seq = self.query(
            node_id,
            seed_time,
            start=-(start + padding_size),
            length=length + 2 * padding_size,
            zero_pad_on_overflow=True,
        )
        time = seed_time - (start * self.step_size)
        return seq, time

    def adjust_train_mask(self, mask: Tensor, length: int) -> Tensor:
        r"""Adjusts the training mask to exclude the last `length - 1` training
        timesteps. Note that this means that we may train a forecast past an
        entity's end time.
        """
        max_time = self.time[mask].max()
        max_time = max_time - self.step_size * (length - 1)
        mask = (self.time <= max_time) & mask

        if mask.sum() == 0:
            raise ValueError("No valid training examples")

        return mask

    def adjust_eval_mask(self, mask: Tensor, length: int) -> Tensor:
        r"""Adjusts the validation/test mask to exclude the last `length - 1`
        evaluation timesteps across all `length`-sized distinct time windows.
        Note that this means that we may predict a forecast past an
        entity's end time.
        """
        # Example: If we have 5 timesteps in the mask from 1 to 5 and use a
        # forecast length of 2, then we want to run validation/test on [2, 3]
        # and [4, 5], and the mask should only include the timestamps 2 and 4.
        # NOTE This is a bit unexpected due to the behavior of entity filters,
        # e.g., it is not guaranteed that an entity is valid for the
        # specific start times of the distinct time windows, but may be valid
        # in between. Currently, we are completely dropping these entities.
        max_time = self.time[mask].max()
        max_time = max_time - self.step_size * (length - 1)
        time = self.time - max_time
        mask = ((time % (length * self.step_size)) == 0) & mask

        if mask.sum() == 0:
            raise ValueError("No valid evaluation examples")

        return mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({list(self.mat.size())})'
