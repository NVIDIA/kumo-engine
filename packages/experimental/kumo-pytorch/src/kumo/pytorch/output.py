# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common token buffers and padded/packed storage for selected rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from kumo.pytorch.contract.batch import Flatten, Packed


def offsets(lengths: Any) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))


def layout_array(values: Any, boundaries: np.ndarray, layout: Any, *, fill: Any = 0) -> Any:
    """Store one row axis; never change its values or sequence boundaries."""

    if isinstance(layout, Packed):
        return values
    import pyarrow as pa

    lengths = np.diff(boundaries)
    width = int(lengths.max(initial=0))
    owners = np.repeat(np.arange(len(lengths)), lengths)
    positions = np.arange(int(boundaries[-1])) - np.repeat(boundaries[:-1], lengths)
    if isinstance(values, torch.Tensor):
        result = torch.full((len(lengths), width, *values.shape[1:]), fill, dtype=values.dtype)
        result[owners, positions] = values
        return result
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    indices = np.full((len(lengths), width), -1, dtype=np.int64)
    indices[owners, positions] = np.arange(len(values))
    import pyarrow.compute as pc

    flat = pc.take(values, pa.array(indices.reshape(-1), mask=indices.reshape(-1) < 0))
    return (
        pa.FixedSizeListArray.from_arrays(flat, width)
        if width
        else pa.array([[] for _ in lengths], type=pa.list_(values.type))
    )


def sequence_layout(boundaries: np.ndarray, layout: Any, **arrays: Any) -> dict[str, Any]:
    valid = torch.ones(int(boundaries[-1]), dtype=torch.bool)
    result = {
        name: layout_array(value, boundaries, layout, fill=-1 if name == "event_indices" else 0)
        for name, value in arrays.items()
    }
    result["valid"] = layout_array(valid, boundaries, layout, fill=False)
    if isinstance(layout, Packed):
        result["offsets"] = torch.from_numpy(boundaries.copy())
    return result


def kernel_metadata(value: Any) -> Any:
    """Attach checked attention-kernel metadata to explicit packed boundaries."""

    if not isinstance(value, dict):
        return value
    result = {name: kernel_metadata(item) for name, item in value.items()}
    for name, item in value.items():
        if name != "offsets" and not name.endswith("_offsets"):
            continue
        if not isinstance(item, torch.Tensor) or item.ndim != 1 or not item.numel():
            continue
        if int(item[-1]) > np.iinfo(np.int32).max:
            raise ValueError(f"packed boundary {name} exceeds int32 attention-kernel range")
        prefix = name[: -len("offsets")]
        result[prefix + "cu_seqlens"] = item.to(torch.int32)
        result[prefix + "max_seqlen"] = int(torch.diff(item).max()) if len(item) > 1 else 0
    return result


@dataclass(frozen=True)
class TokenRows:
    """Flat encoded positions with one boundary per retained source row."""

    values: np.ndarray
    boundaries: np.ndarray
    positions: np.ndarray
    fields: np.ndarray
    slots: np.ndarray
    field_valid: np.ndarray
    events: np.ndarray
    occurrences: np.ndarray
    source_indices: np.ndarray | None = None

    def select(self, indices: np.ndarray, boundaries: np.ndarray) -> TokenRows:
        return TokenRows(
            values=self.values[indices],
            boundaries=boundaries,
            positions=self.positions[indices],
            fields=self.fields[indices],
            slots=self.slots[indices],
            field_valid=self.field_valid[indices],
            events=self.events[indices],
            occurrences=self.occurrences[indices],
            source_indices=None if self.source_indices is None else self.source_indices[indices],
        )

    def output(
        self, layout: Any, *, pad: int, field_ids: bool, field_pad: int = 0
    ) -> dict[str, Any]:
        result = sequence_layout(
            self.boundaries,
            layout,
            value_ids=torch.from_numpy(self.values.copy()),
            position_ids=torch.from_numpy(self.positions.copy()),
            source_event_indices=torch.from_numpy(self.events.copy()),
        )
        if not isinstance(layout, Packed):
            result["value_ids"][~result["valid"]] = pad
        if field_ids:
            result["field_ids"] = layout_array(
                torch.from_numpy(self.fields.copy()), self.boundaries, layout
            )
            result["field_valid"] = layout_array(
                torch.from_numpy(self.field_valid.copy()), self.boundaries, layout, fill=False
            )
            if field_pad:
                result["field_ids"][~result["field_valid"]] = field_pad
        return result


def flatten_rows(
    rows: TokenRows, sample_offsets: np.ndarray, config: Flatten, roles: Any, *, packer=None
) -> TokenRows:
    """Flatten per sample with one shared clipping rule for every parallel buffer.

    A native packer operates on position indices, so values and all provenance
    follow exactly the same retained positions without separate packing passes.
    """

    import pyarrow as pa

    lengths = np.diff(sample_offsets)
    nonempty = np.flatnonzero(lengths)
    retained: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in lengths]
    before, separator, after = (
        -1 if config.before else None,
        -2 if config.separator else None,
        -3 if config.after else None,
    )
    if packer is not None and len(nonempty):
        start, stop = sample_offsets[[0, -1]]
        token_take = np.arange(rows.boundaries[start], rows.boundaries[stop], dtype=np.int64)
        event_ids = np.repeat(np.arange(stop - start), np.diff(rows.boundaries[start : stop + 1]))
        packed_offsets, packed = packer(
            pa.array(offsets(lengths[nonempty]), type=pa.int64()),
            [(pa.array(event_ids, type=pa.int64()), pa.array(token_take, type=pa.int64()))],
            config.max_tokens,
            config.keep == "last",
            config.prefer_event_boundary,
            separator,
            config.separator_placement == "between",
            before,
            after,
        )
        bounds = packed_offsets.to_numpy(zero_copy_only=False)
        ids = packed.to_numpy(zero_copy_only=False)
        for position, owner in enumerate(nonempty):
            retained[owner] = ids[bounds[position] : bounds[position + 1]]
    else:
        for owner in nonempty:
            segments = []
            for event in range(sample_offsets[owner], sample_offsets[owner + 1]):
                segment = np.arange(rows.boundaries[event], rows.boundaries[event + 1])
                if separator is not None and (
                    config.separator_placement == "after_each"
                    or event + 1 < sample_offsets[owner + 1]
                ):
                    segment = np.concatenate((segment, [separator]))
                segments.append(segment)
            boundaries = offsets([len(item) for item in segments])
            ids = np.concatenate(segments) if segments else np.empty(0, dtype=np.int64)
            budget = config.max_tokens - int(before is not None) - int(after is not None)
            lo, hi = 0, len(ids)
            if hi > budget:
                if config.keep == "last":
                    lo = hi - budget
                    if config.prefer_event_boundary:
                        candidate = boundaries[np.searchsorted(boundaries, lo)]
                        lo = int(candidate) if candidate < hi else lo
                else:
                    hi = budget
                    if config.prefer_event_boundary:
                        candidate = boundaries[np.searchsorted(boundaries, hi, side="right") - 1]
                        hi = int(candidate) if candidate else hi
            retained[owner] = np.asarray(
                ([] if before is None else [before])
                + ids[lo:hi].tolist()
                + ([] if after is None else [after]),
                dtype=np.int64,
            )
    for owner in np.flatnonzero(lengths == 0):
        retained[owner] = np.asarray(
            ([] if before is None else [before]) + ([] if after is None else [after]),
            dtype=np.int64,
        )
    selected = np.concatenate(retained) if len(retained) else np.empty(0, dtype=np.int64)
    boundaries = offsets([len(item) for item in retained])
    real = selected >= 0

    def take(values, fill):
        result = np.full(len(selected), fill, dtype=values.dtype)
        result[real] = values[selected[real]]
        return result

    values = take(rows.values, 0)
    for index, role in ((-1, config.before), (-2, config.separator), (-3, config.after)):
        if role is not None:
            values[selected == index] = roles[role]
    return TokenRows(
        values=values,
        boundaries=boundaries,
        positions=take(rows.positions, 0),
        fields=take(rows.fields, 0),
        slots=take(rows.slots, -1),
        field_valid=take(rows.field_valid, False),
        events=take(rows.events, -1),
        occurrences=take(rows.occurrences, -1),
        source_indices=None if rows.source_indices is None else take(rows.source_indices, -1),
    )
