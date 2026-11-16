# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bisect
import itertools
from dataclasses import dataclass
from typing import Iterable, List

import pyarrow as pa


@dataclass
class _TakeIndices:
    # for each chunk in the source, list of indices to use in output
    per_chunk_idx: List[List[int]]
    # for each element of output, intermediate position
    reassemble_idx: List[int]


def _get_list_offsets(lengths: Iterable[int]):
    return [0] + list(itertools.accumulate(lengths))


def _get_arr_offsets(arr: pa.lib.ChunkedArray):
    return _get_list_offsets([len(c) for c in arr.chunks])


def _resolve_take_indices(idx: Iterable[int],
                          offsets: List[int]) -> _TakeIndices:
    per_chunk_idx = [[] for _ in offsets[:-1]]
    chunk_reassemble_idx = []

    # each element of output flows through an intermediate position
    # source -> intermediate defined by per_chunk_idx
    # intermediate -> output defined by chunk_reassemble_idx
    for i in idx:
        chunk_idx = bisect.bisect_right(offsets, i) - 1
        idx_in_chunk = i - offsets[chunk_idx]
        cri = (chunk_idx, len(per_chunk_idx[chunk_idx]))
        chunk_reassemble_idx.append(cri)
        per_chunk_idx[chunk_idx].append(idx_in_chunk)

    lengths = [len(a) for a in per_chunk_idx]
    r_offsets = _get_list_offsets(lengths)
    # resolve chunk_reassemble_idx to flat indices
    reassemble_idx = [r_offsets[c] + i for (c, i) in chunk_reassemble_idx]

    return _TakeIndices(per_chunk_idx, reassemble_idx)


# chunked_take handles the case where arr is too large to concatenate
# N.B. will fail if the portion of arr selected by idx is still too large
def chunked_take(arr: pa.lib.ChunkedArray, idx: Iterable[int]):
    if arr.length() == 0 or len(idx) == 0:  # type: ignore
        return pa.array([])
    offsets = _get_arr_offsets(arr)
    indices = _resolve_take_indices(idx, offsets)

    taken_chunks = []
    for c_data, c_idx in zip(arr.chunks, indices.per_chunk_idx):
        if len(c_idx) > 0:
            taken_chunks.append(c_data.take(c_idx))
        else:
            taken_chunks.append(c_data.slice(length=0))
    taken_unordered = pa.chunked_array(taken_chunks)
    taken = taken_unordered.take(indices.reassemble_idx)
    return taken
