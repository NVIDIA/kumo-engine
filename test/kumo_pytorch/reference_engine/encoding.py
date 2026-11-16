"""Arena-owned columnar acceleration hooks for Kumo's fitted encoders.

Fitting and token grammar interpretation stay in Kumo. These hooks only look up
fitted vocabulary positions and pack already encoded grammar-part columns.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class VocabularyLookup:
    """Reuse sorted fitted keys when vocabulary construction dominates lookup.

    Small vocabularies/large batches retain Arrow's native hash lookup; sorted
    lookup avoids rebuilding a large hash table for a small batch.
    """

    def __init__(self, keys):
        keys = pc.cast(keys, pa.string())
        self._keys_array = keys
        self._keys = self._positions = None
        self._null_position = None

    def _sort_keys(self):
        keys = self._keys_array
        valid = pc.is_valid(keys).to_numpy(zero_copy_only=False)
        values = keys.to_numpy(zero_copy_only=False)
        positions = np.flatnonzero(valid)
        self._keys, first = np.unique(values[valid], return_index=True)
        self._positions = positions[first].astype(np.int64, copy=False)
        missing = np.flatnonzero(~valid)
        self._null_position = int(missing[0]) if len(missing) else None
        self._keys.flags.writeable = False
        self._positions.flags.writeable = False

    def lookup(self, values):
        # Measured crossover: hashing wins for AML-sized vocabularies and large
        # probe batches. Keep that path unchanged instead of forcing sorting.
        if len(self._keys_array) < max(4096, 32 * len(values)):
            return pc.index_in(values, value_set=self._keys_array)
        if self._keys is None:
            self._sort_keys()
        values = pc.cast(values, pa.string())
        valid = pc.is_valid(values).to_numpy(zero_copy_only=False)
        strings = pc.fill_null(values, "").to_numpy(zero_copy_only=False)
        positions = np.zeros(len(values), dtype=np.int64)
        known = np.zeros(len(values), dtype=bool)
        if len(self._keys):
            found = np.searchsorted(self._keys, strings)
            safe = np.minimum(found, len(self._keys) - 1)
            known = valid & (found < len(self._keys)) & (self._keys[safe] == strings)
            positions[known] = self._positions[safe[known]]
        if self._null_position is not None:
            positions[~valid] = self._null_position
            known |= ~valid
        return pa.array(positions, mask=~known, type=pa.int64())


def _integers(values, name):
    if not isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = pa.array(values)
    if not pa.types.is_integer(values.type) or values.null_count:
        raise ValueError(f"{name} must contain non-null integers")
    return pc.cast(values, pa.int64(), safe=True).to_numpy(zero_copy_only=False)


def _ranges(starts, lengths):
    """Flatten contiguous ranges without a Python loop over samples or events."""
    prefixes = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    return np.repeat(starts - prefixes[:-1], lengths) + np.arange(
        prefixes[-1], dtype=np.int64
    )


def pack_token_stream(
    offsets,
    parts,
    max_tokens,
    keep_last,
    truncate_at_event_boundary,
    separator_id,
    separator_between,
    bos_id,
    eos_id,
):
    """Pack Kumo's event/part token columns with vectorized stable scatter/gather.

    The ordering is sample, event, grammar part, token within a part. Separators
    belong to their preceding event before truncation, as in Kumo's reference
    implementation. If no whole event fits, the oversized boundary event is
    sliced so a valid small token budget still produces content.
    """
    offsets = _integers(offsets, "sample event offsets")
    if not len(offsets) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("sample event offsets must start at zero and be nondecreasing")
    prefix, suffix = int(bos_id is not None), int(eos_id is not None)
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, Integral)
        or max_tokens <= prefix + suffix
    ):
        raise ValueError("max_tokens must leave room for content after boundary tokens")
    event_count = int(offsets[-1])
    counts = np.zeros(event_count, dtype=np.int64)
    columns = []
    for events, tokens in parts:
        events, tokens = (
            _integers(events, "part event ids"),
            _integers(tokens, "part token ids"),
        )
        if (
            len(events) != len(tokens)
            or np.any(events < 0)
            or np.any(events >= event_count)
        ):
            raise ValueError(
                "part event ids and token ids must align with existing events"
            )
        # Kumo emits ordered event IDs. Retain stable semantics for direct callers.
        if np.any(np.diff(events) < 0):
            order = np.argsort(events, kind="stable")
            events, tokens = events[order], tokens[order]
        part_counts = np.bincount(events, minlength=event_count)
        columns.append((events, tokens, part_counts))
        counts += part_counts
    separators = np.zeros(event_count, dtype=bool)
    if separator_id is not None:
        separators[:] = True
        if separator_between:
            nonempty = np.diff(offsets) > 0
            separators[offsets[1:][nonempty] - 1] = False
        counts += separators
    event_offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    packed = np.empty(event_offsets[-1], dtype=np.int64)
    filled = np.zeros(event_count, dtype=np.int64)
    for events, tokens, part_counts in columns:
        within = np.arange(len(events), dtype=np.int64) - np.repeat(
            np.r_[0, np.cumsum(part_counts, dtype=np.int64)[:-1]], part_counts
        )
        packed[event_offsets[events] + filled[events] + within] = tokens
        filled += part_counts
    if separator_id is not None:
        positions = np.flatnonzero(separators)
        packed[event_offsets[positions] + filled[positions]] = separator_id
    lo, hi = event_offsets[offsets[:-1]].copy(), event_offsets[offsets[1:]].copy()
    maximum = int(max_tokens) - prefix - suffix
    overflow = hi - lo > maximum
    if keep_last:
        clipped = np.maximum(lo, hi - maximum)
        if truncate_at_event_boundary:
            begins = np.searchsorted(event_offsets, hi[overflow] - maximum, side="left")
            candidate = event_offsets[begins]
            clipped[overflow] = np.where(
                begins < offsets[1:][overflow], candidate, hi[overflow] - maximum
            )
        lo = clipped
    else:
        clipped = np.minimum(hi, lo + maximum)
        if truncate_at_event_boundary:
            ends = (
                np.searchsorted(event_offsets, lo[overflow] + maximum, side="right") - 1
            )
            candidate = event_offsets[ends]
            clipped[overflow] = np.where(
                ends > offsets[:-1][overflow], candidate, lo[overflow] + maximum
            )
        hi = clipped
    lengths = hi - lo
    output_offsets = np.r_[0, np.cumsum(lengths + prefix + suffix, dtype=np.int64)]
    output = np.empty(output_offsets[-1], dtype=np.int64)
    if bos_id is not None:
        output[output_offsets[:-1]] = bos_id
    if eos_id is not None:
        output[output_offsets[1:] - 1] = eos_id
    output[_ranges(output_offsets[:-1] + prefix, lengths)] = packed[
        _ranges(lo, lengths)
    ]
    return pa.array(output_offsets), pa.array(output)
