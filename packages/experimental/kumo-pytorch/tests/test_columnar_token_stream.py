# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Columnar packing preserves sample boundaries and token provenance."""

import numpy as np
import pyarrow as pa
import pytest

from kumo.pytorch.contract.batch import Flatten, Vocabulary
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.encoding import fit_encoder
from kumo.pytorch.output import TokenRows, flatten_rows


@pytest.mark.parametrize("separator", ["between", "after_each", "none"])
@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("maximum", [1, 9, 100])
@pytest.mark.parametrize("samples", [[0, 3, 3, 4], [1, 3, 3, 4], [0, 0, 0, 0]])
def test_native_packing_matches_python_boundaries_and_provenance(
    separator, keep, boundary, maximum, samples
):
    # Four events, including one without tokens; the middle sample has no events.
    rows = TokenRows(
        values=np.asarray([10, 11, 12, 20, 21, 30], dtype=np.int64),
        boundaries=np.asarray([0, 3, 3, 5, 6], dtype=np.int64),
        positions=np.asarray([0, 0, 1, 0, 0, 0], dtype=np.int64),
        fields=np.asarray([4, 5, 5, 4, 5, 4], dtype=np.int64),
        slots=np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64),
        field_valid=np.ones(6, dtype=bool),
        events=np.asarray([0, 0, 0, 2, 2, 3], dtype=np.int64),
        occurrences=np.zeros(6, dtype=np.int64),
        source_indices=np.arange(6, dtype=np.int64) + 10,
    )
    samples = np.asarray(samples, dtype=np.int64)
    config = Flatten(
        max_tokens=maximum,
        keep=keep,
        prefer_event_boundary=boundary,
        separator=None if separator == "none" else "separator",
        separator_placement="between" if separator == "none" else separator,
        before="bos" if maximum > 2 else None,
        after="eos" if maximum > 2 else None,
    )
    roles = {"separator": 100, "bos": 101, "eos": 102}
    expected = flatten_rows(rows, samples, config, roles)
    actual = flatten_rows(
        rows, samples, config, roles, packer=pytest.importorskip("diskgraph").pack_token_stream
    )
    for name in (
        "values",
        "boundaries",
        "positions",
        "fields",
        "slots",
        "field_valid",
        "events",
        "occurrences",
        "source_indices",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert np.diff(actual.boundaries).max() <= maximum
    # No operation turns the empty middle sample into another sample's history.
    empty = actual.values[actual.boundaries[1] : actual.boundaries[2]].tolist()
    assert empty == ([101, 102] if maximum > 2 else [])
    marker = actual.values >= 100
    assert not actual.field_valid[marker].any()
    assert (actual.events[marker] == -1).all()


@pytest.mark.parametrize(
    "dtype,values",
    [
        (DType("float64"), [1.0, 1e-5, 1e20]),
        (DType("bool"), [True, False]),
        (DType("int64"), [1, 2]),
        (DType("string"), ["a", "b"]),
    ],
)
def test_category_columns_use_typed_identity(dtype, values):
    encoder = fit_encoder("category", Vocabulary(), dtype, values)
    actual = encoder.encode_arrow(pa.array(values, type=dtype.to_arrow()))["values"]
    assert actual.tolist() == [encoder.encode(value=value) for value in values]


def test_float32_input_cannot_masquerade_as_a_float64_column():
    encoder = fit_encoder("category", Vocabulary(), DType("float64"), [0.1])
    with pytest.raises(ValueError, match="received Arrow float"):
        encoder.encode_arrow(pa.array([0.1], type=pa.float32()))


def test_tail_clipping_preserves_text_piece_positions():
    rows = TokenRows(
        values=np.asarray([10, 11, 12]),
        boundaries=np.asarray([0, 3]),
        positions=np.asarray([0, 1, 2]),
        fields=np.asarray([4, 4, 4]),
        slots=np.asarray([0, 0, 0]),
        field_valid=np.ones(3, dtype=bool),
        events=np.asarray([0, 0, 0]),
        occurrences=np.zeros(3, dtype=np.int64),
    )
    actual = flatten_rows(rows, np.asarray([0, 1]), Flatten(max_tokens=2, keep="last"), {})
    assert actual.values.tolist() == [11, 12]
    assert actual.positions.tolist() == [1, 2]
