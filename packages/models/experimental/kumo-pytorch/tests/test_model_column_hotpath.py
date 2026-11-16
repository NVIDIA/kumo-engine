# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed model columns retain exact values without rebuilding row records."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch

from kumo.pytorch.assembly import _column_output, _model_rows, _PreparedInput
from kumo.pytorch.contract.batch import (
    Columns,
    Encode,
    EventSource,
    Fields,
    Input,
    Padded,
    Raw,
    Vocabulary,
)
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.data.model import ModelSequence
from kumo.pytorch.encoding import OutputBinding, fit_encoder


@pytest.mark.parametrize(
    "mapping,expected",
    [
        ((("b", 7), ("a", 3), ("café", 12)), [3, 8, 9, 12, 7]),
        ((), [9, 8, 9, 9, 9]),
    ],
)
def test_arrow_vocabulary_preserves_missing_unknown_and_sparse_ids(mapping, expected):
    encoder = fit_encoder(
        "category", Vocabulary(values=mapping, missing_id=8, unknown_id=9), DType("string"), ()
    )
    values = pa.array(["a", None, "?", "café", "b"])
    encoded = encoder.encode_arrow(values)
    assert encoded["values"].tolist() == expected
    assert encoded["values"].dtype == torch.long
    assert encoded["valid"].tolist() == [True, False, True, True, True]
    assert encoder.encode_arrow(pa.array([], type=pa.string()))["values"].numel() == 0


def test_fitted_vocabulary_arrays_are_cached_once():
    encoder = fit_encoder("category", Vocabulary(values=(("b", 7), ("a", 3))), DType("string"), ())
    for _ in range(3):
        assert encoder.encode_arrow(pa.array(["a", "b"]))["values"].tolist() == [3, 7]
    assert encoder.arrow_keys is encoder.arrow_keys
    assert encoder.entry_ids is encoder.entry_ids


def test_supplied_ids_outside_int64_refuse_before_execution():
    with pytest.raises(ValueError, match="IDs"):
        Vocabulary(values=(("a", 2**64),))


def _declarations(dtype, name="amount"):
    return {name: ((("events", "orders", name), dtype, "scalar"),)}


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("limit", [None, 1, 2, 8])
def test_one_ordered_stream_does_not_sort_again(monkeypatch, keep, limit):
    sequence = ModelSequence(
        pa.array([0, 3, 3, 5]),
        pa.array([2, 2, 5, 1, 4]),
        {"amount": pa.array([10.0, None, 30.0, 40.0, 50.0])},
    )
    source = EventSource(events=("orders",), max_events=limit, keep=keep)
    values = SimpleNamespace(events={"orders": sequence}, batch_size=3)
    selected = [[0, 1, 2], [], [3, 4]]
    if limit is not None:
        selected = [row[:limit] if keep == "first" else row[-limit:] for row in selected]
    indices = pa.array([index for row in selected for index in row], type=pa.int64())

    def no_sort(*_args, **_kwargs):
        raise AssertionError("already ordered stream was sorted or synthesized again")

    monkeypatch.setattr(np, "lexsort", no_sort)
    monkeypatch.setattr(pa, "nulls", no_sort)
    actual = _model_rows(source, values, _declarations(DType("float64")))
    np.testing.assert_array_equal(actual.boundaries, np.cumsum([0, *map(len, selected)]))
    assert actual.columns["amount"].equals(sequence.fields["amount"].take(indices))
    assert actual.sources.tolist() == ["orders"] * len(indices)


@pytest.mark.parametrize(
    "config,dtype,values",
    [
        (Raw(), DType("float64"), [1.0, None, 2.0]),
        (Vocabulary(values=(("a", 3),)), DType("string"), ["a", None, "?"]),
    ],
)
def test_encoded_tensors_are_writable_without_aliasing(config, dtype, values):
    encoder = fit_encoder("field", config, dtype, ())
    source = pa.array(values, type=dtype.to_arrow())
    left, right = encoder.encode_arrow(source), encoder.encode_arrow(source)
    expected = right["values"].clone()
    left["values"][0] = -100
    left["valid"][0] = False
    assert source.to_pylist() == values
    assert torch.equal(right["values"], expected)
    assert right["valid"][0]
    assert torch.equal(encoder.encode_arrow(source)["values"], expected)


@pytest.mark.parametrize(
    "times,expected",
    [
        ([5, 1, 1], [10, 11, 50]),
        ([2, None, 1], [10, 20, 99]),
    ],
)
def test_unordered_or_nullable_clock_uses_stable_general_order(times, expected):
    ids = [50, 10, 11] if times[0] == 5 else [20, 99, 10]
    sequence = ModelSequence(pa.array([0, 3]), pa.array(times), {"id": pa.array(ids)})
    values = SimpleNamespace(events={"orders": sequence}, batch_size=1)
    actual = _model_rows(
        EventSource(events=("orders",)), values, _declarations(DType("int64"), "id")
    )
    assert actual.columns["id"].to_pylist() == expected


@pytest.mark.parametrize("empty", [False, True])
def test_padding_preserves_presence_and_empty_histories(empty):
    sequence = ModelSequence(
        pa.array([0, 0, 0] if empty else [0, 2, 2]),
        pa.array([] if empty else [1, 2], type=pa.int64()),
        {"amount": pa.array([] if empty else [10, None], type=pa.float64())},
    )
    values = SimpleNamespace(events={"orders": sequence}, batch_size=2)
    declaration = _declarations(DType("float64"))
    spec = Input(
        source=EventSource(events=("orders",)),
        representation=Fields(fields=(Encode(field="amount", encoding=Raw()),), row=Columns()),
        layout=Padded(),
    )
    prepared = _PreparedInput(
        spec=spec,
        declarations=declaration,
        state={"amount": OutputBinding(encoder=fit_encoder("amount", Raw(), DType("float64"), ()))},
    )
    output = _column_output(prepared, _model_rows(spec.source, values, declaration), model=True)
    assert "timestamps" not in output
    if empty:
        assert output["valid"].shape == (2, 0)
        assert output["amount"]["values"].shape == (2, 0)
    else:
        assert output["valid"].tolist() == [[True, True], [False, False]]
        assert output["amount"]["values"].tolist() == [[10, 0], [0, 0]]
        assert output["amount"]["valid"].tolist() == [[True, False], [False, False]]


def test_interleaved_components_keep_source_order_and_values():
    values = SimpleNamespace(
        batch_size=2,
        events={
            "orders": ModelSequence(
                pa.array([0, 2, 2]), pa.array([1, 4]), {"amount": pa.array([10.0, 40.0])}
            ),
            "payments": ModelSequence(
                pa.array([0, 1, 2]), pa.array([2, 3]), {"amount": pa.array([20.0, 30.0])}
            ),
        },
    )
    declaration = {
        "amount": (
            (("events", "orders", "amount"), DType("float64"), "scalar"),
            (("events", "payments", "amount"), DType("float64"), "scalar"),
        )
    }
    actual = _model_rows(EventSource(events=("orders", "payments")), values, declaration)
    assert actual.sources.tolist() == ["orders", "payments", "orders", "payments"]
    assert actual.columns["amount"].to_pylist() == [10.0, 20.0, 40.0, 30.0]
    assert actual.boundaries.tolist() == [0, 3, 4]


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("limit", [None, 1, 3])
def test_native_components_bypass_python_sort_without_changing_fields(
    monkeypatch, keep, order, limit
):
    native = pytest.importorskip("diskgraph")
    if not hasattr(native, "merge_event_streams"):
        pytest.skip("native event merging is unavailable")
    values = SimpleNamespace(
        batch_size=3,
        events={
            "orders": ModelSequence(
                pa.array([0, 3, 3, 4]),
                pa.array([1, 3, 3, 8]),
                {"amount": pa.array([None, 2.0, 3.0, 4.0])},
            ),
            "payments": ModelSequence(
                pa.array([0, 2, 3, 3]),
                pa.array([3, 3, 9]),
                {"amount": pa.array([10.0, None, 30.0])},
            ),
        },
    )
    declarations = {
        "amount": (
            (("events", "orders", "amount"), DType("float64"), "scalar"),
            (("events", "payments", "amount"), DType("float64"), "scalar"),
        )
    }
    source = EventSource(events=("payments", "orders"), max_events=limit, keep=keep, order=order)
    expected = _model_rows(source, values, declarations)
    values.merge_event_streams = native.merge_event_streams

    def no_sort(*args, **kwargs):
        raise AssertionError("native component merging must bypass Python lexsort")

    monkeypatch.setattr(np, "lexsort", no_sort)
    actual = _model_rows(source, values, declarations)
    assert actual.boundaries.tolist() == expected.boundaries.tolist()
    assert actual.sources.tolist() == expected.sources.tolist()
    assert actual.columns["amount"].equals(expected.columns["amount"])
