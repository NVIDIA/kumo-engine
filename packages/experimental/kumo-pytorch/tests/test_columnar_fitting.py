# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected numeric occurrences retain exact statistics without Python boxing."""

from decimal import Decimal

import pyarrow as pa
import pytest

from kumo.pytorch import api, assembly
from kumo.pytorch.encoding import fit_encoder
from tests.test_mixed_graph_fitting import _contract
from tests.test_model_history_limits import history_data as history_data
from tests.test_preview import _equal as assert_equal


@pytest.mark.parametrize(
    "config", [api.ZScore(), api.Buckets(bins=7), api.Buckets(bins=7, separate_zero=True)]
)
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "dtype,values",
    [
        (api.DType("int8"), [-128, None, 0, 0, 1, 127, 127]),
        (api.DType("int64"), [-(2**63), None, 0, 0, 2**53 + 1, 2**63 - 1]),
        (api.DType("uint64"), [0, None, 0, 2**53 + 1, 2**64 - 1, 2**64 - 1]),
        (api.DType("float32"), [-1.0, None, -0.0, 0.0, 0.1, 0.1, 1e20]),
        (api.DType("float64"), [-1e100, None, -0.0, 0.0, 0.1, 0.1, 1e100]),
        (
            api.DType("decimal128", precision=38, scale=10),
            [
                Decimal("-0.1"),
                None,
                Decimal(0),
                Decimal("1.0000000001"),
                Decimal("1.0000000001"),
                Decimal("999999999.1234567890"),
            ],
        ),
        (
            api.DType("decimal256", precision=76, scale=10),
            [
                Decimal("-0.1"),
                None,
                Decimal(0),
                Decimal("1.0000000001"),
                Decimal("1.0000000001"),
                Decimal("1e50"),
            ],
        ),
        (api.DType("float64"), []),
        (api.DType("float64"), [None, None]),
        (api.DType("float64"), [0.0, 0.0, None]),
        (api.DType("float64"), [1.5, 1.5, 1.5]),
    ],
)
def test_numeric_arrow_fitting_matches_scalar_observations(dtype, values, config, chunked):
    expected = fit_encoder("numeric", config, dtype, values)
    column = pa.array(values, type=dtype.to_arrow())
    if chunked:
        column = pa.chunked_array([column.slice(0, 2), column.slice(2), column.slice(0, 0)])
    actual = fit_encoder("numeric", config, dtype, column)
    assert actual.state_dict() == expected.state_dict()
    assert_equal(actual.encode_arrow(column), expected.encode_arrow(column))


def test_numeric_arrow_fitting_does_not_normalize_python_scalars(monkeypatch):
    dtype = api.DType("float64")
    values = pa.chunked_array([[1.0, None, 1.0], [1.0, 9.0]], type=pa.float64())
    expected = fit_encoder("numeric", api.Buckets(bins=4), dtype, values.to_pylist())

    def boxed(*args, **kwargs):
        pytest.fail("typed numeric fitting normalized a Python observation")

    monkeypatch.setattr(api.DType, "normalize", boxed)
    actual = fit_encoder("numeric", api.Buckets(bins=4), dtype, values)
    assert actual.state_dict() == expected.state_dict()
    # Repeated occurrences must contribute their full weight to the quantile.
    assert actual.boundaries == (1.0, 3.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("config", [api.ZScore(), api.Buckets(bins=4)])
def test_numeric_arrow_fitting_rejects_nonfinite_values(value, config):
    for values in ([1.0, value], pa.chunked_array([[1.0], [value]])):
        with pytest.raises(ValueError, match="require finite values"):
            fit_encoder("numeric", config, api.DType("float64"), values)


def test_mismatched_arrow_dtype_retains_scalar_validation():
    with pytest.raises(ValueError, match="not a valid.*int64"):
        fit_encoder("numeric", api.ZScore(), api.DType("int64"), pa.array([True]))


@pytest.mark.parametrize("shared,target", [(True, "computed"), (False, "aligned")])
def test_selected_history_fitting_keeps_columns_and_exact_state(
    history_data, monkeypatch, shared, target
):
    schema, dataset, _memory, index, requests = history_data
    contract = _contract(schema, shared, target)
    monkeypatch.setattr(assembly, "_FIT_CHUNK", 2)
    original = assembly.fit_encoder

    def boxed_reference(*, values, **kwargs):
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            values = values.to_pylist()
        return original(values=values, **kwargs)

    with api.prepare(contract, dataset, api.DiskGraphEngine(index=index, num_threads=1)) as plan:
        with monkeypatch.context() as previous:
            previous.setattr(assembly, "fit_encoder", boxed_reference)
            reference = plan.materialize(requests, fit_split="train")
        with reference:
            expected_state = reference.fitted_state
            expected = list(reference.batches(batch_size=2, output_format="model"))
        observed = []
        arrow_values = assembly._arrow_values
        input_fitting = assembly._fit_inputs

        def reject_numeric_boxing(column):
            if pa.types.is_floating(column.type) or pa.types.is_integer(column.type):
                pytest.fail("numeric input observations were boxed during fitting")
            return arrow_values(column)

        def columnar_fit(*, values, **kwargs):
            if isinstance(kwargs["config"], (api.Buckets, api.ZScore)) and len(values):
                observed.append(values)
            return original(values=values, **kwargs)

        def checked_inputs(*args, **kwargs):
            with monkeypatch.context() as inputs:
                inputs.setattr(assembly, "_arrow_values", reject_numeric_boxing)
                return input_fitting(*args, **kwargs)

        with monkeypatch.context() as candidate:
            candidate.setattr(assembly, "_fit_inputs", checked_inputs)
            candidate.setattr(assembly, "fit_encoder", columnar_fit)
            actual = plan.materialize(requests, fit_split="train")
        with actual:
            assert any(isinstance(values, pa.ChunkedArray) for values in observed)
            assert actual.fitted_state == expected_state
            assert actual.fitted_state.fingerprint == expected_state.fingerprint
            actual_batches = list(actual.batches(batch_size=2, output_format="model"))
            assert len(actual_batches) == len(expected)
            for batch, previous in zip(actual_batches, expected, strict=True):
                assert_equal(batch.inputs, previous.inputs)
                assert_equal(batch.targets, previous.targets)
                assert_equal(batch.metadata, previous.metadata)
