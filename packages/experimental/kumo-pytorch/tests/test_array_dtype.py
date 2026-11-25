# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape numeric column contracts, physical types, and compatibility."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.test_type_resolution import _source_contract


@pytest.mark.parametrize("name", ["float32", "float64", "int16", "int64", "uint64"])
def test_array_types_round_trip_and_infer_fixed_width(name):
    dtype = api.DType(name, shape=(2, 3))
    physical = api.DType(name).to_arrow()
    assert dtype.to_arrow() == pa.list_(physical, 6)
    assert api.DType.from_arrow(dtype.to_arrow()) == api.DType(name, shape=(6,))
    for arrow in (pa.list_(physical, 6), pa.list_(physical), pa.large_list(physical)):
        assert api.DType.from_arrow(arrow, shape=(2, 3)) == dtype

    contract = _source_contract({"id": api.DType("int64"), "value": dtype})
    assert api.DataLoaderSpec.from_json(contract.to_json()).schema == contract.schema
    one_dimensional = _source_contract(
        {"id": api.DType("int64"), "value": api.DType(name, shape=(6,))}
    )
    assert one_dimensional.fingerprint != contract.fingerprint


def test_scalar_storage_fingerprint_is_unchanged():
    contract = _source_contract({"id": api.DType("int64"), "value": api.DType("float32")})
    assert '"shape"' not in contract.to_json()
    assert "shape" not in api.DType("float32").to_dict()
    # Captured from this scalar-only declaration before array types were added.
    assert (
        contract.fingerprint
        == "sha256:e647094a011cd70164fd27f733631e84f887866380810790dc3e772265a39469"
    )


@pytest.mark.parametrize("shape", [4, (0,), (2, -1), (2, False), (True,), (3.0,), "2"])
def test_invalid_array_shape(shape):
    with pytest.raises(ValueError, match="positive integer"):
        api.DType("int32", shape=shape)


@pytest.mark.parametrize("name", ["string", "bytes", "bool", "date32", "decimal128"])
def test_non_tensor_element_types_are_rejected(name):
    with pytest.raises(ValueError, match="integer or floating-point"):
        api.DType(name, shape=(1,))


def test_shape_and_element_validation():
    dtype = api.DType("uint8", shape=(2, 2))
    assert dtype.normalize(None) is None
    assert dtype.normalize([1, 2, 3, 4]) == (1, 2, 3, 4)
    for value in ([[1, 2], [3, 4]], [1, 2, 3], [1, 2, 256, 4], [1, None, 3, 4]):
        with pytest.raises(ValueError, match="column"):
            dtype.normalize(value, path="table.column")
    with pytest.raises(ValueError, match="explicit array shape"):
        api.DType.from_arrow(pa.list_(pa.uint8()))
    with pytest.raises(ValueError, match="physical list width"):
        api.DType.from_arrow(pa.list_(pa.uint8(), 3), shape=(2, 2))
    with pytest.raises(ValueError, match="element type"):
        api.DType.from_arrow(pa.list_(pa.list_(pa.uint8())), shape=(2, 2))


def test_arrow_validation_uses_only_visible_rows_and_elements():
    dtype = api.DType("int32", shape=(2,))
    bad = pa.array([[1], None, [10, 11], [33, None]], type=pa.large_list(pa.int32()))
    dtype.validate_array(bad.slice(1, 2))
    dtype.validate_array(pa.chunked_array([bad.slice(0, 0), bad.slice(1, 2)]))
    with pytest.raises(ValueError, match="2 elements per row"):
        dtype.validate_array(bad)
    with pytest.raises(ValueError, match="elements cannot be null"):
        dtype.validate_array(bad.slice(3))
    fixed = pa.FixedSizeListArray.from_arrays(
        pa.array([None, 4, 5, 6], type=pa.int32()), 2, mask=pa.array([True, False])
    )
    dtype.validate_array(fixed)
    with pytest.raises(ValueError, match="physical array type"):
        dtype.validate_array(pa.array([[1, 2]], type=pa.list_(pa.int16())))


def test_variable_parquet_shape_is_declared_and_fixed_width_infers(tmp_path):
    path = tmp_path / "users.parquet"
    dataset = api.Dataset({"users": api.Parquet(str(path))})
    vector = api.DType("float32", shape=(2, 2))
    for physical in (pa.list_(pa.float32(), 4), pa.list_(pa.float32())):
        pq.write_table(
            pa.table({"id": [1], "value": pa.array([[1, 2, 3, 4]], type=physical)}), path
        )
        resolved = _source_contract({"id": None, "value": vector}).resolve(dataset=dataset)
        assert resolved.schema.tables[0].columns["value"] == vector
        if pa.types.is_fixed_size_list(physical):
            assert _source_contract().resolve(dataset=dataset).schema.tables[0].columns[
                "value"
            ] == api.DType("float32", shape=(4,))
        else:
            with pytest.raises(ValueError, match="explicit array shape"):
                _source_contract().resolve(dataset=dataset)


def test_keys_are_scalar():
    with pytest.raises(ValueError, match="primary key must be a scalar"):
        api.TableSpec(
            "users", "id", {"id": api.DType("int32", shape=(2,))}, time_column=api.NON_TEMPORAL
        )


def test_constants_and_external_labels_keep_scalar_semantics():
    dtype = api.DType("float32", shape=(2,))
    with pytest.raises(ValueError, match="scalar dtype"):
        api.Constant([2.0, 3.0], dtype)
    with pytest.raises(ValueError, match="scalar dtype"):
        api.ExternalTarget(dtype)
