# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_mapper import (
    SequenceTensorMapper,
    _row_path,
    _tiled_mask,
)
from tests.support.mapper_reference import sequence_pandas_reference
from torch_frame.data import MultiEmbeddingTensor


# Retain the previous native implementation to verify optimized padding paths.
def _ref_from_arrow(self, arr, *, device=None):
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    is_list = isinstance(arr, (pa.ListArray, pa.LargeListArray))
    if is_list and len(arr) > 0:
        offsets_np = arr.offsets.to_numpy(zero_copy_only=False)
        values_arr = arr.values
        if (
            arr.null_count == 0
            and offsets_np[0] == 0
            and offsets_np[-1] == len(values_arr)
        ):
            self._is_string_dtype = False
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*non-writable.*")
                values = torch.from_numpy(values_arr.to_numpy(zero_copy_only=False)).to(
                    device, torch.get_default_dtype()
                )
            offsets = torch.from_numpy(offsets_np.astype(np.int64)).to(device)
            seq_len = offsets[1:] - offsets[:-1]
            n_rows = len(seq_len)
            drop_count = (seq_len - self.max_length).clamp(min=0)
            seq_len = seq_len.clamp(max=self.max_length)
            values = values[_tiled_mask(seq_len, drop_count)]
            padding_len = self.max_length - seq_len
            result = torch.full(
                (n_rows * self.max_length,), float("NaN"), device=device
            )
            result[_tiled_mask(seq_len, padding_len)] = values
            return MultiEmbeddingTensor(
                num_rows=n_rows,
                num_cols=1,
                values=result.view(n_rows, self.max_length),
                offset=torch.tensor([0, self.max_length], device=device),
            )

    return sequence_pandas_reference(self, arr.to_pandas(), device=device)


def _ref_post_forward(self, array, device=None):
    if len(array) == 0:
        return MultiEmbeddingTensor(
            num_rows=0,
            num_cols=1,
            values=torch.empty(0, self.max_length, device=device),
            offset=torch.tensor([0, self.max_length], device=device),
        )
    assert isinstance(array, pa.ListArray)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-writable.*")
        values = torch.from_numpy(array.values.to_numpy()).to(
            device, torch.get_default_dtype()
        )

    offsets = torch.from_numpy(array.offsets.to_numpy()).to(device)
    seq_len = offsets[1:] - offsets[:-1]
    n_rows = len(seq_len)

    drop_count = (seq_len - self.max_length).clamp(min=0)
    seq_len = seq_len.clamp(max=self.max_length)
    values = values[_tiled_mask(seq_len, drop_count)]

    padding_len = self.max_length - seq_len
    result = torch.full((n_rows * self.max_length,), float("NaN"), device=device)
    result[_tiled_mask(seq_len, padding_len)] = values
    return MultiEmbeddingTensor(
        num_rows=n_rows,
        num_cols=1,
        values=result.view(n_rows, self.max_length),
        offset=torch.tensor([0, self.max_length], device=device),
    )


def _assert_identical(ref: MultiEmbeddingTensor, out: MultiEmbeddingTensor) -> None:
    assert type(out) is MultiEmbeddingTensor
    assert (out.num_rows, out.num_cols) == (ref.num_rows, ref.num_cols)
    assert out.values.dtype == ref.values.dtype
    assert out.values.shape == ref.values.shape
    assert out.values.device == ref.values.device
    bits = torch.int64 if ref.values.dtype == torch.float64 else torch.int32
    assert torch.equal(
        out.values.contiguous().view(bits), ref.values.contiguous().view(bits)
    )
    assert torch.equal(out.offset, ref.offset)


L = 4

ROWS: dict[str, list] = {
    "all_full": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
    "full_null_empty": [[1.0, 2.0, 3.0, 4.0], None, [], [5.0, float("nan"), 7.0, 8.0]],
    "uniform_longer": [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        None,
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ],
    "uniform_shorter": [[1.0, 2.0], [], [3.0, 4.0], None],
    "ragged": [[1.0, 2.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0], None, []],
    "value_nulls": [[1.0, None, 3.0, 4.0], None, [None, None, None, None]],
    "all_null": [None, None, None],
    "empty": [],
}


def _cases() -> list[tuple[str, pa.Array | pa.ChunkedArray]]:
    cases: list[tuple[str, pa.Array | pa.ChunkedArray]] = []
    for list_type in [pa.list_, pa.large_list]:
        for value_type in [pa.float32(), pa.float64()]:
            typ = list_type(value_type)
            tag = f"{typ}"
            for name, rows in ROWS.items():
                cases.append((f"{tag}-{name}", pa.array(rows, type=typ)))
            full = pa.array(ROWS["full_null_empty"], type=typ)
            cases.append(
                (
                    f"{tag}-chunked",
                    pa.chunked_array(
                        [
                            full,
                            pa.array([], type=typ),
                            pa.array(ROWS["all_full"], type=typ),
                        ]
                    ),
                )
            )
            cases.append((f"{tag}-sliced_mid", full.slice(1, 2)))
            cases.append(
                (
                    f"{tag}-sliced_after_empty",
                    pa.array([None, [], [1.0, 2.0, 3.0, 4.0]], type=typ).slice(1),
                )
            )
            cases.append((f"{tag}-sliced_head", full.slice(0, 2)))
            cases.append(
                (
                    f"{tag}-null_slot_with_values",
                    type(full).from_arrays(
                        pa.array([0, 4, 8, 12], type=full.offsets.type),
                        pa.array(np.arange(12), type=value_type),
                        mask=pa.array([False, True, False]),
                    ),
                )
            )
    return cases


CASES = _cases()


@pytest.fixture(params=[torch.float32, torch.float64], ids=str)
def default_dtype(request):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    yield request.param
    torch.set_default_dtype(prev)


@pytest.mark.parametrize("arr", [c[1] for c in CASES], ids=[c[0] for c in CASES])
@pytest.mark.parametrize("max_length", [L, 2, 1, 0])
def test_from_arrow_matches_reference(arr, max_length, default_dtype) -> None:
    ref_mapper = SequenceTensorMapper(min_length=0, max_length=max_length)
    mapper = SequenceTensorMapper(min_length=0, max_length=max_length)
    ref = _ref_from_arrow(ref_mapper, arr)
    out = mapper(arr)
    _assert_identical(ref, out)
    assert mapper._is_string_dtype is ref_mapper._is_string_dtype


POST_CASES = [c for c in CASES if isinstance(c[1], pa.ListArray)]


@pytest.mark.parametrize(
    "arr", [c[1] for c in POST_CASES], ids=[c[0] for c in POST_CASES]
)
@pytest.mark.parametrize("max_length", [L, 2, 1, 0])
def test_post_forward_matches_reference(arr, max_length, default_dtype) -> None:
    mapper = SequenceTensorMapper(min_length=0, max_length=max_length)
    try:
        ref = _ref_post_forward(mapper, arr)
    except (pa.ArrowInvalid, IndexError, RuntimeError) as error:
        with pytest.raises(type(error)):
            mapper.post_forward(arr)
        return
    _assert_identical(ref, mapper.post_forward(arr))


def test_output_does_not_alias_arrow_buffer() -> None:
    # Zero-copy dtype.
    arr = pa.array(ROWS["all_full"], type=pa.list_(pa.float32()))
    mapper = SequenceTensorMapper(min_length=0, max_length=L)
    for out in (mapper(arr), mapper.post_forward(arr)):
        out.values.fill_(0.0)
        assert arr.to_pylist() == ROWS["all_full"]


@pytest.mark.parametrize(
    "lens, spans, expected",
    [
        ([4, 4, 4], True, "dense"),
        ([6, 6, 0], True, "uniform"),
        ([2, 2], True, "uniform"),
        ([4, 4, 4], False, "ragged"),
        ([2, 4], True, "ragged"),
        ([], True, "ragged"),
    ],
)
def test_row_path_choice(lens, spans, expected) -> None:
    seq_len = np.array(lens, dtype=np.int64)
    row_len = int(seq_len.max()) if len(lens) else 0
    assert _row_path(seq_len, row_len, L, spans) == expected


def test_zero_rows_do_not_raise() -> None:
    mapper = SequenceTensorMapper(min_length=0, max_length=L)
    out = mapper._pad_or_truncate(np.empty(0), np.zeros(1, dtype=np.int64), None)
    assert out.values.shape == (0, L)


FIXED_ROWS = {
    "full": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    "longer": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
    "shorter": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    "nan_values": [[1.0, float("nan"), 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    "value_nulls": [[1.0, None, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    "null_row": [[1.0, 2.0, 3.0, 4.0], None, [5.0, 6.0, 7.0, 8.0]],
    "empty": [],
}


@pytest.mark.parametrize("value_type", [pa.float32(), pa.float64()], ids=str)
@pytest.mark.parametrize("name", list(FIXED_ROWS))
@pytest.mark.parametrize("max_length", [L, 2, 1, 0])
def test_fixed_size_list_matches_pandas_path(
    name, value_type, max_length, default_dtype
) -> None:
    rows = FIXED_ROWS[name]
    width = len(next((r for r in rows if r is not None), [0] * L))
    arr = pa.array(rows, type=pa.list_(value_type, width))
    chunked = pa.chunked_array([arr.slice(0, 1), arr.slice(1)])
    for a in (arr, chunked, arr.slice(1)):
        ref_mapper = SequenceTensorMapper(min_length=0, max_length=max_length)
        mapper = SequenceTensorMapper(min_length=0, max_length=max_length)
        ref = sequence_pandas_reference(ref_mapper, a.to_pandas())
        _assert_identical(ref, mapper(a))


def test_fixed_size_list_output_does_not_alias_arrow_buffer() -> None:
    arr = pa.array(FIXED_ROWS["full"], type=pa.list_(pa.float32(), L))
    mapper = SequenceTensorMapper(min_length=0, max_length=L)
    mapper(arr).values.fill_(0.0)
    assert arr.to_pylist() == FIXED_ROWS["full"]


@pytest.mark.parametrize(
    "value_type, rows, expected",
    [
        (
            pa.string(),
            [["0", None], ["1", "2.5"], None],
            [[0.0, np.nan], [1.0, 2.5], [np.nan, np.nan]],
        ),
        (
            pa.large_string(),
            [["0", None], ["1", "2.5"], None],
            [[0.0, np.nan], [1.0, 2.5], [np.nan, np.nan]],
        ),
        (
            pa.string(),
            [[" 0 ", None], [" +1e0 ", " 2.5 "], None],
            [[0.0, np.nan], [1.0, 2.5], [np.nan, np.nan]],
        ),
        (
            pa.bool_(),
            [[True, None], [False, True], None],
            [[1.0, np.nan], [0.0, 1.0], [np.nan, np.nan]],
        ),
        (pa.null(), [[None, None], [None, None], None], [[np.nan, np.nan]] * 3),
    ],
    ids=[
        "numeric-strings",
        "large-numeric-strings",
        "numpy-numeric-strings",
        "nullable-booleans",
        "null-leaves",
    ],
)
@pytest.mark.parametrize("layout", ["list", "large-list", "fixed-size-list"])
@pytest.mark.parametrize("max_length", [1, 2, 3])
@pytest.mark.usefixtures("default_dtype")
def test_sequence_leaves_are_coerced_to_floats(
    value_type: pa.DataType,
    rows: list[list[str | bool | None] | None],
    expected: list[list[float]],
    layout: str,
    max_length: int,
) -> None:
    """Numeric leaves preserve nulls, row boundaries, and the default dtype."""
    if layout == "large-list":
        array_type = pa.large_list(value_type)
    else:
        array_type = pa.list_(value_type, 2 if layout == "fixed-size-list" else -1)
    array = pa.array(rows, type=array_type)
    expected_values = torch.full((3, max_length), float("nan"))
    width = min(2, max_length)
    expected_values[:, :width] = torch.tensor(expected)[:, :width]
    sources = (
        (array, expected_values),
        (array.slice(0, 2), expected_values[:2]),
        (array.slice(1), expected_values[1:]),
        (pa.chunked_array([array[:1], array[1:]]), expected_values),
    )

    for source, expected_rows in sources:
        result = SequenceTensorMapper(0, max_length)(source)

        torch.testing.assert_close(result.values, expected_rows, equal_nan=True)
        assert result.values.dtype == torch.get_default_dtype()
        assert result.offset.tolist() == [0, max_length]


def test_sequence_rejects_nonnumeric_string_leaves() -> None:
    """Invalid numeric leaves raise a parsing error instead of losing data."""
    array = pa.array([["not-a-number"]])

    with pytest.raises(ValueError, match="could not convert string to float"):
        SequenceTensorMapper(0, 3)(array)


@pytest.mark.parametrize("value_type", [pa.string(), pa.large_string()], ids=str)
@pytest.mark.usefixtures("default_dtype")
def test_string_encoded_sequences_use_numpy_parsing(value_type: pa.DataType) -> None:
    """String-encoded sequences retain NumPy whitespace and exponent parsing."""
    array = pa.array(["[ 1 , +2.5e0 ]", None, "[]"], type=value_type)
    expected = torch.tensor(
        [[1.0, 2.5, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    )

    result = SequenceTensorMapper(0, 3)(array)

    torch.testing.assert_close(result.values, expected, equal_nan=True)
