# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw array tensors maintain independent value-shape and row-layout axes."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.encoding import fit_encoder


@pytest.mark.parametrize(
    "physical", [pa.list_(pa.int32()), pa.large_list(pa.int32()), pa.list_(pa.int32(), 4)]
)
def test_raw_array_encoding_handles_slices_chunks_and_nulls(physical):
    encoder = fit_encoder("image", api.Raw(), api.DType("int32", shape=(2, 2)), ())
    source = pa.array([[0] * 4, None, [1, 2, 3, 4], [5, 6, 7, 8]], type=physical)
    chunked = pa.chunked_array([source.slice(1, 2), source.slice(3, 1)])
    output = encoder.encode_arrow(chunked, model=True)
    assert output["values"].dtype == torch.int32
    assert output["values"].tolist() == [[[0, 0], [0, 0]], [[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert output["valid"].tolist() == [False, True, True]
    output["values"][1].fill_(99)
    assert source[2].as_py() == [1, 2, 3, 4]
    empty = encoder.encode_arrow(pa.chunked_array([], type=physical))
    assert empty["values"].shape == (0, 2, 2) and empty["valid"].shape == (0,)


def test_null_parent_hides_child_storage():
    encoder = fit_encoder("image", api.Raw(), api.DType("int64", shape=(2,)), ())
    src = pa.ListArray.from_arrays([0, 1, 3], [None, 5, 6], mask=pa.array([True, False]))
    output = encoder.encode_arrow(src)
    assert output["values"].tolist() == [[0, 0], [5, 6]]
    assert output["valid"].tolist() == [False, True]


def _contract(layout, encoding):
    dtype = api.DType("uint32", shape=(2, 2))
    table = api.TableSpec(
        "pins", "id", {"id": api.DType("int64"), "image": dtype}, time_column=api.NON_TEMPORAL
    )
    entity = api.EntitySpec("pins")
    values = {"image": api.Field(column="image")}
    sample = api.SampleSpec(
        entity,
        entity_values=values,
        candidate_sets={"positive": api.CandidateSetSpec(entity, values=values)},
    )
    fields = api.Fields((api.Encode("image", api.UseEncoding("shared")),))
    batch = api.BatchSpec(
        encodings={"shared": encoding},
        inputs={
            "query": api.Input(api.EntitySource(), fields, api.Separate()),
            "positive": api.Input(api.CandidateSource("positive"), fields, layout),
        },
    )
    return api.DataLoaderSpec(api.DataSchema((table,)), sample, batch)


@pytest.mark.parametrize("layout", [api.Packed(), api.Padded()])
def test_array_fields_keep_shape_through_candidate_layout(tmp_path, layout):
    spec = _contract(layout, api.Raw())
    source = tmp_path / "pins.parquet"
    pq.write_table(
        pa.table(
            {"id": [1, 2], "image": pa.array([[0, 1, 2, 3], None], type=pa.list_(pa.uint32()))}
        ),
        source,
    )
    dataset = api.Dataset({"pins": api.Parquet(str(source))})
    engine = api.DiskGraphEngine(num_threads=1)
    requests = [
        api.SampleRequest(
            "a", 1, 10, candidates={"positive": (api.CandidateRequest(2), api.CandidateRequest(1))}
        ),
        api.SampleRequest("b", 2, 10),
    ]
    with (
        api.prepare(spec, dataset, engine) as plan,
        plan.materialize(requests) as task,
    ):
        task.to_artifact().save(tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    with (
        api.prepare(artifact.contract, dataset, engine) as plan,
        plan.materialize(requests, fitted=artifact.fitted_state) as task,
    ):
        batch = next(task.request_batches(requests, batch_size=2))
    query = batch.inputs["query"]["image"]
    assert query["values"].dtype == torch.uint32
    assert query["values"].tolist() == [[[0, 1], [2, 3]], [[0, 0], [0, 0]]]
    assert query["valid"].tolist() == [True, False]
    positives = batch.inputs["positive"]["image"]
    if isinstance(layout, api.Packed):
        assert positives["values"].shape == (2, 2, 2)
        assert positives["valid"].tolist() == [False, True]
    else:
        assert positives["values"].shape == (2, 2, 2, 2)
        assert positives["valid"].tolist() == [[False, True], [False, False]]
        assert positives["values"][1].tolist() == [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    assert positives["values"].reshape(-1, 2, 2)[1].tolist() == [[0, 1], [2, 3]]


@pytest.mark.parametrize(
    "encoding", [api.ZScore(), api.Vocabulary(), api.Buckets(bins=2), api.Text()]
)
def test_array_inputs_require_raw_including_shared_encodings(encoding):
    spec = _contract(api.Packed(), encoding)
    with pytest.raises(ValueError, match="array features require Raw"):
        spec.validate()
    with pytest.raises(ValueError, match="array features require Raw"):
        fit_encoder("shared", encoding, api.DType("uint32", shape=(2, 2)), ())
