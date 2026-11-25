# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover logical feature shapes from a DiskGraph index and a fitted artifact."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from tests.test_existing_index import snapshot
from tests.test_type_resolution import _source_contract


@pytest.mark.parametrize("name,first", [("float32", -3.5), ("uint64", 2**64 - 1)])
def test_saved_shape_rebinds_index_and_raw_encoder(tmp_path, monkeypatch, name, first):
    import kumo.settings

    monkeypatch.setattr(kumo.settings, "_configured", tmp_path)
    dtype = api.DType(name, shape=(2, 2))
    spec = _source_contract({"id": api.DType("int64"), "value": dtype})
    spec = replace(spec, batch=replace(spec.batch, batching=api.Batching(max_samples=2)))
    source = tmp_path / "users.parquet"
    rows = [[first, 1, 2, 3], None]
    pq.write_table(
        pa.table(
            {"id": [4, 5], "value": pa.array(rows, type=pa.list_(dtype.element_dtype.to_arrow()))}
        ),
        source,
    )
    requests = (api.SampleRequest("full", 4, None), api.SampleRequest("empty", 5, None))
    dataset = api.Dataset({"users": api.Parquet(str(source))})
    engine = api.DiskGraphEngine(num_threads=1)
    with api.prepare(spec, dataset, engine) as plan, plan.materialize(requests) as task:
        task.to_artifact().save(tmp_path / "artifact")

    # No base Parquet reads can recover the shape once the original source is gone.
    source.unlink()
    artifact = api.Artifact.load(tmp_path / "artifact")
    rebound = api.DiskGraphEngine.from_snapshot(snapshot(engine.index), num_threads=1)
    assert rebound.schema == artifact.contract.schema == spec.schema
    with (
        api.prepare(contract=artifact.contract, engine=rebound) as plan,
        plan.materialize(requests, fitted=artifact.fitted_state) as task,
    ):
        item = next(task.request_batches(requests, output_format="model")).inputs["values"]["value"]
        assert item["valid"].tolist() == [True, False]
        assert item["values"].dtype == getattr(torch, name)
        assert item["values"].tolist() == [[[first, 1], [2, 3]], [[0, 0], [0, 0]]]
