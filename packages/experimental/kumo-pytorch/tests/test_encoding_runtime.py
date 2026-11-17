# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derived encoder caches preserve values, errors, and portable fitted state."""

import pickle

import pyarrow as pa
import pytest
import torch

from kumo.pytorch.contract.batch import Vocabulary
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.encoding import FittedSpace, fit_encoder


@pytest.mark.parametrize("physical", [pa.string(), pa.large_string()])
def test_cached_string_lookup_keeps_null_unknown_and_sparse_ids(physical):
    diskgraph = pytest.importorskip("diskgraph")
    encoder = fit_encoder(
        "category",
        Vocabulary(values=(("a", 7), ("café", 20), ("", 99))),
        DType("string"),
        (),
    )
    values = pa.array(["prefix", "a", "new", None, "café", "", "suffix"], type=physical)
    values = values.slice(1, 5)
    written = encoder.state_dict()
    builds = []

    def compile_lookup(keys):
        builds.append(keys)
        return diskgraph.StringLookup(keys).lookup

    for _ in range(2):
        encoder._cache_lookup(compile_lookup)
        encoded = encoder.encode_arrow(values)
        assert encoded["values"].tolist() == [7, 1, 0, 20, 99]
        assert encoded["valid"].tolist() == [True, True, False, True, True]
    assert len(builds) == 1
    assert encoder.state_dict() == written
    restored = pickle.loads(pickle.dumps(encoder))
    assert restored.state_dict() == written
    # Compiled native objects are process-local; restoring state needs no binary.
    assert "_lookup" not in restored.__dict__
    assert restored.encode_arrow(values)["values"].tolist() == [7, 1, 0, 20, 99]


def test_remap_cache_keeps_members_separate_and_outputs_independent():
    space = FittedSpace(roles={}, members={"a": {2: 9, 7: 4}, "b": {2: 3, 7: 8}}, size=10)
    written = space.state_dict()
    values = torch.tensor([[7, 2], [2, 7]])
    for _ in range(2):
        actual = space.remap("a", values)
        assert actual.tolist() == [[4, 9], [9, 4]]
        actual.fill_(-1)
        assert space.remap("b", values).tolist() == [[8, 3], [3, 8]]
        with pytest.raises(ValueError, match="no allocation"):
            space.remap("a", torch.tensor([3]))
    assert space.state_dict() == written
    restored = pickle.loads(pickle.dumps(space))
    assert restored.state_dict() == written
    assert "_remap_tables" not in restored.__dict__
    assert torch.equal(restored.remap("a", values), space.remap("a", values))
    assert space.remap("a", torch.empty((2, 0), dtype=torch.long)).shape == (2, 0)


def test_diskgraph_rebuilds_runtime_lookups_from_portable_fitted_state(monkeypatch):
    pytest.importorskip("diskgraph")
    from conformance.assertions import bound_tables
    from kumo.pytorch import api
    from kumo.pytorch.engines.diskgraph import DiskGraphSource
    from tests.test_field_outputs import _contract, _tables
    from tests.test_inference_lifecycle import assert_same_batches

    builds = []
    original = DiskGraphSource.string_lookup

    def compile_lookup(source, keys):
        builds.append(keys)
        return original(source, keys)

    monkeypatch.setattr(DiskGraphSource, "string_lookup", compile_lookup)
    contract = _contract(flat=True, masking=True)
    requests = [api.SampleRequest("history", 1, 40), api.SampleRequest("empty", 2, 40)]
    with (
        bound_tables(contract, _tables(), "diskgraph") as (dataset, engine),
        api.prepare(contract, dataset, engine) as plan,
    ):
        with plan.materialize(requests) as task:
            written = task.fitted_state.to_dict()
            expected = list(task.batches(batch_size=2, output_format="model"))
            assert_same_batches(list(task.batches(batch_size=2, output_format="model")), expected)
            assert task.fitted_state.to_dict() == written
            fitted = pickle.loads(pickle.dumps(task.fitted_state))
        assert len(builds) == 1
        with plan.materialize(requests, fitted=fitted) as task:
            assert_same_batches(list(task.batches(batch_size=2, output_format="model")), expected)
            assert task.fitted_state.to_dict() == written
        assert len(builds) == 2
