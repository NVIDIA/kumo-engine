# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch request creation preserves scalar-reader values and contract identity."""

from dataclasses import replace
from types import MappingProxyType
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.anchors import PositionalAnchors
from kumo.pytorch.engines.parquet_anchors import ParquetAnchors
from tests.test_contract_serialization import _contract


def test_immutable_fingerprints_are_computed_once_and_replace_invalidates():
    contract, _ = _contract()
    with patch.object(api.DataLoaderSpec, "to_dict", autospec=True, wraps=None) as encode:
        encode.side_effect = lambda spec: original(spec)
        _ = contract.fingerprint
        _ = contract.fingerprint
        assert encode.call_count == 1
    changed = replace(
        contract, batch=replace(contract.batch, batching=api.Batching(max_samples=17))
    )
    assert changed.fingerprint != contract.fingerprint
    assert changed.input_fingerprint == contract.input_fingerprint
    assert "_fingerprint" not in contract.to_json()


original = api.DataLoaderSpec.to_dict


def test_accepted_mutable_sequences_do_not_get_stale_fingerprints():
    contract, _ = _contract()
    relation = api.RelationSpec("friend", "users", "id", "users", "id")
    schema = replace(contract.schema, relations=(relation,))
    layers = [api.GraphLayer((api.GraphSample("friend", 1),))]
    sample = replace(contract.sample, neighborhoods={"g": api.NeighborhoodSpec(layers)})
    mutable = replace(contract, schema=schema, sample=sample)
    first = mutable.fingerprint
    first_inputs = mutable.input_fingerprint
    layers.append(api.GraphLayer((api.GraphSample("friend", 2),)))
    assert mutable.fingerprint != first
    assert mutable.input_fingerprint != first_inputs


@pytest.mark.parametrize("selection", [api.ALL_ANCHORS, frozenset({"train", None})])
@pytest.mark.parametrize("explicit_id", [False, True])
def test_arrow_batch_fetch_matches_scalar_reader(tmp_path, selection, explicit_id):
    path = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [7, 2, 9, 4],
                "sid": ["a", "b", "c", "d"],
                "at": [10, 20, 30, 40],
                "split": ["train", None, "valid", "train"],
                "target": [1.0, None, 2.0, 3.0],
                "text": ["é", None, "", "x"],
            }
        ),
        path,
    )
    reader = ParquetAnchors(
        api.AnchorTable(
            source=api.Parquet(str(path), columns={"id": "entity"}),
            entity_id="entity",
            sample_id="sid" if explicit_id else None,
            as_of_time="at",
            split="split",
            targets={"y": "target"},
            metadata=("text",),
        )
    )
    for positions in ([2, 0, 2, -1, 1], [], [0]):
        expected = PositionalAnchors.fetch(reader, selection, positions)
        assert reader.fetch(selection, positions) == expected
    with pytest.raises(IndexError):
        reader.fetch(selection, [99])
    with pytest.raises(ValueError, match="no anchors in split"):
        reader.fetch(frozenset({"unknown"}), [])


def test_mutable_filter_values_are_not_cached():
    contract, _ = _contract()
    allowed = [1]
    with pytest.raises(ValueError, match="explicit sequence"):
        api.RowFilter("id", "in", MappingProxyType({"1": True}))
    contract = replace(
        contract,
        sample=replace(
            contract.sample,
            entity=replace(
                contract.sample.entity,
                filters=(api.RowFilter("id", "in", allowed),),
            ),
        ),
    )
    contract.validate()
    before = contract.fingerprint, contract.input_fingerprint
    allowed.append(2)
    assert (contract.fingerprint, contract.input_fingerprint) != before
    restored = api.DataLoaderSpec.from_json(contract.to_json())
    assert (contract.fingerprint, contract.input_fingerprint) == (
        restored.fingerprint,
        restored.input_fingerprint,
    )
