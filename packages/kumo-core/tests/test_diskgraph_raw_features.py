# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw feature normalization preserves Arrow chunks and sampled row order."""

import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.runtime_adapter import (
    RawFeatureBlock,
    _collect_raw_features,
    runtime_batch_to_hetero,
)
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper
from kumoapi.typing import Stype


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_collect_raw_features_preserves_arrow_buffers(
    single: bool, empty: bool
) -> None:
    """Normalize both native shapes to tables without copying column buffers."""
    array = pa.array([], type=pa.int64()) if empty else pa.array([3, 1, 2])
    batches = [pa.record_batch([array], names=["value"])]
    if not single:
        batches.append(pa.record_batch([array], names=["value"]))
    block: RawFeatureBlock = {
        "record_batches": batches[0] if single else batches,
    }
    result = _collect_raw_features({"items": block}, {})
    table = result.tables["items"]
    assert isinstance(table, pa.Table)
    assert table.num_rows == len(array) * len(batches)
    assert table.column("value").num_chunks == len(batches)
    for chunk in table.column("value").chunks:
        assert chunk.buffers() == array.buffers()
        assert chunk.buffers()[1].address == array.buffers()[1].address


def test_runtime_raw_feature_chunks_encode_in_sample_order() -> None:
    """The runtime exposes tables and encodes matching features and row IDs."""
    batches = [
        pa.record_batch({"value": [3.0, None]}),
        pa.record_batch({"value": [1.0, 2.0]}),
    ]
    ids = pa.array([8, 4, 9, 2])
    hetero, tables, row_ids = runtime_batch_to_hetero(
        {
            "task": {"input_ids": pa.array([0]), "seed_node_ids": pa.array([8])},
            "raw_features": {"items": {"record_batches": batches, "node_ids": ids}},
        },
        train_table_name="seeds",
        mapper_bank={"items": {"value": NumericalTensorMapper()}},
        feature_schema={"node_types": {"items": {"mapper_key": "items"}}},
    )
    assert tables["items"].column("value").num_chunks == 2
    assert row_ids["items"].tolist() == ids.to_pylist()
    actual = hetero["items"].tf.feat_dict[Stype.numerical].flatten()
    assert actual.allclose(torch.tensor([3.0, float("nan"), 1.0, 2.0]), equal_nan=True)
