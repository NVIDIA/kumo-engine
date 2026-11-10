# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw sampling features retain Arrow chunks through tensor encoding."""

from unittest.mock import Mock

import pyarrow as pa
import pytest
from kumo_core.diskgraph import runtime_adapter
from kumo_core.diskgraph.runtime_adapter import (
    RawFeatureBlock,
    _collect_raw_features,
    attach_tensor_frames,
)
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper
from kumoapi.typing import Stype
from torch_geometric.data import HeteroData


@pytest.mark.parametrize("single_batch", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_collect_raw_features_keeps_arrow_buffers(
    single_batch: bool, empty: bool
) -> None:
    """Wrap batches as tables without copying buffers or losing empty schemas."""
    batches = [pa.record_batch({"age": [1.0, None]}), pa.record_batch({"age": [3.0]})]
    if empty:
        batches = [batch.slice(0, 0) for batch in batches]
    block: RawFeatureBlock = {"record_batches": batches[0] if single_batch else batches}

    result = _collect_raw_features({"User": block}, {})

    table = result.tables["User"]
    assert isinstance(table, pa.Table)
    originals = batches[:1] if single_batch else batches
    assert table.num_rows == sum(batch.num_rows for batch in originals)
    assert table.schema == batches[0].schema
    assert table.column("age").num_chunks == len(originals)
    for chunk, batch in zip(table.column("age").chunks, originals):
        assert [
            buffer.address if buffer is not None else None for buffer in chunk.buffers()
        ] == [
            buffer.address if buffer is not None else None
            for buffer in batch.column("age").buffers()
        ]


def test_raw_feature_table_encodes_all_batches() -> None:
    """Feature attachment encodes all rows in their original batch order."""
    block: RawFeatureBlock = {
        "record_batches": [
            pa.record_batch({"age": [1.0]}),
            pa.record_batch({"age": [2.0]}),
        ],
        "node_ids": pa.array([10, 20], type=pa.uint64()),
    }
    raw = _collect_raw_features({"User": block}, {})
    hetero = HeteroData()

    attach_tensor_frames(
        hetero,
        raw.tables,
        {"User": {"age": NumericalTensorMapper()}},
        {"node_types": {"User": {"mapper_key": "User"}}},
    )

    assert hetero["User"].tf.feat_dict[Stype.numerical].tolist() == [[1.0], [2.0]]
    assert raw.row_ids["User"].tolist() == [10, 20]
    assert raw.tables["User"].column("age").num_chunks == 2


def test_missing_raw_features_use_an_empty_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsampled node types pass an empty Arrow table to the feature mapper."""
    table_to_tf = Mock(wraps=runtime_adapter._table_to_tf)
    monkeypatch.setattr(runtime_adapter, "_table_to_tf", table_to_tf)
    hetero = HeteroData()
    hetero["User"].num_nodes = 0

    attach_tensor_frames(
        hetero,
        {},
        {"User": {"age": NumericalTensorMapper()}},
        {"node_types": {"User": {"mapper_key": "User"}}},
    )

    table_to_tf.assert_called_once()
    table = table_to_tf.call_args.args[0]
    assert isinstance(table, pa.Table)
    assert table.num_rows == 0
    assert table.column_names == ["age"]
    assert hetero["User"].tf.feat_dict[Stype.numerical].shape == (0, 1)
