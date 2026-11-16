# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit encoder kernels and task-level token state remain deterministic."""

from dataclasses import replace
from unittest.mock import patch

import pyarrow as pa
import pytest
import torch
from torch.utils._pytree import tree_flatten

from kumo.pytorch.api import Buckets, DataLoaderSpec, DType, Encode, Fields, Text, Vocabulary
from kumo.pytorch.encoding import fit_encoder
from tests.helpers import written_state
from tests.reference_models.batch_definitions import PRAGMA_BATCH
from tests.reference_models.fixtures import adapted_batch, model_task


@pytest.mark.parametrize(
    "dtype,config,values",
    (
        (DType(name="float64"), Buckets(bins=8), [12.5, 100.0, 13.0, 25.0]),
        (DType(name="string"), Vocabulary(), ["USD", "EUR", "USD"]),
        (DType(name="string"), Text(vocabulary_size=64), ["monthly music plan", "payroll deposit"]),
    ),
)
def test_column_encoding_matches_record_encoding(dtype, config, values):
    records = fit_encoder(identity="field", config=config, dtype=dtype, values=values)
    column = pa.array(values, type=dtype.to_arrow())
    columns = fit_encoder(identity="field", config=config, dtype=dtype, values=column.to_pylist())
    assert columns.entries == records.entries
    assert columns.boundaries == records.boundaries
    assert columns.tokenizer_json == records.tokenizer_json
    actual = columns.encode_arrow(column)
    expected = records.encode_values(values)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], expected[name]) for name in actual)


@pytest.mark.parametrize("values", [[], [""], ["monthly"]])
def test_text_encoding_has_integer_piece_offsets(values):
    # no values: a batch whose samples read no text rows (one serving request with no history)
    dtype = DType(name="string")
    encoder = fit_encoder("field", Text(vocabulary_size=64), dtype, ["monthly music plan"])
    encoded = encoder.encode_arrow(pa.array(values, type=dtype.to_arrow()))
    offsets = encoded["piece_offsets"]
    assert offsets.dtype == torch.int64
    assert offsets[0] == 0 and len(offsets) == len(values) + 1
    assert offsets[-1] == len(encoded["values"])


def test_bound_input_encoders_round_trip_and_encoding(tmp_path):
    with model_task("pragma") as task:
        original = next(task.batches(batch_size=2, seed=1729))
        task.to_artifact().save(tmp_path / "encoders")
        assert written_state(tmp_path / "encoders") == task.fitted_state.to_dict()
        bound = DataLoaderSpec.from_json(task.bound_contract().to_json())
        assert bound == task.bound_contract()
        encoded = adapted_batch(task, "pragma")
        assert encoded.history_elapsed_seconds.tolist() == [[0.0, 25.0, 20.0, 5.0, 0.0]] * 2
        assert encoded.token_count > 0
    with model_task("pragma", batch=bound.batch, sample=bound.sample, schema=bound.schema) as task:
        restored = next(task.batches(batch_size=2, seed=1729))
        for name in ("entity", "history"):
            assert torch.equal(
                restored.inputs[name]["value_ids"], original.inputs[name]["value_ids"]
            )


def test_forbidden_payload_is_rejected_without_reading_rows():
    with pytest.raises(ValueError, match="forbidden"):
        Fields(
            fields=(Encode(field="event_payload", encoding=Text()),),
            forbidden_fields=("event_payload",),
        )


def test_masking_and_batching_are_deterministic():
    target = replace(
        PRAGMA_BATCH.targets["masked_values"],
        token_probability=0.5,
        event_probability=0.0,
        key_probability=0.0,
        unknown_probability=0.0,
    )
    policy = replace(PRAGMA_BATCH, targets={"masked_values": target})
    with model_task("pragma", batch=policy) as task:
        first = next(task.batches(batch_size=2, seed=7))
        second = next(task.batches(batch_size=2, seed=7))
        assert first.targets["masked_values"]["valid"].any()
        assert not first.targets["masked_values"]["valid"].all()
        assert torch.equal(
            first.inputs["history"]["value_ids"], second.inputs["history"]["value_ids"]
        )
        assert torch.equal(
            first.targets["masked_values"]["values"], second.targets["masked_values"]["values"]
        )
        assert first.inputs["history"]["sequence"]["event_indices"].shape == (2, 4)


def test_masking_reused_inputs_matches_fresh_fit_without_refitting_encoders():
    target = replace(
        PRAGMA_BATCH.targets["masked_values"],
        token_probability=0.4,
        event_probability=0.2,
        key_probability=0.3,
        unknown_probability=0.25,
    )
    policy = replace(PRAGMA_BATCH, targets={"masked_values": target})
    with model_task("pragma", targets=False) as task:
        saved = task.fitted_state.inputs
    with model_task("pragma", batch=policy) as task:
        expected = next(task.batches(batch_size=2, seed=7, epoch=3))
    with (
        patch(
            "kumo.pytorch.assembly.fit_encoder",
            side_effect=AssertionError("masking refitted an input"),
        ),
        model_task("pragma", batch=policy, fitted=saved) as task,
    ):
        actual = next(task.batches(batch_size=2, seed=7, epoch=3))
    actual_values, actual_structure = tree_flatten((actual.inputs, actual.targets))
    expected_values, expected_structure = tree_flatten((expected.inputs, expected.targets))
    assert actual_structure == expected_structure
    for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
        if isinstance(expected_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value
