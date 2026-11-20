# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LI Medium's four task meanings on tiny native fixtures, not real-data evidence."""

from decimal import Decimal
from test.kumo_pytorch.aml_inputs import SECOND
from test.kumo_pytorch.bindings import load_bindings
from test.kumo_pytorch.contracts import binding_contract
from test.kumo_pytorch.population_anchors import project_binding

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import kumo.pytorch.api as r
from kumo.pytorch.engines.native_anchors import NativeAnchors

from .fixtures import CUTOFF, build_fixture

LI_BINDINGS = ("L01", "L02", "L03", "L04")


def source_anchors(path):
    stamp = int(CUTOFF.timestamp()) * SECOND
    pq.write_table(
        pa.table(
            {
                "sample_id": [
                    "excluded",
                    "train-0",
                    "train-1",
                    "train-2",
                    "valid-0",
                    "test-0",
                ],
                "entity_id": ["account-0001"] * 6,
                "as_of_time": pa.array([stamp] * 6, type=pa.timestamp("ns")),
                "graph_as_of_time": pa.array(
                    [stamp - SECOND] * 6, type=pa.timestamp("ns")
                ),
                # Supervised tasks must select forecast_split, never this column.
                "split": ["train"] * 6,
                "forecast_split": [None, "train", "train", "train", "valid", "test"],
                "involvement": pa.array([1, 1, 0, None, 1, None], type=pa.int32()),
                "next_amount": pa.array(
                    [
                        Decimal("999"),
                        Decimal("-2.5000000001"),
                        None,
                        Decimal("0"),
                        Decimal("1.25"),
                        None,
                    ],
                    type=pa.decimal128(38, 10),
                ),
                "next_currency": ["EXCLUDED", "USD", "EUR", None, "JPY", None],
                "next_delay_seconds": [999.0, 0.0, 2.5, None, 86400.0, None],
                "next_format": ["EXCLUDED", "wire", "ach", None, "card", None],
            }
        ),
        path,
        row_group_size=2,
    )
    return path


@pytest.mark.parametrize("ref", LI_BINDINGS)
def test_native_binding_targets_masks_and_train_only_fitting(
    tmp_path, monkeypatch, ref
):
    binding = load_bindings()[ref]
    fixture = build_fixture()
    dataset = fixture.parquet_dataset(tmp_path / "source")
    anchor_path = source_anchors(tmp_path / "original.parquet")
    projected = tmp_path / "projected.parquet"
    anchors, report = project_binding(
        anchor_path, projected, {"account-0001": 1}, binding
    )
    assert report["counts"] == {"train": 3, "valid": 1, "test": 1}
    assert report["mode"] == "full"  # Full tiny fixture, not the accepted population.
    assert report["binding_id"] == binding["binding_id"]
    rows = pq.read_table(projected).to_pylist()
    assert [row["sample_id"] for row in rows] == [
        "train-0",
        "train-1",
        "train-2",
        "valid-0",
        "test-0",
    ]
    assert [row["anchor_row_position"] for row in rows] == [1, 2, 3, 4, 5]
    graph = ref in {"L01", "L03"}
    assert all(
        row["as_of_time"] == int(CUTOFF.timestamp()) - int(graph) for row in rows
    )
    assert all(
        row["prediction_time_ns"] == int(CUTOFF.timestamp()) * SECOND for row in rows
    )
    if ref == "L01":
        assert [row["amount"] for row in rows] == [-2.5, None, 0.0, 1.25, None]
        assert [row["currency"] for row in rows] == ["USD", "EUR", None, "JPY", None]

    calls = []
    original = NativeAnchors._typed_batches

    def typed(owner, *args, **kwargs):
        calls.append(True)
        yield from original(owner, *args, **kwargs)

    monkeypatch.setattr(NativeAnchors, "_typed_batches", typed)
    contract = binding_contract(
        fixture.schema, binding, max_events=len(fixture.tables["transactions"])
    )
    engine = r.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with r.prepare(contract, dataset, engine) as plan:
        task = plan.materialize(
            anchors, fit_split="train", encoder_cutoff=report["encoder_cutoff_seconds"]
        )
        assert isinstance(task._anchors, NativeAnchors)
        state = task.fitted_state.to_dict()
        categories = {
            "L01": ("currency", {"EUR": 2, "USD": 3}),
            "L03": ("label", {"ach": 2, "wire": 3}),
        }
        if ref in categories:
            alias, expected = categories[ref]
            encoder = state["encoders"][state["targets"][alias]["bindings"][0][1]]
            assert dict(encoder["entries"]) == expected
            assert encoder["roles"] == {"missing": 0, "unknown": 1}

        expected_values = {
            "L01": {"amount": [-2.5, 0.0, 0.0, 1.25, 0.0], "currency": [3, 2, 0, 1, 0]},
            "L02": {"label": [0.0, 2.5, 0.0, 86400.0, 0.0]},
            "L03": {"label": [3, 2, 0, 1, 0]},
            "L04": {"label": [1.0, 0.0, 0.0, 1.0, 0.0]},
        }[ref]
        for epoch in (0, 1):
            batches = [
                batch
                for split in ("train", "valid", "test")
                for batch in task.batches(
                    split=split,
                    epoch=epoch,
                    batch_size=2,
                    shuffle=False,
                    output_format="model",
                )
            ]
            assert [s for b in batches for s in b.metadata["sample_ids"]] == [
                row["sample_id"] for row in rows
            ]
            assert all(set(b.targets) == set(expected_values) for b in batches)
            assert not any(b.metadata["is_padding"].any() for b in batches)
            for alias, values in expected_values.items():
                for batch in batches:
                    shape = (len(batch.metadata["sample_ids"]),)
                    assert batch.targets[alias]["values"].shape == shape
                    assert batch.targets[alias]["valid"].shape == shape
                actual = torch.cat([b.targets[alias]["values"] for b in batches])
                valid = torch.cat([b.targets[alias]["valid"] for b in batches])
                expected_valid = (
                    [True, False, True, True, False]
                    if alias == "amount"
                    else [True, True, False, True, False]
                )
                assert valid.dtype == torch.bool and valid.tolist() == expected_valid
                # Valid zeros remain eligible; UNKNOWN categories remain eligible.
                assert actual.dtype == (
                    torch.int64
                    if ref in categories and alias == categories[ref][0]
                    else torch.float32
                )
                torch.testing.assert_close(
                    actual, torch.tensor(values, dtype=actual.dtype), rtol=0, atol=0
                )
            if ref == "L04":
                assert "masked_values" not in batches[0].targets
                assert not any(
                    isinstance(t, r.MaskedValuesTarget)
                    for t in contract.batch.targets.values()
                )
        assert len(calls) == 6
        assert task.fitted_state.to_dict() == state
