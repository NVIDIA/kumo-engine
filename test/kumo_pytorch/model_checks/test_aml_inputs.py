from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import kumo.pytorch.api as r

from test.kumo_pytorch.aml_inputs import SECOND, requests_for, select_anchors
from test.kumo_pytorch.contracts import build_contract, counts
from .fixtures import build_fixture


def anchor_file(tmp_path, *, damage=None):
    stamps = [
        int(datetime(2022, 9, day, tzinfo=timezone.utc).timestamp()) * SECOND
        for day in (6, 7, 8, 9)
    ]
    table = pa.table(
        {
            "sample_id": ["excluded", "fit", "valid", "test"],
            "entity_id": ["leading-0007"] * 4,
            "as_of_time": pa.array(stamps, type=pa.timestamp("ns")),
            "graph_as_of_time": pa.array(
                [t - SECOND for t in stamps], type=pa.timestamp("ns")
            ),
            "split": ["train", "train", "valid", "test"],
            "forecast_split": [None, "train", "valid", "test"],
            "involvement": pa.array([None, None, 1, 0], type=pa.int32()),
        }
    )
    if damage:
        table = damage(table)
    path = tmp_path / "anchors.parquet"
    pq.write_table(table, path, row_group_size=2)
    return path


def test_selection_keeps_original_order_null_targets_and_split_rules(tmp_path):
    path = anchor_file(tmp_path)
    selected = select_anchors(
        path, split_columns=("split", "forecast_split"), limit=1, graph=True
    )
    assert selected["split"]["train"][0]["sample_id"] == "excluded"
    fit = selected["forecast_split"]["train"][0]
    assert fit["sample_id"] == "fit" and fit["physical_row_position"] == 1
    assert fit["involvement"] is None
    request = requests_for(
        [fit], {"leading-0007": 42}, split="train", graph=True, native=False
    )[0]
    assert request.sample_id == "fit" and request.entity_id == 42
    assert request.targets[0].value is None
    assert request.as_of_time == fit["as_of_time"] // SECOND - 1
    assert request.metadata["prediction_time_ns"] == fit["as_of_time"]
    assert request.metadata["registered_entity_id"] == "leading-0007"
    with pytest.raises(ValueError, match="relabel"):
        requests_for(
            [fit], {"leading-0007": 42}, split="test", graph=True, native=False
        )
    native = requests_for(
        [fit], {"leading-0007": 42}, split="train", graph=False, native=True
    )[0]
    assert native.targets == () and native.as_of_time == fit["as_of_time"] // SECOND


def test_wrong_v2_clock_is_rejected(tmp_path):
    path = anchor_file(
        tmp_path,
        damage=lambda table: table.set_column(
            table.schema.get_field_index("graph_as_of_time"),
            "graph_as_of_time",
            table["as_of_time"],
        ),
    )
    with pytest.raises(ValueError, match="v2 graph cutoff"):
        select_anchors(path, split_columns=("forecast_split",), limit=1, graph=True)


def test_missing_required_split_is_a_failure(tmp_path):
    path = anchor_file(tmp_path, damage=lambda table: table.slice(0, 3))
    with pytest.raises(ValueError, match="test anchor"):
        select_anchors(path, split_columns=("forecast_split",), limit=1)


def test_combined_history_matches_unshifted_sequence_and_count_boundaries():
    fixture = build_fixture()
    originals = fixture.training
    shifted = tuple(
        replace(
            a,
            as_of_time=a.as_of_time - timedelta(seconds=1),
            metadata={"prediction_time_ns": int(a.as_of_time.timestamp()) * SECOND},
        )
        for a in originals
    )
    cutoff = max(a.as_of_time for a in shifted)
    histories = []
    for family, anchors in (("nuformer", originals), ("combined", shifted)):
        contract = build_contract(
            fixture.schema, family, max_events=len(fixture.tables["transactions"])
        )
        with r.prepare(contract, fixture.memory_dataset(), r.InMemoryEngine()) as plan:
            task = plan.materialize(anchors, fit_split="train", encoder_cutoff=cutoff)
            batch = next(task.batches(anchors, seed=0))
            histories.append(batch.inputs["history"])
            learned = [
                value
                for state in task.fitted_state["encoders"].values()
                for value, _ in state.get("entries", ())
            ]
            assert "wire" in learned and "new_format" not in learned
    for name in ("value_ids", "valid"):
        torch.testing.assert_close(
            histories[0][name], histories[1][name], rtol=0, atol=0
        )

    # Independently count [T-24h,T) in raw rows; events at T and after are excluded.
    contract = counts(
        fixture.schema,
        "accounts",
        [("count", ("account_touches",), "event_time", -86400)],
        normalize=False,
        inclusive=True,
    )
    anchors = tuple(replace(a, targets=()) for a in shifted)
    with r.prepare(contract, fixture.memory_dataset(), r.InMemoryEngine()) as plan:
        task = plan.materialize(anchors, fit_split="train", encoder_cutoff=cutoff)
        batch = next(task.batches(anchors, batch_size=2))
        expected = [
            sum(
                row["account_id"] == a.entity_id
                and a.as_of_time - timedelta(days=1) <= row["event_time"] < a.as_of_time
                for row in fixture.tables["account_events"]
            )
            for a in originals
        ]
        assert batch.inputs["aggregates"]["count"]["values"].tolist() == expected
