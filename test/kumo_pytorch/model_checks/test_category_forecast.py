# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native packed forecast coverage with independent labels and corruption checks."""

import hashlib
from copy import deepcopy
from test.kumo_pytorch.bindings import load_bindings, selection
from test.kumo_pytorch.contracts import binding_contract
from test.kumo_pytorch.data import prepare_binding, schema_for
from test.kumo_pytorch.forecast_features import category_features
from test.kumo_pytorch.model_checks import test_count_forecast as count_fixture
from test.kumo_pytorch.models import build_model
from test.kumo_pytorch.nightly import run_case
from test.kumo_pytorch.population_anchors import projected_rows

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import kumo.pytorch.api as r


def fixture(tmp_path):
    data, files, original = count_fixture.fixture(tmp_path)
    binding = load_bindings()["M03"]
    binding["dataset"] = original["dataset"]
    binding["dataset"]["source_options"]["event_identity_order"] = "timestamp"

    def write(key, table):
        path = data / key
        pq.write_table(table, path)
        files[key].update(
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    paths = binding["dataset"]["source_options"]["tables"]["transactions"]["paths"]
    for i, key in enumerate(paths):
        table = pq.ParquetFile(data / key).read()
        amount = table["amount"].to_pylist()
        amount[:4] = [-1.0, 0.0, 1.0, None]
        table = table.set_column(
            table.schema.get_field_index("amount"),
            "amount",
            pa.array(amount, pa.float64()),
        )
        if i != 1:
            for field in ["event_type", "event_subtype"]:
                values = table[field].to_pylist()
                values[6] = None if i == 2 and field == "event_type" else float(i + 1)
                table = table.set_column(
                    table.schema.get_field_index(field),
                    field,
                    pa.array(values, pa.float64()),
                )
        tied = table.slice(2, 1)
        for field, value in (("amount", -2.0), ("currency", 7.0), ("event_type", 6.0)):
            tied = tied.set_column(
                tied.schema.get_field_index(field),
                field,
                pa.array([value], pa.float64()),
            )
        write(key, pa.concat_tables([table, tied]))
    table = pq.read_table(data / "anchors.parquet")
    for field in ["event_type", "event_subtype"]:
        values = [1, None, None if field == "event_type" else 3, 4, 5]
        table = table.append_column(
            "next_" + field, pa.array(values, pa.int64())
        ).append_column(
            "next_" + field + "_valid", pa.array([v is not None for v in values])
        )
    # Same entity in two anchors must have the same matching group ID.
    duplicate = table.slice(0, 1).set_column(
        table.schema.get_field_index("sample_id"),
        "sample_id",
        pa.array(["repeat-client"]),
    )
    table = pa.concat_tables([table, duplicate])
    write("anchors.parquet", table)
    return data, files, binding


def prepared(tmp_path):
    data, files, binding = fixture(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    return binding, out, prepare_binding(data, out, binding, files, limit=None)


@pytest.mark.parametrize("mode", ["smoke", "training", "full", "lifecycle"])
def test_category_forecast_campaigns(tmp_path, mode):
    binding, out, (dataset, anchors, population, mapping, capacity) = prepared(tmp_path)
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        out,
        mode=mode,
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    if mode in {"smoke", "training"} and report.get("outcome") != "refused":
        assert report["feature_coverage"]["F16"] == {"train": 4, "valid": 1, "test": 1}
        assert all(
            v["status"] == "passed" for v in report["feature_assertions"].values()
        )


@pytest.fixture
def native(tmp_path):
    binding, out, (dataset, anchors, population, mapping, capacity) = prepared(tmp_path)
    contract = binding_contract(
        schema_for(dataset, binding), binding, max_events=capacity
    )
    with r.prepare(
        contract, dataset, r.DiskGraphEngine(index=out / "probe-index", num_threads=1)
    ) as plan:
        with plan.materialize(
            anchors,
            fit_split="train",
            encoder_cutoff=population["encoder_cutoff_seconds"],
        ) as task:
            batch = next(
                iter(
                    task.batches(
                        split="train",
                        epoch=0,
                        batch_size=4,
                        shuffle=False,
                        output_format="model",
                    )
                )
            )
            yield binding, out, mapping, task, batch


@pytest.mark.parametrize(
    "fault",
    [
        "composite",
        "offsets",
        "source_position",
        "chronology",
        "field",
        "aligned",
        "threshold",
        "aligned_mask",
        "matching",
        "matching_valid",
        "label",
        "label_valid",
        "leak",
    ],
)
def test_category_oracle_rejects_corruption(native, fault):
    binding, out, mapping, task, original = native
    check = category_features(
        binding,
        out / "anchors.parquet",
        mapping,
        task.fitted_state.to_dict(),
        mode="smoke",
    )
    check(original)
    batch = deepcopy(original)
    if fault == "composite":
        batch.inputs["history"]["value_ids"][0] += 1
    elif fault == "offsets":
        batch.inputs["history"]["offsets"][1] -= 1
    elif fault == "source_position":
        batch.inputs["history"]["source_event_indices"][0] += 1
    elif fault == "chronology":
        batch.inputs["history"]["value_ids"][:2] = batch.inputs["history"]["value_ids"][
            :2
        ].flip(0)
    elif fault == "field":
        batch.inputs["events"]["currency"]["values"][0] += 1
    elif fault == "aligned":
        batch.targets["aligned"]["event_type"]["values"][0] += 1
    elif fault == "threshold":
        batch.targets["aligned"]["positive_amount"]["values"][0] += 1
    elif fault == "aligned_mask":
        batch.targets["aligned"]["event_type"]["valid"][0] = False
    elif fault == "matching":
        batch.targets["matching"]["values"][0] += 1
    elif fault == "matching_valid":
        batch.targets["matching"]["valid"][0] = False
    elif fault == "label":
        batch.targets["event_type"]["values"][0] += 1
    elif fault == "label_valid":
        batch.targets["event_type"]["valid"][0] = False
    else:
        batch.inputs["events"]["amount"] = batch.inputs["events"]["currency"]
    with pytest.raises(AssertionError, match="Feature assertions failed"):
        check(batch)


def test_auxiliary_targets_do_not_enter_forecast_loss(native):
    binding, _, _, task, batch = native
    model = build_model(task, selection(binding), "nuformer")
    before = model.training_step(batch)
    changed = deepcopy(batch)
    changed.targets["aligned"]["positive_amount"]["values"] += 100
    changed.targets["matching"]["values"] += 100
    torch.testing.assert_close(
        model.training_step(changed).loss, before.loss, rtol=0, atol=0
    )
    assert set(before.logits) == {"event_type", "event_subtype"}
    assert before.weights == {"event_type": 2, "event_subtype": 3}


def test_category_projection_keeps_independent_validity(tmp_path):
    _, out, _ = prepared(tmp_path)
    rows = {r["sample_id"]: r for r in projected_rows(out / "anchors.parquet", "train")}
    assert rows["s1"]["event_type"] is None and rows["s1"]["event_subtype"] is None
    assert rows["s2"]["event_type"] is None and rows["s2"]["event_subtype"] == "3"
    assert set(rows) == {"s0", "s1", "s2", "repeat-client"}


def test_packed_model_preserves_native_batch_and_matches_padded(native):
    binding, _, _, task, batch = native
    model = build_model(task, selection(binding), "nuformer")
    original = deepcopy(batch.inputs["history"])
    output = model(batch)
    padded = deepcopy(batch)
    history = padded.inputs["history"]
    bounds = history["offsets"].tolist()
    from torch.nn.utils.rnn import pad_sequence

    for key in ("value_ids", "valid"):
        history[key] = pad_sequence(
            [history[key][a:b] for a, b in zip(bounds, bounds[1:])], batch_first=True
        )
    expected = model(padded)
    for key in output:
        torch.testing.assert_close(output[key], expected[key], rtol=0, atol=0)
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(batch.inputs["history"][key], value)
        else:
            assert batch.inputs["history"][key] == value


def test_equal_time_permutation_keeps_joint_input_target_records(native):
    binding, out, mapping, task, batch = native
    check = category_features(
        binding,
        out / "anchors.parquet",
        mapping,
        task.fitted_state.to_dict(),
        mode="smoke",
    )
    order = torch.arange(batch.inputs["history"]["value_ids"].numel())
    order[2], order[3] = 3, 2
    for group in [batch.inputs["events"], batch.targets["aligned"]]:
        for item in group.values():
            if isinstance(item, dict):
                for key in ["values", "valid"]:
                    item[key] = item[key][order]
    batch.inputs["history"]["value_ids"] = batch.inputs["history"]["value_ids"][order]
    check(batch)
    batch.targets["aligned"]["positive_amount"]["values"][2:4] = batch.targets[
        "aligned"
    ]["positive_amount"]["values"][2:4].flip(0)
    with pytest.raises(AssertionError, match="Feature assertions failed"):
        check(batch)


@pytest.mark.parametrize("fault", ["label", "validity", "null_validity", "clock"])
def test_category_source_and_projection_reject_wrong_semantics(tmp_path, fault):
    from test.kumo_pytorch.targets import verify_target_fitting

    data, files, binding = fixture(tmp_path)
    path = data / "anchors.parquet"
    table = pq.read_table(path)
    column = {
        "label": "next_event_type",
        "validity": "next_event_type_valid",
        "null_validity": "next_event_type_valid",
        "clock": "as_of_time",
    }[fault]
    values = table[column].to_pylist()
    if fault == "label":
        values[0] = 123
    elif fault == "validity":
        values[0] = False
    elif fault == "null_validity":
        values[0] = None
    else:
        from datetime import timedelta

        values[0] += timedelta(seconds=1)
    table = table.set_column(
        table.schema.get_field_index(column),
        column,
        pa.array(values, type=table[column].type),
    )
    pq.write_table(table, path)
    files["anchors.parquet"].update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    out = tmp_path / "out"
    out.mkdir()
    if fault == "null_validity":
        with pytest.raises(ValueError, match="non-null boolean"):
            prepare_binding(data, out, binding, files, limit=None)
        return
    dataset, anchors, pop, mapping, capacity = prepare_binding(
        data, out, binding, files, limit=None
    )
    contract = binding_contract(
        schema_for(dataset, binding), binding, max_events=capacity
    )
    with r.prepare(
        contract, dataset, r.DiskGraphEngine(index=out / "index", num_threads=1)
    ) as plan:
        with plan.materialize(
            anchors, fit_split="train", encoder_cutoff=pop["encoder_cutoff_seconds"]
        ) as task:
            verify_target_fitting(
                binding,
                projected_rows(out / "anchors.parquet", "train"),
                task.fitted_state.to_dict(),
            )
            with pytest.raises(AssertionError, match="earliest|validity"):
                category_features(
                    binding,
                    out / "anchors.parquet",
                    mapping,
                    task.fitted_state.to_dict(),
                    mode="smoke",
                )


def test_category_empty_history_keeps_forecast_and_packed_boundaries(tmp_path):
    from test.kumo_pytorch.model_checks.test_count_forecast import DAY, STAMP

    data, files, binding = fixture(tmp_path)
    key = binding["dataset"]["source_options"]["tables"]["transactions"]["paths"][0]
    path = data / key
    table = pq.ParquetFile(path).read()
    import pyarrow.compute as pc

    values = pc.cast(table["event_time"], pa.int64()).to_pylist()
    values = [STAMP + 30 * DAY if value < STAMP else value for value in values]
    table = table.set_column(
        table.schema.get_field_index("event_time"),
        "event_time",
        pa.array(values, pa.timestamp("ns")),
    )
    pq.write_table(table, path)
    files[key].update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    out = tmp_path / "out"
    out.mkdir()
    dataset, anchors, pop, mapping, capacity = prepare_binding(
        data, out, binding, files, limit=None
    )
    report = run_case(
        dataset,
        anchors,
        pop,
        binding,
        out,
        mode="smoke",
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", report.get("traceback", report)

    contract = binding_contract(
        schema_for(dataset, binding), binding, max_events=capacity
    )
    with r.prepare(
        contract, dataset, r.DiskGraphEngine(index=out / "empty-index", num_threads=1)
    ) as plan:
        with plan.materialize(
            anchors, fit_split="train", encoder_cutoff=pop["encoder_cutoff_seconds"]
        ) as task:
            batch = next(
                iter(
                    task.batches(
                        split="train",
                        epoch=0,
                        shuffle=False,
                        batch_size=2,
                        output_format="model",
                    )
                )
            )
            assert batch.metadata["sample_ids"][0] == "s0"
            for stream in (
                batch.inputs["history"],
                batch.inputs["events"],
                batch.targets["aligned"],
            ):
                assert stream["offsets"].tolist() == [0, 0, 5]
            model = build_model(task, selection(binding), "nuformer")
            assert not model.backbone.encode_kumo(batch).user_repr[0].any()


def test_auxiliary_fitting_rejects_evaluation_label_leak(native):
    binding, out, mapping, task, _ = native
    state = task.fitted_state.to_dict()
    key = next(
        k for k in state["encoders"] if k == '["targets","aligned","event_type"]'
    )
    state["encoders"][key]["entries"].append(["evaluation-only", 100])
    with pytest.raises(AssertionError, match="train histories only"):
        category_features(
            binding, out / "anchors.parquet", mapping, state, mode="smoke"
        )


@pytest.mark.parametrize("fault", ["start", "monotonicity", "end"])
def test_packed_model_rejects_invalid_ownership(native, fault):
    binding, _, _, task, batch = native
    offsets = batch.inputs["history"]["offsets"].tolist()
    if fault == "start":
        offsets[0] = 1
    elif fault == "monotonicity":
        offsets[2] = offsets[1] - 1
    else:
        offsets[-1] += 1
    batch.inputs["history"]["offsets"] = torch.tensor(offsets, dtype=torch.int64)
    with pytest.raises(ValueError, match="partition"):
        build_model(task, selection(binding), "nuformer")(batch)


def test_distinct_nanoseconds_cannot_permute_within_one_second(native):
    binding, out, mapping, task, batch = native
    check = category_features(
        binding,
        out / "anchors.parquet",
        mapping,
        task.fitted_state.to_dict(),
        mode="smoke",
    )
    check(batch)
    order = torch.arange(batch.inputs["history"]["value_ids"].numel())
    order[3], order[4] = 4, 3
    for group in [batch.inputs["events"], batch.targets["aligned"]]:
        for item in group.values():
            if isinstance(item, dict):
                for key in ["values", "valid"]:
                    item[key] = item[key][order]
    batch.inputs["history"]["value_ids"] = batch.inputs["history"]["value_ids"][order]
    with pytest.raises(AssertionError, match="Feature assertions failed"):
        check(batch)


def test_chronological_source_keys_are_stable_and_preserve_nanoseconds(tmp_path):
    from test.kumo_pytorch.partitioned_sources import prepare_sources

    data, files, binding = fixture(tmp_path)
    full, _ = prepare_sources(data, tmp_path / "full", binding["dataset"], files, None)
    bounded, _ = prepare_sources(
        data, tmp_path / "bounded", binding["dataset"], files, {"z-client"}
    )
    import pyarrow.compute as pc

    expected = pq.read_table(full.tables["transactions"].path)
    expected = expected.filter(pc.equal(expected["root_id"], 4))
    actual = pq.read_table(bounded.tables["transactions"].path)
    assert expected.equals(actual)
    rows = actual.to_pylist()
    by_key = sorted(rows, key=lambda r: r["event_id"])
    assert [r["event_time_ns"] for r in by_key] == sorted(
        r["event_time_ns"] for r in rows
    )
    assert len({r["event_id"] for r in rows}) == len(rows)


def test_chronological_keys_cover_signed_clock_boundaries_and_duplicate_times():
    from test.kumo_pytorch.partitioned_sources import chronological_keys

    times = [2**63 - 1, -1, -(2**63), 0, -1]
    physical = [2, 10, 11, 12, 2]
    keys = chronological_keys(
        pa.array(times, pa.int64()), pa.array(physical, pa.int64())
    ).to_pylist()
    ordered = sorted(range(len(keys)), key=lambda i: keys[i])
    assert ordered == [2, 4, 1, 3, 0]
    assert all(len(k) == 41 for k in keys) and len(set(keys)) == len(keys)
