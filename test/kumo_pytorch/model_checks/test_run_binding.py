# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LI runner must deliver native anchors and retain honest training evidence."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from test.kumo_pytorch import nightly as run_binding
from test.kumo_pytorch.bindings import load_bindings
from test.kumo_pytorch.data import prepare_binding as prepare_sources
from test.kumo_pytorch.data import schema_for
from test.kumo_pytorch.nightly import main as nightly_main
from test.kumo_pytorch.nightly import run_case

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.pytorch.api as r
from kumo.pytorch.engines.native_anchors import NativeAnchors
from kumo.pytorch.task import Task

from .test_aml_bindings import LI_BINDINGS, source_anchors
from .test_aml_source import account, source_files, transaction


def main(argv):
    return nightly_main(["run", *argv])


def canonical_inputs(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    members = [account(1, "0007", "li-medium"), account(2, "0008", "li-medium")]
    tx, accounts, _ = source_files(
        root,
        [
            transaction(),
            transaction(timestamp="2022/09/02 00:20", payment_format="ACH"),
            transaction(
                timestamp="2022/09/03 00:20",
                from_bank="2",
                from_account="0008",
                to_bank="1",
                to_account="0007",
            ),
        ],
        members,
    )
    original = source_anchors(root / "anchors.parquet")
    table = pq.read_table(original)
    table = table.set_column(
        table.schema.get_field_index("entity_id"),
        "entity_id",
        pa.array([members[0]] * len(table)),
    )
    # Keep valid evaluation rows in addition to the all-missing test anchor.
    parts = [table]
    for split in ("valid", "test"):
        for i in (1, 2):
            row = table.slice(4, 1)
            row = row.set_column(
                row.schema.get_field_index("sample_id"),
                "sample_id",
                pa.array([f"{split}-{i}"]),
            )
            row = row.set_column(
                row.schema.get_field_index("forecast_split"),
                "forecast_split",
                pa.array([split]),
            )
            parts.append(row)
    pq.write_table(pa.concat_tables(parts), original, row_group_size=2)
    paths = {
        "datasets/aml-data/v8/li-medium/transactions.parquet": tx,
        "anchors/aml-data/v8/li-medium/_bindings/v1/accounts.parquet": accounts,
        **{
            f"anchors/aml-data/v8/li-medium/account-daily/{v}/anchors.parquet": original
            for v in ("v1", "v2")
        },
    }
    files = {}
    for key, path in paths.items():
        destination = root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        files[key] = {
            "key": key,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return root, files


@pytest.mark.parametrize("ref", LI_BINDINGS)
@pytest.mark.parametrize("mode", ("training", "full"))
def test_li_runner_native_delivery_and_training(tmp_path, monkeypatch, ref, mode):
    calls = []
    original = NativeAnchors._typed_batches

    def native(owner, *args, **kwargs):
        calls.append(True)
        yield from original(owner, *args, **kwargs)

    monkeypatch.setattr(NativeAnchors, "_typed_batches", native)
    root, files = canonical_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    binding = load_bindings()[ref]
    dataset, anchors, population, mapping, capacity = prepare_sources(
        root,
        output,
        binding,
        files,
        limit=3 if mode == "training" else None,
    )
    assert mapping["variant"] == "li-medium"
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode=mode,
        mapping=mapping,
        capacity=capacity,
        batch_size=3,
        seed=17,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    assert len(calls) >= 3
    assert report["full_population_verified"] == (mode == "full")
    assert {name: row["delivered"] for name, row in report["splits"].items()} == {
        "train": 3,
        "valid": 3,
        "test": 3,
    }
    assert all(
        all(n > 0 for n in row["eligible_targets"].values())
        for row in report["splits"].values()
    )
    if mode == "training":
        if ref == "L01":
            from test.kumo_pytorch.feature_checks import validate_evidence

            validate_evidence(report, binding["feature_checks"], "training")
            assert report["feature_coverage"] == {
                feature: dict.fromkeys(("train", "valid", "test"), 3)
                for feature in binding["feature_checks"]["features"]
            }
        assert report["reproducible"]
        assert all(
            r["completed_updates"] == 100 and all(r["checks"].values())
            for r in report["repeats"]
        )
        assert (
            report["evaluation_model_sha256"]
            == report["repeats"][0]["final_model_sha256"]
        )
        assert all(
            row["eligible"] == 3 for row in report["evaluation"]["valid"].values()
        )
        assert all(
            row["eligible"] == 2 for row in report["evaluation"]["test"].values()
        )
        assert (output / "repeat-0/model.pt").is_file()
    else:
        assert "repeats" not in report
    assert json.loads((output / "report.json").read_text()) == report


def test_full_binding_cannot_claim_bounded_projection(tmp_path):
    root, files = canonical_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    binding = load_bindings()["L02"]
    dataset, anchors, population, mapping, capacity = prepare_sources(
        root, output, binding, files, limit=3
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="full",
        capacity=capacity,
        mapping=mapping,
    )
    assert report["status"] == "failed" and not report["full_population_verified"]
    assert "Bounded anchors" in report["error"]
    assert not (output / "index").exists()


def test_binding_cli_records_missing_pinned_inputs(tmp_path):
    output = tmp_path / "output"
    assert (
        main(
            [
                "--binding",
                "L01",
                "--mode",
                "full",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads((output / "report.json").read_text())
    assert report["stage"] == "verify_inputs" and report["status"] == "failed"
    assert not report["full_population_verified"]
    assert (output / "report.md").is_file()


def test_binding_cli_rejects_full_limit(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--binding",
                "L01",
                "--mode",
                "full",
                "--anchors-per-split",
                "3",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("stage", ("prepare", "execute"))
def test_binding_refusal_must_precede_execution(tmp_path, monkeypatch, stage):
    root, files = canonical_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    binding = load_bindings()["L03"]
    dataset, anchors, population, mapping, capacity = prepare_sources(
        root, output, binding, files, limit=None
    )
    original = run_binding.contract_for

    def unsupported(*args, **kwargs):
        contract = original(*args, **kwargs)
        entity = replace(
            contract.sample.entity,
            filters=(r.RowFilter("account_id", "greater_than", 0),),
        )
        return replace(contract, sample=replace(contract.sample, entity=entity))

    if stage == "prepare":
        monkeypatch.setattr(run_binding, "contract_for", unsupported)
    else:

        def late(*args, **kwargs):
            contract = unsupported(
                schema_for(dataset, binding), binding, max_events=capacity
            )
            r.prepare(
                contract, dataset, r.DiskGraphEngine(index=tmp_path / "late-index")
            )

        monkeypatch.setattr(run_binding, "execute", late)
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="full",
        capacity=capacity,
        mapping=mapping,
    )
    assert not report["full_population_verified"]
    if stage == "prepare":
        assert report["status"] == "passed" and report["outcome"] == "refused"
        assert report["refusal"]["issues"][0]["code"] == "DG-ROW-FILTER"
        assert "splits" not in report
    else:
        assert report["status"] == "failed" and "outcome" not in report
        assert "after case execution began" in report["error"]


@pytest.mark.parametrize("stage", ("probe", "training", "evaluation"))
def test_later_binding_batches_cannot_change_source_targets(
    tmp_path, monkeypatch, stage
):
    root, files = canonical_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    binding = load_bindings()["L04"]
    dataset, anchors, population, mapping, capacity = prepare_sources(
        root, output, binding, files, limit=3
    )
    original = Task.batches
    calls = Counter()
    changed = []

    def corrupt(task, *args, **kwargs):
        split = kwargs["split"]
        calls[split] += 1
        mutate = (
            (stage == "probe" and split == "train" and calls[split] == 2)
            or (stage == "training" and kwargs.get("shuffle", False))
            or (stage == "evaluation" and split == "valid" and calls[split] > 1)
        )
        for batch in original(task, *args, **kwargs):
            if mutate:
                target = batch.targets["label"]
                target["values"][target["valid"]] = (
                    1 - target["values"][target["valid"]]
                )
                changed.append(True)
            yield batch

    monkeypatch.setattr(Task, "batches", corrupt)
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="training",
        capacity=capacity,
        mapping=mapping,
        batch_size=3,
        seed=17,
    )
    assert changed
    assert all(row["status"] == "passed" for row in report["splits"].values())
    assert report["status"] == "failed" and "outcome" not in report
    assert report["stage"] == ("evaluation" if stage == "evaluation" else "training")
    failed = report if stage == "evaluation" else report["repeats"][0]
    assert "verify_stored_batch" in failed["traceback"]


def test_large_valid_amount_preserves_training_progress(tmp_path):
    root, files = canonical_inputs(tmp_path)
    key = "anchors/aml-data/v8/li-medium/account-daily/v2/anchors.parquet"
    path = root / key
    table = pq.read_table(path)
    amounts = pa.array(
        [
            Decimal("1000000000000") if v is not None else None
            for v in table["next_amount"].to_pylist()
        ],
        type=pa.decimal128(38, 10),
    )
    table = table.set_column(
        table.schema.get_field_index("next_amount"), "next_amount", amounts
    )
    pq.write_table(table, path)
    files[key].update(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size
    )
    output = tmp_path / "output"
    output.mkdir()
    binding = load_bindings()["L01"]
    dataset, anchors, population, mapping, capacity = prepare_sources(
        root, output, binding, files, limit=3
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="training",
        capacity=capacity,
        mapping=mapping,
        batch_size=3,
        seed=17,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    assert all(
        1e11 < repeat["final_loss"] < repeat["initial_loss"]
        for repeat in report["repeats"]
    )


@pytest.mark.parametrize("mode", ["lifecycle", "histories"])
def test_li_combined_lifecycle_and_source_histories(tmp_path, mode):
    from .test_binding_features import prepare_features

    binding, output, (dataset, anchors, population, mapping, capacity) = (
        prepare_features(tmp_path)
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode=mode,
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", json.dumps(report, indent=2)
    if mode == "lifecycle":
        assert set(report["lifecycle"]["checks"]) == {
            "inference",
            "finetuning",
            "mid_epoch_resume",
        }
    else:
        assert len(report["histories"]["cases"]) == 2
        assert all(
            c["samples"] == 12
            for c in report["histories"]["cases"]
            if c.get("outcome") != "refused"
        )


def test_finetuning_rechecks_training_target_vocabulary(tmp_path, monkeypatch):
    from copy import deepcopy
    from test.kumo_pytorch import targets
    from test.kumo_pytorch.training_lifecycle import finetune

    from .test_binding_features import prepare_features

    binding, output, (dataset, anchors, population, mapping, capacity) = (
        prepare_features(tmp_path)
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="lifecycle",
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    # Mid-epoch acceptance is a separate required case; this check targets refitting.
    assert report["lifecycle"]["checks"]["finetuning"]["status"] == "passed"
    directory = output / "finetuning"
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config["finetune_parent_pid"] = -1
    path.write_text(json.dumps(config))
    original, called = targets.verify_target_fitting, []

    def wrong_vocabulary(binding, rows, state):
        state = deepcopy(state)
        encoder = targets.fitted_target_encoder(state, "currency")
        entries = list(encoder["entries"])
        assert len(entries) == 2
        encoder["entries"] = [
            [entries[0][0], entries[1][1]],
            [entries[1][0], entries[0][1]],
        ]
        called.append(True)
        return original(binding, rows, state)

    monkeypatch.setattr(targets, "verify_target_fitting", wrong_vocabulary)
    with pytest.raises(AssertionError):
        finetune(directory)
    assert called
