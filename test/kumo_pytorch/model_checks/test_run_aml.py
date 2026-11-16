"""Exercise the real-data runner on a tiny raw AML-format Parquet fixture."""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys

import pytest
import torch
import kumo.pytorch.api as r

from test.kumo_pytorch.aml_source import (
    account_identity_map,
    prepare_aml_sources,
    prepared_dataset,
)
from test.kumo_pytorch.run_aml import run_case
from test.kumo_pytorch.lifecycle import input_digest
from .test_aml_source import account, source_files, transaction


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    directory = tmp_path_factory.mktemp("raw-aml")
    source, binding, sums = source_files(
        directory,
        rows=[
            transaction(),
            transaction(
                timestamp="2022/09/02 03:00",
                from_bank="2",
                from_account="0008",
                to_bank="1",
                to_account="0007",
                amount_paid="-1.2345678901",
            ),
            transaction(timestamp="2022/09/03 00:00", to_bank="1", to_account="0007"),
            transaction(
                timestamp="2022/09/05 03:00", payment_format="FINETUNE_ONLY_FORMAT"
            ),
            transaction(timestamp="2022/09/06 03:00", payment_format="HELD_OUT_FORMAT"),
            transaction(timestamp="2022/09/08 00:00", payment_format="FUTURE_ONLY"),
        ],
    )
    prepared = directory / "mapped"
    prepare_aml_sources(source, binding, prepared, "hi-small", expected_checksums=sums)
    groups = {}
    for split, day in (("train", 6), ("valid", 7), ("test", 8)):
        stamp = int(datetime(2022, 9, day, tzinfo=timezone.utc).timestamp()) * 10**9
        groups[split] = [
            {
                "sample_id": split,
                "entity_id": account(1, "0007"),
                "as_of_time": stamp,
                "graph_as_of_time": stamp - 10**9,
                "split": split,
                "forecast_split": split,
                "involvement": 1,
                "physical_row_position": day - 6,
            }
        ]
    native_groups = {
        **groups,
        "train": [
            {
                **groups["train"][0],
                "sample_id": "pretrain-train",
                "as_of_time": int(datetime(2022, 9, 4, tzinfo=timezone.utc).timestamp())
                * 10**9,
            }
        ],
    }
    # Two distinct fixture anchors allow a checkpoint inside the training epoch.
    for selected in (groups, native_groups):
        selected["train"].append(
            {
                **selected["train"][0],
                "sample_id": selected["train"][0]["sample_id"] + "-2",
                "physical_row_position": 10,
            }
        )
    selections = {
        "v1": {"split": native_groups, "forecast_split": groups},
        "v2": {"forecast_split": groups},
    }
    yield prepared_dataset(prepared), account_identity_map(prepared), selections


CASES = [
    (record, family)
    for record in json.loads(
        (Path(__file__).parents[1] / "workloads/aml.json").read_text()
    )
    for family in record["models"]
]


@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[f"{r['selector']['workload_id']}-{f}" for r, f in CASES],
)
def test_raw_files_through_training_and_evaluation(mapped, selection, family, tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_case(
            *mapped,
            selection,
            family,
            directory=tmp_path,
            capacity=5,
            batch_size=1,
            steps=1,
            threads=1,
            seed=0,
        )
    finally:
        torch.set_num_threads(previous)
    assert result["status"] == "passed" and result["optimizer_updates"] == 1
    assert result["complete_epoch"] is False
    assert result["fit_split"] == "train" and result["fit_requests"] == 2
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    for split in ("train", "valid", "test"):
        expected_id = "pretrain-train" if native and split == "train" else split
        assert result["splits"][split]["sample_ids"] == [expected_id]
        assert result["splits"][split]["target_weight"] > 0
    assert (
        result["encoder_cutoff_seconds"]
        == int(datetime(2022, 9, 4 if native else 6, tzinfo=timezone.utc).timestamp())
        - 1
    )
    lifecycle = result["lifecycle"]
    assert lifecycle["status"] == "passed"
    assert lifecycle["parent_pid"] == os.getpid() != lifecycle["pid"]
    assert lifecycle["fitted_state_sha256"] == result["fitted_state_sha256"]
    assert lifecycle["sample_ids"] == ["valid", "test"]
    assert lifecycle["no_refitting"] and lifecycle["targets_absent"]
    if family != "graphsage":
        expected_tokens = 30 if family == "pragma" else 54
        assert lifecycle["history_tokens"] == [expected_tokens, expected_tokens]
        assert all(count > 0 for count in lifecycle["unknown_tokens"])
    finetuning = result["finetuning"]
    assert finetuning["status"] == "passed" and not finetuning["input_refitting"]
    assert finetuning["parent_pid"] == os.getpid() != finetuning["pid"]
    assert finetuning["head_replaced"] == native
    assert finetuning["split"] == "forecast_split/train"
    pretrained = r.Artifact.load(tmp_path / "lifecycle/artifact")
    tuned = r.Artifact.load(tmp_path / "finetuning/artifact")
    assert pretrained.fitted_state.inputs == tuned.fitted_state.inputs
    assert (
        finetuning["input_state_sha256"] == pretrained.fitted_state.inputs.fingerprint
    )
    first = finetuning["first_update"]
    resumed = finetuning["resume"]
    assert resumed["status"] == "passed" and resumed["completed_updates"] == 2
    assert resumed["parent_pid"] == finetuning["pid"] != resumed["pid"]
    assert resumed["pid"] != os.getpid()
    assert first["sample_ids"] == resumed["next_update"]["sample_ids"] == ["train"]
    assert first["target_weight"] == resumed["next_update"]["target_weight"] == 1
    if family != "graphsage":
        assert first["history_tokens"] == (24 if family == "pragma" else 43)
    if native:
        assert first["unknown_tokens"] > 0, "Finetuning fitted its new category"
    mid_epoch = result["mid_epoch_resume"]
    assert mid_epoch["status"] == "passed"
    assert mid_epoch["parent_pid"] == os.getpid() != mid_epoch["pid"]
    assert mid_epoch["final_iterator"]["batches"] == 2
    assert len(mid_epoch["remaining"]) == 1
    # A validly bound but wrong model checkpoint must still fail prediction
    # equivalence; generic malformed/incompatible artifacts have package tests.
    if selection["selector"]["workload_id"] == "next-token-pretrain":
        resume_dir = tmp_path / "finetuning"
        checkpoint = torch.load(resume_dir / "optimizer.pt", weights_only=True)
        checkpoint["optimizer"]["state"] = {}
        checkpoint["optimizer_sha256"] = input_digest(checkpoint["optimizer"])
        torch.save(checkpoint, resume_dir / "optimizer.pt")
        (resume_dir / "result.json").unlink()
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "test.kumo_pytorch.training_lifecycle",
                str(resume_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode != 0 and "Resumed update differs" in child.stderr
        assert not (resume_dir / "result.json").exists()
        directory = tmp_path / "lifecycle"
        artifact = r.Artifact.load(directory / "artifact")
        weights = r.load_weights(directory / "weights.pt", artifact.fitted_state)
        r.save_weights(
            directory / "weights.pt",
            {key: torch.zeros_like(value) for key, value in weights.items()},
            artifact.fitted_state,
        )
        (directory / "result.json").unlink()
        child = subprocess.run(
            [sys.executable, "-m", "test.kumo_pytorch.lifecycle", str(directory)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode != 0 and "not equal" in child.stderr
        assert not (directory / "result.json").exists()
