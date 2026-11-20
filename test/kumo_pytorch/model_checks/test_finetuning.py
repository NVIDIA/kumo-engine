# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finetuning starts from persisted state in a fresh interpreter, without refitting."""

import json
import os
from dataclasses import replace
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.lifecycle import input_digest
from test.kumo_pytorch.models import build_model
from test.kumo_pytorch.training_lifecycle import check_training

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import CUTOFF, build_fixture, probe_anchor_table
from .test_aml import CASES


def save_pretrained(directory, selection, family):
    fixture = build_fixture()
    dataset = fixture.parquet_dataset(directory / "sources")
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    graph = family in {"graphsage", "combined"}
    stamp = int(CUTOFF.timestamp())
    training = tuple(
        replace(
            row,
            as_of_time=stamp - int(graph),
            targets=() if native else row.targets,
            metadata={"prediction_time_ns": stamp * 10**9} if graph else {},
        )
        for row in fixture.training
    )
    later = stamp + 86400
    finetuning = tuple(
        replace(
            row,
            as_of_time=later - int(graph),
            metadata={"prediction_time_ns": later * 10**9} if graph else {},
        )
        for row in fixture.training
    )
    capacity = len(fixture.tables["transactions"])
    contract = build_contract(
        fixture.schema, family, native=native, max_events=capacity
    )
    lifecycle = directory / "lifecycle"
    lifecycle.mkdir()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            r.prepare(
                contract,
                dataset,
                r.DiskGraphEngine(index=directory / "index", num_threads=1),
            ) as plan,
            plan.materialize(
                probe_anchor_table(plan, training),
                fit_split="train",
                encoder_cutoff=stamp - 1,
            ) as task,
        ):
            model = build_model(
                task, selection, family, dim=8, max_seq_len=11 * capacity, seed=17
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
            result = model.training_step(
                next(task.batches(batch_size=2, seed=17, output_format="model"))
            )
            assert result.weight > 0 and torch.isfinite(result.loss)
            result.loss.backward()
            optimizer.step()
            task.to_artifact().save(lifecycle / "artifact")
            r.save_weights(
                lifecycle / "weights.pt", model.state_dict(), task.fitted_state
            )
            config = {
                "selection": selection,
                "family": family,
                "model": {"dim": 8, "max_seq_len": 11 * capacity, "seed": 17},
                "sources": {
                    name: source.path for name, source in dataset.tables.items()
                },
                "index": str(directory / "index"),
                "batch_size": 2,
                "threads": 1,
                "seed": 17,
                "parent_pid": os.getpid(),
                "kumo_api": r.__file__,
                "fitted_state_sha256": task.fitted_state.fingerprint,
            }
            (lifecycle / "config.json").write_text(json.dumps(config))
    finally:
        torch.set_num_threads(previous)
    return lifecycle, finetuning


@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[f"{s['selector']['workload_id']}-{f}" for s, f in CASES],
)
def test_finetunes_in_fresh_process_without_input_refitting(
    tmp_path, selection, family
):
    lifecycle, requests = save_pretrained(tmp_path, selection, family)
    original = r.Artifact.load(lifecycle / "artifact").fitted_state
    result = check_training(lifecycle, requests)
    tuned = r.Artifact.load(tmp_path / "finetuning/artifact").fitted_state
    assert result["status"] == "passed" and result["input_refitting"] is False
    assert result["parent_pid"] == os.getpid() != result["pid"]
    assert result["resume"]["parent_pid"] == result["pid"]
    assert len({os.getpid(), result["pid"], result["resume"]["pid"]}) == 3
    assert tuned.inputs == original.inputs
    assert tuned.fit["inputs"] == original.fit["inputs"]
    assert result["input_state_content_sha256"] == input_digest(
        original.inputs.to_dict()
    )
    assert result["first_update"]["sample_ids"] == [r.sample_id for r in requests]
    assert result["head_replaced"] == (
        selection["selector"]["workload_id"] != "account-involvement-next24h"
    )
    if family != "graphsage":
        assert result["first_update"]["unknown_tokens"] > 0, (
            "Finetuning fitted its new categories"
        )
    assert result["resume"]["status"] == "passed"
