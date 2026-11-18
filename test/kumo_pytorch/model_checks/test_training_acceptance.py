# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training acceptance harness checks; fixture results are not real-AML acceptance."""

import json
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch import training_acceptance as acceptance
from test.kumo_pytorch.models.aml import StepResult
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

import kumo.pytorch.api as r

from .fixtures import build_fixture
from .test_aml import CASES, fit_case
from .test_run_aml import mapped  # noqa: F401 - pytest fixture

CASE_IDS = [f"{s['selector']['workload_id']}-{f}" for s, f in CASES]


def test_preflight_refusal_does_not_claim_training(mapped, tmp_path, monkeypatch):
    original = acceptance.build_contract

    def filtered(*args, **kwargs):
        contract = original(*args, **kwargs)
        entity = replace(
            contract.sample.entity,
            filters=(r.RowFilter("account_id", "greater_than", 0),),
        )
        return replace(contract, sample=replace(contract.sample, entity=entity))

    monkeypatch.setattr(acceptance, "build_contract", filtered)
    selection, family = CASES[0]
    result = acceptance.run_case(
        *mapped, selection, family, directory=tmp_path / "refused-training", capacity=5
    )
    assert result["status"] == "passed" and result["outcome"] == "refused"
    assert result["refusal"]["issues"][0]["code"] == "DG-ROW-FILTER"
    assert result["repetitions"] == [] and "reproducible" not in result


@pytest.fixture(params=CASES, ids=CASE_IDS)
def fitted(request, tmp_path_factory):
    yield from fit_case(*request.param, "diskgraph", tmp_path_factory)


def test_original_models_meet_training_acceptance(fitted, tmp_path):
    task, _, selection, family = fitted
    fixture = build_fixture()
    graph = family in {"graphsage", "combined"}
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    requests = tuple(
        replace(
            q,
            as_of_time=q.as_of_time - timedelta(seconds=int(graph)),
            targets=() if native else q.targets,
            metadata={"prediction_time_ns": int(q.as_of_time.timestamp()) * 10**9}
            if graph
            else {},
        )
        for q in fixture.training
    )
    results = [
        acceptance.train_repetition(
            task,
            requests,
            selection,
            family,
            directory=tmp_path / "training-acceptance" / f"repeat-{i}",
            capacity=len(fixture.tables["transactions"]),
            seed=17,
            batch_size=2,
        )
        for i in range(2)
    ]
    assert acceptance.compare_repetitions(*results), results
    for result in results:
        assert result["completed_updates"] == 100 and len(result["curve"]) == 100
        assert result["status"] == "passed", {
            n: p for n, p in result["parameters"].items() if not p.get("updated")
        }


@pytest.mark.parametrize("selection,family", CASES, ids=CASE_IDS)
def test_raw_aml_training_acceptance(mapped, selection, family, tmp_path):
    result = acceptance.run_case(
        *mapped,
        selection,
        family,
        directory=tmp_path / "training-acceptance",
        capacity=5,
        batch_size=1,
    )
    # The raw fixture deliberately gives forecast and pretraining different
    # anchors. Classification must not silently switch to the pretraining split.
    pretraining = selection["selector"]["workload_id"] in {
        "masked-value-pretrain",
        "next-token-pretrain",
    }
    expected_ids = (
        {"pretrain-train", "pretrain-train-2"} if pretraining else {"train", "train-2"}
    )
    assert {a["sample_id"] for a in result["training_anchors"]} == expected_ids
    assert result["status"] == "passed", result
    if result.get("outcome") == "refused":
        assert not result["repetitions"]
        return
    assert result["reproducible"] and all(
        x["completed_updates"] == 100 for x in result["repetitions"]
    )


class LinearConsumer(torch.nn.Module):
    """Small harness oracle, not a substitute for the required AML consumers."""

    def __init__(self, fault):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2]))
        if fault == "unused":
            self.unused = torch.nn.Parameter(torch.tensor([1.0]))
        if fault == "gradient":
            self.weight.register_hook(lambda gradient: gradient * float("nan"))
        self.fault = fault

    def training_step(self, batch):
        logits = self.weight * batch.inputs["x"]
        valid = batch.targets["label"]["valid"]
        loss = (
            F.binary_cross_entropy_with_logits(
                logits[valid], batch.targets["label"]["values"][valid]
            )
            if valid.any()
            else logits.sum() * 0
        )
        if self.fault == "loss":
            loss = loss * float("nan")
        return StepResult(logits, loss, int(valid.sum()))


@pytest.mark.parametrize(
    "fault", [None, "unused", "gradient", "loss", "no_update", "empty_prefix"]
)
def test_acceptance_detects_broken_training(tmp_path, monkeypatch, fault):
    batch = r.Batch(
        inputs={"x": torch.tensor([-1.0, 2.0])},
        targets={
            "label": {
                "values": torch.tensor([0.0, 1.0]),
                "valid": torch.tensor([True, True]),
            }
        },
        metadata={"sample_ids": ["a", "b"], "is_padding": torch.tensor([False, False])},
    )
    requests = tuple(
        r.SampleRequest(
            name, i, 1, split="train", targets=(r.TargetValue("label", label),)
        )
        for i, (name, label) in enumerate((("a", 0), ("b", 1)))
    )
    batches = [batch]
    if fault == "empty_prefix":
        requests += (
            r.SampleRequest(
                "c", 3, 1, split="train", targets=(r.TargetValue("label", None),)
            ),
        )
        batches.insert(
            0,
            r.Batch(
                inputs={"x": torch.tensor([0.0])},
                targets={
                    "label": {
                        "values": torch.tensor([0.0]),
                        "valid": torch.tensor([False]),
                    }
                },
                metadata={"sample_ids": ["c"], "is_padding": torch.tensor([False])},
            ),
        )
    task = SimpleNamespace(
        fitted_state=SimpleNamespace(fingerprint="fixed"),
        batches=lambda *args, **kwargs: iter(batches),
    )
    monkeypatch.setattr(
        acceptance, "build_model", lambda *a, **k: LinearConsumer(fault)
    )
    if fault == "no_update":
        monkeypatch.setattr(torch.optim.AdamW, "step", lambda self: None)
    result = acceptance.train_repetition(
        task,
        requests,
        {"selector": {"workload_id": "account-involvement-next24h"}},
        "combined",
        directory=tmp_path / "trial",
        capacity=1,
        seed=17,
        batch_size=2,
    )
    if fault in (None, "empty_prefix"):
        assert result["status"] == "passed" and result["completed_updates"] == 100
        assert result["final_loss"] < result["initial_loss"]
        changed = deepcopy(result)
        changed["curve"][0]["loss"] += 1
        assert acceptance.compare_repetitions(result, result)
        assert not acceptance.compare_repetitions(result, changed)
        if fault == "empty_prefix":
            assert len(result["zero_target_batches"]) == 100
            assert all(
                row["sample_ids"] == ["c"] for row in result["zero_target_batches"]
            )
    else:
        assert result["status"] == "failed"
        if fault == "unused":
            assert not result["checks"]["every_parameter_updated"]
            assert result["parameters"]["unused"]["nonzero_gradient_steps"] == 0
        elif fault == "no_update":
            assert not result["checks"]["loss_decreased"]
            assert not result["checks"]["every_parameter_updated"]
        else:
            assert "Nonfinite" in result["error"]


def test_cli_keeps_failed_case_and_runs_all_five(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "verify_inputs", lambda _: {})
    monkeypatch.setattr(acceptance, "load_selections", lambda *a, **kw: {})
    monkeypatch.setattr(acceptance, "prepare", lambda *a: (None, None, {}, 1))
    calls = []

    def run(*args, **kwargs):
        selection, family = args[3:5]
        workload = selection["selector"]["workload_id"]
        calls.append((workload, family))
        return {
            "case_id": f"{workload}-{family}",
            "family": family,
            "status": "failed" if family == "combined" else "passed",
            "repetitions": [],
        }

    monkeypatch.setattr(acceptance, "run_case", run)
    output = tmp_path / "campaign"
    assert acceptance.main(["--data-dir", str(tmp_path), "--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert len(calls) == 5 and set(calls) == acceptance.CASES
    assert report["status"] == "failed"
    assert [case["status"] for case in report["cases"]].count("failed") == 1
    assert len({case["case_id"] for case in report["cases"]}) == 5
    summary = (output / "report.md").read_text()
    assert all(case["case_id"] in summary for case in report["cases"])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown"])
def test_training_matrix_cannot_drop_or_substitute_a_case(fault):
    # Start from the actual catalog, whose selection can name several models.
    selections = json.loads((acceptance.ROOT / "workloads/aml.json").read_text())
    assert len(acceptance.training_cases(selections)) == 5
    if fault == "missing":
        selections.pop()
    elif fault == "duplicate":
        selections.append(deepcopy(selections[0]))
    else:
        selections[-1]["models"] = ["unknown"]
    with pytest.raises(AssertionError, match="exactly once"):
        acceptance.training_cases(selections)


@pytest.mark.parametrize(
    "selection,family", [(s, f) for s, f in CASES if f in {"graphsage", "combined"}]
)
def test_root_only_graph_preserves_predictions(selection, family, tmp_path_factory):
    from test.kumo_pytorch.models.aml import build_model
    from test.kumo_pytorch.models.graphsage import GraphSAGEModel
    from test.kumo_pytorch.models.graphsage._relbench import HeteroGraphSAGE

    with closing(fit_case(selection, family, "diskgraph", tmp_path_factory)) as cases:
        task, batch, _, _ = next(cases)
        model = build_model(task, selection, family, dim=16, max_seq_len=512, seed=17)
        model.eval()
        pruned = (
            model.backbone.graph_encoder if family == "combined" else model.backbone
        )
        full = GraphSAGEModel(pruned.config)
        # Restore the original all-node final layer for an independent baseline.
        full.gnn = HeteroGraphSAGE(
            list(full._node_keys.values()),
            list(full._edge_keys.values()),
            full.config.channels,
            aggr=full.config.aggr,
            num_layers=full.config.num_layers,
        )
        missing = full.load_state_dict(pruned.state_dict(), strict=False)
        assert len(missing.missing_keys) == (10 if family == "combined" else 8)
        assert not missing.unexpected_keys
        full.eval()
        with torch.no_grad():
            actual_graph = pruned.encode_kumo(batch)
            expected_graph = full.encode_kumo(batch)
            assert set(actual_graph.node_embeddings) == {pruned.config.root_type}
            torch.testing.assert_close(
                actual_graph.seed_embeddings,
                expected_graph.seed_embeddings,
                rtol=0,
                atol=0,
            )
            actual = model.backbone.forward_kumo(batch)
            if family == "combined":
                model.backbone.graph_encoder = full
            else:
                model.backbone = full
            expected = model.backbone.forward_kumo(batch)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
