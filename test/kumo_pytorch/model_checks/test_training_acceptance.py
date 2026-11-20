# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training acceptance harness checks; fixture results are not real-AML acceptance."""

from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch import nightly
from test.kumo_pytorch import training_acceptance as acceptance
from test.kumo_pytorch.population_anchors import projected_rows

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import build_fixture
from .test_aml import CASES, fit_case
from .test_population import projected_case
from .test_run_aml import mapped  # noqa: F401 - pytest fixture

CASE_IDS = [f"{s['selector']['workload_id']}-{f}" for s, f in CASES]


def test_preflight_refusal_does_not_claim_training(mapped, tmp_path, monkeypatch):
    original = nightly.contract_for

    def filtered(*args, **kwargs):
        contract = original(*args, **kwargs)
        entity = replace(
            contract.sample.entity,
            filters=(r.RowFilter("account_id", "greater_than", 0),),
        )
        return replace(contract, sample=replace(contract.sample, entity=entity))

    monkeypatch.setattr(nightly, "contract_for", filtered)
    selection, family = CASES[0]
    directory = tmp_path / "refused-training"
    result = nightly.run_case(
        *projected_case(mapped, selection, family, directory, repeats=1),
        directory,
        mode="training",
        capacity=5,
        mapping={"scope": {"kind": "full_source"}},
    )
    assert result["status"] == "passed" and result["outcome"] == "refused"
    assert result["refusal"]["issues"][0]["code"] == "DG-ROW-FILTER"
    assert not result.get("repeats") and "reproducible" not in result
    assert result["stage"] == "prepare" and "fitted_state_sha256" not in result


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
    directory = tmp_path / "training-acceptance"
    result = nightly.run_case(
        *projected_case(mapped, selection, family, directory, repeats=1),
        directory,
        mode="training",
        capacity=5,
        batch_size=1,
        seed=17,
        mapping={"scope": {"kind": "full_source"}},
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
    assert {
        a["sample_id"] for a in projected_rows(directory / "anchors.parquet", "train")
    } == expected_ids
    assert result["status"] == "passed", result
    if result.get("outcome") == "refused":
        assert not result.get("repeats")
        return
    assert result["reproducible"] and all(
        x["completed_updates"] == 100 for x in result["repeats"]
    )
    for repeat in result["repeats"]:
        assert {
            sample
            for step in repeat["curve"] + repeat["zero_target_batches"]
            for sample in step["sample_ids"]
        } == expected_ids
    assert set(result["evaluation"]) == {"valid", "test"}


@pytest.mark.parametrize(
    "selection,family", [(s, f) for s, f in CASES if f in {"graphsage", "combined"}]
)
def test_root_only_graph_preserves_predictions(selection, family, tmp_path_factory):
    from test.kumo_pytorch.models import build_model
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
