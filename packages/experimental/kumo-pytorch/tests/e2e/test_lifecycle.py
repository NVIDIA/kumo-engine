# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""H&M lifecycle checks over shared saved runs."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("diskgraph")
pytest.importorskip("torch_geometric")

from e2e.datasets.hm import HMDataset  # noqa: E402
from e2e.flows.common import (  # noqa: E402
    CustomerRows,
    anchors,
    dataset,
    read_predictions,
    serving_requests,
)
from kumo.pytorch.api import (  # noqa: E402
    Artifact,
    Batch,
    DiskGraphEngine,
    ServingAssembler,
    load_weights,
    prepare,
)
from tests.e2e.hm_support import (  # noqa: E402
    REORDER_TOLERANCE,
    Report,
    Run,
    _tie_cut_requests,
)
from tests.e2e.runs import FLOWS  # noqa: E402
from tests.graph_semantics import SourceRows  # noqa: E402
from tests.serving_assertions import assert_serving_batches  # noqa: E402


def test_a_fresh_process_scores_the_test_anchors_as_training_did(
    runs: dict[str, list[Run]], flow: str
) -> None:
    for run in runs[flow]:
        trained = json.loads((run.work / "train.json").read_text())
        inferred = json.loads((run.work / "infer.json").read_text())
        assert inferred["test_digest"] == trained["test_digest"], run
        in_training = read_predictions(run.work / "predictions-train-process.parquet")
        assert run.predictions == in_training, f"{run}: a fresh process scored differently"


def test_serving_equals_reference_batch_inference(
    runs: dict[str, list[Run]],
    flow: str,
    hm: HMDataset,
    reference: dict[str, dict[int, Batch]],
) -> None:
    """Serving satisfies the same source semantics, including eligible graph ties."""

    store = CustomerRows.open(hm.root)
    artifact = Artifact.load(runs[flow][0].work / "artifact")
    server = ServingAssembler(artifact)
    source = SourceRows(server.contract.schema, store.tables)
    requests = serving_requests(hm.root, store, count=16)
    assert sorted(reference[flow]) == sorted(int(r.sample_id) for r in requests)
    for request in requests:
        served = server.assemble(store.lookup(server.rows_spec, request), [request], seed=0)
        assert_serving_batches(
            served,
            [reference[flow][int(request.sample_id)]],
            artifact,
            source,
            [request],
        )


def test_diskgraph_batch_inference_equals_the_reference(
    runs: dict[str, list[Run]],
    flow: str,
    hm: HMDataset,
    reference: dict[str, dict[int, Batch]],
    report: Report,
) -> None:
    """The flow's DiskGraph test predictions equal the model's on reference batches."""

    run = runs[flow][0]
    _, model = importlib.import_module(FLOWS[flow]).load_model(run.work)
    scored = run.predictions
    with torch.no_grad():
        gaps = {
            sample_id: abs(torch.sigmoid(model(batch)).item() - scored[sample_id])
            for sample_id, batch in reference[flow].items()
        }
    if flow == "gnn":
        tied = _tie_cut_requests(hm)
        gaps = {sample_id: gap for sample_id, gap in gaps.items() if sample_id not in tied}
    assert len(gaps) >= 8, gaps
    report.record(f"{flow}.reference", {"compared": len(gaps), "max_gap": max(gaps.values())})
    assert max(gaps.values()) <= REORDER_TOLERANCE, gaps


def test_weights_and_state_are_refused_under_another_contract_or_fit(
    runs: dict[str, list[Run]], hm: HMDataset, root: Path
) -> None:
    gnn, pragma = runs["gnn"][0].work, runs["pragma"][0].work
    gnn_artifact = Artifact.load(gnn / "artifact")
    pragma_artifact = Artifact.load(pragma / "artifact")
    pretrained = Artifact.load(pragma / "pretrained")
    with pytest.raises(ValueError, match="trained against contract_fingerprint"):
        load_weights(gnn / "model.pt", pragma_artifact.fitted_state)
    with pytest.raises(ValueError, match="trained against contract_fingerprint"):
        load_weights(pragma / "model.pt", gnn_artifact.fitted_state)
    with pytest.raises(ValueError, match="trained against"):
        load_weights(pragma / "model.pt", pretrained.fitted_state)
    engine = DiskGraphEngine(index=root / "refusal-index")
    with prepare(pragma_artifact.contract, dataset(hm.root), engine) as plan:
        with pytest.raises(ValueError, match="fitted for different inputs"):
            plan.materialize(anchors(hm.root), fitted=gnn_artifact.fitted_state, inference=True)


def test_finetuning_keeps_the_pretrained_token_ids(runs: dict[str, list[Run]]) -> None:
    work = runs["pragma"][0].work
    pretrained = Artifact.load(work / "pretrained")
    tuned = Artifact.load(work / "artifact")
    for declared in ("encodings", "id_spaces", "field_names"):
        assert getattr(tuned.bound_contract.batch, declared) == getattr(
            pretrained.bound_contract.batch, declared
        )
    assert tuned.bound_contract.batch.inputs == pretrained.bound_contract.batch.inputs
    assert tuned.output_schema["inputs"] == pretrained.output_schema["inputs"]


def test_every_test_anchor_is_scored_once(
    runs: dict[str, list[Run]], flow: str, hm: HMDataset
) -> None:
    anchors = pq.read_table(hm.anchors)
    test = anchors.filter(pc.equal(anchors["split"], "test"))
    expected = set(test["sample_id"].to_pylist())
    for run in runs[flow]:
        table = pq.read_table(run.work / "predictions.parquet")
        assert sorted(table["sample_id"].to_pylist()) == sorted(expected)
