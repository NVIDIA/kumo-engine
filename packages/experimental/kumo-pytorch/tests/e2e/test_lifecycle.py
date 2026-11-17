# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2a and T2b: both H&M flows end to end on DiskGraph, each stage in a fresh process.

``pytest --t2a tests/e2e`` runs them at S (minutes on a CPU), ``pytest --t2b
tests/e2e`` at M (the nightly tier, about 15 minutes on a laptop); ``--e2e-dir DIR`` keeps every
run, its predictions and ``<tier>/metrics.json``. ``tiers.py`` holds what differs.

Lifecycle checks are exact. Model checks compare against the tier's bands, whose
reasoning is stated next to each value. Two known gaps are asserted as they
stand, so a fix fails them and the gap is lifted into the main checks.
"""

from __future__ import annotations

import importlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("diskgraph")
pytest.importorskip("torch_geometric")

from e2e.datasets.hm import (  # noqa: E402
    HMDataset,
    fingerprint_mismatches,
)
from e2e.flows.common import (  # noqa: E402
    CustomerRows,
    anchors,
    dataset,
    oracle_auroc,
    read_predictions,
    serving_requests,
)
from e2e.flows.hm_gnn import RECENT_PURCHASES  # noqa: E402
from e2e.metrics import auroc  # noqa: E402
from kumo.pytorch.api import (  # noqa: E402
    Artifact,
    Batch,
    Dataset,
    DiskGraphEngine,
    InMemoryEngine,
    InMemoryRows,
    SampleRequest,
    ServingAssembler,
    load_weights,
    prepare,
)
from tests.e2e import runs as runs_module  # noqa: E402
from tests.e2e.runs import FLOWS  # noqa: E402
from tests.e2e.tiers import TIERS, Tier  # noqa: E402
from tests.test_inference_lifecycle import assert_same_batches  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[2]
# Where only the order of a float32 sum changes (one request per reference batch against
# hundreds; DiskGraph reordering equal-time purchases), a probability moves by a few
# units in the last place (6e-8 each); 1e-6 allows that, and no changed input.
REORDER_TOLERANCE = 1e-6


# Per flow and test cutoff: test-anchor scores without, then with, the added rows.
FutureScores = dict[str, dict[str, tuple[dict[int, float], dict[int, float]]]]


@dataclass(frozen=True)
class Run:
    flow: str
    seed: int
    work: Path
    seconds: float

    @property
    def predictions(self) -> dict[int, float]:
        return read_predictions(self.work / "predictions.parquet")


class Report:
    """Every measured value, written to ``metrics.json`` whether or not checks pass."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[str, object] = {}

    def record(self, key: str, value: object) -> None:
        self.values[key] = value
        self.path.write_text(json.dumps(self.values, indent=2, sort_keys=True) + "\n")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(tier, id=tier.name, marks=getattr(pytest.mark, tier.name)) for tier in TIERS
    ],
)
def tier(request: pytest.FixtureRequest) -> Tier:
    return request.param


@pytest.fixture(scope="session")
def root(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, tier: Tier
) -> Path:
    chosen: Path | None = request.config.getoption("--e2e-dir")
    if chosen is None:
        return tmp_path_factory.mktemp(tier.name)
    (chosen / tier.name).mkdir(parents=True, exist_ok=True)
    return chosen / tier.name


@pytest.fixture(scope="session")
def report(root: Path) -> Report:
    return Report(root / "metrics.json")


@pytest.fixture(scope="session")
def hm(root: Path, tier: Tier) -> HMDataset:
    return runs_module.dataset(root, tier)


@pytest.fixture(scope="session")
def data(hm: HMDataset, root: Path, tier: Tier) -> HMDataset:
    return runs_module.training_data(hm, root, tier)


@pytest.fixture(scope="session")
def baseline_auroc(tier: Tier) -> float:
    """kumo-engine's lowest test AUROC over its recorded runs."""

    baseline = json.loads((REPOSITORY / "e2e" / "baselines" / tier.baseline).read_text())
    return float(baseline["summary"]["test_auroc"]["min"])


@pytest.fixture(scope="session")
def runs(data: HMDataset, root: Path, report: Report, tier: Tier) -> dict[str, list[Run]]:
    """Every flow and seed of the tier, trained here or by a CI job before (``runs.py``)."""

    done: dict[str, list[Run]] = {}
    for flow in FLOWS:
        done[flow] = []
        for seed in tier.seeds[flow]:
            work, seconds = runs_module.run(data, root, flow, seed)
            done[flow].append(Run(flow, seed, work, seconds))
            report.record(f"{flow}.seed{seed}.wall_seconds", round(seconds, 1))
    return done


@pytest.fixture(params=sorted(FLOWS))
def flow(request: pytest.FixtureRequest) -> str:
    return request.param


def test_the_baseline_was_recorded_on_this_dataset(hm: HMDataset, tier: Tier) -> None:
    baseline = json.loads((REPOSITORY / "e2e" / "baselines" / tier.baseline).read_text())
    recorded = baseline["dataset"]
    assert (recorded["scale"], recorded["seed"]) == (tier.scale, 0)
    assert recorded["generator_version"] == hm.metadata.generator_version
    mismatched = fingerprint_mismatches(
        hm.metadata, recorded["table_sha256"], recorded["float_sums"]
    )
    assert not mismatched, f"re-record the baseline: {mismatched} differ"


def test_a_fresh_process_scores_the_test_anchors_as_training_did(
    runs: dict[str, list[Run]], flow: str
) -> None:
    for run in runs[flow]:
        trained = json.loads((run.work / "train.json").read_text())
        inferred = json.loads((run.work / "infer.json").read_text())
        assert inferred["test_digest"] == trained["test_digest"], run
        in_training = read_predictions(run.work / "predictions-train-process.parquet")
        assert run.predictions == in_training, f"{run}: a fresh process scored differently"


@pytest.fixture(scope="session")
def future_scores(runs: dict[str, list[Run]], hm: HMDataset, root: Path) -> FutureScores:
    scores: FutureScores = {}
    for flow, module in FLOWS.items():
        scores[flow] = {}
        for cutoff in hm.metadata.cutoffs["test"]:
            at = datetime.fromisoformat(cutoff)
            base = root / f"{flow}-future-{at.date().isoformat()}"
            pair = []
            for name, future in (("plain", False), ("future", True)):
                data = _dataset_at(hm, at, base / name, future=future)
                out = base / f"{name}-predictions"
                subprocess.run(
                    [sys.executable, "-m", module, "infer", "--data", str(data)]
                    + ["--work", str(runs[flow][0].work), "--out", str(out)],
                    cwd=REPOSITORY,
                    check=True,
                )
                pair.append(read_predictions(out / "predictions.parquet"))
            scores[flow][cutoff] = (pair[0], pair[1])
    return scores


@pytest.fixture(scope="session")
def reference(runs: dict[str, list[Run]], hm: HMDataset) -> dict[str, dict[int, Batch]]:
    """Reference (in-memory) batch inference for the served requests, over every customer
    and article and each served customer's purchases at any time (both contracts reach
    purchases only through their customer; a missing row would fail the checks)."""

    store = CustomerRows.open(hm.root)
    requests = serving_requests(hm.root, store, count=16)
    tables = dict(store.tables)
    served = pa.array([request.entity_id for request in requests])
    purchases = tables["transactions"]
    tables["transactions"] = purchases.filter(pc.is_in(purchases["customer_id"], served))
    rows = Dataset({name: InMemoryRows(table.to_pylist()) for name, table in tables.items()})
    return {
        flow: _reference_batches(Artifact.load(flow_runs[0].work / "artifact"), rows, requests)
        for flow, flow_runs in runs.items()
    }


def test_serving_equals_reference_batch_inference(
    runs: dict[str, list[Run]],
    flow: str,
    hm: HMDataset,
    reference: dict[str, dict[int, Batch]],
) -> None:
    """Each request, served from the rows the flow looks up by following rows_spec,
    gives exactly the reference batch-inference batch (``reference``)."""

    store = CustomerRows.open(hm.root)
    server = ServingAssembler(Artifact.load(runs[flow][0].work / "artifact"))
    requests = serving_requests(hm.root, store, count=16)
    assert sorted(reference[flow]) == sorted(int(r.sample_id) for r in requests)
    for request in requests:
        served = server.assemble(store.lookup(server.rows_spec, request), [request], seed=0)
        assert_same_batches(served, [reference[flow][int(request.sample_id)]])


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


def test_rows_after_the_cutoff_change_no_prediction(
    future_scores: FutureScores,
    flow: str,
    hm: HMDataset,
    report: Report,
) -> None:
    times = _purchase_times(hm)
    customers = _test_customers(hm)
    for cutoff, (plain, future) in future_scores[flow].items():
        assert plain.keys() == future.keys()
        assert len(plain) == hm.metadata.sizes["customers"]
        if flow == "pragma":
            assert future == plain, f"rows after {cutoff} changed a score"
            continue
        limit = _epoch(datetime.fromisoformat(cutoff))
        clear = [s for s in plain if not _cuts_a_tie(times, customers[s], limit)]
        gap = max(abs(future[s] - plain[s]) for s in clear)
        report.record(f"gnn.future_rows.{cutoff}", {"compared": len(clear), "max_gap": gap})
        assert gap <= REORDER_TOLERANCE, f"rows after {cutoff} changed a score"


def test_rows_after_the_cutoff_reach_no_gnn_input(
    runs: dict[str, list[Run]],
    future_scores: FutureScores,
    hm: HMDataset,
    root: Path,
) -> None:
    """Every anchor, tied ones included: the same customer columns and purchases at
    the same times, and articles that are the sampled purchases' own, all dated
    before the cutoff, with or without the added rows. Untied anchors also keep the
    same articles; a tied one may keep another purchase of the same session.
    The cutoff is inclusive, so "before" includes rows dated exactly at it."""

    artifact = Artifact.load(runs["gnn"][0].work / "artifact")
    times = _purchase_times(hm)
    customers = _test_customers(hm)
    for cutoff in future_scores["gnn"]:
        limit = _epoch(datetime.fromisoformat(cutoff))
        base = root / f"gnn-future-{datetime.fromisoformat(cutoff).date().isoformat()}"
        seen = [
            _gnn_inputs(artifact, base / name, root / "input-index") for name in ("plain", "future")
        ]
        assert seen[0].keys() == seen[1].keys()
        assert len(seen[0]) == hm.metadata.sizes["customers"]
        for sample_id, inputs in seen[1].items():
            plain = seen[0][sample_id]
            assert inputs.columns == plain.columns, sample_id
            assert inputs.purchase_times == plain.purchase_times, sample_id
            if not _cuts_a_tie(times, customers[sample_id], limit):
                assert inputs.articles == plain.articles, sample_id
            for found in (inputs, plain):
                assert all(at <= limit for at in found.purchase_times), sample_id
                assert all(at <= limit for _, at in found.articles), sample_id
                assert [i for i, _ in found.articles] == found.purchased_articles, sample_id


def test_an_article_listed_after_the_cutoff_reaches_no_gnn_input(
    runs: dict[str, list[Run]], hm: HMDataset, root: Path
) -> None:
    """The article hop filters an article by its own clock: every customer's newest
    purchase, one second before the cutoff, is of an article listed after it. The
    purchase is sampled; its article is not."""

    artifact = Artifact.load(runs["gnn"][0].work / "artifact")
    cutoff = datetime.fromisoformat(hm.metadata.cutoffs["test"][0])
    data = _dataset_at(hm, cutoff, root / "late-article", future=False)
    articles = pq.read_table(data / "tables" / "articles.parquet")
    late = _future_articles(articles, cutoff + timedelta(days=1)).slice(0, 1)
    (late_id,) = late["article_id"].to_pylist()
    pq.write_table(pa.concat_tables([articles, late]), data / "tables" / "articles.parquet")
    transactions = pq.read_table(data / "tables" / "transactions.parquet")
    customers = sorted(set(transactions["customer_id"].to_pylist()))
    buyers = set(customers)
    first = int(pc.max(transactions["transaction_id"]).as_py()) + 1
    bought = pa.table(
        {
            "transaction_id": pa.array(range(first, first + len(customers)), pa.int64()),
            "t_dat": pa.array(
                [cutoff - timedelta(seconds=1)] * len(customers), transactions["t_dat"].type
            ),
            "customer_id": pa.array(customers, pa.int64()),
            "article_id": pa.array([late_id] * len(customers), pa.int64()),
            "price": pa.array([0.05] * len(customers), pa.float64()),
            "sales_channel_id": pa.array([1] * len(customers), pa.int64()),
        }
    ).select(transactions.column_names)
    pq.write_table(
        pa.concat_tables([transactions, bought]), data / "tables" / "transactions.parquet"
    )
    limit = _epoch(cutoff)
    seen = _gnn_inputs(artifact, data, root / "late-article-index")
    owners = _test_customers(hm)
    assert len(seen) == hm.metadata.sizes["customers"]
    checked = 0
    for sample_id, inputs in seen.items():
        if owners[sample_id] not in buyers:
            continue
        checked += 1
        assert limit - 1 in inputs.purchase_times, sample_id
        assert late_id in inputs.purchased_articles, sample_id
        assert late_id not in [i for i, _ in inputs.articles], sample_id
    assert checked > 0.75 * len(seen), checked


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


def test_the_contract_label_is_the_generator_label(
    runs: dict[str, list[Run]], hm: HMDataset, root: Path
) -> None:
    artifact = Artifact.load(runs["gnn"][0].work / "artifact")
    oracle = pq.read_table(hm.labels, columns=["sample_id", "label"])
    labels = dict(zip(oracle["sample_id"].to_pylist(), oracle["label"].to_pylist(), strict=True))
    engine = DiskGraphEngine(index=root / "label-index")
    seen = 0
    with prepare(artifact.contract, dataset(hm.root), engine) as plan:
        with plan.materialize(anchors(hm.root), fitted=artifact.fitted_state) as task:
            for split in ("train", "val", "test"):
                for batch in task.batches(split=split, output_format="model"):
                    positive = (batch.targets["label"]["values"] > 0).tolist()
                    for sample_id, label in zip(
                        batch.metadata["sample_ids"], positive, strict=True
                    ):
                        assert label == labels[sample_id], sample_id
                        seen += 1
    assert seen == hm.metadata.sizes["anchors"]


def test_test_auroc_is_within_the_band(
    runs: dict[str, list[Run]],
    flow: str,
    hm: HMDataset,
    baseline_auroc: float,
    report: Report,
    tier: Tier,
) -> None:
    ceiling = hm.metadata.ceiling_auroc["test"]
    upper = ceiling + tier.leak_standard_errors * hm.metadata.ceiling_auroc_se["test"]
    recency = _recency_auroc(hm)
    lower = recency + tier.above_recency
    areas = [oracle_auroc(hm.root, run.predictions) for run in runs[flow]]
    report.record(f"{flow}.test_auroc", areas)
    report.record(f"{flow}.band", {"lower": lower, "upper": upper, "ceiling": ceiling})
    report.record("recency_only.test_auroc", recency)
    report.record(
        f"{flow}.vs_kumo_engine",
        {
            "kumo_engine_lowest": baseline_auroc,
            "difference": [area - baseline_auroc for area in areas],
        },
    )
    for run, area in zip(runs[flow], areas, strict=True):
        assert area >= lower, f"seed {run.seed}: test AUROC {area:.4f} below {lower:.4f}"
        assert area <= upper, f"seed {run.seed}: {area:.4f} beats the ceiling; a leak?"
    assert max(areas) - min(areas) <= tier.seed_spread, areas


def test_every_test_anchor_is_scored_once_with_a_sane_distribution(
    runs: dict[str, list[Run]], flow: str, hm: HMDataset, report: Report, tier: Tier
) -> None:
    anchors = pq.read_table(hm.anchors)
    test = anchors.filter(pc.equal(anchors["split"], "test"))
    expected = set(test["sample_id"].to_pylist())
    positive_rate = hm.metadata.positive_rate["test"]
    for run in runs[flow]:
        table = pq.read_table(run.work / "predictions.parquet")
        assert sorted(table["sample_id"].to_pylist()) == sorted(expected)
        scores = table["score"].to_numpy()
        quantiles = np.quantile(scores, [0.1, 0.5, 0.9]).tolist()
        report.record(
            f"{flow}.seed{run.seed}.predictions",
            {"mean": float(scores.mean()), "std": float(scores.std()), "q10_50_90": quantiles},
        )
        assert not np.isnan(scores).any()
        assert scores.std() > 0.01, "a constant model"
        ratio = scores.mean() / positive_rate
        factor = tier.calibration_factor
        assert 1 / factor <= ratio <= factor, (scores.mean(), ratio)


def test_point_lookups_rank_active_customers_above_dormant_ones(
    runs: dict[str, list[Run]], flow: str, hm: HMDataset, report: Report
) -> None:
    anchors = pq.read_table(hm.anchors)
    test = anchors.filter(pc.equal(anchors["split"], "test"))
    samples: dict[int, list[int]] = {}
    for sample_id, customer in zip(
        test["sample_id"].to_pylist(), test["customer_id"].to_pylist(), strict=True
    ):
        samples.setdefault(customer, []).append(sample_id)
    for run in runs[flow]:
        predictions = run.predictions
        mean = float(np.mean(list(predictions.values())))
        by_kind: dict[str, list[float]] = {"active": [], "dormant": []}
        for lookup in hm.metadata.point_lookups:
            by_kind[lookup.kind] += [predictions[s] for s in samples[lookup.customer_id]]
        report.record(
            f"{flow}.seed{run.seed}.point_lookups",
            {kind: [min(values), max(values)] for kind, values in by_kind.items()},
        )
        assert min(by_kind["active"]) > max(by_kind["dormant"])
        assert min(by_kind["active"]) > mean


def test_known_gap_recent_sampling_ties_depend_on_the_engine_and_on_later_rows(
    runs: dict[str, list[Run]],
    future_scores: FutureScores,
    hm: HMDataset,
    root: Path,
    reference: dict[str, dict[int, Batch]],
) -> None:
    """Recent sampling orders by time, then primary key (docs/limitations.md).
    When the 16th and 17th most recent purchases share a timestamp, DiskGraph's choice
    follows its index, so rows added after the cutoff change it, and the in-memory
    engine compares keys as strings. The GNN's DiskGraph checks above skip those
    samples; when this fails, lift the gap from them."""

    times = _purchase_times(hm)
    customers = _test_customers(hm)
    changed = 0
    for cutoff, (plain, future) in future_scores["gnn"].items():
        limit = _epoch(datetime.fromisoformat(cutoff))
        tied = [s for s in plain if _cuts_a_tie(times, customers[s], limit)]
        changed += sum(future[s] != plain[s] for s in tied)
    assert changed > 0

    artifact = Artifact.load(runs["gnn"][0].work / "artifact")
    tied_requests = _tie_cut_requests(hm)
    assert tied_requests, "no served request cuts a tie; the gap can no longer be seen here"
    sample_id = min(tied_requests)
    in_memory = reference["gnn"][sample_id]
    engine = DiskGraphEngine(index=root / "tie-index")
    with prepare(artifact.contract, dataset(hm.root), engine) as plan:
        with plan.materialize(
            [tied_requests[sample_id]], fitted=artifact.fitted_state, inference=True
        ) as task:
            (batch,) = list(task.batches(output_format="model"))
    purchases = [
        set(b.inputs["graph"]["nodes"]["transactions"]["ids"].tolist()) for b in (in_memory, batch)
    ]
    assert len(purchases[0]) == len(purchases[1]) == RECENT_PURCHASES
    assert purchases[0] != purchases[1]


def test_a_history_batch_without_any_event_assembles(
    runs: dict[str, list[Run]], hm: HMDataset, root: Path
) -> None:
    """A customer with no purchase before the cutoff gets a batch with no event, served
    or scored alone, on either engine. The flows still serve only customers who
    bought before, because their model reads at least one event per batch."""

    artifact = Artifact.load(runs["pragma"][0].work / "artifact")
    request = _request_without_history(hm)
    store, server = CustomerRows.open(hm.root), ServingAssembler(artifact)
    (served,) = server.assemble(store.lookup(server.rows_spec, request), [request])
    engine = DiskGraphEngine(index=root / "empty-index")
    with prepare(artifact.contract, dataset(hm.root), engine) as plan:
        with plan.materialize([request], fitted=artifact.fitted_state, inference=True) as task:
            (scored,) = list(task.batches(output_format="model"))
    for batch in (served, scored):
        assert batch.inputs["history"]["value_ids"].shape[0] == 0
        assert not batch.inputs["history"]["sequence"]["valid"].any()


def _reference_batches(
    artifact: Artifact, rows: Dataset, requests: list[SampleRequest]
) -> dict[int, Batch]:
    with prepare(artifact.contract, rows, InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=artifact.fitted_state, inference=True) as task:
            batches = task.batches(batch_size=1, seed=0)
            return {int(batch.metadata["sample_ids"][0]): batch for batch in batches}


def _seconds(values: pa.ChunkedArray) -> npt.NDArray[np.int64]:
    return values.cast(pa.timestamp("s")).cast(pa.int64()).to_numpy()


def _purchase_times(hm: HMDataset) -> dict[int, npt.NDArray[np.int64]]:
    table = pq.read_table(hm.table("transactions"), columns=["customer_id", "t_dat"])
    customers = table["customer_id"].to_numpy()
    times = _seconds(table["t_dat"])
    order = np.lexsort((times, customers))
    customers, times = customers[order], times[order]
    starts = np.flatnonzero(np.r_[True, customers[1:] != customers[:-1]])
    return {
        int(customers[start]): times[start:stop]
        for start, stop in zip(starts, np.r_[starts[1:], customers.size], strict=True)
    }


def _recency_auroc(hm: HMDataset) -> float:
    """Test AUROC of "the more recent the last purchase, the likelier the next one"."""

    anchors = pq.read_table(hm.anchors)
    test = pc.equal(anchors["split"], "test").to_numpy(zero_copy_only=False)
    times = _purchase_times(hm)
    scores = []
    for customer, cutoff in zip(
        anchors["customer_id"].to_numpy()[test], _seconds(anchors["cutoff"])[test], strict=True
    ):
        past = times.get(int(customer), np.empty(0, np.int64))
        past = past[past < cutoff]
        scores.append(float(past.max() - cutoff) if past.size else -math.inf)
    labels = pq.read_table(hm.labels, columns=["label"])["label"].to_numpy()[test]
    return auroc(labels, np.array(scores))


@dataclass(frozen=True)
class GnnInputs:
    """What one anchor's GNN input holds, by the rows' own times."""

    columns: list[tuple[float, bool]]
    purchase_times: list[int]
    # (article_id, listed_at) of every sampled article.
    articles: list[tuple[int, int]]
    # The distinct articles the sampled purchases reference.
    purchased_articles: list[int]


def _gnn_inputs(artifact: Artifact, data: Path, index: Path) -> dict[int, GnnInputs]:
    """Per anchor: its customer columns, its sampled purchases' times and its articles."""

    transactions = pq.read_table(data / "tables" / "transactions.parquet")
    article_of = dict(
        zip(
            transactions["transaction_id"].to_pylist(),
            transactions["article_id"].to_pylist(),
            strict=True,
        )
    )
    times = {
        name: dict(
            zip(
                table[key].to_pylist(),
                _seconds(table[clock]).tolist(),
                strict=True,
            )
        )
        for name, key, clock in (
            ("transactions", "transaction_id", "t_dat"),
            ("articles", "article_id", "listed_at"),
        )
        for table in [pq.read_table(data / "tables" / f"{name}.parquet")]
    }
    shutil.rmtree(index, ignore_errors=True)
    found: dict[int, GnnInputs] = {}
    with prepare(artifact.contract, dataset(data), DiskGraphEngine(index=index)) as plan:
        with plan.materialize(anchors(data), fitted=artifact.fitted_state, inference=True) as task:
            for batch in task.batches(split="test", output_format="model"):
                found.update(_per_sample(batch, times, article_of))
    return found


def _per_sample(
    batch: Batch, times: Mapping[str, Mapping[int, int]], article_of: Mapping[int, int]
) -> dict[int, GnnInputs]:
    customer = batch.inputs["customer"]
    nodes = batch.inputs["graph"]["nodes"]
    purchases, articles = nodes["transactions"], nodes["articles"]
    own: list[list[int]] = [[] for _ in batch.metadata["sample_ids"]]
    for owner, purchase in zip(purchases["batch"].tolist(), purchases["ids"].tolist(), strict=True):
        own[owner].append(purchase)
    bought: list[list[tuple[int, int]]] = [[] for _ in batch.metadata["sample_ids"]]
    for owner, article in zip(articles["batch"].tolist(), articles["ids"].tolist(), strict=True):
        bought[owner].append((article, times["articles"][article]))
    result = {}
    for row, sample_id in enumerate(batch.metadata["sample_ids"]):
        columns = [
            (float(field["values"][row]), bool(field["valid"][row]))
            for _, field in sorted(customer.items())
        ]
        result[int(sample_id)] = GnnInputs(
            columns,
            sorted(times["transactions"][i] for i in own[row]),
            sorted(bought[row]),
            sorted({article_of[i] for i in own[row]}),
        )
    return result


def _epoch(value: datetime) -> int:
    return int((value - datetime(1970, 1, 1)).total_seconds())


def _serving_cutoff(hm: HMDataset) -> datetime:
    return datetime.fromisoformat(hm.metadata.cutoffs["test"][-1])


def _cuts_a_tie(times: Mapping[int, npt.NDArray[np.int64]], customer: int, cutoff: int) -> bool:
    """Whether the 16 most recent purchases before the cutoff end inside a session."""

    past = np.sort(times.get(customer, np.empty(0, np.int64)))
    past = past[past < cutoff][::-1]
    return past.size > RECENT_PURCHASES and past[RECENT_PURCHASES - 1] == past[RECENT_PURCHASES]


def _test_customers(hm: HMDataset) -> dict[int, int]:
    anchors = pq.read_table(hm.anchors, columns=["sample_id", "customer_id"])
    return dict(
        zip(anchors["sample_id"].to_pylist(), anchors["customer_id"].to_pylist(), strict=True)
    )


def _tie_cut_requests(hm: HMDataset) -> dict[int, SampleRequest]:
    times = _purchase_times(hm)
    return {
        int(request.sample_id): request
        for request in serving_requests(hm.root, CustomerRows.open(hm.root), count=16)
        if _cuts_a_tie(times, request.entity_id, _epoch(request.as_of_time))
    }


def _request_without_history(hm: HMDataset) -> SampleRequest:
    cutoff = _serving_cutoff(hm)
    anchors = pq.read_table(hm.anchors)
    at = anchors.filter(pc.equal(anchors["cutoff"], pa.scalar(cutoff, anchors["cutoff"].type)))
    times = _purchase_times(hm)
    limit = _epoch(cutoff)
    for sample_id, customer in zip(
        at["sample_id"].to_pylist(), at["customer_id"].to_pylist(), strict=True
    ):
        past = times.get(customer, np.empty(0, np.int64))
        if not (past < limit).any():
            return SampleRequest(sample_id, customer, cutoff)
    raise AssertionError("every customer bought before the last test cutoff")


def _dataset_at(hm: HMDataset, cutoff: datetime, out: Path, *, future: bool) -> Path:
    """The test anchors at ``cutoff``; with ``future``, purchases after it added."""

    shutil.rmtree(out, ignore_errors=True)
    (out / "tables").mkdir(parents=True)
    shutil.copy(hm.table("customers"), out / "tables" / "customers.parquet")
    anchors = pq.read_table(hm.anchors)
    at = pc.equal(anchors["cutoff"], pa.scalar(cutoff, anchors["cutoff"].type))
    pq.write_table(
        anchors.filter(pc.and_(at, pc.equal(anchors["split"], "test"))),
        out / "anchors.parquet",
    )
    articles = pq.read_table(hm.table("articles"))
    transactions = pq.read_table(hm.table("transactions"))
    if future:
        added = _future_articles(articles, cutoff)
        articles = pa.concat_tables([articles, added])
        transactions = pa.concat_tables(
            [transactions, _future_rows(transactions, added["article_id"].to_pylist(), cutoff)]
        )
    pq.write_table(articles, out / "tables" / "articles.parquet")
    pq.write_table(transactions, out / "tables" / "transactions.parquet")
    return out


# A row exactly at the cutoff is history (the cutoff is inclusive), so none is added there.
FUTURE_OFFSETS = (
    timedelta(seconds=1),
    timedelta(days=3, hours=12),
    timedelta(days=7) - timedelta(seconds=1),
    timedelta(days=30),
)


def _future_articles(articles: pa.Table, cutoff: datetime) -> pa.Table:
    """New articles listed after the cutoff, copying existing attributes."""

    first = int(pc.max(articles["article_id"]).as_py()) + 1
    added = articles.slice(0, len(FUTURE_OFFSETS))
    added = added.set_column(
        added.schema.get_field_index("article_id"),
        "article_id",
        pa.array(range(first, first + len(FUTURE_OFFSETS)), pa.int64()),
    )
    return added.set_column(
        added.schema.get_field_index("listed_at"),
        "listed_at",
        pa.array([cutoff + offset for offset in FUTURE_OFFSETS], articles["listed_at"].type),
    )


def _future_rows(transactions: pa.Table, articles: list[int], cutoff: datetime) -> pa.Table:
    """For every customer, purchases just after the cutoff, inside the label window
    and after it, of articles listed at the same times: inference must read none."""

    customers = sorted(set(transactions["customer_id"].to_pylist()))
    first = int(pc.max(transactions["transaction_id"]).as_py()) + 1
    rows = [
        (customer, cutoff + offset, article)
        for customer in customers
        for offset, article in zip(FUTURE_OFFSETS, articles, strict=True)
    ]
    return pa.table(
        {
            "transaction_id": pa.array(range(first, first + len(rows)), pa.int64()),
            "t_dat": pa.array([at for _, at, _ in rows], transactions["t_dat"].type),
            "customer_id": pa.array([c for c, _, _ in rows], pa.int64()),
            "article_id": pa.array([a for _, _, a in rows], pa.int64()),
            "price": pa.array([0.05] * len(rows), pa.float64()),
            "sales_channel_id": pa.array([1] * len(rows), pa.int64()),
        }
    ).select(transactions.column_names)
