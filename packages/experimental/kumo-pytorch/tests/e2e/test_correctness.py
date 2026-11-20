# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""H&M correctness checks over shared saved runs."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("diskgraph")
pytest.importorskip("torch_geometric")

from e2e.datasets.hm import HMDataset  # noqa: E402
from e2e.flows.common import CustomerRows, anchors, dataset  # noqa: E402
from e2e.flows.hm_gnn import RECENT_PURCHASES  # noqa: E402
from kumo.pytorch.api import (  # noqa: E402
    Artifact,
    Batch,
    DiskGraphEngine,
    ServingAssembler,
    prepare,
)
from tests.e2e.hm_support import (  # noqa: E402
    REORDER_TOLERANCE,
    FutureScores,
    Report,
    Run,
    _cuts_a_tie,
    _dataset_at,
    _epoch,
    _future_articles,
    _gnn_inputs,
    _purchase_times,
    _request_without_history,
    _test_customers,
    _tie_cut_requests,
)


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


def test_known_gap_recent_sampling_ties_depend_on_the_engine_and_on_later_rows(
    runs: dict[str, list[Run]],
    future_scores: FutureScores,
    hm: HMDataset,
    root: Path,
    reference: dict[str, dict[int, Batch]],
) -> None:
    """Recent sampling orders by time, then primary key (docs/limitations.md).
    When the 16th and 17th most recent purchases share a timestamp, DiskGraph's choice
    follows its index, so rows added after the cutoff can change it.
    The GNN's DiskGraph checks above skip those
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
    native = reference["gnn"][sample_id]
    engine = DiskGraphEngine(index=root / "tie-index")
    with prepare(artifact.contract, dataset(hm.root), engine) as plan:
        with plan.materialize(
            [tied_requests[sample_id]], fitted=artifact.fitted_state, inference=True
        ) as task:
            (batch,) = list(task.batches(output_format="model"))
    purchases = [
        set(b.inputs["graph"]["nodes"]["transactions"]["ids"].tolist()) for b in (native, batch)
    ]
    assert len(purchases[0]) == len(purchases[1]) == RECENT_PURCHASES
    # A tied boundary permits different identities, but never an older event.
    table = next(table for table in artifact.contract.schema.tables if table.name == "transactions")
    requested = tied_requests[sample_id]
    eligible = {
        row[table.primary_key]: _epoch(row[table.time_column])
        for row in CustomerRows.open(hm.root).tables["transactions"].to_pylist()
        if row["customer_id"] == requested.entity_id
        and _epoch(row[table.time_column]) <= _epoch(requested.as_of_time)
    }
    boundary = sorted(eligible.values(), reverse=True)[RECENT_PURCHASES - 1]
    required = {key for key, at in eligible.items() if at > boundary}
    allowed = {key for key, at in eligible.items() if at >= boundary}
    assert all(required <= ids <= allowed for ids in purchases)


def test_a_history_batch_without_any_event_assembles(
    runs: dict[str, list[Run]], hm: HMDataset, root: Path
) -> None:
    """A customer with no purchase before the cutoff gets a batch with no event, served
    or scored alone, on DiskGraph. The flows still serve only customers who
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
