# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared H&M fixtures and helpers; all check groups reuse the same saved runs."""

from __future__ import annotations

import json
import math
import shutil
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
from e2e.flows.hm_gnn import RECENT_PURCHASES  # noqa: E402
from e2e.metrics import auroc  # noqa: E402
from kumo.pytorch.api import (  # noqa: E402
    Artifact,
    Batch,
    Dataset,
    DiskGraphEngine,
    SampleRequest,
    prepare,
)
from tests.native_fixtures import prepare_rows  # noqa: E402

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


def _reference_batches(
    artifact: Artifact, rows: Dataset, requests: list[SampleRequest]
) -> dict[int, Batch]:
    with prepare_rows(artifact.contract, rows) as plan:
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
