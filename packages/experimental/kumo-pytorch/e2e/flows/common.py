# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What both H&M flows share: the tables, the anchors, the caller's row lookup.

A flow reads the directory `e2e.datasets.hm` writes. Everything here is what a
user's own pipeline would hold: how the Parquet files map to a kumo-pytorch schema,
where the anchors come from, how a request handler looks up one customer's rows,
the training loop, and how predictions are keyed and written.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch import nn

from e2e.metrics import auroc
from kumo.pytorch.api import (
    NON_TEMPORAL,
    AnchorTable,
    Artifact,
    Batch,
    DataSchema,
    Dataset,
    DiskGraphEngine,
    DType,
    Parquet,
    RelationSpec,
    Rows,
    SampleRequest,
    ServingAssembler,
    TableSpec,
    Task,
    TimeWindow,
    prepare,
)
from kumo.pytorch.data.records import cutoff_to_epoch
from kumo.pytorch.serving import RowLimit, ServingRowsSpec

DAY = 86_400
HORIZON_DAYS = 7
TABLES = ("customers", "articles", "transactions")

INT64, FLOAT64, STRING = DType("int64"), DType("float64"), DType("string")

# Timestamp columns are left for the Parquet metadata to type (contract.resolve).
SCHEMA = DataSchema(
    tables=(
        TableSpec(
            "customers",
            "customer_id",
            {
                "customer_id": INT64,
                "age": INT64,
                "club_member_status": STRING,
                "fashion_news_frequency": STRING,
                "postal_area": STRING,
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "articles",
            "article_id",
            {
                "article_id": INT64,
                "product_type_name": STRING,
                "product_group_name": STRING,
                "colour_group_name": STRING,
                "index_group_name": STRING,
                "department_name": STRING,
                "price_band": STRING,
                "base_price": FLOAT64,
                "listed_at": None,
            },
            time_column="listed_at",
        ),
        TableSpec(
            "transactions",
            "transaction_id",
            {
                "transaction_id": INT64,
                "t_dat": None,
                "customer_id": INT64,
                "article_id": INT64,
                "price": FLOAT64,
                "sales_channel_id": INT64,
            },
            time_column="t_dat",
        ),
    ),
    relations=(
        RelationSpec(
            "customer_transactions", "customers", "customer_id", "transactions", "customer_id"
        ),
        RelationSpec(
            "article_transactions", "articles", "article_id", "transactions", "article_id"
        ),
    ),
)


def purchases(start: int | None, stop: int) -> Rows:
    """The customer's transactions in ``(cutoff + start, cutoff + stop]``."""

    return Rows(("customer_transactions",), TimeWindow("t_dat", start, stop))


# The label: any purchase in the 7 days from the cutoff. Training reads these
# rows; inference never does.
NEXT_WEEK = purchases(0, HORIZON_DAYS * DAY)


def dataset(data: Path) -> Dataset:
    return Dataset({name: Parquet(str(data / "tables" / f"{name}.parquet")) for name in TABLES})


def anchors(data: Path) -> AnchorTable:
    """The anchor table upstream produced: one row per customer per weekly cutoff."""

    return AnchorTable(
        source=Parquet(str(data / "anchors.parquet")),
        entity_id="customer_id",
        as_of_time="cutoff",
        sample_id="sample_id",
        split="split",
    )


def is_positive(batch: Batch) -> torch.Tensor:
    """The binary label from the generated purchase count."""

    return (batch.targets["label"]["values"] > 0).float()


class BatchDigest:
    """A running hash of every input tensor and sample ID a pass emits."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def update(self, batch: Batch) -> None:
        self._hash.update(repr(list(batch.metadata["sample_ids"])).encode())
        self._add("inputs", batch.inputs)

    def hexdigest(self) -> str:
        return self._hash.hexdigest()

    def _add(self, path: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                self._add(f"{path}.{key}", value[key])
        elif isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous()
            self._hash.update(f"{path}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            self._hash.update(tensor.numpy().tobytes())
        else:
            self._hash.update(f"{path}={value!r}".encode())


@dataclass
class Predictions:
    """Scores keyed by sample ID, and the digest of the batches that produced them."""

    scores: dict[int, float] = field(default_factory=dict)
    digest: BatchDigest = field(default_factory=BatchDigest)

    def add(self, batch: Batch, probabilities: torch.Tensor) -> None:
        self.digest.update(batch)
        padding = batch.metadata["is_padding"].tolist()
        for sample_id, score, pad in zip(
            batch.metadata["sample_ids"], probabilities.tolist(), padding, strict=True
        ):
            if pad:
                continue
            if sample_id in self.scores:
                raise ValueError(f"sample {sample_id} was scored twice")
            self.scores[int(sample_id)] = float(score)

    def write(self, path: Path) -> None:
        ids = sorted(self.scores)
        table = pa.table(
            {
                "sample_id": pa.array(ids, pa.int64()),
                "score": pa.array([self.scores[i] for i in ids], pa.float64()),
            }
        )
        pq.write_table(table, path)


def read_predictions(path: Path) -> dict[int, float]:
    table = pq.read_table(path)
    return dict(zip(table["sample_id"].to_pylist(), table["score"].to_pylist(), strict=True))


def oracle_auroc(data: Path, predictions: Mapping[int, float]) -> float:
    """AUROC against the generator's labels; a real pipeline has no oracle."""

    labels = pq.read_table(data / "oracle" / "labels.parquet")
    pairs = [
        (label, predictions[sample_id])
        for sample_id, label in zip(
            labels["sample_id"].to_pylist(), labels["label"].to_pylist(), strict=True
        )
        if sample_id in predictions
    ]
    return auroc(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


@dataclass(frozen=True)
class CustomerRows:
    """The caller's row store for serving: what ``rows_spec`` lists, looked up per request."""

    tables: Mapping[str, pa.Table]

    @classmethod
    def open(cls, data: Path) -> CustomerRows:
        return cls({name: pq.read_table(data / "tables" / f"{name}.parquet") for name in TABLES})

    def lookup(self, spec: ServingRowsSpec, request: SampleRequest) -> dict[str, pa.Table]:
        """Follow ``spec`` for one request, hop by hop: each route's join, time window and
        limit; then only the required tables and columns."""

        cutoff = cutoff_to_epoch(request.as_of_time)
        fetched: dict[str, pa.Table] = {}
        routes = sorted(
            ((table, route) for table in spec.tables.values() for route in table.routes),
            key=lambda item: item[1].hop,
        )
        for table, route in routes:
            rows = self.tables[table.name]
            join = route.join
            if join.request_field == "entity_id":
                rows = rows.filter(pc.equal(rows[join.column], request.entity_id))
            elif join.request_field is not None:
                raise ValueError(f"the H&M lookup has no {join.request_field} to join {table.name}")
            else:
                source = fetched[join.from_table][join.from_column]
                rows = rows.filter(pc.is_in(rows[join.column], value_set=source.combine_chunks()))
            if route.time_column is not None:
                window = route.window
                start, stop = (None, 0) if window is None else (window.start, window.stop)
                seconds = _seconds(rows[route.time_column])
                keep = np.ones(len(rows), dtype=bool)
                if start is not None:
                    keep &= seconds > cutoff + start
                if stop is not None:
                    keep &= seconds <= cutoff + stop
                rows = rows.filter(pa.array(keep))
            if route.limit is not None:
                rows = rows.take(_kept(rows, route.limit, join.column))
            if table.name in fetched:
                rows = pa.concat_tables([fetched[table.name], rows])
                key = next(n for n, c in table.columns.items() if c.role == "primary key")
                _, first = np.unique(rows[key].to_numpy(), return_index=True)
                rows = rows.take(np.sort(first))
            fetched[table.name] = rows
        return {
            name: fetched.get(name, self.tables[name].schema.empty_table()).select(
                [column for column, spec_column in table.columns.items() if spec_column.required]
            )
            for name, table in spec.tables.items()
            if table.required
        }


def _seconds(column: pa.ChunkedArray) -> np.ndarray:
    """Whole UTC seconds, rounding a fractional instant up as stored clocks do."""

    if column.null_count:
        raise ValueError(
            "a time column the loader filters on has a NULL; the loader refuses such a row"
        )
    if pa.types.is_date32(column.type):
        return column.cast(pa.int32()).to_numpy().astype(np.int64) * DAY
    per_second = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[column.type.unit]
    values = column.cast(pa.int64()).to_numpy()
    return -(-values // per_second)


def _kept(rows: pa.Table, limit: RowLimit, join_column: str) -> np.ndarray:
    """Positions of the rows a limit keeps per group, every row tied at the last one included."""

    seconds = _seconds(rows[limit.time_column])
    groups = (
        np.zeros(len(rows), dtype=np.int64)
        if limit.per == "request"
        else np.unique(rows[join_column].to_numpy(), return_inverse=True)[1]
    )
    kept: list[int] = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        times = seconds[members]
        ordered = members[np.argsort(-times if limit.order == "newest" else times, kind="stable")]
        if len(ordered) > limit.count:
            boundary = seconds[ordered[limit.count - 1]]
            tied = [i for i in ordered[limit.count :] if seconds[i] == boundary]
            ordered = np.concatenate((ordered[: limit.count], tied)).astype(np.int64)
        kept.extend(ordered.tolist())
    return np.array(sorted(kept), dtype=np.int64)


def serving_requests(data: Path, store: CustomerRows, count: int) -> list[SampleRequest]:
    """The point-lookup customers, then others who bought before, at the last test cutoff.

    Customers with no purchase at or before the cutoff are left out: kumo-pytorch assembles
    their batch, but the flows' models read at least one event per batch.
    """

    metadata = json.loads((data / "metadata.json").read_text())
    table = pq.read_table(data / "anchors.parquet")
    table = table.filter(pc.equal(table["split"], "test"))
    cutoff = pc.max(table["cutoff"])
    table = table.filter(pc.equal(table["cutoff"], cutoff))
    sample_ids = dict(
        zip(table["customer_id"].to_pylist(), table["sample_id"].to_pylist(), strict=True)
    )
    transactions = store.tables["transactions"]
    bought = set(
        transactions.filter(pc.less(transactions["t_dat"], cutoff))["customer_id"].to_pylist()
    )
    chosen = [lookup["customer_id"] for lookup in metadata["point_lookups"]]
    chosen += [c for c in sorted(bought) if c not in chosen][: count - len(chosen)]
    return [SampleRequest(sample_ids[c], c, cutoff.as_py()) for c in chosen]


def score(model: nn.Module, batches: Iterable[Batch]) -> Predictions:
    predictions = Predictions()
    model.eval()
    with torch.no_grad():
        for batch in batches:
            predictions.add(batch, torch.sigmoid(model(batch)))
    return predictions


def validation_auroc(model: nn.Module, task: Task) -> float:
    labels, scores = [], []
    model.eval()
    with torch.no_grad():
        for batch in task.batches(split="val", output_format="model"):
            labels.append(is_positive(batch))
            scores.append(torch.sigmoid(model(batch)))
    return auroc(torch.cat(labels).bool().numpy(), torch.cat(scores).double().numpy())


def fit_classifier(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task: Task,
    *,
    seed: int,
    max_epochs: int = 12,
    patience: int = 3,
) -> tuple[list[dict[str, float]], float]:
    """A plain loop over the train split, keeping the weights with the best val AUROC."""

    history: list[dict[str, float]] = []
    best, best_state, stale = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in task.batches(
            split="train", epoch=epoch, seed=seed, shuffle=True, output_format="model"
        ):
            loss = nn.functional.binary_cross_entropy_with_logits(model(batch), is_positive(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val = validation_auroc(model, task)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auroc": val})
        print(f"train: epoch {epoch} loss {np.mean(losses):.4f} val AUROC {val:.4f}")
        if val > best:
            best, stale = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale == patience:
                break
    model.load_state_dict(best_state)
    return history, best


def batch_inference(artifact: Artifact, model: nn.Module, data: Path, out: Path) -> None:
    """Score the test anchors in inference mode, keyed by sample ID, in this process."""

    clock = Stopwatch()
    out.mkdir(parents=True, exist_ok=True)
    engine = DiskGraphEngine(index=out / "index")
    with prepare(artifact.contract, dataset(data), engine) as plan:
        with plan.materialize(anchors(data), fitted=artifact.fitted_state, inference=True) as task:
            predictions = score(model, task.batches(split="test", output_format="model"))
    clock.lap("infer")
    predictions.write(out / "predictions.parquet")
    write_json(
        out / "infer.json", {"test_digest": predictions.digest.hexdigest(), "seconds": clock.laps}
    )
    print(f"infer: scored {len(predictions.scores)} test anchors")


def serve(artifact: Artifact, model: nn.Module, data: Path) -> None:
    """Answer single requests from the caller's rows, as a request handler would."""

    store = CustomerRows.open(data)
    server = ServingAssembler(artifact)
    print(f"serve: rows to fetch per request\n{server.rows_spec}")
    served: dict[str, float] = {}
    model.eval()
    for request in serving_requests(data, store, count=16):
        (batch,) = server.assemble(store.lookup(server.rows_spec, request), [request])
        with torch.no_grad():
            served[str(request.sample_id)] = torch.sigmoid(model(batch)).item()
    print(f"serve: answered {len(served)} requests")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Stopwatch:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(now - self._start, 3)
        self._start = now
