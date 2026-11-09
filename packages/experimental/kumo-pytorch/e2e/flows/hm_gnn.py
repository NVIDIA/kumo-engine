# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GNN for "does the customer buy in the next 7 days", from training to serving.

    python -m e2e.datasets.hm --scale S --seed 0 --out /tmp/hm-s
    python -m e2e.flows.hm_gnn --data /tmp/hm-s --work /tmp/hm-gnn

runs three stages, each in its own process, on DiskGraph over the generated
Parquet:

1. ``train``: materialize the train/val/test anchors, train a small PyG
   heterogeneous GNN with early stopping on val, then write the artifact and
   the weights. It also scores the test anchors in this process, for the
   lifecycle check that a fresh process scores them identically.
2. ``infer``: load both in a fresh process and score the test anchors in
   inference mode; predictions are keyed by ``sample_id``.
3. ``serve``: load both again; for a few requests, look up the customer's rows
   (the caller's job), let kumo-pytorch assemble the batch, and answer with the same
   forward pass.

kumo-pytorch supplies the batches, the artifact and the request assembly; the model,
the training loop, the checkpoint and the row lookup are this file's.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter

from e2e.flows.common import (
    DAY,
    NEXT_WEEK,
    SCHEMA,
    Stopwatch,
    anchors,
    batch_inference,
    dataset,
    fit_classifier,
    oracle_auroc,
    purchases,
    read_predictions,
    score,
    serve,
    write_json,
)
from kumo.pytorch.api import (
    Aggregate,
    Artifact,
    Batch,
    Batching,
    BatchSpec,
    Columns,
    DataLoaderSpec,
    DiskGraphEngine,
    Encode,
    EntitySource,
    EntitySpec,
    Field,
    Fields,
    Graph,
    GraphLayer,
    GraphSample,
    Input,
    LabelTarget,
    NeighborhoodSource,
    NeighborhoodSpec,
    Raw,
    SampleSpec,
    Separate,
    Vocabulary,
    ZScore,
    load_weights,
    prepare,
    save_weights,
)

BATCH_SIZE = 512
RECENT_PURCHASES = 16
SESSION_WINDOWS = {"sessions_7d": 7, "sessions_30d": 30, "sessions_90d": 90, "sessions_365d": 365}
NUMERICAL = ("age", "recency", *SESSION_WINDOWS)
CATEGORICAL = ("club", "news")
ARTICLE_CATEGORICAL = ("product_type_name", "index_group_name", "colour_group_name")


def contract() -> DataLoaderSpec:
    sample = SampleSpec(
        entity=EntitySpec("customers", entity_type="customer"),
        entity_values={
            "age": Field(column="age"),
            "club": Field(column="club_member_status"),
            "news": Field(column="fashion_news_frequency"),
            "recency": Aggregate(purchases(None, 0), "recency", "t_dat"),
            # A session's purchases share one timestamp, so distinct times count visits.
            **{
                name: Aggregate(purchases(-days * DAY, 0), "count_distinct", "t_dat")
                for name, days in SESSION_WINDOWS.items()
            },
        },
        # The customer's most recent purchases before the cutoff, then each one's article.
        neighborhoods={
            "recent": NeighborhoodSpec(
                layers=(
                    GraphLayer(
                        (GraphSample("customer_transactions", RECENT_PURCHASES, strategy="recent"),)
                    ),
                    GraphLayer(
                        (
                            GraphSample(
                                "article_transactions", 1, direction="reverse", strategy="recent"
                            ),
                        )
                    ),
                )
            )
        },
        targets={"purchases_next_week": Aggregate(NEXT_WEEK, "count")},
    )
    batch = BatchSpec(
        inputs={
            "customer": Input(
                EntitySource(),
                Fields(
                    (
                        *(Encode(name, ZScore()) for name in NUMERICAL),
                        *(Encode(name, Vocabulary()) for name in CATEGORICAL),
                    ),
                    row=Columns(),
                ),
                Separate(),
            ),
            "graph": Input(
                NeighborhoodSource("recent"),
                Graph(
                    node_fields={
                        "transactions": Fields(
                            (Encode("price", ZScore()), Encode("sales_channel_id", Vocabulary()))
                        ),
                        "articles": Fields(
                            (
                                Encode("base_price", ZScore()),
                                *(Encode(name, Vocabulary()) for name in ARTICLE_CATEGORICAL),
                            )
                        ),
                    }
                ),
                None,
            ),
        },
        targets={"label": LabelTarget("purchases_next_week", Raw())},
        batching=Batching(max_samples=BATCH_SIZE),
    )
    return DataLoaderSpec(SCHEMA, sample, batch)


EDGE_TYPES = (
    ("customers", "customer_transactions", "transactions"),
    ("transactions", "rev_customer_transactions", "customers"),
    ("articles", "article_transactions", "transactions"),
    ("transactions", "rev_article_transactions", "articles"),
)


class PurchaseGNN(nn.Module):
    """Customer features plus two rounds of message passing over customer -> purchase -> article."""

    def __init__(self, output_schema: Mapping[str, Any], hidden: int = 64) -> None:
        super().__init__()
        customer = output_schema["inputs"]["customer"]
        node_fields = output_schema["inputs"]["graph"]
        purchase, article = node_fields["transactions"], node_fields["articles"]
        self.categorical = nn.ModuleDict(
            {name: nn.Embedding(customer[name]["vocabulary_size"], 8) for name in CATEGORICAL}
        )
        self.customer_in = nn.Linear(2 * len(NUMERICAL) + 8 * len(CATEGORICAL), hidden)
        self.channel = nn.Embedding(purchase["sales_channel_id"]["vocabulary_size"], 4)
        self.purchase_in = nn.Linear(1 + 4, hidden)
        self.article_categorical = nn.ModuleDict(
            {
                name: nn.Embedding(article[name]["vocabulary_size"], 8)
                for name in ARTICLE_CATEGORICAL
            }
        )
        self.article_in = nn.Linear(1 + 8 * len(ARTICLE_CATEGORICAL), hidden)
        self.layers = nn.ModuleList(
            HeteroConv({edge: SAGEConv(hidden, hidden) for edge in EDGE_TYPES}) for _ in range(2)
        )
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: Batch) -> torch.Tensor:
        customer, graph = batch.inputs["customer"], batch.inputs["graph"]
        numbers = [customer[name]["values"] for name in NUMERICAL]
        present = [customer[name]["valid"].float() for name in NUMERICAL]
        codes = [self.categorical[name](customer[name]["values"]) for name in CATEGORICAL]
        features = torch.relu(
            self.customer_in(torch.cat([torch.stack(numbers + present, -1), *codes], -1))
        )

        nodes = graph["nodes"]
        root = nodes["customers"]["batch"]
        fields = nodes["transactions"]["features"]
        purchase = torch.cat(
            (
                fields["price"]["values"].unsqueeze(-1),
                self.channel(fields["sales_channel_id"]["values"]),
            ),
            -1,
        )
        fields = nodes["articles"]["features"]
        article = torch.cat(
            (
                fields["base_price"]["values"].unsqueeze(-1),
                *(
                    self.article_categorical[name](fields[name]["values"])
                    for name in ARTICLE_CATEGORICAL
                ),
            ),
            -1,
        )
        x = {
            "customers": features[root],
            "transactions": torch.relu(self.purchase_in(purchase)),
            "articles": torch.relu(self.article_in(article)),
        }
        bought = graph["edges"]["customer_transactions"]["index"]
        of_article = graph["edges"]["article_transactions"]["index"]
        edges = {
            EDGE_TYPES[0]: bought,
            EDGE_TYPES[1]: bought.flip(0),
            EDGE_TYPES[2]: of_article,
            EDGE_TYPES[3]: of_article.flip(0),
        }
        # Two rounds carry article information to the customer through each purchase.
        for layer in self.layers:
            x = {**x, **{name: torch.relu(value) for name, value in layer(x, edges).items()}}
        pooled = scatter(x["customers"], root, dim=0, dim_size=batch.batch_size, reduce="mean")
        return self.head(torch.cat((pooled, features), -1)).squeeze(-1)


def train(data: Path, work: Path, seed: int) -> None:
    torch.manual_seed(seed)
    clock = Stopwatch()
    engine = DiskGraphEngine(index=work / "index")
    with prepare(contract(), dataset(data), engine) as plan:
        clock.lap("prepare")
        with plan.materialize(anchors(data), fit_split="train") as task:
            clock.lap("materialize")
            model = PurchaseGNN(task.output_schema)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            history, best = fit_classifier(model, optimizer, task, seed=seed)
            clock.lap("fit")
            task.to_artifact().save(work / "artifact")
            save_weights(work / "model.pt", model.state_dict(), task.fitted_state)
            fitted = task.fitted_state
        # The same scoring a fresh process does, here in the training process.
        with plan.materialize(anchors(data), fitted=fitted, inference=True) as test:
            predictions = score(model, test.batches(split="test", output_format="model"))
        clock.lap("score_test")
    predictions.write(work / "predictions-train-process.parquet")
    write_json(
        work / "train.json",
        {
            "seed": seed,
            "history": history,
            "best_val_auroc": best,
            "test_digest": predictions.digest.hexdigest(),
            "seconds": clock.laps,
        },
    )
    print(f"train: best val AUROC {best:.4f}; artifact and weights written")


def load_model(work: Path) -> tuple[Artifact, PurchaseGNN]:
    artifact = Artifact.load(work / "artifact")
    model = PurchaseGNN(artifact.output_schema)
    model.load_state_dict(load_weights(work / "model.pt", artifact.fitted_state))
    return artifact, model.eval()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", nargs="?", choices=("train", "infer", "serve"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="where infer writes; defaults to --work")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        train(args.data, args.work, args.seed)
    elif args.stage == "infer":
        batch_inference(*load_model(args.work), args.data, args.out or args.work)
    elif args.stage == "serve":
        serve(*load_model(args.work), args.data)
    else:
        common = ["--data", str(args.data), "--work", str(args.work)]
        for stage in ("train", "infer", "serve"):
            extra = ["--seed", str(args.seed)] if stage == "train" else []
            subprocess.run(
                [sys.executable, "-m", __spec__.name, stage, *common, *extra], check=True
            )
        predictions = read_predictions(args.work / "predictions.parquet")
        print(
            f"test AUROC against the generator's labels: {oracle_auroc(args.data, predictions):.4f}"
        )


if __name__ == "__main__":
    main()
