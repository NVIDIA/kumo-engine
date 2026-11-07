# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shop app of the inference-and-serving guide, with its three stages.

    python -m tests.lifecycle_app [WORKDIR]

runs ``train``, ``infer`` and ``serve`` each in its own process, the way a
deployment would. The definitions from ``DAY`` through ``rows_for`` are the
guide's ``shop.py`` block verbatim; ``tests/test_inference_lifecycle.py`` checks
that they stay identical.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kumo.pytorch import api

DAY = 86_400
TRAIN_AT, SCORE_AT = 20 * DAY, 34 * DAY
SEGMENTS = ("loyal", "casual", "dormant")
RATES = {"loyal": 0.8, "casual": 0.25, "dormant": 0.03}
CATEGORIES = ("books", "games", "music")
Tables = dict[str, list[dict[str, Any]]]


def tables() -> Tables:
    """Synthetic users whose segment sets how often they buy: the planted signal."""
    rng = random.Random(0)
    users, purchases = [], []
    for user_id in range(1, 61):
        segment = SEGMENTS[user_id % 3]
        users.append({"user_id": user_id, "segment": segment})
        for day in range(40):
            if rng.random() < RATES[segment] / 3:
                purchases.append(
                    {
                        "purchase_id": len(purchases) + 1,
                        "user_id": user_id,
                        "at": day * DAY + rng.randrange(DAY),
                        "amount": round(rng.uniform(2.0, 60.0), 2),
                        "category": rng.choice(CATEGORIES),
                    }
                )
    return {"users": users, "purchases": purchases}


def contract() -> api.DataLoaderSpec:
    users = {"user_id": api.DType("int64"), "segment": api.DType("string")}
    purchases = {
        "purchase_id": api.DType("int64"),
        "user_id": api.DType("int64"),
        "at": api.DType("timestamp", unit="s", timezone="UTC"),
        "amount": api.DType("float64"),
        "category": api.DType("string"),
    }
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "user_id", users, time_column=api.NON_TEMPORAL),
            api.TableSpec("purchases", "purchase_id", purchases, time_column="at"),
        ),
        relations=(api.RelationSpec("user_purchases", "users", "user_id", "purchases", "user_id"),),
    )
    history = api.Rows(("user_purchases",), api.TimeWindow("at", stop=0))  # up to the cutoff
    next_week = api.Rows(("user_purchases",), api.TimeWindow("at", start=0, stop=7 * DAY))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        entity_values={
            "segment": api.Field(column="segment"),
            "purchases_so_far": api.Aggregate(history, "count", dtype=api.DType("int64")),
        },
        events={
            "purchases": api.EventSpec(
                history,
                fields={
                    "amount": api.Field(column="amount"),
                    "category": api.Field(column="category"),
                },
                limit=8,
                order="descending",
            )
        },
        # Read from rows after the cutoff: computed in training, never in inference.
        targets={
            "purchases_next_week": api.Aggregate(next_week, "count", dtype=api.DType("int64"))
        },
    )
    batch = api.BatchSpec(
        inputs={
            "user": api.Input(
                api.EntitySource(),
                api.Fields(
                    (
                        api.Encode("segment", api.Vocabulary()),
                        api.Encode("purchases_so_far", api.ZScore()),
                    ),
                    row=api.Columns(),
                ),
                api.Separate(),
            ),
            "recent": api.Input(
                api.EventSource(("purchases",)),
                api.Fields(
                    (api.Encode("amount", api.ZScore()), api.Encode("category", api.Vocabulary())),
                    row=api.Columns(),
                ),
                api.Padded(),
            ),
        },
        targets={"active": api.LabelTarget("purchases_next_week", api.Raw())},
        batching=api.Batching(max_samples=16),
    )
    return api.DataLoaderSpec(schema, sample, batch)


class ActivityModel(nn.Module):
    """A user model over the batch as the package emits it, sized from the fitted state."""

    def __init__(self, output_schema: Mapping[str, Any]) -> None:
        super().__init__()
        user = output_schema["inputs"]["user"]
        recent = output_schema["inputs"]["recent"]
        self.segment = nn.Embedding(user["segment"]["vocabulary_size"], 8)
        self.category = nn.Embedding(recent["category"]["vocabulary_size"], 8)
        self.head = nn.Sequential(nn.Linear(8 + 1 + 8 + 1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, batch: api.Batch) -> torch.Tensor:
        user, recent = batch.inputs["user"], batch.inputs["recent"]
        valid = recent["valid"].unsqueeze(-1).float()
        events = torch.cat(
            (
                self.category(recent["category"]["values"]),
                recent["amount"]["values"].unsqueeze(-1),
            ),
            dim=-1,
        )
        pooled = (events * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        features = torch.cat(
            (
                self.segment(user["segment"]["values"]),
                user["purchases_so_far"]["values"].unsqueeze(-1),
                pooled,
            ),
            dim=-1,
        )
        return self.head(features).squeeze(-1)


def anchors(at: int, *, split: str | None) -> list[api.SampleRequest]:
    return [
        api.SampleRequest(f"user-{user_id}-at-{at // DAY}", user_id, at, split=split)
        for user_id in range(1, 61)
    ]


def dataset(rows: Tables) -> api.Dataset:
    return api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})


def rows_for(user_id: int, as_of: int) -> Tables:
    """The caller's lookup of what `rows_spec` lists: the user and purchases up to `as_of`."""
    rows = tables()
    return {
        "users": [row for row in rows["users"] if row["user_id"] == user_id],
        "purchases": [
            row for row in rows["purchases"] if row["user_id"] == user_id and row["at"] <= as_of
        ],
    }


def train(workdir: Path) -> None:
    torch.manual_seed(0)
    with api.prepare(contract(), dataset(tables()), api.InMemoryEngine()) as plan:
        with plan.materialize(anchors(TRAIN_AT, split="train"), fit_split="train") as task:
            model = ActivityModel(task.output_schema)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
            for epoch in range(30):
                for batch in task.batches(split="train", epoch=epoch, shuffle=True):
                    labels = (batch.targets["active"]["values"] > 0).float()
                    loss = nn.functional.binary_cross_entropy_with_logits(model(batch), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            task.to_artifact().save(workdir / "artifact")
            api.save_weights(workdir / "model.pt", model.state_dict(), task.fitted_state)
    print(f"train: final loss {loss.item():.3f}; artifact and weights written")


def describe(batch: api.Batch) -> None:
    """Print what an inference batch holds: input tensors, targets and sample IDs."""

    def leaves(prefix: str, value: Any) -> Iterator[tuple[str, torch.Tensor]]:
        if isinstance(value, Mapping):
            for key, inner in value.items():
                yield from leaves(f"{prefix}.{key}", inner)
        else:
            yield prefix, value

    lines = [
        (path, f"{tuple(tensor.shape)} {tensor.dtype}")
        for path, tensor in leaves("inputs", batch.inputs)
    ]
    lines.append(("targets", repr(batch.targets)))
    lines.append(("metadata['sample_ids'][:3]", repr(batch.metadata["sample_ids"][:3])))
    for label, text in lines:
        print(f"  {label:<36} {text}")


def load_model(workdir: Path) -> tuple[api.Artifact, ActivityModel]:
    artifact = api.Artifact.load(workdir / "artifact")
    model = ActivityModel(artifact.output_schema)
    model.load_state_dict(api.load_weights(workdir / "model.pt", artifact.fitted_state))
    return artifact, model.eval()


def infer(workdir: Path) -> None:
    artifact, model = load_model(workdir)
    predictions: dict[str, float] = {}
    with api.prepare(artifact.contract, dataset(tables()), api.InMemoryEngine()) as plan:
        scoring = anchors(SCORE_AT, split=None)
        with plan.materialize(scoring, fitted=artifact.fitted_state, inference=True) as task:
            with torch.no_grad():
                for batch in task.batches():
                    if not predictions:
                        print("infer: first batch")
                        describe(batch)
                    scores = torch.sigmoid(model(batch)).tolist()
                    predictions.update(zip(batch.metadata["sample_ids"], scores, strict=True))
    (workdir / "predictions.json").write_text(json.dumps(predictions, indent=2, sort_keys=True))
    by_segment = {segment: 0.0 for segment in SEGMENTS}
    for user_id in range(1, 61):
        by_segment[SEGMENTS[user_id % 3]] += (
            predictions[f"user-{user_id}-at-{SCORE_AT // DAY}"] / 20
        )
    print("infer: mean score by segment", {k: round(v, 2) for k, v in by_segment.items()})


def serve(workdir: Path) -> None:
    artifact, model = load_model(workdir)
    server = api.ServingAssembler(artifact)
    print(server.rows_spec)
    predictions = json.loads((workdir / "predictions.json").read_text())
    matches = []
    for user_id in (1, 2, 3):
        request = api.SampleRequest(f"user-{user_id}-at-{SCORE_AT // DAY}", user_id, SCORE_AT)
        (batch,) = server.assemble(rows_for(user_id, SCORE_AT), [request])
        with torch.no_grad():
            score = torch.sigmoid(model(batch)).item()
        # A forward pass over a batch of one may differ from one over sixteen in the last bits.
        matches.append(math.isclose(score, predictions[request.sample_id], abs_tol=1e-6))
        print(f"serve: {request.sample_id} scores {score:.3f}")
    print("serving equals batch inference:", all(matches))


def main() -> None:
    stages = {"train": train, "infer": infer, "serve": serve}
    if len(sys.argv) == 3 and sys.argv[1] in stages:
        stages[sys.argv[1]](Path(sys.argv[2]))
        return
    workdir = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(tempfile.mkdtemp())
    for stage in stages:
        subprocess.run(
            [sys.executable, "-m", "tests.lifecycle_app", stage, str(workdir)], check=True
        )


if __name__ == "__main__":
    main()
