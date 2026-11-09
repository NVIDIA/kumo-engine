# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PRAGMA: pre-train on transaction histories, LoRA fine-tune, then infer and serve.

    python -m e2e.datasets.hm --scale S --seed 0 --out /tmp/hm-s
    python -m e2e.flows.hm_pragma --data /tmp/hm-s --work /tmp/hm-pragma

runs four stages, each in its own process, on DiskGraph over the generated
Parquet:

1. ``pretrain``: tokenize each customer's profile and recent transactions as
   field-name/value tokens, train a small hierarchical encoder (fields → event,
   events → history) to recover masked values, and save the artifact and the
   encoder weights.
2. ``finetune``: start from the pre-trained artifact's bound contract (every
   token ID pinned), swap the masked-value target for the 7-day purchase label,
   freeze the encoder, add LoRA adapters and a classifier head, and train them
   with early stopping on val. It also scores the test anchors in this process.
3. ``infer``: load the fine-tuned artifact and weights in a fresh process and
   score the test anchors in inference mode.
4. ``serve``: load them again and answer a few requests from caller-supplied
   rows.

The encoder is a scaled-down PRAGMA (width 64, one field layer, two history
layers) so the flow runs on a CPU in minutes; the lifecycle is the point.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

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
    Buckets,
    CalendarPart,
    DataLoaderSpec,
    Derived,
    DiskGraphEngine,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    EventsOf,
    EventSource,
    EventSpec,
    EventTimeDelta,
    Field,
    FieldNames,
    Fields,
    IdSpace,
    Input,
    LabelTarget,
    MaskedValuesTarget,
    Nested,
    Padded,
    Raw,
    SampleSpec,
    Tokens,
    Vocabulary,
    load_weights,
    prepare,
    save_weights,
)
from kumo.pytorch.models.fusion import add_transformer_lora
from kumo.pytorch.models.layers import TransformerStack

CALENDAR = ("weekday", "hour")
ROLES = ("pad", "unknown", "missing", "mask", "entity", "event")
NUMERICAL_BUCKETS = 16
INT64 = DType("int64")
MAX_EVENTS = 48
WIDTH = 64
LORA_RANK = 8
# Measured at M: 3 and 8 epochs score 0.0005 higher test AUROC and take 1.8x as long;
# fine-tuning is within 0.0025 of its best val AUROC after one epoch.
PRETRAIN_EPOCHS = 2
FINETUNE_EPOCHS = 3


def pretraining_contract() -> DataLoaderSpec:
    sample = SampleSpec(
        entity=EntitySpec("customers", entity_type="customer"),
        entity_values={
            "age": Field(column="age"),
            "club": Field(column="club_member_status"),
            "news": Field(column="fashion_news_frequency"),
            # A session's purchases share one timestamp, so distinct times count visits.
            "sessions_30d": Aggregate(purchases(-30 * DAY, 0), "count_distinct", "t_dat"),
            "sessions_365d": Aggregate(purchases(-365 * DAY, 0), "count_distinct", "t_dat"),
        },
        events={
            "transactions": EventSpec(
                purchases(None, 0),
                fields={
                    "price": Field(column="price"),
                    "channel": Field(column="sales_channel_id"),
                    "article": Field(column="article_id"),
                    "time": Field(column="t_dat"),
                    **{
                        part: Derived(
                            operation=CalendarPart(part=part), inputs=("time",), dtype=INT64
                        )
                        for part in CALENDAR
                    },
                    "age_seconds": EventTimeDelta(unit="seconds", dtype=DType("float64")),
                },
            )
        },
        # Fine-tuning reuses this sample and supervises this target; pre-training
        # emits no label.
        targets={"purchases_next_week": Aggregate(NEXT_WEEK, "count")},
    )

    def tokens(encodings: dict[str, Any]) -> Fields:
        # Every field encodes into the one "pragma" ID space, field names included.
        return Fields(
            tuple(
                Encode(name, encoding, id_space="pragma") for name, encoding in encodings.items()
            ),
            row=Tokens(max_tokens=len(encodings)),
            field_names="fields",
            field_space="pragma",
        )

    numerical, categorical = Buckets(bins=NUMERICAL_BUCKETS), Vocabulary()
    batch = BatchSpec(
        id_spaces={"pragma": IdSpace(specials={role: f"[{role.upper()}]" for role in ROLES})},
        field_names={"fields": FieldNames()},
        inputs={
            "profile": Input(
                EntitySource(),
                tokens(
                    {
                        "age": numerical,
                        "club": categorical,
                        "news": categorical,
                        "sessions_30d": numerical,
                        "sessions_365d": numerical,
                    }
                ),
                Padded(),
            ),
            "history": Input(
                EventSource(("transactions",), max_events=MAX_EVENTS, keep="last"),
                tokens({"price": numerical, "channel": categorical, "article": categorical}),
                Nested(fields=Padded(), events=Padded()),
            ),
            "history_calendar": Input(
                EventsOf("history"),
                Fields(tuple(Encode(part, Raw()) for part in CALENDAR)),
                Padded(),
            ),
            "history_timing": Input(
                EventsOf("history"), Fields((Encode("age_seconds", Raw()),)), Padded()
            ),
        },
        targets={
            "masked": MaskedValuesTarget(
                input="history",
                token_probability=0.15,
                event_probability=0.10,
                key_probability=0.0,
                unknown_probability=0.10,
            )
        },
        batching=Batching(max_samples=256),
    )
    return DataLoaderSpec(SCHEMA, sample, batch)


def finetuning_contract(pretrained: Artifact) -> DataLoaderSpec:
    """The pre-trained bound contract, with the label in place of masked values."""

    bound = pretrained.bound_contract
    return replace(
        bound,
        batch=replace(bound.batch, targets={"label": LabelTarget("purchases_next_week", Raw())}),
    )


class PragmaEncoder(nn.Module):
    """Fields → event vectors → a history over the profile and events.

    Returns the customer summary and, for every event field, its context, which
    the masked-value head reads during pre-training.
    """

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocabulary_size, WIDTH)
        self.event_summary = nn.Parameter(torch.randn(WIDTH) * 0.02)
        self.customer_summary = nn.Parameter(torch.randn(WIDTH) * 0.02)
        self.calendar = nn.Linear(2, WIDTH)
        self.elapsed = nn.Linear(1, WIDTH)
        self.fields = TransformerStack(1, WIDTH, 2 * WIDTH, 4, 0.1)
        self.history = TransformerStack(2, WIDTH, 2 * WIDTH, 4, 0.1)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        profile, events = batch.inputs["profile"], batch.inputs["history"]
        sequence = events["sequence"]
        size = profile["field_ids"].shape[0]

        fields = self.tokens(events["field_ids"]) + self.tokens(events["value_ids"])
        summary = self.event_summary.expand(fields.shape[0], 1, WIDTH)
        valid = torch.cat((torch.ones_like(events["valid"][:, :1]), events["valid"]), 1)
        encoded = self.fields(torch.cat((summary, fields), 1), valid)
        # Each event row's calendar parts, read at its (sample, position) in the history.
        parts = batch.inputs["history_calendar"]
        rows = (events["owner"], events["history_position"])
        calendar = torch.stack([parts[part]["values"][rows].float() for part in CALENDAR], -1)
        calendar = calendar / torch.tensor([7.0, 24.0])
        event_vectors = encoded[:, 0] + self.calendar(calendar)

        slots = event_vectors[sequence["event_indices"].clamp(min=0)]
        elapsed = batch.inputs["history_timing"]["age_seconds"]["values"].float()
        days = torch.log1p(elapsed.clamp(min=0) / DAY).unsqueeze(-1)
        slots = (slots + self.elapsed(days)) * sequence["valid"].unsqueeze(-1)
        profile_tokens = self.tokens(profile["field_ids"]) + self.tokens(profile["value_ids"])
        start = self.customer_summary.expand(size, 1, WIDTH)
        history = self.history(
            torch.cat((start, profile_tokens, slots), 1),
            torch.cat(
                (torch.ones_like(profile["valid"][:, :1]), profile["valid"], sequence["valid"]), 1
            ),
        )
        in_slots = history[:, 1 + profile_tokens.shape[1] :][sequence["valid"]]
        rows = sequence["event_indices"][sequence["valid"]]
        event_context = torch.zeros_like(event_vectors).index_copy(0, rows, in_slots)
        return history[:, 0], encoded[:, 1:] + event_context.unsqueeze(1)


class MaskedValueModel(nn.Module):
    """Pre-training: recover each masked field value, with the embeddings tied."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.encoder = PragmaEncoder(vocabulary_size)

    def loss(self, batch: Batch) -> torch.Tensor:
        _, context = self.encoder(batch)
        target = batch.targets["masked"]
        logits = context[target["valid"]] @ self.encoder.tokens.weight.T
        return F.cross_entropy(logits, target["values"][target["valid"]])


class PurchaseClassifier(nn.Module):
    """Fine-tuning: the encoder frozen, LoRA adapters on its Transformers, a new head."""

    def __init__(self, encoder: PragmaEncoder) -> None:
        super().__init__()
        add_transformer_lora(encoder, rank=LORA_RANK)
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))

    def forward(self, batch: Batch) -> torch.Tensor:
        customer, _ = self.encoder(batch)
        return self.head(customer).squeeze(-1)


def vocabulary_size(manifest: Mapping[str, Any]) -> int:
    """Every token and field name is numbered in the one "pragma" ID space."""

    return manifest["spaces"]["pragma"]["size"]


def pretrain(data: Path, work: Path, seed: int, epochs: int = PRETRAIN_EPOCHS) -> None:
    torch.manual_seed(seed)
    clock = Stopwatch()
    history = []
    engine = DiskGraphEngine(index=work / "index")
    with prepare(pretraining_contract(), dataset(data), engine) as plan:
        with plan.materialize(anchors(data), fit_split="train") as task:
            clock.lap("materialize")
            model = MaskedValueModel(vocabulary_size(task.manifest))
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
            for epoch in range(epochs):
                model.train()
                total, steps = 0.0, 0
                for batch in task.batches(
                    split="train", epoch=epoch, seed=seed, shuffle=True, output_format="model"
                ):
                    loss = model.loss(batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total, steps = total + loss.item(), steps + 1
                history.append({"epoch": epoch, "masked_value_loss": total / steps})
                print(f"pretrain: epoch {epoch} masked-value loss {total / steps:.4f}")
            clock.lap("fit")
            task.to_artifact().save(work / "pretrained")
            save_weights(work / "pretrained.pt", model.encoder.state_dict(), task.fitted_state)
    write_json(work / "pretrain.json", {"seed": seed, "history": history, "seconds": clock.laps})


def finetune(data: Path, work: Path, seed: int, epochs: int = FINETUNE_EPOCHS) -> None:
    torch.manual_seed(seed)
    clock = Stopwatch()
    pretrained = Artifact.load(work / "pretrained")
    engine = DiskGraphEngine(index=work / "index")
    with prepare(finetuning_contract(pretrained), dataset(data), engine) as plan:
        with plan.materialize(anchors(data), fit_split="train") as task:
            clock.lap("materialize")
            encoder = PragmaEncoder(vocabulary_size(task.manifest))
            encoder.load_state_dict(load_weights(work / "pretrained.pt", pretrained.fitted_state))
            model = PurchaseClassifier(encoder)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=3e-3)
            history, best = fit_classifier(model, optimizer, task, seed=seed, max_epochs=epochs)
            clock.lap("fit")
            task.to_artifact().save(work / "artifact")
            save_weights(work / "model.pt", model.state_dict(), task.fitted_state)
            fitted = task.fitted_state
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
            "trainable_parameters": sum(p.numel() for p in trainable),
            "test_digest": predictions.digest.hexdigest(),
            "seconds": clock.laps,
        },
    )
    print(f"finetune: best val AUROC {best:.4f}; artifact and weights written")


def load_model(work: Path) -> tuple[Artifact, PurchaseClassifier]:
    artifact = Artifact.load(work / "artifact")
    model = PurchaseClassifier(PragmaEncoder(vocabulary_size(artifact.manifest)))
    model.load_state_dict(load_weights(work / "model.pt", artifact.fitted_state))
    return artifact, model.eval()


STAGES = ("pretrain", "finetune", "infer", "serve")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="where infer writes; defaults to --work")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "pretrain":
        pretrain(args.data, args.work, args.seed, args.pretrain_epochs)
    elif args.stage == "finetune":
        finetune(args.data, args.work, args.seed, args.finetune_epochs)
    elif args.stage == "infer":
        batch_inference(*load_model(args.work), args.data, args.out or args.work)
    elif args.stage == "serve":
        serve(*load_model(args.work), args.data)
    else:
        common = ["--data", str(args.data), "--work", str(args.work), "--seed", str(args.seed)]
        common += ["--pretrain-epochs", str(args.pretrain_epochs)]
        common += ["--finetune-epochs", str(args.finetune_epochs)]
        for stage in STAGES:
            subprocess.run([sys.executable, "-m", __spec__.name, stage, *common], check=True)
        predictions = read_predictions(args.work / "predictions.parquet")
        area = oracle_auroc(args.data, predictions)
        print(f"test AUROC against the generator's labels: {area:.4f}")


if __name__ == "__main__":
    main()
