# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and display a package binding's full-population training result."""

import json
from math import isclose, isfinite

from e2e.datasets.binding_sources import PINS, _sources
from tests.e2e.binding_plan import NIGHTLY_TRAINING, binding_heads
from tests.e2e.binding_report import require_fields

# Display units are separate from the raw SI measurements retained in JSON.
METRICS = (
    ("ingestion_seconds", "Ingestion (s)", 1),
    ("materialization_seconds", "Materialize / fit (s)", 1),
    ("training_seconds", "Training (s)", 1),
    ("samples_per_second", "Samples/s", 1),
    ("tokens_per_second", "Tokens/s", 1),
    ("peak_ram_bytes", "Peak RAM (GiB)", 2**30),
    ("average_gpu_utilization_percent", "Mean GPU (%)", 1),
    ("ssd_bytes", "SSD (GiB)", 2**30),
)


def validate_training(data, binding, plan):
    if not data.get("complete"):
        raise ValueError(f"{data.get('stage', 'setup')}: {data.get('error', 'interrupted')}")
    require_fields(
        data,
        dict(
            binding=binding,
            revision=plan["revision"],
            run_id=plan["run_id"],
            run_attempt=plan["attempt"],
            tracked_diff_sha256=None,
            complete=True,
            stage="complete",
            scope="full_source",
            device="cuda",
            fit_split="train",
            workload="binding-train",
        ),
        binding,
    )
    pins = {}
    for name in _sources(binding)[0]:
        pins.update(json.loads((PINS / f"{name}.json").read_text())["objects"])
    for key in ("source_manifest", "anchor_manifest"):
        require_fields(data, {key + "_sha256": pins[key]["sha256"]}, binding)
    counts = data["split_counts"]
    expected = dict(zip(("train", "valid", "test"), NIGHTLY_TRAINING[binding], strict=True))
    if counts != expected:
        raise ValueError(f"{binding}: expected registered splits {expected}, received {counts}")
    for split, rows, epochs in (
        ("train", data["epochs"], [1, 2]),
        ("valid", data["validation"], [1, 2]),
        ("test", [data["test"]], [2]),
    ):
        if [r["epoch"] for r in rows] != epochs:
            raise ValueError(f"{binding}: incomplete {split} epochs")
        for r in rows:
            require_fields(r, dict(split=split, samples=counts[split]), binding)
            if not (
                r["elapsed_ns"] > 0
                and r["tokens"] >= 0
                and (
                    0 < r["optimizer_updates"] <= r["samples"]
                    if split == "train"
                    else r["optimizer_updates"] == 0
                )
            ):
                raise ValueError(f"{binding}: invalid {split} counters")
            if set(r["heads"]) != set(binding_heads(binding)):
                raise ValueError(f"{binding}: missing or unexpected loss head")
            for name, head in r["heads"].items():
                n, total, mean = (head[k] for k in ("eligible", "loss_sum", "mean_loss"))
                if (
                    type(n) is not int
                    or n < 0
                    or not isfinite(total)
                    or (not isclose(mean, total / n) if n else total != 0 or mean is not None)
                ):
                    raise ValueError(f"{binding}: invalid {split} {name} loss")
                eligible = data["expected_eligible"][name]
                if eligible is not None and n != eligible[split]:
                    raise ValueError(f"{binding}: {split} {name} eligibility differs from source")
    epochs = data["epochs"]
    seconds = sum(e["elapsed_ns"] for e in epochs) / 1e9
    gpu = [e["gpu"] for e in epochs]
    if (
        any(g["samples"] <= 0 or not 0 <= g["mean_percent"] <= 100 for g in gpu)
        or not gpu[0]["device_uuid"]
        or gpu[0]["device_uuid"] != gpu[1]["device_uuid"]
    ):
        raise ValueError(f"{binding}: invalid GPU utilization samples")
    metrics = data["metrics"]
    for key, _, _ in METRICS:
        if not isfinite(metrics[key]) or (
            metrics[key] < 0
            if key in {"tokens_per_second", "average_gpu_utilization_percent"}
            else metrics[key] <= 0
        ):
            raise ValueError(f"{binding}: invalid {key}")
    derived = dict(
        training_seconds=seconds,
        samples_per_second=2 * counts["train"] / seconds,
        tokens_per_second=sum(e["tokens"] for e in epochs) / seconds,
        average_gpu_utilization_percent=sum(g["mean_percent"] * g["samples"] for g in gpu)
        / sum(g["samples"] for g in gpu),
    )
    if not all(isclose(metrics[k], v) for k, v in derived.items()):
        raise ValueError(f"{binding}: aggregate metrics differ from the training epochs")


def training_tables(rows):
    if not rows:
        return []
    lines = [
        "",
        "### Full-split GPU training",
        "",
        "| Binding | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "| --- " * (1 + len(METRICS)) + "|",
    ]
    losses = [
        "",
        "| Binding | Epoch | Split | Anchors | Optimizer steps | Head | Eligible | Mean loss |",
        "| --- " * 8 + "|",
    ]
    for row in rows:
        result = row.get("training")
        if not result:
            lines.append(f"| {row['binding']} ({row['status']}) |" + " — |" * len(METRICS))
            continue
        numbers = [f"{result['metrics'][k] / scale:,.3f}" for k, _, scale in METRICS]
        lines.append("| " + " | ".join([row["binding"], *numbers]) + " |")
        for split in (*result["epochs"], *result["validation"], result["test"]):
            for head, loss in split["heads"].items():
                mean = "—" if loss["mean_loss"] is None else f"{loss['mean_loss']:.6g}"
                losses.append(
                    f"| {row['binding']} | {split['epoch']} | {split['split']} | "
                    f"{split['samples']:,} | {split['optimizer_updates']:,} | {head} | "
                    f"{loss['eligible']:,} | {mean} |"
                )
    return lines + losses
