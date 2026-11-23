# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and display the scheduled extract or manual full-population training result."""

import hashlib
import json
from datetime import timedelta
from math import isclose, isfinite

from e2e.datasets.binding_sources import PINS, _sources
from tests.e2e.binding_plan import (
    MEASURED_UPDATES,
    NIGHTLY_HISTORY_EVENTS,
    NIGHTLY_TRAINING,
    TRAINING_BATCH_SIZE,
    WARMUP_UPDATES,
    binding_heads,
)
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


def validate_training(data, binding, plan, *, scope="full_source"):
    if scope not in ("extract", "full_source"):
        raise ValueError(f"Unexpected GPU training scope {scope}")
    bounded = scope == "extract"
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
            scope=scope,
            device="cuda",
            fit_split="train",
            workload="binding-throughput" if bounded else "binding-train",
            history_limit=NIGHTLY_HISTORY_EVENTS,
        ),
        binding,
    )
    counts = data["split_counts"]
    pins = {}
    for name in _sources(binding)[0]:
        pins.update(json.loads((PINS / f"{name}.json").read_text())["objects"])
    for key in ("source_manifest", "anchor_manifest"):
        require_fields(data, {key + "_sha256": pins[key]["sha256"]}, binding)
    if bounded:
        pin = (PINS / "throughput" / f"{binding.lower()}.json").read_bytes()
        extract = json.loads(pin)
        expected = extract["split_counts"]
        require_fields(
            data,
            dict(
                extract_manifest_sha256=hashlib.sha256(pin).hexdigest(),
                batch_size=TRAINING_BATCH_SIZE,
                expected_eligible=extract["expected_eligible"],
            ),
            binding,
        )
        train = [data["warmup"], data["measurement"]]
        for row, phase, updates in zip(
            train, ("warmup", "measurement"), (WARMUP_UPDATES, MEASURED_UPDATES), strict=True
        ):
            require_fields(row, dict(phase=phase, optimizer_updates=updates), binding)
            if (
                row["samples"] % TRAINING_BATCH_SIZE
                or row["samples"] < updates * TRAINING_BATCH_SIZE
            ):
                raise ValueError(f"{binding}: invalid {phase} training batch sizes")
        if sum(r["samples"] for r in train) > counts["train"]:
            raise ValueError(f"{binding}: training windows exceed the source split")
        if not (
            isfinite(data["staging_seconds"])
            and data["staging_seconds"] >= 0
            and isfinite(data["extract_seconds"])
            and data["extract_seconds"] > 0
        ):
            raise ValueError(f"{binding}: invalid staging or extract duration")
    else:
        expected = dict(zip(("train", "valid", "test"), NIGHTLY_TRAINING[binding], strict=True))
        train = data["epochs"]
    if counts != expected:
        raise ValueError(f"{binding}: expected registered splits {expected}, received {counts}")
    for split, rows, epochs in (
        ("train", train, [1, 1] if bounded else [1, 2]),
        ("valid", data["validation"], [1] if bounded else [1, 2]),
        ("test", [data["test"]], [1] if bounded else [2]),
    ):
        if [r["epoch"] for r in rows] != epochs:
            raise ValueError(f"{binding}: incomplete {split} epochs")
        for r in rows:
            require_fields(r, {"split": split}, binding)
            if not (bounded and split == "train"):
                require_fields(r, {"samples": counts[split]}, binding)
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
            if bounded and sum(h["eligible"] for h in r["heads"].values()) < r["optimizer_updates"]:
                raise ValueError(f"{binding}: insufficient eligible targets in {split}")
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
                if (
                    eligible is not None
                    and not (bounded and split == "train")
                    and n != eligible[split]
                ):
                    raise ValueError(f"{binding}: {split} {name} eligibility differs from source")
    if bounded:
        for head in binding_heads(binding):
            eligible = data["expected_eligible"][head]
            if (
                eligible is not None
                and sum(r["heads"][head]["eligible"] for r in train) > eligible["train"]
            ):
                raise ValueError(f"{binding}: training {head} eligibility exceeds source")
    gpu = [e["gpu"] for e in train]
    if (
        any(g["samples"] <= 0 or not 0 <= g["mean_percent"] <= 100 for g in gpu)
        or not gpu[0]["device_uuid"]
        or any(g["device_uuid"] != gpu[0]["device_uuid"] for g in gpu)
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
    epochs = [data["measurement"]] if bounded else train
    gpu = [e["gpu"] for e in epochs]
    seconds = sum(e["elapsed_ns"] for e in epochs) / 1e9
    derived = dict(
        training_seconds=seconds,
        samples_per_second=sum(e["samples"] for e in epochs) / seconds,
        tokens_per_second=sum(e["tokens"] for e in epochs) / seconds,
        average_gpu_utilization_percent=sum(g["mean_percent"] * g["samples"] for g in gpu)
        / sum(g["samples"] for g in gpu),
    )
    if not all(isclose(metrics[k], v) for k, v in derived.items()):
        raise ValueError(f"{binding}: aggregate metrics differ from the measured training")


def training_tables(rows, wall_seconds):
    if not rows:
        return []
    bounded = rows[0].get("scope") == "extract"
    lines = [
        "",
        "### Bounded GPU training throughput" if bounded else "### Full-split GPU training",
        "",
        f"Each anchor uses up to {NIGHTLY_HISTORY_EVENTS} recent events and up to "
        f"{NIGHTLY_HISTORY_EVENTS} neighbors per relation. Counts use their full windows.",
    ]
    if wall_seconds is not None:
        lines += [
            "",
            f"First GPU job start to last finish took **{timedelta(seconds=wall_seconds)}**.",
        ]
    columns = [label for _, label, _ in METRICS]
    if bounded:
        columns = ["Stage extract (s)", "Verify / open (s)", "Build index (s)", *columns[1:]]
    lines += [
        "",
        "| Binding / model | " + " | ".join(columns) + " |",
        "| --- " * (1 + len(columns)) + "|",
    ]
    losses = [
        "",
        "| Binding | Epoch | Window | Anchors | Optimizer steps | Elapsed (s) "
        "| Head | Eligible | Mean loss |",
        "| --- " * 9 + "|",
    ]
    for row in rows:
        result = row.get("training")
        model = f"{row['binding']} · {row['representation']}"
        if not result:
            lines.append(f"| {model} ({row['status']}) |" + " — |" * len(columns))
            continue
        numbers = [f"{result['metrics'][k] / scale:,.3f}" for k, _, scale in METRICS]
        if bounded:
            numbers = [
                f"{result['staging_seconds']:,.3f}",
                f"{result['extract_seconds']:,.3f}",
                *numbers,
            ]
        lines.append("| " + " | ".join([model, *numbers]) + " |")
        train = [result["warmup"], result["measurement"]] if bounded else result["epochs"]
        for split in (*train, *result["validation"], result["test"]):
            for head, loss in split["heads"].items():
                mean = "—" if loss["mean_loss"] is None else f"{loss['mean_loss']:.6g}"
                losses.append(
                    f"| {row['binding']} | {split['epoch']} | {split.get('phase', split['split'])} "
                    f"| {split['samples']:,} | {split['optimizer_updates']:,} "
                    f"| {split['elapsed_ns'] / 1e9:.3f} | {head} | "
                    f"{loss['eligible']:,} | {mean} |"
                )
    return (
        lines
        + [
            "",
            (
                f"Rates cover {MEASURED_UPDATES} optimizer updates after "
                f"{WARMUP_UPDATES} warmup updates on the pinned extract. "
                if bounded
                else "Rates cover both training passes. "
            )
            + "Tokens count valid history inputs. "
            "M04 uses graph inputs. RAM is peak process RSS. SSD includes staged source and index.",
        ]
        + losses
    )
