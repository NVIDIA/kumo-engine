# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure native loading through CUDA optimization on one pinned workload."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import torch

from e2e.datasets.throughput import load_fixture, pin_path
from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors

BATCH_SIZE = 32
SEED = 17


def _window(batches, model, optimizer, device, count):
    samples = history_tokens = updates = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(count):
        batch = next(batches, None)
        if batch is None or batch.batch_size != BATCH_SIZE or batch.metadata["is_padding"].any():
            raise ValueError("Throughput requires a full training batch at every step")
        samples += batch.batch_size
        if "history" in batch.inputs:
            history_tokens += int(batch.inputs["history"]["valid"].sum())
        loss = model.batch_loss(batch, device)
        if loss is not None:
            torch._assert_async(loss.isfinite(), "Nonfinite benchmark loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    if not updates:
        raise ValueError("Training window had no eligible optimizer updates")
    return dict(
        batches=count,
        samples=samples,
        history_tokens=history_tokens,
        optimizer_updates=updates,
        seconds=seconds,
    )


def measure(workload, root, index, device):
    import kumo.settings

    from tests.e2e.throughput_models import make_model

    kumo.settings.configure(scratch_dir=index)
    start = perf_counter()
    contract, dataset, anchors, cutoff = load_fixture(workload, root)
    opened = perf_counter()
    with api.prepare(contract, dataset, api.DiskGraphEngine(num_threads=4)) as plan:
        indexed = perf_counter()
        with plan.materialize(anchors, fit_split="train", encoder_cutoff=cutoff) as task:
            fitted = perf_counter()
            if not isinstance(task._anchors, NativeAnchors):
                raise ValueError("Throughput requires native DiskGraph Parquet anchors")
            torch.manual_seed(SEED)
            model = make_model(workload, task).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
            fallback = RuntimeError("Throughput attempted Python batch delivery")
            with (
                patch.object(task._anchors, "_requests", side_effect=fallback),
                patch.object(task._assembler, "batches", side_effect=fallback),
            ):
                batches = task.batches(
                    split="train",
                    batch_size=BATCH_SIZE,
                    seed=SEED,
                    epoch=1,
                    shuffle=False,
                    output_format="model",
                )
                warmup = _window(batches, model, optimizer, device, 10)
                timed = _window(batches, model, optimizer, device, 100)
    return dict(
        fixture_open_seconds=opened - start,
        index_seconds=indexed - opened,
        fit_seconds=fitted - indexed,
        batch_size=BATCH_SIZE,
        seed=SEED,
        epoch=1,
        warmup=warmup,
        timed=timed,
        samples_per_second=timed["samples"] / timed["seconds"],
        history_tokens_per_second=None
        if workload == "M04"
        else timed["history_tokens"] / timed["seconds"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("M04", "S02", "L01"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--index", type=Path, required=True, help="Scratch root for a fresh managed index"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Throughput requires a CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    args.index.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.index) as work:
        report = measure(args.workload, args.data_root, Path(work) / "index", device)
    report.update(
        workload=args.workload,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        fixture_sha256=hashlib.sha256(pin_path(args.workload).read_bytes()).hexdigest(),
        runner=os.getenv("RUNNER_NAME"),
        gpu=torch.cuda.get_device_name(device),
        staging_seconds=float(os.environ.get("KUMO_PYTORCH_STAGE_SECONDS", 0)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    timed = report["timed"]
    tokens = report["history_tokens_per_second"]
    rate = "N/A (graph-only)" if tokens is None else f"{tokens:,.1f}"
    summary = (
        f"### {args.workload} DiskGraph throughput\n\n"
        "| Timed batches | Samples | Optimizer updates | Timed seconds "
        "| Samples/s | History tokens/s |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {timed['batches']} | {timed['samples']} | {timed['optimizer_updates']} | "
        f"{timed['seconds']:.3f} | {report['samples_per_second']:,.1f} | {rate} |\n\n"
        f"Stage {report['staging_seconds']:.2f} s · "
        f"Open fixture {report['fixture_open_seconds']:.2f} s · "
        f"Build index {report['index_seconds']:.2f} s · Fit {report['fit_seconds']:.2f} s\n"
    )
    print(summary)
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as out:
            out.write(summary)


if __name__ == "__main__":
    main()
