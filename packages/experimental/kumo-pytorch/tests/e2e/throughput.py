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
from tests.e2e.throughput_resources import training_gpu_window, workload_resources

BATCH_SIZE = 32
SEED = 17


def _window(batches, count, model=None, optimizer=None, device=None):
    samples = history_tokens = updates = 0
    on_cuda = device is not None and device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(count):
        batch = next(batches, None)
        if batch is None or batch.batch_size != BATCH_SIZE or batch.metadata["is_padding"].any():
            raise ValueError("Throughput requires a full training batch at every step")
        samples += batch.batch_size
        if "history" in batch.inputs:
            history_tokens += int(batch.inputs["history"]["valid"].sum())
        if model is None:
            continue
        loss = model.batch_loss(batch, device)
        if loss is not None:
            torch._assert_async(loss.isfinite(), "Nonfinite benchmark loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    if on_cuda:
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    result = dict(batches=count, samples=samples, history_tokens=history_tokens, seconds=seconds)
    if model is not None:
        if not updates:
            raise ValueError("Training window had no eligible optimizer updates")
        result["optimizer_updates"] = updates
    return result


def _rates(window, workload):
    return dict(
        samples_per_second=window["samples"] / window["seconds"],
        history_tokens_per_second=None
        if workload == "M04"
        else window["history_tokens"] / window["seconds"],
    )


def measure(workload, root, index, device):
    from tests.e2e.throughput_models import make_model

    start = perf_counter()
    contract, dataset, anchors, cutoff = load_fixture(workload, root)
    opened = perf_counter()
    with api.prepare(contract, dataset, api.DiskGraphEngine(index=index, num_threads=4)) as plan:
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
                options = dict(
                    split="train",
                    batch_size=BATCH_SIZE,
                    seed=SEED,
                    epoch=1,
                    shuffle=False,
                    output_format="model",
                )
                batches = task.batches(**options)
                warmup = _window(batches, 10, model, optimizer, device)
                with training_gpu_window(device) as gpu:
                    timed = _window(batches, 100, model, optimizer, device)
                batches = task.batches(**options)
                loader_warmup = _window(batches, 10)
                loader_timed = _window(batches, 100)
                for replay, training in ((loader_warmup, warmup), (loader_timed, timed)):
                    if any(replay[k] != training[k] for k in ("samples", "history_tokens")):
                        raise ValueError("Loader replay sample or history token count changed")
    return dict(
        fixture_open_seconds=opened - start,
        index_seconds=indexed - opened,
        fit_seconds=fitted - indexed,
        batch_size=BATCH_SIZE,
        seed=SEED,
        epoch=1,
        warmup=warmup,
        timed=timed,
        training_gpu=gpu,
        loader_only=dict(
            warmup=loader_warmup, timed=loader_timed, **_rates(loader_timed, workload)
        ),
        **_rates(timed, workload),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("M04", "S02", "L01"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--index", type=Path, required=True, help="Scratch directory for a fresh index"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Throughput requires a CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    args.index.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.index) as work:
        index = Path(work) / "index"
        with workload_resources(args.workload, args.data_root, index) as resources:
            report = measure(args.workload, args.data_root, index, device)
    report.update(
        resources=resources,
        workload=args.workload,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        fixture_sha256=hashlib.sha256(pin_path(args.workload).read_bytes()).hexdigest(),
        runner=os.getenv("RUNNER_NAME"),
        gpu=torch.cuda.get_device_name(device),
        staging_seconds=float(os.environ.get("KUMO_PYTORCH_STAGE_SECONDS", 0)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = _summary(report)
    print(summary)
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as out:
            out.write(summary)


def _summary(report):
    gpu = report["training_gpu"]
    gpu_mean = f"{gpu['mean_percent']:.1f}%" if gpu["samples"] else "N/A (no samples)"
    resources = report["resources"]
    lines = [
        f"### {report['workload']} DiskGraph throughput\n",
        "| Window | Batches | Samples | Seconds | Samples/s | History tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, run in (("Training", report), ("Loader only (warm replay)", report["loader_only"])):
        timed = run["timed"]
        tokens = run["history_tokens_per_second"]
        rate = "N/A (graph-only)" if tokens is None else f"{tokens:,.1f}"
        lines.append(
            f"| {name} | {timed['batches']} | {timed['samples']} | {timed['seconds']:.3f} | "
            f"{run['samples_per_second']:,.1f} | {rate} |"
        )
    mib = 1024**2
    lines.extend(
        [
            "",
            f"Stage {report['staging_seconds']:.2f} s · "
            f"Open fixture {report['fixture_open_seconds']:.2f} s · "
            f"Build index {report['index_seconds']:.2f} s · Fit {report['fit_seconds']:.2f} s",
            f"\nTimed training: {report['timed']['optimizer_updates']} optimizer updates · "
            f"Mean GPU utilization {gpu_mean} ({gpu['samples']} device samples)",
            f"\nPeak trainer RSS {resources['peak_rss_bytes'] / mib:,.1f} MiB · "
            f"Peak sampled scratch (staged data + index) "
            f"{resources['peak_sampled_scratch_bytes'] / mib:,.1f} MiB · "
            f"Process storage I/O {resources['process_read_bytes'] / mib:,.1f} MiB read, "
            f"{resources['process_write_bytes'] / mib:,.1f} MiB written",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
