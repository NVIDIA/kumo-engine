# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time repeated CPU-batch transfer plus the same fixed GNN training workload.

Run this identical driver with each checkout first on PYTHONPATH. Source identity
comes from the imported graph module, so baseline runs need no source changes.
Sampling/model transfer are outside cycle timing. No task-quality claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import platform
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from benchmarks.ablations.graph_backends.benchmark import Settings, _Regressor, _state_hash, inputs
from torch import nn

from tfm.contracts import graph as graph_contract
from tfm.extras.kumo import engine_version
from tfm.layers import gnn as graph_layers
from tfm.runtime.common import source_provenance


def run(settings: Settings, output: Path) -> dict[str, Any]:
    """Measure transfer-inclusive cycles, preserving the fixed-batch model recipe."""
    device = torch.device(settings.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This execution comparison requires CUDA; no fallback")
    started_at = datetime.now(UTC).isoformat()
    sampler, anchors, spec = inputs(settings)
    source = Path(inspect.getfile(graph_contract)).resolve()
    arms = []
    for sharing in ("shared", "disjoint"):
        cpu_batch = sampler.sample(anchors, "users", sharing=sharing)
        torch.manual_seed(settings.seed)
        model = _Regressor(graph_layers.GraphEncoder(spec), settings.width)
        initial_hash = _state_hash(model)
        model.to(device).train()
        target = torch.randn(
            settings.batch_size, generator=torch.Generator().manual_seed(settings.seed + 1)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.002)
        transfers, cycles, losses = [], [], []
        for step in range(settings.warmup + settings.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            if step == settings.warmup:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            batch = cpu_batch.to(device)
            torch.cuda.synchronize(device)
            transferred = time.perf_counter()
            prediction = model(batch)
            loss = (prediction - target).square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite benchmark loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize(device)
            end = time.perf_counter()
            if step >= settings.warmup:
                transfers.append(1000 * (transferred - start))
                cycles.append(1000 * (end - start))
                losses.append(float(loss.detach()))
        arms.append(
            {
                "sharing": sharing,
                "initial_state_sha256": initial_hash,
                "nodes": sum(len(x) for x in cpu_batch.node_features.values()),
                "edges": sum(e.shape[1] for e in cpu_batch.edge_index.values()),
                "batch_transfer_ms": transfers,
                "batch_transfer_ms_median": statistics.median(transfers),
                "transfer_and_train_ms": cycles,
                "transfer_and_train_ms_median": statistics.median(cycles),
                "throughput": {
                    "value": cpu_batch.batch_size * 1000 / statistics.median(cycles),
                    "unit": "roots/s",
                    "statistic": "actual roots divided by median measured cycle seconds",
                    "scope": "blocking batch transfer+forward+MSE+backward+clipping+AdamW; construction, sampling and model transfer excluded",
                },
                "losses": losses,
                "final_prediction": prediction.detach().cpu().tolist(),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            }
        )
        del optimizer, model, batch, cpu_batch, target, prediction, loss
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "settings": asdict(settings),
        "precision": "fp32",
        "arms": arms,
        **source_provenance(),
        "contract_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "kumo_engine": engine_version(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "scope": "Repeated blocking CPU-batch transfer, forward/backward, clipping and AdamW on fixed sampled graphs; sampling and model transfer excluded. Model/recipe identical across revisions. No native loader or convergence claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    result = run(Settings(warmup=args.warmup, steps=args.steps), args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
