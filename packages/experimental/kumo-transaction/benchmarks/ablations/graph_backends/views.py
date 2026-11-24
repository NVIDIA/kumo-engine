# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare complete batch cycles across source revisions on fixed synthetic inputs.

Run the identical driver against each checkout using PYTHONPATH. Sampling is
excluded; CPU reconstruction/validation, transfer and model work are included.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import statistics
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from benchmarks.ablations.graph_backends.benchmark import _state_hash

from tfm.api.graph import graph_specs
from tfm.compositions.gnn import GNNSequence
from tfm.config import RecipeConfig
from tfm.contracts import graph as contracts
from tfm.data.graph_recipe import prepare_graph_data
from tfm.extras.kumo import engine_version
from tfm.layers.gnn import GNNEncoderModel, GNNSequenceModel
from tfm.runtime.common import source_provenance


def _hash_inputs(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(item.contiguous().numpy().tobytes())
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict) or hasattr(item, "items"):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (set, frozenset)):
            for child in sorted(item, key=repr):
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _reconstruct(batch: Any) -> Any:
    graph = replace(batch.graph)
    if isinstance(batch, contracts.GraphTrainingBatch):
        return replace(batch, graph=graph, targets=replace(batch.targets))
    return replace(batch, graph=graph, history=replace(batch.history))


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Measure three execution workloads without changing architecture or recipe."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required; no fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(args.deterministic)
    device = torch.device("cuda")
    root = Path(inspect.getfile(contracts)).resolve().parents[3]
    config_data = yaml.safe_load(
        (root / "recipes/gnn_sequence/config.pretrain_smoke.yaml").read_text()
    )
    config_data["data"]["n_entities"] = 64
    config_data["training"]["batch_size"] = 32
    config_data["sequence_training"]["batch_size"] = 32
    g = config_data["model"]["composition"]["granularity"]
    g.update(depth=4, fanout=10, hidden_size=32)
    g["decoder"].update(hidden_size=32, num_heads=4)
    config = RecipeConfig.model_validate(config_data)
    data = prepare_graph_data(config)
    graph, sequence = graph_specs(config, data)
    results = []
    for arm in ("gnn_training", "sequence_training", "validation"):
        torch.manual_seed(42)
        encoder = GNNEncoderModel(graph, data.pretrain[0].targets.columns)
        model = (
            encoder
            if arm == "gnn_training"
            else GNNSequenceModel(GNNSequence(graph, sequence), encoder.encoder)
        )
        original = (
            data.pretrain[0]
            if arm == "gnn_training"
            else (
                data.hybrid_train[0] if arm == "sequence_training" else data.probes["validation"][0]
            )
        )
        identity = _state_hash(model)
        model.to(device).train(arm != "validation")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        max_grad_norm = 1.0
        samples: dict[str, list[float]] = {
            k: [] for k in ("construction_ms", "transfer_ms", "model_ms", "cycle_ms")
        }
        losses = []
        for step in range(args.warmup + args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if step == args.warmup:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            batch = _reconstruct(original)
            constructed = time.perf_counter()
            batch = getattr(batch, "for_execution", batch.to)(device)
            torch.cuda.synchronize()
            moved = time.perf_counter()
            with torch.set_grad_enabled(arm != "validation"):
                output = model(batch, compute_loss=arm != "validation")
                if arm != "validation":
                    output.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
            torch.cuda.synchronize()
            end = time.perf_counter()
            if step >= args.warmup:
                for key, duration in zip(
                    samples,
                    (constructed - start, moved - constructed, end - moved, end - start),
                    strict=True,
                ):
                    samples[key].append(1000 * duration)
                if output.loss is not None:
                    losses.append(float(output.loss.detach()))
        results.append(
            {
                "arm": arm,
                "effective_settings": {
                    "root_count": original.graph.batch_size,
                    "model_mode": "eval" if arm == "validation" else "train",
                    "optimizer": None
                    if arm == "validation"
                    else {
                        "name": type(optimizer).__name__,
                        **{
                            key: optimizer.param_groups[0][key]
                            for key in ("lr", "weight_decay", "betas", "eps", "amsgrad")
                        },
                    },
                    "max_grad_norm": None if arm == "validation" else max_grad_norm,
                    "optimizer_steps": 0 if arm == "validation" else args.warmup + args.steps,
                },
                "input_sha256": _hash_inputs(original),
                "initial_state_sha256": identity,
                "samples": samples,
                "median_ms": {key: statistics.median(values) for key, values in samples.items()},
                "throughput": {
                    "value": original.graph.batch_size
                    * 1000
                    / statistics.median(samples["cycle_ms"]),
                    "unit": "roots/s",
                    "statistic": "actual roots divided by median cycle seconds",
                    "scope": "constructor+transfer+forward"
                    if arm == "validation"
                    else "constructor+transfer+forward+backward+clipping+AdamW",
                },
                "peak_memory": {
                    "allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                    "reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                    "scope": "PyTorch CUDA allocator peak after warmup across measured cycles; includes model, optimizer and live batch tensors",
                },
                "losses": losses,
                "final_embeddings": output.embeddings.detach().cpu().tolist(),
                "nodes": sum(len(x) for x in original.graph.node_features.values()),
                "edges": sum(x.shape[1] for x in original.graph.edge_index.values()),
            }
        )
        del batch, model, encoder, optimizer, output
        torch.cuda.empty_cache()
    return {
        **source_provenance(),
        "contract_sha256": hashlib.sha256(
            Path(inspect.getfile(contracts)).read_bytes()
        ).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "kumo_engine": engine_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "warmup": args.warmup,
        "steps": args.steps,
        "config": config.model_dump(mode="json"),
        "results": results,
        "scope": "Synthetic fixed sampled batches; CPU reconstruction/validation, blocking transfer, forward and (training only) backward/clipping/AdamW. Sampling and model transfer excluded. No quality or native-loader claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([r["median_ms"] for r in result["results"]]))


if __name__ == "__main__":
    main()
