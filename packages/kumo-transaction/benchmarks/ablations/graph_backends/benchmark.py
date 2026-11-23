# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Short controlled training-cost comparison; no full-pipeline or quality claim.

Run from the repository root with the kumo extra. All three arms consume the
same sampled per-root neighborhoods. Only Engine shared/disjoint have identical
initial model weights; legacy intentionally changes architecture. Graph unions
also change BatchNorm/node weighting and possibly context, not just runtime.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
from benchmarks.ablations.graph_backends.legacy import LegacyGraphEncoder
from torch import nn

from tfm.compositions.gnn import GNNEncoder
from tfm.contracts.graph import GraphAnchor, GraphBatch
from tfm.data.graph import GraphLink, GraphRow, GraphSampler, GraphTable, RelationalGraph
from tfm.extras.kumo import engine_version
from tfm.layers.gnn import GraphEncoder
from tfm.runtime.common import source_provenance


@dataclass(frozen=True)
class Settings:
    seed: int = 0
    entities: int = 64
    events_per_entity: int = 12
    transfers_per_entity: int = 4
    batch_size: int = 32
    width: int = 512
    depth: int = 6
    fanout: int = 10
    warmup: int = 2
    steps: int = 5
    device: str = "cuda"


class _Regressor(nn.Module):
    def __init__(self, encoder: nn.Module, width: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(width, 1)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        return self.head(self.encoder(batch)).flatten()


def inputs(settings: Settings) -> tuple[GraphSampler, tuple[GraphAnchor, ...], GNNEncoder]:
    """Overlapping generated relational data, independent of any task outcomes."""
    if (
        min(
            settings.entities,
            settings.events_per_entity,
            settings.transfers_per_entity,
            settings.batch_size,
            settings.width,
            settings.depth,
            settings.fanout,
            settings.steps,
        )
        < 1
        or settings.batch_size > settings.entities
        or settings.warmup < 0
    ):
        raise ValueError("Invalid benchmark sizes")
    generator = torch.Generator().manual_seed(settings.seed)
    now = datetime(2024, 1, 10, tzinfo=UTC)
    rows = [
        GraphRow("users", str(i), {"x": float(torch.rand((), generator=generator))})
        for i in range(settings.entities)
    ]
    roles = (
        ("users", "events", "events"),
        ("users", "sent", "transfers"),
        ("users", "received", "transfers"),
    )
    links = []
    for owner in range(settings.entities):
        for j in range(settings.events_per_entity):
            key, at = f"{owner}/{j}", now - timedelta(hours=j + 1)
            rows.append(
                GraphRow("events", key, {"x": float(torch.rand((), generator=generator))}, at, at)
            )
            links.append(GraphLink(roles[0], str(owner), key, at))
        for j in range(settings.transfers_per_entity):
            key, at = f"{owner}/{j}", now - timedelta(hours=j + 1)
            recipient = int(torch.randint(settings.entities, (), generator=generator))
            rows.append(
                GraphRow(
                    "transfers", key, {"x": float(torch.rand((), generator=generator))}, at, at
                )
            )
            links.extend(
                (
                    GraphLink(roles[1], str(owner), key, at),
                    GraphLink(roles[2], str(recipient), key, at),
                )
            )
    graph = RelationalGraph(
        (
            GraphTable("users", ("x",), immutable=True),
            GraphTable("events", ("x",)),
            GraphTable("transfers", ("x",)),
        ),
        roles,
        tuple(rows),
        tuple(links),
        frozenset({"label"}),
        now,
    )
    anchors = tuple(GraphAnchor(f"sample-{i}", str(i), now) for i in range(settings.batch_size))
    spec = GNNEncoder(
        tuple((t.name, len(t.features)) for t in graph.tables),
        graph.directed_relations,
        "users",
        tuple((t, f"rev_{r}", s) for s, r, t in graph.relations),
        hidden_size=settings.width,
        depth=settings.depth,
    )
    return GraphSampler(graph, depth=settings.depth, fanout=settings.fanout), anchors, spec


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    name: str, batch: GraphBatch, model: nn.Module, target: torch.Tensor, settings: Settings
) -> dict[str, Any]:
    device = torch.device(settings.device)
    _synchronize(device)
    start = time.perf_counter()
    model = model.to(device).train()
    batch, target = batch.to(device), target.to(device)
    _synchronize(device)
    transfer_ms = 1000 * (time.perf_counter() - start)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.002)
    latency, losses = [], []
    for step in range(settings.warmup + settings.steps):
        optimizer.zero_grad(set_to_none=True)
        if step == settings.warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        start = time.perf_counter()
        prediction = model(batch)
        loss = (prediction - target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite benchmark loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _synchronize(device)
        if step >= settings.warmup:
            latency.append(1000 * (time.perf_counter() - start))
            losses.append(float(loss.detach()))
    result = {
        "arm": name,
        "status": "ok",
        "sharing": batch.sharing,
        "nodes": sum(len(x) for x in batch.node_features.values()),
        "edges": sum(x.shape[1] for x in batch.edge_index.values()),
        "samples": batch.batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_and_batch_to_device_ms": transfer_ms,
        "step_ms_median": statistics.median(latency),
        "throughput": {
            "value": batch.batch_size * 1000 / statistics.median(latency),
            "unit": "roots/s",
            "statistic": "actual roots divided by median measured step seconds",
            "scope": "forward+MSE+backward+clipping+AdamW; batch/model transfer, construction and sampling excluded",
        },
        "step_ms": latency,
        "losses": losses,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20
        if device.type == "cuda"
        else None,
    }
    # Clear gradients/state before returning so one arm never inflates another's peak.
    optimizer.zero_grad(set_to_none=True)
    return result


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(settings: Settings, output: Path) -> dict[str, Any]:
    """Time training steps on fixed pre-sampled batches, recording all conditions."""
    started_at = datetime.now(UTC).isoformat()
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("The fixed benchmark requires the FP32 default dtype")
    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no CPU fallback")
    sampler, anchors, spec = inputs(settings)
    batches, preparation = {}, {}
    for sharing in ("shared", "disjoint"):
        start = time.perf_counter()
        batches[sharing] = sampler.sample(anchors, "users", sharing=sharing)
        preparation[sharing] = 1000 * (time.perf_counter() - start)
    # Exact identical initial Engine model state in the two batching arms.
    torch.manual_seed(settings.seed)
    template = _Regressor(GraphEncoder(spec), settings.width)
    state = copy.deepcopy(template.state_dict())
    engine_source = Path(inspect.getfile(type(template.encoder.gnn)))
    provenance = {
        "distribution": "kumo-engine",
        "version": engine_version(),
        "module": type(template.encoder.gnn).__module__,
        "gnn_source_sha256": hashlib.sha256(engine_source.read_bytes()).hexdigest(),
    }
    del template
    target = torch.randn(
        settings.batch_size, generator=torch.Generator().manual_seed(settings.seed + 1)
    )
    results = []
    for name, sharing in (
        ("engine_shared", "shared"),
        ("engine_disjoint", "disjoint"),
        ("legacy_disjoint", "disjoint"),
    ):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch.manual_seed(settings.seed)
        model = _Regressor(
            LegacyGraphEncoder(spec) if name == "legacy_disjoint" else GraphEncoder(spec),
            settings.width,
        )
        if name != "legacy_disjoint":
            model.load_state_dict(state)
        initial_hash = _state_hash(model)
        try:
            result = _measure(name, batches[sharing], model, target, settings)
        except torch.OutOfMemoryError:
            result = {"arm": name, "status": "out_of_memory", "sharing": sharing}
        result["initial_state_sha256"] = initial_hash
        results.append(result)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "precision": "fp32",
        "settings": asdict(settings),
        "engine": provenance,
        **source_provenance(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": version("torch-geometric"),
        "torch_scatter_installed": find_spec("torch_scatter") is not None,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "sampling_ms": preparation,
        "arms": results,
        "scope": "Generated fixed-batch model forward/backward/AdamW timing; sampling and transfer reported separately. Engine uses validated hop trimming and no extra input ReLU; legacy uses the same ordered input graph without trimming. No native loader, full-pipeline timing, or quality comparison. Sharing also changes BatchNorm weighting/context.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in Settings.__dataclass_fields__.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=str if name == "device" else int,
            default=field.default,
        )
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    print(json.dumps(run(Settings(**args), output), indent=2))


if __name__ == "__main__":
    main()
