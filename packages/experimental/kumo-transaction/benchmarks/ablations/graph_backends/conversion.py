# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument the fixed execution driver for constructor/compilation ablations.

Run the identical file against baseline, combined-edge, and accumulator sources.
Use --warmup 0 --steps 1 first to compare exact outputs/state/gradients before
collecting repeated timings. Snapshot hashing occurs outside measured cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from benchmarks.ablations.graph_backends import views
from torch import nn

from tfm.contracts.validation import ValidatedBatch
from tfm.runtime.common import source_provenance


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Record compilation within CPU reconstruction and hash final model state."""
    original_reconstruct = views._reconstruct
    original_seal = ValidatedBatch._seal
    original_hash = views._state_hash
    original_empty = torch.cuda.empty_cache
    compilation: list[float] = []
    snapshots: list[dict[str, str]] = []
    active: nn.Module | None = None
    in_constructor = False
    elapsed = 0.0

    def seal(batch: Any) -> None:
        nonlocal elapsed
        if not in_constructor:
            original_seal(batch)
            return
        start = time.perf_counter()
        original_seal(batch)
        elapsed += time.perf_counter() - start

    def reconstruct(batch: Any) -> Any:
        nonlocal in_constructor, elapsed
        elapsed = 0.0
        in_constructor = True
        try:
            return original_reconstruct(batch)
        finally:
            in_constructor = False
            compilation.append(1000 * elapsed)

    def remember(model: nn.Module) -> str:
        nonlocal active
        active = model
        return original_hash(model)

    def snapshot_and_release() -> None:
        nonlocal active
        if active is not None:
            digest = hashlib.sha256()
            for name, parameter in active.named_parameters():
                digest.update(name.encode())
                gradient = parameter.grad
                if gradient is None:
                    digest.update(b"None")
                else:
                    digest.update(str((gradient.dtype, tuple(gradient.shape))).encode())
                    digest.update(gradient.detach().cpu().contiguous().numpy().tobytes())
            snapshots.append(
                {
                    "final_state_sha256": original_hash(active),
                    "final_gradients_sha256": digest.hexdigest(),
                }
            )
            active = None
        original_empty()

    with (
        patch.object(views, "_reconstruct", reconstruct),
        patch.object(ValidatedBatch, "_seal", seal),
        patch.object(views, "_state_hash", remember),
        patch.object(torch.cuda, "empty_cache", snapshot_and_release),
    ):
        result = views.run(args)
    width = args.warmup + args.steps
    assert len(compilation) == width * len(result["results"])
    assert len(snapshots) == len(result["results"])
    for index, row in enumerate(result["results"]):
        measured = compilation[index * width + args.warmup : (index + 1) * width]
        row["samples"]["compilation_ms"] = measured
        row["median_ms"]["compilation_ms"] = statistics.median(measured)
        row.update(snapshots[index])
    result.update(source_provenance())
    result["execution_driver_sha256"] = result["driver_sha256"]
    result["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["instrumentation"] = (
        "Compilation measures _seal inside each timed CPU reconstruction. Thin timing wrappers "
        "are identical across revisions and included in constructor/cycle cost. Final state "
        "(including buffers) and gradient hashing occurs after each phase, outside timing. "
        "Compilation is a subset of constructor time, not an additional stage."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive; warmup must be nonnegative")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([row["median_ms"] for row in result["results"]]))


if __name__ == "__main__":
    main()
