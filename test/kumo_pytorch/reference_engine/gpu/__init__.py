# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU backend for Kumo PyTorch: GPU sampling, encoding and batch delivery for Kumo contracts.

`prepare` takes the place of ``kumo.prepare(contract, dataset, engine)`` for a
plan that loads and batches on the GPU, using Kumo's shared table statistics for stored-column fitting. Upstream Kumo PyTorch has no backend hook,
so this package offers its own entry point in the same shape. It returns a
`GpuPlan`, whose ``materialize`` returns a `GpuTask`.

Importing this package loads no GPU library (cuDF, CuPy, RMM, cuGraph).
tests/environment/test_import_isolation.py checks that, so CPU-only arena code and CI can
import it. Each GPU library is imported only inside the code that uses it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BatchMemoryError", "GpuPlan", "GpuTask", "measure_model_bytes", "prepare"]


def __getattr__(name: str) -> Any:
    """`GpuPlan` and `GpuTask`, from `test.kumo_pytorch.reference_engine.gpu.plan`, and `BatchMemoryError`, when first asked for (they
    import no GPU library either)."""

    if name == "BatchMemoryError":
        from test.kumo_pytorch.reference_engine.gpu.batching.admission import BatchMemoryError

        return BatchMemoryError
    if name in ("GpuPlan", "GpuTask"):
        from test.kumo_pytorch.reference_engine.gpu.plan import gpu_plan, gpu_task

        return gpu_plan.GpuPlan if name == "GpuPlan" else gpu_task.GpuTask
    raise AttributeError(f"module 'test.kumo_pytorch.reference_engine.gpu' has no attribute {name!r}")


def prepare(contract: Any, dataset: Any, *, inference: bool = False, split_dir: Any = None,
            split: str = "if_missing", num_partitions: int | None = None, model_bytes: int | None = None) -> Any:
    """Prepare `contract` over `dataset` on the GPU: upstream ``prepare`` without its ``engine``.

    Returns a `test.kumo_pytorch.reference_engine.gpu.plan.gpu_plan.GpuPlan`, which offers ``materialize`` and ``close``
    as Kumo PyTorch's ``Plan`` does. A dataset too big for the GPU is split by entity into
    `split_dir` (or reuses the split there) and streamed: in `num_partitions`
    partitions when given, else as many as the room the model's measured peak
    (`model_bytes`, see `measure_model_bytes`) leaves needs. See `test.kumo_pytorch.reference_engine.gpu.plan.prepare.prepare`.
    """

    from test.kumo_pytorch.reference_engine.gpu.plan.prepare import prepare as _prepare

    return _prepare(contract, dataset, inference=inference, split_dir=split_dir, split=split,
                    num_partitions=num_partitions, model_bytes=model_bytes)


def measure_model_bytes(step: Any, *args: Any, **kwargs: Any) -> int:
    """The GPU memory a training step takes at its peak: `step(*args, **kwargs)` run once on the current GPU,
    PyTorch's peak reserved memory read after it (weights, gradients, optimizer state and the step's
    activations). Pass it to `prepare` as ``model_bytes``. Run it on a batch as big as training's."""

    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step(*args, **kwargs)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_reserved())
