# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timing hooks for the copied GPU preparation path; no model consumer."""

import inspect
from functools import wraps


def bind(recorder, *, split_dir, device):
    """Return prepare/observation callbacks for any Kumo contract and dataset."""
    import torch

    from ..reference_engine import gpu

    def synchronize():
        torch.cuda.synchronize(device)

    def timed(operation):
        def decorate(original):
            @wraps(original)
            def call(*args, **kwargs):
                synchronize()
                with recorder.span(operation) as details:
                    details["implementation"] = (
                        f"{original.__module__}.{original.__qualname__}"
                    )
                    details["synchronization"] = (
                        "device synchronized before and after operation"
                    )
                    try:
                        return original(*args, **kwargs)
                    finally:
                        synchronize()

            return call

        return decorate

    def encoder_fit(original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            from kumo.pytorch.api import Buckets, Text, Vocabulary, ZScore

            config = signature.bind(*args, **kwargs).arguments["config"]
            operation = (
                "vocabulary.fit"
                if isinstance(config, Vocabulary)
                else "statistics.fit"
                if isinstance(config, (Buckets, ZScore))
                else "bpe.fit"
                if isinstance(config, Text) and config.state is None
                else None
            )
            return (timed(operation)(original) if operation else original)(
                *args, **kwargs
            )

        return call

    def observe(observers):
        import diskgraph

        from kumo.pytorch import column_stats

        from ..reference_engine.gpu.batching import fit
        from ..reference_engine.gpu.encoding import (
            encoders,
            table_categories,
            table_stats,
            targets,
        )

        for owner, name, operation in (
            (table_stats, "pin_table_encoders", "input.fit"),
            (fit, "fit_inputs", "input.fit"),
            (targets, "fit_targets", "target.fit"),
            (table_categories, "pin_categories", "vocabulary.fit"),
            (diskgraph, "write_column_stats_for_parquet", "statistics.fit"),
        ):
            observers.patch(owner, name, timed(operation))
        observers.patch(encoders, "fit_encoder", encoder_fit)
        observers.patch(fit, "fit_encoder", encoder_fit)
        observers.patch(column_stats, "fit_from_stats", observers._encoder_fit)

    def prepare(contract, dataset):
        with recorder.span("reference.load") as details:
            details["scope"] = (
                "GPU context, contract checks, routing, data loading and indexes; synchronized completion"
            )
            torch.cuda.set_device(device)
            properties = torch.cuda.get_device_properties(device)
            recorder.write(
                "device",
                selected_device=str(device),
                name=properties.name,
                uuid=str(getattr(properties, "uuid", "unavailable")),
                total_memory_bytes=properties.total_memory,
                compute_capability=[properties.major, properties.minor],
                multiprocessors=properties.multi_processor_count,
                torch_cuda_version=torch.version.cuda,
            )
            plan = gpu.prepare(contract, dataset, split_dir=split_dir)
            try:
                synchronize()
            except BaseException:
                plan.close()
                raise
            return plan

    return prepare, observe
