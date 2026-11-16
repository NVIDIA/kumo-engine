"""Full PDF measurement inventory; absent collectors remain visible obligations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    unit: str
    scope: str = "run"


OPERATIONS = (
    "ingestion",
    "source.read",
    "source.convert",
    "index.build",
    "reference.load",
    "index.reopen",
    "materialization",
    "anchors.prepare",
    "input.fit",
    "target.fit",
    "vocabulary.fit",
    "statistics.fit",
    "bpe.fit",
    "task.setup",
    "state.load",
    "batch_production",
    "native.sample",
    "native.retrieve",
    "native.aggregate",
    "python.plan",
    "python.postprocess",
    "python.merge",
    "python.encode",
    "python.tokenize",
    "python.labels",
    "python.masks",
    "python.assemble",
    "reference.fit",
    "reference.plan",
    "reference.batch",
)
EPOCH_OPERATIONS = frozenset(
    name
    for name in OPERATIONS
    if name.startswith(("native.", "python."))
    or name in {"batch_production", "reference.plan", "reference.batch"}
)
METRICS = {
    f"{name}.elapsed_ns": Metric("ns", "epoch" if name in EPOCH_OPERATIONS else "run")
    for name in OPERATIONS
}
METRICS.update(
    {
        f"{stage}.peak_host_bytes": Metric(
            "bytes", "epoch" if stage == "batch_production" else "run"
        )
        for stage in ("ingestion", "materialization", "batch_production")
    }
)
METRICS.update(
    {
        "source.rows": Metric("rows"),
        "source.bytes": Metric("bytes"),
        "converted.rows": Metric("rows"),
        "converted.bytes": Metric("bytes"),
        "index.bytes": Metric("bytes"),
        "rows.examined": Metric("rows"),
        "rows.decoded": Metric("rows"),
    }
)
METRICS.update(
    {
        name: Metric(unit, "epoch")
        for name, unit in {
            "samples": "count",
            "padding_samples": "count",
            "real_samples": "count",
            "tokens": "count",
            "nodes": "count",
            "edges": "count",
            "events": "count",
            "output_elements": "count",
            "output_bytes": "bytes",
            "batches": "count",
            "samples_per_second": "samples/s",
            "tokens_per_second": "tokens/s",
            "nodes_per_second": "nodes/s",
            "edges_per_second": "edges/s",
            "execution.elapsed_ns": "ns",
            "iterator.startup.elapsed_ns": "ns",
            "batch.request.elapsed_ns": "ns",
            "first_batch_ns": "ns",
            "first_request_ns": "ns",
            "batch_latency_p50_ns": "ns",
            "batch_latency_p95_ns": "ns",
            "later_batch_latency_p50_ns": "ns",
            "later_batch_latency_p95_ns": "ns",
            "batch_wait_ns": "ns",
            "transfer.host.elapsed_ns": "ns",
            "transfer.device_ns": "ns",
            "training.step.elapsed_ns": "ns",
            "steps": "count",
            "optimizer_updates": "count",
            "steps_per_second": "steps/s",
            "gpu.peak_allocated_bytes": "bytes",
            "gpu.peak_reserved_bytes": "bytes",
            "gpu.utilization_percent": "percent",
        }.items()
    }
)

# Additional diagnostic spans do not replace any of the required measurements.
EXTRAS = {
    f"model.{name}.elapsed_ns": Metric("ns", "epoch")
    for name in (
        "adapt",
        "forward",
        "loss",
        "backward",
        "optimizer",
    )
}

CONDITIONS = (
    "steady_state",
    "first_and_later_epochs",
    "cold_and_warm_caches",
    "fresh_fit_and_state_reuse",
    "independent_repeated_runs",
)


def validate_inapplicable(declarations):
    for name, reason in declarations.items():
        if name not in METRICS:
            raise ValueError(f"Unknown metric in applicability declaration: {name}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Inapplicable metric {name} requires a reason")
        if name in {
            "samples",
            "real_samples",
            "samples_per_second",
            "batch_production.elapsed_ns",
        }:
            raise ValueError(f"Every workload requires {name}")
