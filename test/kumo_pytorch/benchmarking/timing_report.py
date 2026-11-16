"""Summarize loader-only campaigns without asserting cross-engine batch equality."""

import json
import statistics

from .metrics import TIMING_METRICS
from .reporting import _text, write_json


def timing_summary(cases, output):
    rows = []
    for case in cases:
        campaign_path = output / case["name"] / "results.json"
        if not campaign_path.is_file():
            continue
        campaign = json.loads(campaign_path.read_text())
        workers = []
        for run in campaign["runs"]:
            path = output / case["name"] / run["name"] / "results.json"
            if path.is_file():
                workers.append((run, json.loads(path.read_text())))
        for mode in ("fresh", "reused"):
            selected = [(run, result) for run, result in workers if run["mode"] == mode]
            if not selected:
                continue
            epochs = sorted(
                {e["scope"]["epoch"] for _, r in selected for e in r["epochs"]}
            )
            for epoch in (None, *epochs):
                metrics = {}
                for name, spec in TIMING_METRICS.items():
                    if (epoch is None) != (spec.scope == "run"):
                        continue
                    values = []
                    for run, result in selected:
                        # A failed worker's partial measurements remain in its own
                        # report, never in the successful-run timing summary.
                        if not run["execution_completed"]:
                            continue
                        observed = [
                            row
                            for row in result["measurements"].get(name, [])
                            if epoch is None or row["scope"].get("epoch") == epoch
                        ]
                        if observed and all(
                            row["status"] == "available" for row in observed
                        ):
                            values.append(sum(row["value"] for row in observed))
                    metrics[name] = {
                        "values": values,
                        "median": statistics.median(values) if values else None,
                        "min": min(values) if values else None,
                        "max": max(values) if values else None,
                        "unit": spec.unit,
                    }
                rows.append(
                    {
                        "case": case["name"],
                        "output_devices": sorted(
                            {
                                w["identity"].get(
                                    "output_device", w.get("device", "unspecified")
                                )
                                for _, r in selected
                                for w in r.get("workloads", {}).values()
                            }
                        ),
                        "status": campaign["status"],
                        "mode": mode,
                        "epoch": epoch,
                        "metrics": metrics,
                        "successful_runs": sum(
                            run["execution_completed"] for run, _ in selected
                        ),
                        "expected_runs": sum(
                            run["mode"] == mode for run in campaign["runs"]
                        ),
                    }
                )
    write_json(output / "timings.json", {"measurement_profile": "timing", "rows": rows})
    lines = [
        "## Loader timing comparison",
        "",
        (
            "Medians of successful workers only; each cell retains repetitions and ranges in timings.json. "
            "Failed workers retain partial measurements and errors in their linked reports. "
            "Nested stages overlap and must not be added. Caches are uncontrolled. "
            "First/later epochs and fresh/reused fitting stay separate. "
            "GPU timings synchronize device work; output locations are shown. "
            "This compares timings, not batch equality, training or model quality."
        ),
        "",
    ]
    tables = (
        (
            None,
            (
                ("Ingestion ms", "ingestion.elapsed_ns", 1e6),
                ("Source verification ms", "source.read.elapsed_ns", 1e6),
                ("Source conversion ms", "source.convert.elapsed_ns", 1e6),
                ("Index build ms", "index.build.elapsed_ns", 1e6),
                ("Reference load ms", "reference.load.elapsed_ns", 1e6),
                ("Index reopen ms", "index.reopen.elapsed_ns", 1e6),
                ("Materialization ms", "materialization.elapsed_ns", 1e6),
                ("State load ms", "state.load.elapsed_ns", 1e6),
            ),
        ),
        (
            "epoch",
            (
                ("Samples", "real_samples", 1),
                ("Loader ms", "batch_production.elapsed_ns", 1e6),
                ("First batch ms", "first_batch_ns", 1e6),
                ("Batch p50 ms", "batch_latency_p50_ns", 1e6),
                ("Batch p95 ms", "batch_latency_p95_ns", 1e6),
                ("Samples/s", "samples_per_second", 1),
            ),
        ),
    )
    for scope, columns in tables:
        headings = [
            "Case",
            "Status",
            "Output",
            "Fit",
            "Epoch",
            "Successful / expected runs",
        ]
        lines += [
            "| " + " | ".join(headings + [c[0] for c in columns]) + " |",
            "| "
            + " | ".join("---" for _ in range(len(headings) + len(columns)))
            + " |",
        ]
        for row in rows:
            if (scope is None) != (row["epoch"] is None):
                continue
            values = [
                f"[{row['case']}]({row['case']}/report.md)",
                row["status"],
                ", ".join(row["output_devices"]) or "unspecified",
                row["mode"],
                "preparation" if row["epoch"] is None else row["epoch"],
                f"{row['successful_runs']} / {row['expected_runs']}",
            ]
            for _, name, scale in columns:
                value = row["metrics"][name]["median"]
                values.append(
                    f"{value / scale:.3f}" if value is not None else "— (see coverage)"
                )
            lines.append("| " + " | ".join(_text(v) for v in values) + " |")
        lines.append("")
    return lines
