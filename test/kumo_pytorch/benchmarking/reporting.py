"""Report every requirement; do not turn partial observations into acceptance."""

import json
import statistics
from collections import defaultdict

from .metrics import CONDITIONS, required_metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def report(directory, config, identities, errors):
    profile = config.get("measurement_profile", "full")
    metrics = required_metrics(profile)
    observations = defaultdict(list)
    epochs = []
    backend_profiles = []
    native_iterations = []
    gpu_observations = []
    fitting_states = []
    cache_conditions = []
    dataset_inputs = []
    steady_windows = []
    devices = []
    with (directory / "observations.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "measurement":
                observations[row["metric"]].append(row)
            elif row["kind"] == "epoch_result":
                epochs.append(row)
            elif row["kind"] == "native_iteration":
                native_iterations.append(row)
            elif row["kind"] == "backend_profile":
                backend_profiles.append(row)
            elif row["kind"].startswith("gpu_"):
                gpu_observations.append(row)
            elif row["kind"] == "fitting_state":
                fitting_states.append(row)
            elif row["kind"] == "cache_condition":
                cache_conditions.append(row)
            elif row["kind"] == "dataset_inputs":
                dataset_inputs.append(row)
            elif row["kind"] == "steady_window":
                steady_windows.append(row)
            elif row["kind"] == "device":
                devices.append(row)
    coverage = {}
    for name, spec in metrics.items():
        cases = []
        for repetition in range(config["repetitions"]):
            declaration = identities.get(repetition, {}).get("inapplicable", {})
            expected = range(config["epochs"]) if spec.scope == "epoch" else (None,)
            for epoch in expected:
                rows = [
                    row
                    for row in observations[name]
                    if row["scope"].get("repetition") == repetition
                    and (epoch is None or row["scope"].get("epoch") == epoch)
                ]
                reason = declaration.get(name)
                if reason and rows:
                    status, reason = (
                        "error",
                        "Measurement contradicts declared inapplicability",
                    )
                elif reason:
                    status = "not_applicable"
                elif not rows:
                    status, reason = "missing", "Required measurement was not collected"
                elif any(row["status"] != "available" for row in rows):
                    status = "incomplete"
                    reason = "; ".join(
                        sorted({row["reason"] for row in rows if row["reason"]})
                    )
                else:
                    status = "measured"
                cases.append(
                    {
                        "repetition": repetition,
                        "epoch": epoch,
                        "status": status,
                        "reason": reason,
                    }
                )
        good = {"measured", "not_applicable"}
        status = (
            "complete"
            if all(case["status"] in good for case in cases)
            else "incomplete"
        )
        coverage[name] = {"status": status, "unit": spec.unit, "cases": cases}

    # These experiments require separate execution/evidence. Epoch labels alone
    # cannot certify cold caches, independence, steady state or fitting reuse.
    conditions = {
        name: {"status": "pending", "reason": "Experiment evidence not yet implemented"}
        for name in CONDITIONS
    }
    if (
        config["epochs"] >= 2
        and len(epochs) == config["epochs"] * config["repetitions"]
        and all(row["complete"] for row in epochs)
    ):
        conditions["first_and_later_epochs"] = {
            "status": "measured",
            "reason": "Separate complete epoch windows retained",
        }
    measured = sum(item["status"] == "complete" for item in coverage.values())
    steady_gpu_complete = True
    for window in steady_windows:
        device = identities.get(window["scope"]["repetition"], {}).get("device", "cpu")
        if device == "cpu" or profile == "timing":
            continue
        samples = [
            sample
            for record in gpu_observations
            if record["kind"] == "gpu_utilization_samples"
            and record["scope"] == window["scope"]
            for sample in record["samples"]
            if window["start_ns"]
            <= sample["start_ns"]
            <= sample["end_ns"]
            <= window["end_ns"]
        ]
        good_samples = [sample for sample in samples if sample["status"] == "available"]
        measured_gpu = len(good_samples) >= 2 and len(good_samples) == len(samples)
        steady_gpu_complete &= measured_gpu
        window["gpu_sampling"] = {
            "status": "measured" if measured_gpu else "incomplete",
            "samples": samples,
            "sample_count": len(good_samples),
            "mean_utilization_percent": statistics.mean(
                s["value"] for s in good_samples
            )
            if measured_gpu
            else None,
            "leading_gap_ns": good_samples[0]["start_ns"] - window["start_ns"]
            if good_samples
            else None,
            "trailing_gap_ns": window["end_ns"] - good_samples[-1]["end_ns"]
            if good_samples
            else None,
            "scope": "device-wide samples; at least two successful in-window reads required, not a time integral",
        }
    if (
        len(steady_windows) == config["repetitions"] * config["epochs"]
        and steady_gpu_complete
    ):
        conditions["steady_state"] = {
            "status": "measured",
            "reason": "Explicit warmup and measured post-warmup windows with latency variation retained; no equilibrium assumption",
        }
    cache_pairs = []
    for repetition in range(config["repetitions"]):
        rows = [
            row
            for row in cache_conditions
            if row["scope"].get("repetition") == repetition
        ]
        completed_epochs = {
            row["scope"]["epoch"]
            for row in epochs
            if row["scope"].get("repetition") == repetition and row["complete"]
        }
        good = [
            row
            for row in rows
            if row["status"] == "verified"
            and row["scope"].get("epoch") in completed_epochs
        ]
        # Pair the same selected files; opening a new directory is not a cache experiment.
        signatures = [
            {
                tuple(
                    (f["device"], f["inode"], f["size"], f["mtime_ns"])
                    for f in row["files"]
                )
                for row in good
                if row["mode"] == mode
            }
            for mode in ("cold", "warm")
        ]
        cache_pairs.append(
            bool(signatures[0] & signatures[1]) and len(good) == len(rows)
        )
    if all(cache_pairs):
        conditions["cold_and_warm_caches"] = {
            "status": "measured",
            "reason": "Verified selected Linux file-page residency before separate epochs; application/device caches uncontrolled",
        }
    if profile == "timing":
        conditions["cold_and_warm_caches"] = {
            "status": "outside_scope",
            "reason": "Loader timing comparison does not control caches",
        }
    result = {
        "schema_version": 1,
        "measurement_profile": profile,
        "status": "failed" if errors else "incomplete",
        "execution_status": "failed" if errors else "completed",
        "metric_coverage": {"complete": measured, "total": len(metrics)},
        "experiments": conditions,
        "workloads": identities,
        "epochs": epochs,
        "errors": errors,
        "measurements": dict(observations),
        "backend_profiles": backend_profiles,
        "native_iterations": native_iterations,
        "gpu_observations": gpu_observations,
        "fitting_states": fitting_states,
        "cache_conditions": cache_conditions,
        "dataset_inputs": dataset_inputs,
        "steady_windows": steady_windows,
        "devices": devices,
    }
    write_json(
        directory / "coverage.json", {"metrics": coverage, "experiments": conditions}
    )
    write_json(directory / "results.json", result)
    lines = [
        "# Benchmark results",
        "",
        (
            f"Status: **{result['status']}**. "
            f"Execution: {result['execution_status']}. Profile: {profile}. Metric coverage: {measured}/{len(metrics)}."
        ),
        "",
        "This is measurement evidence, not a performance-regression pass.",
        "",
        "| Repetition | Epoch | Complete pass | Samples/s | Tokens/s | Nodes/s | Edges/s | p95 latency (ns) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in epochs:
        values = row["metrics"]
        cells = [row["scope"]["repetition"], row["scope"]["epoch"], row["complete"]]
        cells += [
            values.get(key, "unavailable")
            for key in (
                "samples_per_second",
                "tokens_per_second",
                "nodes_per_second",
                "edges_per_second",
                "batch_latency_p95_ns",
            )
        ]
        lines.append("| " + " | ".join(_text(v) for v in cells) + " |")
    if native_iterations:
        lines += [
            "",
            "## Native AnchorTable iteration",
            "",
            "Length counts verify planning and cache reuse; exclusive length-planning time is unavailable.",
            "",
            "| Repetition | Epoch | Native complete | Merged-history batches | Lengths computed | Cache hits | Same cache reused |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for epoch in native_iterations:
            for row in epoch["iterators"]:
                values = [
                    epoch["scope"]["repetition"],
                    epoch["epoch"],
                    epoch["complete"],
                    row["merged_history_batches"],
                    row.get("lengths_computed", "not applicable"),
                    row.get("length_cache_hits", "not applicable"),
                    row.get("length_cache_reused", "not applicable"),
                ]
                lines.append("| " + " | ".join(_text(v) for v in values) + " |")
    if devices:
        lines += [
            "",
            "## Execution devices",
            "",
            "```json",
            json.dumps(devices, indent=2),
            "```",
        ]
    memory_rows = [
        row
        for name, rows in observations.items()
        if name.endswith((".peak_host_bytes", ".peak_anonymous_bytes"))
        for row in rows
    ]
    if memory_rows:
        lines += [
            "",
            "## Host-memory measurement method",
            "",
            "Total host peaks include native allocations, descendants, file-cache and kernel charges. Anonymous peaks sample memory.stat anon separately, excluding file/shmem charges; neither is process RSS. Anonymous statistics can lag allocations, and the two peaks need not occur together.",
            "",
            "| Method | Precision | Interval (s) | Samples | Largest gap (ns) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for method in sorted(
            {row["details"].get("method", "unspecified") for row in memory_rows}
        ):
            rows = [
                row
                for row in memory_rows
                if row["details"].get("method", "unspecified") == method
            ]
            sampled = method == "sampled_cgroup_v2_memory_current"
            lines.append(
                "| "
                + " | ".join(
                    _text(v)
                    for v in (
                        method,
                        "Sampled maximum; between-sample peaks may be missed"
                        if sampled
                        else "Kernel high-water mark"
                        if method == "cgroup_v2_memory_peak_same_fd_reset"
                        else "unspecified",
                        sorted(
                            {
                                row["details"].get("sample_interval_seconds")
                                for row in rows
                            }
                        )
                        if sampled
                        else "not applicable",
                        sum(row["details"].get("sample_count", 0) for row in rows)
                        if sampled
                        else "not applicable",
                        max(row["details"].get("max_gap_ns", 0) for row in rows)
                        if sampled
                        else "not applicable",
                    )
                )
                + " |"
            )
        lines += [
            "",
            "Raw observations retain stage boundaries and individual samples. Boundary reads include observation overhead; short stages may have boundary reads only.",
        ]
    lines += [
        "",
        "## All required measurements",
        "",
        "Values retain their repetition/epoch and operation scope. Nested timings are not additive.",
        "",
        "| Metric | Unit | Coverage | Observations (repetition, epoch, value/status) |",
        "| --- | --- | --- | --- |",
    ]
    for name, item in coverage.items():
        grouped = defaultdict(list)
        for row in observations[name]:
            grouped[
                (
                    row["scope"].get("repetition"),
                    row["scope"].get("epoch"),
                    row["details"].get("outcome"),
                    row["status"],
                )
            ].append(row)
        values = []
        for key, rows in grouped.items():
            if key[-1] != "available":
                value = "; ".join(sorted({r["reason"] for r in rows}))
            elif len(rows) == 1:
                value = rows[0]["value"]
            else:
                numbers = [r["value"] for r in rows]
                if name.endswith((".peak_host_bytes", ".peak_anonymous_bytes")):
                    value = f"peak={max(numbers)}, measured_intervals={len(numbers)}"
                else:
                    value = f"n={len(rows)}, min={min(numbers)}, median={statistics.median(numbers)}, max={max(numbers)}"
            values.append(f"{key}: {value}")
        gaps = [
            f"({c['repetition']}, {c['epoch']}): {c['status']}: {c['reason']}"
            for c in item["cases"]
            if c["status"] != "measured"
        ]
        lines.append(
            f"| {name} | {item['unit']} | {item['status']} | "
            + _text("; ".join(values + gaps))
            + " |"
        )
    if backend_profiles:
        lines += [
            "",
            "## Native profile evidence",
            "",
            "Each native timing/count links to a profile observation ID. Full snapshots are in results.json and observations.jsonl.",
            "Parent/native durations overlap; graph host sampling time is not a Rust-only breakdown.",
            "",
            "| Method | Profile | Observations |",
            "| --- | --- | --- |",
        ]
        grouped_profiles = defaultdict(int)
        for row in backend_profiles:
            grouped_profiles[row["method"], row["profile_name"]] += 1
        for (method, profile), count in grouped_profiles.items():
            lines.append(f"| {_text(method)} | {_text(profile)} | {count} |")
    sampling = [
        row for row in gpu_observations if row["kind"] == "gpu_utilization_samples"
    ]
    if sampling:
        lines += [
            "",
            "## GPU sampling coverage",
            "",
            "Utilization is device-wide sampled diagnostics. Sample coverage does not certify steady state.",
            "",
            "| Repetition / epoch | GPU UUID | Samples / failed | Interval (s) | Leading / trailing gap (ns) | Sample span fraction |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in sampling:
            c = row["coverage"]
            cells = [
                f"{row['scope'].get('repetition')} / {row['scope'].get('epoch')}",
                c["uuid"],
                f"{c['sample_count']} / {c['failed_samples']}",
                c["interval_seconds"],
                f"{c['leading_gap_ns']} / {c['trailing_gap_ns']}",
                c["sample_span_fraction"],
            ]
            lines.append("| " + " | ".join(_text(value) for value in cells) + " |")
    if cache_conditions:
        lines += [
            "",
            "## File-page-cache conditions",
            "",
            "Residency snapshots cover only the explicitly selected Linux file pages.",
            "",
            "| Repetition / epoch | Mode | Status | Resident / total pages |",
            "| --- | --- | --- | --- |",
        ]
        for row in cache_conditions:
            scope = row["scope"]
            lines.append(
                f"| {scope.get('repetition')} / {scope.get('epoch')} | {row['mode']} | {row['status']} | {row.get('resident_pages', 'unavailable')} / {row.get('pages', 'unavailable')} |"
            )
    if fitting_states:
        lines += [
            "",
            "## Fitted state",
            "",
            "| Repetition | Mode | Fingerprint |",
            "| --- | --- | --- |",
        ]
        for row in fitting_states:
            lines.append(
                f"| {row['scope']['repetition']} | {row['mode']} | {row['fingerprint']} |"
            )
    if steady_windows:
        lines += [
            "",
            "## Post-warmup windows",
            "",
            "| Repetition / epoch | Warmup batches | Measured batches | Samples/s | p50 / p95 (ns) | Latency CV |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in steady_windows:
            lines.append(
                f"| {row['scope']['repetition']} / {row['scope']['epoch']} | {row['warmup_batches']} | {row['measured_batches']} | {row['samples_per_second']} | {row['p50_ns']} / {row['p95_ns']} | {row['latency_cv']} |"
            )
    lines += ["", "## Experiment coverage", ""]
    lines += [
        f"- {name}: {item['status']} — {item['reason']}"
        for name, item in conditions.items()
    ]
    lines += ["", "## Failures", ""] + [f"- {_text(error)}" for error in errors]
    lines += [
        "",
        "## Configuration and identity",
        "",
        "```json",
        json.dumps(
            {"configuration": config, "workloads": identities},
            indent=2,
            allow_nan=False,
        ),
        "```",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    return result
