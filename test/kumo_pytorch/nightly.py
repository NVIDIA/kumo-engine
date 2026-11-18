# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog-driven CI shards and a combined report; no native imports for planning."""

import argparse
import json
import os
import re
from pathlib import Path

from .bindings import (
    binding_for,
    binding_label,
    contract_for,
    dataset_definition,
    load_bindings,
    prepare_binding,
    selected_bindings,
    selection,
)

ROOT = Path(__file__).resolve().parent
MODES = (
    "training",
    "index",
    "timing-diskgraph",
    "timing-polars",
    "timing-gpu",
    "length-planning",
)


def selected_cases(case=None, *, campaign=None):
    cases = [
        (selection(binding), binding["representation"])
        for binding in selected_bindings(campaign=campaign)
        if campaign is not None
        or "fitting" in binding["campaigns"]
        or any(
            ("benchmark-training" if mode == "training" else mode)
            in binding["campaigns"]
            for mode in MODES
        )
    ]
    names = [s["case_name"] for s, _ in cases]
    if (
        not names
        or len(names) != len(set(names))
        or any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) for name in names)
    ):
        raise ValueError("Nightly cases must have unique safe names")
    if case is None:
        return cases
    if case not in names:
        raise ValueError(f"Unknown nightly case: {case}")
    return [cases[names.index(case)]]


def matrix(*, campaign=None):
    return {
        "include": [
            {
                "case": s["case_name"],
                "dataset": dataset_definition(binding_for(s, f))["inventory"],
                "label": binding_label(binding_for(s, f)),
            }
            for s, f in selected_cases(campaign=campaign)
        ]
    }


def benchmark_matrix():
    return {
        "include": [
            {**entry, "mode": mode}
            for entry in matrix()["include"]
            for mode in MODES
            if ("benchmark-training" if mode == "training" else mode)
            in binding_for(*selected_cases(entry["case"])[0])["campaigns"]
        ]
    }


def campaign_resources(execution):
    runners = {
        "cpu": "ubuntu-24.04",
        "rtxpro6000": {
            "group": "nv-gpu-amd64-rtxpro6000-1gpu",
            "labels": "linux-amd64-gpu-rtxpro6000-latest-1",
        },
    }
    runner = execution.get("runner", "cpu")
    if runner not in runners:
        raise ValueError(f"Unknown campaign runner: {runner}")
    minimum = execution.get("min_free_gib", 0)
    if type(minimum) is not int or minimum < 0:
        raise ValueError("Campaign min_free_gib must be a nonnegative integer")
    return {
        **execution,
        "runner": json.dumps(runners[runner]),
        "min_free_gib": minimum,
    }


def campaign_matrix(campaign):
    modes = {
        "training": ("training",),
        "population": ("smoke", "full"),
        "lifecycle": ("lifecycle",),
        "histories": ("histories",),
        "benchmark": (
            "benchmark-training",
            "index",
            "timing-diskgraph",
            "timing-polars",
            "timing-gpu",
            "length-planning",
        ),
    }[campaign]
    return {
        "include": [
            {
                "binding": b["ref"],
                "mode": mode,
                "dataset": dataset_definition(b)["inventory"],
                "label": binding_label(b),
                **campaign_resources(b["execution"]),
            }
            for b in selected_bindings()
            for mode in modes
            if mode in b["campaigns"]
        ]
    }


def execution_plan(seed_start=0):
    """The same catalog selection for PR, scheduled and manual executions."""
    if seed_start < 0:
        raise ValueError("Conformance seed start must be nonnegative")
    population = campaign_matrix("population")["include"]
    return {
        "conformance": {
            "include": [
                {"shard": shard, "shards": 3, "seed_start": seed_start, "seeds": 12}
                for shard in range(3)
            ]
        },
        "correctness": {
            "include": [r for r in population if r["mode"] == "smoke"]
            + campaign_matrix("histories")["include"]
        },
        "training": campaign_matrix("training"),
        "lifecycle": campaign_matrix("lifecycle"),
        "population": {"include": [r for r in population if r["mode"] == "full"]},
        "benchmarks": benchmark_matrix(),
        "fitting": {
            "include": [
                {**row, "repeat": repeat}
                for row in matrix(campaign="fitting")["include"]
                for repeat in range(3)
            ]
        },
    }


def benchmark_config(case, mode, data_dir):
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    selection, family = selected_cases(case)[0]
    binding = binding_for(selection, family)
    if ("benchmark-training" if mode == "training" else mode) not in binding[
        "campaigns"
    ]:
        raise ValueError(f"Binding {binding['ref']} does not enable {mode}")
    config = json.loads((ROOT / "workloads/aml-benchmark.json").read_text())
    config.update(
        data_dir=str(data_dir),
        binding=binding_for(selection, family)["ref"],
        family=family,
        workload=selection["selector"]["workload_id"],
        cache_mode="cold-warm",
    )
    if mode.startswith("timing-"):
        backend = mode.removeprefix("timing-")
        config.update(
            backend=backend,
            training=False,
            cache_mode="uncontrolled",
            device="cuda:0" if backend == "gpu" else "cpu",
        )
    if mode == "length-planning":
        config.update(
            length_planning=True,
            training=False,
            device="cpu",
            cache_mode="uncontrolled",
        )
    return [
        {
            "name": case,
            "adapter": "test.kumo_pytorch.benchmarking.kumo:factory",
            "config": config,
        }
    ]


def combine_fitting(directory, case):
    """Require all three paired repetitions, including matching spec hashes."""
    from .benchmarking.fitting_memory import compare

    directory.mkdir(parents=True, exist_ok=True)
    runs, errors = [], []
    for repeat in range(3):
        try:
            report = json.loads(
                (directory / f"repeat-{repeat}/report.json").read_text()
            )
            assert report["status"] == "passed"
            assert report["selected_repeat"] == repeat and report["repetitions"] == 3
            assert len(report["cases"]) == 1 and report["cases"][0]["case"] == case
            selected = report["cases"][0]
            assert (
                selected["status"] == "passed" and selected["selected_repeat"] == repeat
            )
            assert len(selected["runs"]) == 2
            assert all(run["repeat"] == repeat for run in selected["runs"])
            runs.extend(selected["runs"])
        except Exception as error:  # noqa: BLE001 - absent repetitions cannot pass
            errors.append(f"repeat-{repeat}: {type(error).__name__}: {error}")
    comparison = compare(runs, counts=(25000, 400000), repetitions=3)
    status = "passed" if not errors and comparison["status"] == "passed" else "failed"
    result = {
        "status": status,
        "cases": [
            {"case": case, "status": status, "comparison": comparison, "runs": runs}
        ],
        "errors": errors,
    }
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# Fitting memory: {case}",
        "",
        f"Status: **{status}**",
        "",
        "Three independent runners; each builds one index and fits both population sizes against it.",
        "Specifications, source/anchor identities, threads and cutoff must match across repetitions.",
        "",
    ]
    lines.extend(f"- [Repetition {r}](repeat-{r}/report.md)" for r in range(3))
    lines += ["", "```json", json.dumps(comparison, indent=2), "```", *errors]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return result


def validate_binding_report(report, binding, mode, revision):
    from .refusals import record_child_refusal

    assert report["status"] == "passed", report.get("error", report["status"])
    assert report["revision"] == revision, "Report belongs to another revision"
    assert not report["dirty_checkout"], "CI report used a modified checkout"
    assert report["mode"] == mode and report["ref"] == binding["ref"]
    assert report["binding_id"] == binding["binding_id"]
    assert report["selector"] == binding["selector"]
    assert report["family"] == binding["representation"]
    if report.get("outcome") == "refused":
        assert report["stage"] == "prepare" and not report["full_population_verified"]
        assert all(
            row.get("delivered", 0) == 0 for row in report.get("splits", {}).values()
        ), "Refusal after delivery"
        record_child_refusal(report["refusal"])
        return "refused"
    assert report["stage"] == "complete"
    assert report.get("outcome", "executed") == "executed"
    import math

    from .bindings import task_definition
    from .refusals import record_child_refusal

    objective = task_definition(binding)
    targets = set(binding["targets"]) or {objective["kind"]}
    assert set(report["splits"]) == {"train", "valid", "test"}
    for split in report["splits"].values():
        assert split["status"] == "passed"
        assert split["delivered"] == split["expected"] and split["delivered"] > 0
        assert set(split["eligible_targets"]) == targets
        assert all(
            n > 0 and (objective["kind"] != "stored" or n <= split["delivered"])
            for n in split["eligible_targets"].values()
        )
    assert report["full_population_verified"] == (mode == "full")
    requirements = binding.get("feature_checks", {})
    if mode in requirements.get("modes", ()):
        from .feature_checks import validate_evidence

        validate_evidence(report, requirements, mode)
        coverage = report["feature_coverage"]
        assert set(coverage) == set(requirements["features"]), (
            "Missing feature coverage"
        )
        expected = {s: row["delivered"] for s, row in report["splits"].items()}
        assert all(counts == expected for counts in coverage.values()), (
            "Features were not checked for every selected anchor and split"
        )
    if mode == "full":
        assert report["population"]["mode"] == "full"
        assert report["mapping"]["scope"]["kind"] == "full_source"
    elif mode == "training":
        assert report["reproducible"] and len(report["repeats"]) == 2
        for repeat in report["repeats"]:
            assert repeat["status"] == "passed" and repeat["completed_updates"] == 100
            assert set(repeat["checks"]) == {
                "100_updates",
                "every_parameter_has_nonzero_gradient",
                "every_parameter_updated",
                "loss_decreased",
            }
            assert all(repeat["checks"].values())
        assert (
            report["evaluation_model_sha256"]
            == report["repeats"][0]["final_model_sha256"]
        )
        assert set(report["evaluation"]) == {"valid", "test"}
        for split, outputs in report["evaluation"].items():
            assert set(outputs) == targets
            for name, output in outputs.items():
                assert (
                    output["eligible"]
                    == report["splits"][split]["eligible_targets"][name]
                )
                assert math.isfinite(output["mean_loss"])
    if mode == "lifecycle":
        checks = report["lifecycle"]["checks"]
        assert report["lifecycle"]["status"] == "passed"
        assert set(checks) == {"inference", "finetuning", "mid_epoch_resume"}
        assert all(row["status"] == "passed" for row in checks.values())
        assert (
            checks["inference"]["no_refitting"]
            and checks["inference"]["targets_absent"]
        )
        assert checks["finetuning"]["input_refitting"] is False
        assert checks["finetuning"]["resume"]["status"] == "passed"
        assert checks["mid_epoch_resume"]["remaining"]
        for check in checks.values():
            assert check["pid"] != check["parent_pid"]
    if mode == "histories":
        checks = report["histories"]
        assert checks["status"] == "passed"
        assert [row["name"] for row in checks["cases"]] == [
            "windows",
            "filters-and-windows",
        ]
        for row in checks["cases"]:
            assert row["status"] == "passed"
            if row.get("outcome") == "refused":
                assert row["stage"] == "prepare"
                record_child_refusal(row["refusal"])
                continue
            assert row["samples"] > 0
            assert set(row["events_by_input"]) == {
                f"{keep}_{order}"
                for keep in ("all", "first", "last")
                for order in ("ascending", "descending")
            }
            assert row["events_by_input"]["all_ascending"] > 0
    return "executed"


def summarize_bindings(directory, revision):
    bindings = load_bindings()
    rows = []
    for binding in selected_bindings():
        ref = binding["ref"]
        for mode in (
            m
            for m in ("training", "smoke", "full", "lifecycle", "histories")
            if m in binding["campaigns"]
        ):
            path = directory / mode / ref / "report.json"
            row = {"ref": ref, "mode": mode, "report": str(path.relative_to(directory))}
            try:
                report = json.loads(path.read_text())
                row.update(
                    outcome=validate_binding_report(
                        report, bindings[ref], mode, revision
                    ),
                    status="passed",
                )
                if report.get("refusal"):
                    row["refusal"] = report["refusal"]
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
                row.update(status="failed", error=str(error) or type(error).__name__)
            rows.append(row)
    return {
        "status": "passed" if all(r["status"] == "passed" for r in rows) else "failed",
        "revision": revision,
        "cases": rows,
    }


def summarize(directory, revision):
    """Missing or unsuccessful shards fail; keep every worker's linked reports."""
    from .benchmarking.timing_report import timing_summary

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, timing_cases = [], []
    for entry in matrix()["include"]:
        case = entry["case"]
        binding = binding_for(*selected_cases(case)[0])
        kinds = [
            row["mode"] for row in benchmark_matrix()["include"] if row["case"] == case
        ]
        if "fitting" in binding["campaigns"]:
            combine_fitting(directory / "fitting" / case, case)
            kinds.append("fitting")
        for kind in kinds:
            base = directory / kind / case
            path = base / ("results.json" if kind in MODES else "report.json")
            row = {
                "case": case,
                "ref": binding["ref"],
                "category": "benchmark",
                "kind": kind,
                "status": "missing",
            }
            try:
                result = json.loads(path.read_text())
                row["status"] = result["status"]
                if any(
                    c["status"] not in {"passed", "complete"} for c in result["cases"]
                ):
                    raise ValueError("Shard contains unsuccessful cases")
                if kind in MODES:
                    if [c["name"] for c in result["cases"]] != [case]:
                        raise ValueError("Shard reported a different case")
                else:
                    identities = [
                        c.get("case") or f"{c['selector']['workload_id']}-{c['family']}"
                        for c in result["cases"]
                    ]
                    if identities != [case]:
                        raise ValueError("Shard reported a different case")
                row["report"] = str((base / "report.md").relative_to(directory))
                if kind.startswith("timing-"):
                    timing_cases.append({"name": f"{kind}/{case}/{case}"})
            except Exception as error:  # noqa: BLE001 - missing evidence must stay visible
                row.update(status="failed", error=str(error))
            rows.append(row)
    for entry in summarize_bindings(directory, revision)["cases"]:
        rows.append(
            {
                **entry,
                "case": entry["ref"],
                "kind": entry["mode"],
                "category": "binding",
            }
        )
    status = (
        "passed"
        if all(r["status"] in {"passed", "complete"} for r in rows)
        else "failed"
    )
    result = {"status": status, "shards": rows}
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# PyTorch binding results",
        "",
        f"Status: **{status}**",
        "",
        "| Dataset / Binding | Correctness | Training | Lifecycle | Population | Benchmarks |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    groups = {
        "Correctness": ("smoke", "histories"),
        "Training": ("training",),
        "Lifecycle": ("lifecycle",),
        "Population": ("full",),
        "Benchmarks": (*MODES, "fitting"),
    }
    # Model-training benchmarks and acceptance both use "training" in reports.
    for binding in selected_bindings():
        cells = []
        for group, kinds in groups.items():
            selected = [
                r
                for r in rows
                if r["ref"] == binding["ref"]
                and r["kind"] in kinds
                and (
                    r["category"] == "benchmark"
                    if group == "Benchmarks"
                    else r["category"] == "binding"
                )
            ]
            if not selected:
                cells.append(
                    "Legacy campaign"
                    if group == "Lifecycle"
                    and "legacy-lifecycle" in binding["campaigns"]
                    else "Not enabled"
                )
                continue
            cells.append(
                "<br>".join(
                    f"[{r['kind']}: {r.get('outcome') if r.get('outcome') == 'refused' else r['status']}]({r['report']})"
                    if "report" in r
                    else f"{r['kind']}: {r['status']}"
                    for r in selected
                )
            )
        lines.append(f"| {binding_label(binding)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Binding | Experiment | Status | Report |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        link = f"[report]({row['report']})" if "report" in row else "Missing report"
        label = row["status"] + (
            " (refused)" if row.get("outcome") == "refused" else ""
        )
        lines.append(f"| {row['ref']} | {row['kind']} | {label} | {link} |")
    lines += ["", *timing_summary(timing_cases, directory)]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return result


def execute(
    task,
    binding,
    projection,
    population,
    directory,
    report,
    *,
    mode,
    capacity,
    batch_size,
    seed,
    plan,
    dataset,
    anchors,
    threads,
):
    from .population import check_population, write_report
    from .stage_timing import StageTiming
    from .training_acceptance import train_task

    timing = StageTiming(report, lambda: write_report(directory, report))
    features = None
    requirements = binding.get("feature_checks", {})
    if mode in requirements.get("modes", ()):
        from .bindings import resolve

        with timing.span("features.prepare", checkpoint=True):
            features = resolve(requirements["factory"])(
                binding,
                projection,
                report["mapping"],
                task.fitted_state.to_dict(),
                mode=mode,
            )
        report["feature_coverage"] = features.coverage
        report["feature_assertions"] = features.evidence
    check_population(
        task,
        binding,
        projection,
        population,
        directory,
        report,
        batch_size=batch_size,
        seed=seed,
        features=features,
    )
    report["full_population_verified"] = mode == "full"
    if mode == "training":
        with timing.span("training", checkpoint=True):
            train_task(
                task,
                binding,
                projection,
                population,
                directory,
                report,
                capacity=capacity,
                batch_size=batch_size,
                seed=seed,
                features=features,
            )
    if mode == "lifecycle":
        from .lifecycle import run_binding as lifecycle

        with timing.span("lifecycle", checkpoint=True):
            report["lifecycle"] = lifecycle(
                plan,
                task,
                dataset,
                anchors,
                binding,
                directory,
                capacity=capacity,
                batch_size=batch_size,
                threads=threads,
                seed=seed,
            )
        assert report["lifecycle"]["status"] == "passed", (
            "Lifecycle checks failed; see individual results"
        )
    if mode == "histories":
        from .bindings import resolve

        with timing.span("histories", checkpoint=True):
            report["histories"] = resolve(binding["history_checks"]["factory"])(
                dataset,
                binding,
                projection,
                report["mapping"],
                directory / "histories",
                threads=threads,
            )
        assert report["histories"]["status"] == "passed", (
            "Source history probes failed; see individual results"
        )
    return report


def run_case(
    dataset,
    anchors,
    population,
    binding,
    directory,
    *,
    mode,
    capacity,
    mapping,
    batch_size=4,
    threads=1,
    seed=0,
    report=None,
):
    import kumo.pytorch.api as r

    from .data import schema_for
    from .population import write_report
    from .refusals import case_failure, executing_case
    from .stage_timing import StageTiming

    report = {
        **(report or {}),
        "status": "running",
        "stage": "contract",
        "ref": binding["ref"],
        "binding_id": binding["binding_id"],
        "selector": binding["selector"],
        "family": binding["representation"],
        "mode": mode,
        "full_population_verified": False,
        "population": population,
        "mapping": mapping,
    }
    write_report(directory, report)
    timing = StageTiming(report, lambda: write_report(directory, report))
    try:
        if mode not in {"training", "smoke", "full", "lifecycle", "histories"}:
            raise ValueError(f"Unknown binding execution mode: {mode}")
        if mode == "full" and population["mode"] != "full":
            raise ValueError("Bounded anchors cannot qualify full-population delivery")
        if mode == "full" and mapping["scope"]["kind"] != "full_source":
            raise ValueError("Bounded sources cannot qualify full-population delivery")
        contract = contract_for(
            schema_for(dataset, binding), binding, max_events=capacity
        )
        (directory / "contract.json").write_text(contract.to_json())
        report["stage"] = "prepare"
        with timing.span("ingestion", checkpoint=True):
            plan = r.prepare(
                contract,
                dataset,
                r.DiskGraphEngine(index=directory / "index", num_threads=threads),
            )
        with plan, executing_case():
            report["stage"] = "materialize"
            with timing.span("materialization", checkpoint=True):
                task = plan.materialize(
                    anchors,
                    fit_split="train",
                    encoder_cutoff=population["encoder_cutoff_seconds"],
                )
            report["stage"] = "execute"
            execute(
                task,
                binding,
                directory / "anchors.parquet",
                population,
                directory,
                report,
                mode=mode,
                capacity=capacity,
                batch_size=batch_size,
                seed=seed,
                plan=plan,
                dataset=dataset,
                anchors=anchors,
                threads=threads,
            )
        report.update(status="passed", stage="complete")
    except Exception as error:  # noqa: BLE001 - persist execution failures in the case report
        report.update(case_failure(error, allow_refusal=report["stage"] == "prepare"))
    write_report(directory, report)
    return report


def run_binding(args):
    import importlib.metadata
    import subprocess
    import time

    import torch

    from .data import verify_dataset
    from .population import write_report
    from .refusals import case_failure
    from .stage_timing import StageTiming

    binding = load_bindings()[args.binding]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "status": "running",
        "stage": "verify_inputs",
        "ref": args.binding,
        "mode": args.mode,
        "binding_id": binding["binding_id"],
        "selector": binding["selector"],
        "full_population_verified": False,
    }
    timing = StageTiming(
        report, lambda: write_report(args.output, report), checkpoint=True
    )
    try:
        torch.set_num_threads(args.threads)
        report["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        report["dirty_checkout"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        report["versions"] = {
            p: importlib.metadata.version(p) for p in ("torch", "pyarrow", "diskgraph")
        }
        write_report(args.output, report)
        with timing.span("source.verify", checkpoint=True):
            files = verify_dataset(args.data_dir, binding)
        report["inputs"] = list(files.values())
        report["stage"] = "prepare_sources"
        write_report(args.output, report)
        dataset, anchors, population, mapping, capacity = prepare_binding(
            args.data_dir,
            args.output,
            binding,
            files,
            limit=None
            if args.mode == "full"
            else args.anchors_per_split or (32 if args.mode == "training" else 4),
            recorder=timing,
        )
        report["mapping"] = mapping
        report["history_capacity"] = capacity
        report.update(
            run_case(
                dataset,
                anchors,
                population,
                binding,
                args.output,
                mode=args.mode,
                capacity=capacity,
                mapping=mapping,
                batch_size=args.batch_size,
                threads=args.threads,
                seed=args.seed,
                report=report,
            )
        )
    except Exception as error:  # noqa: BLE001 - input/preparation failures must retain evidence
        report.update(case_failure(error, allow_refusal=False))
    report["seconds"] = time.perf_counter() - started
    write_report(args.output, report)
    lines = [
        f"# {args.binding}: {binding['selector']['workload_id']} / {binding['representation']}",
        "",
        f"Mode: **{args.mode}**; status: **{report['status']} ({report.get('outcome', 'executed')})**",
        "",
        f"Full population verified: **{report['full_population_verified']}**",
        "",
        "| Split | Delivered / expected | Eligible targets |",
        "| --- | --- | --- |",
    ]
    for name, split in report.get("splits", {}).items():
        lines.append(
            f"| {name} | {split['delivered']} / {split['expected']} | {split['eligible_targets']} |"
        )
    for i, repeat in enumerate(report.get("repeats", [])):
        lines.append(
            f"\nRepetition {i}: {repeat['status']}; updates {repeat['completed_updates']}; probe loss {repeat.get('initial_loss')} -> {repeat.get('final_loss')}. [Details](repeat-{i}/report.json)."
        )
    if report.get("feature_coverage"):
        lines += ["", "| Feature | Checked anchors by split |", "| --- | --- |"]
        lines.extend(
            f"| {name} | {counts} |"
            for name, counts in report["feature_coverage"].items()
        )
    if report.get("feature_assertions"):
        lines += [
            "",
            "| Assertion | Features | Status | Checked anchors by phase/split |",
            "| --- | --- | --- | --- |",
        ]
        for name, row in report["feature_assertions"].items():
            lines.append(
                f"| {name} | {', '.join(row['features'])} | {row['status']} | {row['checked_anchors']} |"
            )
            if row.get("error"):
                lines += ["", f"{name}: {row['error']}"]
    for kind in ("lifecycle", "histories"):
        if kind in report:
            lines += [
                "",
                f"### {kind.title()}",
                "",
                "```json",
                json.dumps(report[kind], indent=2),
                "```",
            ]
    if report.get("error"):
        lines += ["", "```text", report["error"], "```"]
    lines += [
        "",
        "| Stage | Seconds | Calls | Failed calls |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in report.get("stage_timings", {}).items():
        lines.append(
            f"| {name} | {row['seconds']:.3f} | {row['calls']} | {row['failures']} |"
        )
    lines += [
        "",
        "Stage spans do not overlap. Campaign total also includes uninstrumented setup and report writing.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{report['status']}: {args.output / 'report.md'}")
    return int(report["status"] != "passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seed-start", type=int)
    matrix_parser = commands.add_parser("matrix")
    matrix_parser.add_argument(
        "--campaign",
        choices=("training", "population", "lifecycle", "histories", "benchmark"),
    )
    run = commands.add_parser("run")
    run.add_argument("--binding", required=True)
    run.add_argument(
        "--mode",
        choices=("training", "smoke", "full", "lifecycle", "histories"),
        required=True,
    )
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--anchors-per-split", type=int)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--seed", type=int, default=17)
    config = commands.add_parser("benchmark-config")
    config.add_argument("--case", required=True)
    config.add_argument("--mode", choices=MODES, required=True)
    config.add_argument("--data-dir", type=Path, required=True)
    config.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("summary")
    report.add_argument("--directory", type=Path, required=True)
    report.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        import time

        selected = execution_plan(
            args.seed_start
            if args.seed_start is not None
            else int(time.time()) // 86400 * 12
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selected, indent=2) + "\n")
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a") as stream:
                stream.writelines(
                    f"{group}={json.dumps(rows)}\n" for group, rows in selected.items()
                )
        return 0
    if args.command == "run":
        binding = load_bindings().get(args.binding)
        if binding is None or args.mode not in binding["campaigns"]:
            parser.error("Binding does not declare the requested campaign")
        if (
            args.batch_size < 1
            or args.threads < 1
            or (args.anchors_per_split is not None and args.anchors_per_split < 1)
        ):
            parser.error("Batch size, threads and anchor limit must be positive")
        if args.mode == "full" and args.anchors_per_split is not None:
            parser.error("Full-population runs cannot have an anchor limit")
        return run_binding(args)
    if args.command == "matrix":
        print(json.dumps(campaign_matrix(args.campaign) if args.campaign else matrix()))
    elif args.command == "benchmark-config":
        args.output.write_text(
            json.dumps(benchmark_config(args.case, args.mode, args.data_dir))
        )
    else:
        return int(summarize(args.directory, args.revision)["status"] != "passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
