"""Catalog-driven CI shards and a combined report; no native imports for planning."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ("training", "index", "timing-diskgraph", "timing-polars", "timing-gpu")


def selected_cases(case=None):
    cases = [
        (selection, family)
        for selection in json.loads((ROOT / "workloads/aml.json").read_text())
        for family in selection["models"]
    ]
    names = [f"{s['selector']['workload_id']}-{f}" for s, f in cases]
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


def matrix():
    return {
        "include": [
            {"case": f"{s['selector']['workload_id']}-{f}"} for s, f in selected_cases()
        ]
    }


def benchmark_config(case, mode, data_dir):
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    selection, family = selected_cases(case)[0]
    config = json.loads((ROOT / "workloads/aml-benchmark.json").read_text())
    config.update(
        data_dir=str(data_dir),
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
    return [
        {
            "name": case,
            "adapter": "test.kumo_pytorch.aml_benchmark:factory",
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


def summarize(directory):
    """Missing or unsuccessful shards fail; keep every worker's linked reports."""
    from .benchmarking.timing_report import timing_summary

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, timing_cases = [], []
    for entry in matrix()["include"]:
        case = entry["case"]
        combine_fitting(directory / "fitting" / case, case)
        for kind in (*MODES, "fitting", "smoke", "full"):
            base = directory / kind / case
            path = base / ("results.json" if kind in MODES else "report.json")
            row = {"case": case, "kind": kind, "status": "missing"}
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
                    selected = result["cases"][0]
                    refused = selected.get("outcome") == "refused"
                    if refused:
                        evidence = selected["refusal"]
                        assert kind in {"smoke", "full"}
                        assert evidence["stage"] == "preflight"
                        assert evidence["execution"] == "not_started"
                        assert evidence["issues"] and all(
                            all(
                                issue.get(key)
                                for key in ("code", "path", "reason", "suggestion")
                            )
                            for issue in evidence["issues"]
                        )
                        assert result["full_population_verified"] is False
                        assert selected["full_population_verified"] is False
                        assert selected["splits"] and all(
                            split["status"] == "not_executed"
                            and split["delivered"] == 0
                            for split in selected["splits"].values()
                        )
                        row.update(outcome="refused", refusal=evidence)
                    if (
                        kind == "full"
                        and not refused
                        and not result["full_population_verified"]
                    ):
                        raise ValueError("Full population was not verified")
                row["report"] = str((base / "report.md").relative_to(directory))
                if kind.startswith("timing-"):
                    timing_cases.append({"name": f"{kind}/{case}/{case}"})
            except Exception as error:  # noqa: BLE001 - missing evidence must stay visible
                row.update(status="failed", error=str(error))
            rows.append(row)
    status = (
        "passed"
        if all(r["status"] in {"passed", "complete"} for r in rows)
        else "failed"
    )
    result = {"status": status, "shards": rows}
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Parallel AML nightly",
        "",
        f"Status: **{status}**",
        "",
        "| Case | Experiment | Status | Report |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        link = f"[report]({row['report']})" if "report" in row else "Missing report"
        label = row["status"] + (
            " (refused)" if row.get("outcome") == "refused" else ""
        )
        lines.append(f"| {row['case']} | {row['kind']} | {label} | {link} |")
    lines += ["", *timing_summary(timing_cases, directory)]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("matrix")
    config = commands.add_parser("benchmark-config")
    config.add_argument("--case", required=True)
    config.add_argument("--mode", choices=MODES, required=True)
    config.add_argument("--data-dir", type=Path, required=True)
    config.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("summary")
    report.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "matrix":
        print(json.dumps(matrix()))
    elif args.command == "benchmark-config":
        args.output.write_text(
            json.dumps(benchmark_config(args.case, args.mode, args.data_dir))
        )
    else:
        return int(summarize(args.directory)["status"] != "passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
