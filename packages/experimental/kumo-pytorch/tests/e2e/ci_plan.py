# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan the package workflow and reconcile its current-attempt jobs and test evidence."""

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
from time import time
from xml.etree import ElementTree

from tests.e2e.binding_plan import BINDINGS, NIGHTLY_TRAINING, selections
from tests.e2e.binding_report import BindingReport, require_fields
from tests.e2e.nightly_report import training_tables, validate_training
from tests.e2e.runs import (
    AML_MODES,
    CUDA_REGRESSIONS,
    HI_SMALL_BINDINGS,
    TIMING_CASES,
    _aml_evidence,
    _cuda_junit_error,
)
from tests.e2e.tiers import T2A, T2B, matrix

WORKFLOW = ".github/workflows/kumo-pytorch-tests.yml"
PACKAGE_JOB = "Package tests / Package and dataset tests"
AML = ("aml", "l01-cuda", "hi-small-memory")
STREAMING = ("all", "smoke", "training", "lifecycle", "full")
CONFORMANCE = [f"kumo-pytorch-t2b-conformance-{i}" for i in range(3)]


def gh(endpoint, *, pages=False):
    """Read from the current repository with the workflow's read-only token."""
    args = ["gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{endpoint}"]
    if pages:
        args += ["--paginate", "--slurp"]
    return json.loads(subprocess.check_output(args, text=True))


def make_plan(event_name, event, *, ref, **identity):
    pr = event.get("pull_request", {})
    repository = identity["repository"]
    inputs = event.get("inputs") or {}
    suite = event_name in {"pull_request", "schedule"} or (
        event_name == "workflow_dispatch" and not inputs
    )
    if event_name == "pull_request" and pr["head"]["repo"]["full_name"] != repository:
        raise ValueError(
            "The full package suite requires private datasets and a same-repository PR"
        )
    mode = "package" if suite else inputs.get("mode")
    if not suite and (
        event_name != "workflow_dispatch"
        or mode not in ("cuda", "binding-full", "binding-train", *AML, *STREAMING)
    ):
        raise ValueError(f"Unknown benchmark mode {mode}")
    if repository != "NVIDIA/kumo-engine" or (not pr and not ref.startswith("refs/heads/")):
        raise ValueError("Package CI requires a trusted repository branch")
    binding = inputs.get("binding", "")
    training = []
    if suite or mode == "binding-train":
        if not suite and binding != "all" and binding not in NIGHTLY_TRAINING:
            raise ValueError(f"Unknown training binding: {binding}")
        training = list(NIGHTLY_TRAINING) if suite or binding == "all" else [binding]
    plan = dict(
        **identity,
        event=event_name,
        ref=ref,
        source_revision=pr["head"]["sha"] if pr else identity["revision"],
        mode=mode,
        torch_backend="cu130" if mode in ("cuda", "aml", "l01-cuda", "binding-train") else "cpu",
        binding=binding,
        training_bindings=training,
        seed_start=int(time()) // 86400 * 12 if suite else None,
        suite=suite,
    )
    plan["jobs"] = planned_jobs(plan)
    plan["selections"] = selection_rows(plan)
    return plan


def binding_label(binding):
    dataset = {"S": "Synthetic", "M": "MBD", "L": "AML"}[binding[0]]
    return f"{dataset} {binding}"


def training_matrix(plan):
    rows = [dict(label=binding_label(b), binding=b) for b in plan["training_bindings"]]
    if plan["suite"]:
        for tier, dataset in ((T2A, "H&M Small"), (T2B, "H&M Medium")):
            for row in matrix(tier)["include"]:
                flow, seed = row["flow"], str(row["seed"])
                rows.append(
                    dict(
                        label=f"{dataset} · {flow} (seed {seed})",
                        tier=tier.name,
                        flow=flow,
                        seed=seed,
                    )
                )
    return {"include": rows}


def training_job(label, worker, plan):
    prefix = "Training · " if plan["suite"] else "GPU training / "
    return f"{prefix}{label} / {worker}"


def planned_jobs(plan):
    mode = plan["mode"]
    expected = {
        "Select suite jobs" if plan["suite"] else "Select benchmark": True,
        "Build / diskgraph-wheel": True,
    }
    if plan["training_bindings"]:
        for row in training_matrix(plan)["include"]:
            expected[training_job(row["label"], "GPU" if "binding" in row else "CPU", plan)] = True
    else:
        expected["GPU training"] = False
    if not plan["suite"]:
        expected.update(
            {
                "dataset-source": True,
                "cuda": mode == "cuda",
                "binding-full": mode == "binding-full",
                "aml-benchmark": mode in AML,
                "streaming-25b": mode in STREAMING,
            }
        )
        return expected

    expected[PACKAGE_JOB] = True
    expected["Distribution / Install from wheel"] = True
    for dataset in ("H&M Small", "H&M Medium"):
        expected[f"Lifecycle · {dataset} / Check saved models"] = True
    for i in range(1, len(CONFORMANCE) + 1):
        expected[f"Package tests / Engine conformance {i}/{len(CONFORMANCE)}"] = True
    return expected


def account_jobs(plan, raw):
    expected = plan["jobs"]
    inactive_workers = {
        training_job(row["label"], "CPU" if "binding" in row else "GPU", plan)
        for row in training_matrix(plan)["include"]
    }
    jobs, errors = {}, []
    for job in raw:
        name = job["name"]
        if name in {"Publish results", "kumo-pytorch-gate"}:
            continue  # These are evaluated at the gate and in the completed scheduled run.
        if job["conclusion"] == "skipped" and name in inactive_workers:
            continue
        if name in jobs or (job["head_sha"], job["run_attempt"], job["run_id"]) != (
            plan["source_revision"],
            plan["attempt"],
            plan["run_id"],
        ):
            errors.append(f"Duplicate or stale job: {name}")
        jobs[name] = {"result": job["conclusion"], "url": job["html_url"]}
        if name == PACKAGE_JOB:
            jobs[name]["steps"] = {
                s["name"]: s["conclusion"]
                for s in job.get("steps", [])
                if s["name"] in {"Test suite", "Style", "Package build"}
            }
    for name in sorted(jobs.keys() | expected.keys()):
        actual = jobs.get(name, {}).get("result", "missing")
        want = ("success" if expected[name] else "skipped") if name in expected else "absent"
        if actual != want:
            errors.append(f"{name}: expected {want}, observed {actual}")
    return jobs, errors


def artifact_name(name, attempt):
    return name if attempt == 1 else f"{name}-attempt-{attempt}"


def evidence_names(plan):
    names = {r["artifact"] for r in plan["selections"] if r["status"] == "required"}
    if plan["suite"]:
        names.add("kumo-pytorch-tests")
        names.update(CONFORMANCE)
    return {artifact_name(name, plan["attempt"]) for name in names}


def read_json(path):
    return json.loads(path.read_text())


def only(directory, pattern):
    paths = list(directory.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {directory.name}, found {len(paths)}")
    return paths[0]


def receipt_evidence(directory, rows, revision):
    # Source-oracle validation and the entire test, including teardown, precede publication.
    outcomes = {}
    for case in ElementTree.parse(only(directory, "*.xml")).iter("testcase"):
        prop = case.find("properties/property[@name='nodeid']")
        if prop is not None:
            passed = all(case.find(t) is None for t in ("error", "failure", "skipped"))
            outcomes.setdefault(prop.get("value"), []).append(passed)
    for row in rows:
        nodeid = row["nodeid"]
        key = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
        try:
            path = only(directory, f"binding-{row['mode']}-{key}.json")
            data = read_json(path)
            fields = {k: row[k] for k in ("binding", "mode", "selector", "representation", "scope")}
            oracle = data["expected"].copy()
            checks = oracle.pop("checks")
            if row["mode"] == "full" and set(oracle["counts"]) != {"train", "valid", "test"}:
                raise ValueError("Missing full-population splits")
            outcome = BindingReport(**fields, **oracle).validate(
                data, revision, checks, allowed_refusal=row.get("refusal")
            )
            if outcomes.get(nodeid) != [True]:
                raise ValueError(f"Expected one passing JUnit case for {nodeid}")
            row.update(
                status=outcome,
                splits=data["splits"],
                checks=data["checks"],
                report=str(path.relative_to(directory)),
            )
        except (KeyError, ValueError, OSError, TypeError) as error:
            row.update(status="missing or invalid", error=str(error))


def benchmark_evidence(directory, rows, revision, mode):
    if mode == "aml":
        manifest = only(directory, "campaign.json")
        campaign = read_json(manifest)
        require_fields(campaign, {"complete": True, "revision": revision}, manifest)
        pairs = [(r["binding"], r["repetition"]) for r in campaign["hi_small_pairs"]]
        if pairs != list(product(HI_SMALL_BINDINGS, range(3))) or set(campaign["reports"]) != set(
            AML_MODES
        ):
            raise ValueError("Incomplete AML selections")
        reports = {name: manifest.parent / path for name, path in campaign["reports"].items()}
    else:
        name = "hi-small" if mode == "hi-small-memory" else mode
        reports = {name: only(directory, "timings.json")}
    for name, path in reports.items():
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Report outside this artifact")
        data = read_json(path)
        if name == "cuda":
            require_fields(
                data,
                dict(
                    revision=revision,
                    tracked_diff_sha256=None,
                    device="cuda",
                    workload=TIMING_CASES["cuda"],
                ),
                path,
            )
            (worker,) = data["workers"]
            require_fields(worker, {"complete": True, "returncode": 0}, path)
            epochs = read_json(path.parent / worker["results"])["training"]["epochs"]
            if [e["epoch"] for e in epochs] != [0, 1] or not all(
                e["cuda"]["complete"] and sum(e["batch_sizes"]) == 33 for e in epochs
            ):
                raise ValueError("Expected two completed 33-anchor CUDA epochs")
            error = _cuda_junit_error(
                path.parent / worker["junit"], [TIMING_CASES["cuda"], *CUDA_REGRESSIONS]
            )
            if error:
                raise ValueError(error)
            nodes = [
                r["nodeid"]
                for r in selections()
                if r["mode"] == "training" and r["binding"] in NIGHTLY_TRAINING
            ]
            nodes.append("tests/e2e/test_aml.py::test_l01_nightly_outlier")
            if error := _cuda_junit_error(only(directory, "binding-cuda.xml"), nodes):
                raise ValueError(error)
        else:
            _aml_evidence(name, path, revision)
        for row in rows:
            if row["mode"] == name:
                row.update(status="executed", report=str(path.relative_to(directory)))
                if name == "hi-small":
                    binding, rep = row["binding"], row["repetition"]
                    summary = data["summary"][binding]
                    row["summary"] = summary["growth_ratio"]
                    for size in ("25000", "400000"):
                        row[size] = {k: v["values"][rep] for k, v in summary[size].items()}
                        row[size]["report"] = str(
                            path.parent.relative_to(directory)
                            / binding
                            / str(rep)
                            / size
                            / "results.json"
                        )
                else:
                    row["summary"] = data["summary"]


def selection_rows(plan):
    mode = plan["mode"]
    rows = []
    for row in selections():
        mode_name = row["mode"]
        if mode_name == "full":
            if mode != "binding-full" or plan["binding"] != row["binding"]:
                continue
            row["artifact"] = f"binding-full-{row['binding']}"
        elif plan["suite"]:
            row["artifact"] = "kumo-pytorch-tests"
        else:
            continue
        # The six default lifecycle bindings exclude the separately grouped HI Small probes.
        if mode_name == "histories" and row["binding"] == "L01":
            row["campaign"] = "correctness"
        elif (
            mode_name in ("correctness", "training", "lifecycle", "full")
            and row["group"] != "hi-small-lifecycle"
        ):
            row["campaign"] = mode_name
        row["status"] = "required"
        rows.append(row)
    if mode == "binding-full" and not rows:
        raise ValueError(f"Unknown full-population binding: {plan['binding']}")
    rows.extend(
        dict(
            binding=b,
            mode="two-epoch-training",
            representation=BINDINGS[b][-1],
            group="GPU training",
            scope="full_source",
            artifact=f"binding-train-{b}",
            status="required",
        )
        for b in plan["training_bindings"]
    )
    for campaign in STREAMING[1:]:
        if mode not in ("all", campaign):
            continue
        node = (
            "test_streaming_weekly_session_forecast[25b-full]"
            if campaign == "smoke"
            else f"test_streaming_campaign[{campaign}]"
        )
        rows.append(
            dict(
                binding="T01",
                mode=campaign,
                representation="graphsage",
                group="real 25B",
                artifact="streaming-25b-" + mode,
                status="required",
                selector="25b-full" if campaign == "smoke" else "streaming-25b-full.json",
                scope="full_source" if campaign == "full" else "bounded",
                nodeid="tests/e2e/test_streaming.py::" + node,
            )
        )
        if campaign == "smoke":
            for selector in ("bounded", "25b"):
                rows.append(
                    dict(
                        rows[-1],
                        selector=selector,
                        nodeid=(
                            f"tests/e2e/test_streaming.py::test_streaming_weekly_session_forecast[{selector}]"
                        ),
                    )
                )
    for name in (*AML_MODES, "cuda"):
        if not (
            mode == name
            or (mode == "aml" and name in AML_MODES)
            or (mode == "hi-small-memory" and name == "hi-small")
        ):
            continue
        pairs = (
            product(HI_SMALL_BINDINGS, range(3))
            if name == "hi-small"
            else [("33 anchors" if name == "cuda" else "L01", None)]
        )
        for binding, rep in pairs:
            rows.append(
                dict(
                    binding=binding,
                    mode=name,
                    group="benchmark",
                    representation="Linear" if name == "cuda" else BINDINGS[binding][-1],
                    repetition=rep,
                    artifact=mode + "-measurements",
                    status="required",
                )
            )
    return rows


def results(plan, jobs, artifacts, directory):
    checked, errors = account_jobs(plan, jobs)
    rows = [r.copy() for r in plan["selections"]]
    # The Actions API lists artifacts across attempts. The publisher suffixes retries.
    url = f"https://github.com/{plan['repository']}/actions/runs/{plan['run_id']}/artifacts/"
    links = {
        a["name"]: url + str(a["id"])
        for a in artifacts
        if not a["expired"]
        and a["workflow_run"]["id"] == plan["run_id"]
        and a["workflow_run"]["head_sha"] == plan["source_revision"]
    }
    matched = evidence_names(plan) & {a["name"] for a in artifacts if not a["expired"]}

    def evidence(name):
        name = artifact_name(name, plan["attempt"])
        if name in links:
            if (directory / name).is_dir():
                return directory / name
            # download-artifact extracts a single match directly into the destination.
            if (
                matched == {name}
                and directory.is_dir()
                and any(p.is_file() for p in directory.iterdir())
            ):
                return directory
        raise ValueError(f"Missing current-attempt artifact {name}")

    inventory = Counter(r["campaign"] for r in rows if "campaign" in r)
    report = dict(
        plan=plan,
        jobs=checked,
        errors=errors,
        selections=rows,
        inventory=inventory,
        artifacts=links,
    )
    if plan["training_bindings"]:
        workers = {j["name"]: j for j in jobs}
        try:
            intervals = []
            for binding in plan["training_bindings"]:
                job = workers[training_job(binding_label(binding), "GPU", plan)]
                intervals.append(
                    (
                        datetime.fromisoformat(job["started_at"]),
                        datetime.fromisoformat(job["completed_at"]),
                    )
                )
            if any(end < start for start, end in intervals):
                raise ValueError("job finished before starting")
            report["training_wall_seconds"] = (
                max(end for _, end in intervals) - min(start for start, _ in intervals)
            ).total_seconds()
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Missing or invalid GPU job timing: {exc}")
    for name in sorted({r["artifact"] for r in rows if r["status"] == "required"}):
        group = [r for r in rows if r["status"] == "required" and r["artifact"] == name]
        try:
            source = evidence(name)
            if group[0]["group"] == "GPU training":
                (row,) = group
                path = only(source, "report.json")
                data = read_json(path)
                validate_training(data, row["binding"], plan)
                row.update(status="executed", report=path.name, training=data)
            elif group[0]["group"] == "benchmark":
                benchmark_evidence(source, group, plan["revision"], plan["mode"])
            else:
                receipt_evidence(source, group, plan["revision"])
            for row in group:
                row["artifact"] = artifact_name(name, plan["attempt"])
                if row["status"] not in ("executed", "refused"):
                    errors.append(
                        f"{row['binding']} {row['mode']}: {row.get('error', row['status'])}"
                    )
        except (KeyError, TypeError, ValueError, OSError, ElementTree.ParseError) as exc:
            for row in group:
                if row["status"] == "required":
                    row.update(status="missing or invalid", error=str(exc))
            errors.append(f"{name}: {exc}")
    if plan["suite"]:
        from conformance.shards import verify_shards

        try:
            paths = [only(evidence(name), "conformance-shard-*.json") for name in CONFORMANCE]
            report["conformance_cases"] = verify_shards(
                paths,
                revision=plan["revision"],
                run_id=plan["run_id"],
                attempt=plan["attempt"],
                seed_start=plan["seed_start"],
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"Conformance: {exc}")
    report["status"] = "failed" if errors else "passed"
    return report


def render(report):
    rows = [f"### Package suite accounting · {report['status']}"]
    rows += training_tables(
        [r for r in report["selections"] if r["group"] == "GPU training"],
        report.get("training_wall_seconds"),
    )
    rows += [
        "",
        "| Job | Outcome |",
        "| --- | --- |",
    ]
    for name, job in sorted(report["jobs"].items()):
        rows.append(f"| [{name}]({job['url']}) | {job['result']} |")
        for step, outcome in job.get("steps", {}).items():
            rows.append(f"| ↳ {step} | {outcome} |")
    rows += [
        "",
        "| Dataset / binding | Model or representation | Mode | Group | Outcome |",
        "| --- | --- | --- | --- | --- |",
    ]
    datasets = {
        "E": "HI Small",
        "L": "LI Medium",
        "M": "MBD",
        "S": "Synthetic Medium",
        "T": "Streaming 25B",
    }
    for row in report["selections"]:
        state = row["status"]
        if row["status"] in {"executed", "refused", "missing or invalid"} and (
            link := report["artifacts"].get(row["artifact"])
        ):
            state = f'[{state}]({link} "{row.get("report", "")}")'
        label = row["mode"] + (f" {row['repetition']}" if row.get("repetition") is not None else "")
        binding = row["binding"]
        cells = (
            f"{datasets.get(binding[:1], '')} {binding}".strip(),
            row.get("representation", ""),
            label,
            row.get("group", ""),
            state,
        )
        rows.append("| " + " | ".join(cells) + " |")
    if report["errors"]:
        rows += ["", "Required work needing attention", ""]
        rows.extend(f"- {error}" for error in report["errors"])
    return "\n".join(rows)


def gate(plan, report):
    require_fields(
        report,
        {"plan": plan, "errors": [], "status": "passed"},
        "gate",
    )
    jobs = {name: row["result"] for name, row in report["jobs"].items()}
    if jobs != {
        name: "success" if required else "skipped" for name, required in plan["jobs"].items()
    }:
        raise ValueError("Final job set or outcome does not match the plan")
    expected = plan["selections"]

    def identity(row):
        return row["binding"], row["mode"], row.get("selector"), row.get("repetition")

    observed = {identity(r): r for r in report["selections"]}
    if len(observed) != len(report["selections"]) or set(observed) != {
        identity(r) for r in expected
    }:
        raise ValueError("Final selection set does not match the plan")
    for row in expected:
        actual = observed[identity(row)]
        require_fields(
            actual, {k: v for k, v in row.items() if k not in {"artifact", "status"}}, "selection"
        )
        accepted = {"executed", "refused"} if row.get("refusal") else {"executed"}
        if row["status"] != "required":
            accepted = {row["status"]}
        if actual["status"] not in accepted:
            raise ValueError(f"Incomplete {identity(row)}: {actual['status']}")


def scheduled(report, run, jobs):
    plan = report["plan"]
    require_fields(
        plan, {"event": "schedule", "ref": "refs/heads/main", "suite": True}, "nightly evidence"
    )
    require_fields(
        run,
        dict(
            path=WORKFLOW,
            event="schedule",
            head_branch="main",
            head_sha=plan["revision"],
            id=plan["run_id"],
            run_attempt=plan["attempt"],
            status="completed",
        ),
        "scheduled run",
    )
    observed, errors = account_jobs(plan, jobs)
    if observed != report["jobs"] or not set(errors).issubset(report["errors"]):
        raise ValueError("Scheduled evidence differs from actual job outcomes")
    final = {
        j["name"]: j["conclusion"]
        for j in jobs
        if j["name"] in {"Publish results", "kumo-pytorch-gate"}
    }
    success = not report["errors"]
    if success:
        gate(plan, report)
    if final != {
        "Publish results": "success",
        "kumo-pytorch-gate": "success" if success else "failure",
    } or run["conclusion"] != ("success" if success else "failure"):
        raise ValueError(
            "Scheduled gate or workflow conclusion differs from the published evidence"
        )
    return {
        "scheduled_trigger": "verified",
        "tests": run["conclusion"],
        "run": run["html_url"],
        "revision": plan["revision"],
        "attempt": plan["attempt"],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "results", "gate", "scheduled"):
        command = commands.add_parser(name)
        if name in {"plan", "results"}:
            command.add_argument("--output", type=Path, required=True)
        if name in {"results", "gate"}:
            command.add_argument("--plan", type=Path)
        if name in {"gate", "scheduled"}:
            command.add_argument("--results", type=Path, required=True)
        if name == "results":
            command.add_argument("--directory", type=Path, required=True)
    args = cli.parse_args()
    if args.cmd != "scheduled":
        identity = dict(
            revision=os.environ["GITHUB_SHA"],
            run_id=int(os.environ["GITHUB_RUN_ID"]),
            attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
            repository=os.environ["GITHUB_REPOSITORY"],
        )
    if args.cmd == "plan":
        event = read_json(Path(os.environ["GITHUB_EVENT_PATH"]))
        name = os.environ["GITHUB_EVENT_NAME"]
        data = make_plan(
            name,
            event,
            ref=os.environ["GITHUB_REF"],
            **identity,
        )
        # Verify the executor paths against this checkout before scheduling workers.
        paths = (
            {
                "tests/e2e/test_dataset_staging.py",
                "tests/test_notebooks.py",
                "tests/test_docs_code.py",
                "tests/e2e/test_lifecycle.py",
            }
            if data["suite"]
            else set()
        )
        paths.update(
            r["nodeid"].split("::")[0]
            for r in data["selections"]
            if r["status"] == "required" and r.get("nodeid")
        )
        for path in paths:
            if not Path(path).is_file():
                raise ValueError(f"Planned test path is missing: {path}")
        if output := os.getenv("GITHUB_OUTPUT"):
            with open(output, "a") as handle:
                print(f"training={json.dumps(training_matrix(data))}", file=handle)
                names = sorted(evidence_names(data))
                pattern = "{" + ",".join(names) + "}" if len(names) > 1 else "".join(names)
                print(f"evidence_pattern={pattern}", file=handle)
                print(f"plan={json.dumps(data, separators=(',', ':'))}", file=handle)
    else:
        if args.cmd == "scheduled":
            report = read_json(args.results)
            plan = report["plan"]
        else:
            plan = read_json(args.plan) if args.plan else json.loads(os.environ["PLAN_JSON"])
            require_fields(plan, identity, "workflow run")
        if args.cmd == "gate":
            gate(plan, read_json(args.results))
            return
        endpoint = f"actions/runs/{plan['run_id']}"
        attempt = f"{endpoint}/attempts/{plan['attempt']}"
        jobs = [j for page in gh(attempt + "/jobs?per_page=100", pages=True) for j in page["jobs"]]
        if args.cmd == "scheduled":
            print(json.dumps(scheduled(report, gh(attempt), jobs), indent=2))
            return
        artifacts = [
            a
            for page in gh(endpoint + "/artifacts?per_page=100", pages=True)
            for a in page["artifacts"]
        ]
        data = results(plan, jobs, artifacts, args.directory)
        markdown = render(data)
        args.output.with_suffix(".md").write_text(markdown + "\n")
        if summary := os.getenv("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as handle:
                print(markdown, file=handle)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
