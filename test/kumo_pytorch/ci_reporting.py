# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concise Actions output; complete command logs and reports stay in artifacts.

This is a standalone, standard-library script so installation failures can also
be reported, including from outside the checkout in installed-wheel tests.
"""

import argparse
import ast
import collections
import glob
import html
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

BAD = {"failed", "failure", "error", "incomplete", "missing", "pending"}
LIMIT = 12
NO_REASON = "See the detailed report for this check."
REPOSITORY = Path(__file__).resolve().parents[2]


def brief(value, limit=320):
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(value))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def cell(value):
    return html.escape(brief(value)).replace("|", "&#124;")


@lru_cache(maxsize=128)
def source_descriptions(path):
    """Read test explanations without importing tests or their dependencies."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, UnicodeError):
        return {}
    descriptions = {}

    def visit(body, prefix=()):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, (*prefix, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ""
                paragraphs = re.split(r"\n\s*\n", doc.strip())
                description = {"title": brief(paragraphs[0])} if doc else {}
                for paragraph in paragraphs[1:]:
                    key, separator, value = paragraph.partition(":")
                    if separator and key in {"Expected", "Evidence"}:
                        description[key.lower()] = brief(value)
                descriptions[(*prefix, node.name)] = description

    visit(tree.body)
    return descriptions


def test_description(classname, name):
    """JUnit retains the exact ID; presentation uses the same checkout's docs."""
    function, _, variant = name.partition("[")
    parts = classname.split(".")
    description = {}
    if all(part.isidentifier() for part in (*parts, function)):
        for root in (
            REPOSITORY,
            REPOSITORY / "test/kumo_pytorch",
            REPOSITORY / "packages/experimental/kumo-pytorch",
        ):
            for end in range(len(parts), 0, -1):
                path = root.joinpath(*parts[:end]).with_suffix(".py")
                if path.is_file():
                    description = source_descriptions(path).get(
                        (*parts[end:], function), {}
                    )
                    break
            if description:
                break
    title = description.get("title") or re.sub(r"^test_", "", function).replace(
        "_", " "
    )
    return {
        **description,
        "title": title[:1].upper() + title[1:],
        "variant": brief(variant.removesuffix("]")),
    }


def observation(message):
    """Explain only what the diagnostic establishes, without guessing a cause."""
    if "CONTRACT AMBIGUITY" in message:
        label = re.search(r"CONTRACT AMBIGUITY [A-Z][0-9]+", message)
        if label and " Observed " in message:
            return brief(f"{label[0]}: {message.rsplit(' Observed ', 1)[1]}")
        return brief(message.removeprefix("AssertionError: "))
    if "Differing items:" in message:
        differences = message.split("Differing items:", 1)[1]
        keys = sorted(set(re.findall(r"['\"](\w+)['\"]:\s*['\"]sha256:", differences)))
        if keys:
            return f"Fingerprint mismatch in {', '.join(keys)}. The compared identities differ."
        message = "Compared mappings differ: " + differences
    split = re.search(
        r"metadata\.splits\[(\d+)\].*?assert (.+?) == (.+?)(?:\n|$)", message, re.S
    )
    if split:
        left, right = split[2].strip(), split[3].strip()
        return brief(
            f"Split metadata differs at position {split[1]}: "
            f"{'missing' if left == 'None' else left} versus {'missing' if right == 'None' else right}."
        )
    mismatch = re.search(r"At index (\d+) diff: ([^\n]+) != ([^\n]+)", message)
    if mismatch:
        return brief(
            f"Values differ at position {mismatch[1]}: {mismatch[2]} versus {mismatch[3]}."
        )
    explanations = (
        (
            "this state was saved for a different task",
            "Iterator restore was rejected because the saved task identity differs.",
        ),
        (
            "one sample exceeds Batching.max_tokens",
            "Batch planning rejected a sample as larger than the token budget.",
        ),
        (
            "Overflow when unpacking long long",
            "Integer conversion overflowed the runtime's signed integer range.",
        ),
    )
    error_text = message
    exception = re.match(r"^([\w.]+(?:Error|Invalid|Exception)): (.*)", message, re.S)
    if exception and exception[1] != "AssertionError":
        error_text = exception[2]
    for detail, explanation in explanations:
        if error_text.startswith(detail):
            return explanation
    if error_text.startswith("Integer value ") and (
        "not in range: -9007199254740992 to 9007199254740992" in error_text
    ):
        return "An integer-to-float conversion exceeded the supported safe range."
    if error_text.startswith("unsupported operand type(s) for ** or pow(): 'NoneType'"):
        return "A calculation attempted exponentiation on a missing value (None) and raised an error."
    if error_text.startswith("post-fit text token lengths missing for "):
        field = error_text.split("post-fit text token lengths missing for ", 1)[
            1
        ].partition("\n")[0]
        resource = f" for {field}" if field else " (field not reported)"
        return brief(
            f"Native batch planning could not find token-length metadata{resource}."
        )
    # Keep useful values but omit opaque hashes and pytest's formatting advice.
    message = re.sub(r"sha256:[0-9a-f]+", "[fingerprint]", message)
    message = "\n".join(
        line
        for line in message.splitlines()
        if not line.strip().startswith(("Omitting ", "Use -v"))
    )
    return brief(message)


def check_title(row):
    title = row.get("title") or row["check"].replace("_", " ")
    if row.get("variant"):
        title += f" [{row['variant']}]"
    return brief(title)


def reason(row):
    if not isinstance(row, dict):
        return brief(row) if row else NO_REASON
    for key in ("error", "reason", "message"):
        if row.get(key):
            return brief(row[key])
    if row.get("traceback"):
        return brief(str(row["traceback"]).strip().splitlines()[-1])
    return NO_REASON


def json_checks(value, path=""):
    """Read statuses, not arbitrary tensors, hashes, measurements or predictions."""
    rows = []
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str):
            refused = value.get("outcome") == "refused"
            rows.append(
                {
                    "check": path
                    or value.get("ref")
                    or value.get("name")
                    or "Campaign",
                    "status": "refused (accepted)" if refused else status,
                    "reason": reason(value.get("refusal", {}))
                    if refused
                    else reason(value),
                }
            )
        for key, child in value.items():
            if key in {
                "checks",
                "cases",
                "shards",
                "lifecycle",
                "histories",
                "repeats",
                "feature_assertions",
                "experiments",
                "metrics",
                "errors",
                "results",
                "comparison",
                "runs",
            }:
                rows.extend(json_checks(child, f"{path} / {key}".strip(" /")))
            elif path.endswith(
                ("checks", "feature_assertions", "experiments", "metrics")
            ):
                if isinstance(child, bool):
                    rows.append(
                        {
                            "check": f"{path} / {key}",
                            "status": "passed" if child else "failed",
                            "reason": "Requirement satisfied."
                            if child
                            else "Requirement was not met.",
                        }
                    )
                else:
                    rows.extend(json_checks(child, f"{path} / {key}"))
        if not status and value.get("error"):
            rows.append(
                {"check": path or "Error", "status": "failed", "reason": reason(value)}
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            label = (
                child.get("name", child.get("ref", child.get("path", index + 1)))
                if isinstance(child, dict)
                else index + 1
            )
            if path.endswith("errors") and isinstance(child, str):
                rows.append({"check": path, "status": "failed", "reason": brief(child)})
            else:
                rows.extend(json_checks(child, f"{path} / {label}"))
    return rows


def read_report(path):
    if path.suffix == ".xml":
        counts = collections.Counter()
        rows = []
        for case in ET.parse(path).iter("testcase"):
            failure = case.find("failure")
            if failure is None:
                failure = case.find("error")
            refused = case.find("./properties/property[@name='capability_refusals']")
            status = (
                "failed"
                if failure is not None
                else "skipped"
                if case.find("skipped") is not None
                else "refused (accepted)"
                if refused is not None
                else "passed"
            )
            counts[status] += 1
            if status != "passed":
                detail = failure if failure is not None else case.find("skipped")
                message = (
                    "Explicit capability refusal; see the structured JUnit evidence."
                )
                if detail is not None:
                    text = (detail.text or "").strip().splitlines()
                    message = detail.get("message") or (text[-1] if text else status)
                description = test_description(
                    case.get("classname", ""), case.get("name", "")
                )
                rows.append(
                    {
                        **description,
                        "check": f"{case.get('classname', '')}::{case.get('name', '')}",
                        "status": status,
                        "reason": observation(message),
                    }
                )
        return rows, dict(counts)
    return json_checks(json.loads(path.read_text())), {}


def tail_reason(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        lines = stream.read().decode(errors="replace").splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    relevant = [
        line
        for line in lines
        if re.search(r"error|failed|exception|assertion|not found|fatal", line, re.I)
    ]
    return brief((relevant or lines or ["Command failed without output."])[-1])


def collect_reports(patterns, directory, since_ns):
    rows, counts, notes = [], collections.Counter(), []
    seen = set()
    for pattern in patterns.splitlines():
        for name in sorted(
            glob.glob(os.path.expandvars(pattern.strip()), recursive=True)
        ):
            path = Path(name)
            if (
                not path.is_file()
                or path.resolve() in seen
                or path.stat().st_mtime_ns < since_ns
            ):
                continue
            seen.add(path.resolve())
            saved = directory / f"{len(seen):03d}-{path.name}"
            shutil.copyfile(path, saved)
            try:
                found, totals = read_report(path)
                rows.extend({**row, "report": saved.name} for row in found)
                counts.update(totals)
            except (ValueError, ET.ParseError, OSError) as error:
                notes.append(
                    f"Cannot read {path.name}: {brief(error)}. Original retained in artifact."
                )
    return rows, dict(counts), notes


def console(result):
    print(
        f"{result['title']}: {result['status'].upper()} ({result['seconds']:.1f}s; exit {result['exit_code']})",
        flush=True,
    )
    if result["tests"]:
        print(
            "Tests: "
            + ", ".join(f"{n} {s}" for s, n in sorted(result["tests"].items())),
            flush=True,
        )
    failures = [r for r in result["checks"] if r["status"] in BAD]
    # Specific nested failures precede the enclosing campaign's generic error.
    failures.sort(key=lambda r: r["check"].count(" / "), reverse=True)
    for row in failures[:5]:
        print(f"- {check_title(row)}", flush=True)
        if row.get("expected"):
            print(f"  Expected: {row['expected']}", flush=True)
        print(f"  Observed: {row['reason']}", flush=True)
        if row.get("evidence"):
            print(f"  Evidence: {row['evidence']}", flush=True)
    if len(failures) > 5:
        print(
            f"- {len(failures) - 5} more reported failures; see the job summary and artifact.",
            flush=True,
        )
    for note in result["notes"]:
        print(brief(note), flush=True)
    if result["exit_code"]:
        message = (
            f"{check_title(failures[0])}: {failures[0]['reason']}"
            if failures
            else "Command failed; see the detailed log."
        )
        escape = lambda s: (
            str(s).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        )
        print(f"::error::{escape(result['title'] + ': ' + message)}", flush=True)
    print(
        f"Full output: {result['log']} in this job's diagnostics artifact.", flush=True
    )


def run(args):
    root = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "pytorch-ci-logs"
    directory = root / (
        re.sub(r"[^a-zA-Z0-9_-]+", "-", args.title)[:64] + "-" + uuid.uuid4().hex[:8]
    )
    directory.mkdir(parents=True)
    log = directory / "console.log"
    started_ns, started = time.time_ns(), time.monotonic()
    heartbeat = started
    print(
        f"Starting: {args.title}. Full output is being saved to the diagnostics artifact.",
        flush=True,
    )
    interrupted = []
    process = None

    def signal_children(signum):
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def cancel(signum, _frame):
        interrupted.append((signum, time.monotonic()))
        signal_children(signum)

    old = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        with log.open("wb") as output:
            process = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if interrupted:
                        if time.monotonic() - interrupted[0][1] > 5:
                            signal_children(signal.SIGKILL)
                    elif time.monotonic() - heartbeat >= 30:
                        print(
                            f"Still running: {args.title} ({time.monotonic() - started:.0f}s; {log.stat().st_size:,} log bytes saved)",
                            flush=True,
                        )
                        heartbeat = time.monotonic()
            code = 128 + interrupted[0][0] if interrupted else process.returncode
            if code < 0:
                code = 128 - code
    except OSError as error:
        with log.open("a") as output:
            output.write(f"{type(error).__name__}: {error}\n")
        code = 127
    finally:
        if interrupted:
            # The shell can exit before a worker that ignored the first signal.
            signal_children(signal.SIGKILL)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    try:
        checks, counts, notes = collect_reports(
            os.environ.get("KUMO_CI_REPORTS", ""), directory, started_ns
        )
    except OSError as error:
        checks, counts, notes = [], {}, [f"Report collection failed: {brief(error)}"]
    if code and not any(
        row["status"] in BAD and row["reason"] != NO_REASON for row in checks
    ):
        checks.append(
            {"check": args.title, "status": "failed", "reason": tail_reason(log)}
        )
    result = {
        "title": args.title,
        "status": "cancelled" if interrupted else "failed" if code else "passed",
        "exit_code": code,
        "seconds": time.monotonic() - started,
        "tests": counts,
        "checks": checks,
        "notes": notes,
        "log": str(log.relative_to(root)),
    }
    temporary = directory / "step.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(directory / "step.json")
    console(result)
    return code


def summary(root, title, artifact_url, job_status):
    records = [json.loads(p.read_text()) for p in sorted(root.glob("*/step.json"))]
    lines = [f"## {cell(title)}", "", f"Job result: **{cell(job_status)}**.", ""]
    if artifact_url:
        lines += [f"[Download full logs and detailed reports]({artifact_url})", ""]
    else:
        lines += [
            "**Diagnostics artifact unavailable.** Inspect the artifact-upload step; full logs may not have been retained.",
            "",
        ]
    lines += ["| Step | Result | Time | Tests |", "| --- | --- | --- | --- |"]
    for row in records:
        tests = ", ".join(f"{n} {s}" for s, n in sorted(row["tests"].items())) or "—"
        lines.append(
            f"| {cell(row['title'])} | {cell(row['status'])} | {row['seconds']:.1f}s | {cell(tests)} |"
        )
    failures = [(r, c) for r in records for c in r["checks"] if c["status"] in BAD]
    failures.sort(key=lambda pair: pair[1]["check"].count(" / "), reverse=True)
    if failures:
        lines += [
            "",
            "### What failed",
            "",
            "| What was tested | Expected | Observed |",
            "| --- | --- | --- |",
        ]
        for record, check in failures[:LIMIT]:
            lines.append(
                f"| {cell(record['title'])} / {cell(check_title(check))} | {cell(check.get('expected', '—'))} | {cell(check['reason'])} |"
            )
        if len(failures) > LIMIT:
            lines += [
                "",
                f"Showing {LIMIT} of {len(failures)} reported failures. All details are retained in the artifact.",
            ]
        lines += [
            "",
            "<details>",
            "<summary>Exact test IDs and documentation</summary>",
            "",
        ]
        for record, check in failures[:LIMIT]:
            lines.append(f"- {cell(record['title'])}: {html.escape(check['check'])}")
            if check.get("evidence"):
                lines.append(f"  Evidence: {cell(check['evidence'])}")
        lines += ["", "</details>"]
    outcomes = [(r, c) for r in records for c in r["checks"] if c["status"] not in BAD]
    if outcomes:
        lines += [
            "",
            "<details>",
            "<summary>Other check outcomes</summary>",
            "",
            "| Step / check | Result |",
            "| --- | --- |",
        ]
        for record, check in outcomes[:LIMIT]:
            lines.append(
                f"| {cell(record['title'])} / {cell(check['check'])} | {cell(check['status'])} |"
            )
        if len(outcomes) > LIMIT:
            lines += [
                "",
                f"{len(outcomes) - LIMIT} additional outcomes are in the detailed reports.",
            ]
        lines += ["", "</details>"]
    for record in records:
        accepted = [r for r in record["checks"] if r["status"] == "refused (accepted)"]
        if accepted:
            lines += [
                "",
                f"**Accepted capability refusals in {cell(record['title'])}:** {len(accepted)}. These are reported separately from executed checks.",
            ]
        lines += (
            ["", *[f"- {cell(note)}" for note in record["notes"]]]
            if record["notes"]
            else []
        )
    incomplete = [
        p for p in root.glob("*/console.log") if not p.with_name("step.json").exists()
    ]
    if incomplete or not records:
        lines += [
            "",
            "Some steps did not produce a completed report. Check setup/cancellation status and the retained partial logs; absence of a report does not mean a pass.",
        ]
    if job_status not in {"success", "passed"} and not failures:
        lines += [
            "",
            "No test failure detail was produced. Inspect the failed setup, action, or artifact-upload step in this job.",
        ]
    lines += [
        "",
        "Detailed logs, tracebacks and copied JSON/JUnit reports are in the linked artifact. Original campaign artifacts are retained separately."
        if artifact_url
        else "Original campaign artifacts may still be available in the run's Artifacts list.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--title", required=True)
    runner.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    if args.action == "run":
        return run(args)
    root = Path(os.environ["RUNNER_TEMP"]) / "pytorch-ci-logs"
    text = summary(
        root,
        os.environ["KUMO_CI_TITLE"],
        os.environ.get("KUMO_CI_ARTIFACT_URL", ""),
        os.environ["KUMO_CI_JOB_STATUS"],
    )
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
