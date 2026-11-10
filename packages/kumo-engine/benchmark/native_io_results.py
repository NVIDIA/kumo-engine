# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive benchmark variants and replay the committed matrix validation."""

import argparse
import gzip
import hashlib
import importlib
import json
import sys
from pathlib import Path

from io_utils import fsutil


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: str) -> dict:
    return json.loads(fsutil.read_text(path))


def dump(path: str, value: dict) -> None:
    fsutil.write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def driver(repo: str):
    benchmark = str(Path(repo).resolve() / "packages/kumo-engine/benchmark")
    sys.path.insert(0, benchmark)
    module = importlib.import_module("native_io_matrix")
    if str(Path(module.__file__).resolve().parent) != benchmark:
        raise ValueError("Wrong canonical matrix module imported")
    return module


def code_oracle(repo: str) -> dict:
    paths = ("benchmark/native_io_matrix.py", "benchmark/native_io_resources.py",
             "benchmark/native_io_isolation.py", "test/wheel_gpu_smoke.py")
    root = str(Path(repo).resolve() / "packages/kumo-engine")
    return {path: sha(fsutil.read_binary(root + "/" + path)) for path in paths}


def report(module, rows: list[dict], repetitions: int, seed: int, atol: float) -> dict:
    for row in rows:
        spec = row["matrix_spec"]
        module.validate_native(row, spec)
        evidence = module.training_evidence(row, spec)
        if row["training_evidence"] != evidence:
            raise ValueError("Saved training evidence does not match raw snapshots")
        if not spec["pilot"]:
            if evidence["interval_seconds"] < max(30, spec["min_training_seconds"]):
                raise ValueError("Formal trial did not meet training duration")
            if spec["regime"] == "constrained" and not evidence["has_pressure_evidence"]:
                raise ValueError("Formal constrained training lacks measured pressure")
    summary = module.summarize(rows, repetitions, seed, atol)
    del summary["trials"]
    summary["training_pressure"] = [dict(
        task=row["task"], regime=row["regime"], backend=row["backend"],
        repeat=row["repeat"], evidence=row["training_evidence"])
        for row in rows]
    return summary


def describe(module, rows: list[dict], oracles: dict) -> dict:
    controls, trials = {}, []
    for row in rows:
        signature = module.trial_signature(row)
        key = sha(signature.encode())
        oracles.setdefault(key, json.loads(signature))
        controls[row["task"] + "/" + row["regime"]] = key
        trials.append(dict(task=row["task"], regime=row["regime"],
                           backend=row["backend"], repeat=row["repeat"],
                           command=module.runner_command(row["matrix_spec"])))
    return dict(control_oracles=controls, trials=trials)


def cross_correctness(module, rows: list[dict], atol: float) -> None:
    for task in sorted({row["task"] for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        module.validate_work(selected)
        module.validate_predictions(selected, atol)
        if any(row["requested_model_plan"] != selected[0]["requested_model_plan"]
               for row in selected):
            raise ValueError("Cross-regime/cache model plans differ for " + task)


def package(args, module) -> dict:
    if fsutil.exists(args.output_dir):
        raise FileExistsError(args.output_dir)
    raw, variants, oracles, validated_rows = [], {}, {}, []
    for entry in args.variant:
        name, separator, directory = entry.partition("=")
        if not separator or not name or name in variants:
            raise ValueError("Use unique NAME=RESULTS_DIR variants")
        rows = []
        for path in sorted(fsutil.listdir(directory, suffix=".json")):
            if fsutil.isdir(path):
                continue
            text = fsutil.read_text(path)
            row = json.loads(text)
            if "matrix_spec" not in row or "phase_seconds" not in row:
                continue
            rows.append(row)
            raw.append(dict(variant=name, source_file=path,
                            source_sha256=sha(text.encode()), source_text=text))
        try:
            summary = report(module, rows, args.repetitions,
                             args.order_seed, args.prediction_atol)
        except (KeyError, TypeError, ValueError) as error:
            if not args.diagnostic:
                raise
            variants[name] = dict(status="DIAGNOSTIC_INVALID", reason=str(error),
                                  records=len(rows), input_dir=directory)
            continue
        metadata = describe(module, rows, oracles)
        variants[name] = dict(status="DIAGNOSTIC" if args.diagnostic else "VALIDATED",
                              input_dir=directory, summary=summary, **metadata)
        validated_rows.extend(rows)
    if not args.diagnostic:
        cross_correctness(module, validated_rows, args.prediction_atol)
    payload = "".join(canonical(item) + "\n" for item in raw).encode()
    archive = gzip.compress(payload, compresslevel=6, mtime=0)
    fsutil.makedirs(args.output_dir)
    fsutil.write_binary(args.output_dir + "/raw-trials.jsonl.gz", archive)
    manifest = dict(schema_version=2, diagnostic=args.diagnostic,
                    canonical_code=code_oracle(args.repo_root),
                    helper_sha256=sha(fsutil.read_binary(__file__)),
                    repetitions=args.repetitions, order_seed=args.order_seed,
                    prediction_atol=args.prediction_atol,
                    archive_sha256=sha(archive), archive_records=len(raw),
                    variants=variants)
    dump(args.output_dir + "/controls.json", oracles)
    manifest["controls_sha256"] = sha(fsutil.read_binary(
        args.output_dir + "/controls.json"))
    dump(args.output_dir + "/manifest.json", manifest)
    return dict(status="PACKAGED", variants=len(variants), records=len(raw))


def replay(args, module) -> dict:
    manifest = load(args.replay + "/manifest.json")
    if code_oracle(args.repo_root) != manifest["canonical_code"]:
        raise ValueError("Replay requires the exact committed validator sources")
    if sha(fsutil.read_binary(args.replay + "/controls.json")) != manifest["controls_sha256"]:
        raise ValueError("Control oracle hash mismatch")
    archive = fsutil.read_binary(args.replay + "/raw-trials.jsonl.gz")
    if sha(archive) != manifest["archive_sha256"]:
        raise ValueError("Raw archive hash mismatch")
    variants = {name: [] for name in manifest["variants"]}
    records = gzip.decompress(archive).decode().splitlines()
    if len(records) != manifest["archive_records"]:
        raise ValueError("Archive record count mismatch")
    for text in records:
        record = json.loads(text)
        if sha(record["source_text"].encode()) != record["source_sha256"]:
            raise ValueError("Original trial hash mismatch")
        variants[record["variant"]].append(json.loads(record["source_text"]))
    checked, validated_rows, oracles = 0, [], {}
    for name, rows in variants.items():
        saved = manifest["variants"][name]
        if saved["status"] == "DIAGNOSTIC_INVALID":
            continue
        summary = report(module, rows, manifest["repetitions"],
                         manifest["order_seed"], manifest["prediction_atol"])
        if canonical(summary) != canonical(saved["summary"]):
            raise ValueError("Canonical summary replay differs: " + name)
        metadata = describe(module, rows, oracles)
        if any(metadata[key] != saved[key] for key in metadata):
            raise ValueError("Control or executed-command replay differs: " + name)
        validated_rows.extend(rows)
        checked += 1
    if canonical(oracles) != canonical(load(args.replay + "/controls.json")):
        raise ValueError("Reconstructed control oracle differs")
    if not manifest["diagnostic"]:
        cross_correctness(module, validated_rows, manifest["prediction_atol"])
    return dict(status="REPLAY_PASS", validated_variants=checked,
                diagnostic=manifest["diagnostic"], records=len(records))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run packages/kumo-engine/benchmark/native_io_results.py "
        "from the repository root; "
        "replay validates saved trials without training.")
    parser.add_argument(
        "--repo-root", default=str(Path.cwd()),
        help="Repository containing the canonical validators (default: current directory)")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--replay")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=424)
    parser.add_argument("--prediction-atol", type=float, default=1e-6)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if bool(args.replay) == bool(args.output_dir):
        parser.error("Select either --replay or --output-dir")
    if not args.replay and not args.variant:
        parser.error("Packaging requires --variant NAME=RESULTS_DIR")
    module = driver(args.repo_root)
    result = replay(args, module) if args.replay else package(args, module)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
