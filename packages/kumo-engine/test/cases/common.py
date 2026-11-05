# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Engine run and result checks. Requirements need no Engine import."""

import argparse
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path


def parse_args(case, scales, *, family, prediction_rows=20,
               metric_tolerance=.25, prediction_tolerance=1e-5):
    parser = argparse.ArgumentParser(description=f"Run the {case} Engine case")
    parser.add_argument("--scale", choices=tuple(scales), required=True)
    parser.add_argument("--requirements", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument(
        "--scratch", type=Path,
        default=Path(os.environ.get("DISKGRAPH_DATA_ROOT", "/mnt/localssd")))
    parser.add_argument("--source-sha")
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    args.case, args.scratch_gib = case, scales[args.scale]
    if args.requirements:
        print(json.dumps({"scratch_gib": args.scratch_gib}))
        parser.exit()
    if args.reference is None or not re.fullmatch(r"[0-9a-f]{40}",
                                                  args.source_sha or ""):
        parser.error(
            "--reference and the Engine product --source-sha are required")
    args.source = f"{os.environ['ENGINE_E2E_DATA_S3'].rstrip('/')}/{family}/{args.scale}/"
    args.region = os.environ.get("AWS_REGION", "us-west-2")
    args.prediction_rows = prediction_rows
    args.metric_tolerance, args.prediction_tolerance = metric_tolerance, prediction_tolerance
    return args


def sources(args):
    import kumo.engine as kumo
    kumo.configure(scratch_dir=args.scratch)
    return kumo.S3Connector(root_dir=args.source, region=args.region)


def table(sources, name, key, clock, columns):
    from kumoapi.table import Column, TableDefinition

    import kumo.engine as kumo
    return kumo.Table(
        sources, name, definition=TableDefinition(
            source_table=sources[name].to_source_table(), pkey=key,
            time_col=clock, cols=[
                Column(
                    column, stype, dtype,
                    timestamp_format="ms" if stype == "timestamp" else None,
                    multicategorical_delimiter="|"
                    if stype == "multicategorical" else None)
                for column, stype, dtype in columns
            ]))


def table_counts(training_table):
    manifest = json.loads(training_table.manifest_path.read_text())
    path = training_table.manifest_path.parent / manifest["split_stats"]["path"]
    stats = json.loads(path.read_text())
    splits = {}
    for split in ("train", "val", "test"):
        values = stats[f"{split}_table_stats"]
        counts = (values.get("target_distribution") or {}).get("count", {})
        splits[split] = {"rows": values["num_train_rows"]}
        if "true" in counts or "false" in counts:
            splits[split]["positive_rows"] = counts.get("true", 0)
    return {
        "rows": sum(part["rows"] for part in splits.values()),
        "splits": splits
    }


def evaluation_steps(split, batch_size, min_class_rows):
    """Size a prefix for expected support in the less common class."""
    rows, positives = split["rows"], split.get("positive_rows", 0)
    minority = min(positives, rows - positives)
    if minority <= 0:
        raise ValueError("Binary evaluation needs both classes in each split")
    population = min(rows, math.ceil(min_class_rows * rows / minority))
    return math.ceil(population / batch_size)


def table_fingerprint(training_table, scratch):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as parquet

    from kumo.engine import FileConnector, TrainingTableOutputConfig
    destination = scratch / "export"
    training_table.export(
        TrainingTableOutputConfig(FileConnector(root_dir=destination),
                                  "answers"))
    total, count, schema = 0, 0, None
    try:
        for path in sorted(destination.rglob("*.parquet")):
            reader = parquet.ParquetFile(path)
            here = sorted([field.name, str(field.type)]
                          for field in reader.schema_arrow)
            if schema is not None and schema != here:
                raise ValueError("Training-table parts have different schemas")
            schema = here
            for batch in reader.iter_batches(batch_size=65536):
                frame = batch.to_pandas().sort_index(axis=1)
                for field in batch.schema:
                    if (pa.types.is_list(field.type)
                            or pa.types.is_large_list(field.type)
                            or pa.types.is_fixed_size_list(field.type)):
                        frame[field.name] = [
                            repr(value)
                            for value in batch.column(field.name).to_pylist()
                        ]
                total = (total + int(
                    pd.util.hash_pandas_object(frame,
                                               index=False).sum())) % 2**64
                count += len(frame)
    finally:
        shutil.rmtree(destination)
    if not count:
        raise ValueError("Training table is empty")
    return {
        "algorithm": "pandas-row-sum-v1",
        "schema": schema,
        "rows": count,
        "sum": total
    }


def prediction_table(training_table, query, snapshot, args, generation):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    from kumo.engine import FileConnector, PredictionTable

    # Native stage files already hold the same source IDs as the public export.
    spec = json.loads(
        (training_table.staging_dir / "tt_ingest_spec.json").read_text())
    rows, remaining = [], args.prediction_rows
    for stage in spec["split_stages"]:
        if stage["key"] != "test":
            continue
        for part in stage["parts"]:
            reader = parquet.ParquetFile(training_table.staging_dir / part)
            keys = [
                key for key in ("ENTITY1", "TIMESTAMP", "autoregression_labels")
                if key in reader.schema_arrow.names
            ]
            for batch in reader.iter_batches(batch_size=remaining,
                                             columns=keys):
                rows.append(pa.Table.from_batches([batch]).slice(0, remaining))
                remaining -= len(rows[-1])
                if not remaining:
                    break
            if not remaining:
                break
    if not rows:
        raise ValueError("Held-out prediction population is empty")
    directory = args.scratch / "prediction-input"
    directory.mkdir()
    path = directory / "inputs.parquet"
    parquet.write_table(pa.concat_tables(rows), path)
    return PredictionTable.from_source_table(
        FileConnector(root_dir=directory)[path.name], pq=query,
        snapshot=snapshot, plan=generation)


def check_predictions(first, repeat, reloaded, tolerance):
    import numpy as np
    import pandas as pd
    frames = [
        value if isinstance(value, pd.DataFrame) else value.frame
        for value in (first, repeat, reloaded)
    ]
    keys = [
        key for key in ("ENTITY", "ANCHOR_TIMESTAMP", "TIMESTAMP", "CLASS")
        if key in frames[0]
    ]
    score = next(key for key in ("SCORE", "True_PROB", "TARGET_PRED")
                 if key in frames[0])
    values = []
    for frame in frames:
        if (frame.empty or "ENTITY" not in keys or any(key not in frame
                                                       for key in keys)
                or frame[keys].isna().any().any()
                or frame.duplicated(keys).any()):
            raise ValueError(
                "Prediction coordinates must be nonempty and unique")
        scores = frame.set_index(keys)[score].sort_index().astype(float)
        if not np.isfinite(scores.to_numpy()).all():
            raise ValueError("Prediction scores must be finite")
        values.append(scores)
    maximum = 0.0
    for current in values[1:]:
        if not current.index.equals(values[0].index):
            raise ValueError("Repeated prediction coordinates differ")
        maximum = max(
            maximum,
            float(np.abs(current.to_numpy() - values[0].to_numpy()).max()))
    if maximum > tolerance:
        raise ValueError("Repeated prediction scores exceed tolerance")
    return {"max_abs_difference": maximum}


def compare_reference(result, expected, tolerance):
    for key in ("graph", "training_table"):
        if result[key] != expected[key]:
            raise ValueError(f"Reference {key} differs")
    if "metrics" not in result:
        return
    if result["metrics"].keys() != expected["metrics"].keys():
        raise ValueError("Reference metric names differ")
    for name, value in result["metrics"].items():
        baseline = expected["metrics"][name]
        difference, band = abs(value - baseline), abs(baseline) * tolerance
        if not math.isfinite(baseline) or (
                difference > band and
                not math.isclose(difference, band, rel_tol=1e-12, abs_tol=0)):
            raise ValueError(
                f"Reference metric band exceeded: {name}: {value} vs {baseline}"
            )


def _gpu():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits"
        ], text=True, timeout=3)
        return [line.rsplit(", ", 2) for line in output.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []


def _observe(result, scratch, stopped):
    import psutil
    process, disk_start = psutil.Process(), shutil.disk_usage(scratch).used
    while not stopped.is_set():
        values = result["resources"]
        try:
            values["ram_peak_bytes"] = max(
                values["ram_peak_bytes"],
                sum(member.memory_info().rss
                    for member in [process, *process.children(recursive=True)]))
            values["scratch_peak_delta_bytes"] = max(
                values["scratch_peak_delta_bytes"],
                shutil.disk_usage(scratch).used - disk_start)
            values["gpu_peak_sampled_mib"] = max(
                values["gpu_peak_sampled_mib"],
                sum(int(row[2]) for row in _gpu()))
        except (psutil.Error, OSError):
            pass
        stopped.wait(2)


def run(graph, query, split, task, overrides=None, *, args,
        prediction_kwargs=None, min_eval_class_rows=0):
    import psutil
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo
    result = {
        "case": args.case,
        "scale": args.scale,
        "source_sha": args.source_sha,
        "status": "failed",
        "resources": {
            "ram_peak_bytes": 0,
            "scratch_peak_delta_bytes": 0,
            "gpu_peak_sampled_mib": 0
        }
    }
    started, stopped, observer = time.monotonic(), threading.Event(), None
    temporary = None

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous_signal = signal.signal(signal.SIGTERM, interrupted)
    try:
        expected = None
        if not args.write_reference:
            expected = json.loads(args.reference.read_text())
            if (expected.get("status") != "passed"
                    or expected.get("case") != args.case
                    or expected.get("scale") != args.scale or not re.fullmatch(
                        r"[0-9a-f]{40}", expected.get("source_sha", ""))):
                raise ValueError(
                    "Reference must be a passed result for this case and scale")
        args.scratch.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.scratch).free < args.scratch_gib * 2**30:
            raise ValueError(
                f"Case requires {args.scratch_gib} GiB of free local SSD scratch"
            )
        temporary = tempfile.TemporaryDirectory(
            prefix=f"{args.case}-{args.scale}-", dir=args.scratch)
        args.scratch = Path(temporary.name)
        kumo.configure(scratch_dir=args.scratch)
        result["hardware"] = {
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
            "gpu": _gpu()
        }
        observer = threading.Thread(target=_observe,
                                    args=(result, args.scratch, stopped),
                                    daemon=True)
        observer.start()
        graph.validate()
        query = kumo.PredictiveQuery(graph, query)
        query.validate()
        snapshot = graph.snapshot()
        stats = json.loads((Path(snapshot.index_path) /
                            "ingest_stats.json").read_text())["stats"]
        result["graph"] = {
            key: stats[key]
            for key in ("node_counts", "edge_counts")
        }
        training_table = query.generate_training_table(snapshot, plan=split)
        result["training_table"] = table_counts(training_table)
        if args.scale in ("100M", "1B"):
            result["training_table"]["fingerprint"] = table_fingerprint(
                training_table, args.scratch)
        if expected is not None:
            compare_reference(result, expected, args.metric_tolerance)
        population = prediction_table(training_table, query, snapshot, args,
                                      split)
        plan = query.suggest_model_plan(snapshot)
        for key, value in {
                "training_job.num_experiments": 2,
                "training_job.pruning": None,
                "training_job.manual_seed": 0,
                "training_job.refit_trainval": False,
                "training_job.refit_full": False,
                "training_job.disable_explain": True,
                "optimization.max_epochs": 3,
                "optimization.min_steps_per_epoch": 200,
                "optimization.max_steps_per_epoch": 200,
                "optimization.max_val_steps": 100,
                "optimization.max_test_steps": 200,
                **(overrides or {})
        }.items():
            section, field = key.split(".", 1)
            setattr(getattr(plan, section), field, value)
        if min_eval_class_rows:
            batch_size = min(plan.optimization.batch_size)
            result["evaluation"] = {}
            for stage in ("val", "test"):
                field = f"max_{stage}_steps"
                steps = max(
                    getattr(plan.optimization, field),
                    evaluation_steps(result["training_table"]["splits"][stage],
                                     batch_size, min_eval_class_rows))
                setattr(plan.optimization, field, steps)
                result["evaluation"][stage] = {
                    "initial_batch_size": batch_size,
                    "max_steps": steps,
                    "min_class_rows": min_eval_class_rows
                }
        trainer = kumo.Trainer(plan, accelerator="gpu")
        fitted = trainer.fit(training_table, snapshot=snapshot)
        result["metrics"] = {
            key: float(value)
            for key, value in fitted.metrics.items()
        }
        if not result["metrics"] or not all(
                math.isfinite(value) for value in result["metrics"].values()):
            raise ValueError("Training metrics must be nonempty and finite")

        def predict(model, destination):
            return model.predict(
                population, snapshot=snapshot,
                **({
                    "binary_classification_threshold": .5
                } if task == "binary" else {}), **(prediction_kwargs or {}),
                output_config=kumo.PredictionOutputConfig(
                    connector=kumo.FileConnector(root_dir=args.scratch /
                                                 "predictions"),
                    output_paths={"predictions": destination},
                    extra_fields=[MetadataField.ANCHOR_TIMESTAMP]
                    if task != "static_link" else []))

        first, repeated = predict(trainer, "first"), predict(trainer, "repeat")
        del trainer
        restored = kumo.Trainer.load(fitted.bundle_dir, accelerator="gpu")
        result["predictions"] = check_predictions(first, repeated,
                                                  predict(restored, "reload"),
                                                  args.prediction_tolerance)
        if expected is not None:
            compare_reference(result, expected, args.metric_tolerance)
        result["status"] = "passed"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        stopped.set()
        if observer is not None:
            observer.join(timeout=4)
        result["total_seconds"] = time.monotonic() - started
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n")
        finally:
            if temporary is not None:
                temporary.cleanup()
    if args.write_reference:
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
