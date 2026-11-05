# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small shared checks for the standalone Engine cases. No Engine import here."""

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse


def parse_args(case, scratch_gib):
    parser = argparse.ArgumentParser(description=f"Run the {case} Engine case")
    parser.add_argument("--scale", choices=tuple(scratch_gib), required=True)
    parser.add_argument("--requirements", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument(
        "--scratch", type=Path,
        default=Path(os.environ.get("DISKGRAPH_DATA_ROOT", "/mnt/localssd")))
    parser.add_argument("--source-sha")
    parser.add_argument("--prediction-rows", type=int)
    parser.add_argument("--metric-tolerance", type=float)
    parser.add_argument("--performance-tolerance", type=float)
    parser.add_argument("--prediction-tolerance", type=float)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--compatibility", action="store_true")
    parser.add_argument("--older-model", type=Path)
    parser.add_argument("--older-predictions", type=Path)
    args = parser.parse_args()
    args.case, args.scratch_gib = case, scratch_gib[args.scale]
    if args.requirements:
        print(json.dumps({"scratch_gib": args.scratch_gib}))
        parser.exit()
    return args


def _mix(values):
    """SplitMix64 finalizer, evaluated in native NumPy code."""
    import numpy as np
    with np.errstate(over="ignore"):
        values = (values ^ (values >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        values = (values ^ (values >> 27)) * np.uint64(0x94d049bb133111eb)
    return values ^ (values >> 31)


def _values_hash(array, seed, unordered=False):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy().astype("int64")
        child = array.values.slice(int(offsets[0]),
                                   int(offsets[-1] - offsets[0]))
        offsets = offsets - offsets[0]
        values = _values_hash(child, seed)
        if not unordered:
            positions = np.arange(len(child)) - np.repeat(
                offsets[:-1], np.diff(offsets))
            values = _mix(values ^ _mix(positions.astype("uint64") + seed))
        sums = np.concatenate(
            (np.zeros(1, dtype="uint64"), np.cumsum(values, dtype="uint64")))
        result = _mix(sums[offsets[1:]] - sums[offsets[:-1]]
                      ^ np.diff(offsets).astype("uint64"))
    else:
        if pa.types.is_nested(array.type):
            raise TypeError(
                f"Unsupported training-table hash type: {array.type}")
        if pa.types.is_null(array.type):
            scalar = array.to_numpy(zero_copy_only=False)
        else:
            empty = "" if pa.types.is_string(
                array.type) or pa.types.is_large_string(array.type) else (
                    b"" if pa.types.is_binary(array.type)
                    or pa.types.is_large_binary(array.type) else
                    False if pa.types.is_boolean(array.type) else 0)
            scalar = array.fill_null(pa.scalar(
                empty, type=array.type)).to_numpy(zero_copy_only=False)
        result = _mix(
            pd.util.hash_array(scalar, hash_key=f"{int(seed):016x}") ^ seed)
    return np.where(array.is_null().to_numpy(zero_copy_only=False),
                    _mix(np.full(len(array), seed ^ 0x1729, dtype="uint64")),
                    result)


def _row_hashes(table, lane=0, unordered_targets=False):
    """Versioned noncryptographic row fingerprint, including nested answers."""
    import numpy as np
    import pyarrow as pa
    seed = np.uint64((lane + 1) * 0x1f123bb5)
    result = np.full(len(table), seed, dtype="uint64")
    frequency = "__kumo_internal_frequency_column__"
    for name in sorted(table.column_names):
        if unordered_targets and name == frequency:
            continue
        array = table[name].combine_chunks()
        if unordered_targets and name == "TARGET1" and frequency in table.column_names:
            if not pa.types.is_list(array.type):
                raise TypeError("Frequency weights require list targets")
            weights = table[frequency].combine_chunks()
            if not array.offsets.equals(weights.offsets):
                raise ValueError(
                    "Targets and frequency weights have different lengths")
            paired = _mix(
                _values_hash(array.values, seed)
                ^ _mix(_values_hash(weights.values, seed + np.uint64(17))))
            offsets = array.offsets.to_numpy()
            sums = np.concatenate(
                (np.zeros(1, dtype="uint64"), np.cumsum(paired,
                                                        dtype="uint64")))
            target_null = array.is_null().to_numpy(
                zero_copy_only=False).astype("uint64")
            weight_null = weights.is_null().to_numpy(
                zero_copy_only=False).astype("uint64")
            values = _mix(sums[offsets[1:]] - sums[offsets[:-1]]
                          ^ _mix(np.diff(offsets).astype("uint64"))
                          ^ _mix(target_null + np.uint64(0x3569))
                          ^ _mix(weight_null + np.uint64(0x8127)))
        else:
            values = _values_hash(array, seed, unordered_targets
                                  and name == "TARGET1")
        column_seed = np.uint64(
            int.from_bytes(
                hashlib.sha256(name.encode()).digest()[:8], "little"))
        result = _mix(result ^ values ^ column_seed)
    return result


def _gpu():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits"
        ], text=True, timeout=3)
        return [
            line.strip().rsplit(", ", 2)
            for line in output.strip().splitlines()
        ]
    except (OSError, subprocess.SubprocessError):
        return []


class CaseRun:
    def __init__(self, args):
        self.args = args
        self.data = {
            "case": args.case,
            "scale": args.scale,
            "source_sha": args.source_sha,
            "status": "failed"
        }
        self._stop = threading.Event()
        self._sampler = None
        self._sample = None
        self._owned = None

    def start(self):
        args = self.args
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha or ""):
            raise ValueError(
                "--source-sha must identify the real Engine product commit")
        for key in ("input", "reference", "prediction_rows", "metric_tolerance",
                    "performance_tolerance", "prediction_tolerance"):
            if getattr(args, key) is None:
                raise ValueError(f"--{key.replace('_', '-')} is required")
        if args.prediction_rows < 1 or any(
                not math.isfinite(value) or value < 0
                for value in (args.metric_tolerance, args.performance_tolerance,
                              args.prediction_tolerance)):
            raise ValueError(
                "Prediction size must be positive and tolerances finite and nonnegative"
            )
        self.data["settings"] = {
            key: getattr(args, key)
            for key in ("prediction_rows", "metric_tolerance",
                        "performance_tolerance", "prediction_tolerance")
        }
        binding = json.loads(args.input.read_text())
        source = urlparse(binding["source"])
        if (source.scheme != "s3" or not source.netloc or source.query
                or source.fragment or binding["scale"] != args.scale
                or not binding["revision"]
                or not binding["immutable_prefix_policy"]
                or not binding["read_only_role"]):
            raise ValueError(
                "Input requires a reviewed immutable S3 prefix and a read-only test role"
            )
        objects = binding["objects"]
        if not objects or any(
                not item["version_id"] or item["version_id"] == "null"
                or not item["key"].startswith(source.path.strip("/") +
                                              "/") or item["size"] < 0
                for item in objects):
            raise ValueError(
                "Input requires an exact, versioned source-object inventory")
        args.source, args.region = binding["source"], binding["region"]
        self.data["input"] = {
            key: binding[key]
            for key in ("source", "revision", "immutable_prefix_policy",
                        "read_only_role")
        }
        self.data["input"]["manifest_sha256"] = hashlib.sha256(
            args.input.read_bytes()).hexdigest()
        args.scratch.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.scratch).free < args.scratch_gib * 2**30:
            raise ValueError(
                f"Case requires {args.scratch_gib} GiB of free local SSD scratch"
            )
        args.scratch = Path(
            tempfile.mkdtemp(prefix=f"{args.case}-{args.scale}-",
                             dir=args.scratch))
        self._owned = args.scratch
        self.started = time.monotonic()
        self._disk_used = shutil.disk_usage(args.scratch).used
        self.data["performance"] = {}
        self.data["resources"] = {
            "scratch_peak_delta_bytes": 0,
            "gpu_peak_sampled_mib": 0,
            "process_tree_ram_peak_sampled_bytes": 0
        }
        gpu = _gpu()
        self.data["hardware"] = {
            "cpu":
            next((line.split(":", 1)[1].strip()
                  for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "unknown"),
            "memory_kib":
            int(Path("/proc/meminfo").read_text().split()[1]),
            "gpu": [row[:2] for row in gpu]
        }
        self._sampler = threading.Thread(target=self._observe, daemon=True)
        self._sampler.start()

    def _observe(self):
        import psutil
        process = psutil.Process()
        while not self._stop.is_set():
            values = self.data["resources"]
            values["scratch_peak_delta_bytes"] = max(
                values["scratch_peak_delta_bytes"],
                shutil.disk_usage(self.args.scratch).used - self._disk_used)
            gpu = _gpu()
            values["gpu_peak_sampled_mib"] = max(
                values["gpu_peak_sampled_mib"], sum(int(row[2]) for row in gpu))
            try:
                resident = sum(
                    member.memory_info().rss
                    for member in [process, *process.children(recursive=True)])
                values["process_tree_ram_peak_sampled_bytes"] = max(
                    values["process_tree_ram_peak_sampled_bytes"], resident)
            except psutil.Error:
                pass
            self._stop.wait(2)

    @contextmanager
    def phase(self, name):
        start = time.monotonic()
        try:
            yield
        finally:
            self.data["performance"][f"{name}_seconds"] = time.monotonic(
            ) - start

    def record_graph(self, snapshot):
        stats = json.loads((Path(snapshot.index_path) /
                            "ingest_stats.json").read_text())["stats"]
        self.data["graph"] = {
            key: stats[key]
            for key in ("node_counts", "edge_counts")
        }

    def record_table(self, table, *, unordered_targets=False):
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        from kumo.engine import FileConnector, TrainingTableOutputConfig
        destination = self.args.scratch / "export"
        table.export(
            TrainingTableOutputConfig(FileConnector(root_dir=destination),
                                      "answers"))
        totals, count, schema = [0] * 4, 0, None
        for path in sorted(destination.rglob("*.parquet")):
            reader = pq.ParquetFile(path)
            here = sorted(
                (field.name, str(field.type)) for field in reader.schema_arrow)
            if schema is not None and schema != here:
                raise ValueError(
                    "Training-table parts have different logical schemas")
            schema = here
            for batch in reader.iter_batches(batch_size=65536):
                frame = pa.Table.from_batches([batch])
                for lane in range(4):
                    totals[lane] = (totals[lane] + int(
                        _row_hashes(frame, lane, unordered_targets).sum(
                            dtype="uint64"))) % 2**64
                count += len(frame)
                selected = frame.filter(pa.compute.equal(frame["SPLIT"], 2))
                if not len(selected):
                    continue
                keys = [
                    name for name in ("ENTITY1", "TIMESTAMP",
                                      "autoregression_labels")
                    if name in selected.column_names
                ]
                selected = selected.select(keys)
                if self._sample is not None:
                    selected = pa.concat_tables([self._sample, selected])
                hashes = _row_hashes(
                    selected.select([
                        key for key in keys if key != "autoregression_labels"
                    ]))
                order = np.argsort(hashes,
                                   kind="stable")[:self.args.prediction_rows]
                self._sample = selected.take(pa.array(order))
        if not count or self._sample is None:
            raise ValueError(
                "Training table and its held-out prediction population must be nonempty"
            )
        self.data["tools"] = {
            "pandas": pd.__version__,
            "pyarrow": pa.__version__
        }
        body = {
            "algorithm": "arrow-pandas-multiset-v1",
            "schema": schema,
            "rows": count,
            "sums": [int(total) for total in totals]
        }
        self.data["training_table"] = {
            **body, "fingerprint":
            hashlib.sha256(json.dumps(body,
                                      sort_keys=True).encode()).hexdigest()
        }
        self.data["prediction_rows"] = len(self._sample)

    def prediction_table(self, table, query, snapshot, *, forecast=False):
        import pyarrow.parquet as pq

        from kumo.engine import FileConnector, PredictionTable
        keys = [
            key for key in self._sample.column_names
            if forecast or key != "autoregression_labels"
        ]
        directory = self.args.scratch / "prediction-input"
        directory.mkdir()
        path = directory / "inputs.parquet"
        pq.write_table(self._sample.select(keys), path)
        # Public and internal generation plans share the same JSON schema.
        from kumoapi.json_serde import to_json_dict
        from kumoapi.model_plan import TrainingTableGenerationPlan
        plan = TrainingTableGenerationPlan(
            **to_json_dict(table.generation_plan))
        return PredictionTable.from_source_table(
            FileConnector(root_dir=directory)[path.name], pq=query,
            snapshot=snapshot, plan=plan)

    def record_metrics(self, metrics):
        values = {key: float(value) for key, value in metrics.items()}
        if not values or not all(
                math.isfinite(value) for value in values.values()):
            raise ValueError("Training metrics must be nonempty and finite")
        self.data["metrics"] = values

    def check_predictions(self, first, repeat, reloaded, *, task, top_k=None,
                          forecast_steps=12, time_column=None, classes=(),
                          population=None):
        import numpy as np
        import pandas as pd
        frames = [
            value.copy()
            if isinstance(value, pd.DataFrame) else value.frame.copy()
            for value in (first, repeat, reloaded)
        ]
        selected = self._sample if population is None else population
        expected = selected.to_pandas().rename(columns={
            "ENTITY1": "ENTITY",
            "TIMESTAMP": "ANCHOR_TIMESTAMP"
        })
        keys = [
            key for key in ("ENTITY", "ANCHOR_TIMESTAMP") if key in expected
        ]
        expected = expected[keys].drop_duplicates()
        if len(expected) != len(selected):
            raise ValueError("Prediction inputs repeat an entity and anchor")
        ranking = task in ("static_link", "temporal_link", "multilabel_ranking")
        score = "SCORE" if "SCORE" in frames[
            0] else "True_PROB" if task == "binary" else "TARGET_PRED"
        coordinate = "CLASS" if "CLASS" in frames[
            0] else time_column if task == "forecasting" else None
        columns = keys + ([coordinate] if coordinate else [])
        for frame in frames:
            if (frame.empty or any(key not in frame for key in columns)
                    or frame[columns].isna().any().any()
                    or frame.duplicated(columns).any() or
                    not np.isfinite(frame[score].to_numpy(dtype=float)).all()):
                raise ValueError(
                    "Prediction coordinates and scores must be complete, unique and finite"
                )
            populations = frame[keys].drop_duplicates()
            if not populations.merge(expected, how="outer",
                                     indicator=True)["_merge"].eq("both").all():
                raise ValueError(
                    "Prediction output differs from the capped input population"
                )
            counts = frame.groupby(keys, dropna=False).size()
            if task == "forecasting" and not counts.eq(forecast_steps).all():
                raise ValueError("Forecast output omits a selected horizon")
            if task == "forecasting":
                offsets = (pd.to_datetime(frame[time_column]) - pd.to_datetime(
                    frame["ANCHOR_TIMESTAMP"])) / pd.Timedelta(days=7)
                if not offsets.isin(range(forecast_steps)).all():
                    raise ValueError(
                        "Forecast output contains a foreign horizon")
            if ranking and (top_k is None or not counts.eq(top_k + 1).all()):
                raise ValueError(
                    "Ranking output requires the complete top-K plus boundary item"
                )
            if ranking:
                ranked = frame.assign(
                    RANK=frame["RANK"] if "RANK" in frame else frame.
                    groupby(keys)[score].rank(method="first", ascending=False) -
                    1).sort_values(keys + ["RANK"])
                ranks = ranked.groupby(keys)["RANK"].agg(["min", "max", "size"])
                if (ranked.duplicated(keys + ["RANK"]).any()
                        or not ranks["min"].isin([0, 1]).all()
                        or not (ranks["max"] - ranks["min"]).eq(ranks["size"] -
                                                                1).all() or
                    (ranked[score] - ranked.groupby(keys)[score].cummin()
                     > self.args.prediction_tolerance).any()):
                    raise ValueError(
                        "Prediction ranks contain duplicates or exceed a near tie"
                    )
            if task in ("binary", "regression") and not counts.eq(1).all():
                raise ValueError("Scalar output contains repeated predictions")
            if task == "binary":
                probabilities = frame[["False_PROB", "True_PROB"]].to_numpy()
                if (not ((probabilities >= 0) & (probabilities <= 1)).all()
                        or not np.allclose(probabilities.sum(axis=1), 1,
                                           atol=1e-6, rtol=0)
                        or not frame["TARGET_PRED"].eq(
                            frame["True_PROB"] > .5).all()):
                    raise ValueError(
                        "Binary probabilities or decisions are invalid")
            if task in ("multiclass", "multilabel", "multilabel_ranking"):
                if not classes or set(
                        frame["CLASS"]) != set(classes) or not counts.eq(
                            len(classes)).all():
                    raise ValueError("Class output omits a selected class")
                if not frame[score].between(0, 1).all():
                    raise ValueError(
                        "Class probabilities fall outside zero to one")
                if task != "multilabel" and not np.allclose(
                        frame.groupby(keys)[score].sum(), 1, atol=1e-6, rtol=0):
                    raise ValueError("Class probabilities do not sum to one")
                if task == "multilabel" and not frame["PREDICTED"].eq(
                        frame[score] > .5).all():
                    raise ValueError(
                        "Multilabel decisions differ from their thresholds")
                if task == "multiclass":
                    choices = frame[frame["PREDICTED"]]
                    maxima = frame.groupby(keys)[score].max().rename("maximum")
                    if (not choices.groupby(keys).size().reindex(
                            counts.index, fill_value=0).eq(1).all()
                            or not choices.merge(maxima, on=keys)[score].eq(
                                choices.merge(maxima,
                                              on=keys)["maximum"]).all()):
                        raise ValueError(
                            "Multiclass decisions do not select one highest score"
                        )
        base = frames[0].set_index(columns)[score]
        maximum = 0.0
        for frame in frames[1:]:
            current = frame.set_index(columns)[score]
            common = base.index.intersection(current.index)
            if not ranking and (len(common) != len(base)
                                or len(current) != len(base)):
                raise ValueError(
                    "Repeated prediction output omits a coordinate")
            if len(common):
                delta = np.abs(base.loc[common].to_numpy() -
                               current.loc[common].to_numpy())
                maximum = max(maximum, float(delta.max()))
                if (delta > self.args.prediction_tolerance).any():
                    raise ValueError(
                        "Repeated prediction scores exceed the chosen absolute tolerance"
                    )
            if task == "multiclass":
                choices = frames[0].loc[frames[0]["PREDICTED"],
                                        keys + ["CLASS"]].merge(
                                            frame.loc[frame["PREDICTED"],
                                                      keys + ["CLASS"]],
                                            on=keys,
                                            suffixes=("_before", "_after"))
                old_index, new_index = [
                    pd.MultiIndex.from_frame(
                        choices[keys + [f"CLASS_{side}"]].rename(
                            columns={f"CLASS_{side}": "CLASS"}))
                    for side in ("before", "after")
                ]
                for scores in (base, current):
                    if (np.abs(scores.loc[old_index].to_numpy() -
                               scores.loc[new_index].to_numpy())
                            > self.args.prediction_tolerance).any():
                        raise ValueError(
                            "A class decision changed outside a near tie")
            if ranking:
                ordered_before = frames[0].sort_values(
                    keys + [score])[score].to_numpy()
                ordered_after = frame.sort_values(keys +
                                                  [score])[score].to_numpy()
                rank_delta = np.abs(ordered_before - ordered_after)
                maximum = max(maximum, float(rank_delta.max()))
                if (rank_delta > self.args.prediction_tolerance).any():
                    raise ValueError(
                        "Repeated ranking score vectors exceed the chosen absolute tolerance"
                    )
                outputs = [
                    source[columns +
                           [score]].assign(_order=source["RANK"] if "RANK" in
                                           source else -source[score])
                    for source in (frames[0], frame)
                ]
                joined = outputs[0].merge(outputs[1], on=columns, how="outer",
                                          suffixes=("_before", "_after"),
                                          indicator=True)
                shared = joined[joined["_merge"].eq("both")]
                for order, values in (("before", "after"), ("after", "before")):
                    ordered = shared.sort_values(keys + [f"_order_{order}"])
                    scores = ordered[f"{score}_{values}"]
                    if (scores -
                            ordered.groupby(keys)[f"{score}_{values}"].cummin()
                            > self.args.prediction_tolerance).any():
                        raise ValueError("A ranking changed outside a near tie")
                for suffix, side, source in (("before", "left_only", frames[0]),
                                             ("after", "right_only", frame)):
                    missing = joined[joined["_merge"].eq(side)]
                    boundary = source.groupby(keys)[score].min().rename(
                        "boundary")
                    missing = missing.merge(boundary, on=keys)
                    if (missing[f"{score}_{suffix}"] - missing["boundary"]
                            > self.args.prediction_tolerance).any():
                        raise ValueError(
                            "A ranking change exceeds the top-K boundary tolerance"
                        )
        self.data["predictions"] = {
            "complete": True,
            "max_abs_difference": maximum
        }

    def finish(self):
        self.data["performance"]["total_seconds"] = time.monotonic(
        ) - self.started
        self.data["resources"]["ram_peak_kib"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
        required = ("graph", "training_table", "metrics", "predictions")
        if any(key not in self.data for key in required):
            raise ValueError(
                "Case omitted a required graph, table, metric or prediction check"
            )
        reference = self.args.reference
        if self.args.write_reference:
            self.data["status"] = "passed"
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_text(
                json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        else:
            expected = json.loads(reference.read_text())
            if (expected.get("status") != "passed"
                    or expected.get("case") != self.args.case
                    or expected.get("scale") != self.args.scale
                    or not re.fullmatch(r"[0-9a-f]{40}",
                                        expected.get("source_sha", ""))):
                raise ValueError(
                    "Reference must be a passed result for this case and scale with a product commit"
                )
            for key in ("input", "hardware", "graph", "training_table",
                        "prediction_rows", "settings"):
                if self.data[key] != expected[key]:
                    raise ValueError(f"Reference {key} differs")
            for group, tolerance in (("metrics", self.args.metric_tolerance),
                                     ("performance",
                                      self.args.performance_tolerance)):
                if self.data[group].keys() != expected[group].keys():
                    raise ValueError(f"Reference {group} names differ")
                for name, value in self.data[group].items():
                    baseline = expected[group][name]
                    if (not isinstance(baseline, (int, float))
                            or not math.isfinite(baseline)):
                        raise ValueError(
                            f"Reference {group}.{name} must be finite")
                    difference = value - baseline if group == "performance" else abs(
                        value - baseline)
                    if difference > abs(baseline) * tolerance:
                        raise ValueError(
                            f"Reference band exceeded: {group}.{name}: {value} vs {baseline}"
                        )
        self.data["status"] = "passed"


@contextmanager
def run_case(args):
    run = CaseRun(args)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run.start()
        yield run
        run.finish()
    except BaseException as error:
        run.data["status"] = "failed"
        run.data["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        run._stop.set()
        if run._sampler is not None:
            run._sampler.join(timeout=4)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(run.data, indent=2, sort_keys=True) + "\n")
        if run._owned is not None:
            shutil.rmtree(run._owned)
