# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare public and specialized DiskGraph loaders on a common temporal OmniSage workload."""

import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace
from unittest.mock import patch

import kumo.settings
import torch

from kumo.pytorch import api
from tests.e2e.omnisage_custom import CustomLoader
from tests.e2e.omnisage_fixture import CANDIDATE_NAMES, make_fixture, spec
from tests.e2e.omnisage_training import released_model, train_step
from tests.e2e.omnisage_verify import Oracle
from tests.e2e.s3_endpoint import local_s3


def ms_since(start):
    return (perf_counter_ns() - start) / 1e6


def peak_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def epoch(batches):
    counts = Counter()
    latencies = []
    start = previous = perf_counter_ns()
    for batch in batches():
        delivered = perf_counter_ns()
        latencies.append((delivered - previous) / 1e6)
        counts["anchors"] += len(batch.metadata["sample_ids"])
        for graph in batch.inputs.values():
            counts["nodes"] += sum(len(nodes["ids"]) for nodes in graph["nodes"].values())
            counts["scored_edges"] += sum(len(edge["score"]) for edge in graph["edges"].values())
        for name in CANDIDATE_NAMES:
            counts[name + "_candidates"] += int(batch.inputs[name]["candidate_offsets"][-1])
        # Keep a single yielded batch live, as in a synchronous training loop.
        previous = perf_counter_ns()
    return {"ms": ms_since(start), "batch_ms": latencies, "counts": dict(counts)}


def training(batches, model_file, steps):
    torch.manual_seed(913)
    model = released_model(model_file)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    # Warm up the model and optimizer before timing from the next iterator.
    train_step(next(iter(batches())), model, optimizer)
    probe = model.aggregator.output_proj.weight
    before = probe.detach().clone()
    observations = []
    start = perf_counter_ns()
    iterator = iter(batches())
    for _ in range(steps):
        batch = next(iterator)
        loss, pairs = train_step(batch, model, optimizer)
        observations.append(
            {"anchors": len(batch.metadata["sample_ids"]), "pairs": pairs, "loss": loss}
        )
    elapsed = ms_since(start)
    delta = float((probe.detach() - before).norm())
    if not math.isfinite(delta) or not delta > 0:
        raise AssertionError("OmniSage optimizer did not produce a finite parameter update")
    return {
        "ms": elapsed,
        "steps": observations,
        "parameter_delta_l2": delta,
        "anchors_per_s": 1000 * sum(step["anchors"] for step in observations) / elapsed,
    }


def worker(pipe, label, workload, index, root, batch_size, threads, seed, model_file):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        kumo.settings.configure(scratch_dir=root)
        with ExitStack() as stack:
            base_rss = peak_mib()
            start = perf_counter_ns()
            if label == "package":
                binding = SimpleNamespace(index_path=index, id="omnisage@benchmark")
                engine = api.DiskGraphEngine.from_snapshot(
                    binding, num_threads=threads, random_seed=seed
                )
                plan = stack.enter_context(api.prepare(spec(workload, batch_size), engine=engine))
                task = stack.enter_context(plan.materialize(workload.anchors))
                fail = AssertionError("model batches must use the native task path")
                stack.enter_context(patch.object(task._anchors, "_requests", side_effect=fail))
                stack.enter_context(patch.object(task._assembler, "batches", side_effect=fail))

                def batches():
                    return task.batches(output_format="model", shuffle=False)

                cache_bytes = None
            else:
                loader = stack.enter_context(
                    CustomLoader(
                        index,
                        workload.anchors.source,
                        batch_size,
                        width=workload.width,
                        seed=seed,
                        num_threads=threads,
                    )
                )
                batches, cache_bytes = loader.batches, loader.auxiliary_bytes
            pipe.send(
                {
                    "open_and_stage_ms": ms_since(start),
                    "startup_peak_rss_mib": base_rss,
                    "custom_cache_bytes": cache_bytes,
                }
            )
            while True:
                command, arg = pipe.recv()
                if command == "stop":
                    break
                if command == "epoch":
                    result = epoch(batches)
                    result["peak_rss_mib"] = peak_mib()
                elif command == "train":
                    result = training(batches, model_file, arg)
                elif command == "verify":
                    result = Oracle(workload).verify_epoch(batches())
                else:
                    raise ValueError(command)
                pipe.send(result)
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
    finally:
        pipe.close()


def receive(pipe):
    result = pipe.recv()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


@contextmanager
def serving(*args):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=worker, args=(child, *args))
    proc.start()
    child.close()
    try:
        yield parent, receive(parent)
    finally:
        if proc.is_alive():
            with suppress(BrokenPipeError, ConnectionResetError):
                parent.send(("stop", None))
        parent.close()
        proc.join(10)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def request(pipe, command, arg=None):
    pipe.send((command, arg))
    return receive(pipe)


def summary(samples):
    return {
        "median": statistics.median(samples),
        "p95": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
    }


def measure_pair(workload, index, scratch, size, args):
    labels = ("package", "custom")
    runs = {k: [] for k in labels}
    records = {}
    pipes = {}
    with ExitStack() as stack:
        for label in labels:
            pipes[label], records[label] = stack.enter_context(
                serving(
                    label, workload, index, scratch, size, args.threads, args.seed, args.model_file
                )
            )
            first = request(pipes[label], "epoch")
            records[label]["first_epoch_ms"] = first["ms"]
            records[label]["first_batch_ms"] = first["batch_ms"][0]
        for repeat in range(args.repeats):
            for label in labels if repeat % 2 == 0 else reversed(labels):
                runs[label].append(request(pipes[label], "epoch"))
        for label in labels:
            samples = runs[label]
            times = [r["ms"] for r in samples]
            counts = [r["counts"] for r in samples]
            assert all(c["anchors"] == args.anchors for c in counts)
            metrics = {
                f"{key}_per_s": statistics.median(
                    [1000 * c[key] / ms for c, ms in zip(counts, times, strict=True)]
                )
                for key in ("anchors", "nodes", "scored_edges")
            }
            records[label].update(
                metrics,
                epoch_ms=summary(times),
                epoch_ms_runs=times,
                batch_ms=summary([t for r in samples for t in r["batch_ms"]]),
                warm_batches=sum(len(r["batch_ms"]) for r in samples),
                epoch_counts=counts,
                loader_peak_rss_mib=max(r["peak_rss_mib"] for r in samples),
            )
        if args.model_file:
            n = min(args.train_steps, math.ceil(args.anchors / size))
            for repeat in range(args.train_repeats):
                for label in labels if repeat % 2 == 0 else reversed(labels):
                    records[label].setdefault("training_runs", []).append(
                        request(pipes[label], "train", n)
                    )
            for label in labels:
                records[label]["training_anchors_per_s"] = statistics.median(
                    r["anchors_per_s"] for r in records[label]["training_runs"]
                )
        for label in labels:
            records[label]["verified"] = request(pipes[label], "verify")
            verified = records[label]["verified"]
            assert verified["anchors"] == args.anchors
            assert all(verified[n + "_candidates"] for n in CANDIDATE_NAMES)
    return records


def measure(args):
    torch.set_num_threads(1)
    with (
        tempfile.TemporaryDirectory() as tmp,
        local_s3() if args.storage == "s3" else nullcontext() as client,
    ):
        root = Path(tmp)
        scratch = root / "scratch"
        kumo.settings.configure(scratch_dir=scratch)
        workload = make_fixture(
            root / "input",
            pins=args.pins,
            anchors=args.anchors,
            width=args.width,
            client=client,
            component_pins=args.component_pins,
        )
        engine = api.DiskGraphEngine(num_threads=args.threads, random_seed=args.seed)
        start = perf_counter_ns()
        with api.prepare(spec(workload, args.batch_sizes[0]), workload.dataset, engine):
            build_ms = ms_since(start)
        result = {
            "shared_base_build_ms": build_ms,
            "base_index_bytes": sum(
                f.stat().st_size for f in engine.index.rglob("*") if f.is_file()
            ),
            "parquet_bytes": {k: v.stat().st_size for k, v in workload.paths.items()},
            "batch_sizes": {
                size: measure_pair(workload, engine.index, scratch, size, args)
                for size in args.batch_sizes
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=("s3", "local"), default="s3")
    parser.add_argument("--pins", type=int, default=4096)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--component-pins", type=int, default=32)
    parser.add_argument("--width", type=int, default=1152)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model-file", type=Path, help="atg-research/omnisage/model.py")
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--train-repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            args.repeats,
            args.anchors,
            args.threads,
            args.train_steps,
            args.train_repeats,
            *args.batch_sizes,
        )
        < 1
    ):
        parser.error("sizes and repetition counts must be positive")
    if args.model_file and args.width != 1152:
        parser.error("training requires --width 1152")
    results = measure(args)
    results["settings"] = {
        **vars(args),
        "model_file": str(args.model_file) if args.model_file else None,
    }
    results["environment"] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    results["measurement"] = (
        "Independent CPU processes reuse the same immutable native DiskGraph index. "
        "One full first epoch warms each loader. Subsequent epochs alternate order. "
        "Peak RSS is process high-water before model training and source-truth checks. "
        "Model timing includes loading, tensor adaptation, forward, backward and SGD. "
        "Setup excludes synthetic source generation and S3 upload."
    )
    if args.model_file:
        revision = subprocess.run(
            ["git", "-C", str(args.model_file.resolve().parent), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
        )
        if revision.returncode == 0:
            results["model_revision"] = revision.stdout.strip()
        results["model_sha256"] = hashlib.sha256(args.model_file.read_bytes()).hexdigest()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
