# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time the three joint OmniSage objectives and both DiskGraph providers on one CPU."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import tempfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import kumo.settings
import numpy as np
import torch

from kumo.pytorch import api
from tests.e2e.omnisage_paper_custom import custom_graphs
from tests.e2e.omnisage_paper_data import (
    make_paper_fixture,
    paper_batches,
    paper_graph_spec,
    paper_selection,
)
from tests.e2e.omnisage_paper_model import OmniSagePaperModel, PaperConfig, optimizer_for
from tests.e2e.s3_endpoint import local_s3


def _ms(start):
    return (perf_counter() - start) * 1000


def _counts(batch):
    counts = Counter(
        pair_examples=sum(len(pair[1]) for pair in batch.pairs.values()),
        feature_roots=sum(map(len, batch.features.values())),
        user_sequences=len(batch.history),
        next_pin_targets=int(((batch.history[:, 1:] >= 0) & (batch.history[:, :-1] >= 0)).sum()),
        future_pin_targets=int((batch.future >= 0).sum()),
        uniform_negative_roots=sum(map(len, batch.negatives.values())),
        graph_roots=sum(map(len, batch.root_ids.values())),
        graph_nodes=sum(len(n["ids"]) for g in batch.graphs.values() for n in g["nodes"].values()),
        graph_edges=sum(
            e["index"].shape[1] for g in batch.graphs.values() for e in g["edges"].values()
        ),
    )
    return counts


def _means(rows):
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0]}


def _batches(args, label, fixture, engine, schedule, seed):
    return paper_batches(
        fixture,
        engine=engine,
        num_threads=args.threads,
        seed=seed,
        requests=schedule,
        graph_loader_factory=custom_graphs if label == "custom" else None,
    )


def _graph_digest(batch):
    """Compare every hydrated graph tensor outside of the timed batch fetch."""
    digest = hashlib.sha256()

    def update(name, tensor):
        array = tensor.detach().contiguous().numpy()
        digest.update(repr((name, array.shape, array.dtype.str)).encode())
        digest.update(array.reshape(-1))

    for root, graph in sorted(batch.graphs.items()):
        for table, node in sorted(graph["nodes"].items()):
            for key in ("ids", "batch"):
                update((root, table, key), node[key])
            for name, feature in sorted(node["features"].items()):
                for key, values in sorted(feature.items()):
                    update((root, table, name, key), values)
        for bank, edges in sorted(graph["edges"].items()):
            for key in ("index", "score", "rank"):
                update((root, bank, key), edges[key])
    return digest.hexdigest()


def _run_loader_arm(args, label, fixture, engine, schedule, verify):
    records = []
    start = perf_counter()
    with _batches(args, label, fixture, engine, schedule, args.seed) as batches:
        setup_ms = _ms(start)
        for step in range(len(schedule)):
            start = perf_counter()
            batch = next(batches)
            elapsed = _ms(start)
            row = {
                "warmup": step < args.warmup_steps,
                "gather_ms": elapsed,
                "counts": _counts(batch),
            }
            if verify:
                row["graph_sha256"] = _graph_digest(batch)
            records.append(row)
        if next(batches, None) is not None:
            raise AssertionError("loader emitted an extra batch beyond the request schedule")
    measured = records[args.warmup_steps :]
    total = sum((row["counts"] for row in measured), Counter())
    gather_ms = sum(row["gather_ms"] for row in measured)
    return {
        "provider": label,
        "loader_setup_ms": setup_ms,
        "steps": records,
        "measured_counts": dict(total),
        "measured_gather_ms": gather_ms,
        "gather_ms_per_batch": gather_ms / len(measured),
        "roots_per_second": total["graph_roots"] * 1000 / gather_ms,
        "nodes_per_second": total["graph_nodes"] * 1000 / gather_ms,
        "edges_per_second": total["graph_edges"] * 1000 / gather_ms,
    }


def _loader_benchmark(args, fixture, engine, result):
    start = perf_counter()
    schedule = paper_selection(fixture, engine, args.seed)
    selection_ms = _ms(start)
    build_ms = result["shared_index_build_ms"]
    result["shared_selection_ms"] = selection_ms
    measured_schedule = schedule[args.warmup_steps :]
    cutoffs = {t for r in measured_schedule for ticks in r.times.values() for t in ticks.tolist()}
    result["measured_schedule"] = {
        "batches": len(measured_schedule),
        # Two logged pair sources, two graph pair sources and a user sequence source.
        "source_anchor_examples": 5 * fixture.batch_size * len(measured_schedule),
        "distinct_graph_cutoffs": len(cutoffs),
        "earliest_graph_cutoff": min(cutoffs),
        "latest_graph_cutoff": max(cutoffs),
    }
    for trial in range(args.repeats):
        arms = []
        for label in ("package", "custom") if trial % 2 == 0 else ("custom", "package"):
            arm = _run_loader_arm(args, label, fixture, engine, schedule, verify=trial == 0)
            gather = arm["gather_ms_per_batch"]
            arm["selection_plus_gather_ms_per_batch"] = gather + selection_ms / len(schedule)
            arm["cold_full_pass_ms_per_batch"] = (
                build_ms
                + selection_ms
                + arm["loader_setup_ms"]
                + sum(row["gather_ms"] for row in arm["steps"])
            ) / len(schedule)
            arms.append(arm)
        for step, (left, right) in enumerate(zip(arms[0]["steps"], arms[1]["steps"], strict=True)):
            if left["counts"] != right["counts"] or left.get("graph_sha256") != right.get(
                "graph_sha256"
            ):
                raise AssertionError(f"providers returned different hydrated graphs at step {step}")
        result["repeats"].append({"arms": arms})
    summary = {}
    for label in ("package", "custom"):
        arms = [arm for rep in result["repeats"] for arm in rep["arms"] if arm["provider"] == label]
        steps = [row["gather_ms"] for arm in arms for row in arm["steps"] if not row["warmup"]]
        summary[label] = {
            metric: statistics.median(arm[metric] for arm in arms)
            for metric in (
                "loader_setup_ms",
                "gather_ms_per_batch",
                "selection_plus_gather_ms_per_batch",
                "cold_full_pass_ms_per_batch",
                "roots_per_second",
                "nodes_per_second",
                "edges_per_second",
            )
        }
        summary[label]["pooled_batch_p50_ms"], summary[label]["pooled_batch_p95_ms"] = (
            np.percentile(steps, [50, 95]).tolist()
        )
    result["loader_summary"] = summary
    result["tensor_verified_paired_batches"] = len(schedule)
    result["package_over_custom_time_ratio"] = {
        metric: summary["package"][metric] / summary["custom"][metric]
        for metric in (
            "gather_ms_per_batch",
            "selection_plus_gather_ms_per_batch",
            "cold_full_pass_ms_per_batch",
        )
    }
    return result


def _run_arm(args, label, config, fixture, engine, schedule, seed):
    # Fresh parameters, optimizer, Count-Min sketches and LR schedule for each provider.
    torch.manual_seed(seed)
    wall = perf_counter()
    model = OmniSagePaperModel(config).train()
    epoch_steps = args.warmup_steps + args.steps
    optimizer, scheduler = optimizer_for(model, epoch_steps, args.warmup_steps)
    use_fp16 = args.precision == "fp16"
    scaler = torch.amp.GradScaler("cpu", enabled=use_fp16, init_scale=32.0)
    probe = model.entities["pins"].cls
    before = probe.detach().clone()
    model_setup = _ms(wall)

    records = []
    setup = perf_counter()
    with _batches(args, label, fixture, engine, schedule, seed) as batches:
        loader_setup = _ms(setup)
        for step in range(epoch_steps):
            start = perf_counter()
            batch = next(batches)
            gathered = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.float16, enabled=use_fp16):
                losses = model(batch)
            forwarded = perf_counter()
            scaler.scale(losses["total"]).backward()
            backward = perf_counter()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            end = perf_counter()
            timing = {
                "graph_gather_ms": (gathered - start) * 1000,
                "forward_ms": (forwarded - gathered) * 1000,
                "backward_ms": (backward - forwarded) * 1000,
                "adamw_ms": (end - backward) * 1000,
                "train_ms": (end - gathered) * 1000,
                "step_ms": (end - start) * 1000,
            }
            metrics = {name: loss.detach().item() for name, loss in losses.items()}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(f"{label} step {step} has a nonfinite loss: {metrics}")
            records.append(
                dict(
                    warmup=step < args.warmup_steps,
                    timing=timing,
                    losses=metrics,
                    counts=_counts(batch),
                )
            )
    wall_ms = _ms(wall)
    parameter_delta = (before - probe.detach()).abs().max().item()
    actual_updates = int(optimizer.state[probe]["step"].item())
    finite = all(parameter.isfinite().all().item() for parameter in model.parameters())
    if not finite or actual_updates != epoch_steps or parameter_delta <= 0:
        raise FloatingPointError(
            f"{label} completed {actual_updates}/{epoch_steps} updates, "
            f"parameter delta {parameter_delta}, finite={finite}"
        )
    measured = records[args.warmup_steps :]
    summed = Counter()
    for row in measured:
        summed.update(row["counts"])
    return {
        "provider": label,
        "model_setup_ms": model_setup,
        "loader_setup_ms": loader_setup,
        "arm_wall_ms": wall_ms,
        "steps": records,
        "measured_mean_ms": _means([r["timing"] for r in measured]),
        "measured_mean_loss": _means([r["losses"] for r in measured]),
        "measured_counts": dict(summed),
        "optimizer_updates": actual_updates,
        "finite_parameters": finite,
        "probe_max_abs_update": parameter_delta,
    }


def benchmark(args):
    if args.steps < 1 or args.warmup_steps < 1 or args.repeats < 1 or args.threads < 1:
        raise ValueError("steps, warmup steps, repeats and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = PaperConfig() if args.profile == "published" else PaperConfig.cpu()
    if not args.loader_only and args.history_length > config.max_history:
        raise ValueError(f"history length exceeds the configured maximum of {config.max_history}")
    result = {
        "mode": "graph_loader" if args.loader_only else "joint_training",
        "source": "https://arxiv.org/html/2504.17811v3",
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "output"
        },
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "repeats": [],
    }
    if not args.loader_only:
        result["model"] = asdict(config)
    with (
        tempfile.TemporaryDirectory() as tmp,
        local_s3() if args.storage == "s3" else nullcontext() as client,
    ):
        tmp = Path(tmp)
        kumo.settings.configure(scratch_dir=tmp / "index")
        start = perf_counter()
        options = dict(
            steps=args.steps + args.warmup_steps,
            batch_size=args.batch_size,
            history_length=args.history_length,
            random_negatives=args.random_negatives,
            seed=args.seed,
            client=client,
        )
        if args.events is None:
            fixture = make_paper_fixture(tmp / "source", pins=args.pins, **options)
        else:
            from tests.e2e.omnisage_event_fixture import make_event_fixture

            fixture, stats = make_event_fixture(
                tmp / "source",
                args.events,
                max_events=args.max_events,
                future_window_seconds=args.future_window_seconds,
                **options,
            )
            result["dataset"] = {"id": args.dataset_id or args.events.stem, **stats}
        result["fixture_write_ms"] = _ms(start)
        start = perf_counter()
        engine = api.DiskGraphEngine(num_threads=args.threads, random_seed=args.seed)
        with api.prepare(
            paper_graph_spec(fixture.schema, "pins", args.batch_size), fixture.dataset, engine
        ):
            pass
        build_ms = _ms(start)
        result["shared_index_build_ms"] = build_ms
        if args.loader_only:
            return _loader_benchmark(args, fixture, engine, result)
        for trial in range(args.repeats):
            seed = args.seed + trial
            start = perf_counter()
            requests = paper_selection(fixture, engine, seed)
            selection_ms = _ms(start)
            arms = []
            for label in ("package", "custom") if trial % 2 == 0 else ("custom", "package"):
                arm = _run_arm(args, label, config, fixture, engine, requests, seed)
                # Each arm is charged one index build and one selection pass as if run alone.
                # Use the whole actual epoch as denominator when amortizing cold setup.
                epoch_steps = len(arm["steps"])
                data_setup_ms = build_ms + selection_ms + arm["loader_setup_ms"]
                arm["cold_data_setup_ms_per_step"] = data_setup_ms / epoch_steps
                arm["cold_data_setup_plus_measured_step_ms"] = (
                    arm["measured_mean_ms"]["step_ms"] + data_setup_ms / epoch_steps
                )
                arm["cold_training_epoch_wall_ms"] = build_ms + selection_ms + arm["arm_wall_ms"]
                arm["cold_training_epoch_ms_per_step"] = (
                    arm["cold_training_epoch_wall_ms"] / epoch_steps
                )
                arms.append(arm)
            max_difference = 0.0
            for left, right in zip(arms[0]["steps"], arms[1]["steps"], strict=True):
                if left["counts"] != right["counts"]:
                    raise AssertionError("providers returned different work for the same schedule")
                for name, value in left["losses"].items():
                    other = right["losses"][name]
                    if not math.isclose(value, other, rel_tol=1e-4, abs_tol=1e-5):
                        raise AssertionError(f"provider losses differ for {name}: {value}, {other}")
                    max_difference = max(max_difference, abs(value - other))
            result["repeats"].append(
                {
                    "seed": seed,
                    "shared_selection_ms": selection_ms,
                    "max_absolute_loss_difference": max_difference,
                    "arms": arms,
                }
            )
    medians = {}
    for label in ("package", "custom"):
        arms = [a for rep in result["repeats"] for a in rep["arms"] if a["provider"] == label]
        rows = [
            {
                **arm["measured_mean_ms"],
                "cold_data_setup_plus_measured_step_ms": arm[
                    "cold_data_setup_plus_measured_step_ms"
                ],
                "cold_training_epoch_ms_per_step": arm["cold_training_epoch_ms_per_step"],
            }
            for arm in arms
        ]
        medians[label] = {key: statistics.median(row[key] for row in rows) for key in rows[0]}
    result["median_over_repeats_ms"] = medians
    result["package_over_custom_time_ratio"] = {
        k: v / medians["custom"][k] for k, v in medians["package"].items()
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--storage", choices=("local", "s3"), default="local")
    p.add_argument("--profile", choices=("published", "small"), default="published")
    p.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--pins", type=int, default=64)
    p.add_argument(
        "--events", type=Path, help="NPZ with integer actor, item and time (seconds) columns"
    )
    p.add_argument("--dataset-id", help="name used in the real dataset result")
    p.add_argument("--max-events", type=int, default=100_000)
    p.add_argument(
        "--future-window-seconds", type=int, help="override the event fixture's future window"
    )
    p.add_argument("--loader-only", action="store_true", help="time paired graph hydration passes")
    p.add_argument("--steps", type=int, default=2, help="measured batches per arm and repeat")
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="initial excluded batches and training LR warmup steps",
    )
    p.add_argument(
        "--batch-size", type=int, default=2, help="examples per task source in a joint update"
    )
    p.add_argument("--history-length", type=int, default=4)
    p.add_argument(
        "--random-negatives", type=int, default=4, help="uniform draws per target type in each step"
    )
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, help="write JSON to a file instead of stdout")
    args = p.parse_args()
    output = json.dumps(benchmark(args), indent=2, allow_nan=False)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
