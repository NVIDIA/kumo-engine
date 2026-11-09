# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record kumo-engine's baseline on the H&M-shaped dataset; developer-run, not CI.

Trains kumo-engine's in-process GNN on ``PREDICT COUNT(transactions.*, 0, 7, days) > 0
FOR EACH customers.customer_id`` with the generator's own anchors, labels and
splits (supplied as a custom training table), once per seed, then scores the
test anchors. It first checks that kumo-engine's own PQL labels equal the
generator's at every anchor. Run from this package's root directory in a venv with kumo-engine
installed (see ``e2e/README.md``)::

    python -m e2e.baselines.kumo_engine_hm \\
        --kumo-engine-repo /path/to/kumo-engine --work /tmp/hm-baseline

The recorded JSON is checked in once per kumo-engine pin, not recomputed by the end-to-end tests.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from kumo.engine import (
    FileConnector,
    Graph,
    GraphSnapshot,
    PredictionTable,
    PredictiveQuery,
    Table,
    Trainer,
    TrainingTable,
    TrainingTableOutputConfig,
    configure,
)
from kumoapi.graph import Edge
from kumoapi.json_serde import from_json
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.train import TrainingTableSpec

from e2e.datasets.hm import GENERATOR_VERSION, SCALES, HMDataset, generate, load
from e2e.metrics import auroc

QUERY = "PREDICT COUNT(transactions.*, 0, 7, days) > 0 FOR EACH customers.customer_id"
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
PLAN = Path(__file__).with_name("kumo_engine_hm_plan.json")
Json: TypeAlias = "None | bool | int | float | str | list[Json] | dict[str, Json]"


def model_plan(seed: int) -> ModelPlan:
    """The pinned single-trial plan; only the seed varies, so AutoML can't pick another trial."""
    plan = from_json(PLAN.read_text(), ModelPlan)
    plan.training_job.manual_seed = seed
    return plan


def build_graph(dataset: HMDataset) -> Graph:
    files = FileConnector(root_dir=str(dataset.tables_dir))
    return Graph(
        tables={
            "customers": Table.from_source_table(
                files["customers.parquet"], primary_key="customer_id"
            ).infer_metadata(),
            "articles": Table.from_source_table(
                files["articles.parquet"], primary_key="article_id", time_column="listed_at"
            ).infer_metadata(),
            "transactions": Table.from_source_table(
                files["transactions.parquet"], primary_key="transaction_id", time_column="t_dat"
            ).infer_metadata(),
        },
        edges=[
            Edge(src_table="transactions", fkey="customer_id", dst_table="customers"),
            Edge(src_table="transactions", fkey="article_id", dst_table="articles"),
        ],
    )


def anchor_frame(dataset: HMDataset) -> pd.DataFrame:
    anchors = pq.read_table(dataset.anchors).to_pandas()
    oracle = pq.read_table(dataset.labels).to_pandas()
    if not anchors["sample_id"].equals(oracle["sample_id"]):
        raise RuntimeError(f"{dataset.labels} is not aligned with {dataset.anchors}")
    return anchors.assign(label=oracle["label"], true_prob=oracle["true_prob"])


def write_rows(anchors: pd.DataFrame, rows_dir: Path) -> FileConnector:
    rows_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ENTITY1": anchors["customer_id"],
            "TIMESTAMP": anchors["cutoff"],
            "TARGET1": anchors["label"],
            "SPLIT": anchors["split"].map(SPLIT_CODES).astype("int64"),
        }
    ).to_parquet(rows_dir / "training_rows.parquet", index=False)
    # Predictions carry only the entity, so each cutoff is scored as its own table.
    for cutoff, test in anchors[anchors["split"] == "test"].groupby("cutoff"):
        pd.DataFrame({"ENTITY1": test["customer_id"], "TIMESTAMP": test["cutoff"]}).to_parquet(
            rows_dir / f"test_{cutoff:%Y%m%d}.parquet", index=False
        )
    return FileConnector(root_dir=str(rows_dir))


def pql_label_agreement(
    dataset: HMDataset,
    anchors: pd.DataFrame,
    query: PredictiveQuery,
    snapshot: GraphSnapshot,
    export_dir: Path,
) -> dict[str, int]:
    """Check that kumo-engine's own PQL labels equal the generator's at every anchor."""
    cutoffs = dataset.metadata.cutoffs
    end = (pd.Timestamp(cutoffs["test"][-1]) + pd.Timedelta(days=7)).date()
    ranges = (
        (cutoffs["train"][0][:10], cutoffs["val"][0][:10]),
        (cutoffs["val"][0][:10], cutoffs["test"][0][:10]),
        (cutoffs["test"][0][:10], str(end)),
    )
    split = "TimeRangeSplit([" + ", ".join(f'("{a}", "{b}")' for a, b in ranges) + "])"
    generated = query.generate_training_table(
        snapshot, plan=TrainingTableGenerationPlan(split=split, timeframe_step=7)
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    generated.export(TrainingTableOutputConfig(FileConnector(root_dir=str(export_dir)), "pql"))
    pql = pd.concat(
        pd.read_parquet(part) for part in sorted((export_dir / "pql").glob("*.parquet"))
    )
    pql = pql.assign(TIMESTAMP=pd.to_datetime(pql["TIMESTAMP"]).astype("datetime64[ns]"))
    matched = anchors.assign(cutoff=anchors["cutoff"].astype("datetime64[ns]")).merge(
        pql, left_on=["customer_id", "cutoff"], right_on=["ENTITY1", "TIMESTAMP"]
    )
    disagreements = int(
        (matched["TARGET1"].isna() | (matched["label"] != matched["TARGET1"].astype(bool))).sum()
    )
    if len(matched) != len(anchors) or disagreements:
        raise RuntimeError(
            f"PQL labels: {len(matched)} of {len(anchors)} anchors generated, "
            f"{disagreements} labels differ"
        )
    return {
        "anchors": len(anchors),
        "matched": len(matched),
        "disagreements": disagreements,
        "pql_rows_outside_anchors": len(pql) - len(matched),
    }


def distribution(scores: npt.NDArray[np.float64]) -> dict[str, Json]:
    return {
        "count": int(scores.size),
        "nan": int(np.isnan(scores).sum()),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "quantiles": {str(q): float(np.quantile(scores, q)) for q in QUANTILES},
        "predicted_positive_rate_at_0.5": float(np.mean(scores >= 0.5)),
    }


@dataclass(frozen=True)
class SeedRun:
    seed: int
    rerun: bool
    fit_seconds: float
    predict_seconds: float
    kumo_metrics: dict[str, float]
    test_auroc: float
    holdout_test_auroc: float
    point_lookups: list[dict[str, Json]]
    scores: npt.NDArray[np.float64]
    resolved_plan: dict[str, Json]
    epochs_run: int

    def to_json(self) -> dict[str, Json]:
        return {
            "seed": self.seed,
            "rerun": self.rerun,
            "fit_seconds": round(self.fit_seconds, 1),
            "predict_seconds": round(self.predict_seconds, 1),
            "kumo_metrics": dict(self.kumo_metrics),
            "test_auroc": self.test_auroc,
            "epochs_run": self.epochs_run,
            "holdout_test_auroc": self.holdout_test_auroc,
            "test_prediction_distribution": distribution(self.scores),
            "point_lookups": list(self.point_lookups),
        }


def run_seed(
    seed: int,
    rerun: bool,
    dataset: HMDataset,
    anchors: pd.DataFrame,
    query: PredictiveQuery,
    snapshot: GraphSnapshot,
    rows: FileConnector,
) -> SeedRun:
    training_table = TrainingTable.from_source_table(
        rows["training_rows.parquet"],
        TrainingTableSpec(split_col="SPLIT"),
        pq=query,
        snapshot=snapshot,
    )
    trainer = Trainer(model_plan=model_plan(seed))
    began = time.perf_counter()
    fit = trainer.fit(training_table, snapshot=snapshot)
    fit_seconds = time.perf_counter() - began

    test = anchors[anchors["split"] == "test"]
    scored_cutoffs = []
    began = time.perf_counter()
    for cutoff, rows_at in test.groupby("cutoff"):
        prediction_table = PredictionTable.from_source_table(
            rows[f"test_{cutoff:%Y%m%d}.parquet"],
            pq=query,
            snapshot=snapshot,
            plan=trainer.training_table_generation_plan,
        )
        predicted = trainer.predict(
            prediction_table, snapshot=snapshot, binary_classification_threshold=0.5
        ).frame
        scored_cutoffs.append(
            rows_at.merge(
                predicted[["ENTITY", "True_PROB"]].rename(columns={"ENTITY": "customer_id"}),
                on="customer_id",
                how="left",
                validate="one_to_one",
            )
        )
    predict_seconds = time.perf_counter() - began
    scored = pd.concat(scored_cutoffs, ignore_index=True)
    scores = scored["True_PROB"].to_numpy(dtype=np.float64)
    if len(scored) != len(test) or np.isnan(scores).any():
        raise RuntimeError(f"seed {seed}: not every test anchor was scored exactly once")
    holdout = fit.holdout_df()
    lookups: list[dict[str, Json]] = []
    for lookup in dataset.metadata.point_lookups:
        rows_for = scored[scored["customer_id"] == lookup.customer_id].sort_values("cutoff")
        lookups.append(
            {
                "customer_id": lookup.customer_id,
                "kind": lookup.kind,
                "expected": lookup.expected,
                "scores": [float(score) for score in rows_for["True_PROB"]],
            }
        )
    (resolved_plan,) = Path(fit.bundle_dir).parent.rglob("model_plan.json")
    (progress,) = Path(fit.bundle_dir).parent.rglob("train_progress.json")
    plan = json.loads(resolved_plan.read_text())
    plan["training_job"].pop("manual_seed")
    return SeedRun(
        seed=seed,
        rerun=rerun,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        kumo_metrics={name: float(value) for name, value in sorted(fit.metrics.items())},
        test_auroc=auroc(scored["label"].to_numpy(), scores),
        holdout_test_auroc=auroc(
            holdout["TARGET"].to_numpy(dtype=bool), holdout["True_PROB"].to_numpy(np.float64)
        ),
        point_lookups=lookups,
        scores=scores,
        resolved_plan=plan,
        epochs_run=int(json.loads(progress.read_text())["num_epochs_completed"]),
    )


def summary(values: list[float]) -> dict[str, Json]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def training_budget(
    runs: list[SeedRun], anchors: pd.DataFrame, gnn_runs: list[Path]
) -> dict[str, Json]:
    """Samples each side trains on: kumo-engine's capped steps, the GNN flow's full passes."""
    optimization = runs[0].resolved_plan["optimization"]
    assert isinstance(optimization, dict)
    batch_size = int(optimization["batch_size"][0])
    train_anchors = int((anchors["split"] == "train").sum())
    # kumo-core keeps the last partial batch and clamps to [min, max] steps per epoch.
    steps = min(
        max(-(-train_anchors // batch_size), int(optimization["min_steps_per_epoch"])),
        int(optimization["max_steps_per_epoch"]),
    )
    budget: dict[str, Json] = {
        "train_anchors": train_anchors,
        "kumo_engine": {
            "batch_size": batch_size,
            "steps_per_epoch": steps,
            "samples_per_epoch": steps * batch_size,
            "max_epochs": int(optimization["max_epochs"]),
            "epochs_run": [run.epochs_run for run in runs],
        },
    }
    if gnn_runs:
        trained = [json.loads((work / "train.json").read_text()) for work in gnn_runs]
        budget["gnn_flow"] = {
            "runs": [str(work.name) for work in gnn_runs],
            "samples_per_epoch": train_anchors,
            "epochs_run": [len(run["history"]) for run in trained],
            "best_epoch": [
                max(run["history"], key=lambda epoch: epoch["val_auroc"])["epoch"]
                for run in trained
            ],
        }
    return budget


def kumo_engine_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def hardware() -> dict[str, Json]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "accelerator": os.environ["KUMO_ACCELERATOR"],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kumo-engine-repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="scratch directory")
    parser.add_argument("--scale", choices=sorted(SCALES), default="S")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, help="default: hm-<scale>.json next to this script")
    parser.add_argument(
        "--gnn-runs",
        type=Path,
        nargs="*",
        default=[],
        help="work directories of GNN flow runs on the same dataset, for the budget",
    )
    args = parser.parse_args()
    out: Path = args.out or Path(__file__).with_name(f"hm-{args.scale.lower()}.json")
    # kumo-core reads its Lightning accelerator from here at fit time; the baseline is CPU-only.
    os.environ["KUMO_ACCELERATOR"] = "cpu"

    data_root = args.work / f"hm-{args.scale.lower()}-{args.data_seed}"
    dataset = (
        load(data_root)
        if (data_root / "metadata.json").exists()
        else generate(args.scale, args.data_seed, data_root)
    )
    found = (dataset.metadata.scale, dataset.metadata.seed, dataset.metadata.generator_version)
    if found != (args.scale, args.data_seed, GENERATOR_VERSION):
        raise RuntimeError(f"{data_root} holds a different dataset {found}; delete it")
    configure(scratch_dir=args.work / "kumo-scratch")
    anchors = anchor_frame(dataset)
    rows = write_rows(anchors, args.work / "rows")
    graph = build_graph(dataset)
    began = time.perf_counter()
    snapshot = graph.snapshot()
    snapshot_seconds = time.perf_counter() - began
    query = PredictiveQuery(graph=graph, query=QUERY)
    labels = pql_label_agreement(dataset, anchors, query, snapshot, args.work / "pql-export")

    runs = [run_seed(seed, False, dataset, anchors, query, snapshot, rows) for seed in args.seeds]
    if any(run.resolved_plan != runs[0].resolved_plan for run in runs):
        raise RuntimeError("seeds resolved to different model plans")
    runs.append(run_seed(args.seeds[0], True, dataset, anchors, query, snapshot, rows))
    meta = dataset.metadata
    record: dict[str, Json] = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "generator": "e2e.datasets.hm",
            "generator_version": meta.generator_version,
            "scale": meta.scale,
            "seed": meta.seed,
            "sizes": dict(meta.sizes),
            "table_sha256": dict(meta.table_sha256),
            "float_sums": dict(meta.float_sums),
            "ceiling_auroc": dict(meta.ceiling_auroc),
            "ceiling_auroc_se": dict(meta.ceiling_auroc_se),
            "positive_rate": dict(meta.positive_rate),
        },
        "task": {
            "query": QUERY,
            "anchors": "the generator's anchors.parquet with oracle/labels.parquet labels",
            "cutoffs": {split: list(values) for split, values in meta.cutoffs.items()},
            "point_lookups": [asdict(lookup) for lookup in meta.point_lookups],
            "pql_label_agreement": dict(labels),
        },
        "kumo_engine": {
            "commit": kumo_engine_commit(args.kumo_engine_repo),
            "pinned_plan": PLAN.name,
            "resolved_model_plan": runs[0].resolved_plan,
        },
        "hardware": hardware(),
        "snapshot_seconds": round(snapshot_seconds, 1),
        "training_budget": training_budget(runs, anchors, args.gnn_runs),
        "runs": [run.to_json() for run in runs],
        "same_seed_rerun": {
            "seed": args.seeds[0],
            "test_auroc_delta": runs[-1].test_auroc - runs[0].test_auroc,
            "max_abs_score_delta": float(np.abs(runs[-1].scores - runs[0].scores).max()),
            "max_abs_score_delta_second_vs_first_seed": float(
                np.abs(runs[1].scores - runs[0].scores).max()
            )
            if len(args.seeds) > 1
            else None,
        },
        "summary": {
            "runs": "every run above, the same-seed rerun included",
            "test_auroc": summary([run.test_auroc for run in runs]),
            "test_prediction_mean": summary([float(run.scores.mean()) for run in runs]),
            "fit_seconds": summary([run.fit_seconds for run in runs]),
        },
    }
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
