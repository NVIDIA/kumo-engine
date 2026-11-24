# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relative lift of every probe arm over a reference arm, per seed, as Markdown.

Lift is `x / x_ref - 1` (docs/concepts/evaluation-protocol.md), computed per
seed on the same rows, then summarized as median [min, max]. The reference is
`tabular_xgboost` by default, this kit's stand-in for a production model.

    uv run python benchmarks/lift_table.py \\
        benchmarks/results/blueprint_v2_seeds.json \\
        [--baselines benchmarks/results/...tabular_baselines.json] [--reference raw]

`--baselines` merges a `tabular_baselines.py` summary into the sweep's
per-seed metrics, matched by seed, for sweeps that ran without the tabular arms.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

_FORECASTS = ("churn_6m", "default_12m", "ltv_positive", "aml")
_TASKS = ("fraud", "history_window", "single_event", *_FORECASTS)


def _split(key: str) -> tuple[str, str]:
    """`(task, arm)` for a metrics key: `churn_6m_embedding`, `lora_fraud`, `raw`, ..."""
    if key.startswith("lora_"):
        return key[len("lora_") :], "lora"
    for task in _TASKS[1:]:
        if key.startswith(f"{task}_"):
            return task, key[len(task) + 1 :]
    return "fraud", key


def per_seed_metrics(sweep: dict[str, Any], baselines: dict[str, Any] | None) -> dict[int, dict]:
    runs = {int(run["seed"]): dict(run["metrics"]) for run in sweep["runs"]}
    if baselines is not None:
        for run in baselines["runs"]:
            seed = int(run["seed"])
            if seed not in runs:
                raise SystemExit(f"Seed {seed} of --baselines is not in the sweep.")
            runs[seed].update(run["metrics"])
    return runs


def lift_rows(
    runs: dict[int, dict], reference: str, metrics: tuple[str, ...]
) -> list[tuple[str, str, dict[str, list[float]]]]:
    by_task: dict[str, dict[str, dict[str, list[float]]]] = {}
    for seed_metrics in runs.values():
        arms: dict[str, dict[str, dict[str, float]]] = {}
        for key, values in seed_metrics.items():
            task, arm = _split(key)
            arms.setdefault(task, {})[arm] = values
        for task, task_arms in arms.items():
            ref = task_arms.get(reference)
            if ref is None:
                continue
            for arm, values in task_arms.items():
                if arm == reference:
                    continue
                for metric in metrics:
                    if ref.get(metric) and metric in values:
                        by_task.setdefault(task, {}).setdefault(arm, {}).setdefault(
                            metric, []
                        ).append(values[metric] / ref[metric] - 1.0)
    order = {task: i for i, task in enumerate(_TASKS)}
    return [
        (task, arm, lifts)
        for task in sorted(by_task, key=lambda t: order.get(t, len(order)))
        for arm, lifts in sorted(by_task[task].items())
    ]


def gain_recovered(runs: dict[int, dict], metric: str = "pr_auc") -> dict[str, list[float]]:
    """Per task, `(raw+emb - raw) / (tabular - raw)`: the share of the hand-built
    history features' gain that the embedding recovers, per seed.

    Skipped for a seed whose history features gain nothing over the raw arm.
    """
    out: dict[str, list[float]] = {}
    for seed_metrics in runs.values():
        arms: dict[str, dict[str, dict[str, float]]] = {}
        for key, values in seed_metrics.items():
            task, arm = _split(key)
            arms.setdefault(task, {})[arm] = values
        for task, a in arms.items():
            needed = ("xgboost_raw", "xgboost_raw_embedding", "tabular_xgboost")
            if not all(metric in a.get(arm, {}) for arm in needed):
                continue
            raw = a["xgboost_raw"][metric]
            gap = a["tabular_xgboost"][metric] - raw
            if gap > 0:
                out.setdefault(task, []).append((a["xgboost_raw_embedding"][metric] - raw) / gap)
    return out


def _cell(xs: list[float]) -> str:
    if not xs:
        return "–"
    med = statistics.median(xs)
    return f"{med:+.0%} [{min(xs):+.0%}, {max(xs):+.0%}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--baselines", type=Path, default=None)
    parser.add_argument("--reference", default="tabular_xgboost")
    args = parser.parse_args()
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    baselines = (
        None if args.baselines is None else json.loads(args.baselines.read_text(encoding="utf-8"))
    )
    runs = per_seed_metrics(sweep, baselines)
    metrics = ("pr_auc", "roc_auc")
    print(f"Lift over `{args.reference}`, median [min, max] over seeds {sorted(runs)}.\n")
    print("| Task | Arm | PR-AUC lift | ROC-AUC lift |")
    print("|---|---|---|---|")
    for task, arm, lifts in lift_rows(runs, args.reference, metrics):
        print(
            f"| `{task}` | `{arm}` | {_cell(lifts.get('pr_auc', []))} | {_cell(lifts.get('roc_auc', []))} |"
        )
    recovered = gain_recovered(runs)
    if recovered:
        print("\nShare of `tabular_xgboost`'s PR-AUC gain over `xgboost_raw` that")
        print("`xgboost_raw_embedding` recovers, median [min, max].\n")
        print("| Task | Gain recovered |")
        print("|---|---|")
        for task, xs in recovered.items():
            print(f"| `{task}` | {_cell(xs)} |")


if __name__ == "__main__":
    main()
