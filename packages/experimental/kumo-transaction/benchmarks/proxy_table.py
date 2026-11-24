# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print `proxy_sweep.py` summaries as one Markdown table per metric group.

    uv run python benchmarks/proxy_table.py outputs/proxy/stage1.json [more.json ...]

Columns are PR-AUC per task and readout, `xgboost_raw_embedding` beside the
cached `xgboost_raw` and `tabular_xgboost`, the held-out losses, same-user
retrieval on the default readout, and training stability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def rows(runs: list[dict[str, Any]], tasks: list[str], views: list[str]) -> list[str]:
    header = ["variant"]
    for task in tasks:
        header += [f"{task} {view}" for view in views] + [f"{task} raw+emb"]
    header += ["held-out loss", "retrieval@1 last", "final loss", "grad max", "train s"]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run in runs:
        proxy = run["proxy"]
        stats = run.get("train_stats") or {}
        cells = [f"`{run['name']}`"]
        for task in tasks:
            cells += [_fmt(proxy.get(f"{task}/{view}/pr_auc")) for view in views]
            cells.append(_fmt(proxy.get(f"{task}/xgboost_raw_embedding/pr_auc")))
        heldout = sum(v for k, v in proxy.items() if k.startswith("heldout/"))
        cells += [
            _fmt(heldout),
            _fmt(proxy.get("health/last/retrieval_top1")),
            _fmt(stats.get("final_loss")),
            _fmt(stats.get("grad_norm_max")),
            _fmt(run["seconds"]["train"]),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return out


def reference(runs: list[dict[str, Any]], tasks: list[str]) -> list[str]:
    proxy = runs[0]["proxy"]
    out = ["| task | xgboost_raw | tabular_xgboost |", "|---|---|---|"]
    for task in tasks:
        out.append(
            f"| {task} | {_fmt(proxy.get(f'{task}/xgboost_raw/pr_auc'))} "
            f"| {_fmt(proxy.get(f'{task}/tabular_xgboost/pr_auc'))} |"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--views", nargs="+", default=["last", "mean", "last_mean"])
    args = parser.parse_args()
    runs: list[dict[str, Any]] = []
    for path in args.summaries:
        runs += json.loads(path.read_text(encoding="utf-8"))["runs"]
    tasks = sorted(
        {key.split("/")[0] for run in runs for key in run["proxy"]} - {"heldout", "health"}
    )
    print("\n".join(rows(runs, tasks, args.views)))
    print()
    print("\n".join(reference(runs, tasks)))


if __name__ == "__main__":
    main()
