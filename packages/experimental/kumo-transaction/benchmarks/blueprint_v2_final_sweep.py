# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the Blueprint v2 final sweep with the selection rule fixed before it ran.

`benchmarks/blueprint_v2_final_sweep.sh` writes `A.json` to `D.json` under its
run root; this reads them and writes `score.json` beside them. The scored
record is `benchmarks/results/blueprint_v2_final_sweep.md`.

    uv run python benchmarks/blueprint_v2_final_sweep.py outputs/final_sweep

Rule:
- Headline tasks: fraud, transaction-level fraud, and churn on PR-AUC; LTV on
  ROC-AUC. Lift is x / xgboost_raw - 1 on the same seed's rows.
- Arms: `lora` and `surprise` (raw + embedding + next-event surprise). One arm
  is used across all four tasks.
- An arm is eligible if its lift is positive at every seed on every headline task.
- B, C, or D replaces A if one of its eligible arms beats the same arm of A on
  median lift on at least three of the four tasks. If more than one candidate
  qualifies, the one beating A on more tasks wins, then the higher worst
  seed-and-task lift.
- The winner's arm is its eligible arm with the higher worst seed-and-task lift.
- Default in 12 months is reported, never scored.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

CANDIDATES = ("A", "B", "C", "D")
SEEDS = (42, 43, 44)
# task label -> (metric-key prefix, LoRA arm, metric)
TASKS: dict[str, tuple[str, str, str]] = {
    "fraud": ("", "lora_fraud", "pr_auc"),
    "fraud_txn": ("history_window_", "lora_history_window", "pr_auc"),
    "churn_6m": ("churn_6m_", "lora_churn_6m", "pr_auc"),
    "ltv_positive": ("ltv_positive_", "lora_ltv_positive", "roc_auc"),
}
FOOTNOTE = {"default_12m": ("default_12m_", "lora_default_12m", "pr_auc")}
ARMS = ("lora", "surprise")


def arm_key(task: tuple[str, str, str], arm: str) -> str:
    prefix, lora, _ = task
    if arm == "lora":
        return lora
    if arm == "surprise":
        return f"{prefix}xgboost_raw_embedding:surprise"
    if arm == "embedding":
        return f"{prefix}xgboost_raw_embedding"
    if arm == "tabular":
        return f"{prefix}tabular_xgboost"
    if arm == "raw":
        return f"{prefix}xgboost_raw"
    raise KeyError(arm)


def per_seed(record: dict[str, Any], tasks: dict[str, tuple[str, str, str]]) -> dict[str, Any]:
    """Every arm's metric and lift per task and seed."""
    runs = {run["seed"]: run["metrics"] for run in record["runs"]}
    missing = [s for s in SEEDS if s not in runs]
    if missing:
        raise SystemExit(f"Seeds {missing} missing from {record.get('config')}")
    out: dict[str, Any] = {}
    for label, task in tasks.items():
        metric = task[2]
        rows: dict[str, Any] = {}
        for seed in SEEDS:
            m = runs[seed]
            raw = m[arm_key(task, "raw")][metric]
            row: dict[str, Any] = {
                "raw": raw,
                "n_eval": m[arm_key(task, "raw")]["n_eval"],
                "n_eval_positive": m[arm_key(task, "raw")]["n_eval_positive"],
            }
            for arm in ("embedding", "surprise", "lora", "tabular"):
                block = m[arm_key(task, arm)]
                row[arm] = {
                    "pr_auc": block["pr_auc"],
                    "roc_auc": block["roc_auc"],
                    "value": block[metric],
                    "lift": block[metric] / raw - 1,
                }
            row["raw_pr_auc"] = m[arm_key(task, "raw")]["pr_auc"]
            row["raw_roc_auc"] = m[arm_key(task, "raw")]["roc_auc"]
            rows[str(seed)] = row
        out[label] = {"metric": metric, "seeds": rows}
    return out


def arm_summary(table: dict[str, Any], arm: str) -> dict[str, Any]:
    lifts = {task: [table[task]["seeds"][str(s)][arm]["lift"] for s in SEEDS] for task in TASKS}
    flat = [x for xs in lifts.values() for x in xs]
    return {
        "lifts": lifts,
        "median": {task: statistics.median(xs) for task, xs in lifts.items()},
        "eligible": all(x > 0 for x in flat),
        "worst": min(flat),
    }


def main(root: Path) -> None:
    present = [c for c in CANDIDATES if (root / f"{c}.json").exists()]
    if "A" not in present:
        raise SystemExit("Candidate A is required.")
    records = {c: json.loads((root / f"{c}.json").read_text(encoding="utf-8")) for c in present}
    tables = {c: per_seed(r, TASKS) for c, r in records.items()}
    footnotes = {c: per_seed(r, FOOTNOTE) for c, r in records.items()}
    arms = {c: {arm: arm_summary(tables[c], arm) for arm in ARMS} for c in present}

    challengers = []
    for c in present:
        if c == "A":
            continue
        for arm in ARMS:
            s = arms[c][arm]
            if not s["eligible"]:
                continue
            beaten = sum(s["median"][t] > arms["A"][arm]["median"][t] for t in TASKS)
            if beaten >= 3:
                challengers.append((beaten, s["worst"], c, arm))
    if challengers:
        challengers.sort(reverse=True)
        winner = challengers[0][2]
        reason = f"{winner} beats A on {challengers[0][0]} of 4 tasks with an eligible arm"
    else:
        winner = "A"
        reason = "no candidate beats A on three of four tasks with an eligible arm"

    eligible = [arm for arm in ARMS if arms[winner][arm]["eligible"]]
    pool = eligible or list(ARMS)
    arm = max(pool, key=lambda a: arms[winner][a]["worst"])

    result = {
        "candidates": present,
        "winner": winner,
        "reason": reason,
        "arm": arm,
        "arm_eligible": bool(eligible),
        "challengers": [
            {"beaten": b, "worst": w, "candidate": c, "arm": a} for b, w, c, a in challengers
        ],
        "arms": arms,
        "tables": tables,
        "footnotes": footnotes,
        "commits": {c: r.get("commit") for c, r in records.items()},
        "overrides": {c: r.get("overrides") for c, r in records.items()},
        "seconds": {c: [run["seconds"] for run in r["runs"]] for c, r in records.items()},
    }
    (root / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    pct = lambda x: f"{x:+.0%}"  # noqa: E731
    lines = [f"winner: {winner} ({reason}); arm: {arm}", ""]
    for c in present:
        for a in ARMS:
            s = arms[c][a]
            cells = ", ".join(
                f"{t} {pct(s['median'][t])} [{pct(min(s['lifts'][t]))}, {pct(max(s['lifts'][t]))}]"
                for t in TASKS
            )
            flag = "eligible" if s["eligible"] else "not eligible"
            lines.append(f"{c} {a:8s} {flag:12s} worst {pct(s['worst'])}: {cells}")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
