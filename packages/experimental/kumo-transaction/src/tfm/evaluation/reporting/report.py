# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON / Markdown reports and model cards."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def metrics_markdown(metrics: dict[str, dict[str, float]]) -> str:
    """Render the probe table with per-row support and an under-support flag.

    Rows with fewer than `MIN_TRUSTWORTHY_POSITIVES` evaluation positives are
    marked, so a rendered report cannot pass an unsupported score off as a
    finding (ADR 0008).
    """
    from tfm.evaluation.fraud.metrics import MIN_TRUSTWORTHY_POSITIVES, is_supported

    lines = [
        "# Probe evaluation",
        "",
        "| Probe | ROC-AUC | PR-AUC | P@k | R@k | ECE | n eval | n pos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    thin = False
    for name, vals in metrics.items():
        supported = is_supported(vals)
        thin = thin or not supported
        flag = "" if supported else " ⚠"
        lines.append(
            f"| {name}{flag} | {vals['roc_auc']:.4f} | {vals['pr_auc']:.4f} | "
            f"{vals['precision_at_k']:.4f} | {vals['recall_at_k']:.4f} | "
            f"{vals['calibration']:.4f} | {int(vals.get('n_eval', 0))} | "
            f"{int(vals.get('n_eval_positive', 0))} |"
        )
    lines.append("")
    if thin:
        lines.append(
            f"⚠ Fewer than {MIN_TRUSTWORTHY_POSITIVES} evaluation positives. Those rows "
            "are diagnostics, not results: at this support a ranking metric is "
            "determined by one or two rows."
        )
        lines.append("")
    lines.append("This is a reference-architecture run, not a production fraud or credit model.")
    return "\n".join(lines) + "\n"


def headline_markdown(metrics: dict[str, dict[str, float]]) -> str:
    """Per task: raw-feature XGBoost, raw plus the embedding, and the lift between them.

    Empty when the run scored no XGBoost arms (the CPU smoke configs).
    """
    suffix = "xgboost_raw"
    rows = []
    for name in metrics:
        if not name.endswith(suffix):
            continue
        task = name[: -len(suffix)]
        raw, combined = metrics[name], metrics.get(f"{task}xgboost_raw_embedding")
        if combined is None:
            continue
        reference = metrics.get(f"{task}tabular_xgboost", {}).get("pr_auc")
        lift = combined["pr_auc"] / raw["pr_auc"] - 1 if raw["pr_auc"] > 0 else float("nan")
        rows.append(
            f"| {task.rstrip('_') or 'fraud'} | {raw['pr_auc']:.4f} | {combined['pr_auc']:.4f} | "
            f"{lift:+.1%} | {'-' if reference is None else f'{reference:.4f}'} | "
            f"{int(raw.get('n_eval_positive', 0))} |"
        )
    if not rows:
        return ""
    return "\n".join(
        [
            "## Headline",
            "",
            "PR-AUC of raw-feature XGBoost against raw features plus the frozen embedding,",
            "on the same rows; `tabular_xgboost` is hand-built history aggregates.",
            "",
            "| Task | xgboost_raw | xgboost_raw_embedding | Lift | tabular_xgboost | n pos |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )


def model_card_markdown(
    *,
    recipe: str,
    seed: int,
    architecture: str,
    data: str,
    metrics: dict[str, dict[str, float]],
    limitations: str,
) -> str:
    now = datetime.now(UTC).isoformat()
    return "\n".join(
        [
            f"# Model card — {recipe}",
            "",
            f"- Generated: {now}",
            f"- Seed: {seed}",
            f"- Architecture: {architecture}",
            f"- Data: {data}",
            "- Support tier: see docs/reference/compatibility-matrix.md",
            "",
            headline_markdown(metrics),
            "## Metrics",
            "",
            metrics_markdown(metrics),
            "## Limitations",
            "",
            limitations,
            "",
        ]
    )
