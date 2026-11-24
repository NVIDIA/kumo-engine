# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tfm.evaluation.reporting import headline_markdown, model_card_markdown


def _arm(pr_auc: float, positives: int = 40) -> dict[str, float]:
    return {
        "roc_auc": 0.7,
        "pr_auc": pr_auc,
        "precision_at_k": 0.1,
        "recall_at_k": 0.1,
        "calibration": 0.01,
        "n_eval": 1000,
        "n_eval_positive": positives,
    }


def test_headline_pairs_raw_with_raw_plus_embedding_per_task() -> None:
    metrics = {
        "xgboost_raw": _arm(0.20),
        "xgboost_raw_embedding": _arm(0.25),
        "tabular_xgboost": _arm(0.30),
        "churn_6m_xgboost_raw": _arm(0.40, positives=300),
        "churn_6m_xgboost_raw_embedding": _arm(0.30, positives=300),
        "history_window_xgboost_raw_embedding:single_event": _arm(0.9),
    }
    table = headline_markdown(metrics)
    rows = [line for line in table.splitlines() if line.startswith("| ") and "Task" not in line]
    assert rows == [
        "| fraud | 0.2000 | 0.2500 | +25.0% | 0.3000 | 40 |",
        "| churn_6m | 0.4000 | 0.3000 | -25.0% | - | 300 |",
    ]


def test_runs_without_xgboost_arms_have_no_headline() -> None:
    metrics = {"raw": _arm(0.2), "embedding": _arm(0.3), "combined": _arm(0.3)}
    assert headline_markdown(metrics) == ""
    card = model_card_markdown(
        recipe="r", seed=0, architecture="a", data="d", metrics=metrics, limitations="none"
    )
    assert "## Headline" not in card
    assert "## Metrics" in card
