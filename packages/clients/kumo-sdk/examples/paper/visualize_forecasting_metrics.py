# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot forecast overlay from --save output."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_overlay(
    run_dir: Path,
    dataset: str,
    entity: str,
    output_dir: Path | None = None,
) -> Path:
    """Overlay ground truth and per-model predictions for one entity."""
    pred_dir = run_dir / "predictions" / dataset
    history = pd.read_parquet(pred_dir / "history.parquet")
    model_paths = sorted(p for p in pred_dir.glob("*.parquet")
                         if p.name != "history.parquet")
    preds = pd.concat([pd.read_parquet(p) for p in model_paths],
                      ignore_index=True)

    ent = str(entity)
    history["time"] = pd.to_datetime(history["time"])
    preds["time"] = pd.to_datetime(preds["time"])
    hist = history[history["entity"].astype(str) == ent].sort_values("time")
    fig, ax = plt.subplots(figsize=(12, 5))
    split_styles = {
        "train": "#b0b0b0",
        "context": "#707070",
        "test": "#202020"
    }
    max_train = 100
    for split, color in split_styles.items():
        part = hist[hist["split"] == split]
        if not part.empty:
            if split == "train":
                part = part.tail(max_train)
            ax.plot(part["time"], part["y_true"], color=color, linewidth=0.8,
                    marker=".", markersize=2, label=f"y_true ({split})")
    context_start = hist.loc[hist["split"] == "context", "time"].min()
    if pd.notna(context_start):
        ax.axvline(context_start, color="#999", linestyle=":", linewidth=0.8)
    test_start = hist.loc[hist["split"] == "test", "time"].min()
    if pd.notna(test_start):
        ax.axvline(test_start, color="#555", linestyle=":", linewidth=0.8)

    test_truth = hist[hist["split"] == "test"][["time", "y_true"]]
    for model, mdf in preds[preds["entity"].astype(str) == ent].groupby(
            "model"):
        merged = mdf.merge(test_truth, on="time",
                           how="inner").sort_values("time")
        if not merged.empty:
            ax.plot(merged["time"], merged["prediction"], linewidth=0.8,
                    marker="o", markersize=2, label=model)

    ax.set_title(f"{dataset} / entity={entity}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_dir = output_dir or run_dir / "figures" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_{dataset}_{entity}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast overlay plot.")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/forecasting"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--entity", type=str, required=True)
    args = parser.parse_args()
    plot_forecast_overlay(args.run_dir, args.dataset, args.entity)


if __name__ == "__main__":
    main()
