# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local synthetic tables used by the standalone SDM notebooks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def breast_mass(seed=42):
    rng = np.random.default_rng(seed)
    n = 20000
    frame = pd.DataFrame({
        "Age": rng.uniform(20, 90, n).round(3),
        "Shape": rng.integers(1, 5, n),
        "Margin": rng.integers(1, 6, n),
        "Density": rng.uniform(1, 4, n).round(4),
    })
    risk = (frame.Age - 55) / 20 + (frame.Shape - 2.5) * 0.7 + (frame.Margin -
                                                                3) * 0.5
    frame["Severity"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    return frame


def vehicle(seed=42):
    rng = np.random.default_rng(seed)
    n = 7000
    label = rng.integers(0, 4, n)
    centers = rng.normal(0, 2, (4, 18))
    frame = pd.DataFrame(
        (centers[label] + rng.normal(size=(n, 18))).round(4),
        columns=[f"silhouette_{i}" for i in range(18)],
    )
    frame["target"] = np.array(["bus", "sedan", "hatchback", "van"])[label]
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["breast_mass", "vehicle"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = {
        "breast_mass": breast_mass,
        "vehicle": vehicle
    }[args.dataset](args.seed)
    frame.to_csv(args.output / f"{args.dataset}.csv", index=False)
