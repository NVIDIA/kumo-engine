# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate local synthetic passages with noisy readability targets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(output, seed=42):
    rng = np.random.default_rng(seed)
    n = 1800
    complexity = rng.uniform(0, 1, n)
    simple = "The small bird rests near the green tree. "
    complex_text = (
        "Ecological interdependence illustrates substantial environmental variability. "
    )
    passages = [
        ((simple * (10 - int(c * 8))) + (complex_text * (1 + int(c * 8))))
        for c in complexity
    ]
    frame = pd.DataFrame({
        "ID":
        np.arange(n),
        "Title": [f"Synthetic passage {i}" for i in range(n)],
        "Excerpt":
        passages,
        "Word Count": [len(t.split()) for t in passages],
        "BT Easiness": (1 - 3 * complexity + rng.normal(0, 0.2, n)).round(4),
    })
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "passages.parquet"
    frame.to_parquet(path, index=False, compression=None, use_dictionary=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate(args.output, args.seed))
