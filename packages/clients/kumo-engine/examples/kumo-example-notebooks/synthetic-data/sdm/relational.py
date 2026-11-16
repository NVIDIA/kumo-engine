# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small seeded sales graphs for relational SDM examples, written locally."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(output, seed=42):
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    customers = pd.DataFrame({
        "CUSTOMER": np.arange(300),
        "ADDRESSID": np.arange(300),
        "segment": rng.integers(0, 3, 300),
        "budget": rng.uniform(50, 500, 300).round(2),
        "__index_level_0__": np.arange(300),
    })
    addresses = pd.DataFrame({
        "ADDRESSID":
        np.arange(300),
        "region":
        rng.choice(["north", "south", "west"], 300),
        "__index_level_0__":
        np.arange(300),
    })
    n = 4500
    ids = rng.integers(0, 300, n)
    segment = customers.segment.to_numpy()[ids]
    dates = pd.date_range("2023-01-01", periods=n, freq="4h")
    sales = pd.DataFrame({
        "SALESDOCUMENT":
        np.arange(n),
        "CUSTOMER":
        ids,
        "CREATIONDATE":
        dates.strftime("%Y-%m-%d"),
        "CREATIONTIME":
        dates.strftime("%H:%M:%S"),
        "value":
        (customers.budget.to_numpy()[ids] * rng.uniform(0.5, 1.5, n)).round(2),
        "__index_level_0__":
        np.arange(n),
    })
    for column in [
            "SALESOFFICE",
            "SALESGROUP",
            "CUSTOMERPAYMENTTERMS",
            "SHIPPINGCONDITION",
            "INCOTERMSCLASSIFICATION",
    ]:
        # Region/segment creates predictive signal; independent noise avoids a lookup-only target.
        sales[column] = np.where(
            rng.random(n) < 0.85, segment, rng.integers(0, 3, n)).astype(str)
    items = pd.DataFrame({
        "SALESDOCUMENT": np.arange(n),
        "quantity": rng.integers(1, 10, n),
        "__index_level_0__": np.arange(n),
    })
    for column in ["SOLDTOPARTY", "SHIPTOPARTY", "BILLTOPARTY", "PAYERPARTY"]:
        items[column] = ids
    for column in ["PLANT", "SHIPPINGPOINT", "INCOTERMSCLASSIFICATION"]:
        items[column] = np.where(
            rng.random(n) < 0.85, segment, rng.integers(0, 3, n)).astype(str)
    frames = {"customers": customers, "addresses": addresses}
    for name, frame in [("salesdocuments", sales),
                        ("salesdocument_items", items)]:
        for split, subset in [
            ("train", frame.iloc[:3600]),
            ("test", frame.iloc[3600:]),
        ]:
            frames[f"{name}_{split}"] = subset
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    # Temporal entity tasks share this graph. Future task labels are never graph features.
    for split, start in [
        ("train", "2023-01-01"),
        ("val", "2024-01-01"),
        ("test", "2025-01-01"),
    ]:
        rows = []
        for timestamp in pd.date_range(start, periods=4, freq="MS"):
            score = (customers.budget.to_numpy() / 100 +
                     customers.segment.to_numpy() + rng.normal(0, 0.6, 300))
            rows.append(
                pd.DataFrame({
                    "CUSTOMER": np.arange(300),
                    "timestamp": timestamp,
                    "target": (score > 3.8).astype(int),
                }))
        pd.concat(rows, ignore_index=True).to_csv(output / f"task_{split}.csv",
                                                  index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate(args.output, args.seed))
