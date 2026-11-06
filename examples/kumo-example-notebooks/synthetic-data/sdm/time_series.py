# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write small local seasonal series with trend, weather effects and anomalies."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(output, seed=42):
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in range(16):
        t = np.arange(960)
        rows.append(
            pd.DataFrame({
                "item_id":
                item,
                "timestamp":
                pd.date_range("2025-01-01", periods=len(t), freq="h"),
                "target":
                (20 + item + t * 0.01 + 5 * np.sin(t * 2 * np.pi / 24) +
                 rng.normal(0, 0.5, len(t))).round(3),
            }))
    pd.concat(rows).to_csv(output / "hourly.csv", index=False)
    t = np.arange(744)
    load = 100 + 30 * np.sin(t * 2 * np.pi / 24) + rng.normal(0, 5, len(t))
    renewables = 25 + 20 * np.maximum(0, np.sin(t * 2 * np.pi / 24))
    energy = pd.DataFrame({
        "Ampirion Load Forecast":
        load,
        "PV+Wind Forecast":
        renewables,
        "target":
        0.7 * load - 0.5 * renewables + rng.normal(0, 2, len(t)),
    }).round(3)
    energy.iloc[:720].to_parquet(output / "energy_train.parquet", index=False)
    energy.iloc[720:].assign(target=np.nan).to_parquet(
        output / "energy_test.parquet", index=False)
    rows = []
    for store in range(16):
        t = np.arange(133)
        promo = rng.integers(0, 2, len(t))
        customers = 50 + store + 20 * promo + 10 * np.sin(t * 2 * np.pi / 52)
        rows.append(
            pd.DataFrame({
                "id":
                store,
                "timestamp":
                pd.date_range("2023-01-01", periods=len(t), freq="W"),
                "Open":
                1,
                "Promo":
                promo,
                "SchoolHoliday": (t % 52 > 40).astype(int),
                "StateHoliday": (t % 26 == 0).astype(int),
                "Customers":
                customers.round(2),
                "Sales": (customers * 8 + rng.normal(0, 5, len(t))).round(2),
                "step":
                t,
            }))
    retail = pd.concat(rows)
    retail.loc[retail.step < 120].drop(columns="step").to_parquet(
        output / "retail_train.parquet", index=False)
    retail.loc[retail.step >= 120].drop(columns=["step", "Customers"]).assign(
        Sales=np.nan).to_parquet(output / "retail_test.parquet", index=False)
    t = np.arange(6000)
    anomaly = pd.DataFrame({
        "step":
        t,
        "sensor_a":
        np.sin(t / 12) + rng.normal(0, 0.03, len(t)),
        "sensor_b":
        np.cos(t / 20) + rng.normal(0, 0.03, len(t)),
    })
    anomaly.loc[140:149, ["sensor_a", "sensor_b"]] += 3
    anomaly.round(4).to_csv(output / "anomaly.csv", index=False)
    weather = pd.DataFrame({
        "observed_weather":
        20 + 5 * np.sin(t / 365 * 2 * np.pi),
        "planned_price":
        1 + 0.1 * np.cos(t / 30),
    })
    weather["ice_cream"] = (30 + 2 * weather.observed_weather -
                            20 * weather.planned_price +
                            rng.normal(0, 1, len(t)))
    weather["cold_drinks"] = (10 + weather.observed_weather -
                              8 * weather.planned_price +
                              rng.normal(0, 0.5, len(t)))
    weather.round(3).to_csv(output / "weather.csv", index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate(args.output, args.seed))
