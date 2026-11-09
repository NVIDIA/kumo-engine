# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate small synthetic hospitality graphs without network or warehouse writes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ("restaurant", "recipes", "booking")


def tables(dataset, seed=42, rows=16000):
    rng = np.random.default_rng(seed)
    users = np.arange(200)
    restaurants = np.arange(80)
    user = rng.integers(200, size=rows)
    day = rng.integers(0, 730, size=rows)
    keep = (user % 4 != 0) | (day % 30 < 10)
    user, day = user[keep], day[keep]
    item = user % 4 * 20 + rng.integers(20, size=len(user))
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(day, unit="D")
    cuisines = np.array(["cuisine_a", "cuisine_b", "cuisine_c", "cuisine_d"])
    if dataset == "restaurant":
        return {
            "users":
            pd.DataFrame({
                "user_id": users,
                "age": 20 + users % 50,
                "gender": np.where(users % 2, "female", "male"),
                "preferred_cuisine": cuisines[users % 4],
            }),
            "restaurants":
            pd.DataFrame({
                "restaurant_id":
                restaurants,
                "name": [f"Synthetic restaurant {i}" for i in restaurants],
                "cuisine":
                cuisines[restaurants // 20],
                "rating":
                3 + restaurants % 3,
                "price_range":
                1 + restaurants % 3,
                "location": [f"district_{i // 20}" for i in restaurants],
            }),
            "interactions":
            pd.DataFrame({
                "user_id":
                user,
                "restaurant_id":
                item,
                "interaction_type":
                np.array(["visit", "review", "favorite", "reservation"
                          ])[(user + rng.integers(0, 2, len(user))) % 4],
                "timestamp":
                dates,
            }),
            "social_graph":
            pd.DataFrame({
                "user_a":
                np.tile(users, 4),
                "user_b":
                (np.tile(users, 4) + np.repeat([4, 8, 12, 16], 200)) % 200,
            }),
        }
    if dataset == "recipes":
        return {
            "customers":
            pd.DataFrame({
                "customer_id": users,
                "preference": users % 4
            }),
            "menus":
            pd.DataFrame({
                "menu_id": np.arange(4),
                "category": np.arange(4)
            }),
            "recipes":
            pd.DataFrame({
                "rid": restaurants,
                "category": restaurants // 20,
                "start_date": pd.Timestamp("2022-01-01"),
                "end_date": pd.Timestamp("2027-01-01"),
            }),
            "orders":
            pd.DataFrame({
                "customer_id": user,
                "menu_id": user % 4,
                "rid": item,
                "order_date": dates,
            }),
        }
    if dataset == "booking":
        return {
            "users_relational":
            pd.DataFrame({
                "hashed_user_id":
                users,
                "age_gender_bkt":
                users % 8,
                "timestamp_first_active":
                pd.date_range("2022-01-01", periods=len(users),
                              freq="h").strftime("%Y%m%d%H%M%S"),
                "age":
                20 + users % 50,
                "country_destination":
                np.array([
                    "destination_a",
                    "destination_b",
                    "destination_c",
                    "destination_d",
                ])[users % 4],
            }),
            "sessions_relational":
            pd.DataFrame({
                "hashed_user_id":
                user,
                "action": [f"browse_destination_{i % 4}" for i in user],
                "action_detail":
                np.where(user % 2, "view_listing", "search_results"),
                "secs_elapsed":
                30 + user % 100,
            }),
            "age_gender_bkts_relational":
            pd.DataFrame({
                "age_gender_bkt": np.arange(8),
                "age_band": np.arange(8) // 2,
                "gender": np.where(np.arange(8) % 2, "female", "male"),
            }),
        }
    raise ValueError(dataset)


def generate(dataset, output_dir, seed=42):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import io

    encoded = {}
    total = 0
    row_count = 12000
    for _ in range(3):
        encoded = {}
        for name, frame in tables(dataset, seed, row_count).items():
            if dataset in {"adventureworks", "matching", "restaurant"}:
                encoded[name + ".csv"] = frame.to_csv(
                    index=False, date_format="%Y-%m-%d %H:%M:%S").encode()
            else:
                stream = io.BytesIO()
                frame.to_parquet(stream, index=False, compression=None,
                                 use_dictionary=False)
                encoded[name + ".parquet"] = stream.getvalue()
        total = sum(map(len, encoded.values()))
        if 850000 <= total <= 1150000:
            break
        row_count = max(2000, int(row_count * 1000000 / total))
    for name, data in encoded.items():
        (output / name).write_bytes(data)
    print(f"{dataset}: {total:,} bytes in {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.dataset, args.output_dir, args.seed)
