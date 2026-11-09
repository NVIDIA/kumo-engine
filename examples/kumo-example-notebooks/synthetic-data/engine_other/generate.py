# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate small, seeded Engine datasets locally; never connects to a service.

Run with --dataset NAME --output-dir NEW_DIRECTORY. Requires numpy, pandas,
and pyarrow. Output Parquet tables retain the schemas used by the examples.
Synthetic evaluation metrics demonstrate a workflow, not real-world quality.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(dataset, seed=42):
    """Return related tables with noisy behavioral or seasonal signal."""
    rng = np.random.default_rng(seed)
    n = {
        "netflix": 72000,
        "movielens": 36000,
        "mlops": 30000
    }.get(dataset, 12000)
    if dataset in ("movielens", "netflix"):
        user = pd.DataFrame({
            "userId": np.arange(240),
            "preferred_genre": np.arange(240) % 5
        })
        movie = pd.DataFrame({
            "movieId": np.arange(100),
            "genre": np.arange(100) % 5,
            "quality": rng.uniform(2, 5, 100),
        })
        u = rng.integers(0, len(user), n)
        m = rng.integers(0, len(movie), n)
        rating = pd.DataFrame({
            "userId":
            u,
            "movieId":
            m,
            "timestamp":
            pd.Timestamp("2000-01-01") +
            pd.to_timedelta(rng.integers(0, 6 * 365 * 86400, n), unit="s"),
            "rating":
            np.clip(
                movie.quality.to_numpy()[m] + (u % 5 == m % 5) +
                rng.normal(0, 0.8, n),
                1,
                5,
            ).round(1),
        })
        if dataset == "netflix":
            return {
                "Users":
                user.rename(columns={"userId": "userID"}),
                "Movies":
                movie.rename(columns={
                    "movieId": "MovieID"
                }).assign(YearOfRelease=1995),
                "Ratings":
                rating.rename(columns={
                    "userId": "userID",
                    "movieId": "movieID",
                    "timestamp": "date",
                }),
            }
        pairs = (rating.rename(columns={
            "userId": "user_id",
            "movieId": "movie_id"
        }).sort_values("timestamp").reset_index(drop=True))
        history_end = len(pairs) * 2 // 3
        return {
            "user": user,
            "movie": movie,
            "rating": rating,
            "users": user.rename(columns={"userId": "user_id"}),
            "movies": movie.rename(columns={"movieId": "movie_id"}),
            "ratings": pairs.iloc[:history_end].copy(),
            "train_table": pairs.iloc[history_end:].copy(),
        }
    if dataset == "transformer":
        t = np.arange(n)
        load = 20 + 7 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, n)
        rows = pd.DataFrame({
            "entity":
            0,
            "date":
            pd.date_range("2020-01-01", periods=n, freq="h"),
            "OT":
            30 + 0.8 * load + rng.normal(0, 1, n),
        })
        for i, col in enumerate(
            ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")):
            rows[col] = load * (1 - i / 10) + rng.normal(0, 0.5, n)
        return {
            "entity_table": pd.DataFrame({"entity": [0]}),
            "target_table": rows
        }
    if dataset == "sunspots":
        t = np.arange(2400)
        return {
            "sunspots_entity_table":
            pd.DataFrame({"object": [0]}),
            "sunspots_target_table":
            pd.DataFrame({
                "object":
                0,
                "Date":
                pd.date_range("1800-01-01", periods=len(t), freq="MS"),
                "Sunspots":
                np.maximum(
                    0,
                    60 + 50 * np.sin(2 * np.pi * t / 132) +
                    rng.normal(0, 8, len(t)),
                ),
            }),
        }
    if dataset == "home_credit":
        a = np.arange(800)
        income = rng.lognormal(11, 0.5, len(a))
        debt = rng.uniform(0.05, 0.9, len(a))
        tables = {
            "application":
            pd.DataFrame({
                "SK_ID_CURR": a,
                "INCOME": income,
                "DEBT_RATIO": debt,
                "TARGET": rng.random(len(a)) < 0.05 + 0.6 * debt,
            })
        }
        for name, key in [
            ("bureau", "SK_ID_BUREAU"),
            ("previous_application", "SK_ID_PREV"),
        ]:
            tables[name] = pd.DataFrame({
                key:
                np.arange(1600),
                "SK_ID_CURR":
                np.arange(1600) % 800,
                "AMOUNT":
                income[np.arange(1600) % 800] * debt[np.arange(1600) % 800],
            })
        for name in (
                "bureau_balance",
                "POS_CASH_balance",
                "credit_card_balance",
                "installments_payments",
        ):
            ref = rng.integers(0, 1600, 12000)
            tables[name] = pd.DataFrame({
                "SK_ID_BUREAU" if name == "bureau_balance" else "SK_ID_PREV":
                ref,
                "SK_ID_CURR":
                ref % 800,
                "AMOUNT":
                income[ref % 800] * debt[ref % 800] / 12 +
                rng.normal(0, 20, len(ref)),
                "MONTH":
                rng.integers(-24, 0, len(ref)),
            })
        return tables
    if dataset == "financial":
        ids = np.arange(1000)
        income = rng.uniform(1500, 7000, len(ids))
        district = pd.DataFrame({"DISTRICT_ID": np.arange(20)})
        for c in ("A4", "A5", "A6", "A7", "A8", "A9", "A11", "A14", "A16"):
            district[c] = rng.uniform(1, 100, 20)
        good = rng.random(len(ids)) < (income / 8000)
        tables = {
            "DISTRICT":
            district,
            "ACCOUNT":
            pd.DataFrame({
                "ACCOUNT_ID": ids,
                "DISTRICT_ID": ids % 20,
                "INCOME": income
            }),
            "CLIENT":
            pd.DataFrame({
                "CLIENT_ID": ids,
                "DISTRICT_ID": ids % 20
            }),
            "DISP":
            pd.DataFrame({
                "DISP_ID": ids,
                "ACCOUNT_ID": ids,
                "CLIENT_ID": ids
            }),
            "CARD":
            pd.DataFrame({
                "CARD_ID": ids,
                "DISP_ID": ids
            }),
            "LOAN":
            pd.DataFrame({
                "LOAN_ID": ids,
                "ACCOUNT_ID": ids,
                "AMOUNT": rng.uniform(1000, 20000, len(ids)),
                "STATUS": np.where(good, "A", "B"),
            }),
        }
        for name, key in [("TRANS", "TRANS_ID"), ("ORDERS", "ORDER_ID")]:
            a = rng.integers(0, len(ids), 18000)
            tables[name] = pd.DataFrame({
                key:
                np.arange(len(a)),
                "ACCOUNT_ID":
                a,
                "AMOUNT":
                income[a] / 10 + rng.normal(0, 30, len(a)),
            })
        return tables
    if dataset == "stack":
        users = pd.DataFrame({
            "Id": np.arange(300),
            "topic": np.arange(300) % 10,
            "CreationDate": pd.Timestamp("2010-01-01"),
        })
        posts = pd.DataFrame({
            "Id": np.arange(600),
            "OwnerUserId": np.arange(600) % 300,
            "ParentId": np.arange(600),
            "AcceptedAnswerId": np.arange(600),
            "topic": np.arange(600) % 10,
            "CreationDate": pd.Timestamp("2011-01-01"),
        })
        tables = {"users": users, "posts": posts}
        for name in ("comments", "postHistory", "postLinks", "votes", "badges"):
            count = 18000 if name == "comments" else 4000
            u = rng.integers(0, 300, count)
            p = rng.integers(0, 60, count) * 10 + u % 10
            tables[name] = pd.DataFrame({
                "Id":
                np.arange(count),
                "UserId":
                u,
                "PostId":
                p,
                "RelatedPostId":
                p,
                "Date" if name == "badges" else "CreationDate":
                pd.Timestamp("2012-01-01") + pd.to_timedelta(
                    rng.integers(0, 10 * 365 * 86400, count), unit="s"),
            })
        return tables
    if dataset == "mlops":
        u = np.arange(300)
        loyalty = rng.uniform(0.1, 1, len(u))
        selected = rng.choice(u, n, p=loyalty / loyalty.sum())
        return {
            "users":
            pd.DataFrame({
                "USER_ID": u,
                "loyalty": loyalty
            }),
            "items":
            pd.DataFrame({
                "ITEM_ID": np.arange(100),
                "category": np.arange(100) % 5
            }),
            "orders":
            pd.DataFrame({
                "ORDER_ID":
                np.arange(n),
                "USER_ID":
                selected,
                "ITEM_ID":
                rng.integers(0, 100, n),
                "ORDER_TIME":
                pd.Timestamp("2024-01-01") +
                pd.to_timedelta(rng.integers(0, 730 * 86400, n), unit="s"),
                "AMOUNT":
                10 + 100 * loyalty[selected] + rng.normal(0, 2, n),
            }),
        }
    if dataset == "modules":
        i = np.arange(6000)
        return {
            "sensors":
            pd.DataFrame({
                "sensor_id":
                i,
                "reading":
                20 + np.sin(i / 24) + rng.normal(0, 0.1, len(i)),
            }),
            "users":
            pd.DataFrame({
                "user_id": np.arange(300),
                "loyalty": rng.uniform(0, 1, 300)
            }),
            "orders":
            pd.DataFrame({
                "order_id": i,
                "user_id": i % 300,
                "amount": 10 + i % 300 + rng.normal(0, 2, len(i)),
            }),
            "interests":
            pd.DataFrame({
                "interests": ["books|music", "sports|travel"] * 300,
                "product_ids": [[1, 2], [3, 4]] * 300,
            }),
            "prompts":
            pd.DataFrame({
                "prompt": [
                    f"Explain this synthetic score: account {j}, visits {j % 20}, predicted purchase probability {(0.1 + 0.04 * (j % 20)):.2f}."
                    for j in i
                ]
            }),
        }
    if dataset == "fraud":
        customers = pd.DataFrame({
            "cc_num": [f"synthetic-card-{i}" for i in range(300)],
            "age":
            rng.integers(18, 80, 300),
        })
        merchants = pd.DataFrame({
            "merchant": [f"synthetic-merchant-{i}" for i in range(50)],
            "risk":
            rng.uniform(0, 1, 50),
        })
        merchant_ids = rng.integers(0, 50, 8000)
        transactions = pd.DataFrame({
            "trans_num": [f"synthetic-transaction-{i}" for i in range(8000)],
            "cc_num":
            rng.choice(customers.cc_num, 8000),
            "merchant":
            merchants.merchant.to_numpy()[merchant_ids],
            "amt":
            rng.lognormal(4, 1, 8000),
            "trans_date_trans_time":
            pd.Timestamp("2019-01-01") +
            pd.to_timedelta(rng.integers(0, 600 * 86400, 8000), unit="s"),
        })
        flagged = (rng.random(8000)
                   < 0.02 + 0.4 * merchants.risk.to_numpy()[merchant_ids])
        reports = transactions.loc[
            flagged, ["trans_num", "trans_date_trans_time"]].rename(
                columns={"trans_date_trans_time": "report_time"})
        reports["report_time"] += pd.Timedelta(days=2)
        reports["is_real_fraud"] = 1
        population = transactions.tail(100)[[
            "trans_num", "trans_date_trans_time"
        ]].rename(columns={
            "trans_num": "ENTITY1",
            "trans_date_trans_time": "TIMESTAMP"
        })
        return {
            "customers": customers,
            "merchants": merchants,
            "transactions": transactions,
            "fraud_reports": reports,
            "prediction_population": population,
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "movielens",
            "netflix",
            "transformer",
            "sunspots",
            "home_credit",
            "financial",
            "stack",
            "mlops",
            "modules",
            "fraud",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in generate(args.dataset, args.seed).items():
        frame.to_parquet(args.output_dir / f"{name}.parquet", index=False,
                         compression=None)
    print(
        f"Wrote {sum(p.stat().st_size for p in args.output_dir.iterdir()):,} bytes to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
