"""Generate deterministic synthetic retail graphs as local files (about 1 MB each)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = (
    "fashion",
    "customerltv",
    "coupon",
    "olist",
    "instacart",
    "m5",
    "matching",
    "stores",
    "tpch",
)


def tables(dataset, seed=42, rows=12000):
    rng = np.random.default_rng(seed)
    n, m = 200, 80
    u = np.arange(n)
    p = np.arange(m)
    customer = rng.integers(n, size=rows)
    day = rng.integers(0, 1095, size=rows)
    # Lower-activity customers stop purchasing early, providing churn negatives.
    keep = (customer % 4 != 0) | (day < 650)
    customer, day = customer[keep], day[keep]
    count = len(day)
    item = (customer % 4 * 20 + rng.integers(20, size=count)) % m
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(day, unit="D")
    amount = np.round(10 + item * 1.5 + rng.normal(0, 2, count), 2)
    ids = np.arange(count)
    if dataset == "fashion":
        transactions = pd.DataFrame(
            {
                "customer_id": customer,
                "article_id": item,
                "t_dat": dates,
                "price": amount,
                "sales_channel_id": 1 + customer % 2,
            }
        )
        return {
            "customers": pd.DataFrame(
                {"customer_id": u, "age": 20 + u % 50, "segment": u % 4}
            ),
            "articles": pd.DataFrame(
                {
                    "article_id": p,
                    "category": p // 20,
                    "detail_desc": [f"Synthetic garment {i}" for i in p],
                }
            ),
            "transactions_train": transactions,
            "transactions": transactions,
            "article_pairs": pd.DataFrame(
                [(a, b) for a in p for b in p if a != b and a // 20 == b // 20],
                columns=["article_1", "article_2"],
            ),
        }
    if dataset == "customerltv":
        return {
            "customer": pd.DataFrame({"CustomerID": u, "segment": u % 4}),
            "stock": pd.DataFrame({"StockCode": p, "category": p // 20}),
            "transaction": pd.DataFrame(
                {
                    "CustomerID": customer,
                    "StockCode": item,
                    "InvoiceDate": dates,
                    "Quantity": 1 + customer % 6,
                    "UnitPrice": amount,
                    "Description": [f"Synthetic stock {i}" for i in item],
                }
            ),
        }
    if dataset == "coupon":
        return {
            "user_list": pd.DataFrame({"USER_ID_hash": u, "segment": u % 4}),
            "coupon_list_train": pd.DataFrame(
                {
                    "COUPON_ID_hash": p,
                    "category": p // 20,
                    "DISPFROM": pd.Timestamp("2022-01-01"),
                    "DISPEND": pd.Timestamp("2027-01-01"),
                }
            ),
            "coupon_detail_train": pd.DataFrame(
                {
                    "USER_ID_hash": customer,
                    "COUPON_ID_hash": item,
                    "I_DATE": dates,
                    "amount": amount,
                }
            ),
            "coupon_visit_train": pd.DataFrame(
                {"USER_ID_hash": customer, "COUPON_ID_hash": item, "I_DATE": dates}
            ),
        }
    if dataset == "olist":
        return {
            "olist_customers_dataset": pd.DataFrame(
                {
                    "customer_id": u,
                    "region": u % 4,
                    "customer_city": [f"City {i % 4}" for i in u],
                }
            ),
            "olist_products_dataset": pd.DataFrame(
                {
                    "product_id": p,
                    "category": p // 20,
                    "product_category_name": [f"Category {i // 20}" for i in p],
                }
            ),
            "olist_item_orders_dataset_unique": pd.DataFrame(
                {
                    "order_id": ids,
                    "customer_id": customer,
                    "product_id": item,
                    "order_purchase_timestamp": dates,
                    "price": amount,
                }
            ),
            "olist_order_payments_dataset": pd.DataFrame(
                {"order_id": ids, "payment_value": amount}
            ),
            "olist_order_reviews_dataset": pd.DataFrame(
                {
                    "review_id": ids,
                    "order_id": ids,
                    "review_score": 1 + customer % 5,
                    "review_comment_message": [
                        f"Synthetic review {i % 5}" for i in customer
                    ],
                }
            ),
        }
    if dataset == "instacart":
        return {
            "user": pd.DataFrame({"user_id": u, "household_size": 1 + u % 6}),
            "order_with_count": pd.DataFrame(
                {
                    "order_id": ids,
                    "user_id": customer,
                    "timestamp": dates,
                    "item_count": 2 + customer % 6 + rng.poisson(3, count),
                }
            ),
        }
    if dataset == "m5":
        days = pd.date_range("2023-01-01", periods=1095)
        return {
            "sales_entities": pd.DataFrame({"id": p, "category": p // 20}),
            "sales_events": pd.DataFrame(
                {
                    "id": item,
                    "date": dates,
                    "variable": [f"d_{d}" for d in day],
                    "value": 2 + item % 10 + day % 7,
                }
            ),
            "sell_prices": pd.DataFrame(
                {"id": item, "date": dates, "sell_price": amount}
            ),
            "calendar": pd.DataFrame(
                {
                    "d": [f"d_{d}" for d in range(1095)],
                    "date": days,
                    "weekday": days.dayofweek,
                }
            ),
        }
    if dataset == "stores":
        return {
            "users": pd.DataFrame(
                {
                    "user_id": u,
                    "location_id": u % 4,
                    "created_time": pd.Timestamp("2022-01-01"),
                }
            ),
            "stores": pd.DataFrame(
                {
                    "store_id": p,
                    "created_time": pd.Timestamp("2022-01-01"),
                    "location_id": p // 20,
                }
            ),
            "orders": pd.DataFrame(
                {
                    "user_id": customer,
                    "location_id": customer % 4,
                    "store_id": item,
                    "order_time": dates,
                }
            ),
        }
    if dataset == "matching":
        a = pd.DataFrame(
            {
                "id": np.arange(1500),
                "title": [f"Synthetic item {i}" for i in range(1500)],
                "brand": [f"Brand {i % 20}" for i in range(1500)],
                "price": 10 + np.arange(1500) % 100,
            }
        )
        left = rng.integers(1500, size=rows)
        label = rng.integers(2, size=rows)
        right = np.where(label, left, (left + rng.integers(1, 1500, size=rows)) % 1500)
        pairs = pd.DataFrame({"ltable_id": left, "rtable_id": right, "label": label})
        return {
            "tableA": a,
            "tableB": a.copy(),
            "train": pairs.iloc[: rows * 3 // 5],
            "valid": pairs.iloc[rows * 3 // 5 : rows * 4 // 5],
            "test": pairs.iloc[rows * 4 // 5 :],
        }
    if dataset == "tpch":
        return {
            "customer": pd.DataFrame(
                {
                    "c_custkey": u,
                    "c_name": [f"Synthetic customer {i}" for i in u],
                    "c_nationkey": u % 5,
                    "c_mktsegment": u % 4,
                    "c_acctbal": 100 + u,
                }
            ),
            "orders": pd.DataFrame(
                {
                    "o_orderkey": ids,
                    "o_custkey": customer,
                    "o_orderdate": dates,
                    "o_totalprice": amount,
                    "o_shippriority": customer % 3,
                    "o_orderstatus": np.where(customer % 2, "F", "O"),
                    "o_orderpriority": np.where(customer % 2, "HIGH", "LOW"),
                }
            ),
            "lineitem": pd.DataFrame(
                {
                    "lineitem_id": ids,
                    "l_shipmode": "TRUCK",
                    "l_orderkey": ids,
                    "l_partkey": item,
                    "l_suppkey": item % 10,
                    "l_linenumber": 1,
                    "l_quantity": 1 + customer % 4,
                    "l_extendedprice": amount,
                    "l_discount": 0.05,
                    "l_tax": 0.02,
                    "l_shipdate": dates + pd.Timedelta(days=2),
                    "l_commitdate": dates + pd.Timedelta(days=1),
                    "l_receiptdate": dates + pd.Timedelta(days=3),
                    "l_returnflag": "N",
                    "l_linestatus": "O",
                }
            ),
            "part": pd.DataFrame(
                {
                    "p_partkey": p,
                    "p_name": [f"Synthetic part {i}" for i in p],
                    "p_brand": p % 5,
                    "p_size": 1 + p % 10,
                    "p_container": "BOX",
                    "p_retailprice": 20 + p * 2,
                    "p_type": p // 20,
                }
            ),
        }
    raise ValueError(dataset)


def generate(dataset, output_dir, seed=42):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import io

    row_count = 12000
    for _ in range(3):
        encoded = {}
        for name, frame in tables(dataset, seed, row_count).items():
            if dataset == "matching":
                encoded[name + ".csv"] = frame.to_csv(
                    index=False, date_format="%Y-%m-%d %H:%M:%S"
                ).encode()
            else:
                stream = io.BytesIO()
                frame.to_parquet(
                    stream, index=False, compression=None, use_dictionary=False
                )
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
