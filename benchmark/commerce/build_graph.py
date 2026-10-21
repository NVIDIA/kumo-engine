# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builds the kumo.engine Graph and PQL task for a generated rung.

    <kumo venv>/bin/python kumo/build_graph.py --data <root>/commerce/1B --graph commerce \
        --scratch <scratch dir> [--snapshot] [--training-table]

--data is the rung directory (local; one sub-directory of Parquet parts per table).
Without flags it only resolves schemas and validates the graph and the query.
"""
import argparse
import time

from kumoapi.graph import Edge
from kumoapi.typing import Stype

from kumo.engine import FileConnector, Graph, PredictiveQuery, Table, configure

DELIM = "|"

COMMERCE_QUERY = ("PREDICT LIST_DISTINCT(purchases.merchant_id, 0, 7, days) RANK TOP 10 "
                  "FOR EACH customers.customer_id")

def table(files, name, pk=None, time_col=None, stypes=None, multicat=()):
    t = Table.from_source_table(files[name], primary_key=pk, time_column=time_col)
    for col, stype in (stypes or {}).items():
        t = t.with_column(col, stype=stype)
    for col in multicat:
        t = t.with_column(col, stype=Stype.multicategorical, multicategorical_delimiter=DELIM)
    return t


def commerce_graph(files):
    cat, num, idt = Stype.categorical, Stype.numerical, Stype.ID
    tables = {
        "customers": table(files, "customers", "customer_id", "signup_time", {
            "region": cat, "city": cat, "membership": cat, "platform": cat, "marketing_opt_in": cat}),
        "merchants": table(files, "merchants", "merchant_id", "created_time", {
            "region": cat, "primary_category": cat, "price_tier": cat, "is_partner": cat,
            "avg_rating": num}, multicat=["tags"]),
        "purchases": table(files, "purchases", "purchase_id", "purchase_time", {
            "customer_id": idt, "merchant_id": idt, "channel": cat, "fulfillment": cat,
            "item_count": num, "subtotal": num, "fees": num, "tip": num,
            "is_first_purchase": cat, "rating": num}),
        "merchant_views": table(files, "merchant_views", None, "view_time", {
            "customer_id": idt, "merchant_id": idt, "source": cat}),
        "searches": table(files, "searches", None, "search_time", {
            "customer_id": idt, "query_term": cat, "category": cat, "result_page": num}),
        "favorites": table(files, "favorites", None, "created_time", {
            "customer_id": idt, "merchant_id": idt}),
        "reviews": table(files, "reviews", None, "review_time", {
            "customer_id": idt, "merchant_id": idt, "stars": num}),
    }
    edges = [Edge(src, "customer_id", "customers")
             for src in ["purchases", "merchant_views", "searches", "favorites", "reviews"]]
    edges += [Edge(src, "merchant_id", "merchants")
              for src in ["purchases", "merchant_views", "favorites", "reviews"]]
    return Graph(tables=tables, edges=edges), COMMERCE_QUERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--graph", required=True, choices=["commerce"])
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--training-table", action="store_true")
    args = ap.parse_args()
    configure(scratch_dir=args.scratch)
    files = FileConnector(root_dir=args.data)
    graph, query = commerce_graph(files)
    graph.validate()
    for name, t in graph._named_tables.items():
        print(name, [(c.name, str(c.stype), str(c.dtype)) for c in t.definition.cols])
    pq = PredictiveQuery(graph, query)
    pq.validate()
    print("graph and query valid; task type:", pq.get_task_type())
    if args.snapshot or args.training_table:
        t0 = time.time()
        snap = graph.snapshot()
        print(f"snapshot {snap.index_path} in {time.time() - t0:.0f}s")
        if args.training_table:
            t0 = time.time()
            tt = pq.generate_training_table(snap)
            print(f"training table in {time.time() - t0:.0f}s: {tt}")


if __name__ == "__main__":
    main()
