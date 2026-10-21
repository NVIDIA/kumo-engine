# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared schemas: column types, keys, time columns, FKs and Kumo stypes.

Column tuple: (name, arrow type, nullable, stype). Stypes are Kumo's.
"""
import pyarrow as pa

I64, F64, STR, BOOL, TS = pa.int64(), pa.float64(), pa.string(), pa.bool_(), pa.timestamp("ms")


def _t(pk, time, fks, cols):
    return {"pk": pk, "time": time, "fks": fks, "cols": cols}


COMMERCE = {
    "customers": _t("customer_id", "signup_time", {}, [
        ("customer_id", I64, False, "ID"), ("signup_time", TS, False, "timestamp"),
        ("region", STR, False, "categorical"), ("city", STR, False, "categorical"),
        ("membership", STR, False, "categorical"), ("platform", STR, False, "categorical"),
        ("marketing_opt_in", BOOL, False, "categorical")]),
    "merchants": _t("merchant_id", "created_time", {}, [
        ("merchant_id", I64, False, "ID"), ("created_time", TS, False, "timestamp"),
        ("region", STR, False, "categorical"), ("primary_category", STR, False, "categorical"),
        ("tags", STR, True, "multicategorical"), ("price_tier", STR, False, "categorical"),
        ("is_partner", BOOL, False, "categorical"), ("avg_rating", F64, True, "numerical")]),
    "purchases": _t("purchase_id", "purchase_time",
                    {"customer_id": "customers", "merchant_id": "merchants"}, [
        ("purchase_id", I64, False, "ID"), ("purchase_time", TS, False, "timestamp"),
        ("customer_id", I64, False, "ID"), ("merchant_id", I64, False, "ID"),
        ("channel", STR, False, "categorical"), ("fulfillment", STR, False, "categorical"),
        ("item_count", I64, False, "numerical"), ("subtotal", F64, False, "numerical"),
        ("fees", F64, False, "numerical"), ("tip", F64, True, "numerical"),
        ("is_first_purchase", BOOL, False, "categorical"), ("rating", I64, True, "numerical")]),
    "merchant_views": _t(None, "view_time",
                         {"customer_id": "customers", "merchant_id": "merchants"}, [
        ("view_time", TS, False, "timestamp"), ("customer_id", I64, False, "ID"),
        ("merchant_id", I64, False, "ID"), ("source", STR, False, "categorical")]),
    "searches": _t(None, "search_time", {"customer_id": "customers"}, [
        ("search_time", TS, False, "timestamp"), ("customer_id", I64, False, "ID"),
        ("query_term", STR, False, "categorical"), ("category", STR, True, "categorical"),
        ("result_page", I64, False, "numerical")]),
    "favorites": _t(None, "created_time",
                    {"customer_id": "customers", "merchant_id": "merchants"}, [
        ("created_time", TS, False, "timestamp"), ("customer_id", I64, False, "ID"),
        ("merchant_id", I64, False, "ID")]),
    "reviews": _t(None, "review_time",
                  {"customer_id": "customers", "merchant_id": "merchants"}, [
        ("review_time", TS, False, "timestamp"), ("customer_id", I64, False, "ID"),
        ("merchant_id", I64, False, "ID"), ("stars", I64, False, "numerical")]),
}

GRAPHS = {"commerce": COMMERCE}

# inclusive value bounds; the generator clamps to them and the validator checks them
BOUNDS = {
    "commerce": {
        "merchants": {"avg_rating": (1.0, 5.0)},
        "purchases": {"item_count": (1, None), "subtotal": (0.01, None), "fees": (0.0, None),
                      "tip": (0.0, None), "rating": (1, 5)},
        "searches": {"result_page": (1, None)},
        "reviews": {"stars": (1, 5)},
    },
}

QUERIES = {
    "commerce": ("PREDICT LIST_DISTINCT(purchases.merchant_id, 0, 7, days) RANK TOP 10 "
                 "FOR EACH customers.customer_id"),
}


def arrow_schema(graph, table):
    return pa.schema([pa.field(n, t, nullable=nl) for n, t, nl, _ in GRAPHS[graph][table]["cols"]])


def nullable(graph, table):
    return {n for n, _, nl, _ in GRAPHS[graph][table]["cols"] if nl}


def kinds(graph, table):
    """Column name -> 'categorical' | 'multicategorical' | other, for format checks."""
    return {n: s for n, _, _, s in GRAPHS[graph][table]["cols"]}


def describe(graph):
    """JSON description of a graph's tables for the manifest."""
    return {"delimiter": "|", "timestamp_unit": "ms", "query": QUERIES[graph], "tables": {
        t: {"primary_key": s["pk"], "time_column": s["time"],
            "foreign_keys": [{"column": c, "references": f"{p}.{GRAPHS[graph][p]['pk']}"} for c, p in s["fks"].items()],
            "columns": [{"name": n, "dtype": str(ty), "nullable": nl, "stype": st} for n, ty, nl, st in s["cols"]]}
        for t, s in GRAPHS[graph].items()}}


def int_columns(graph, table):
    return {n for n, t, _, _ in GRAPHS[graph][table]["cols"] if t == I64}
