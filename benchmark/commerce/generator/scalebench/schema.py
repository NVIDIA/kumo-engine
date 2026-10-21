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

FRAUD = {
    "accounts": _t("account_id", "created_time", {}, [
        ("account_id", I64, False, "ID"), ("created_time", TS, False, "timestamp"),
        ("email_domain", STR, False, "categorical"), ("country", STR, False, "categorical"),
        ("account_type", STR, False, "categorical"), ("loyalty_tier", STR, False, "categorical"),
        ("is_verified", BOOL, False, "categorical")]),
    "devices": _t("device_id", "first_seen_time", {}, [
        ("device_id", I64, False, "ID"), ("first_seen_time", TS, False, "timestamp"),
        ("os", STR, False, "categorical"), ("browser", STR, False, "categorical"),
        ("device_type", STR, False, "categorical"), ("language", STR, False, "categorical"),
        ("timezone", STR, False, "categorical")]),
    "ip_addresses": _t("ip_id", None, {}, [
        ("ip_id", I64, False, "ID"), ("isp", STR, False, "categorical"),
        ("country", STR, False, "categorical"), ("region", STR, False, "categorical"),
        ("is_proxy", BOOL, False, "categorical"), ("is_datacenter", BOOL, False, "categorical")]),
    "payment_methods": _t("payment_method_id", None, {}, [
        ("payment_method_id", I64, False, "ID"), ("card_network", STR, False, "categorical"),
        ("card_type", STR, False, "categorical"), ("issuing_country", STR, False, "categorical"),
        ("payer_name", STR, False, "multicategorical")]),
    "merchants": _t("merchant_id", None, {}, [
        ("merchant_id", I64, False, "ID"), ("category", STR, False, "categorical"),
        ("country", STR, False, "categorical"), ("city", STR, False, "categorical"),
        ("channel", STR, False, "categorical")]),
    "transactions": _t("transaction_id", "transaction_time", {
        "account_id": "accounts", "device_id": "devices", "ip_id": "ip_addresses",
        "payment_method_id": "payment_methods", "merchant_id": "merchants"}, [
        ("transaction_id", I64, False, "ID"), ("transaction_time", TS, False, "timestamp"),
        ("account_id", I64, False, "ID"), ("device_id", I64, False, "ID"),
        ("ip_id", I64, False, "ID"), ("payment_method_id", I64, False, "ID"),
        ("merchant_id", I64, False, "ID"), ("is_fraud", I64, False, "numerical"),
        ("amount_usd", F64, False, "numerical"), ("quantity", I64, False, "numerical"),
        ("currency", STR, False, "categorical"), ("channel", STR, False, "categorical"),
        ("billing_country", STR, True, "categorical"),
        ("billing_postal_codes", STR, True, "multicategorical"),
        ("risk_rule_hits", STR, True, "multicategorical"),
        ("email_matches_payer", BOOL, False, "categorical"),
        ("installments", I64, False, "numerical"), ("traveler_name", STR, True, "categorical")]),
}

GRAPHS = {"commerce": COMMERCE, "fraud": FRAUD}

# inclusive value bounds; the generator clamps to them and the validator checks them
BOUNDS = {
    "commerce": {
        "merchants": {"avg_rating": (1.0, 5.0)},
        "purchases": {"item_count": (1, None), "subtotal": (0.01, None), "fees": (0.0, None),
                      "tip": (0.0, None), "rating": (1, 5)},
        "searches": {"result_page": (1, None)},
        "reviews": {"stars": (1, 5)},
    },
    "fraud": {
        "transactions": {"is_fraud": (0, 1), "amount_usd": (0.01, None), "quantity": (1, None),
                         "installments": (0, None)},
    },
}

QUERIES = {
    "commerce": ("PREDICT LIST_DISTINCT(purchases.merchant_id, 0, 7, days) RANK TOP 10 "
                 "FOR EACH customers.customer_id"),
    "fraud": ("PREDICT COUNT(transactions.* WHERE transactions.is_fraud = 1, 0, 1, days) > 0 "
              "FOR EACH accounts.account_id ASSUMING COUNT(transactions.*, 0, 1, days) > 0"),
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
