# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered selections exercised by package-owned dataset tests."""

# The package test IDs also distinguish the bounded sources from the offline fixtures.
BINDINGS = {
    "E01": ("aml", "hi_small_account_history[e01-graph]", "aml-e01.json", "graph"),
    "E02": (
        "aml",
        "hi_small_account_history[e02-combined]",
        "aml-e01.json",
        "aggregates+graph+history",
    ),
    "E03": (
        "aml",
        "hi_small_account_history[e03-masked-values]",
        "aml-e01.json",
        "history+history_calendar+history_timing",
    ),
    "E04": ("aml", "hi_small_account_history[e04-next-token]", "aml-e01.json", "history"),
    "E05": ("aml", "hi_small_account_history[e05-involvement]", "aml-e01.json", "history"),
    "L01": ("aml", "li_medium_l01_combined_forecast", "aml-l01.json", "combined"),
    "S01": (
        "synthetic",
        "registered_history[s01-masked-values]",
        "synthetic-medium/account",
        "pragma",
    ),
    "S02": (
        "synthetic",
        "registered_history[s02-next-token]",
        "synthetic-medium/customer",
        "nuformer",
    ),
    "S04": (
        "synthetic",
        "registered_history[s04-positive-ltv]",
        "synthetic-medium/ltv",
        "combined",
    ),
    "M03": ("mbd", "mbd_transaction_forecast[m03-category]", "mbd-m03.json", "nuformer"),
    "M04": ("mbd", "mbd_transaction_forecast[m04-count]", "mbd-m04.json", "graphsage"),
}


def selections():
    rows = []
    for binding, (suite, test, selector, representation) in BINDINGS.items():
        hi = binding.startswith("E")
        nodeid = f"tests/e2e/test_{suite}.py::test_{test}"
        modes = ["correctness", "training", "lifecycle", "full"]
        if binding in ("L01", "E05"):
            modes.extend(("histories", "histories-filtered"))
        for mode in modes:
            full = mode == "full"
            history = mode.startswith("histories")
            group = "package"
            if hi and mode in ("correctness", "lifecycle", "histories", "histories-filtered"):
                kind = "source" if mode == "correctness" else "histories" if history else mode
                group = f"hi-small-{kind}"
            rows.append(
                dict(
                    binding=binding,
                    mode=mode,
                    selector=selector,
                    group=group,
                    representation="packed" if history else representation,
                    scope="full_source" if full else "bounded",
                    nodeid=f"tests/e2e/test_binding_population.py::test_full_binding[{binding}]"
                    if full
                    else nodeid,
                    **({"refusal": "DG-ROW-FILTER"} if mode == "histories-filtered" else {}),
                )
            )
    return rows
