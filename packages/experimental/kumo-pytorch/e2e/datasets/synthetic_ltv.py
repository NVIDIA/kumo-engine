# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined graph, history and aggregate outputs for the positive-LTV workload."""

from dataclasses import replace

from kumo.pytorch import api


def _aggregates(events):
    windows = (
        ("all_history", None),
        ("24h", -86400),
        ("7d", -7 * 86400),
        ("30d", -30 * 86400),
        ("180d", -180 * 86400),
    )
    values = {
        f"{window}_{family}_count": api.Aggregate(
            api.Rows(event.rows.path, api.TimeWindow("event_time", start)), "count"
        )
        for window, start in windows
        for family, event in events.items()
    }
    cards = values["30d_card_txns_count"].source
    number, integer = api.DType("float64"), api.DType("int64")
    for op in ("sum", "mean", "min", "max", "std_population", "var_population"):
        values[f"card_amount_{op}"] = api.Aggregate(cards, op, "amount", dtype=number)
    for key in ("mcc", "channel"):
        values[f"card_distinct_{key}"] = api.Aggregate(cards, "count_distinct", key, dtype=integer)
    values.update(
        card_channel_mode=api.Aggregate(cards, "mode", "channel", dtype=api.DType("string")),
        card_recency_at_execution=api.Aggregate(cards, "recency", "event_time", dtype=number),
        prediction_offset=api.Constant(1.0, number),
        card_recency_seconds=api.Derived(
            "add", ("card_recency_at_execution", "prediction_offset"), dtype=number
        ),
        recent_card_share=api.Derived(
            "divide", ("24h_card_txns_count", "30d_card_txns_count"), dtype=number
        ),
    )
    groups = {
        keys: api.GroupBy(cards, keys, {"n": api.Reduce("count")})
        for keys in (("mcc",), ("channel",), ("mcc", "channel"))
    }
    for keys, group in groups.items():
        prefix = "card_" + "_".join(keys)
        values[prefix + "_groups"] = api.Aggregate(group, "count", dtype=integer)
        values[prefix + "_max_count"] = api.Aggregate(group, "max", "n", dtype=integer)
    per_mcc = api.GroupBy(groups["mcc", "channel"], ("mcc",), {"channels": api.Reduce("count")})
    values["card_mean_channels_per_mcc"] = api.Aggregate(per_mcc, "mean", "channels", dtype=number)
    return values


def ltv_contract(history, graph_fields, capacity):
    values = _aggregates(history.sample.events)
    features = {}
    for family, fields in graph_fields.items():
        numerical = (api.Encode("amount", api.ZScore()),) if "amount" in fields else ()
        features[family] = api.Fields(
            (
                *numerical,
                *(api.Encode(n, api.Vocabulary()) for n in fields if n != "amount"),
                api.Encode("event_time_seconds", api.Raw()),
            )
        )
    layer = api.GraphLayer(
        tuple(
            api.GraphSample(event.rows.path[0], capacity, time_column="event_time")
            for event in history.sample.events.values()
        )
    )
    sample = replace(
        history.sample,
        neighborhoods={"history_graph": api.NeighborhoodSpec((layer,))},
        entity_values=values,
        targets={"label": api.ExternalTarget(api.DType("float32"))},
    )
    checks = tuple(
        api.Encode(n, api.Vocabulary() if n == "card_channel_mode" else api.Raw()) for n in values
    )
    model = tuple(
        api.Encode(n, api.ZScore())
        for n in values
        if n not in {"card_channel_mode", "card_recency_at_execution", "prediction_offset"}
    )
    inputs = dict(
        history.batch.inputs,
        graph=api.Input(api.NeighborhoodSource("history_graph"), api.Graph(features), None),
        aggregates=api.Input(api.EntitySource(), api.Fields(model), api.Separate()),
        aggregate_checks=api.Input(api.EntitySource(), api.Fields(checks), api.Separate()),
    )
    return replace(
        history,
        sample=sample,
        batch=replace(
            history.batch, inputs=inputs, targets={"label": api.LabelTarget("label", api.Raw())}
        ),
    )
