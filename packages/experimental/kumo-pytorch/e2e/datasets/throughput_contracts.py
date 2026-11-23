# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-facing batch contracts for the three bounded DiskGraph workloads."""

from kumo.pytorch import api

HISTORY_EVENTS = 64
L01_AMOUNTS = ("amount_paid", "amount_received")
L01_CATEGORIES = ("payment_currency", "receiving_currency", "payment_format")
L01_ROLES = ("sender", "receiver")
MBD_CATEGORIES = (
    "currency",
    "event_type",
    "event_subtype",
    "src_type11",
    "src_type12",
    "dst_type11",
    "dst_type12",
    "src_type21",
    "src_type22",
    "src_type31",
    "src_type32",
)
_CALENDAR = {"hour": range(24), "weekday": range(7), "day": range(1, 32)}


def _graph(fields):
    return api.Input(
        api.NeighborhoodSource("history_graph"),
        api.Graph({"transactions": api.Fields(fields)}),
        None,
    )


def mbd(capacity):
    relation = api.RelationSpec(
        "client_transactions", "roots", "root_id", "transactions", "root_id"
    )
    schema = api.DataSchema(
        (
            api.TableSpec("roots", "root_id", time_column=api.NON_TEMPORAL),
            api.TableSpec("transactions", "event_id", time_column="event_time"),
        ),
        (relation,),
    )
    layer = api.GraphLayer(
        (api.GraphSample(relation.name, min(capacity, HISTORY_EVENTS), time_column="event_time"),)
    )
    sample = api.SampleSpec(
        api.EntitySpec("roots", entity_type="client"),
        neighborhoods={"history_graph": api.NeighborhoodSpec((layer,))},
        targets={"count": api.ExternalTarget(api.DType("float32"))},
    )
    graph = _graph(
        (
            api.Encode("amount", api.ZScore()),
            *(api.Encode(name, api.Vocabulary()) for name in MBD_CATEGORIES),
            api.Encode("event_time_seconds", api.Raw()),
        )
    )
    return api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec({"graph": graph}, {"count": api.LabelTarget("count", api.Raw())}),
    )


def _history_event(relation, fields, *, graph_clock=False):
    values = {name: api.Field(name) for name in fields}
    values["__event_time"] = api.Field("event_time")
    if graph_clock:
        # Graphs execute at T-1 second; history age is measured at prediction time T.
        values.update(
            __execution_age=api.EventTimeDelta("seconds"),
            __prediction_offset=api.Constant(1.0, api.DType("float64")),
            age_range=api.Derived("add", ("__execution_age", "__prediction_offset")),
        )
    else:
        values["age_range"] = api.EventTimeDelta("seconds")
    for name in _CALENDAR:
        values[name] = api.Derived(api.CalendarPart(name), ("__event_time",))
    rows = api.Rows((relation,), api.TimeWindow("event_time", stop=0 if graph_clock else -1))
    return api.EventSpec(rows, values, order="descending", limit=HISTORY_EVENTS)


def _history(events, encoders, layout, capacity):
    source = api.EventSource(
        tuple(events), max_events=HISTORY_EVENTS, keep="last", order="ascending"
    )
    encoders = {"age_range": api.Buckets(bins=64), **encoders}
    for name, domain in _CALENDAR.items():
        encoders[name] = api.Vocabulary(values=tuple((i, i + 2) for i in domain))
    fields = api.Fields(
        tuple(api.Encode(name, encoder, "history") for name, encoder in encoders.items()),
        row=api.Tokens(missing="error"),
        history=api.Flatten(
            (len(encoders) + 1) * capacity - 1, keep="first", separator="separator"
        ),
    )
    return api.Input(source, fields, layout)


_HISTORY_SPACE = {
    "history": api.IdSpace(
        {
            **{n: f"[{n.upper()}]" for n in ("entity", "event", "mask", "pad", "unknown")},
            "separator": "[SEP]",
        }
    )
}


def synthetic_medium(capacity):
    families = {
        "card_txns": ("amount", "mcc", "channel"),
        "transfers": ("amount", "transfer_type", "cross_border", "direction"),
        "logins": ("ip_region", "success"),
        "profile_changes": ("change_type",),
        "disputes": ("reason",),
        "lifelong_events": ("event_type",),
        "plan_events": ("plan",),
    }
    tables = [api.TableSpec("roots", "root_id", time_column="available_at")]
    relations, events, encoders = [], {}, {}
    for family, fields in families.items():
        tables.append(api.TableSpec(family, "event_id", time_column="event_time"))
        relation = api.RelationSpec(f"root_{family}", "roots", "root_id", family, "root_id")
        relations.append(relation)
        names = ("source_kind", *fields)
        events[family] = _history_event(relation.name, names)
        encoders.update(
            (n, api.Buckets(bins=64) if n == "amount" else api.Vocabulary()) for n in names
        )
    sample = api.SampleSpec(api.EntitySpec("roots", entity_type="customer"), events=events)
    batch = api.BatchSpec(
        {"history": _history(events, encoders, api.Padded(), capacity)},
        {"next_token": api.NextTokenTarget("history")},
        id_spaces=_HISTORY_SPACE,
        batching=api.Batching(max_samples=32),
    )
    return api.DataLoaderSpec(api.DataSchema(tuple(tables), tuple(relations)), sample, batch)


def aml_li_medium(capacity):
    tables = (
        api.TableSpec("accounts", "account_id", time_column="first_observed_time"),
        api.TableSpec("transactions", "transaction_id", time_column="event_time"),
        api.TableSpec("account_events", "event_id", time_column="event_time"),
    )
    relates = tuple(
        api.RelationSpec(role, "accounts", "account_id", "transactions", f"{role}_id")
        for role in L01_ROLES
    )
    touches = api.RelationSpec(
        "account_touches", "accounts", "account_id", "account_events", "account_id"
    )
    schema = api.DataSchema(tables, (*relates, touches))
    layers = (
        api.GraphLayer(tuple(api.GraphSample(role, HISTORY_EVENTS) for role in L01_ROLES)),
        api.GraphLayer(tuple(api.GraphSample(role, 1, direction="reverse") for role in L01_ROLES)),
    )
    buckets, category = api.Buckets(bins=64), api.Vocabulary()
    encoders = dict(
        direction=category,
        amount_paid=buckets,
        payment_currency=category,
        amount_received=buckets,
        receiving_currency=category,
        payment_format=category,
    )
    events = {"transactions": _history_event(touches.name, encoders, graph_clock=True)}
    counts = {
        f"{window}_{role}_count": api.Aggregate(
            api.Rows((path,), api.TimeWindow("event_time", start)), "count"
        )
        for window, start in (
            ("all_history", None),
            ("24h", -86400),
            ("7d", -7 * 86400),
            ("30d", -30 * 86400),
        )
        for role, path in (
            ("total_touching", touches.name),
            ("sender", "sender"),
            ("receiver", "receiver"),
        )
    }
    sample = api.SampleSpec(
        api.EntitySpec("accounts", entity_type="account"),
        neighborhoods={"history_graph": api.NeighborhoodSpec(layers)},
        targets={
            "amount": api.ExternalTarget(api.DType("float32")),
            "currency": api.ExternalTarget(api.DType("string")),
        },
        events=events,
        entity_values=counts,
    )
    graph = _graph(
        (
            *(api.Encode(n, api.ZScore()) for n in L01_AMOUNTS),
            *(api.Encode(n, category) for n in L01_CATEGORIES),
            api.Encode("event_time_seconds", api.Raw()),
        )
    )
    aggregates = api.Input(
        api.EntitySource(),
        api.Fields(tuple(api.Encode(n, api.ZScore()) for n in counts)),
        api.Separate(),
    )
    batch = api.BatchSpec(
        {
            "graph": graph,
            "history": _history(events, encoders, api.Packed(), capacity),
            "aggregates": aggregates,
        },
        {
            "amount": api.LabelTarget("amount", api.Raw()),
            "currency": api.LabelTarget("currency", category),
        },
        id_spaces=_HISTORY_SPACE,
        batching=api.Batching(max_samples=32),
    )
    return api.DataLoaderSpec(schema, sample, batch)
