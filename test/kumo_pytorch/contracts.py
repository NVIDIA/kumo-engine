# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML prototype input declarations adapted from Arena typed_contracts.

Shared by real-data runs and tiny model checks. Full-history capacities must
come from the prepared source, not arbitrary truncation limits.
"""

from dataclasses import replace
from test.kumo_pytorch.models.aml import AMOUNTS as GRAPH_AMOUNT_FIELDS
from test.kumo_pytorch.models.aml import CATEGORIES as GRAPH_CATEGORY_FIELDS

import kumo.pytorch.api as r

GRAPH_TIMESTAMP_FIELD = "event_time_seconds"


def build_contract(schema, family, *, native=False, max_events):
    """Declare full selected-root histories; v2 graph requests execute at T-1s."""
    if family not in {"graphsage", "combined", "pragma", "nuformer"}:
        raise ValueError("Unknown AML model family")
    if type(max_events) is not int or max_events < 1:
        raise ValueError("max_events must bound the complete selected-root histories")
    graph = family in {"graphsage", "combined"}
    if native and graph:
        raise ValueError("Graph AML workloads use external involvement labels")
    if graph:
        graph_spec = aml_graph(schema, max_transactions=max_events, inclusive=True)
        if family == "graphsage":
            return replace(
                graph_spec,
                sample=replace(
                    graph_spec.sample,
                    targets={"label": r.ExternalTarget(r.DType("float32"))},
                ),
                batch=replace(
                    graph_spec.batch, targets={"label": r.LabelTarget("label", r.Raw())}
                ),
            )
    fields = tuple(
        (name, name, "numerical" if name in GRAPH_AMOUNT_FIELDS else "categorical")
        for name in (
            "direction",
            "amount_paid",
            "payment_currency",
            "amount_received",
            "receiving_currency",
            "payment_format",
        )
    )
    seq = sequence(
        schema,
        "nuformer" if graph else family,
        families=(("transactions", ("account_touches",), "event_time", fields),),
        root_table="accounts",
        root_type="account",
        max_events=max_events,
        native=native,
        inclusive=graph,
    )
    if not graph:
        return seq
    agg = counts(
        schema,
        "accounts",
        (
            (f"{window}_{role}_count", (relation,), "event_time", start)
            for window, start in (
                ("all_history", None),
                ("24h", -86400),
                ("7d", -604800),
                ("30d", -2592000),
            )
            for role, relation in (
                ("total_touching", "account_touches"),
                ("sender", "sender"),
                ("receiver", "receiver"),
            )
        ),
        inclusive=True,
    )
    return replace(
        seq,
        sample=replace(
            seq.sample,
            neighborhoods=graph_spec.sample.neighborhoods,
            entity_values=agg.sample.entity_values,
        ),
        batch=replace(
            seq.batch,
            inputs={**seq.batch.inputs, **graph_spec.batch.inputs, **agg.batch.inputs},
        ),
    )


def schema_columns(schema):
    return {field.name: r.DType.from_arrow(field.type) for field in schema}


def window(clock, start=None, stop=0, *, inclusive=False):
    offset = 0 if inclusive else -1
    return r.TimeWindow(
        clock,
        None if start is None else start + offset,
        None if stop is None else stop + offset,
    )


def buckets(boundaries=None):
    return (
        r.Buckets(bins=64)
        if boundaries is None
        else r.Buckets(boundaries=tuple(boundaries))
    )


def _events(families, *, pragma, inclusive=False):
    result = {}
    for family, path, clock, fields in families:
        values = {alias: r.Field(column) for alias, column, _ in fields}
        values["__event_time"] = r.Field(clock)
        values.update(
            {
                name: r.Derived(r.CalendarPart(name), ("__event_time",))
                for name in ("hour", "weekday", "day")
            }
        )
        age = "age_seconds" if pragma else "age_range"
        if inclusive:
            # These executable common profiles have whole-second prediction
            # boundaries. Restore ages to the original prediction clock through
            # declared native expressions, before encoder fitting.
            values["__execution_age"] = r.EventTimeDelta("seconds")
            values["__prediction_offset"] = r.Constant(1.0, r.DType("float64"))
            values[age] = r.Derived("add", ("__execution_age", "__prediction_offset"))
        else:
            values[age] = r.EventTimeDelta("seconds")
        result[family] = r.EventSpec(
            r.Rows(path, window(clock, inclusive=inclusive)), values
        )
    return result


def sequence(
    schema,
    representation,
    *,
    families,
    root_table,
    root_type,
    max_events,
    native=False,
    numerical_boundaries=None,
    age_boundaries=None,
    field_state=None,
    targets=None,
    inclusive=False,
):
    pragma = representation == "pragma"
    if field_state is not None:
        raise ValueError(
            "Typed Kumo fitted state must be reused through Plan.materialize(fitted=...), not a builder field_state argument"
        )
    cuts = numerical_boundaries or {}
    kinds = {alias: kind for _, _, _, fields in families for alias, _, kind in fields}
    events = _events(families, pragma=pragma, inclusive=inclusive)
    sample = r.SampleSpec(
        r.EntitySpec(root_table, entity_type=root_type),
        events=events,
        targets={} if native else {"label": r.ExternalTarget(r.DType("float32"))},
    )
    source = r.EventSource(tuple(events))
    roles = {
        name: f"[{name.upper()}]"
        for name in ("pad", "unknown", "mask", "entity", "event")
    }
    encodes = tuple(
        r.Encode(
            name,
            buckets(cuts.get(name)) if kind == "numerical" else r.Vocabulary(),
            "history",
        )
        for name, kind in kinds.items()
    )
    if pragma:
        history = r.Fields(
            encodes,
            row=r.Tokens(max_tokens=len(kinds), keep="first", missing="omit"),
            field_names="fields",
            field_space="history",
            forbidden_fields=("entity_id", "sample_id", "label", "split"),
        )
        inputs = {
            "history": r.Input(
                source, history, r.Nested(fields=r.Packed(), events=r.Padded())
            ),
            "history_calendar": r.Input(
                r.EventsOf("history"),
                r.Fields(
                    tuple(r.Encode(n, r.Raw()) for n in ("hour", "weekday", "day"))
                ),
                r.Padded(),
            ),
            "history_timing": r.Input(
                r.EventsOf("history"),
                r.Fields((r.Encode("age_seconds", r.Raw()),)),
                r.Padded(),
            ),
        }
        native_targets = {
            "masked_values": r.MaskedValuesTarget(
                "history",
                0.15,
                0.10,
                0.10,
                0.10,
                fields=tuple(kinds),
                combine="union",
                ensure_at_least_one=True,
                supervise_unknown=False,
            )
        }
    else:
        roles["separator"] = "[SEP]"
        ordered = (r.Encode("age_range", buckets(age_boundaries), "history"), *encodes)
        # Calendar ranges are fixed rather than learned from the training cohort.
        ordered = (
            *ordered,
            *(
                r.Encode(
                    n, r.Vocabulary(values=tuple((i, i + 2) for i in values)), "history"
                )
                for n, values in (
                    ("hour", range(24)),
                    ("weekday", range(7)),
                    ("day", range(1, 32)),
                )
            ),
        )
        history = r.Fields(
            ordered,
            row=r.Tokens(missing="error"),
            history=r.Flatten(
                (len(ordered) + 1) * max_events - 1,
                keep="first",
                prefer_event_boundary=True,
                separator="separator",
            ),
        )
        inputs = {"history": r.Input(source, history, r.Padded())}
        native_targets = {"next_token": r.NextTokenTarget("history")}
    return r.DataLoaderSpec(
        schema,
        sample,
        r.BatchSpec(
            inputs,
            native_targets
            if native
            else (targets or {"label": r.LabelTarget("label", r.Raw())}),
            id_spaces={"history": r.IdSpace(roles)},
            field_names={"fields": r.FieldNames()} if pragma else {},
            batching=r.Batching(max_samples=32),
        ),
    )


def counts(schema, root, declarations, *, normalize=True, inclusive=False):
    if type(normalize) is not bool:
        raise TypeError("normalize must be boolean")
    # Declare unnormalized counts as float32 through Kumo's supported encoding.
    # This uses ordinary float32 precision, not exact integer preservation.
    encoding = (
        r.ZScore() if normalize else r.ZScore(mean=0.0, scale=1.0, dtype="float32")
    )
    values = {
        name: r.Aggregate(
            r.Rows(path, window(clock, start, inclusive=inclusive)), "count"
        )
        for name, path, clock, start in declarations
    }
    return r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec(root), entity_values=values),
        r.BatchSpec(
            {
                "aggregates": r.Input(
                    r.EntitySource(),
                    r.Fields(tuple(r.Encode(n, encoding) for n in values)),
                    r.Separate(),
                )
            }
        ),
    )


def aml_graph(schema, *, max_transactions, inclusive=False):
    if type(max_transactions) is not int or max_transactions < 1:
        raise ValueError(
            "max_transactions must be a positive verified source-row bound"
        )
    if not inclusive:
        raise ValueError(
            "AML graph workloads require strict history (event_time < cutoff); "
            "the supported Kumo GraphSample API provides inclusive cutoffs only. "
            "Exclusive graph sampling is unsupported."
        )
    neighborhood = r.NeighborhoodSpec(
        (
            r.GraphLayer(
                tuple(
                    r.GraphSample(role, max_transactions)
                    for role in ("sender", "receiver")
                )
            ),
            r.GraphLayer(
                tuple(
                    r.GraphSample(role, 1, direction="reverse")
                    for role in ("sender", "receiver")
                )
            ),
        )
    )
    features = r.Fields(
        tuple(
            [r.Encode(name, r.ZScore()) for name in GRAPH_AMOUNT_FIELDS]
            + [r.Encode(name, r.Vocabulary()) for name in GRAPH_CATEGORY_FIELDS]
            + [r.Encode(GRAPH_TIMESTAMP_FIELD, r.Raw())]
        )
    )
    return r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("accounts", entity_type="account"),
            neighborhoods={"history_graph": neighborhood},
        ),
        r.BatchSpec(
            {
                "graph": r.Input(
                    r.NeighborhoodSource("history_graph"),
                    r.Graph({"transactions": features}),
                    None,
                )
            }
        ),
    )
