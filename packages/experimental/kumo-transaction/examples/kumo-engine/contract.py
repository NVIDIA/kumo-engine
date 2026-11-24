# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The relational schema, one anchored sample, and its model-facing batch layout."""

from pathlib import Path

from kumo.pytorch import api

# Each stream owns its fields. A transaction never emits an app's screen token.
# Within a stream, a nullable column (e.g. counterparty on a card transaction)
# becomes an explicit missing token. A name shared by streams, such as channel,
# uses one value encoder but retains each component's field-name identity.
SOURCES = {
    "transaction": (
        "transactions",
        (
            "amount",
            "currency",
            "mcc",
            "merchant",
            "txn_type",
            "channel",
            "country",
            "counterparty",
        ),
    ),
    "app": (
        "app_events",
        ("screen", "action", "device_id", "os", "app_version", "geo_country"),
    ),
    "trading": (
        "trades",
        ("instrument", "side", "quantity", "price", "order_type", "venue"),
    ),
    "communication": (
        "communications",
        ("comm_event", "campaign_id", "channel", "template"),
    ),
}
EVENT_FIELDS = tuple(dict.fromkeys(("source", *(f for _, fs in SOURCES.values() for f in fs))))
NUMERIC_FIELDS = {"amount", "quantity", "price"}
STATIC_PROFILE_FIELDS = (
    "age_band",
    "country",
    "device_count",
    "housing",
    "income_band",
    "kyc_tier",
    "marital",
    "occupation",
)
FIRST_TRANSACTION = "first_transaction"
FIRST_TRANSACTION_AGE = "seconds_since_first_transaction"
PROFILE_FIELDS = (*STATIC_PROFILE_FIELDS, FIRST_TRANSACTION)
TIME_FIELDS = ("hour", "dow", "dom", "age_seconds")


def make_contract(max_events: int = 128, batch_size: int = 128) -> api.DataLoaderSpec:
    """Describe the default Blueprint-v2 workload using Kumo's declarative API."""
    string = api.DType("string")
    timestamp = api.DType("timestamp", unit="s", timezone="UTC")

    # 1. SCHEMA: user_id is the root key; each event has its own primary key
    # and a foreign key back to users. Static profile attributes have no clock.
    # Event ts columns are the clocks used for point-in-time traversal.
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "users",
                "user_id",
                {"user_id": string, **{f: string for f in STATIC_PROFILE_FIELDS}},
                time_column=api.NON_TEMPORAL,
            ),
            *(
                api.TableSpec(
                    table,
                    "event_id",
                    {
                        "event_id": string,
                        "user_id": string,
                        "ts": timestamp,
                        "source": string,
                        **{
                            f: api.DType("float64") if f in NUMERIC_FIELDS else string
                            for f in fields
                        },
                    },
                    time_column="ts",
                )
                for table, fields in SOURCES.values()
            ),
        ),
        relations=tuple(
            api.RelationSpec(f"user_{table}", "users", "user_id", table, "user_id")
            for table, _ in SOURCES.values()
        ),
    )

    # 2. SAMPLE: an anchor is (user_id, as_of_time). Rows follows a relation
    # from that user. TimeWindow offsets are seconds relative to this anchor;
    # stop=0 includes its timestamp and excludes future rows.
    # First-transaction age intentionally uses ALL prior transaction history,
    # independently of the 730-day / 128-event context used by the decoder.
    first_transaction = api.GroupBy(
        source=api.Rows(("user_transactions",), api.TimeWindow("ts", stop=0)),
        keys=("user_id",),
        values={"first_at": api.Reduce(operation="min", column="ts")},
    )
    events = {
        table: api.EventSpec(
            api.Rows((f"user_{table}",), api.TimeWindow("ts", start=-730 * 86400, stop=0)),
            {
                **{f: api.Field(column=f) for f in ("source", *fields)},
                "event_time": api.Field(column="ts"),
                "hour": api.Derived(api.CalendarPart("hour"), ("event_time",)),
                "dow": api.Derived(api.CalendarPart("weekday"), ("event_time",)),
                "dom": api.Derived(api.CalendarPart("day"), ("event_time",)),
                "age_seconds": api.EventTimeDelta(unit="seconds"),
            },
        )
        for table, fields in SOURCES.values()
    }
    sample = api.SampleSpec(
        entity=api.EntitySpec(table="users", entity_type="user"),
        entity_values={
            **{f: api.Field(column=f) for f in STATIC_PROFILE_FIELDS},
            # This token names the fact; it does not encode an absolute date.
            FIRST_TRANSACTION: api.Constant(value=FIRST_TRANSACTION, dtype=string),
            # recency returns anchor minus first_at in raw seconds, or NULL
            # before the first transaction. Zero seconds is a valid value.
            FIRST_TRANSACTION_AGE: api.Aggregate(
                source=first_transaction, operation="recency", column="first_at"
            ),
        },
        events=events,
        # Targets are absent: the existing blueprint model constructs both
        # next-event supervision and MLM corruption from the resulting batch.
    )

    # 3. ENCODING: timestamped columns fit up to the latest train anchor;
    # static attributes use the supplied snapshot. Other encoders fit on
    # train samples. Categorical fields use vocabularies; numbers use buckets.
    # bins=17 reserves one zero bucket and up to 16 nonzero quantile buckets.
    # Both keys and values join the same model embedding namespace.
    def tokens(fields: tuple[str, ...]) -> api.Fields:
        return api.Fields(
            fields=tuple(
                api.Encode(
                    field=f,
                    encoding=api.Buckets(bins=17, separate_zero=True)
                    if f in NUMERIC_FIELDS
                    else api.Vocabulary(max_size=4096),
                    id_space="model",
                )
                for f in fields
            ),
            row=api.Tokens(missing="emit"),
            field_names="fields",
            field_space="model",
        )

    # 4. BATCH: EventSource combines the streams, selects the latest 128
    # events globally, and orders the retained events from oldest to newest.
    # Nested(Padded, Padded) emits real event-token rows [E,T], plus a padded
    # [B,N] event-index grid. The adapter gathers these into model [B,N,T].
    # Model-format task iteration retrieves and merges streams natively.
    batch = api.BatchSpec(
        id_spaces={
            "model": api.IdSpace(
                # Match the IDs used by Blueprint's model and MLM objective.
                # Kumo assigns them directly; the adapter passes IDs through.
                specials={
                    "pad": "[PAD]",
                    "unknown": "[UNK]",
                    "missing": "[MISSING]",
                    "mask": "[MASK]",
                    "entity": "[USR]",
                    "event": "[EVT]",
                },
                special_ids={
                    "pad": 0,
                    "mask": 1,
                    "unknown": 2,
                    "entity": 3,
                    "event": 4,
                    "missing": 5,
                },
            )
        },
        field_names={"fields": api.FieldNames()},
        inputs={
            "history": api.Input(
                source=api.EventSource(
                    events=tuple(events),
                    max_events=max_events,
                    keep="last",
                    order="ascending",
                ),
                representation=tokens(EVENT_FIELDS),
                layout=api.Nested(fields=api.Padded(), events=api.Padded()),
            ),
            # EventsOf follows history's exact selection and ordering. Raw
            # time columns therefore align with that [B,N] grid without a join.
            "history_aux": api.Input(
                source=api.EventsOf(input="history"),
                representation=api.Fields(
                    fields=tuple(api.Encode(field=f, encoding=api.Raw()) for f in TIME_FIELDS),
                    row=api.Columns(),
                ),
                layout=api.Padded(),
            ),
            # Scalar profile tokens already arrive as padded [B,P] arrays.
            "profile": api.Input(
                source=api.EntitySource(),
                representation=tokens(PROFILE_FIELDS),
                layout=api.Padded(),
            ),
            # The numeric age travels separately from its fact token. Its
            # validity mask tells the adapter whether to include that fact.
            "profile_ages": api.Input(
                source=api.EntitySource(),
                representation=api.Fields(
                    fields=(api.Encode(field=FIRST_TRANSACTION_AGE, encoding=api.Raw()),),
                    row=api.Columns(),
                ),
                layout=api.Separate(),
            ),
        },
        # Group similar history lengths within native shuffled anchor pools.
        batching=api.Batching(
            max_samples=batch_size,
            drop_last=False,
            length_input="history",
            bucket_by_length=True,
        ),
    )
    return api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)


def make_dataset(root: Path) -> api.Dataset:
    """Bind this schema to the scalar Parquet files written by seed_data.py."""
    return api.Dataset(
        {
            name: api.Parquet(str(root / "tables" / f"{name}.parquet"))
            for name in ("users", *(table for table, _ in SOURCES.values()))
        }
    )


def make_anchors(path: Path) -> api.AnchorTable:
    """Multiple sample IDs may point to one user at different anchor times."""
    return api.AnchorTable(
        source=api.Parquet(str(path)),
        entity_id="user_id",
        as_of_time="as_of_time",
        sample_id="sample_id",
        split="split",
    )
