# 7. Model input shapes

Published sequence models over transactions disagree about what one model input
should look like. Some want a short token sequence per event, some one long
stream per user, some one ID per event, some separate typed columns. This page
builds all four from the same underlying tables, changing the sampling
(`SampleSpec`) and the batch presentation (`BatchSpec`) as each model requires.

The four shapes follow published models. They reproduce each model's data
requirements, not its full training recipe or reported results; each section
states where it deviates.

| Shape | Per-event output | Across events | Extra inputs | Objective |
| --- | --- | --- | --- | --- |
| [PRAGMA](#pragma-field-tokens-per-event) | Field tokens with field IDs | Grouped events, packed fields | Calendar parts and event ages, aligned to the kept events | Masked value IDs in the shared ID space |
| [nuFormer](#nuformer-one-token-stream-per-user) | Direction, amount bucket, calendar IDs and text pieces | Flattened, with separators and a total budget | None; calculations live in the sample | Next token |
| [PANTHER](#panther-one-behavior-id-per-event) | One composite ID per ordered field tuple | Flattened, one ID per event | A separately encoded profile | Next behavior and a profile matching group |
| [TREASURE](#treasure-independent-typed-columns) | Independent numerical and categorical columns | Grouped, padded events | Held-out labels aligned to the events | Current and next event attributes |

Every model reads the fitted state of its task (vocabulary sizes, ID spaces)
to size its embeddings. Each model's code may reshape the arrays, but it uses
the task's fitted vocabularies, buckets and thresholds instead of fitting its
own. The helper below uses Python requests to inspect model inputs. Use a
[Parquet AnchorTable](05-training-batches.md#training-on-the-diskgraph-engine)
to load a training or benchmark population on DiskGraph.

## The shared data

The data has four tables: users, their accounts, merchants and card
transactions. Times are whole seconds, and every example below predicts at
second 30, so the transactions at seconds 35 and 40 are the future.

```python
import kumo.pytorch as kp

TS = kp.DType(name="timestamp", unit="s", timezone="UTC")

schema = kp.DataSchema(
    tables=(
        kp.TableSpec(
            name="users",
            primary_key="user_id",
            columns={
                "user_id": kp.DType(name="int64"),
                "country": kp.DType(name="string"),
                "risk_segment": kp.DType(name="string"),
            },
            time_column=kp.NON_TEMPORAL,
        ),
        kp.TableSpec(
            name="accounts",
            primary_key="account_id",
            columns={
                "account_id": kp.DType(name="int64"),
                "user_id": kp.DType(name="int64"),
                "account_type": kp.DType(name="string"),
                "opened_at": TS,
                "closed_at": TS,
            },
            time_column="opened_at",
        ),
        kp.TableSpec(
            name="merchants",
            primary_key="merchant_id",
            columns={
                "merchant_id": kp.DType(name="int64"),
                "risk_score": kp.DType(name="float64"),
            },
            time_column=kp.NON_TEMPORAL,
        ),
        kp.TableSpec(
            name="transactions",
            primary_key="transaction_id",
            columns={
                "transaction_id": kp.DType(name="int64"),
                "user_id": kp.DType(name="int64"),
                "account_id": kp.DType(name="int64"),
                "merchant_id": kp.DType(name="int64"),
                "occurred_at": TS,
                "amount": kp.DType(name="float64"),
                "transaction_type": kp.DType(name="string"),
                "currency": kp.DType(name="string"),
                "channel": kp.DType(name="string"),
                "mcc": kp.DType(name="string"),
                "status": kp.DType(name="string"),
                "description": kp.DType(name="string"),
            },
            time_column="occurred_at",
        ),
    ),
    relations=(
        kp.RelationSpec(
            name="user_transactions",
            left_table="users",
            left_key="user_id",
            right_table="transactions",
            right_key="user_id",
        ),
        kp.RelationSpec(
            name="user_accounts",
            left_table="users",
            left_key="user_id",
            right_table="accounts",
            right_key="user_id",
        ),
        kp.RelationSpec(
            name="account_transactions",
            left_table="accounts",
            left_key="account_id",
            right_table="transactions",
            right_key="account_id",
        ),
        kp.RelationSpec(
            name="transaction_merchant",
            left_table="transactions",
            left_key="merchant_id",
            right_table="merchants",
            right_key="merchant_id",
        ),
    ),
)


def transaction(
    id, user, account, merchant, at, amount, kind, currency, channel, mcc, status, text
):
    return {
        "transaction_id": id,
        "user_id": user,
        "account_id": account,
        "merchant_id": merchant,
        "occurred_at": at,
        "amount": amount,
        "transaction_type": kind,
        "currency": currency,
        "channel": channel,
        "mcc": mcc,
        "status": status,
        "description": text,
    }


rows = {
    "users": [
        {"user_id": 1, "country": "US", "risk_segment": "low"},
        {"user_id": 2, "country": "GB", "risk_segment": "medium"},
    ],
    "accounts": [
        {
            "account_id": 11,
            "user_id": 1,
            "account_type": "credit_card",
            "opened_at": 1,
            "closed_at": None,
        },
        {
            "account_id": 21,
            "user_id": 2,
            "account_type": "credit_card",
            "opened_at": 1,
            "closed_at": None,
        },
    ],
    "merchants": [
        {"merchant_id": 50, "risk_score": 0.1},
        {"merchant_id": 51, "risk_score": 0.8},
    ],
    "transactions": [
        transaction(
            101,
            1,
            11,
            50,
            10,
            120.0,
            "deposit",
            "USD",
            "online",
            "5815",
            "approved",
            "salary payment",
        ),
        transaction(
            102,
            1,
            11,
            51,
            20,
            18.5,
            "withdrawal",
            "USD",
            "store",
            "5411",
            "declined",
            "grocery store",
        ),
        transaction(
            201,
            2,
            21,
            50,
            15,
            42.0,
            "fee",
            "GBP",
            "online",
            "5815",
            "approved",
            "monthly account fee",
        ),
        transaction(
            202,
            2,
            21,
            51,
            35,
            75.0,
            "withdrawal",
            "GBP",
            "store",
            "5411",
            "declined",
            "utility bill",
        ),
        transaction(
            103,
            1,
            11,
            50,
            40,
            25.0,
            "withdrawal",
            "USD",
            "online",
            "5815",
            "approved",
            "cash withdrawal",
        ),
    ],
}
dataset = kp.Dataset(tables={name: kp.InMemoryRows(rows=table) for name, table in rows.items()})
past = kp.Rows(path=("user_transactions",), time=kp.TimeWindow(column="occurred_at", stop=0))
users_at_30 = (
    kp.SampleRequest(sample_id="user-1-at-30", entity_id=1, as_of_time=30, split="train"),
    kp.SampleRequest(sample_id="user-2-at-30", entity_id=2, as_of_time=30, split="train"),
)


def first_batch(sample, batch, anchors=users_at_30):
    contract = kp.DataLoaderSpec(schema=schema, sample=sample, batch=batch)
    # The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
    with kp.prepare(contract=contract, dataset=dataset, engine=kp.InMemoryEngine()) as plan:
        with plan.materialize(anchors=anchors, fit_split="train") as task:
            return next(task.batches(requests=anchors, seed=17))
```

`first_batch` fits the encoders on the training anchors and returns the first
batch. The rest of the page only declares samples and batches.

## PRAGMA: field tokens per event

Reference: [PRAGMA: Revolut Foundation
Model](https://arxiv.org/abs/2604.08649).

The model wants a summary of the user, a summary of each event, and a sequence
of event summaries. The batch supplies each event as a short sequence of field
tokens (amount bucket, channel, description pieces), each paired with a field
ID that says which column it came from. The model builds the neural summaries;
the batch only reserves `[USR]` and `[EVT]` IDs for them.

Calendar parts and the event's age (anchor time minus event time) are sample
calculations. They reach the model only because two inputs ask for them, and
`EventsOf(input="history")` keeps them aligned with exactly the events
`history` kept.

```python
pragma_sample = kp.SampleSpec(
    entity=kp.EntitySpec(table="users", entity_type="user"),
    entity_values={
        "country": kp.Field(column="country"),
        "risk_segment": kp.Field(column="risk_segment"),
        "transaction_count": kp.Aggregate(source=past, operation="count"),
    },
    events={
        "transactions": kp.EventSpec(
            rows=past,
            fields={
                "amount": kp.Field(column="amount"),
                "channel": kp.Field(column="channel"),
                "description": kp.Field(column="description"),
                "time": kp.Field(column="occurred_at"),
                "hour": kp.Derived(operation=kp.CalendarPart(part="hour"), inputs=("time",)),
                "weekday": kp.Derived(operation=kp.CalendarPart(part="weekday"), inputs=("time",)),
                "age_seconds": kp.EventTimeDelta(unit="seconds", dtype=kp.DType(name="float64")),
            },
        )
    },
)


def pragma_tokens(*fields):
    return kp.Fields(
        fields=fields,
        row=kp.Tokens(max_tokens=8),
        field_names="fields",  # a field ID beside every value piece
        field_space="pragma",  # in the same ID space as the values
    )


def into_pragma(field, encoding):
    return kp.Encode(field=field, encoding=encoding, id_space="pragma")


pragma_batch = kp.BatchSpec(
    # Country, risk segment and channel share one category mapping on purpose.
    encodings={"strings": kp.Vocabulary(), "description": kp.Text(vocabulary_size=64)},
    id_spaces={
        "pragma": kp.IdSpace(
            specials={
                "pad": "[PAD]",
                "unknown": "[UNK]",
                "missing": "[MISSING]",
                "mask": "[MASK]",
                "entity": "[USR]",
                "event": "[EVT]",
            }
        )
    },
    field_names={"fields": kp.FieldNames()},
    inputs={
        "entity": kp.Input(
            source=kp.EntitySource(),
            representation=pragma_tokens(
                into_pragma("country", kp.UseEncoding(name="strings")),
                into_pragma("risk_segment", kp.UseEncoding(name="strings")),
                into_pragma("transaction_count", kp.Buckets(bins=4, separate_zero=True)),
            ),
            layout=kp.Padded(),
        ),
        "history": kp.Input(
            source=kp.EventSource(events=("transactions",), max_events=16),
            representation=pragma_tokens(
                into_pragma("amount", kp.Buckets(bins=4, separate_zero=True)),
                into_pragma("channel", kp.UseEncoding(name="strings")),
                into_pragma("description", kp.UseEncoding(name="description")),
            ),
            layout=kp.Nested(fields=kp.Packed(), events=kp.Padded()),
        ),
        "history_calendar": kp.Input(
            source=kp.EventsOf(input="history"),
            representation=kp.Fields(
                fields=(
                    kp.Encode(field="hour", encoding=kp.Raw()),
                    kp.Encode(field="weekday", encoding=kp.Raw()),
                )
            ),
            layout=kp.Padded(),
        ),
        "history_timing": kp.Input(
            source=kp.EventsOf(input="history"),
            representation=kp.Fields(fields=(kp.Encode(field="age_seconds", encoding=kp.Raw()),)),
            layout=kp.Padded(),
        ),
    },
    targets={
        "masked_values": kp.MaskedValuesTarget(
            input="history",
            token_probability=0.15,
            event_probability=0.1,
            key_probability=0.1,
            fields=("amount", "channel", "description"),
            ensure_at_least_one=True,
        )
    },
    batching=kp.Batching(max_samples=2),
)

batch = first_batch(pragma_sample, pragma_batch)
history = batch.inputs["history"]
print("inputs:", tuple(batch.inputs))
print("targets:", tuple(batch.targets))
print("event slots:", tuple(history["sequence"]["valid"].shape))
print("tokens per event:", history["field_offsets"].diff().tolist())
print("event owner:", history["owner"].tolist())
```

<!-- test: output -->
```text
inputs: ('entity', 'history', 'history_calendar', 'history_timing')
targets: ('masked_values',)
event slots: (2, 2)
tokens per event: [4, 4, 5]
event owner: [0, 0, 1]
```

`history` is a packed buffer of field tokens shared by all events of the batch,
but nothing is merged: `field_offsets` delimits each event's tokens, `owner`
says which sample an event belongs to, and `sequence` lays the events out per
sample with a validity mask. `value_ids`, `field_ids` and `position_ids` align
piece by piece, so two description pieces share a field ID but not a position.

`MaskedValuesTarget` replaces selected value IDs of `history` with `[MASK]` or
`[UNK]` and returns the originals under `batch.targets["masked_values"]`, with
separate masks for "this slot is real" and "this slot was corrupted". It
changes only the input it names.

Deviations: the model, not the batch, re-expresses event ages relative to the
latest event; dated profile facts (`OccurrencesOf`) are left out to keep the
page short. For pre-training this shape and fine-tuning it on a new label, see
the [transaction model example](../examples/pragma-fraud/).

## nuFormer: one token stream per user

Reference: [Your Spending Needs Attention: Modeling Financial Habits with
Transformers](https://arxiv.org/abs/2507.23267).

Every event contributes direction, amount magnitude, calendar parts and the
description's text pieces, and all events of a user become one sequence for a
causal model. Direction and magnitude are two sample calculations, not two
encodings of one value.

```python
nuformer_sample = kp.SampleSpec(
    entity=kp.EntitySpec(table="users", entity_type="user"),
    events={
        "transactions": kp.EventSpec(
            rows=past,
            fields={
                "amount": kp.Field(column="amount"),
                "transaction_type": kp.Field(column="transaction_type"),
                "description": kp.Field(column="description"),
                "time": kp.Field(column="occurred_at"),
                "direction": kp.Derived(
                    operation=kp.MapValues(
                        groups={
                            "inflow": ("deposit", "interest"),
                            "outflow": ("withdrawal", "fee"),
                        },
                        unknown="error",  # don't guess a direction for another type
                    ),
                    inputs=("transaction_type",),
                    dtype=kp.DType(name="string"),
                ),
                "absolute_amount": kp.Derived(operation="abs", inputs=("amount",)),
                "month": kp.Derived(operation=kp.CalendarPart(part="month"), inputs=("time",)),
                "weekday": kp.Derived(operation=kp.CalendarPart(part="weekday"), inputs=("time",)),
            },
        )
    },
)


def into_tokens(field, encoding):
    return kp.Encode(field=field, encoding=encoding, id_space="tokens")


nuformer_batch = kp.BatchSpec(
    id_spaces={
        "tokens": kp.IdSpace(
            specials={
                "pad": "[PAD]",
                "unknown": "[UNK]",
                "missing": "[MISSING]",
                "separator": "[SEP]",
            }
        )
    },
    inputs={
        "history": kp.Input(
            source=kp.EventSource(events=("transactions",)),
            representation=kp.Fields(
                fields=(
                    into_tokens("direction", kp.Vocabulary(values=(("inflow", 2), ("outflow", 3)))),
                    into_tokens("absolute_amount", kp.Buckets(bins=4)),
                    # Fixed entries cover months and weekdays the fitting data never saw.
                    into_tokens(
                        "month", kp.Vocabulary(values=tuple((m, m + 1) for m in range(1, 13)))
                    ),
                    into_tokens(
                        "weekday", kp.Vocabulary(values=tuple((d, d + 2) for d in range(7)))
                    ),
                    into_tokens("description", kp.Text(vocabulary_size=64)),
                ),
                row=kp.Tokens(missing="skip_event"),
                history=kp.Flatten(max_tokens=128, separator="separator", keep="last"),
            ),
            layout=kp.Padded(),
        )
    },
    targets={"next_token": kp.NextTokenTarget(input="history")},
    batching=kp.Batching(max_samples=2),
)

batch = first_batch(nuformer_sample, nuformer_batch)
print("inputs:", tuple(batch.inputs))
print("targets:", tuple(batch.targets))
print("stream:", tuple(batch.inputs["history"]["value_ids"].shape))
print("next token:", tuple(batch.targets["next_token"]["values"].shape))
```

<!-- test: output -->
```text
inputs: ('history',)
targets: ('next_token',)
stream: (2, 12)
next token: (2, 12)
```

The field tuple fixes the order within an event; `Flatten` joins a user's
events with `[SEP]` and keeps the newest 128 tokens, markers included. It never
joins two users. The local ID 2 for "inflow" is not its ID in the `tokens` ID
space: read those IDs and the space's size from `task.spaces["tokens"]`.

`NextTokenTarget` shifts the already encoded stream. For a stream `[a, b, c, d]`
the input is `[a, b, c]` and the target `[b, c, d]`; padding is invalid and
never enters the loss. No second vocabulary is fitted.

Deviations: the day-of-month token is omitted, and the model (causal attention,
loss, optional low-rank adaptation) is yours.

## PANTHER: one behavior ID per event

Reference: [PANTHER: Generative Pretraining Beyond Language for Sequential User
Behavior Modeling](https://arxiv.org/abs/2510.10102).

A behavior ID stands for one complete ordered tuple of event fields (channel,
amount bucket, merchant category, merchant risk), not for a source row. The
merchant's risk is joined onto each transaction during sampling.

```python
panther_sample = kp.SampleSpec(
    entity=kp.EntitySpec(table="users", entity_type="user"),
    entity_values={"country": kp.Field(column="country")},
    events={
        "transactions": kp.EventSpec(
            rows=past,
            fields={
                "channel": kp.Field(column="channel"),
                "amount": kp.Field(column="amount"),
                "mcc": kp.Field(column="mcc"),
                "merchant_risk": kp.Field(column="risk_score", path=("transaction_merchant",)),
            },
        )
    },
)

specials = {"pad": "[PAD]", "unknown": "[UNK]", "missing": "[MISSING]"}
panther_batch = kp.BatchSpec(
    id_spaces={
        "profile": kp.IdSpace(specials=specials),
        "behaviors": kp.IdSpace(specials=specials),
    },
    inputs={
        "entity": kp.Input(
            source=kp.EntitySource(),
            representation=kp.Fields(
                fields=(kp.Encode(field="country", encoding=kp.Vocabulary(), id_space="profile"),)
            ),
            layout=kp.Separate(),
        ),
        "history": kp.Input(
            source=kp.EventSource(events=("transactions",), max_events=16),
            representation=kp.Fields(
                fields=(
                    kp.Encode(field="channel", encoding=kp.Vocabulary()),
                    kp.Encode(field="amount", encoding=kp.Buckets(bins=4)),
                    kp.Encode(field="mcc", encoding=kp.Vocabulary()),
                    kp.Encode(field="merchant_risk", encoding=kp.Buckets(bins=4)),
                ),
                row=kp.Composite(
                    top_k=16,  # the 16 most frequent tuples get their own ID
                    required_fields=("channel", "amount", "mcc", "merchant_risk"),
                    id_space="behaviors",
                ),
                history=kp.Flatten(max_tokens=16),
            ),
            layout=kp.Padded(),
        ),
    },
    targets={
        "next_behavior": kp.NextTokenTarget(input="history", prefix_input="entity"),
        "entity_match": kp.MatchingGroupTarget(input="entity", fields=("country",)),
    },
    batching=kp.Batching(max_samples=2),
)

batch = first_batch(panther_sample, panther_batch)
print("inputs:", tuple(batch.inputs))
print("targets:", tuple(batch.targets))
print("behaviors:", tuple(batch.inputs["history"]["value_ids"].shape))
print("next behavior:", tuple(batch.targets["next_behavior"]["values"].shape))
```

<!-- test: output -->
```text
inputs: ('entity', 'history')
targets: ('next_behavior', 'entity_match')
behaviors: (2, 1)
next behavior: (2, 2)
```

Fitting encodes each component first, then counts the ordered tuples of those
results. A tuple outside the top 16 gets the unknown behavior ID, and an event
missing a required field is dropped. Text fields can't be components, and
`prepare` raises an error if you use one.

With `prefix_input="entity"` the target has one more slot than the history:
the first behavior is predicted from a profile position the model creates and
fills with the separately encoded country. The batch does not insert a country
token into the history. `MatchingGroupTarget` gives equal group IDs to equal raw
profile values; choosing negatives and the contrastive loss stay in the model.

Deviations: unseen tuples fall back to one unknown ID, not hashing or
reconstruction from components.

## TREASURE: independent typed columns

Reference: [The Visa Payment Foundation Model for High-Volume Transaction
Understanding](https://arxiv.org/abs/2511.19693).

The entity is a credit-card account. Each event keeps its fields as separate
typed columns: amount stays a float, currency and channel become category IDs.
The transaction's status is a label, not an input: it is withheld from the
history and returned aligned with it. A second target is the account's first
transaction after the prediction time.

```python
account_past = kp.Rows(
    path=("account_transactions",), time=kp.TimeWindow(column="occurred_at", stop=0)
)
account_future = kp.Rows(
    path=("account_transactions",), time=kp.TimeWindow(column="occurred_at", start=0, stop=None)
)
event_fields = {name: kp.Field(column=name) for name in ("amount", "currency", "channel", "status")}
treasure_sample = kp.SampleSpec(
    entity=kp.EntitySpec(
        table="accounts",
        entity_type="card_account",
        filters=(kp.RowFilter(column="account_type", operation="equal", value="credit_card"),),
        available_from="opened_at",
    ),
    entity_values={"country": kp.Field(column="country", path=("user_accounts",))},
    events={"transactions": kp.EventSpec(rows=account_past, fields=event_fields)},
    targets={
        "next_transaction": kp.RowTarget(
            rows=account_future, fields=event_fields, selection="first"
        )
    },
)

currency = kp.Vocabulary(values=(("USD", 2), ("GBP", 3)))
channel = kp.Vocabulary(values=(("online", 2), ("store", 3)))
status = kp.Vocabulary(values=(("approved", 2), ("declined", 3)))
treasure_batch = kp.BatchSpec(
    inputs={
        "entity": kp.Input(
            source=kp.EntitySource(),
            representation=kp.Fields(
                fields=(kp.Encode(field="country", encoding=kp.Vocabulary()),)
            ),
            layout=kp.Separate(),
        ),
        "events": kp.Input(
            source=kp.EventSource(events=("transactions",), max_events=64),
            representation=kp.Fields(
                fields=(
                    kp.Encode(field="amount", encoding=kp.Raw()),
                    kp.Encode(field="currency", encoding=currency),
                    kp.Encode(field="channel", encoding=channel),
                ),
                row=kp.Columns(),
            ),
            layout=kp.Padded(),
        ),
    },
    targets={
        "event_labels": kp.AlignedEventTarget(
            input="events",
            fields=(kp.Encode(field="status", encoding=status),),
            derived=(
                kp.ThresholdEventLabel(
                    name="large",
                    field="amount",
                    dtype=kp.DType(name="string"),
                    threshold=50.0,
                    below="normal",
                    above="large",
                    encoding=kp.Vocabulary(values=(("normal", 2), ("large", 3))),
                ),
            ),
        ),
        "next_event": kp.StructuredTarget(
            source="next_transaction",
            fields=(
                kp.Encode(field="amount", encoding=kp.Raw()),
                kp.Encode(field="currency", encoding=currency),
                kp.Encode(field="status", encoding=status),
            ),
        ),
    },
    batching=kp.Batching(max_samples=2),
)

accounts_at_30 = (
    kp.SampleRequest(sample_id="account-11-at-30", entity_id=11, as_of_time=30, split="train"),
    kp.SampleRequest(sample_id="account-21-at-30", entity_id=21, as_of_time=30, split="train"),
)
batch = first_batch(treasure_sample, treasure_batch, accounts_at_30)
events = batch.inputs["events"]
print("inputs:", tuple(batch.inputs))
print("targets:", tuple(batch.targets))
print("event columns:", tuple(events))
print("amount:", events["amount"]["values"].dtype, "currency:", events["currency"]["values"].dtype)
print("next amount valid:", batch.targets["next_event"]["amount"]["valid"].tolist())
```

<!-- test: output -->
```text
inputs: ('entity', 'events')
targets: ('event_labels', 'next_event')
event columns: ('amount', 'currency', 'channel', 'valid')
amount: torch.float64 currency: torch.int64
next amount valid: [True, True]
```

Every column has its own `values` and `valid` arrays, shaped by sample and
event. A missing amount is an invalid slot, not a zero; a missing category and
an unseen one get different IDs. Both accounts have a transaction after second
30, so both next-event targets are valid; an account with none would get an
invalid target, not a made-up zero or negative class.

Status appears only in `event_labels`, aligned to the events `events` kept;
putting it in an input as well is refused. The threshold label reads amount
without removing it from the input. Leave out `threshold` and it is fitted (the
median) on the training anchors only.

Deviations: if the model drops rare categories from its loss, it decides that
from the task's fitted vocabulary sizes; the batch doesn't add another encoder
for it.

## Next

- [Encodings and IDs](03-encodings-and-ids.md): sharing encoders, ID
  spaces and field-name IDs in detail.
- [Training batches](05-training-batches.md): layouts, targets and batching.
- [Supported features and limitations](limitations.md): which of these shapes
  the DiskGraph engine executes today.
- [API reference](api/README.md) for every declaration used here.
