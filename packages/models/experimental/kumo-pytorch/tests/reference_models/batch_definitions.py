# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper-guided data contracts with explicit encoders and model input names."""

from kumo.pytorch.api import (
    AlignedEventTarget,
    BatchSpec,
    Buckets,
    Composite,
    DType,
    Encode,
    EntitySource,
    EventsOf,
    EventSource,
    FieldNames,
    Fields,
    Flatten,
    IdSpace,
    Input,
    MaskedValuesTarget,
    MatchingGroupTarget,
    Nested,
    NextTokenTarget,
    OccurrencesOf,
    Packed,
    Padded,
    Raw,
    Separate,
    Text,
    ThresholdEventLabel,
    Tokens,
    UseEncoding,
    Vocabulary,
)

# These string values deliberately share one learned dictionary. A final ID
# space also accommodates independent numerical, Boolean, integer and BPE IDs.
PRAGMA_BATCH = BatchSpec(
    encodings={
        "strings": Vocabulary(),
        "integers": Vocabulary(),
        "description": Text(vocabulary_size=27_000, min_frequency=1, empty="unknown"),
    },
    id_spaces={
        "pragma": IdSpace(
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
    field_names={"fields": FieldNames()},
    inputs={
        "entity": Input(
            source=EntitySource(),
            representation=Fields(
                fields=(
                    Encode(
                        field="country", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(field="region", encoding=UseEncoding(name="strings"), id_space="pragma"),
                    Encode(
                        field="home_currency",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="income_band", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(
                        field="risk_segment",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="acquisition_channel",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="birth_year",
                        encoding=Buckets(bins=64, separate_zero=True),
                        id_space="pragma",
                    ),
                    Encode(
                        field="active_account_count",
                        encoding=Buckets(bins=64, separate_zero=True),
                        id_space="pragma",
                    ),
                    Encode(
                        field="lifelong_event",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="lifelong_product",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                ),
                row=Tokens(max_tokens=200, occurrence_order="chronological"),
                field_names="fields",
                field_space="pragma",
            ),
            layout=Padded(),
        ),
        "history": Input(
            source=EventSource(
                events=(
                    "banking_transaction",
                    "transfer",
                    "card_transaction",
                    "card_payment",
                    "card_dispute",
                    "investment_transaction",
                    "communication",
                    "product_adoption",
                ),
                max_events=6_500,
            ),
            representation=Fields(
                fields=(
                    Encode(
                        field="event_source",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="amount",
                        encoding=Buckets(bins=64, separate_zero=True),
                        id_space="pragma",
                    ),
                    Encode(
                        field="currency", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(
                        field="channel", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(field="mcc", encoding=UseEncoding(name="strings"), id_space="pragma"),
                    Encode(
                        field="merchant_risk",
                        encoding=Buckets(bins=64, separate_zero=True),
                        id_space="pragma",
                    ),
                    Encode(
                        field="entry_mode", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(
                        field="country", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(
                        field="recurring_candidate_flag", encoding=Vocabulary(), id_space="pragma"
                    ),
                    Encode(field="status", encoding=UseEncoding(name="strings"), id_space="pragma"),
                    Encode(
                        field="description",
                        encoding=UseEncoding(name="description"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="transaction_type",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="counterparty_type",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="transfer_type",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(
                        field="payment_type",
                        encoding=UseEncoding(name="strings"),
                        id_space="pragma",
                    ),
                    Encode(field="reason", encoding=UseEncoding(name="strings"), id_space="pragma"),
                    Encode(
                        field="outcome", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(
                        field="asset_class", encoding=UseEncoding(name="strings"), id_space="pragma"
                    ),
                    Encode(field="side", encoding=UseEncoding(name="strings"), id_space="pragma"),
                    Encode(
                        field="gross_amount",
                        encoding=Buckets(bins=64, separate_zero=True),
                        id_space="pragma",
                    ),
                    Encode(
                        field="product_id", encoding=UseEncoding(name="integers"), id_space="pragma"
                    ),
                ),
                row=Tokens(max_tokens=24),
                field_names="fields",
                field_space="pragma",
            ),
            layout=Nested(fields=Packed(), events=Padded()),
        ),
        # These inputs follow retained events, rather than repeating selection.
        "profile_timing": Input(
            source=OccurrencesOf(input="entity"),
            representation=Fields(fields=(Encode(field="age_seconds", encoding=Raw()),)),
            layout=Padded(),
        ),
        "history_calendar": Input(
            source=EventsOf(input="history"),
            representation=Fields(
                fields=(
                    Encode(field="hour", encoding=Raw()),
                    Encode(field="weekday", encoding=Raw()),
                    Encode(field="day", encoding=Raw()),
                )
            ),
            layout=Padded(),
        ),
        "history_timing": Input(
            source=EventsOf(input="history"),
            representation=Fields(fields=(Encode(field="age_seconds", encoding=Raw()),)),
            layout=Padded(),
        ),
    },
    targets={
        "masked_values": MaskedValuesTarget(
            input="history",
            token_probability=0.15,
            event_probability=0.10,
            key_probability=0.10,
            unknown_probability=0.10,
        )
    },
)

NUFORMER_BATCH = BatchSpec(
    id_spaces={
        "nuformer": IdSpace(
            specials={
                "pad": "[PAD]",
                "unknown": "[UNK]",
                "missing": "[MISSING]",
                "separator": "[SEP]",
            }
        )
    },
    inputs={
        "history": Input(
            source=EventSource(events=("banking_transaction",)),
            representation=Fields(
                fields=(
                    Encode(
                        field="direction",
                        encoding=Vocabulary(values=(("outflow", 2), ("inflow", 3))),
                        id_space="nuformer",
                    ),
                    Encode(field="absolute_amount", encoding=Buckets(bins=21), id_space="nuformer"),
                    Encode(
                        field="month",
                        encoding=Vocabulary(
                            values=tuple((value, value + 1) for value in range(1, 13))
                        ),
                        id_space="nuformer",
                    ),
                    Encode(
                        field="day",
                        encoding=Vocabulary(
                            values=tuple((value, value + 1) for value in range(1, 32))
                        ),
                        id_space="nuformer",
                    ),
                    Encode(
                        field="weekday",
                        encoding=Vocabulary(values=tuple((value, value + 2) for value in range(7))),
                        id_space="nuformer",
                    ),
                    Encode(
                        field="description",
                        encoding=Text(vocabulary_size=8_000),
                        id_space="nuformer",
                    ),
                ),
                row=Tokens(missing="skip_event"),
                history=Flatten(
                    max_tokens=2_048,
                    keep="last",
                    prefer_event_boundary=False,
                    separator="separator",
                ),
            ),
            layout=Padded(),
        )
    },
    targets={"next_token": NextTokenTarget(input="history")},
)

PANTHER_BATCH = BatchSpec(
    id_spaces={
        "profile": IdSpace(specials={"pad": "[PAD]", "unknown": "[UNK]", "missing": "[MISSING]"}),
        "panther": IdSpace(specials={"pad": "[PAD]", "unknown": "[UNK]", "missing": "[MISSING]"}),
    },
    inputs={
        "entity": Input(
            source=EntitySource(),
            representation=Fields(
                fields=(Encode(field="country", encoding=Vocabulary(), id_space="profile"),)
            ),
            layout=Separate(),
        ),
        "history": Input(
            source=EventSource(events=("card_transaction",), max_events=2_048),
            representation=Fields(
                fields=(
                    Encode(field="channel", encoding=Vocabulary()),
                    Encode(field="amount", encoding=Buckets(bins=20)),
                    Encode(field="mcc", encoding=Vocabulary()),
                    Encode(field="merchant_risk", encoding=Buckets(bins=20)),
                ),
                row=Composite(
                    top_k=60_000,
                    required_fields=("channel", "amount", "mcc", "merchant_risk"),
                    id_space="panther",
                ),
                history=Flatten(max_tokens=2_048),
            ),
            layout=Padded(),
        ),
    },
    targets={
        "next_behavior": NextTokenTarget(input="history", prefix_input="entity"),
        "entity_match": MatchingGroupTarget(input="entity", fields=("country",)),
    },
)

TREASURE_BATCH = BatchSpec(
    inputs={
        "entity": Input(
            source=EntitySource(),
            representation=Fields(
                fields=(
                    Encode(field="country", encoding=Vocabulary()),
                    Encode(field="active_account_count", encoding=Raw()),
                )
            ),
            layout=Separate(),
        ),
        "events": Input(
            source=EventSource(events=("card_transaction",), max_events=2_048),
            representation=Fields(
                fields=(
                    Encode(field="amount", encoding=Raw()),
                    Encode(field="channel", encoding=Vocabulary()),
                    Encode(field="country", encoding=Vocabulary()),
                    Encode(field="currency", encoding=Vocabulary()),
                    Encode(field="entry_mode", encoding=Vocabulary()),
                    Encode(field="mcc", encoding=Vocabulary()),
                    Encode(field="recurring_candidate_flag", encoding=Vocabulary()),
                )
            ),
            layout=Padded(),
        ),
    },
    targets={
        "current_event": AlignedEventTarget(
            input="events",
            fields=(Encode(field="status", encoding=Vocabulary()),),
            derived=(
                ThresholdEventLabel(
                    name="synthetic_abnormal",
                    field="amount",
                    dtype=DType(name="string"),
                    encoding=Vocabulary(),
                ),
            ),
        )
    },
)

BATCH_SPECS = {
    "pragma": PRAGMA_BATCH,
    "nuformer": NUFORMER_BATCH,
    "panther": PANTHER_BATCH,
    "treasure": TREASURE_BATCH,
}
