# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tokenizer for the v1 12-field transaction layout.

Field order: AMT MERCH CAT MCC HOUR DOW MONTH CARD CHIP ZIP3 STATE CUST.
`fields` selects and orders a subset; ids are dense over the selected fields.

This implementation is pandas/numpy based. It is not bitwise-identical to the
v1 RAPIDS tokenizer (merchant hashing differs from cuDF `hash_values`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Self

import torch

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.schema import TransactionEvent, TransactionSequence, ensure_utc
from tfm.contracts.tokenizers import LAYOUT_FLAT_FIELDS, EncodedFlatSequence, FieldVocabulary
from tfm.features.tokenizers.registry import register_tokenizer

if TYPE_CHECKING:
    from tfm.config import DataConfig, FeaturesConfig

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")
FIELD_ORDER = (
    "AMT",
    "MERCH",
    "CAT",
    "MCC",
    "HOUR",
    "DOW",
    "MONTH",
    "CARD",
    "CHIP",
    "ZIP3",
    "STATE",
    "CUST",
)

KNOWN_MCCS = [
    -1,
    1711,
    3000,
    3001,
    3005,
    3006,
    3007,
    3008,
    3009,
    3058,
    3066,
    3075,
    3132,
    3144,
    3174,
    3256,
    3260,
    3359,
    3387,
    3389,
    3390,
    3393,
    3395,
    3405,
    3504,
    3509,
    3596,
    3640,
    3684,
    3722,
    3730,
    3771,
    3775,
    3780,
    4111,
    4112,
    4121,
    4131,
    4214,
    4411,
    4511,
    4722,
    4784,
    4814,
    4829,
    4899,
    4900,
    5045,
    5094,
    5192,
    5193,
    5211,
    5251,
    5261,
    5300,
    5310,
    5311,
    5411,
    5499,
    5533,
    5541,
    5621,
    5651,
    5655,
    5661,
    5712,
    5719,
    5722,
    5732,
    5733,
    5812,
    5813,
    5814,
    5815,
    5816,
    5912,
    5921,
    5932,
    5941,
    5942,
    5947,
    5970,
    5977,
    6300,
    7011,
    7210,
    7230,
    7276,
    7349,
    7393,
    7531,
    7538,
    7542,
    7549,
    7801,
    7802,
    7832,
    7922,
    7995,
    7996,
    8011,
    8021,
    8041,
    8043,
    8049,
    8062,
    8099,
    8111,
    8931,
    9402,
]

INDUSTRY_RANGES: tuple[tuple[int, int, str], ...] = (
    (0, 1499, "AGRICULTURAL"),
    (1500, 2999, "CONTRACTED"),
    (3000, 3299, "AIRLINES"),
    (3300, 3499, "CAR_RENTAL"),
    (3500, 3999, "LODGING"),
    (4000, 4799, "TRANSPORTATION"),
    (4800, 4999, "UTILITIES"),
    (5000, 5599, "RETAIL"),
    (5600, 5699, "CLOTHING"),
    (5700, 7299, "MISC_STORES"),
    (7300, 7999, "BUSINESS"),
    (8000, 8999, "PROFESSIONAL"),
    (9000, 9999, "GOVERNMENT"),
)

CHIP_MAPPING = {
    "SWIPE TRANSACTION": "SWIPE",
    "CHIP TRANSACTION": "CHIP",
    "ONLINE TRANSACTION": "ONLINE",
}

ALL_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
    "PR",
    "VI",
    "GU",
    "AS",
    "MP",
    "XX",
    "ONLINE",
]

AMOUNT_THRESHOLDS = (10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0)


def _stable_hash(value: str, buckets: int) -> int:
    digest = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) % buckets


def _industry(mcc: int) -> str:
    for lo, hi, label in INDUSTRY_RANGES:
        if lo <= mcc <= hi:
            return label
    return "GENERAL"


def _amount_bin(amount: float) -> int:
    rank = 0
    for threshold in AMOUNT_THRESHOLDS:
        if amount >= threshold:
            rank += 1
    return rank


@register_tokenizer("transaction_field_tokenizer")
@dataclass
class TransactionFieldTokenizer:
    """Fixed-vocab field tokenizer with global offsets.

    `fields` names the emitted fields in order; the vocabulary holds only
    their tokens. The default is all of `FIELD_ORDER`.
    """

    merchant_hash_size: int = 2000
    max_cust: int = 2999
    fields: tuple[str, ...] = field(default=FIELD_ORDER)
    name: str = "transaction_field_tokenizer"
    layout: str = LAYOUT_FLAT_FIELDS
    support_tier: str = "reference"
    requires_flat_fields = True
    """Reads the 12 transaction fields; adapters drop or alias other sources."""
    uses_profiles = False
    feature_keys: ClassVar[tuple[str, ...]] = ("fields",)
    """The `features` keys `from_config` reads (`ConfiguredTokenizer`)."""
    token_cache_format: ClassVar[str] = "flat_fields"

    @classmethod
    def from_config(cls, features: FeaturesConfig, data: DataConfig) -> Self:
        del data
        return cls() if features.fields is None else cls(fields=tuple(features.fields))

    @classmethod
    def event_width(cls, features: FeaturesConfig) -> int:
        """One token per field and one boundary token, `BOS` or `SEP` (`FixedWidthTokenizer`)."""
        return len(features.fields or FIELD_ORDER) + 1

    def __post_init__(self) -> None:
        self.fields = tuple(self.fields)
        unknown = [f for f in self.fields if f not in FIELD_ORDER]
        if unknown or not self.fields or len(set(self.fields)) != len(self.fields):
            raise UnsupportedCombinationError(
                f"fields must be distinct names from {list(FIELD_ORDER)}; got {list(self.fields)}."
            )
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: dict[int, str] = dict(enumerate(SPECIAL_TOKENS))
        self._field_ranges: list[tuple[int, int]] = []
        for name in self.fields:
            start = len(self.token_to_id)
            self._add_field(name)
            self._field_ranges.append((start, len(self.token_to_id)))

    def _add_field(self, name: str) -> None:
        if name == "AMT":
            self._add_range("AMT", 0, 6, 0)
        elif name == "MERCH":
            self._add_range("MERCH", 0, self.merchant_hash_size - 1, 0)
        elif name == "CAT":
            for label in [rng[2] for rng in INDUSTRY_RANGES] + ["GENERAL"]:
                self._add_token(f"CAT_{label}")
        elif name == "MCC":
            for mcc in KNOWN_MCCS:
                self._add_token(f"MCC_{mcc}")
            self._add_token("MCC_-1")
        elif name == "HOUR":
            self._add_range("HOUR", 0, 23, 2)
        elif name == "DOW":
            self._add_range("DOW", 0, 6, 0)
        elif name == "MONTH":
            self._add_range("MONTH", 1, 12, 2)
        elif name == "CARD":
            self._add_range("CARD", 0, 9, 0)
        elif name == "CHIP":
            for chip in ("SWIPE", "CHIP", "ONLINE", "UNK"):
                self._add_token(f"CHIP_{chip}")
        elif name == "ZIP3":
            self._add_range("ZIP3", 0, 999, 3)
        elif name == "STATE":
            for state in ALL_STATES:
                self._add_token(f"STATE_{state}")
        else:
            self._add_range("CUST", 0, self.max_cust, 0)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def _add_range(self, prefix: str, min_val: int, max_val: int, pad: int) -> None:
        for value in range(min_val, max_val + 1):
            self._add_token(f"{prefix}_{value:0{pad}d}" if pad else f"{prefix}_{value}")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<sep>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def lookup(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def field_vocabulary(self) -> FieldVocabulary:
        """Each field's id range, for objectives that predict a field from its own values."""
        return FieldVocabulary(
            vocab_size=self.vocab_size,
            field_names=self.fields,
            field_ranges=tuple(self._field_ranges),
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            sep_id=self.sep_id,
            eos_id=self.eos_id,
        )

    def fit(self, sequences: Sequence[TransactionSequence]) -> TransactionFieldTokenizer:
        """Vocab is data-independent; fit is a no-op."""
        return self

    def encode(self, sequence: TransactionSequence) -> EncodedFlatSequence:
        self.prepare_merchants([sequence])
        ids = [self.bos_id]
        for index, event in enumerate(sequence.events):
            if index > 0:
                ids.append(self.sep_id)
            ids.extend(self.lookup(tok) for tok in self.event_tokens(event))
        ids.append(self.eos_id)
        return EncodedFlatSequence(
            entity_id=sequence.entity_id,
            sequence_id=sequence.sequence_id or sequence.entity_id,
            token_ids=tuple(ids),
            layout=self.layout,
        )

    def pad_encoded(
        self,
        encoded: EncodedFlatSequence,
        *,
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        ids = self._truncate(list(encoded.token_ids), max_length)
        attn = [1] * len(ids)
        if len(ids) < max_length:
            pad = max_length - len(ids)
            ids.extend([self.pad_id] * pad)
            attn.extend([0] * pad)
        return ids, attn

    def _truncate(self, ids: list[int], max_length: int) -> list[int]:
        """Keep `<bos>` and the most recent whole events that fit, ending at `<eos>`.

        A history is scored at its end, so the events dropped are the oldest.
        """
        if len(ids) <= max_length:
            return ids
        start = len(ids) - (max_length - 1)
        while start < len(ids) and ids[start - 1] != self.sep_id:
            start += 1
        return [self.bos_id, *ids[start:]]

    def _merchant_bucket(self, merchant: str) -> int:
        """Merchant hash bucket. Variants override this and `prepare_merchants`."""
        return _stable_hash(merchant, self.merchant_hash_size)

    def prepare_merchants(self, sequences: Sequence[TransactionSequence]) -> None:
        """Called once per batch before tokenizing, so a variant can hash in bulk."""

    @staticmethod
    def merchant_key(event: TransactionEvent) -> str:
        return str(event.categorical.get("merchant_name", "UNK")).upper()

    def event_tokens(self, event: TransactionEvent) -> list[str]:
        ts = ensure_utc(event.event_timestamp)
        amount = float(event.numeric.get("amount", 0.0))
        merchant = self.merchant_key(event)
        mcc = int(float(event.categorical.get("mcc", event.numeric.get("mcc", -1))))
        card = int(event.numeric.get("card", 0))
        user = int(event.numeric.get("user", 0))
        zip_raw = str(int(event.numeric.get("zip", 0))).zfill(5)
        state = str(event.categorical.get("merchant_state", "XX")).upper().strip() or "XX"
        if state not in ALL_STATES:
            state = "XX"
        chip_raw = str(event.categorical.get("use_chip", "")).upper()
        chip = CHIP_MAPPING.get(chip_raw, "UNK")
        tokens = {
            "AMT": f"AMT_{_amount_bin(amount)}",
            "MERCH": f"MERCH_{self._merchant_bucket(merchant)}",
            "CAT": f"CAT_{_industry(mcc)}",
            "MCC": f"MCC_{mcc if mcc in KNOWN_MCCS else -1}",
            "HOUR": f"HOUR_{ts.hour:02d}",
            "DOW": f"DOW_{ts.weekday()}",
            "MONTH": f"MONTH_{ts.month:02d}",
            "CARD": f"CARD_{max(0, min(card, 9))}",
            "CHIP": f"CHIP_{chip}",
            "ZIP3": f"ZIP3_{int(zip_raw[:3]):03d}",
            "STATE": f"STATE_{state}",
            "CUST": f"CUST_{max(0, min(user, self.max_cust))}",
        }
        return [tokens[name] for name in self.fields]

    def encode_sequence(
        self,
        sequence: TransactionSequence,
        *,
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        return self.pad_encoded(self.encode(sequence), max_length=max_length)

    def collate(
        self,
        sequences: Sequence[TransactionSequence],
        *,
        max_length: int,
    ) -> TransactionBatch:
        self.prepare_merchants(sequences)
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for seq in sequences:
            tok_ids, attn = self.encode_sequence(seq, max_length=max_length)
            ids.append(tok_ids)
            masks.append(attn)
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(masks, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return TransactionBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "tokenizer": self.name,
            "layout": self.layout,
            "support_tier": self.support_tier,
            "field_order": list(self.fields),
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "merchant_hash_size": self.merchant_hash_size,
        }

    def dumps(self) -> str:
        return json.dumps(self.manifest(), indent=2, sort_keys=True)


def event_time_features(event: TransactionEvent) -> dict[str, float]:
    ts: datetime = ensure_utc(event.event_timestamp)
    return {"hour": float(ts.hour), "dow": float(ts.weekday()), "month": float(ts.month)}
