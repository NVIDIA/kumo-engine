# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import builtins
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from kumoapi.encoder import TextEmbedding
from kumoapi.typing import Stype

from kumo_core.encoders.tensor_mapper import SequenceTensorMapper
from kumo_core.embedding_models import resolve_encoder
from kumo_core.encoders.text import normalize_text


def mean(ser: pd.Series) -> float:
    return float(np.mean(ser)) if len(ser) > 0 else 0.0


def std(ser: pd.Series) -> float:
    return float(np.std(ser)) if len(ser) > 1 else 1.0


def min(ser: pd.Series) -> float:
    return float(np.min(ser)) if len(ser) > 0 else 0.0


def max(ser: pd.Series) -> float:
    return float(np.max(ser)) if len(ser) > 0 else 0.0


def quantiles(ser: pd.Series) -> tuple[float, float, float]:
    if len(ser) == 0:
        return (0.0, 0.0, 0.0)
    return tuple(np.quantile(ser, [0.25, 0.5, 0.75]).tolist())


def kurtosis(ser: pd.Series) -> float:
    kurtosis = float(ser.kurtosis())
    return kurtosis if not math.isnan(kurtosis) else 0.0


def category_count(ser: pd.Series) -> list[tuple[Any, int]]:
    counts = ser.value_counts()
    counts = counts.rename_axis('category').reset_index(name='count')
    counts = counts.sort_values(
        by=['count', 'category'],
        ascending=[False, True],
        kind='mergesort',
    )
    return counts.to_numpy().tolist()


def sep_multi(ser: pd.Series) -> str | None:
    index = ser.first_valid_index()
    if index is None or pd.api.types.is_list_like(ser[index]):
        return None

    text = '\n'.join(ser.iloc[:1000].fillna(''))
    white_list = {',', ';', ':', '|', '\t'}
    candidates: dict[str, int] = defaultdict(int)
    for char in text:
        if char in white_list:
            candidates[char] += 1

    if len(candidates) > 0:
        return builtins.max(candidates, key=candidates.get)  # type: ignore

    return None


def num_unique_multi(
    ser: pd.Series,
    sep: str | None,
) -> int:
    sep = sep_multi(ser) if sep == '???' else sep
    ser = ser.astype(str).str.split(sep) if sep is not None else ser
    ser = ser.explode()
    if sep is None:
        ser = ser.dropna()
    ser = ser[ser != '']
    return len(ser.unique())


def sequence_min_length(ser: pd.Series) -> int:
    ser = SequenceTensorMapper.to_list_like(ser)
    return int(np.min(ser.apply(len))) if len(ser) > 0 else 0


def sequence_max_length(ser: pd.Series) -> int:
    ser = SequenceTensorMapper.to_list_like(ser)
    return int(np.max(ser.apply(len))) if len(ser) > 0 else 0


def sequence_mean(ser: pd.Series) -> float:
    ser = SequenceTensorMapper.to_list_like(ser)
    return float(np.mean(np.concatenate(ser.values))) if len(ser) > 0 else 0.0


def sequence_std(ser: pd.Series) -> float:
    ser = SequenceTensorMapper.to_list_like(ser)
    return float(np.std(np.concatenate(ser.values))) if len(ser) > 0 else 1.0


def avg_tokens(ser: pd.Series) -> float:
    ser = normalize_text(ser)
    return float(np.mean(ser.map(len))) if len(ser) > 0 else 0.0


def max_tokens(ser: pd.Series) -> int:
    ser = normalize_text(ser)
    return int(np.max(ser.map(len))) if len(ser) > 0 else 0


def avg_non_char(ser: pd.Series) -> float:
    ser = normalize_text(ser).explode().dropna().iloc[:1000]
    non_char_tokens = total_tokens = 0
    for token in ser:
        if not re.fullmatch('[a-zA-Z]+', token):
            non_char_tokens += 1
        total_tokens += 1
    return non_char_tokens / total_tokens if total_tokens > 0 else 0


def glove_overlap(ser: pd.Series) -> float:
    ser = normalize_text(ser)
    num_tokens = int(ser.map(len).sum())
    if num_tokens == 0:
        return 1.0
    _, index = resolve_encoder(TextEmbedding()).tokenize(ser)
    return index.size(0) / num_tokens


def notna(ser: pd.Series) -> np.ndarray:
    if pd.api.types.is_float_dtype(ser):
        return np.isfinite(ser.to_numpy())

    notna_mask = ser.notna()

    if pd.api.types.infer_dtype(ser) == 'string':
        notna_mask &= ser != ''

    return notna_mask.to_numpy()


@dataclass
class Stats:
    approx: bool
    count: int
    na_frac: float


@dataclass
class NumericalStats(Stats):
    mean: float
    std: float
    min: float
    max: float
    kurtosis: float
    quantiles: tuple[float, float, float]

    @property
    def median(self) -> float:
        return self.quantiles[1]


@dataclass
class CategoricalStats(Stats):
    category_count: list[tuple[Any, int]]

    @property
    def num_unique(self) -> int:
        return len(self.category_count)


@dataclass
class MultiCategoricalStats(Stats):
    sep: str | None
    num_unique: int


@dataclass
class SequenceStats(Stats):
    min_length: int
    max_length: int
    mean: float
    std: float


@dataclass
class TextStats(Stats):
    avg_tokens: float
    max_tokens: int
    avg_non_char: float
    glove_overlap: float


def get_stats(ser: pd.Series, stype: Stype, approx: bool = False) -> Stats:
    if not approx:
        notna_mask = notna(ser)
        na_frac = 1.0 - int(np.sum(notna_mask)) / len(ser)
        ser = ser[notna_mask]
    else:
        num_approx = 1_000 if stype == Stype.text else 10_000

        # Iterate over `ser` in chunks until we have found 10k valid rows:
        sers, notna_masks = [], []
        for start in range(0, len(ser), num_approx):
            _ser = ser.iloc[start:start + num_approx]
            notna_mask = notna(_ser)
            sers.append(_ser[notna_mask])
            notna_masks.append(notna_mask)
            if sum([len(s) for s in sers]) >= num_approx:
                break

        ser = pd.concat(sers)
        na_frac = 1.0 - (sum([int(np.sum(m)) for m in notna_masks]) /
                         sum([len(m) for m in notna_masks]))
        ser = ser.iloc[:num_approx]

    if stype == Stype.numerical:
        return NumericalStats(
            approx=approx,
            count=len(ser),
            na_frac=na_frac,
            mean=mean(ser),
            std=std(ser),
            min=min(ser),
            max=max(ser),
            kurtosis=kurtosis(ser),
            quantiles=quantiles(ser),
        )

    if stype in {Stype.categorical, Stype.ID}:
        return CategoricalStats(
            approx=approx,
            count=len(ser),
            na_frac=na_frac,
            category_count=category_count(ser),
        )

    if stype == Stype.multicategorical:
        sep = sep_multi(ser)
        return MultiCategoricalStats(
            approx=approx,
            count=len(ser),
            na_frac=na_frac,
            sep=sep,
            num_unique=num_unique_multi(ser, sep),
        )

    if stype == Stype.sequence:
        ser = SequenceTensorMapper.to_list_like(ser)
        return SequenceStats(
            approx=approx,
            count=len(ser),
            na_frac=na_frac,
            min_length=sequence_min_length(ser),
            max_length=sequence_max_length(ser),
            mean=sequence_mean(ser),
            std=sequence_std(ser),
        )

    if stype == Stype.text:
        ser = normalize_text(ser)
        return TextStats(
            approx=approx,
            count=len(ser),
            na_frac=na_frac,
            avg_tokens=avg_tokens(ser),
            max_tokens=max_tokens(ser),
            avg_non_char=avg_non_char(ser),
            glove_overlap=glove_overlap(ser),
        )

    return Stats(approx=approx, count=len(ser), na_frac=na_frac)
