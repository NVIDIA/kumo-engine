# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder-stat fixtures shared by core and service tests."""

import kumo_core.features.stats
import numpy as np
import pandas as pd
from kumo_core.encoders.tensor_mapper import SequenceTensorMapper
from kumo_core.features.time import to_datetime
from kumo_core.types.features import TIMESTAMP_PREFIX
from kumoapi.encoder import Datetime, Encoder
from kumoapi.typing import ColStatType


def fit_encoder_stats(
    encoder: Encoder,
    series: pd.Series,
    force_refresh: bool = False,
) -> Encoder:
    """Populate encoder statistics for synthetic test data.

    Args:
        encoder: Encoder whose required statistics are populated in place.
        series: Synthetic column values used to construct the test fixture.
        force_refresh: Whether to discard previously populated statistics.

    Returns:
        The encoder with all required statistics populated.

    Raises:
        ValueError: If the encoder requires an unsupported statistic.
    """
    if force_refresh:
        encoder._stats = {}

    missing_stats = encoder.required_stats - set(encoder._stats.keys())
    if len(missing_stats) == 0:
        return encoder

    if isinstance(encoder, Datetime):
        series = to_datetime(series)

    notna_mask = kumo_core.features.stats.notna(series)
    series = series[notna_mask]

    if pd.api.types.is_timedelta64_dtype(series):
        series = series.dt.total_seconds()

    # Ensure that we first infer the separator:
    if ColStatType.MULTI_CATEGORIES_SEPARATOR in missing_stats:
        sep = kumo_core.features.stats.sep_multi(series)
        encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] = sep

    for stat in encoder.required_stats - set(encoder._stats.keys()):
        if stat == ColStatType.CATEGORY_COUNTS:
            encoder._stats[stat] = kumo_core.features.stats.category_count(series)
        elif stat == ColStatType.MULTI_CATEGORY_COUNTS:
            sep = encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR]
            categories = (
                series.astype(str).str.split(sep) if sep is not None else series
            )
            categories = categories.explode()
            encoder._stats[stat] = kumo_core.features.stats.category_count(categories)
        elif stat == ColStatType.MEAN:
            encoder._stats[stat] = kumo_core.features.stats.mean(series)
        elif stat == ColStatType.STD:
            encoder._stats[stat] = kumo_core.features.stats.std(series)
        elif stat == ColStatType.MIN:
            if isinstance(encoder, Datetime):
                value = np.min(series) if len(series) > 0 else pd.Timestamp("1970")
                encoder._stats[stat] = f"{TIMESTAMP_PREFIX}{value}"
            else:
                encoder._stats[stat] = kumo_core.features.stats.min(series)
        elif stat == ColStatType.MAX:
            if isinstance(encoder, Datetime):
                value = np.max(series) if len(series) > 0 else pd.Timestamp("1970")
                encoder._stats[stat] = f"{TIMESTAMP_PREFIX}{value}"
            else:
                encoder._stats[stat] = kumo_core.features.stats.max(series)
        elif stat == ColStatType.QUANTILES:
            encoder._stats[stat] = kumo_core.features.stats.quantiles(series)
        elif stat == ColStatType.SEQUENCE_MIN_LENGTH:
            series = SequenceTensorMapper.to_list_like(series)
            encoder._stats[stat] = kumo_core.features.stats.sequence_min_length(series)
        elif stat == ColStatType.SEQUENCE_MAX_LENGTH:
            series = SequenceTensorMapper.to_list_like(series)
            encoder._stats[stat] = kumo_core.features.stats.sequence_max_length(series)
        elif stat == ColStatType.SEQUENCE_MEAN:
            series = SequenceTensorMapper.to_list_like(series)
            encoder._stats[stat] = kumo_core.features.stats.sequence_mean(series)
        elif stat == ColStatType.SEQUENCE_STD:
            series = SequenceTensorMapper.to_list_like(series)
            encoder._stats[stat] = kumo_core.features.stats.sequence_std(series)
        else:
            raise ValueError(f"Cannot compute stat '{stat.value}'")

    return encoder
