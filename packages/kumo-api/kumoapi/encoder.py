# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

# flake8: noqa

import warnings
from abc import ABC, abstractmethod
from dataclasses import field, fields
from typing import Any, Dict, Literal, Optional, Set, Union, get_args

from pydantic import PositiveInt
from pydantic.dataclasses import dataclass

from kumoapi.common import StrEnum
from kumoapi.typing import _FORBID_EXTRA_CONFIG, ColStatType, Stype

warnings.filterwarnings('ignore', "fields may not start with an underscore")


class NAStrategy(StrEnum):
    r"""Kumo-supported null value imputation strategies."""
    ZERO = 'zero'  # Fill missing values with zeros.
    MEAN = 'mean'  # Fill missing values with mean.
    SEPARATE = 'separate'  # Regard missing values as a separate category.
    MOST_FREQUENT = 'most_frequent'  # Fill with most frequent value.

    RAISE = 'raise'  # Backward compatibility. Do not use.

    def __repr__(self) -> str:
        return self.value


class Scaler(StrEnum):
    r"""Kumo-supported numerical value scaling strategies."""
    #: Scale values with z-score normalization.
    #: Equivalent to `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
    STANDARD = 'standard'

    #: Scale values by subtracting the minimum value and dividing by the range.
    #: Equivalent to `MinMaxScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_.
    MINMAX = 'minmax'

    #: Scale values by subtracting the median and dividing by the range between
    #: the first and third quartiles. Equivalent to `RobustScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html>`_.
    ROBUST = 'robust'


class EmbeddingScaler(StrEnum):
    r"""Kumo-supported scaling strategies for numerical sequences and/or
    embeddings."""
    #: Scale values with z-score normalization computed over each embedding.
    STANDARD = 'standard'

    #: Scale values with CosineNormalization, dividing by the embedding norm.
    #: Taken from `Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks <https://arxiv.org/abs/1702.05870>`
    COSINE = 'cosine'

    def __repr__(self) -> str:
        return self.value


@dataclass(config=_FORBID_EXTRA_CONFIG)
class Encoder(ABC):
    def __post_init__(self) -> None:
        if hasattr(self, 'na_strategy'):
            self.na_strategy = NAStrategy(self.na_strategy)

        # Let `pydantic` break on invalid `_target_` names. Needed because
        # `pydantic` doesn't check for type-safety in underscore attributes.
        target = getattr(self, '_target_', None)
        if target is not None:
            f = [f for f in fields(self.__class__) if f.name == '_target_'][0]
            if target not in get_args(f.type):
                raise ValueError(f"Unsupported `_target_={target}` for "
                                 f"'{self.__class__.__name__}' encoder")

    @property
    @abstractmethod
    def supported_stypes(self) -> Set[Stype]:
        pass

    @property
    @abstractmethod
    def required_stats(self) -> Set[ColStatType]:
        pass


@dataclass(config=_FORBID_EXTRA_CONFIG)
class Null(Encoder):
    r"""A :class:`Null` encoder skips encoding its corresponding column."""
    name: Literal['Null'] = field(default='Null', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.Null'] = field(
        default='kumoapi.encoder.Null',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return set(Stype)

    @property
    def required_stats(self) -> Set[ColStatType]:
        return set()


@dataclass(config=_FORBID_EXTRA_CONFIG)
class Numerical(Encoder):
    r"""A :class:`Numerical` encoder encodes its corresponding numerical
    column with a normalization specified by :obj:`scaler` and strategy for
    null value imputation specified by :obj:`na_strategy`."""
    #: The specified :obj:`~kumoapi.encoder.Scaler`, one of "standard",
    #: "minmax", or "robust".
    scaler: Optional[Scaler] = None

    #: The specified null value imputation strategy.
    na_strategy: Literal[
        NAStrategy.ZERO,
        NAStrategy.MEAN,
        NAStrategy.SEPARATE,
        NAStrategy.RAISE,
    ] = NAStrategy.MEAN

    name: Literal['Numerical'] = field(default='Numerical', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.Numerical'] = field(
        default='kumoapi.encoder.Numerical',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.numerical}

    @property
    def required_stats(self) -> Set[ColStatType]:
        stats = set()
        if self.na_strategy is NAStrategy.MEAN:
            stats |= {ColStatType.MEAN}
        if self.scaler is Scaler.STANDARD:
            stats |= {ColStatType.MEAN, ColStatType.STD}
        elif self.scaler is Scaler.MINMAX:
            stats |= {ColStatType.MIN, ColStatType.MAX}
        elif self.scaler is Scaler.ROBUST:
            stats |= {ColStatType.QUANTILES}
        return stats


@dataclass(config=_FORBID_EXTRA_CONFIG)
class MaxLogNumerical(Encoder):
    r"""A :class:`MaxLogNumerical` encoder encodes its corresponding numerical
    column, after applying the transformation

    .. math::
        \log \left( \frac{\text{feature} - (\text{min} - 1)}{1.0} \right)

    and using a strategy for null value imputation specified by
    :obj:`na_strategy`."""
    #: The specified null value imputation strategy.
    na_strategy: Literal[
        NAStrategy.ZERO,
        NAStrategy.MEAN,
        NAStrategy.SEPARATE,
        NAStrategy.RAISE,
    ] = NAStrategy.MEAN

    name: Literal['MaxLogNumerical'] = field(
        default='MaxLogNumerical',
        repr=False,
    )
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.MaxLogNumerical'] = field(
        default='kumoapi.encoder.MaxLogNumerical',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.numerical}

    @property
    def required_stats(self) -> Set[ColStatType]:
        if self.na_strategy is NAStrategy.MEAN:
            return {ColStatType.MIN, ColStatType.MEAN}
        return {ColStatType.MIN}


@dataclass(config=_FORBID_EXTRA_CONFIG)
class MinLogNumerical(Encoder):
    r"""A :class:`MinLogNumerical` encoder encodes its corresponding numerical
    column, after applying the transformation

    .. math::
        \log \left( \frac{\text{feature} - (\text{max} + 1)}{-1.0} \right)

    and using a strategy for null value imputation specified by
    :obj:`na_strategy`."""
    #: The specified null value imputation strategy.
    na_strategy: Literal[
        NAStrategy.ZERO,
        NAStrategy.MEAN,
        NAStrategy.SEPARATE,
        NAStrategy.RAISE,
    ] = NAStrategy.MEAN

    name: Literal['MinLogNumerical'] = field(
        default='MinLogNumerical',
        repr=False,
    )
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.MinLogNumerical'] = field(
        default='kumoapi.encoder.MinLogNumerical',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.numerical}

    @property
    def required_stats(self) -> Set[ColStatType]:
        if self.na_strategy is NAStrategy.MEAN:
            return {ColStatType.MAX, ColStatType.MEAN}
        return {ColStatType.MAX}


@dataclass(config=_FORBID_EXTRA_CONFIG)
class Index(Encoder):
    r"""An :class:`Index` encoder encodes its corresponding categorical column
    by assigning each unique value with frequency above :obj:`min_occ` to an
    embedding of size :obj:`channels` from the model plan. Values below this
    frequency are all collapsed to the same embedding."""
    #: The minimum frequency of distinct values.
    min_occ: PositiveInt = 1
    #: The specified null value imputation strategy.
    na_strategy: Literal[
        NAStrategy.ZERO,
        NAStrategy.SEPARATE,
        NAStrategy.MOST_FREQUENT,
        NAStrategy.RAISE,
    ] = NAStrategy.SEPARATE

    name: Literal['Index'] = field(default='Index', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.Index'] = field(
        default='kumoapi.encoder.Index',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.categorical, Stype.ID}

    @property
    def required_stats(self) -> Set[ColStatType]:
        return {ColStatType.CATEGORY_COUNTS}


@dataclass(config=_FORBID_EXTRA_CONFIG)
class Hash(Encoder):
    r"""A :class:`Hash` encoder encodes its corresponding categorical column
    by hashing each value to range :obj:`[0..num_components]`, and using this
    hashed value to determine the corresponding embedding (with size
    :obj:`channels` from the model plan)."""
    #: The number of distinct categories after hashing.
    num_components: PositiveInt
    #: The specified null value imputation strategy.
    na_strategy: Literal[
        NAStrategy.SEPARATE,
        NAStrategy.MOST_FREQUENT,
    ] = NAStrategy.SEPARATE

    name: Literal['Hash'] = field(default='Hash', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.Hash'] = field(
        default='kumoapi.encoder.Hash',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.categorical, Stype.ID}

    @property
    def required_stats(self) -> Set[ColStatType]:
        return {ColStatType.CATEGORY_COUNTS}


@dataclass(config=_FORBID_EXTRA_CONFIG)
class MultiCategorical(Encoder):
    r"""A :class:`MultiCategorical` encoder encodes its corresponding
    multicategorical column by treating each categorical value independently,
    and fusing the results."""
    #: The minimum frequency of distinct values.
    min_occ: PositiveInt = 1
    #: The specified null value imputation strategy.
    na_strategy: Literal[
        NAStrategy.ZERO,
        NAStrategy.SEPARATE,
        NAStrategy.MOST_FREQUENT,
    ] = NAStrategy.ZERO

    name: Literal['MultiCategorical'] = field(
        default='MultiCategorical',
        repr=False,
    )
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.MultiCategorical'] = field(
        default='kumoapi.encoder.MultiCategorical',
        repr=False,
    )

    @property
    def supported_stypes(cls) -> Set[Stype]:
        return {Stype.multicategorical}

    @property
    def required_stats(self) -> Set[ColStatType]:
        return {
            ColStatType.MULTI_CATEGORY_COUNTS,
            ColStatType.MULTI_CATEGORIES_SEPARATOR,
        }


@dataclass(config=_FORBID_EXTRA_CONFIG)
class TextEmbedding(Encoder):
    r"""A :class:`TextEmbedding` encoder embeds text with a word-embedding
    model registered under :obj:`model` in the embedding model store passed
    to the trainer. ``"glove.6B-50d"`` is the default model."""
    #: The name the embedding model is registered under.
    model: str = 'glove.6B-50d'
    na_strategy: Literal[NAStrategy.ZERO] = field(  # No need to show/modify.
        default=NAStrategy.ZERO,
        repr=False,
    )

    name: Literal['TextEmbedding'] = field(default='TextEmbedding', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)
    # The model it was trained with: family, dim, fingerprint and label.
    _identity: Optional[Dict[str, Any]] = field(default=None, repr=False)

    _target_: Literal['kumoapi.encoder.TextEmbedding'] = field(
        default='kumoapi.encoder.TextEmbedding',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.text}

    @property
    def required_stats(self) -> Set[ColStatType]:
        return set()


@dataclass(config=_FORBID_EXTRA_CONFIG)
class NumericalList(Encoder):
    r"""A :class:`NumericalList` encoder encodes numerical sequences by
    treating these sequences as input features with a normalization specified
    by :obj:`scaler`."""
    #: The specified :obj:`~kumoapi.encoder.EmbeddingScaler`, one of "standard"
    #: or "cosine"
    scaler: Optional[EmbeddingScaler] = None

    na_strategy: Literal[NAStrategy.ZERO] = field(  # No need to show/modify.
        default=NAStrategy.ZERO,
        repr=False,
    )
    name: Literal['NumericalList'] = field(default='NumericalList', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.NumericalList'] = field(
        default='kumoapi.encoder.NumericalList',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.sequence}

    @property
    def required_stats(self) -> Set[ColStatType]:
        stats = {
            ColStatType.SEQUENCE_MIN_LENGTH,
            ColStatType.SEQUENCE_MAX_LENGTH,
        }
        if self.scaler is EmbeddingScaler.STANDARD:
            stats |= {ColStatType.SEQUENCE_MEAN, ColStatType.SEQUENCE_STD}
        return stats


@dataclass(repr=False, config=_FORBID_EXTRA_CONFIG)
class Datetime(Encoder):
    r"""A :class:`Datetime` encoder encodes a date or time value, representing
    it with various user-specified granularities."""
    #: Whether to include minute-granularity features.
    include_minute: bool = True
    #: Whether to include hour-granularity features.
    include_hour: bool = True
    #: Whether to include week-granularity features.
    include_day_of_week: bool = True
    #: Whether to include month-granularity features.
    include_day_of_month: bool = True
    #: Whether to include day-of-year-granularity features.
    include_day_of_year: bool = True
    #: Whether to include year-granularity features.
    include_year: bool = True
    num_year_periods: Optional[PositiveInt] = None  # TODO: document?
    na_strategy: Literal[NAStrategy.ZERO] = field(  # No need to show/modify.
        default=NAStrategy.ZERO,
        repr=False,
    )

    name: Literal['Datetime'] = field(default='Datetime', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    _target_: Literal['kumoapi.encoder.Datetime'] = field(
        default='kumoapi.encoder.Datetime',
        repr=False,
    )

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.timestamp}

    @property
    def required_stats(self) -> Set[ColStatType]:
        stats = set()
        if self.include_year and self.num_year_periods is None:
            stats |= {ColStatType.MIN, ColStatType.MAX}
        return stats

    def __repr__(self) -> str:
        kwargs = {  # Only show arguments that diverge from the default:
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.repr and getattr(self, f.name) != f.default
        }
        reprs = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
        return f'{self.__class__.__name__}({reprs})'


@dataclass(config=_FORBID_EXTRA_CONFIG)
class Image(Encoder):
    r"""A :class:`Image` encoder is used to process image URLs."""
    name: Literal['Image'] = field(default='Image', repr=False)
    _stats: Dict[ColStatType, Any] = field(default_factory=dict, repr=False)

    @property
    def supported_stypes(self) -> Set[Stype]:
        return {Stype.image}

    @property
    def required_stats(self) -> Set[ColStatType]:
        return set()


EncoderType = Union[
    Null,
    Numerical,
    MaxLogNumerical,
    MinLogNumerical,
    Index,
    Hash,
    MultiCategorical,
    TextEmbedding,
    NumericalList,
    Datetime,
    Image,
]
