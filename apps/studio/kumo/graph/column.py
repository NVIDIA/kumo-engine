# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from kumo_core import encoder_selection
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumo_core.types import ColStatType, EdgeStatType
from kumo_core.utils.mixins import CastMixin
from kumo_core.utils.names import fully_qualified_table_name, get_table_name
from kumo_core.utils.registry import register
from kumoapi.encoder import Datetime, Encoder, Image
from kumoapi.encoder import Index as _Index
from kumoapi.encoder import Null
from kumoapi.typing import ColStatType as ApiColStatType
from kumoapi.typing import Dtype, Stype

from io_utils import logging
from kumo.config import (
    BaseColumnConfig,
    BoolColumnConfig,
    ColumnConfig,
    ConfigMixin,
    ForeignKeyConfig,
    PrimaryKeyConfig,
    TimeColumnConfig,
)
from kumo.graph.mixin import ConnectorMixin, TableMixin
from kumo.stats import Stats

logger = logging.getLogger(__name__)

# Kept as a compatibility export for callers importing it from this module.
Index = _Index


@register(data_cls=BaseColumnConfig)
class BaseColumn(CastMixin, ConfigMixin, ConnectorMixin, TableMixin):
    # NOTE For `BaseColumn` types, we allow auto-casting of arguments via the
    # `CastMixin`, i.e. these types are treated the same:
    # * `Column(name='AMOUNT')`
    # * `Column(name='AMOUNT', dtype=Dtype.float)`
    # * `'AMOUNT'
    # * ['AMOUNT', Dtype.float]` or `('AMOUNT', Dtype.float)`
    # * `{'name': 'AMOUNT', 'dtype': Dtype.float}`
    r"""A base class for representing columns.

    Args:
        name (str): The name of the column.
        stype (Stype, optional): The semantic type of the column.
            (default: :obj:`None`)
        dtype (Dtype, optional): The data type of the column.
            (default: :obj:`None`)
        na_values: (list, optional): A list of values to interpret as
            :obj:`N/A`.
        metadata (dict, optional): A dict of key to column metadata. Could be
            used to store timestamp format string for date/timestamp column.
            (default: obj:`None`)
        timestamp_unit (str, optional): Denotes the time unit used for
            capturing time since epoch. This is only going to be set for
            columns which denote time since epoch.
        multicategorical_delimiter (str, optional): The proven delimiter for
            a scalar-string multicategorical column. Native list columns do
            not use a delimiter.
    """
    def __init__(
        self,
        name: str,
        stype: Optional[Stype] = None,
        dtype: Optional[Dtype] = None,
        na_values: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp_unit: Optional[str] = None,
        multicategorical_delimiter: Optional[str] = None,
        **kwargs,
    ):
        self._name = name
        self.stype = stype
        self.dtype = dtype
        self.na_values = na_values
        self.metadata = metadata or {}
        self.timestamp_unit = timestamp_unit
        self.multicategorical_delimiter = multicategorical_delimiter

    def to_base(self) -> 'BaseColumn':
        return BaseColumn(
            name=self.name,
            stype=self.stype,
            dtype=self.dtype,
            na_values=self.na_values,
            metadata=self.metadata,
            timestamp_unit=self.timestamp_unit,
            multicategorical_delimiter=self.multicategorical_delimiter,
        )

    def base_config(self) -> BaseColumnConfig:
        return self.to_base().config()

    @property
    def name(self) -> str:
        return self._name

    @property
    def qualified_name(self) -> str:
        """Return the column name qualified by its table when available."""
        return (f"{get_table_name(self.table_name)}.{self.name}"
                if self.has_table else self.name)

    @property
    def pandas(self):
        r"""Returns a :class:`pandas` view to access the data of the underlying
        :obj:`data_source`:

        .. code-block:: python

            graph[table_name, col_name].pandas()  # all data
            graph[table_name, col_name].pandas[:1000]  # first 1000 rows
        """
        from kumo.graph.pandas_view import PandasView
        return PandasView(self.connector, self.table_name, cols=[self],
                          return_series=True)

    @property
    def pandas_indexed(self):
        r"""Returns a :class:`pandas` view to access the data of the underlying
        :obj:`data_source` including the index (NODE_INDEX_COLUMN) column as
        a pandas DataFrame
        .. code-block:: python
            graph[table_name, col_name].pandas()  # all data
            graph[table_name, col_name].pandas[:1000]  # first 1000 rows
        """
        from kumo.graph.pandas_view import PandasView
        return PandasView(self.connector, self.table_name,
                          cols=[self, Column(name=NODE_INDEX_COLUMN)],
                          return_series=False)

    # ``INVALID_FRACTION`` and ``DROPPED_ROWS`` are vestigial cast-phase
    # stats not computed by the DG ingest pipeline. Excluded from
    # ``full_stats`` so ``Table.compute_exact_stats`` does not raise
    # ``UnsupportedInDiskgraphError`` on the DG path; downstream consumers
    # already tolerate absence (default-to-zero / try-except).
    _VESTIGIAL_FULL_STATS: List[ColStatType] = [
        ColStatType.INVALID_FRACTION,
        ColStatType.DROPPED_ROWS,
    ]

    @property
    def full_stats(self) -> List[ColStatType]:
        r"""A list of all supported statistics on this column."""
        if self.dtype is None:
            return []

        excluded = list(self._VESTIGIAL_FULL_STATS)
        # ``STRING_AVG_LEN`` is a string-length stat that the DG ingest
        # pipeline computes only for ``text`` columns; for categorical / ID
        # (string-backed) columns it is neither emitted by DG nor consumed by
        # any categorical/ID encoder. Requesting it there makes the strict
        # ``Table.compute_exact_stats`` DG-cache check raise on a stat that
        # will never be present. Exclude it outside of ``text`` (same class of
        # vestigial-request exclusion as INVALID_FRACTION / DROPPED_ROWS).
        if self.stype != Stype.text:
            excluded.append(ColStatType.STRING_AVG_LEN)

        return [
            stat for stat in list(ColStatType) if stat.supports(self.dtype)
            and stat.supports_stype(self.stype) and stat not in excluded
        ]

    @property
    def full_encoder_stats(self) -> List[ColStatType]:
        r"""A list of all supported statstics on this column used for encoder.
        """
        if self.dtype is None:
            return []

        return [
            stat for stat in list(ColStatType) if stat.supports(self.dtype)
            and stat.supports_stype(self.stype) and stat not in [
                ColStatType.HISTOGRAM,
                ColStatType.HISTOGRAM_HALF_MONTH,
                ColStatType.NUM_IRRATIONAL,
                ColStatType.SUMMARY_STATS_BY_CLASS,
            ]
        ]

    @property
    def minimal_stats(self) -> List[ColStatType]:
        r"""A list of all statistics that this column requires to be computed
        post-materialization."""
        out = [ColStatType.NUM_NA]
        if self.stype == Stype.timestamp:
            # Currently used in the UI only. TODO: refactor out
            out += [ColStatType.MIN, ColStatType.MAX]
        return out

    def sample(self, num_samples=100) -> pd.Series:
        r"""Returns a sample of values (default 100) for the column from the
        underlying :obj:`data_source`:

        .. code-block:: python

            graph[table_name, col_name].sample()  # 100 samples
        """
        df = self.connector.read_table_sample(
            self.table_name,
            cols=[self.base_config()],
            num_rows=num_samples,
        )

        return df[self.name]

    def __getitem__(self, idx: Any) -> pd.DataFrame:
        return self.pandas[idx]

    def update(self, obj: 'BaseColumn', override: bool = True) -> 'BaseColumn':
        for key in self.__dict__.keys():
            if key[0] == '_':  # Skip private attributes for now:
                continue
            value = getattr(obj, key, None)
            if value is not None:
                if override or getattr(self, key, None) is None:
                    setattr(self, key, value)
        return self

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.name}, stype={self.stype}, '
                f'dtype={self.dtype})')


@register(data_cls=PrimaryKeyConfig)
class PrimaryKey(BaseColumn):
    r"""The primary key of a :class:`kumo.graph.Table`."""
    def __init__(
        self,
        name: str,
        dtype: Optional[Dtype] = None,
        drop_duplicates: bool = False,
    ):
        super().__init__(name, Stype.ID, dtype)
        self.drop_duplicates = drop_duplicates

    @property
    def name(self) -> str:
        return self._name

    @name.setter  # No reason to disallow modifications to {pkey.name}.
    def name(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.name}, dtype={self.dtype})')

    @property
    def full_stats(self) -> List[ColStatType]:
        stats = super().full_stats

        # We do not compute category counts on primary keys, as this is too
        # expensive (and primary keys are not encoded anway):
        if ColStatType.CATEGORY_COUNTS in stats:
            stats.remove(ColStatType.CATEGORY_COUNTS)
        return stats


@register(data_cls=ForeignKeyConfig)
class ForeignKey(BaseColumn):
    r"""A foreign key of a :class:`kumo.graph.Table`."""
    def __init__(
        self,
        name: str,
        dst_table: Optional[str] = None,
        dst_key: Optional[str] = None,
        dtype: Optional[Dtype] = None,
        dst_data_source: Optional[str] = None,
        stats: Optional[Dict[Union[EdgeStatType, ColStatType], Stats]] = None,
    ):
        super().__init__(name, Stype.ID, dtype)
        self.dst_table = dst_table
        self.dst_key = dst_key
        self.dst_data_source = dst_data_source

        # TODO(manan) remove, here for backwards compatibility:
        self.stats = stats or {}  # Dict[ColStatType, Stats]

        # Check if the dst_table can be fully-qualified.
        if self.dst_table is not None and self.dst_data_source is not None:
            self.dst_table = fully_qualified_table_name(self.dst_table,
                                                        self.dst_data_source)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.name}, '
                f'dst_table={self.dst_table}, '
                f'dst_key={self.dst_key}, '
                f'dtype={self.dtype})')

    @property
    def just_dst_table(self) -> str:
        if self.dst_table is None:
            raise RuntimeError("Foreign Key must have non-empty dst_table")
        return self.dst_table

    @property
    def dst_table_name(self) -> str:
        return get_table_name(self.just_dst_table)

    def can_link_to_dtype(self, other_name: str, other_dtype: Optional[Dtype],
                          raise_on_error: bool = False) -> bool:
        """Return True if the current foreign key is permitted to link to
        a column with the given dtype in another table.

        In general, we only allow foreign keys between matching dtypes. Integer
        precision differences are allowed when the foreign key can be safely
        widened to the destination key dtype. Floating point precision
        differences remain compatible.

        In the past we've allowed [integer, float, string] and [date, time]
        to be connected, but since the user can currently edit dtype, enforce
        exact matches and let the user decide how cast columns to resolve any
        mismatches by changing dtypes.

        At least one dtype of None is allowed for now since this case comes up
        in testing but shouldn't in prod; TODO: never allow links with None
        """
        if (self.dtype is None or other_dtype is None):
            logging.warn_or_raise(
                f"Links are only allowed between matching datatypes, but "
                f"a link was defined between datatypes {self.name}"
                f"(dtype={self.dtype}) and {other_name}(dtype={other_dtype})"
                ", at least one of which is None. Please update"
                " the datatype selection of one or both of your tables to"
                " ensure that they match.", raise_on_error=raise_on_error)
            return True

        elif self.dtype.is_int() and other_dtype.is_int():
            if self.dtype.can_link_to_dtype(other_dtype):
                return True
            logging.warn_or_raise(
                f"Links are only allowed between matching datatypes or "
                f"lossless integer widening, but a link was defined between "
                f"datatypes {self.name}(dtype={self.dtype}) and "
                f"{other_name}(dtype={other_dtype}). Please update the "
                "datatype selection of one or both of your tables to ensure "
                "that they match, or widen the destination key dtype.",
                raise_on_error=raise_on_error)
            return False

        elif self.dtype.is_float() and other_dtype.is_float():
            # Float keys are now rejected upstream by the graph/table validate
            # guard, so this permissive branch is superseded for float keys.
            return True

        elif self.dtype != other_dtype:
            if self.dtype.can_link_to_dtype(other_dtype):
                logging.warn_or_raise(
                    f"Foreign-key column {self.name}(dtype={self.dtype}) links "
                    f"to {other_name}(dtype={other_dtype}): int/float mismatch, "
                    f"likely due to NaN-promotion in pandas < 2.0. The link will "
                    f"be allowed. To silence this warning use "
                    f"df['{self.name}'].astype('Int64') on the FK column.",
                    raise_on_error=False)
                return True
            logging.warn_or_raise(
                f"Links are only allowed between matching datatypes, but "
                f"a link was defined between datatypes {self.name}"
                f"(dtype={self.dtype}) and {other_name}(dtype={other_dtype})"
                ". Please update the datatype selection of "
                "one or both of your tables to ensure that they match.",
                raise_on_error=raise_on_error)
            return False

        return True


@register(data_cls=ColumnConfig, group='graph')
class Column(BaseColumn):
    r"""A feature column of a :class:`kumo.graph.Table`.
    """
    def __init__(
        self,
        name: str,
        stype: Optional[Stype] = None,
        dtype: Optional[Dtype] = None,
        encoder: Optional[Encoder] = None,
        na_values: Optional[List[str]] = None,
        stats: Optional[Dict[ColStatType, Stats]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp_unit: Optional[str] = None,
        multicategorical_delimiter: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            stype=stype,
            dtype=dtype,
            na_values=na_values,
            metadata=metadata,
            timestamp_unit=timestamp_unit,
            multicategorical_delimiter=multicategorical_delimiter,
            **kwargs,
        )
        self._encoder = encoder

        # TODO(manan) remove, here for backwards compatibility:
        self.stats = stats or {}  # Dict[ColStatType, Stats]

    @property
    def encoder(self) -> Optional[Encoder]:
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: Encoder) -> None:
        if EncoderGenerator.supports_encoder(self, encoder):
            self._encoder = encoder
        else:
            raise ValueError(f'Invalid encoder type ({type(encoder)}) '
                             f'for column {self}')

    # Materialization #########################################################

    def validate(
        self,
        raise_on_error: bool = True,
    ) -> bool:
        status = True

        # 1. The column has unsupported data type
        if self.dtype is Dtype.unsupported:
            status = False
            logging.warn_or_raise(
                f"Column '{self.name}' has unsupported data type",
                raise_on_error)

        # 2. The column has a data type defined, if it is not encoded with a
        # Null() encoder:
        if not isinstance(self.encoder, Null) and self.dtype is None:
            status = False
            logging.warn_or_raise(f"Column '{self.name}' has no data type",
                                  raise_on_error)

        # 3. The column has a semantic type defined (all columns require a
        # semantic type):
        if self.stype is None:
            status = False
            logging.warn_or_raise(f"Column '{self.name}' has no semantic type",
                                  raise_on_error)

        # 4. The semantic type supports the column data type:
        elif (not isinstance(self.encoder, Null)
              and not self.stype.supports_dtype(self.dtype)):
            status = False
            logging.warn_or_raise(
                f"Column '{self.name}' has the semantic type '{self.stype}' "
                f"which is incompatible with its data type '{self.dtype}'",
                raise_on_error)

        return status

    def validate_encoders(
        self,
        data_version: str,
        raise_on_error: bool,
        col_stats: Dict[ColStatType, Stats],
    ) -> bool:
        status = True
        # 1. The encoder is present
        if self.encoder is None:
            logging.warn_or_raise(f"Column '{self.name}' has no encoder",
                                  raise_on_error)
            status = False

        # 2. Don't use `Categorical` for encoding (mostly) distinct values:
        if not (self.encoder is None or isinstance(self.encoder, Null)):
            data_version = data_version or self.connector.get_data_version(
                self.table_name)
            if self.stype == Stype.ID:
                num_unique_stat = col_stats.get(self.base_config(),
                                                {}).get(ColStatType.NUM_UNIQUE)
                if num_unique_stat is not None:
                    num_unique, num_entries = (num_unique_stat.value,
                                               num_unique_stat.num_entries)

                    if num_entries is None:
                        logger.warning(
                            "NUM_UNIQUE statistic for %s has no num_entries",
                            self,
                        )
                    elif encoder_selection.is_mostly_unique_id(
                            num_unique, num_entries):
                        status = False
                        logging.warn_or_raise(
                            f"The '{self.encoder}' encoder of column "
                            f"'{self.name}' is used to encode a column with "
                            f"'{num_unique}' distinct values across "
                            f"'{num_entries}' rows. This will result in a "
                            f"feature representation with mostly unique "
                            f"embeddings per row which may hurt performance. "
                            f"Consider applying a different encoder strategy "
                            f"for this particular column.",
                            raise_on_error=False)
                else:
                    logger.warning(
                        f"NUM_UNIQUE statistic was requested on {self} during "
                        f"column encoder validation but was not found in the "
                        f"backend cache. Soon, this will be treated as an "
                        f"error.")

        return status


@register(data_cls=TimeColumnConfig)
class TimeColumn(BaseColumn):
    r"""The time column of a :class:`kumo.graph.Table`."""
    def __init__(
        self,
        name: str,
        dtype: Optional[Dtype] = Dtype.date,
        is_pseudo: bool = False,
        na_values: Optional[List[str]] = None,
    ):
        self.is_pseudo = is_pseudo  # indicates if the time is pseudo or not.
        super().__init__(name, Stype.timestamp, dtype, na_values)

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Dtype) -> None:
        # The column has a data type that can be converted to timestamp
        if self.is_pseudo:
            if not dtype.is_int():
                raise ValueError(
                    f"Cannot set Pseudo Time Column {self.name} to date type "
                    f"'{dtype}'. Must be of int type.")
        else:
            if dtype is None:
                raise ValueError(
                    f"Time Column '{self.name}' must have a non-empty data "
                    "type.")
            elif not dtype.is_timestamp():
                raise ValueError(
                    f"Cannot set Time Column '{self.name}' to data type "
                    f"'{dtype}'. Must be either '{Dtype.time}' or "
                    f"'{Dtype.date}'.")
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._name

    @name.setter  # No reason to disallow modifications to {time_col.name}.
    def name(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'


@register(data_cls=BoolColumnConfig)
class BoolColumn(BaseColumn):
    r"""A column of a :class:`kumo.graph.Table` holding explicit boolean
    values."""
    def __init__(
        self,
        name: str,
        stype: Optional[Stype] = Stype.categorical,
    ):
        super().__init__(name, stype, Dtype.bool)

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Dtype):
        if dtype is None:
            raise ValueError(f"Boolean column '{self.name}' must have a "
                             f"non-empty data type.")
        if dtype != Dtype.bool:
            raise ValueError(f"Cannot set boolean column '{self.name}' to "
                             f"data type '{dtype}'.")
        self._dtype = dtype

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'


###############################################################################


class EncoderGenerator:
    """Compatibility facade over the shared core selector."""

    NA_THRESHOLD = encoder_selection.NA_THRESHOLD
    SMALL_NUM_ROWS_THRESHOLD = encoder_selection.SMALL_NUM_ROWS_THRESHOLD
    CATEGORY_UNIQUE_THRESHOLD_MULTIPLIER = (
        encoder_selection.CATEGORY_UNIQUE_THRESHOLD_MULTIPLIER)
    MAX_ONE_HOT_CATEGORIES = encoder_selection.MAX_ONE_HOT_CATEGORIES
    MAX_VALUE_TO_STANDARDIZE = encoder_selection.MAX_VALUE_TO_STANDARDIZE
    UNIQUE_ID_THRESHOLD = encoder_selection.UNIQUE_ID_THRESHOLD
    MAX_AUTO_CATEGORICAL_CONVERSION_THRESHOLD = (
        encoder_selection.MAX_AUTO_CATEGORICAL_CONVERSION_THRESHOLD)
    MAX_CATEGORIES = encoder_selection.MAX_CATEGORIES
    MAX_MULTI_LABEL_CATEGORIES = encoder_selection.MAX_MULTI_LABEL_CATEGORIES
    UNIQUE_INT_THRESHOLD = encoder_selection.UNIQUE_INT_THRESHOLD
    KURTOSIS_THRESHOLD = encoder_selection.KURTOSIS_THRESHOLD
    GLOVE_COVERAGE_THRESHOLD = encoder_selection.GLOVE_COVERAGE_THRESHOLD
    NON_CHAR_THRESHOLD = encoder_selection.NON_CHAR_THRESHOLD
    AVG_TOKENS_THRESHOLD = encoder_selection.AVG_TOKENS_THRESHOLD
    REQUIRED_EXACT_STATS = [
        ColStatType(stat_type.value)
        for stat_type in encoder_selection.REQUIRED_EXACT_STATS
    ]
    REQUIRED_APPROX_STATS = [
        ColStatType(stat_type.value)
        for stat_type in encoder_selection.REQUIRED_APPROX_STATS
    ]
    REQUIRED_STATS = REQUIRED_EXACT_STATS + REQUIRED_APPROX_STATS

    @classmethod
    def supported_encoders(cls, dtype: Dtype) -> List[Encoder]:
        return encoder_selection.supported_encoder_types(dtype)

    @classmethod
    def supports_encoder(cls, col: Column, encoder: Encoder) -> bool:
        return encoder_selection.supports_encoder(col.dtype, encoder)

    @staticmethod
    def _shared_stats(
        stats: Dict[ColStatType, Stats],
    ) -> Dict[ApiColStatType, encoder_selection._EncoderStat]:
        supported = {
            stat_type.value: stat_type
            for stat_type in encoder_selection.REQUIRED_STATS
        }
        return {
            supported[stat_type.value]:
            encoder_selection._EncoderStat(
                value=stat.value,
                approx=stat.approx,
                num_entries=stat.num_entries,
            )
            for stat_type, stat in stats.items() if stat_type.value in supported
        }

    @classmethod
    def get_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
        *,
        table_name: str | None = None,
    ) -> Encoder:
        return encoder_selection.select_encoder(
            (f"{table_name}.{col.name}"
             if table_name is not None else col.qualified_name),
            col.dtype,
            col.stype,
            cls._shared_stats(stats),
        )

    @classmethod
    def get_numerical_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return encoder_selection.select_numerical_encoder(
            cls._shared_stats(stats), column_name=col.qualified_name)

    @classmethod
    def get_categorical_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return encoder_selection.select_categorical_encoder(
            col.qualified_name,
            col.dtype,
            cls._shared_stats(stats),
        )

    @classmethod
    def get_multicategorical_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return encoder_selection.select_multicategorical_encoder(
            col.qualified_name,
            cls._shared_stats(stats),
        )

    @classmethod
    def get_id_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return encoder_selection.select_id_encoder(
            col.qualified_name,
            col.dtype,
            cls._shared_stats(stats),
        )

    @staticmethod
    def get_required_text_stats() -> list[ColStatType]:
        return [
            ColStatType.STRING_GLOVE_OVERLAP, ColStatType.STRING_AVG_TOKENS,
            ColStatType.STRING_MAX_TOKENS, ColStatType.STRING_AVG_NON_CHAR
        ]

    @classmethod
    def get_text_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return encoder_selection.select_text_encoder(
            col.qualified_name,
            cls._shared_stats(stats),
        )

    @classmethod
    def get_timestamp_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return Datetime()

    @classmethod
    def get_sequence_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return encoder_selection.select_sequence_encoder(
            col.dtype,
            cls._shared_stats(stats),
            column_name=col.qualified_name,
        )

    @classmethod
    def get_image_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return Image()

    @classmethod
    def get_null_encoder(
        cls,
        col: Column,
        stats: Dict[ColStatType, Stats],
    ) -> Encoder:
        return Null()
