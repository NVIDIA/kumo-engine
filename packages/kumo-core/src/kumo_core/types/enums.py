# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical enums shared by authoring, data, and training code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

# `Dtype` and `Stype` are wire contract, not engine internals: a client and the
# engine must agree on what a column is. They live in `kumoapi.typing` as the
# single definition. The conversions that need torch/pyarrow stay here, in
# `kumo_core.types.conversions`, so this package never pulls the training stack
# into a client install.
from kumoapi.typing import Dtype, Stype

TrainModuleOutput: TypeAlias = dict[str, Any | dict[str, Any] | None]


def persisted_type_token(type_: Dtype | Stype | None) -> str:
    """Renders a `Dtype`/`Stype` as it appears in on-disk keys and hashes.

    Column-statistics cache keys and the table-config dedupe hash are written
    to durable storage, so their spelling is a storage format and not a
    presentation choice. Both used to be built from `str(stype)`, which the
    `kumo_core` copies of these enums rendered as `"kumo.<value>"`. Now that
    the single definition is kumoapi's, `str` gives the bare value -- so
    keeping the prefix here is what stops every already-cached entry from
    missing its key and being recomputed.

    New formats should not adopt the prefix; `parse_col_stat_from_key` already
    strips an optional one, so readers tolerate either.
    """
    if type_ is None:
        return str(None)
    return f'kumo.{type_.value}'


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    MULTILABEL_RANKING = "multilabel_ranking"
    REGRESSION = "regression"
    TEMPORAL_LINK_PREDICTION = "temporal_link_prediction"
    STATIC_LINK_PREDICTION = "static_link_prediction"
    FORECASTING = "forecasting"
    INVALID = "invalid"
    LINK_PREDICTION = "link_prediction"

    @classmethod
    def get_classification_tasks(cls) -> list["TaskType"]:
        return [
            cls.BINARY_CLASSIFICATION,
            cls.MULTICLASS_CLASSIFICATION,
            cls.MULTILABEL_CLASSIFICATION,
        ]

    @classmethod
    def get_node_pred_tasks(cls) -> list["TaskType"]:
        return [
            *cls.get_classification_tasks(),
            cls.MULTILABEL_RANKING,
            cls.REGRESSION,
            cls.FORECASTING,
        ]

    @classmethod
    def get_link_pred_tasks(cls) -> list["TaskType"]:
        return [
            cls.TEMPORAL_LINK_PREDICTION,
            cls.STATIC_LINK_PREDICTION,
            cls.LINK_PREDICTION,
        ]

    @property
    def is_classification(self) -> bool:
        return self in self.get_classification_tasks()

    @property
    def is_node_pred(self) -> bool:
        return self in self.get_node_pred_tasks()

    @property
    def is_link_pred(self) -> bool:
        return self in self.get_link_pred_tasks()

    @property
    def is_multilabel(self) -> bool:
        return self in {
            self.MULTILABEL_CLASSIFICATION,
            self.MULTILABEL_RANKING,
        }

    @property
    def is_ranking(self) -> bool:
        return self is self.MULTILABEL_RANKING or self.is_link_pred

    @property
    def has_multiple_predictions(self) -> bool:
        return (self.is_ranking or self.is_multilabel
                or self is self.MULTICLASS_CLASSIFICATION)

    @property
    def has_categorical_target(self) -> bool:
        return self in {
            *self.get_classification_tasks(),
            self.MULTILABEL_RANKING,
            self.TEMPORAL_LINK_PREDICTION,
            self.STATIC_LINK_PREDICTION,
        }

    @property
    def has_numerical_target(self) -> bool:
        return self in {self.REGRESSION, self.FORECASTING}

    @property
    def uses_arl_labels(self) -> bool:
        return self is self.FORECASTING

    @property
    def has_array_target(self) -> bool:
        return self.is_multilabel or self.is_link_pred

    @property
    def supports_xai(self) -> bool:
        return self is not self.FORECASTING and not self.is_ranking

    @classmethod
    def _missing_(cls, value: str) -> "TaskType | None":
        if value.lower() == "logistic_regression":
            return cls.INVALID
        if value.lower() == "classification":
            return cls.MULTICLASS_CLASSIFICATION
        if value.lower() == "multilabel":
            return cls.MULTILABEL_CLASSIFICATION
        return None


class ConnectorType(str, Enum):
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    BIGQUERY = "bigquery"
    S3 = "s3"
    GCS = "gcs"
    ADLS = "adls"
    FILE = "file"

    @property
    def is_file_based_connector(self) -> bool:
        return self is ConnectorType.FILE

    @property
    def supported_file_suffixes(self) -> list[str]:
        return ["csv", "parquet"] if self is ConnectorType.FILE else []


class ColStatType(str, Enum):
    COUNT = "COUNT"
    NUM_NA = "NUM_NA"
    NA_FRACTION = "NA_FRACTION"
    INVALID_FRACTION = "INVALID_FRACTION"
    NUM_PARSE_FAILURES = "NUM_PARSE_FAILURES"
    DROPPED_ROWS = "DROPPED_ROWS"
    MIN = "MIN"
    MAX = "MAX"
    MEAN = "MEAN"
    QUANTILES = "QUANTILES"
    QUANTILE25 = "QUANTILE25"
    MEDIAN = "MEDIAN"
    QUANTILE75 = "QUANTILE75"
    STD = "STD"
    KURTOSIS = "KURTOSIS"
    HISTOGRAM = "HISTOGRAM"
    NUM_IRRATIONAL = "NUM_IRRATIONAL"
    HISTOGRAM_HALF_MONTH = "HISTOGRAM_HALF_MONTH"
    NUM_UNIQUE = "NUM_UNIQUE"
    NUM_UNIQUE_MULTI = "NUM_UNIQUE_MULTI"
    CATEGORY_COUNTS = "CATEGORY_COUNTS"
    MULTI_CATEGORY_COUNTS = "MULTI_CATEGORY_COUNTS"
    UNIQUE_FRACTION = "UNIQUE_FRACTION"
    MULTI_CATEGORIES_SEPARATOR = "MULTI_CATEGORIES_SEPARATOR"
    STRING_AVG_LEN = "STRING_AVG_LEN"
    STRING_MAX_LEN = "STRING_MAX_LEN"
    STRING_AVG_TOKENS = "STRING_AVG_TOKENS"
    STRING_MAX_TOKENS = "STRING_MAX_TOKENS"
    STRING_GLOVE_OVERLAP = "STRING_GLOVE_OVERLAP"
    STRING_AVG_NON_CHAR = "STRING_AVG_NON_CHAR"
    STRING_ARR_MIN_LEN = "STRING_ARR_MIN_LEN"
    STRING_ARR_MAX_LEN = "STRING_ARR_MAX_LEN"
    SEQUENCE_MAX_LENGTH = "SEQUENCE_MAX_LENGTH"
    SEQUENCE_MIN_LENGTH = "SEQUENCE_MIN_LENGTH"
    SEQUENCE_MEAN = "SEQUENCE_MEAN"
    SEQUENCE_STD = "SEQUENCE_STD"
    SUMMARY_STATS_BY_CLASS = "SUMMARY_STATS_BY_CLASS"

    def supports(self, dtype: Dtype | None) -> bool:
        if self in {
                self.COUNT,
                self.NUM_NA,
                self.NA_FRACTION,
                self.INVALID_FRACTION,
                self.DROPPED_ROWS,
        }:
            return True
        if self in {self.NUM_UNIQUE, self.UNIQUE_FRACTION}:
            return dtype is None or not dtype.is_list()
        if self in {self.MIN, self.MAX}:
            return dtype is not None and (dtype.is_numerical()
                                          or dtype.is_temporal())
        if self in {
                self.MEAN,
                self.QUANTILES,
                self.MEDIAN,
                self.STD,
                self.KURTOSIS,
                self.HISTOGRAM,
                self.NUM_IRRATIONAL,
                self.QUANTILE25,
                self.QUANTILE75,
                self.SUMMARY_STATS_BY_CLASS,
        }:
            return dtype is not None and dtype.is_numerical()
        if self is self.CATEGORY_COUNTS:
            return dtype is not None and (dtype.is_int() or dtype.is_bool() or
                                          dtype.is_string() or dtype.is_float())
        if self in {
                self.NUM_UNIQUE_MULTI,
                self.MULTI_CATEGORY_COUNTS,
                self.MULTI_CATEGORIES_SEPARATOR,
        }:
            return dtype is not None and (dtype.is_string() or dtype.is_list())
        if self in {
                self.STRING_AVG_LEN,
                self.STRING_MAX_LEN,
                self.STRING_AVG_TOKENS,
                self.STRING_MAX_TOKENS,
                self.STRING_GLOVE_OVERLAP,
                self.STRING_AVG_NON_CHAR,
        }:
            return dtype is not None and dtype.is_string()
        if self in {self.SEQUENCE_MAX_LENGTH, self.SEQUENCE_MIN_LENGTH}:
            return dtype is not None and (dtype.is_list() or dtype.is_string())
        if self is self.HISTOGRAM_HALF_MONTH:
            return dtype is not None and dtype.is_temporal()
        return False

    def supports_stype(self, stype: Stype | None) -> bool:
        basic = {
            self.COUNT,
            self.NUM_NA,
            self.NA_FRACTION,
            self.INVALID_FRACTION,
        }
        if stype is not Stype.unsupported and self in basic:
            return True
        supported = {
            Stype.numerical: {
                self.MEAN,
                self.MEDIAN,
                self.STD,
                self.MIN,
                self.MAX,
                self.KURTOSIS,
                self.QUANTILES,
                self.QUANTILE25,
                self.QUANTILE75,
                self.HISTOGRAM,
                self.NUM_IRRATIONAL,
                self.SUMMARY_STATS_BY_CLASS,
            },
            Stype.categorical: {
                self.CATEGORY_COUNTS,
                self.COUNT,
                self.NUM_NA,
                self.NUM_UNIQUE,
                self.SUMMARY_STATS_BY_CLASS,
                self.STRING_AVG_LEN,
            },
            Stype.ID: {
                self.UNIQUE_FRACTION,
                self.CATEGORY_COUNTS,
                self.COUNT,
                self.NUM_NA,
                self.NUM_UNIQUE,
                self.SUMMARY_STATS_BY_CLASS,
                self.STRING_AVG_LEN,
            },
            Stype.multicategorical: {
                self.NUM_UNIQUE_MULTI,
                self.MULTI_CATEGORY_COUNTS,
                self.MULTI_CATEGORIES_SEPARATOR,
            },
            Stype.text: {
                self.STRING_GLOVE_OVERLAP,
                self.STRING_AVG_TOKENS,
                self.STRING_AVG_LEN,
                self.STRING_MAX_TOKENS,
                self.STRING_AVG_NON_CHAR,
            },
            Stype.timestamp: {
                self.NUM_UNIQUE,
                self.MAX,
                self.MEAN,
                self.MIN,
                self.HISTOGRAM_HALF_MONTH,
            },
            Stype.sequence: {
                self.SEQUENCE_MIN_LENGTH,
                self.SEQUENCE_MAX_LENGTH,
            },
            Stype.image: set(),
        }
        if stype in supported:
            return self in supported[stype]
        if stype is not Stype.unsupported and self in {
                self.NUM_UNIQUE,
                self.UNIQUE_FRACTION,
                self.NUM_UNIQUE_MULTI,
        }:
            return True
        return self is not self.CATEGORY_COUNTS


class EdgeStatType(str, Enum):
    EDGE_COUNT = "EDGE_COUNT"
    FRAC_OF_TOTAL_EDGES = "FRAC_OF_TOTAL_EDGES"
    EDGE_DENSITY = "EDGE_DENSITY"
    NUM_SRC_NONZERO_DEGREE = "NUM_SRC_NONZERO_DEGREE"
    NUM_DST_NONZERO_DEGREE = "NUM_DST_NONZERO_DEGREE"
    MAX_SRC_DEGREE = "MAX_SRC_DEGREE"
    MAX_DST_DEGREE = "MAX_DST_DEGREE"
    NUM_SRC_NODES = "NUM_SRC_NODES"
    NUM_DST_NODES = "NUM_DST_NODES"
    FRAC_SRC_NONZERO_DEGREE = "FRAC_SRC_NONZERO_DEGREE"
    FRAC_DST_NONZERO_DEGREE = "FRAC_DST_NONZERO_DEGREE"
    AVG_SRC_DEGREE = "AVG_SRC_DEGREE"
    AVG_DST_DEGREE = "AVG_DST_DEGREE"
    AVG_SRC_NONZERO_DEGREE = "AVG_SRC_NONZERO_DEGREE"
    AVG_DST_NONZERO_DEGREE = "AVG_DST_NONZERO_DEGREE"
    IN_DEGREE_PERCENTILE_LIST = "IN_DEGREE_PERCENTILE_LIST"


class PredTime(Enum):
    MAX_TRAIN = "max_train"
    TODAY = "today"


class AggregationType(str, Enum):
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    ANY = "ANY"
    FIRST = "FIRST"
    LAST = "LAST"
    LIST_DISTINCT = "LIST_DISTINCT"
    LIST = "LIST"


class TableStatType(str, Enum):
    NUM_ROWS = "NUM_ROWS"
    SOURCE_SIZE = "SOURCE_SIZE"
    MATERIALIZED_SIZE = "MATERIALIZED_SIZE"


@dataclass
class TaskTypeWithReason:
    task_type: TaskType
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)


class ExplainMethod(str, Enum):
    GRADIENT = "gradient"
    COLUMN_ANALYSIS = "column_analysis"
    SUBGRAPH_COLLECTOR = "subgraph_collector"


class TargetType(str, Enum):
    TRUTH = "truth"
    PREDICTION = "prediction"


class TrainTableType(str, Enum):
    STATIC_NODE_PRED = "static_node_pred"
    STATIC_LINK_PRED = "static_link_pred"
    TEMPORAL_NODE_PRED = "temporal_node_pred"
    TEMPORAL_LINK_PRED = "temporal_link_pred"

    @classmethod
    def get_temporal_train_table_type(cls) -> list["TrainTableType"]:
        return [cls.TEMPORAL_NODE_PRED, cls.TEMPORAL_LINK_PRED]

    @classmethod
    def get_link_pred_train_table_type(cls) -> list["TrainTableType"]:
        return [cls.TEMPORAL_LINK_PRED, cls.STATIC_LINK_PRED]

    @property
    def is_temporal(self) -> bool:
        return self in self.get_temporal_train_table_type()

    @property
    def is_link_pred(self) -> bool:
        return self in self.get_link_pred_train_table_type()


class PredictionArtifactType(str, Enum):
    PREDICTIONS = "PREDICTIONS"
    EMBEDDINGS = "EMBEDDINGS"

    @property
    def output_file_name(self) -> str:
        return f"{self.value.lower()}.parquet"


__all__ = [
    "AggregationType",
    "ColStatType",
    "ConnectorType",
    "Dtype",
    "EdgeStatType",
    "ExplainMethod",
    "PredTime",
    "PredictionArtifactType",
    "Stype",
    "TableStatType",
    "TargetType",
    "TaskType",
    "TaskTypeWithReason",
    "TrainModuleOutput",
    "TrainTableType",
]
