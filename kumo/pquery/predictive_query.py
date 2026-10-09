# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import uuid
from abc import abstractmethod
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, Union

import kumo_core.types as core_types
from kumo_core.pquery import resolve_target_to_entity_paths
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumo_core.training.options import allow_array_targets
from kumo_core.training.splits import DateOffsetSplit as CoreDateOffsetSplit
from kumo_core.training.splits import (
    default_split_for_query,
    normalize_static_link_split,
)
from kumo_core.utils.registry import register
from kumo_core.utils.time import compute_unix_step
from kumoapi.pquery import QueryType, ValidatedPredictiveQuery
from kumoapi.typing import AggregationType, ProblemType, TimeUnit
from pandas import DataFrame as PandasDataFrame
from pandas import DateOffset, Timestamp

from async_utils.sync import blocking_method
from io_utils import fsutil, logging, parquet_util
from io_utils.json import load_json, save_json
from io_utils.yaml import to_yaml
from kumo.config import ConfigMixin, PredictiveQueryV2Config, TaskConfig
from kumo.config.mixin import construct_object_string_from_config
from kumo.connector import Connector, ParquetConnector
from kumo.data.utils import get_table_path
from kumo.graph import ForeignKey, Graph
from kumo.home import get_home_dir
from kumo.pquery.batch_pred_table_generator import PredictionTableGeneratorV2
from kumo.pquery.generator_utils import get_task_type
from kumo.pquery.train_table_generator import TrainingTableGeneratorV2
from kumo.pquery.utils import col_name, table_name, validate_weight_column
from kumo.pquery.weight_spec import normalize_weight_col
from kumo.stats.train_table_stats import (
    TemporalLinkTrainTableStats,
    TemporalNodeTrainTableStats,
)
from kumo.task import Task
from kumo.task.split import (
    ApproxDateOffsetSplit,
    DateOffsetSplit,
    RandomLinkSplit,
    RandomSplit,
    Split,
    TemporalSplit,
    TimeRangeSplit,
)
from kumo.task.splits import TimeSplit
from kumo.task.splits.parsing import get_split
from kumo.utils.testing import pytest_only
from kumo.validation import (
    ValidationException,
    ValidationResponse,
    ValidationWarning,
)
from kumo.validation.train_table_validation import TrainingTableGenerationError
from monitoring.tracing import TraceOptions, trace

logger = logging.getLogger(__name__)

TRAIN_TABLE_NAME = 'train_table'
CUSTOM_TRAIN_TABLE_NAME = 'custom_train_table'
PRED_TABLE_NAME = 'pred_table'

TableGenerators = Union[
    TrainingTableGeneratorV2,
    PredictionTableGeneratorV2,
]

# An in-process capability used only by the REST request boundary after an
# authorized activity has validated and hydrated a PQuery. It cannot arrive
# through a serialized config or an API request.
_VALIDATED_PQUERY_RECONSTRUCTION = object()


def train_table_source_name(train_table_path: str,
                            is_custom: bool = False) -> str:
    """
    Compose the internal data-source name for a generated training or
    prediction table.
    """
    suffix = md5(train_table_path.encode()).hexdigest()
    table_name = CUSTOM_TRAIN_TABLE_NAME if is_custom else TRAIN_TABLE_NAME
    return f'{table_name}_parquet_{suffix}'


def _force_ttng_lp_array(graph, validated_predictive_query,
                         forecast_length: int) -> bool:
    """True iff this is a link-prediction query -- array-mode is the only LP
    mode under TT-NG, so a persisted/deserialized explicit False must be forced
    on."""
    return get_task_type(graph, validated_predictive_query,
                         forecast_length=forecast_length).task_type.is_link_pred


class PQuerySettings:
    # Default is to enable everything so that tests and examples can continue
    # working without depending on the environment.
    # Note that default values for rest.settings.PredictiveQuerySettings still
    # reflect GA settings.

    # PQ features
    disable_static_node_prediction: bool = False
    disable_static_link_prediction: bool = False
    disable_temporal_link_prediction: bool = False
    disable_multilabel_classification: bool = False
    disable_multilabel_ranking: bool = False
    disable_temporal_node_regression: bool = False
    disable_forecasting: bool = False
    disable_first_last_on_fkey: bool = False
    disable_first_last: bool = False
    use_decoupled_splits: bool = False

    # Other options
    disable_pquery_advanced_options: bool = False


class PredictiveQuery(ConfigMixin):
    r"""
    Predictive Query interface/base class. All implementations of Predictive
    Query must extend from this class.
    """
    TIMESTAMP = 'TIMESTAMP'  # Name of the column with temporal information.
    TARGET = 'TARGET'  # Name of the column with the target labels.

    def __init__(
        self,
        graph: Graph,
        name: str,
        entity: str,
        target: str,
        **kwargs,
    ):
        self.graph = graph
        self.name = name
        self.entity = entity
        self.target = target

    # Used as a public api
    @abstractmethod
    def validate(self) -> ValidationResponse:
        r"""Validates the config and returns any warnings.

        Returns:
            ValidationResponse: The list of encountered warnings.

        Raises:
            ValueError: If any errors were encountered"""
        raise NotImplementedError

    def get_validate_response(self) -> ValidationResponse:
        self._validate_response

    def get_validate_train_table_response(self) -> Optional[ValidationResponse]:
        r"""Return the validation responses triggered during train table
        generation. If this function is called before train table is created
        return `None`.
        """
        return self._validate_train_table_response

    @property
    @abstractmethod
    def target_to_entity_paths(self) -> List[List[Tuple[str, str, str]]]:
        r"""Returns the paths from the target table to the entity."""
        raise NotImplementedError

    @blocking_method()
    def get_train_table_stats(self, job_id: Optional[str] = None) -> Any:
        r"""Gets important statistics of the generated
        :class:`~kumo.task.TrainingTable` in order to rate the quality of a
        :class:`PredictiveQuery`.
        Args:
            job_id: job id to retrieve the training table stats.
        """
        return self.table_generator.get_train_table_stats(job_id)

    def get_train_table_progress(self, job_id: str) -> Any:
        r"""Gets progress and intermediate statistics of the generated
        :class:`~kumo.task.TrainingTable` in order to rate the quality of a
        :class:`PredictiveQuery`.
        Args:
            job_id: job id to retrieve the training table stats.
        """
        return self.table_generator.get_train_table_progress(job_id)

    def get_pred_table_stats(self) -> Any:
        r"""Gets important statistics of the generated
        prediction table"""
        return self.table_generator.get_pred_table_stats()

    @abstractmethod
    def set_table_generator(self):
        pass

    @abstractmethod
    def get_table_generator(self) -> TableGenerators:
        pass

    @property
    def query_type(self) -> QueryType:
        raise NotImplementedError()

    @query_type.setter
    def query_type(self, query_type):
        raise NotImplementedError()

    def get_target_offset(self) -> Tuple[int, int]:
        r"""Returns the start and end offset of target aggregation.
        """
        raise NotImplementedError()

    def get_bp_entity_col_name(self) -> str:
        r"""
        Gets the name of the entity col found in the batch prediction table
        based on whether it's PQLV2 or not
        """
        raise NotImplementedError()

    @blocking_method()
    def has_train_table_time_col(self) -> bool:
        """Returns whether the train table has a time columns"""
        time_col = self.get_table_generator().get_time_col()
        if time_col is None:
            return False
        # Pseudo-time does not count
        return not time_col.is_pseudo

    @abstractmethod
    def get_task_type(self) -> core_types.TaskType:
        pass

    @property
    def entity_table(self) -> str:
        r"""Fully qualified name of the entity table."""
        entity_table_name = table_name(self.entity)
        return self.graph.fully_qualified_table_name(entity_table_name)

    @property
    @abstractmethod
    def entity_table_names(self) -> list[str]:
        pass

    @property
    @pytest_only
    def train_df(self) -> PandasDataFrame:
        """
        NOTE: This method is only introduced to maintain backwards
        compatibility with dozens of tests using predictive_query.train_df
        and expecting a pandas DataFrame. Do not use this attribute/method
        for production workflows because it loads the complete parquet-backed
        training table into the caller's memory.
        """
        logger.warning(
            'Invoked predictive_query.train_df, this is deprecated. '
            'Please use predictive_query.train_table_path instead, which '
            'points to the location of the training table.')
        df = parquet_util.read_parquet(self._train_table_path)
        if NODE_INDEX_COLUMN in df.columns:
            df = df.drop(columns=[NODE_INDEX_COLUMN])
        return df

    @property
    def train_table_path(self) -> str:
        """
        Returns the associated training table path with this predictive query.
        This will point to a parquet dataset which denotes the training table.
        """
        if not hasattr(self, '_train_table_path'):
            raise ValidationException(
                title="Training table not generated",
                message=(
                    "train_table_path doesn't exists for this predictive "
                    "query. Make sure you have called generate_train_table "
                    "function before calling for this property."),
            )
        return self._train_table_path

    @property
    def pred_df(self) -> PandasDataFrame:
        """
        NOTE: This method is only introduced to maintain backwards
        compatibility with dozens of tests using predictive_query.pred_df
        and expecting a pandas DataFrame. Do not use this attribute/method
        for production workflows because it loads the complete parquet-backed
        prediction table into the caller's memory.
        """
        logger.warning(
            'Invoked predictive_query.pred_df, this is deprecated. '
            'Please use predictive_query.pred_table_path instead, which '
            'points to the location of the prediction table.')
        df = parquet_util.read_parquet(self._pred_table_path)
        if NODE_INDEX_COLUMN in df.columns:
            df = df.drop(columns=[NODE_INDEX_COLUMN])
        return df

    @property
    def pred_table_path(self) -> str:
        """
        Returns the associated pred table path with this pquery.
        This will point to a parquet dataset which denotes the pred table.
        """
        if not hasattr(self, '_pred_table_path'):
            raise ValidationException(
                title="Prediction table not generated",
                message=(
                    "pred_table_path doesn't exists for this predictive "
                    "query. Make sure you have called "
                    "generate_current_pred_table function before calling for "
                    "this property."),
            )
        return self._pred_table_path

    @abstractmethod
    def generate_train_table(
        self,
        job_id: Optional[str] = None,
        diskgraph_index_path: Optional[str] = None,
    ) -> Tuple[str, str, ValidationResponse]:
        raise NotImplementedError

    @abstractmethod
    def create_task(
        self,
        persist_dir: Optional[str] = None,
        train_table_path: Optional[str] = None,
        task_path_override: Optional[str] = None,
        job_id: Optional[str] = None,
        job_dir: Optional[str] = None,
        ignore_encoders: bool = False,
        weight_col_name: bool = False,
        train_table_name: Optional[str] = None,
    ) -> Task:
        raise NotImplementedError

    @abstractmethod
    def generate_current_pred_table(
        self,
        job_id: Optional[str] = None,
        training_job_id: Optional[str] = None,
        pred_time: Optional[Union[core_types.PredTime, Timestamp]] = None,
    ) -> Tuple[str, ValidationResponse]:
        raise NotImplementedError

    # Save and load ###########################################################
    def _remove_pandas_connector_from_graph(self):
        from kumo.connector.pandas_connector import PandasConnector
        self._removed_source_dict = {}
        for key, value in self.graph._data_source_dict.items():
            if isinstance(value, PandasConnector):
                self._removed_source_dict[key] = value
        for key in self._removed_source_dict.keys():
            del self.graph._data_source_dict[key]

    def _add_pandas_connector_back_to_graph(self):
        _removed_source_dict = ({} if hasattr(self, '_removed_source_dict') else
                                self._removed_source_dict)
        self.graph._data_source_dict.update(_removed_source_dict)
        self._removed_source_dict = {}

    def customize_config(self, values: Dict[str, Any]) -> Dict[str, Any]:
        return values

    def save(self, name: Optional[str] = None,
             destination: Optional[str] = None) -> str:
        """Save the :class:`kumo.PredictiveQuery` to storage.

        Args:
            name (string, optional): The name of the folder to store the data.
                If set to :obj:`None`, will use the name of the predictive
                query. (default: :obj:`None`)
            destination (string, optional): Base dir the query is saved under
                (``<destination>/queries/<name>``). :obj:`None` falls back to
                the process-global :func:`kumo.get_home_dir`.
        """
        name = name or self.name
        root_dir = osp.join(destination or get_home_dir(), 'queries', name)
        fsutil.makedirs(root_dir, exist_ok=True)

        # TODO: Persist training table DataFrame.
        path = osp.join(root_dir, 'config.json')
        save_json(self.config(), path)
        logger.info("Successfully saved query %s to %s.", self.name, root_dir)
        return name

    def to_yaml(self) -> str:
        r"""Converts the :class:`kumo.PredictiveQuery` to a YAML string.
        """
        return to_yaml(self.config())

    @classmethod
    @abstractmethod
    def load(cls, name: str,
             home_dir: Optional[str] = None) -> 'PredictiveQuery':
        raise NotImplementedError

    def _override_task_path(self, task_path_override: str) -> None:
        r"""Overrides the task with the given task config path.
        The overrding task config should match the generated task config.
        """
        from kumo.task import Task
        task = Task.from_config(load_json(task_path_override, cls=TaskConfig))
        logger.warning(f"Overriding task with path {task_path_override}")
        logger.warning("Comparision of the overrding task config "
                       "and the generated task config:")
        task.diff_config(self.task)
        override_task_graph = task.config().graph
        generated_task_graph = self.task.config().graph
        assert override_task_graph.name == generated_task_graph.name, (
            "The graph name in the overriding task config does not match")
        assert (override_task_graph.data_source_dict ==
                generated_task_graph.data_source_dict), (
                    "The data source dict in the overriding task config "
                    "does not match: "
                    f"{override_task_graph.data_source_dict} vs "
                    f"{generated_task_graph.data_source_dict}")
        override_task_table_names = {
            table.table_name
            for table in override_task_graph.tables
        }
        generated_task_table_names = {
            table.table_name
            for table in generated_task_graph.tables
        }
        assert (override_task_table_names == generated_task_table_names), (
            "The table names in the overriding task config do not match: "
            f"{override_task_table_names} vs {generated_task_table_names}")
        task.train_table = self.task.train_table
        task.name = self.name
        self.task = task


@register(data_cls=PredictiveQueryV2Config)
class PredictiveQueryV2(PredictiveQuery):
    @classmethod
    def _from_validated_config(
        cls,
        cfg: PredictiveQueryV2Config,
        *,
        graph: Graph,
        pquery_settings: PQuerySettings,
    ) -> 'PredictiveQueryV2':
        """Purely reconstruct an activity-validated PQuery snapshot."""
        return super().from_config(
            cfg,
            graph=graph,
            pquery_settings=pquery_settings,
            _validated_reconstruction=_VALIDATED_PQUERY_RECONSTRUCTION,
        )

    def __init__(
        self,
        name: str,
        graph: Graph,
        validated_predictive_query: ValidatedPredictiveQuery,
        timeframe_step: Optional[int] = None,
        train_start_offset: Optional[int] = None,
        train_end_offset: Optional[Union[int, DateOffset]] = None,
        split: Optional[Split] = None,
        pquery_settings: Optional[PQuerySettings] = None,
        forecast_length: Optional[int] = None,
        lag_timesteps: Optional[int] = None,
        drop_earliest_timeframe: Optional[bool] = True,
        list_distinct_target_is_array: Optional[bool] = None,
        lp_bp_layout: Optional[str] = None,
        year_over_year: bool = False,
        start_time: Optional[Union[int, str]] = None,
        end_time: Optional[Union[int, str]] = None,
        weight_col: Optional[str] = None,
        _validated_reconstruction: object = None,
        **kwargs,
    ):
        """
        Initializes a `PredictiveQueryV2` class and perform the following
        additional steps:
        1. Validating target for legitimate date range.
        2. Initializes split.
        3. Sets table_generator to be used by this class.

        Args:
        name (str): Name of the predictive query, also commonly referred to as
            its id.
        graph (kumo.Graph): The graph this predictive query is going to be run
            on.
        validated_predictive_query (ValidatedPredictiveQuery): A parsed and
            validated PQL v2 predictive query defining a problem statement.
        timeframe_step (int, optional): Defines the step size of generating
            time intervals for training table generation. If set to
            :obj:`None`, :obj:`timeframe_step` will be set to
            :obj:`timeframe`. (default: :obj:`None`). This is mostly used
            for overriding the default timeframe step based on the target
            definition when the default timeframe_step leads to low quality
            training table for example due to label imbalance.
        train_start_offset (int, optional): Defines the number of days from
            the most recent entry to include in the train set. For example,
            if set to :obj:`10` and the time unit used in target aggregation
            is `days`, then the train table will only include entries from
            the last 10 days. (default: :obj:`None`)
        train_end_offset (Union[pd.DateOffset, int], optional): Defines the
            time range from the most recent entry not to include in the train
            set. For example if set to :obj:`10` and the time unit used in
            target aggregation is `days`, then the train table will not
            include entries from the last 10 days. If set to
            `pd.DateOffset(months=1, days=2)`, then the train table will not
            include entries from the last one months and 2 days. Has to be
            strictly less than `train_start_offset` (default: :obj:`None`)
        split (Split, optional): The splitting scheme of the table.
            If set to :obj:`None`, will use a 80%/10%/10% random split on
            static queries. On temporal queries, will use the two most recent
            non-overlapping :obj:`timeframe` windows for validation and test
            splits. (default: :obj:`None`)
        forecast_length (int, optional): Length of the future time series
            to forecast. If > 1 or validated_predictive_query.problem_type is
            FORECAST, sets task_type to FORECASTING instead of REGRESSION.
            (default: :obj:`None`)
        lag_timesteps (int, optional): This sets the number of steps
            backward autoregression will take.
            (default: :obj:`None`)
        drop_earliest_timeframe (bool, None): Whether to consider dropping the
            earliest timeframe during train table generation. When
            `drop_earliest_timeframe` is False, the earliest timeframe will not
            be dropped. When `drop_earliest_timeframe` is True, the earliest
            timeframe might be dropped.
        list_distinct_target_is_array (bool, optional): Internal flag whether
            to return targets of list_distinct as an array or not.
            (default: :obj:`None`)
        year_over_year (bool): Whether to use year over year labels
            for forecasting problems. (default: :obj:`False`)
        start_time (int, str, optional): Start time for the training data.
            If provided as a string, it should be in the format 'YYYY-MM-DD'.
            If provided as an integer, it should be less than or equal to 0
            and represents the number of time units from the most recent entry.
        end_time (int, str, optional): End time for the training data.
            If provided as a string, it should be in the format 'YYYY-MM-DD'.
            If provided as an integer, it should be less than or equal to 0
            and represents the number of time units from the most recent entry.
        weight_col (str, optional): Weight column expression or column name to
            use for training example weights. (default: :obj:`None`)
        """
        # TODO: if entity is just a column name
        # we will not be able to reconstruct the entity ast
        # from self.entity
        entity = validated_predictive_query.entity_column
        target = validated_predictive_query.target_ast
        super().__init__(
            graph,
            name,
            entity,
            target,
        )
        self.validated_predictive_query = validated_predictive_query
        self.train_start_offset = train_start_offset
        self.train_end_offset = train_end_offset
        self.graph = graph
        self.name = name
        self.entity = entity
        self.target = target
        self.timeframe_step = timeframe_step
        self.pquery_settings = pquery_settings or PQuerySettings()
        self._validate_train_table_response = ValidationResponse()
        self.drop_earliest_timeframe = drop_earliest_timeframe
        self.lag_timesteps = lag_timesteps
        self.forecast_length = (forecast_length
                                or self.validated_predictive_query.num_forecasts
                                or 1)
        self.year_over_year = year_over_year
        self.start_time = start_time
        self.end_time = end_time
        self.weight_col = normalize_weight_col(weight_col)

        if list_distinct_target_is_array and not allow_array_targets():
            # Should not happen unless we turn ALLOW_ARRAY_TARGETS back off
            raise ValidationException(
                title="Array targets disabled",
                message=("Add ALLOW_ARRAY_TARGETS flag to use this predictive "
                         "query."),
            )
        elif list_distinct_target_is_array is None:
            # Default: array-mode iff this is a link-prediction query -- the
            # only array-target task under TT-NG. A non-LP LIST_DISTINCT (e.g.
            # multilabel CLASSIFY) stays scalar/multicategorical, so it is not
            # rejected as an unpredictable array target.
            list_distinct_target_is_array = _force_ttng_lp_array(
                self.graph, self.validated_predictive_query,
                self.forecast_length)
        elif not list_distinct_target_is_array and _force_ttng_lp_array(
                self.graph, self.validated_predictive_query,
                self.forecast_length):
            # Array-mode is the ONLY LP mode under TT-NG, so a persisted/
            # deserialized explicit False must be forced on for a TT-NG LP query
            # (treat it as None); otherwise the RandomSplit->RandomLinkSplit
            # force below never fires and static-LP silently emits no FK-free
            # lp_resolution. Scoped to TT-NG LP so the legacy exploded/hstack LP
            # path (ALLOW_ARRAY_TARGETS without TT-NG) is left unchanged.
            list_distinct_target_is_array = True

        self.list_distinct_target_is_array = list_distinct_target_is_array
        self.lp_bp_layout = lp_bp_layout

        if isinstance(split, (TemporalSplit, RandomSplit)):
            normalized = normalize_static_link_split(
                split,
                default_is_random_link_split=isinstance(
                    self.get_default_split(), RandomLinkSplit),
            )
            if normalized is not split:
                assert self.list_distinct_target_is_array
                logger.info(f"Overwrite '{split.__class__.__name__}' to "
                            f"'RandomLinkSplit' for static link prediction")
                # Keep the hosted registered subclass for config serialization.
                split = RandomLinkSplit(split.ratio)

        if isinstance(split, (TimeRangeSplit, DateOffsetSplit, TemporalSplit)):
            self._set_split(split)
        elif isinstance(split, TimeSplit):
            self._set_time_split(split)
        elif split is None:
            self.split = self.get_default_split()
        else:
            self.split = split
        self.set_table_generator()
        self._check_pquery_settings()
        if _validated_reconstruction is _VALIDATED_PQUERY_RECONSTRUCTION:
            self._validate_response = self._validate_without_graph_io()
        else:
            if _validated_reconstruction is not None:
                raise ValueError('Invalid validated PQuery reconstruction')
            self._validate_response = self.validate()

    @property
    def is_forecasting(self) -> bool:
        return (self.validated_predictive_query.problem_type
                == ProblemType.FORECAST or self.forecast_length > 1)

    @property
    def use_autoregression(self) -> bool:
        r"""Whether or not we are using autoregressive labels.
        """
        return self.lag_timesteps is not None and self.lag_timesteps > 0

    @property
    def query_type(self) -> QueryType:
        self._query_type = self.validated_predictive_query.query_type
        return self._query_type

    @query_type.setter
    def query_type(self, query_type) -> None:
        self._query_type = query_type

    def get_task_type(self) -> core_types.TaskType:
        return self.table_generator.task_type

    def get_validate_response(self) -> ValidationResponse:
        return self._validate_response

    @property
    def target_to_entity_paths(self) -> List[List[Tuple[str, str, str]]]:
        r"""Returns the paths from the target table to the entity."""
        target_key = self.validated_predictive_query.get_single_target_fkey()
        if target_key is None:
            return []

        target_table = self.graph.fully_qualified_table_name(
            table_name(target_key))
        entity_table = self.graph.fully_qualified_table_name(
            table_name(self.validated_predictive_query.entity_column))
        target_table_pkey = self.graph.table(target_table).pkey_name
        entity_fkey = self.validated_predictive_query.\
            get_single_target_fkey(lhs=True)

        rhs_path = None
        if self.get_task_type().is_link_pred:
            aggr = self.validated_predictive_query.\
                get_final_target_aggregation()
            fkey = self.graph.column_from_fqn(aggr._get_target_column_name())
            rhs_path = (target_table, fkey.name, fkey.dst_table)
        return resolve_target_to_entity_paths(
            target_table=target_table,
            target_key=col_name(target_key),
            entity_table=entity_table,
            target_primary_key=target_table_pkey,
            entity_key=col_name(entity_fkey),
            rhs_path=rhs_path,
        )

    def set_table_generator(self):
        self.table_generator: TrainingTableGeneratorV2 = (
            self.get_table_generator())

    @trace(TraceOptions(trace_params=True))
    def get_table_generator(self) -> TrainingTableGeneratorV2:
        """
        Gets the table_generator implementation used by this class.
        """
        # TODO (xinwei): Change the naming to `get_train_table_generator`
        generator = TrainingTableGeneratorV2(
            graph=self.graph,
            validated_query=self.validated_predictive_query,
            split=self.split,
            train_start_offset=self.train_start_offset,
            train_end_offset=self.train_end_offset,
            timeframe_step=self.timeframe_step,
            drop_earliest_timeframe=self.drop_earliest_timeframe,
            lag_timesteps=self.lag_timesteps,
            forecast_length=self.forecast_length,
            year_over_year=self.year_over_year,
            start_time=self.start_time,
            end_time=self.end_time,
            use_decoupled_splits=self.pquery_settings.use_decoupled_splits,
            weight_col=self.weight_col,
        )
        generator.set_list_distinct_target_is_array(
            self.list_distinct_target_is_array)
        return generator

    def get_batch_pred_table_generator(self) -> PredictionTableGeneratorV2:
        generator = PredictionTableGeneratorV2(
            graph=self.graph,
            validated_query=self.validated_predictive_query,
            forecast_length=self.forecast_length,
        )
        generator.list_distinct_target_is_array = (
            self.list_distinct_target_is_array)
        generator.lp_bp_layout = self.lp_bp_layout
        return generator

    def _set_time_split(
        self,
        split: TimeSplit,
    ):
        r"""Sets the time split parameters (unit, timeframe_size,
        anchor_start_offset) based on the temporal query parameters if they
        are not already set.

        For temporal queries, we can infer these values from the query itself:
        - unit comes from the query's date offset unit.
        - timeframe_size comes from the query's timeframe.
        - anchor_start_offset comes from the query's anchor start offset.

        Args:
            split (TimeSplit): The time split to configure.
        """
        if self.query_type == QueryType.TEMPORAL:
            # Get target date offset range from the query
            target_date_offset = (self.validated_predictive_query.
                                  get_combined_date_offset_range())

            # Set unit if not already set
            if split.unit is None:
                split.unit = target_date_offset.unit

            # Set timeframe_size if not already set
            if split.timeframe_size is None:
                split.timeframe_size = (target_date_offset.end -
                                        target_date_offset.start)

            # Set anchor_start_offset if not already set
            if split.anchor_start_offset is None:
                split.anchor_start_offset = target_date_offset.start

        self.split = split

    def _set_split(
        self,
        split: Union[DateOffsetSplit, TimeRangeSplit, TemporalSplit],
    ):
        r"""Stores `split` correctly for `DateOffsetSplit` and
        `TimeRangeSplit`. There are several parameters in these
        splits, used to prevent overlap, that are specific to the predictive
        query (such as `timeframe`). This method sets them correctly before
        setting `self.split`."""
        if self.query_type == QueryType.TEMPORAL:
            target_date_offset = (self.validated_predictive_query.
                                  get_combined_date_offset_range())
            timeframe = target_date_offset.end - target_date_offset.start
            anchor_start_offset = target_date_offset.start
            unit = target_date_offset.unit
            offset_unit = target_date_offset.unit
        else:
            # Support TimeRangeSplit for static query. We do not support
            # DateOffsetSplit for static query for now because
            # 1. It is not requested by adoption pod
            # 2. For static queries, use cases for DateOffsetSplit can all be
            #    achieved with TimeRangeSplit.
            # 3. One advantage of DateOffsetSplit over TimeRangeSplit for
            # temporal query is that we can easily use DateOffsetSplit to
            # define the number of timeframes we want to include for each
            # split set, which is not relevant for static problems.
            assert not isinstance(split, DateOffsetSplit)
            timeframe = None
            self.timeframe_step = None
            unit = 'days'
            anchor_start_offset = None

        if isinstance(split, TimeRangeSplit):
            self.split = TimeRangeSplit(split.ranges, unit=unit,
                                        timeframe=timeframe,
                                        anchor_start_offset=anchor_start_offset,
                                        timeframe_step=self.timeframe_step,
                                        flexible_boundaries=False)
        elif isinstance(split, DateOffsetSplit):
            self.split = DateOffsetSplit(
                split.offsets, unit=unit, offset_unit=offset_unit,
                timeframe=timeframe, anchor_start_offset=anchor_start_offset,
                timeframe_step=self.timeframe_step, flexible_boundaries=False)
        elif isinstance(split, TemporalSplit):
            self.split = TemporalSplit(
                ratio=split.ratio,
                unit=unit,
                timeframe=timeframe,
                anchor_start_offset=anchor_start_offset,
            )
        else:
            raise ValidationException(
                title="Unsupported split type",
                message=(
                    "`_set_split` is not supported for "
                    f"{split.__class__.__name__}. Supported split types: "
                    "TimeRangeSplit, DateOffsetSplit, TemporalSplit, TimeSplit."
                ),
            )

    @blocking_method()
    def validate(self) -> ValidationResponse:
        # We don't require encoders to be present when constructing the
        # predictive query. However, encoders are required once we create the
        # task from the PQ (and we perform another validation at that point in
        # time)
        self.graph.validate(ignore_encoders=True)
        return self._validate_without_graph_io()

    def _validate_without_graph_io(self) -> ValidationResponse:
        """Validate only PQuery state that cannot access a provider."""
        response = ValidationResponse()
        response.merge(self.table_generator.validate_pre_train_table_generate())

        return response

    def _check_pquery_settings(self):
        task_type = self.get_task_type()
        query_type = self.validated_predictive_query.query_type

        suffix = ('Please go back and change your query '
                  'to fit a supported problem type.')

        if ((task_type == core_types.TaskType.STATIC_LINK_PREDICTION
             and self.pquery_settings.disable_static_link_prediction)
                or (task_type == core_types.TaskType.TEMPORAL_LINK_PREDICTION
                    and self.pquery_settings.disable_temporal_link_prediction)):
            raise ValidationException(
                title="Unsupported operator",
                message=("The `LIST_DISTINCT` operator is currently not "
                         f"supported in this version of Kumo. {suffix}"),
            )
        elif (task_type in {
                core_types.TaskType.REGRESSION,
                core_types.TaskType.FORECASTING,
        } and self.pquery_settings.disable_temporal_node_regression):
            raise ValidationException(
                title="Unsupported problem type",
                message=(
                    "Regression and Forecasting are currently not supported "
                    f"in this version of Kumo. {suffix}"),
            )
        elif (task_type == core_types.TaskType.MULTILABEL_CLASSIFICATION
              and self.pquery_settings.disable_multilabel_classification):
            raise ValidationException(
                title="Unsupported problem type",
                message=("Multilabel classification is currently not "
                         f"supported in this version of Kumo. {suffix}"),
            )
        elif (task_type == core_types.TaskType.MULTILABEL_RANKING
              and self.pquery_settings.disable_multilabel_ranking):
            raise ValidationException(
                title="Unsupported problem type",
                message=("Multilabel ranking is currently not "
                         f"supported in this version of Kumo. {suffix}"),
            )
        elif (task_type == core_types.TaskType.FORECASTING
              and self.pquery_settings.disable_forecasting):
            raise ValidationException(
                title="Unsupported problem type",
                message=("Forecasting is currently not supported "
                         f"in this version of Kumo. {suffix}"),
            )

        if query_type == QueryType.STATIC:
            if self.pquery_settings.disable_static_node_prediction:
                raise ValidationException(
                    title="Unsupported problem type",
                    message=(
                        "Static prediction problems are not supported "
                        f"in this version of Kumo. {suffix} You can use an "
                        "aggregation operator in your PREDICT clause to make "
                        "your problem temporal."),
                )

        final_aggr = self.validated_predictive_query.\
            get_final_target_aggregation()
        if (query_type == QueryType.TEMPORAL and final_aggr is not None):
            if (final_aggr.aggr == AggregationType.FIRST
                    or final_aggr.aggr == AggregationType.LAST):
                if self.pquery_settings.disable_first_last:
                    raise ValidationException(
                        title="Unsupported operator",
                        message=(f"{final_aggr.aggr} operator is currently not "
                                 "supported."),
                    )
                # FIRST/LAST operator on foreign key target is currently
                # blocked on the level of PredictiveQueryValidator
                target_col = self.graph.column_from_fqn(
                    final_aggr._get_target_column_name())
                if isinstance(target_col, ForeignKey):
                    if self.pquery_settings.disable_first_last_on_fkey:
                        raise ValidationException(
                            title="Unsupported operator",
                            message=(
                                f"{final_aggr.aggr} operator is currently not "
                                "supported on foreign key targets "
                                f"(`{target_col}`)."),
                        )

        if task_type == core_types.TaskType.FORECASTING:
            if final_aggr is not None and final_aggr.aggr not in [
                    AggregationType.AVG,
                    AggregationType.COUNT,
                    AggregationType.COUNT_DISTINCT,
                    AggregationType.SUM,
                    AggregationType.MIN,
                    AggregationType.MAX,
            ]:
                raise ValidationException(
                    title="Unsupported aggregation",
                    message=("Forecasting is currently only supported for SUM, "
                             "COUNT, COUNT_DISTINCT, MIN, MAX aggregations. "
                             "Change your target aggregation to one of these."),
                )
            aggregation_unit = (self.validated_predictive_query.
                                get_combined_date_offset_range().unit)
            if (aggregation_unit == TimeUnit.MONTHS):
                raise ValidationException(
                    title="Unsupported aggregation unit",
                    message=(
                        "Forecasting does not support month as aggregation "
                        "unit because different months have different number "
                        "of days. Consider using days instead."),
                )

    def __repr__(self) -> str:
        """
        String representation of the PredictiveQueryV2 class.
        """
        return (
            f'{self.__class__.__name__}(\n'
            f'  name={self.name},\n'
            f'  validated_predictive_query={self.validated_predictive_query},\n'  # noqa E501
            f'  graph={self.graph},\n'
            f')')

    @property
    def entity_table(self) -> str:
        """
        Returns the table name of the entity.
        """
        entity_table_name = table_name(
            self.validated_predictive_query.entity_column)
        return self.graph.fully_qualified_table_name(entity_table_name)

    @property
    def entity_table_names(self) -> list[str]:
        entity_table_names = [
            self.validated_predictive_query.entity_column.split('.')[-2],
        ]
        if self.get_task_type().is_link_pred:
            aggr = self.validated_predictive_query.\
                get_final_target_aggregation()
            fkey = self.graph.column_from_fqn(aggr._get_target_column_name())
            entity_table_names.append(fkey.dst_table.split('.')[-1])
        return entity_table_names

    def get_target_offset(self) -> Tuple[int, int]:
        r"""Returns the start and end offset of the combined target
        aggregation.
        """
        offset_range = (
            self.validated_predictive_query.get_combined_date_offset_range())
        assert not offset_range.is_open
        start, end = offset_range.start, offset_range.end
        assert isinstance(start, int)
        return start, end

    def get_bp_entity_col_name(self) -> str:
        """
        Gets the name of the entity col found in the batch prediction table
        based on whether it's PQLV2 or not
        """
        return TrainingTableGeneratorV2.ENTITY

    async def get_default_split_async(self) -> Split:
        """
        Async way to get the default split for the Predictive Query inferred
        based on the task type. Should be preffered in async functions so it
        doesn't block the async event loop.

        Returns:
            Split: The default split for the PredictiveQuery
        """
        task_type = get_task_type(self.graph, self.validated_predictive_query,
                                  forecast_length=self.forecast_length)
        if self.pquery_settings.use_decoupled_splits:
            return self._get_default_split_v2(task_type.task_type)
        else:
            return self._get_default_split(task_type.task_type)

    def get_default_split(self) -> Split:
        task_type = get_task_type(self.graph, self.validated_predictive_query,
                                  forecast_length=self.forecast_length)
        if self.pquery_settings.use_decoupled_splits:
            return self._get_default_split_v2(task_type.task_type)
        else:
            return self._get_default_split(task_type.task_type)

    def _get_default_split_v2(self, task_type: core_types.TaskType) -> Split:
        r"""Returns the default data split for this predictive query when
        using decoupled splits.
        This method is used when _USE_DECOUPLED_SPLITS is enabled. Currently
        only supports temporal queries with TimeSplit using 80/10/10 fractions.
        Args:
            task_type (TaskType): The type of ML task being performed
        Returns:
            Split: The default split configuration for this predictive query
        Raises:
            NotImplementedError: If task_type is temporal link prediction or
                forecasting.
            ValueError: If query type is not temporal
        """
        assert self.pquery_settings.use_decoupled_splits
        validated_query = self.validated_predictive_query
        query_type = validated_query.query_type
        if query_type == QueryType.STATIC:
            if task_type == core_types.TaskType.STATIC_LINK_PREDICTION:
                if self.list_distinct_target_is_array:
                    return RandomLinkSplit([0.8, 0.1, 0.1])
                # static link prediction should use temporal split over pseudo
                # time. However, we set RandomSplit here for better showing in
                # UI. We overwrite it to TemporalSplit inside Task.
                return RandomSplit([0.8, 0.1, 0.1])
            else:
                entity_table = table_name(validated_query.entity_column)
                if self.graph[entity_table].has_time_col():
                    return TimeSplit(fractions=[0.8, 0.1, 0.1])
                else:
                    return RandomSplit([0.8, 0.1, 0.1])

        elif query_type == QueryType.TEMPORAL:
            offset = validated_query.get_combined_date_offset_range()
            unit = offset.unit
            timeframe = offset.end - offset.start
            if task_type == core_types.TaskType.TEMPORAL_LINK_PREDICTION:

                # NOTE: When timeframe_step is smaller than offset.end, choose
                # `validation_set_offset = offset.end` is good because, the
                # validation set will be guaranteed to have one timeframe if
                # not all data are filtered out.
                # However, when timeframe_step is larger than offset.end, we
                # need to choose `validation_set_offset = self.timeframe_step`
                # to ensure that the validation set has at least one timeframe.
                validation_set_offset = offset.end
                if self.timeframe_step is not None:
                    validation_set_offset = max(validation_set_offset,
                                                self.timeframe_step)
                return TimeSplit(
                    offset_boundaries=[
                        ('-INF', -1 * offset.end - validation_set_offset),
                        (-1 * offset.end - validation_set_offset,
                         -1 * offset.end),
                        (-1 * offset.end, 0),
                    ],
                    unit=unit,
                    timeframe_size=timeframe,
                    anchor_start_offset=offset.start,
                )
            elif task_type == core_types.TaskType.FORECASTING:
                return TimeSplit(
                    indices=[('-INF', -2 * self.forecast_length),
                             (-2 * self.forecast_length, -self.forecast_length),
                             (-self.forecast_length, 0)],
                    unit=unit,
                    timeframe_size=timeframe,
                    anchor_start_offset=offset.start,
                )
            else:
                return TimeSplit(
                    fractions=[0.8, 0.1, 0.1],
                    unit=unit,
                    timeframe_size=timeframe,
                    anchor_start_offset=offset.start,
                )
        else:
            raise ValueError(
                f"No default split defined for query type {query_type}")

    def _get_default_split(self, task_type: core_types.TaskType) -> Split:
        r"""Returns the default data split for this predictive query.
        """
        validated_query = self.validated_predictive_query
        query_type = validated_query.query_type
        if query_type not in {QueryType.STATIC, QueryType.TEMPORAL}:
            raise ValueError(
                f"No default split defined for query type {query_type}")
        offset = (None if query_type == QueryType.STATIC else
                  validated_query.get_combined_date_offset_range())
        descriptor = default_split_for_query(
            task_type,
            query_is_static=query_type == QueryType.STATIC,
            entity_has_time_column=self.graph[table_name(
                validated_query.entity_column)].has_time_col(),
            list_distinct_target_is_array=self.list_distinct_target_is_array,
            offset_start=None if offset is None else offset.start,
            offset_end=None if offset is None else offset.end,
            unit="days" if offset is None else offset.unit,
            forecast_length=self.forecast_length,
            timeframe_step=self.timeframe_step,
        )
        if isinstance(descriptor, CoreDateOffsetSplit):
            split = DateOffsetSplit(**descriptor.to_split_config())
        else:
            split = get_split(repr(descriptor))
        assert split is not None
        return split

    def initialize_splits_post_train_table_generate(
        self,
        train_table_stats: Union[TemporalNodeTrainTableStats,
                                 TemporalLinkTrainTableStats],
    ):
        if (isinstance(self.split, ApproxDateOffsetSplit)
                and not self.split.is_initialized):
            offset = (self.validated_predictive_query.
                      get_combined_date_offset_range())
            unit = offset.unit
            timeframe = offset.end - offset.start

            # fully initially `ApproxDateOffsetSplit`
            self.split.set_date_offset_boundary(
                train_table_stats=train_table_stats, timeframe=timeframe,
                unit=unit, timeframe_step=self.timeframe_step)
        elif isinstance(self.split, DateOffsetSplit):
            # adjust `DateOffsetSplit` as necessary
            if self.query_type == QueryType.TEMPORAL:
                self.split = (
                    self.adjust_date_offset_split_post_train_table_generate(
                        train_table_stats))

    def create_task(
        self,
        persist_dir: str | None = None,
        train_table_path: str | None = None,
        task_path_override: str | None = None,
        job_id: str | None = None,
        job_dir: str | None = None,
        ignore_encoders: bool = False,
        weight_col_name: str | None = None,
        train_table_name: str | None = None,
    ) -> Task:
        """
        Creates a task from a PredictiveQuery which can be materialized to
        setup training table and features for the model.
        This method is responsible for the following:
        1. Generates a training table.
        2. Define train/val/test split config.

        Args:
            persist_dir (string, optional): Training table is saved here as
                `train_table.parquet`. if :obj:`None` it won't be
                saved. (default: :obj:`None`)
            train_table_path (str): Path where the training table is located.
            task_path_override (str, optional): Path to the task config file.
                If not provided, task will be generated by pquery.
                If specified, pquery will use the override path to load the
                task, but will still use the pquery to generate the training
                table. (default: :obj:`None`)
            job_id (str, optional): Job id for retrieving table table stats.
                (default: :obj:`None`)
            job_dir (str, optional): directory for storing training-job
                specific artifacts such as split table stats. Sometimes known
                as home_dir (default: :obj:`None`)
            weight_col_name (str, optional): Name of the weight column in
                the training table. This is currently implemented via
                the customer adding a weight col and supplying it to kumo.
                The custom table is ingested under the training
                job dir.(default: :obj:`None`)
            train_table_name (str, optional): Logical training-table name used
                in the generated task metadata. (default: :obj:`None`)

        Returns:
            A kumo.Task object initialized with graph, training table, etc.
        """
        from kumo.task import Task

        # Require that encoders are present before creating the task
        self.graph.validate(ignore_encoders=ignore_encoders)
        if train_table_path is None:
            train_table_path, job_id, self._validate_train_table_response = (
                self.generate_train_table(job_id=job_id))
        self._train_table_path = train_table_path

        # Note: when `self._validate_train_table_response` contains more than
        # one errors, we will always raise the first one.
        if ((self._validate_train_table_response is not None)
                and (not self._validate_train_table_response.ok)):
            validation_error = self._validate_train_table_response.errors[0]
            raise TrainingTableGenerationError(title=validation_error.title,
                                               message=validation_error.message)

        # Splitting ###########################################################

        self.get_train_table_stats(job_id)

        # TT-NG assigns every row in pass B. Its stage tables and split_info
        # counts/bounds are authoritative; rebuilding DateOffset or
        # ApproxDateOffset from diagnostic stats can change that assignment.

        #######################################################################

        # The generated result is the physical ``tt.parquet`` selected from
        # an immutable bundle. Keep its logical task name independent from
        # that storage filename.
        path_to_train_table = osp.dirname(self._train_table_path)
        train_table_name = train_table_name or TRAIN_TABLE_NAME
        physical_train_table_name = osp.splitext(
            osp.basename(self._train_table_path))[0]
        train_table_connector = ParquetConnector(
            name=train_table_source_name(path_to_train_table, weight_col_name
                                         is not None),
            root_dir=path_to_train_table)

        task_type = self.get_task_type()

        weight_col = None
        if weight_col_name:
            weight_col = validate_weight_column(
                weight_col_name, self._train_table_path, task_type,
                self.list_distinct_target_is_array)
        elif train_table_name != TRAIN_TABLE_NAME:
            raise ValueError(
                f"Unexpected training table name '{train_table_name}' "
                "without a weight column.")

        # NOTE: Right now `target_table_name` is only used in static link
        # prediction for inserting pseudo time in target table
        target_table_name = None
        task_type = self.get_task_type()
        if task_type == core_types.TaskType.STATIC_LINK_PREDICTION:
            target_table_name = self.table_generator.get_target_table_name()

        unix_start_offset = unix_end_offset = unix_step_size = None
        if self.query_type == QueryType.TEMPORAL:
            offset = self.target.date_offset_range
            start = offset.start or 0
            end = offset.end
            step_size = self.timeframe_step or (end - start)

            unix_start_offset = compute_unix_step(start, offset.unit)
            unix_end_offset = compute_unix_step(end, offset.unit)
            unix_step_size = compute_unix_step(step_size, offset.unit)

        self.task = Task(
            self.graph,
            task_name=self.name,
            task_type=task_type,
            train_table_name=train_table_name,
            train_table_source_name=physical_train_table_name,
            data_source=train_table_connector,
            target_col=self.table_generator.get_target_col(),
            fkeys=self.table_generator.get_fkeys(),
            cols=self.table_generator.get_cols(),
            time_col=self.table_generator.get_time_col(),
            entity_include_bit_col=self.table_generator.
            get_entity_include_bit_col(),
            weight_col=weight_col,
            stats_manager=self.table_generator.stats_manager,
            forecast_length=self.forecast_length,
            unix_start_offset=unix_start_offset,
            unix_end_offset=unix_end_offset,
            unix_step_size=unix_step_size,
            split=self.split,
            target_to_entity_paths=self.target_to_entity_paths,
            target_table_name=target_table_name,
            is_sliding_window=False,
            lag_timesteps=self.lag_timesteps,
            job_id=job_id,
            job_dir=job_dir,
            list_distinct_target_is_array=self.list_distinct_target_is_array,
            lp_bp_layout=self.lp_bp_layout,
            ignore_encoders=ignore_encoders,
        )

        # Save all columns that leak information during training:
        self.task.exclude_cols = self.table_generator.get_exclude_cols()

        if task_path_override:
            self._override_task_path(task_path_override)

        return self.task

    def generate_current_pred_table(
        self,
        job_id: str | None = None,
        training_job_id: str | None = None,
        pred_time: core_types.PredTime | Timestamp | None = None,
        diskgraph_index_path: str | None = None,
        destination: str | None = None,
    ) -> Tuple[str, ValidationResponse]:
        """
        Generates current prediction table data as a
        :class:`pandas.DataFrame` of the :class:`~kumo.task.TrainingTable`.
        Returns the path to the generated prediction table.

        Args:
            job_id (str, optional): job id for cache of pred table stats
            training_job_id (str, optional): training job id in the training
                stage.
            pred_time (PredTime or pandas.Timestamp, optional): The timestamp
                to make predictions for (in case of a temporal task).
                If set to :obj:`None`, will use the maximum timestamp in the
                training data as prediction time. (default: :obj:`None`)
            diskgraph_index_path (str, optional): Path to the diskgraph index.
                (default: :obj:`None`)
        """
        if isinstance(pred_time, str):
            pred_time = core_types.PredTime(pred_time.lower())

        # Resolve the destination once here (the sole get_home_dir() fallback
        # for this path); everything below receives a concrete base dir.
        destination = destination or get_home_dir()

        pred_table_path = osp.join(
            get_table_path(self.name, destination, training_table=False),
            PRED_TABLE_NAME)
        if fsutil.exists(pred_table_path):
            logger.info(f"Prediction table already exists at {pred_table_path}"
                        ". Local test may fail but workflow will reattach to "
                        "to existing table generation job.")
        fsutil.makedirs(pred_table_path, exist_ok=True)

        from kumo.data.generic.generate_current_pred_table import (
            TARGET_FILE_NAME,
            generate_current_pred_table_generic,
        )
        from kumo.validation import ValidationResponse, train_table_validation

        # Remove pandas connector from graph (cannot be serialized).
        # TODO(kiriti): Remove this once we have a better way to
        # handle pandas connectors.
        self._remove_pandas_connector_from_graph()

        pquery_config = self.config()
        # Validation response + target sidecar live alongside the pred-table
        # directory so downstream readers find them at a stable location.
        output_file_path = pred_table_path
        pred_table_validation_response_path = osp.join(
            osp.dirname(output_file_path),
            train_table_validation.PRED_TABLE_VALIDATION_RESPONSE_NAME)
        target_file_path = osp.join(osp.dirname(output_file_path),
                                    TARGET_FILE_NAME)

        if not job_id:
            job_id = uuid.uuid4().hex

        generate_current_pred_table_generic(
            pquery_config,
            output_file_path,
            pred_table_validation_response_path,
            destination,
            job_id=job_id,
            training_job_id=training_job_id,
            pred_time=pred_time,
            target_file_path=target_file_path,
            diskgraph_index_path=diskgraph_index_path,
        )

        validation_response = ValidationResponse.load(
            pred_table_validation_response_path,
            train_table_validation.PRED_TABLE_VALIDATION_RESPONSE_NAME)

        if not validation_response.ok:
            return pred_table_path, validation_response

        self._add_pandas_connector_back_to_graph()

        self._pred_table_path = pred_table_path
        return self._pred_table_path, validation_response

    def generate_train_table(
        self,
        job_id: str | None = None,
        diskgraph_index_path: str | None = None,
        destination: str | None = None,
    ) -> Tuple[str, str, ValidationResponse]:
        """
        Generates the training table parquet via the DiskGraph producer and
        sets ``self.train_table_path`` to its on-disk location.

        Args:
            job_id (str, optional): job id for indexing cache of pred table
                stats
            destination (str, optional): Base dir the output table + intermediate
                staging are written under. :obj:`None` falls back to the
                process-global :func:`kumo.get_home_dir`.
        """
        from kumo.data.generic import generate_train_table
        from kumo.validation import ValidationResponse, train_table_validation

        logger.debug(
            "Generating train table with config: weight_col=%s "
            "list_distinct_target_is_array=%s query_type=%s",
            self.config().weight_col,
            self.list_distinct_target_is_array,
            self.query_type,
        )

        # Resolve the destination once here (the sole get_home_dir() fallback
        # for this path); everything below receives a concrete base dir.
        destination = destination or get_home_dir()

        pquery_config = self.config()
        train_table_path = osp.join(
            get_table_path(pquery_config.name, destination), TRAIN_TABLE_NAME)
        train_table_validation_response_path = osp.join(
            osp.dirname(train_table_path),
            train_table_validation.TRAIN_TABLE_VALIDATION_RESPONSE_NAME)

        if not job_id:
            job_id = uuid.uuid4().hex

        published_train_table_path = generate_train_table.generate_train_table_generic(
            pquery_config,
            train_table_path,
            destination,
            train_table_validation_response_path=(
                train_table_validation_response_path),
            job_id=str(job_id),
            diskgraph_index_path=diskgraph_index_path,
        )

        validation_response = ValidationResponse.load(
            train_table_validation_response_path,
            train_table_validation.TRAIN_TABLE_VALIDATION_RESPONSE_NAME)

        if not validation_response.ok:
            return '', str(job_id), validation_response
        if published_train_table_path is None:
            raise RuntimeError(
                'train table generation succeeded without publishing data')

        # TODO (xinwei): Deprecate this and adjust the unit tests
        # NOTE: This is needed to store a copy of train table stats in
        # memory. This is useful for unit testing and adhoc scripts.
        self.get_train_table_stats(job_id)

        validate_post_response = (
            self.table_generator.validate_post_train_table_generate())
        if validate_post_response is not None:
            validation_response.merge(validate_post_response)
        # The task keeps the logical table name stable while reading the
        # immutable bundle's physical ``tt.parquet`` source.
        self._train_table_path = published_train_table_path
        return published_train_table_path, job_id, validation_response

    @property
    def connectors(self) -> List[Connector]:
        """
        Returns the connectors of the :obj:`entity`, :obj:`target`,
        :obj:`temporal_entity_filter`, and :obj:`whatif` (if present)
        tables.
        """
        raise NotImplementedError

    def adjust_date_offset_split_post_train_table_generate(
        self, train_table_stats: Union[TemporalNodeTrainTableStats,
                                       TemporalLinkTrainTableStats]
    ) -> DateOffsetSplit:
        r"""When a custom split is not provided, and that the inferred
        default split is `DateOffsetSplit`, we will adjust the split
        boundary to make sure validation set is not empty even with
        restrictive temporal entity filters.

        Args:
            train_table_stats: (Union[TemporalNodeTrainTableStats,
                TemporalLinkTrainTableStats]): The temporal train table stats.
        Returns:
            (DateOffsetSplit): The adjusted `DateOffsetSplit` method.
        """
        split = self.split
        assert isinstance(split, DateOffsetSplit)

        offset = self.target.date_offset_range
        unit = offset.unit

        # TODO: Support boundary adjustment for custom `DateOffsetSplit`,
        # where `offset.start == 0` and `self.timeframe_step is not None`.
        # NOTE: This is a temporary logic to determine whether a
        # `DateOffsetSplit` is set to the default value, i.e. using the last
        # timeframe as test set and second last timeframe as validation set.
        # We will remove this logic soon.
        is_default_split = False
        if (len(split.offsets) == 2 and split.offsets[-1] == 0
                and -split.offsets[0] == split.timeframe):
            is_default_split = True

        # When the current `DateOffsetSplit` is not a default split and
        # when `offset.start` > 0 or `timeframe_step is not None`,
        # under the current default split inferral logic train/val/test
        # timeframes are not consecutive to prevent potential leakages
        # between splits, as a result it is more chanllenging to adjust
        # date offset split in this case. We will improve this in the
        # future. For now, we will skip date offset split adjustment for
        # this case.
        if (not is_default_split or offset.start > 0
                or self.timeframe_step is not None):
            return split

        # Adjust the split boundary between train/val set such that
        # val set is always not empty when at least 2 timeframes are
        # observed.
        offset = self.target.date_offset_range
        unit = offset.unit
        timeframe = offset.end

        stats_per_timeframe = train_table_stats.stats_per_timeframe
        val_timeframe_index, test_timeframe_index = (
            stats_per_timeframe[1].index,
            stats_per_timeframe[0].index,
        )
        adjusted_timeframe = (val_timeframe_index -
                              test_timeframe_index) * timeframe

        # NOTE: the test set is actually not empty because
        # max sample timestamp is one timeframe larger than the max
        # data timestamp. Hence, setting the test set edge to be 0
        # means that the test set will contain exactly all the
        # samples with the largest timestamp.
        adjust_split = DateOffsetSplit(
            offsets=[-adjusted_timeframe, 0],
            offset_unit=unit,
            timeframe=timeframe,
            timeframe_step=self.timeframe_step,
            anchor_start_offset=split.anchor_start_offset,
            flexible_boundaries=split.flexible_boundaries,
        )

        if adjust_split != split:
            split_str = construct_object_string_from_config(split)
            adjust_split_str = construct_object_string_from_config(adjust_split)
            validation_warning = ValidationWarning(
                title=("DateOffsetSplit is adjusted to avoid empty "
                       "validation data"),
                message=(
                    f"Validation set would be empty with split {split_str}, "
                    f"so we adjusted it to {adjust_split_str} to keep all "
                    f"splits non-empty. Edit the split and retrain if you "
                    f"prefer a different setup."))
            self._validate_train_table_response.warnings.append(
                validation_warning)
            logger.warning(validation_warning.message)
        return adjust_split

    @classmethod
    def load(cls, name: str,
             home_dir: Optional[str] = None) -> 'PredictiveQuery':
        home_dir = home_dir if home_dir is not None else get_home_dir()
        root_dir = osp.join(home_dir, 'queries', name)

        # TODO: Load training table DataFrame.
        path = osp.join(root_dir, 'config.json')
        cfg = load_json(path, cls=PredictiveQueryV2Config)

        return cls.from_config(cfg)
