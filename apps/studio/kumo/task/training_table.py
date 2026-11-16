# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Disable yapf to avoid confict with isort
# yapf: disable
import os.path as osp
from typing import Any

import pandas as pd
from kumo_core.types import TaskType
from kumo_core.types.schema import TableFingerprint, TrainingTableFingerprint
from kumo_core.utils.registry import register
from kumoapi.train import TrainingStage

from io_utils import json, logging
from kumo.config import TableMetadataConfig, TrainingTableConfig
from kumo.config.task import TimePeriod
from kumo.connector import Connector
from kumo.connector.pandas_connector import PandasConnector
from kumo.data import get_active_cache
from kumo.data.backend_cache import BackendCacheManager
from kumo.graph import BoolColumn, Column, ForeignKey, Table, TimeColumn
from kumo.graph.column import BaseColumn
from kumo.graph.utils import persist_pandas_connector
from kumo.home import HomeDirManager, get_home_dir
from kumo.stats import Stats
from kumo.stats.train_table_stats import (
    SplitTrainTableStats,
    TrainTableStats,
    load_train_table_stats_from_cache,
    train_table_split_stats_from_job_dir,
)
from kumo.task.split import RandomSplit, Split
from kumo.validation.train_table_validation import TrainingTableSplitError
from kumo.validation.validation import ValidationException

# yapf: enable

logger = logging.getLogger(__name__)


@register(data_cls=TrainingTableConfig)
class TrainingTable(Table):
    r"""Describes a KUMO training table, consisting of a data source,
    a primary key, foreign keys and column information.

    Args:
        table_name (str): The name of the table.
        connector (Connector): The connector to the underlying data source
            of the training table. This can be any connector, including the
            :obj:`kumo.connector.pandas_connector.PandasConnector`.
        task_name (str): The name of the task associated with the
            training table.
        target_col (Column, optional): The target column to predict.
            Can be :obj:`None` for link prediction.
        fkeys (List[ForeignKey]): The foreign keys of the table.
            At least one foreign key needs to be defined.
        cols (List[Column], optional): The columns of the table.
            (default: :obj:`None`)
        time_col (TimeColumn, optional): The time column of the table.
            (default: :obj:`None`)
        weight_col (BaseColumn, optional): The weight column of the table,
            indicating the sampling probability of training examples.
            Does not necessarily need to sum up to one. (default: :obj:`None`)
        split (Split, optional): The splitting scheme of the table.
            If set to :obj:`None`, will use a 80%/10%/10% random split.
            (default: :obj:`None`)
        task_type (str): Type of the prediction task. Currently
            supported options are: :obj:`"multiclass_classification"` ,
            :obj:`"binary_classification"`, :obj:`"multilabel_classification"`,
            :obj:`"regression"`, :obj:`"static_link_prediction"` or
            :obj:`"temporal_link_prediction"`.
        job_id (str, optional): Job id for storing split table stats.
            (default: :obj:`None`)
        is_pred (bool): Whether the training table object is
            actually a prediction table. Required since we currently use this
            class to represent both training and prediction tables.
        job_dir (str, optional): directory for storing training-job specific
            artifacts such as split table stats. Sometimes known as home_dir
            (default: :obj:`None`)
    """
    def __init__(
        self,
        table_name: str,
        connector: Connector,
        task_name: str | None = None,
        target_col: Column | None = None,
        fkeys: list[ForeignKey] | None = None,
        cols: list[Column] | None = None,
        time_col: TimeColumn | None = None,
        weight_col: BaseColumn | None = None,
        stats_manager: Any | None = None,
        split: Split | None = None,
        source_name: str | None = None,
        # Required to load from TrainingTableConfig since they need to be
        # passed into the super init rather than added later in from_config
        data_version: str | None = None,
        pkey: Any | None = None,
        stats: Any | None = None,
        data_source: Any | None = None,
        task_type: TaskType | None = None,
        job_id: str | None = None,
        is_pred: bool = False,
        job_dir: str | None = None,
        entity_include_bit_col: BoolColumn | None = None,
    ):
        self._connector = connector
        fkeys = fkeys or []
        self._target_col = Column.cast(target_col)
        self._weight_col = BaseColumn.cast(weight_col)
        self._entity_include_bit_col = entity_include_bit_col
        self.is_pred = is_pred
        self.job_id = job_id
        self.job_dir = job_dir
        data_source = connector.name

        super().__init__(
            table_name,
            data_source,
            fkeys=fkeys,
            cols=cols,
            time_col=time_col,
            source_name=source_name,
            data_version=data_version,
            pkey=pkey,
            stats=stats,
        )

        if len(self.fkeys) < 1:
            raise ValidationException(
                "Training table must contain at least one foreign "
                "key pointing to a graph entity table. "
                "Add a foreign key in your predictive query or "
                "training table definition so rows can be "
                "linked into the graph.")

        if split is None:
            logger.info("No split defined - using a 80%/10%/10% random split")
            split = RandomSplit([0.8, 0.1, 0.1])
        self.split = split
        self.task_name = task_name
        self.connector = connector
        self.task_type = task_type

        self.stats_manager = stats_manager

        self._train_table_stats = None
        self._split_train_table_stats = None

    @classmethod
    def customize_from_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        # `end_time_col` argument does not exist at the training table level.
        values.pop('end_time_col', None)
        return values

    def __repr__(self) -> str:
        fkey_names = str(self.fkey_names).replace("'", "")
        col_names = str(self.col_names).replace("'", "")
        return (f'{self.__class__.__name__}(\n'
                f'  name={self.name},\n'
                f'  data_source={self.data_source},\n'
                f'  target_col={self.target_col_name or "???"},\n'
                f'  fkeys={fkey_names},\n'
                f'  cols={col_names},\n'
                f'  time_col={self.time_col_name or "???"},\n'
                f'  weight_col={self.weight_col or "???"},\n'
                f'  entity_include_bit_col='
                f'{self.entity_include_bit_col or "???"},\n'
                f')')

    @property
    def target_col(self) -> Column | None:
        return self._target_col

    @property
    def data_source(self) -> str:
        return self.connector.name

    @property
    def target_col_name(self) -> str | None:
        return None if self.target_col is None else self.target_col.name

    def has_target_col(self) -> bool:
        return self.target_col is not None

    def is_target_col(self, name: str) -> bool:
        return self.target_col is not None and self.target_col.name == name

    @property
    def entity_include_bit_col(self) -> BoolColumn | None:
        return self._entity_include_bit_col

    @property
    def entity_include_bit_col_name(self) -> str | None:
        return (None if self.entity_include_bit_col is None else
                self.entity_include_bit_col.name)

    def has_entity_include_bit_col(self) -> bool:
        return self.entity_include_bit_col is not None

    def is_entity_include_bit_col(self, name: str) -> bool:
        return (self.entity_include_bit_col is not None
                and self.entity_include_bit_col.name == name)

    @property
    def weight_col(self) -> BaseColumn | None:
        return self._weight_col

    @property
    def weight_col_name(self) -> str | None:
        return None if self.weight_col is None else self.weight_col.name

    def has_weight_col(self) -> bool:
        return self.weight_col is not None

    def is_weight_col(self, name: str) -> bool:
        return self.weight_col is not None and self.weight_col.name == name

    def fingerprint(self) -> TableFingerprint:
        return TrainingTableFingerprint(
            connector_name=self.data_source,
            source_name=self.table_name,
        )

    def __contains__(self, name: str) -> bool:
        return (super().__contains__(name) or self.is_target_col(name)
                or self.is_entity_include_bit_col(name))

    def __getitem__(self, name: Any) -> Any:
        if self.is_target_col(name):
            return self.target_col
        elif self.is_entity_include_bit_col(name):
            return self.entity_include_bit_col
        return super().__getitem__(name)

    # Materialization #########################################################

    def validate(self, raise_on_error: bool = True) -> bool:
        status = super().validate(raise_on_error)

        if self.target_col is not None:
            status &= self.target_col.validate(raise_on_error)

        return status

    @property
    def _data_cols(self) -> list[BaseColumn]:
        cols = super()._data_cols
        if self.target_col is not None:
            cols.append(self.target_col)
        if self.weight_col is not None:
            cols.append(self.weight_col)
        if self.entity_include_bit_col is not None and not self.is_pred:
            cols.append(self.entity_include_bit_col)
        return cols

    def prediction_horizon(self, stage: TrainingStage) -> TimePeriod | None:
        """Return a prediction horizon for train/val/test stage.

        Args:
            stage(TrainingStage): Valid values are :obj:`TRAIN`, :obj:`TEST` or
                :obj:`VAL`.
        """
        # Derived from the per-stage stats of the materialized rows, never
        # from the split spec: any split that produced stage row groups has
        # a horizon, whether it drew the boundaries from the time column or
        # read them out of a user-filled column. The absence of a horizon is
        # reported by the `min_date` check below, which is the real
        # precondition -- per-stage time stats must exist.
        split_table_stats = self.get_train_table_stats(after_split=True)
        if split_table_stats is None:
            logger.error(
                f"Expect to have train table stats for {stage.value} set "
                f"for retrieving prediction horizon.")
            return
        stage_table_stats = split_table_stats.get_stage_table_stats(stage)
        assert stage_table_stats is not None

        if stage_table_stats.min_date is None:
            return None
        return TimePeriod.cast(stage_table_stats.min_date,
                               stage_table_stats.max_date, is_pseudo=False)

    def time_period(self, stage: TrainingStage) -> TimePeriod:
        """Returns a time period of train/val/test anchor time.
        Requires per-stage time stats on the materialized training table.

        Args:
            stage(TrainingStage): Valid values are :obj:`TRAIN`, :obj:`TEST` or
                :obj:`VAL`.
        """
        # Looked up, not derived: `time_periods` is precomputed per stage
        # when the training table is materialized, so the precondition is
        # that those stats exist -- not that the split spec happens to name
        # a time-based family. The lookup below already names that
        # precondition when it fails.
        result = self.data.time_periods.get(stage)
        if result is None:
            raise ValidationException(
                f"Time period for stage '{stage}' not found. Available stages are '{list(self.data.time_periods.keys())}'. "
                "Ensure the training table has a real time column configured, "
                "that time values span enough range for multiple splits, and that your predictive query filters "
                "do not remove all rows from that stage.")
        return result

    def persist_pandas_connector(self, destination: str | None = None):
        if isinstance(self.connector, PandasConnector):
            base = destination or get_home_dir()
            root_dir = osp.join(base, 'tasks', self.task_name)
            self.connector = persist_pandas_connector(self.connector,
                                                      root_dir=root_dir)

    def compute_feature_column_exact_stats(self, **kwargs):
        if kwargs.get('skip_compute_stats', False):
            logger.info(
                'Skipping exact stats computation for training table materialization'
            )
            return

        # Convert the PandasConnector to ParquetConnector if present.
        self.persist_pandas_connector()
        # Remove cached data version for previously generated train/pred
        # tables. This is needed for running some of the tests which
        # performs 2 batch predictions in a single test. This also makes
        # sure the process worker for task materialization does not get
        # stale train/pred table version.
        get_active_cache().table_version_cache.remove(self.connector.name)

        cols = self.cols
        if self.has_target_col():
            cols = cols + [self.target_col]
        is_prediction = kwargs.get('prediction', False)
        # We don't have entity_include_bit_col during prediction
        # Instead, for each row in prediction table, we have AUTOREGRESSION_COL
        if self.has_entity_include_bit_col() and not is_prediction:
            cols = cols + [self.entity_include_bit_col]

        # Compute all exact stats on the feature columns of this table during
        # training. During prediction, no need to compute stats.
        if not is_prediction:
            self.compute_exact_stats(is_minimal=False, cols=cols)

    def materialize(self, **kwargs) -> 'TrainingTable':
        from kumo.graph.table_data import TableData

        if self.has_data:
            return self

        self.validate(raise_on_error=True)

        self.compute_feature_column_exact_stats(**kwargs)

        self._data: TableData = TableData(
            self,
            data_version=self.data_version,
        )
        # Initialize in-memory attributes needed for persistence.
        # This is at most an empty ``TableData`` metadata shell (num_nodes +
        # fkey pointers); TT-NG never populates the whole training table (or
        # its stamped train/val/test masks) in memory. The DiskGraph bundle /
        # wheel iterator is the entire training data plane, and the split is
        # owned by the DiskGraph producer (native split-stats sidecar). The
        # legacy whole-table ``execute_split`` mask machinery is dead on
        # TT-NG, so ``materialize`` never runs it.
        self._data.persist_table(self)

        return self

    # Configuration ###########################################################

    def metadata_config(self) -> TableMetadataConfig:
        return TableMetadataConfig(
            table_name=self.table_name,
            source_name=self.source_name,
            data_source=self.data_source,
            data_version=self.data_version,
            pkey=self.pkey.config() if self.pkey else None,
            fkeys=[fkey.config() for fkey in self.fkeys],
            cols=[
                col.base_config() for col in (self.cols + [self.target_col])
                if col is not None
            ],
            time_col=self.time_col.config() if self.time_col else None,
        )

    # Statistics ###########################################################
    def _get_date_offset(self) -> tuple[pd.Timedelta, pd.Timedelta] | None:
        r"""
        Returns:
            A pair of pd.Timestamp objects representing the start and the
            end offset of target aggregation when train table type is
            temporal, None, otherwise.
        """
        table_stats = self.get_train_table_stats(after_split=False)
        if table_stats is not None and table_stats.stats_type.is_temporal:
            first_timeframe_stats = table_stats.stats_per_timeframe[0]
            min_date = pd.Timestamp(first_timeframe_stats.min_date)
            max_date = pd.Timestamp(first_timeframe_stats.max_date)
            anchor_date = pd.Timestamp(first_timeframe_stats.anchor_date)
            return min_date - anchor_date, max_date - anchor_date
        return None

    def _update_and_save_stats_cache(
            self, split_train_table_stats: SplitTrainTableStats):
        cache: BackendCacheManager = get_active_cache()

        # Include split train table stats
        stats = {
            "train_table_stats":
            Stats(split_train_table_stats.train_table_stats, approx=False),
            "val_table_stats":
            Stats(split_train_table_stats.val_table_stats, approx=False),
        }
        if split_train_table_stats.test_table_stats is not None:
            stats['test_table_stats'] = Stats(
                split_train_table_stats.test_table_stats, approx=False)

        if self.job_id is not None:
            # TODO(frg100): Can we remove this and use it in the job dir?
            cache.training_table_split_stats_cache.update(self.job_id, stats)

    def _save_split_train_table_stats_to_job_dir(
            self, split_train_table_stats: SplitTrainTableStats):
        if self.job_dir is not None:
            path = HomeDirManager(self.job_dir).split_train_table_stats(
                self.split_fpid)
            json.save_json(split_train_table_stats, path)

    def _save_train_table_stats_to_job_dir(self,
                                           train_table_stats: TrainTableStats):
        if self.job_dir is not None:
            path = HomeDirManager(self.job_dir).train_table_stats
            json.save_json(train_table_stats, path)

    def get_train_table_stats(
            self,
            after_split: bool) -> TrainTableStats | SplitTrainTableStats | None:
        r"""Gets important statistics of the generated
        :class:`~kumo.task.TrainingTable`.
        Args:
            after_split (bool): Whether stats is generated after the
                splitting of train table.
        """
        # NOTE: This is to makes sure pandas backend does not break as
        # we do not save train table stats for pandas backend at this
        # moment.
        if self.job_id is None:
            return None

        if after_split:
            if self._split_train_table_stats is None:
                self._split_train_table_stats = (
                    SplitTrainTableStats.from_cache(self.job_id))
                # Producer (`execute_split`) and consumer (e.g.
                # materialization activities run in different
                # sub-processes under `@process_pool_activity_defn`, so the
                # in-process `BackendCacheManager` write isn't visible to
                # the consumer's `from_cache` read. The on-disk JSON
                # written by `_save_split_train_table_stats_to_job_dir` is
                # the cross-process source of truth; fall back to it when
                # the cache misses and `job_dir` is set.
                if (self._split_train_table_stats is None
                        and self.job_dir is not None):
                    self._split_train_table_stats = (
                        train_table_split_stats_from_job_dir(
                            self.job_dir, self.split_fpid))
            return self._split_train_table_stats
        else:
            if self._train_table_stats is None:
                self._train_table_stats = load_train_table_stats_from_cache(
                    self.job_id)
            return self._train_table_stats

    @property
    def split_fpid(self) -> str:
        return self.split.fpid

    def save_data(self, path: str):
        """Save the current data in _data to the given path.

        Args:
            path (str): The path to save the data to.
        """
        self._data.save_data(path)

    def load_data(self, path: str):
        """Load the data from the given path.

        Args:
            path (str): The path to load the data from.
        """
        from kumo.graph.table_data import TableData
        self._data = TableData.__new__(TableData)
        self._data.load_data(
            path,
            pkey=self.pkey,  # Should always be None.
            fkeys=self.fkeys,
        )

    # Validation ###########################################################

    def validate_training_table_stats(
            self, stages: list[TrainingStage],
            split_train_table_stats: SplitTrainTableStats):
        r""" Validate training table statistics
        Args:
            stages (TrainingStage): The set of stages to validate stats for;
                `TrainingStage.TRAIN`, `TrainingStage.VAL` or
                `TrainingStage.TEST`.
            split_train_table_stats (SplitTrainTableStats): Training
                table stats.
        """
        logger.info(f"Generated training table split stats:\n"
                    f"{split_train_table_stats} for task {self.task_type}")
        if self.task_type in [
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
                TaskType.MULTILABEL_RANKING,
        ]:
            if not split_train_table_stats:
                raise TrainingTableSplitError(
                    'Split stats unavailable',
                    f'Failed to compute training table split stats for {self.task_type} task. '
                    "Re-generate the training table, and verify the target column is present and non-empty."
                )
            for stage in stages:
                stats = split_train_table_stats.get_stage_table_stats(stage)
                if not stats:
                    # todo: return actual validation class with warnings/errors
                    logger.warning(
                        f'Training split stats for stage {stage} missing.')
                    continue
                if stats.num_targets == 1:
                    target_counts = (stats.target_distribution.get('count', {})
                                     if stats.target_distribution is not None
                                     else {})
                    if target_counts:
                        # In the majority of cases, num_targets == 1 means
                        # that there is only one unique target value.
                        target_val = list(target_counts.keys())[0]
                        raise TrainingTableSplitError(
                            title="Detected imbalanced split",
                            message=  # noqa: E251
                            (f"In training table splitting, {stage} set "
                             f"has only one target value ('{target_val}'). "
                             f"Split: {self.split}, and {stage} table statistics: {stats}.\n"
                             "Modify the source data, adjust the predictive query filters, "
                             "or change the split to include multiple target values."
                             ),
                        )
                    else:
                        raise TrainingTableSplitError(
                            title="Detected imbalanced split",
                            message=  # noqa: E251
                            (f"In training table splitting, {stage} set has "
                             "only one target value; its value is unavailable "
                             "from the bounded split statistics.\n"
                             f"Modify the source data, split configuration ('{self.split}'), "
                             "or predictive query to include multiple target values."
                             ),
                        )
