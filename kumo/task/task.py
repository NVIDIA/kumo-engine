# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os.path as osp
from io import BytesIO
from typing import Any

import pandas as pd
from kumo_core.encoder_selection import select_target_encoder
from kumo_core.encoders.config import EncodersConfig
from kumo_core.resolved_task import ResolvedNumericalTask, ResolvedTaskSampling
from kumo_core.task_dataset import DiskgraphTaskManifest
from kumo_core.train.autoregression import (
    AUTOREGRESSION_COL,
    AUTOREGRESSION_SUFFIX,
    FUTURE_LABELS_COL,
    PADDING_SIZE,
    YOY_COL,
)
from kumo_core.types import ColStatType, TaskType
from kumo_core.utils.names import get_table_name
from kumo_core.utils.registry import register
from kumo_core.version import get_kumo_version
from kumoapi.encoder import (
    Encoder,
    EncoderType,
    Index,
    MultiCategorical,
    Numerical,
    NumericalList,
)
from kumoapi.typing import Dtype, Stype
from pydantic import TypeAdapter
from torch_geometric.typing import EdgeType

from io_utils import (
    AllowArbitraryTypesConfig,
    fsutil,
    logging,
    pydantic_dataclass,
)
from io_utils.json import load_json, save_json
from kumo.config import ConfigMixin, TaskConfig
from kumo.connector import Connector
from kumo.connector.pandas_connector import PandasConnector
from kumo.data import get_active_cache
from kumo.graph import BoolColumn, Column, ForeignKey, Graph, Table, TimeColumn
from kumo.graph.column import BaseColumn, EncoderGenerator
from kumo.graph.mixin import _HasData
from kumo.graph.utils import materialize_encoders
from kumo.stats.stats import Stats
from kumo.task.split import (
    DateOffsetSplit,
    FixedSplit,
    ManualSplit,
    RandomLinkSplit,
    RandomSplit,
    Split,
    TemporalSplit,
    TimeRangeSplit,
)
from kumo.task.splits.time_split import TimeSplit
from kumo.task.training_table import TrainingTable
from kumo.utils.visualization import visualize_graph

logger = logging.getLogger(__name__)


def _lp_prediction_layout_uses_target_sidecar(
    list_distinct_target_is_array: bool | None,
    lp_bp_layout: str | None,
) -> bool:
    if lp_bp_layout is None:
        return bool(list_distinct_target_is_array)
    return lp_bp_layout == 'separate_lhs_rhs'


# The name of the default PandasConnector created by the Task class, if no
# data source name for the training table is provided.
DEFAULT_PANDAS_DATA_SOURCE_NAME = 'pandas'


@register()
@pydantic_dataclass(config=AllowArbitraryTypesConfig)
class PredictionAttributes(ConfigMixin):
    r"""Dataclass for storing prediction-related attributes of a Task, if it
    is in prediction mode."""

    # The original name of the training table that was used to train the model,
    # necessary to preserve the model metagraph across training and prediction:
    train_table_name: str

    # The target column of the training task, used to format predictions:
    # TODO(manan): for some reason, typing this as a Column breaks downstream
    # dataclasses via Pydantic...
    target_col: Any | None = None

    # The number of targets at training time:
    num_targets: int | None = None

    @classmethod
    def customize_from_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        encoder = (values.get('target_col') or {}).get('encoder', None)
        if encoder is not None:
            values['target_col']['encoder'] = TypeAdapter(
                EncoderType).validate_python(encoder)
        return values


@register(data_cls=TaskConfig)
class Task(ConfigMixin, _HasData):
    r"""Describes a Kumo Task, which encapsulates a graph and a training table
    that is associated with a predictive query defined on the graph.

    Args:
        graph (kumo.graph.Graph): A :class:`kumo.Graph` object.
        task_type (str): Type of the prediction task. Currently
            supported options are: :obj:`"multiclass_classification"`,
            :obj:`"binary_classification"`, :obj:`"multilabel_classification"`,
            :obj:`"regression"`, :obj:`"static_link_prediction"` or
            :obj:`"temporal_link_prediction"`.
        train_table_name (str): Name of the training table.
        target_col (Column, optional): The target column to predict for
            tasks. Can be :obj:`None` for link prediction.
        fkeys (List[ForeignKey]): The foreign keys of the table.
            At least one foreign key to the :obj:`graph` needs to be defined.
        data_source (Connector or pd.DataFrame, optional):  Data source of the
            training table in the task. This can either be a connector object
            or a Pandas dataframe. If not provided, the graph's connector
            will be used and raises an error if graph has multiple connectors.
            (default: :obj:`None`)
        train_table (TrainingTable, optional): The training table of the task.
            If not provided, a new training table will be created based on
            `data_source` and `train_table_name`. (default: :obj:`None`)
        metrics (List[str], optional): List of metrics to report
            (:obj:`"acc"`, :obj:`"auprc"`, :obj:`"auroc"` :obj:`"mae"`,
            :obj:`"mse"`,:obj:`"f1"`, :obj:`"hr1"`). If set to :obj:`None`,
            will automatically infer metrics based on :obj:`task_type`.
            (default: :obj:`None`)
        cols (List[Column], optional): The columns of the table.
            (default: :obj:`None`)
        time_col (TimeColumn, optional): The time column of the table.
            (default: :obj:`None`)
        entity_include_bit_col (BoolColumn, optional): The
            ENTITY_INCLUDE_BIT column of the table. In forecasting task,
            we currently do not filter out rows in the training table.
            Instead we have an additional boolean column that tracks if
            the row should be included in the train table.
            (default: :obj:`None`)
        weight_col (BaseColumn, optional): The weight column of the table,
            indicating the sampling probability of training examples.
            Does not necessarily need to sum up to one. (default: :obj:`None`)
        uncensored_col (BoolColumn, optional): The column denoting whether the
            target column holds uncensored data. This column should only be
            specified in :obj:`TIME_TO_NEXT_EVENT` task types and denotes
            whether an event is right-censored or not (i.e. whether a next
            event happened or is only about to happen). (default: :obj:`null`)
        stats_manager (StatsManager, optional): `StatsManager` used to
            cache stats.
        split (Split, optional): The splitting scheme of the table.
            If set to :obj:`None`, will use a 80%/10%/10% random split.
            (default: :obj:`None`)
        target_to_entity_paths (List[List[Tuple[str, str, str]]], optional):
            The paths from the target table to the entity/entities. Only
            applicable in case the :class:`Task` is created via
            :class:`PredictiveQuery`. (default: :obj:`None`)
        target_table_name (str, optional): Name of the target table. Only
            applicable in case the :class:`Task` is created via
            :class:`PredictiveQuery`. (default: :obj:`None`)
        task_name (str, optional): The name of the task. (default: :obj:`None`)
        is_sliding_window (bool): Whether this is a task that will use the
            :obj:`SlidingWindowLightningModule` for evaluation.
            (default: :obj:`False`)
        lag_timesteps (int, optional): This sets the number of steps
            backward autoregression will take. (default: :obj:`None`)
        forecast_length (int, optional): This sets the number of steps
            forecasting will forecast. (default: :obj:`None`)
        unix_start_offset (int, optional):
            :obj:`PredictiveQuery.target.date_offset_range.start`
            in unix time. (default: :obj:`None`)
        unix_end_offset (int, optional):
            :obj:`PredictiveQuery.target.date_offset_range.end`
            in unix time. (default: :obj:`None`)
        unix_step_size (int, optional): :obj:`PredictiveQuery.timeframe_step`
            in unix time. (default: :obj:`None`)
        job_dir (str, optional): Directory for storing training-job
            specific artifacts such as split table stats. Sometimes known
            as home_dir (default: :obj:`None`)
    """
    def __init__(
        self,
        graph: Graph,
        task_type: TaskType | str,
        train_table_name: str,
        target_col: str | Column | None,
        fkeys: list[str] | list[ForeignKey],
        data_source: Connector | pd.DataFrame | None = None,
        train_table: TrainingTable | None = None,
        metrics: list[str] | None = None,
        cols: list[str] | list[Column] | None = None,
        time_col: str | TimeColumn | None = None,
        weight_col: str | BaseColumn | None = None,
        entity_include_bit_col: str | BoolColumn | None = None,
        stats_manager: Any | None = None,
        split: Split | None = None,
        target_to_entity_paths: list[list[EdgeType]] | None = None,
        target_table_name: str | None = None,
        task_name: str | None = None,
        train_table_source_name: str | None = None,
        is_sliding_window: bool = False,
        lag_timesteps: int | None = None,
        forecast_length: int | None = None,
        unix_start_offset: int | None = None,
        unix_end_offset: int | None = None,
        unix_step_size: int | None = None,
        train_table_data_version: str | None = None,
        prediction_attrs: PredictionAttributes | None = None,
        job_id: str | None = None,
        job_dir: str | None = None,
        list_distinct_target_is_array: bool | None = None,
        lp_bp_layout: str | None = None,
        kumo_version: str | None = None,
        ignore_encoders: bool = False,
    ) -> None:
        assert task_name is not None

        self.name = task_name
        self.graph = graph
        self.task_type = TaskType(task_type)
        self.prediction_attrs = prediction_attrs

        self.target_to_entity_paths = target_to_entity_paths or []
        self.is_sliding_window = is_sliding_window
        self.lag_timesteps = lag_timesteps or 0
        self.forecast_length = forecast_length or 1
        self.unix_start_offset = unix_start_offset
        self.unix_end_offset = unix_end_offset
        self.unix_step_size = unix_step_size
        self.list_distinct_target_is_array = list_distinct_target_is_array
        self.lp_bp_layout = lp_bp_layout

        # Column data per table to ignore in the GNN for this specific task:
        # NOTE This is internal only and solely being used in combination with
        # `PredictiveQuery` - we do not expose it in the `Task` interface:
        self.exclude_cols: dict[str, list[str]] = {}

        self.use_lp_shallow_rhs = False
        self.use_lp_id_awareness = False
        self.use_lp_hybrid = False
        self._diskgraph_train_table_col_stats_seeded = False

        # NOTE: This dependency of training table connector being a part of
        # the graph in default case must be removed once Task gets deprecated.
        # Not removing currently since there are a lot of tests which make this
        # default assumption.
        if data_source is None and train_table is None:
            if self.graph.num_data_sources != 1:
                raise ValueError(
                    f"Cannot add table '{train_table_name}' to task without "
                    "specifying a data_source. Provided graph has "
                    f"{len(graph.connectors)} data_sources.")
            data_source = self.graph.connectors[0]

        connector = data_source
        if isinstance(data_source, pd.DataFrame):
            connector = PandasConnector()
            connector.add_df(train_table_name, data_source)

        time_col = TimeColumn.cast(time_col)

        # NOTE: Right now `target_table_name` is only used in static link
        # prediction for inserting pseudo time in target table
        if (self.task_type.is_link_pred and time_col is not None
                and time_col.is_pseudo):
            assert target_table_name is not None
        self.target_table_name = target_table_name

        if not self.is_pred:
            if target_col is None and not self.task_type.is_link_pred:
                raise ValueError(
                    f"Task type {self.task_type} must define an associated "
                    f"target column. Please specify a target column to "
                    f"proceed.")

            if target_col is not None and self.task_type.is_link_pred:
                raise ValueError(
                    f"Task type {self.task_type} does not support a target "
                    f"column (got {target_col}). Please remove the target "
                    f"column to proceed.")

            if (self.task_type.is_link_pred and time_col is not None
                    and time_col.is_pseudo
                    and not self.list_distinct_target_is_array):
                # Static LP scalar targets use pseudo time for row-stage
                # materialization, so ratio-based LP splits become temporal
                # splits over that pseudo time.
                if isinstance(split, (RandomSplit, RandomLinkSplit)):
                    logger.info(
                        "Overwrite '%s' to 'TemporalSplit' for "
                        "static link prediction", split.__class__.__name__)
                    split = TemporalSplit(split.ratio)
                elif isinstance(split, TimeSplit):
                    # TimeSplit over pseudo time is not meaningful; fall back
                    # to a ratio-based temporal split.
                    if split.fractions is not None:
                        ratio = split.fractions
                    else:
                        ratio = [1.0 / split.num_splits] * split.num_splits
                    logger.info("Overwrite 'TimeSplit' to 'TemporalSplit' "
                                "for static link prediction")
                    split = TemporalSplit(ratio)
                if not isinstance(split, TemporalSplit):
                    raise ValueError(
                        "Static link prediction with scalar targets and "
                        "pseudo time requires a TemporalSplit-compatible "
                        "split. "
                        f"task_type={self.task_type}, "
                        f"split_type={type(split).__name__}, "
                        "list_distinct_target_is_array="
                        f"{self.list_distinct_target_is_array}")
            if (self.task_type == TaskType.STATIC_LINK_PREDICTION
                    and self.list_distinct_target_is_array):
                assert time_col is None
                assert isinstance(split, RandomLinkSplit)

        fkeys = [self.graph.fully_qualify_fkey(fkey) for fkey in fkeys]

        self.train_table = train_table or TrainingTable(
            table_name=train_table_name,
            connector=connector,
            task_name=task_name,
            target_col=target_col,
            fkeys=fkeys,
            cols=cols,
            time_col=time_col,
            weight_col=weight_col,
            stats_manager=stats_manager,
            split=split,
            source_name=train_table_source_name,
            task_type=self.task_type,
            data_version=train_table_data_version,
            is_pred=self.is_pred,
            job_id=job_id,
            job_dir=job_dir,
            entity_include_bit_col=entity_include_bit_col,
        )
        self._metrics = metrics

        # DiskGraph is the only supported deploy mode — when encoder
        # inferral is skipped by callers, stats will be injected later via
        # the strict diskgraph path.
        skip_encoder_inference = ignore_encoders

        if not self.is_pred:
            # Target semantic type is required even when encoder inferral
            # is intentionally skipped by callers (e.g. strict diskgraph
            # paths that inject encoder stats later).
            target_col = self.train_table.target_col
            if target_col is not None and target_col.stype is None:
                target_col.stype = self._infer_target_stype()
            if (not skip_encoder_inference and target_col is not None
                    and target_col.encoder is None):
                target_col.encoder = self._infer_target_encoder(target_col)

        # Infer info for previous label columns, used with autoregression:
        for current_col in self.train_table.cols:
            if current_col.name[:-1].endswith(AUTOREGRESSION_SUFFIX):
                if current_col.stype is None:
                    current_col.stype = self._infer_target_stype()
                if (not skip_encoder_inference and current_col.encoder is None):
                    current_col.encoder = self._infer_autoregression_encoder(
                        current_col)
            if current_col.name == AUTOREGRESSION_COL:
                if current_col.stype is None:
                    current_col.stype = Stype.sequence
                if (not skip_encoder_inference and current_col.encoder is None):
                    # TODO: Remove this dependency on `self.lag_timesteps`
                    #       from `Task` and change this logic to instead
                    #       use stats computation
                    if self.is_sliding_window:
                        # NaN value in autoregression label
                        # corresponds to the case where there are not
                        # enough previous aggregated labels or when
                        # the actual target value is NaN
                        na_strategy = 'keep'
                    else:
                        na_strategy = 'zero'
                    current_col.encoder = NumericalList(
                        na_strategy=na_strategy,
                        _stats={
                            ColStatType.SEQUENCE_MIN_LENGTH: self.lag_timesteps,
                            ColStatType.SEQUENCE_MAX_LENGTH: self.lag_timesteps,
                        },
                    )

        # Infer remaining encoders:
        if not skip_encoder_inference:
            self.infer_encoders()

        # Infer links for the training table upon construction:
        self.infer_links()

        # Number of targets is unknown until materialization time for training
        # tasks:
        self.num_targets: int | None = None
        if self.is_pred:
            self.num_targets = self.prediction_attrs.num_targets

        # Set the kumo version for the task to improve test process.
        self.kumo_version = kumo_version
        if self.kumo_version is None:
            self.kumo_version = get_kumo_version()

    # Built-ins ###############################################################

    def __repr__(self) -> str:
        target_col_name = "???"
        if self.is_pred:
            target_col = self.prediction_attrs.target_col
            if target_col is not None:
                target_col_name = target_col.name
        else:
            target_col_name = self.train_table.target_col_name

        return (f'{self.__class__.__name__}(\n'
                f'  name={self.name or "???"},\n'
                f'  type={self.task_type.value},\n'
                f'  graph={self.graph.name or "???"},\n'
                f'  train_table={self.train_table.name},\n'
                f'  target_col={target_col_name},\n'
                f'  time_col={self.train_table.time_col_name or "???"},\n'
                f'  weight_col={self.train_table.weight_col_name or "???"},\n'
                f'  split={self.train_table.split},\n'
                f')')

    def __getitem__(self, name: str) -> Table:
        if (name == self.train_table.name
                or name == self.train_table.source_name):
            return self.train_table
        return self.graph[name]

    # Properties ##############################################################

    @property
    def is_pred(self) -> bool:
        return self.prediction_attrs is not None

    @property
    def is_temporal(self) -> bool:
        # Using a pseudo-time column should not make the task temporal:
        return self.train_table.has_time_col() and (
            not self.train_table.time_col.is_pseudo)

    @property
    def metrics(self) -> list[str]:
        if self._metrics is None:
            self._metrics = self.get_default_metrics()
        return self._metrics

    @property
    def tables(self) -> list[Table]:
        return self.graph.tables + [self.train_table]

    @property
    def edge_strs(self) -> list[str]:
        r"""The list of all the edges in the task in string format"""
        if not self.has_data:
            raise ValueError("Task should be materialized to get edge names")
        return ["__".join(edge) for edge in self.data.edge_types]

    @property
    def unqualified_target_to_entity_paths(
            self) -> list[list[tuple[str, str, str]]]:
        return [[(get_table_name(src), rel, get_table_name(dst))
                 for src, rel, dst in path]
                for path in self.target_to_entity_paths]

    def get_default_metrics(self) -> list[str]:
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return ['acc']
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return ['auroc', 'ap', 'auprc']
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            metrics = []
            for metric in ['ap', 'auroc', 'auprc']:
                for average in ['macro', 'micro', 'per_label']:
                    metrics.append(f'{metric}_{average}')
            return metrics
        if self.task_type == TaskType.REGRESSION:
            return ['mae', 'mse', 'rmse']
        if self.task_type.is_ranking:
            default_metrics = []
            default_top_k_list = [1, 10, 100]
            for k in default_top_k_list:
                default_metrics += [
                    f'map@{k}', f'precision@{k}', f'recall@{k}', f'f1@{k}'
                ]
            return default_metrics
        if self.task_type == TaskType.FORECASTING:
            return ['mae', 'mse', 'rmse']

        raise ValueError(f"Non-supported task '{self.task_type.value}'")

    def get_train_table_stats(self, after_split: bool):
        return self.train_table.get_train_table_stats(after_split=after_split)

    # Split Helpers ###########################################################

    def random_split(self, *args, **kwargs) -> 'Task':
        self.train_table.split = RandomSplit(*args, **kwargs)
        return self

    def fixed_split(self, *args, **kwargs) -> 'Task':
        self.train_table.split = FixedSplit(*args, **kwargs)
        return self

    def temporal_split(self, *args, **kwargs) -> 'Task':
        self.train_table.split = TemporalSplit(*args, **kwargs)
        return self

    def date_offset_split(self, *args, **kwargs) -> 'Task':
        self.train_table.split = DateOffsetSplit(*args, **kwargs)
        return self

    def time_range_split(self, *args, **kwargs) -> 'Task':
        self.train_table.split = TimeRangeSplit(*args, **kwargs)
        return self

    def manual_split(self, *args, **kwargs) -> 'Task':
        self.train_table.split = ManualSplit(*args, **kwargs)
        return self

    # Inferral Helpers ########################################################

    def seed_train_table_col_stats_from_dataframe(
        self,
        df: pd.DataFrame,
    ) -> bool:
        r"""Seed train-table col stats from an in-memory DataFrame.

        DG-native sibling of :meth:`seed_train_table_col_stats_from_manifest` for
        callers with the DataFrame in hand but no connector / manifest sidecar.
        Computes the canonical DG supplemental stats (same entry point as
        ``test.util.cache_seeding``) and writes them to the
        :class:`BackendCacheManager`.

        Args:
            df: Source DataFrame; columns must superset the train-table columns.

        Returns:
            ``True`` if any stats were written (latch flipped); ``False`` if the
            latch was set or there are no declared columns yet (e.g. LP, which
            resolves via the lazy path).
        """
        import kumo.stats.compute_stats_python as compute_stats_python

        if self._diskgraph_train_table_col_stats_seeded:
            return False

        data_version = self.train_table.data_version
        if data_version is None:
            raise RuntimeError(
                'diskgraph strict path requires train table data_version '
                'to seed train_table_col_stats cache from a DataFrame; '
                'set Task(..., train_table_data_version=...) at construction')

        # Declared train-table cols + the separately-stored target_col.
        col_by_name: dict[str, Any] = {
            col.name: col.base_config()
            for col in self.train_table.cols
        }
        if self.train_table.target_col is not None:
            col_by_name[self.train_table.target_col.name] = (
                self.train_table.target_col.base_config())

        # Empty cols (e.g. LP pre-materialization) — defer to the lazy path.
        if not col_by_name:
            logger.info('train_table.cols empty — deferring col stats seeding '
                        'from DataFrame to lazy path during encoder inference')
            return False

        stats_to_add: dict[Any, dict[ColStatType, Stats]] = {}
        for col_name, col_cfg in col_by_name.items():
            if col_name not in df.columns:
                # Skip rather than fabricate; the strict-path raise surfaces the
                # genuine missing-stat error at the right column.
                logger.warning(
                    "seed_train_table_col_stats_from_dataframe: column "
                    "'%s' declared on train table but absent from "
                    "DataFrame; skipping", col_name)
                continue
            ser = df[col_name]
            _, stats_for_col = (
                compute_stats_python.compute_supplemental_col_stats_from_series(
                    ser,
                    data_version=data_version,
                    approx=False,
                ))
            if stats_for_col:
                stats_to_add[col_cfg] = stats_for_col

        if not stats_to_add:
            raise RuntimeError(
                'diskgraph strict path: seed_train_table_col_stats_from_'
                'dataframe produced no usable stats for any declared train '
                'table column; check that the DataFrame columns match the '
                'train table schema')

        fingerprint = self.train_table.fingerprint()
        cache = get_active_cache()
        cache.add_col_stats(fingerprint, stats_to_add)
        cache.save_col_stats_cache(fingerprint, data_version)
        self._diskgraph_train_table_col_stats_seeded = True
        return True

    def seed_train_table_col_stats_from_manifest(
        self,
        manifest_path: str,
    ) -> bool:
        r"""Best-effort seed of train-table col stats from an explicit
        Rust-producer manifest sidecar.

        ``manifest_path`` (e.g. the TT-NG producer manifest) is read directly.
        Returns ``True`` if the manifest was found, parsed, and any stats
        written; ``False`` (no raise) when the file is missing. A
        located-but-malformed manifest (or a missing ``data_version``) still
        raises — those are contract violations, not best-effort misses.
        """
        if self._diskgraph_train_table_col_stats_seeded:
            return False
        # A missing explicit ``manifest_path`` is a best-effort miss.
        if not fsutil.exists(manifest_path):
            return False

        import kumo.diskgraph.diskgraph_native_tt_artifacts as tt_artifacts

        try:
            manifest = tt_artifacts.get_validated_artifact_bundle(manifest_path)
        except Exception:
            # Manifest stale/invalid — skip; encoder inference computes from data.
            return False
        parsed = manifest.get('parsed_train_table_col_stats')
        if not isinstance(parsed, dict):
            # The producer omits the block when nothing needs stats (FK-free LP).
            # Degrade to False when there is nothing to seed; node-pred
            # (target_col present) still raises.
            needs_col_stats = (self.train_table.target_col is not None
                               or bool(self.train_table.cols))
            if not needs_col_stats:
                return False
            raise RuntimeError(
                'diskgraph strict path requires parsed train_table_col_stats '
                'artifact in manifest')

        raw_stats = parsed.get('stats')
        if not isinstance(raw_stats, dict):
            raise RuntimeError(
                'diskgraph strict path requires train_table_col_stats.stats '
                'payload in manifest artifact')

        return self._seed_col_stats_from_raw_stats(
            raw_stats,
            empty_artifact_error=(
                'diskgraph strict path found train_table_col_stats artifact '
                'but no usable stats entries for train table columns'),
        )

    def seed_train_table_col_stats_from_sidecar(
        self,
        sidecar_path: str,
    ) -> bool:
        r"""Seed train-table col stats from the durable
        ``train_table_col_stats.json`` sidecar the DiskGraph producer writes.

        AUTHORITATIVE post-ingest source for node-pred col-stats. Unlike
        :meth:`seed_train_table_col_stats_from_manifest`, it reads the
        self-describing sidecar directly rather than the transient gen
        ``manifest.json`` bundle, whose sibling artifacts (consumed scratch
        staging) may be gone on the PQL node-pred path -- the failure that
        raised ``KeyError: CATEGORY_COUNTS`` at encoder build.

        A missing sidecar is a best-effort miss (``False``, no raise); a
        located-but-malformed sidecar still raises (contract violation).
        """
        if self._diskgraph_train_table_col_stats_seeded:
            return False
        # A missing file is a best-effort miss, not an error.
        if not fsutil.exists(sidecar_path):
            return False

        import kumo.diskgraph.diskgraph_native_tt_artifacts as tt_artifacts
        payload = tt_artifacts.parse_train_table_col_stats_artifact(
            sidecar_path)
        raw_stats = payload.get('stats')
        if not isinstance(raw_stats, dict):
            raise RuntimeError(
                'diskgraph strict path requires train_table_col_stats.stats '
                'payload in durable sidecar artifact')

        return self._seed_col_stats_from_raw_stats(
            raw_stats,
            empty_artifact_error=(
                'diskgraph strict path found train_table_col_stats sidecar '
                'but no usable stats entries for train table columns'),
        )

    def _seed_col_stats_from_raw_stats(
        self,
        raw_stats: dict,
        *,
        empty_artifact_error: str,
    ) -> bool:
        r"""Seed the backend col-stats cache from a producer ``stats`` map.

        Shared core of the manifest- and sidecar-reading seeders (same
        ``stats`` payload shape in both). Returns ``True`` if any stats were
        written, ``False`` if there are no declared columns yet (e.g. LP,
        resolved on the lazy path).
        """
        fingerprint = self.train_table.fingerprint()
        data_version = self.train_table.data_version
        if data_version is None:
            raise RuntimeError(
                'diskgraph strict path requires train table data_version to '
                'seed train_table_col_stats cache')

        col_by_name = {
            col.name: col.base_config()
            for col in self.train_table.cols
        }
        if self.train_table.target_col is not None:
            col_by_name[self.train_table.target_col.name] = (
                self.train_table.target_col.base_config())

        # Empty cols (e.g. LP) — defer to the lazy path in
        # _get_encoder_stats_from_cache_or_fail when columns are available.
        if not col_by_name:
            logger.info('train_table.cols empty — deferring col stats seeding '
                        'to lazy path during encoder inference')
            self._diskgraph_train_table_col_stats_seeded = False
            return False

        # TT-NG manifest keys stats by ENTITY1/TARGET1; kumo's cols use slot-0
        # ENTITY/TARGET, so fall back to the alias map. No-op for legacy.
        _PHYSICAL_TO_LOGICAL = {'ENTITY1': 'ENTITY', 'TARGET1': 'TARGET'}

        stats_to_add: dict[Any, dict[ColStatType, Stats]] = {}
        for stat_key, stat_payload in raw_stats.items():
            if not isinstance(stat_key, str) or not isinstance(
                    stat_payload, dict):
                continue
            parts = stat_key.rsplit('__', 3)
            if len(parts) != 4:
                continue
            col_name, _stype_name, stat_name, approx_flag = parts
            col_cfg = col_by_name.get(col_name)
            if col_cfg is None:
                legacy_name = _PHYSICAL_TO_LOGICAL.get(col_name)
                if legacy_name is not None:
                    col_cfg = col_by_name.get(legacy_name)
            if col_cfg is None:
                continue
            try:
                stat_type = ColStatType(stat_name)
            except Exception:
                continue
            approx = approx_flag.lower() == 'true'
            stat = Stats(
                value=stat_payload.get('value'),
                approx=approx,
                data_version=data_version,
                num_entries=stat_payload.get('num_entries'),
            )
            if col_cfg not in stats_to_add:
                stats_to_add[col_cfg] = {}
            stats_to_add[col_cfg][stat_type] = stat

        if not stats_to_add:
            raise RuntimeError(empty_artifact_error)

        cache = get_active_cache()
        cache.add_col_stats(fingerprint, stats_to_add)
        cache.save_col_stats_cache(fingerprint, data_version)
        self._diskgraph_train_table_col_stats_seeded = True
        return True

    def _get_encoder_stats_from_cache_or_fail(
        self,
        col: Column,
    ) -> dict[ColStatType, Any]:
        """Load encoder stats from cache only; never compute in DiskGraph path."""
        if self.train_table.data_version is None:
            raise RuntimeError(
                'diskgraph strict path requires train table data_version for '
                f"column '{col.name}'")

        fingerprint = self.train_table.fingerprint()
        data_version = self.train_table.data_version
        cache = get_active_cache()

        def _fetch() -> tuple[dict[Any, Any], dict[Any, Any], list[ColStatType],
                              list[ColStatType]]:
            cache.load_col_stats_cache(fingerprint, data_version)

            col_cfg = col.base_config()
            exact_types = [
                stat for stat in EncoderGenerator.REQUIRED_EXACT_STATS
                if stat.supports(col.dtype) and stat.supports_stype(col.stype)
            ]
            approx_types = [
                stat for stat in EncoderGenerator.REQUIRED_APPROX_STATS
                if stat.supports(col.dtype) and stat.supports_stype(col.stype)
            ]

            exact_request = {col_cfg: exact_types} if exact_types else {}
            approx_request = {col_cfg: approx_types} if approx_types else {}

            # DG-native: read directly from BackendCacheManager (DG ingest
            # populates it); the severed legacy factory has no get_col_stats.
            exact_stats = cache.get_col_stats(
                fingerprint,
                exact_request,
                approx=False,
                data_version=data_version,
            ) if exact_request else {}
            approx_stats = cache.get_col_stats(
                fingerprint,
                approx_request,
                approx=True,
                data_version=data_version,
            ) if approx_request else {}

            col_exact = exact_stats.get(col_cfg, {})
            col_approx = approx_stats.get(col_cfg, {})
            missing_exact = [
                stat for stat in exact_types if stat not in col_exact
            ]
            missing_approx = [
                stat for stat in approx_types if stat not in col_approx
            ]
            return col_exact, col_approx, missing_exact, missing_approx

        col_exact, col_approx, missing_exact, missing_approx = _fetch()
        if missing_exact or missing_approx:
            # TT-NG has exactly two authoritative col-stats sources, each with
            # its own availability point, and NEITHER is re-read on demand:
            #   * GRAPH col stats  -> populated in the backend cache during
            #     GRAPH INGESTION (DG writes them as it ingests entity tables).
            #   * TRAIN-TABLE col stats -> seeded from the DiskGraph producer
            #     sidecar during MATERIALIZATION (see
            #     rest/.../materialize_training_task.py ->
            #     Task.seed_train_table_col_stats_from_sidecar).
            # A miss here means the stat is genuinely absent from the cache at
            # the point the encoder needs it -- i.e. a real upstream gap in
            # ingestion, materialization, or the producer sidecar. Under TT-NG
            # there is intentionally NO manifest re-probe / whole-table
            # connector-read fallback: papering over the miss by re-reading the
            # table is exactly the legacy anti-pattern this path removes. Fail
            # loud so the upstream gap gets fixed at its source.
            missing = [s.value for s in (missing_exact + missing_approx)]
            raise RuntimeError(
                f"Encoder stats for '{self.train_table.name}.{col.name}' are "
                f"missing from the backend col-stats cache (data_version="
                f"{data_version}). Missing stat types: {missing}. Under the "
                f"TT-NG / DiskGraph strict path these must already be present: "
                f"graph-column stats are populated during GRAPH INGESTION and "
                f"train-table-column stats are seeded from the DiskGraph "
                f"producer sidecar during MATERIALIZATION "
                f"(Task.seed_train_table_col_stats_from_sidecar). There is "
                f"intentionally no manifest or connector fallback here -- a "
                f"miss indicates a real upstream gap (ingest / materialization "
                f"/ sidecar seed did not cover this column), not something to "
                f"paper over by re-reading the table.")

        merged = dict(col_exact)
        merged.update(col_approx)
        return merged

    def _infer_target_encoder(
        self,
        target_col: Column,
    ) -> Encoder:
        stype = self._infer_target_stype()
        if stype != Stype.numerical:
            return select_target_encoder(stype, {})
        if self.task_type == TaskType.REGRESSION and self.is_sliding_window:
            return Numerical()
        assert target_col is not None
        stats = self._get_encoder_stats_from_cache_or_fail(target_col)
        return select_target_encoder(stype,
                                     EncoderGenerator._shared_stats(stats))

    def _infer_autoregression_encoder(
        self,
        col: Column,
    ) -> Encoder:
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return Index(na_strategy='separate')
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return Index(na_strategy='separate')
        if self.task_type.is_multilabel:
            return MultiCategorical(na_strategy='separate')
        if self.task_type == TaskType.REGRESSION:
            if self.is_sliding_window:
                # TODO: Use MultiNestedTensor so that we can drop
                # deprecated NAStrategy.KEEP.
                encoder_fn = Numerical(na_strategy='keep')
            else:
                encoder_stats = self._get_encoder_stats_from_cache_or_fail(col)
                if str(encoder_stats[ColStatType.MEAN].value) == 'nan':
                    encoder_fn = Numerical()
                else:
                    encoder_fn = EncoderGenerator.get_numerical_encoder(
                        col, encoder_stats)
                encoder_fn.na_strategy = 'mean'
            return encoder_fn

        raise ValueError(f"Non-supported task '{self.task_type.value}'")

    def _infer_target_stype(self) -> Stype:
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return Stype.categorical
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return Stype.categorical
        if self.task_type.is_multilabel:
            return Stype.multicategorical
        if self.task_type == TaskType.REGRESSION:
            return Stype.numerical
        if self.task_type == TaskType.FORECASTING:
            return Stype.numerical

        raise ValueError(f"Non-supported task '{self.task_type.value}'")

    def infer_metadata(self, *args, **kwargs) -> 'Task':
        self.train_table.infer_metadata(*args, **kwargs)
        return self

    def infer_encoders(self, *args, **kwargs) -> 'Task':
        # `train_table.infer_encoders` only knows
        # about column-stats-driven inferral via `EncoderGenerator`; it does
        # not understand task-type-driven target encoders (e.g.
        # `MULTILABEL_RANKING -> MultiCategorical`). Populate the target
        # encoder here so callers that build the task with
        # `ignore_encoders=True` and then explicitly call `infer_encoders`
        # get a fully-encoded task.
        if (not self.is_pred and not self.task_type.is_link_pred
                and self.train_table.target_col is not None
                and self.train_table.target_col.encoder is None):
            target_col = self.train_table.target_col
            target_col.encoder = self._infer_target_encoder(target_col)
        self.train_table.infer_encoders(*args, **kwargs)
        return self

    def infer_links(self, name_inferral: bool = True) -> 'Task':
        connector = self.train_table.connector

        # connector may have link information
        if connector is not None and connector.has_links_support():
            meta_tables = [self.train_table.metadata_config()]
            for table in self.graph:
                if table.data_source == self.train_table.data_source:
                    meta_tables.append(table.metadata_config())

            table_names = [table.table_name for table in meta_tables]
            meta_table = connector.get_table_metadata(table_names)[0]

            for fkey in meta_table.fkeys:
                # FKeys from TableMetadataConfig are not fully qualified.
                fkey = self.graph.fully_qualify_fkey(fkey)
                if self.train_table.has_fkey(fkey.name):
                    self.train_table[fkey.name].dst_table = fkey.dst_table
                    self.train_table[fkey.name].dst_key = fkey.dst_key
                else:
                    self.train_table.add_fkey(ForeignKey.from_config(fkey))

        if name_inferral:
            pkey_to_table: dict[str, str] = {}
            for table in self.graph:
                if table.pkey is not None:
                    pkey_to_table[table.pkey.name] = table.name

            for fkey in self.train_table.fkeys:
                if fkey.dst_table is None:
                    fkey.dst_table = pkey_to_table.get(fkey.name, None)

        return self

    # Trainer Helpers #########################################################

    def _replace_train_table(
        self,
        name: str,
        data_source: Connector | pd.DataFrame | None = None,
    ) -> 'Task':
        r"""Replaces the training table of the task by the table with name
        :obj:`name` defined in the data source :obj:`data_source`.

        Args:
            name (str): Name of the prediction table.
            data_source (Union[Connector, pd.DataFrame], optional):
                The data source to use for the prediction table. Can be either
                a dataframe directly or a connector which points to the
                prediction table data. If set to :obj:`None`, the connector of
                the original training table will be used.
        """
        # Choose a connector for the new table:
        connector = data_source
        if connector is None:
            # Reuse the training table connector:
            connector = self.train_table.connector
        elif isinstance(data_source, pd.DataFrame):
            df, data_source = data_source, DEFAULT_PANDAS_DATA_SOURCE_NAME
            connector = PandasConnector({name: df})
        elif isinstance(data_source, str):
            raise ValueError(
                "data source for the prediction table should either be a "
                "connector of a Pandas Dataframe. Received string: "
                f"{data_source}")

        # Set basic properties:
        pred_table = copy.copy(self.train_table)
        pred_table.clear_data()
        pred_table._data_version = None
        pred_table._name = name
        pred_table._source_name = name
        pred_table.connector = connector
        pred_table.is_pred = True
        if self.task_type.is_link_pred:
            pred_table.lp_is_array = _lp_prediction_layout_uses_target_sidecar(
                self.list_distinct_target_is_array,
                self.lp_bp_layout,
            )
        # TODO(data-platform): drop non-data backend stats caches completely:
        pred_table.stats = {}

        # Construct shallow-copied prediction task:
        task = copy.copy(self)
        task._data = None
        task.train_table = pred_table

        # If we are replacing a prediction table with another prediction table,
        # do nothing. Otherwise, construct `prediction_attrs` and make other
        # necessary modifications:
        if not self.is_pred:
            task.prediction_attrs = PredictionAttributes(
                train_table_name=self.train_table.table_name,
                target_col=self.train_table.target_col,
                num_targets=self.num_targets,
            )

        # Use a very simple split to skip any work/checks that could error
        # out because all the dates are the same.
        task.train_table.split = FixedSplit([1.0, 0.0, 0.0])

        if task.train_table.target_col is not None:
            # Prediction table should have no target column:
            if task.train_table.has_col(self.train_table.target_col.name):
                del task.train_table[self.train_table.target_col.name]
            task.train_table._target_col = None

        return task

    # Persistence #############################################################

    @classmethod
    def from_config(cls, cfg: TaskConfig) -> 'Task':
        # When loading from a saved config, always use a connector for the
        # data_source.
        graph: Graph = Graph.from_config(cfg.graph)
        train_table = TrainingTable.from_config(cfg.train_table)

        # TODO(manan): see comment in the PredictionAttributes class:
        prediction_attrs = PredictionAttributes.from_config(
            cfg.prediction_attrs)
        if (prediction_attrs is not None
                and prediction_attrs.target_col is not None):
            prediction_attrs.target_col = Column.from_config(
                prediction_attrs.target_col)

        task = cls(
            graph=graph,
            train_table_name=cfg.train_table.table_name,
            target_col=Column.from_config(cfg.train_table.target_col),
            task_type=cfg.task_type,
            data_source=train_table.connector,
            train_table=train_table,
            fkeys=[ForeignKey.from_config(k) for k in cfg.train_table.fkeys],
            metrics=cfg.metrics,
            cols=[Column.from_config(col) for col in cfg.train_table.cols],
            time_col=TimeColumn.from_config(cfg.train_table.time_col),
            weight_col=Column.from_config(cfg.train_table.weight_col),
            target_to_entity_paths=cfg.target_to_entity_paths,
            target_table_name=cfg.target_table_name,
            split=Split.from_config(cfg.train_table.split),
            task_name=cfg.name,
            forecast_length=cfg.forecast_length,
            unix_start_offset=cfg.unix_start_offset,
            unix_end_offset=cfg.unix_end_offset,
            unix_step_size=cfg.unix_step_size,
            train_table_data_version=cfg.train_table.data_version,
            prediction_attrs=prediction_attrs,
            is_sliding_window=cfg.is_sliding_window,
            lag_timesteps=cfg.lag_timesteps,
            list_distinct_target_is_array=cfg.list_distinct_target_is_array,
            lp_bp_layout=cfg.lp_bp_layout,
        )
        task.exclude_cols = cfg.exclude_cols

        task.use_lp_shallow_rhs = cfg.use_lp_shallow_rhs
        task.use_lp_id_awareness = cfg.use_lp_id_awareness
        task.use_lp_hybrid = cfg.use_lp_hybrid

        return task

    def persist_train_table_pandas_connector(
            self, destination: str | None = None) -> None:
        r"""If the TrainingTable has a PandasConnector we need to persist it
        to a ParquetConnector. It is also required to rename the associated
        nodes and edges in the graph.

        ``destination`` is the base dir the connector is persisted under
        (``<destination>/tasks/<task_name>``); :obj:`None` falls back to the
        process-global :func:`kumo.get_home_dir`.
        """
        if isinstance(self.train_table.connector, PandasConnector):
            old_name = self.train_table.name
            self.train_table.persist_pandas_connector(destination=destination)
            if self.has_data:
                self.data.rename(old_name, self.train_table.name)

    def save(self, destination: str, name: str | None = None) -> str:
        """Save the :class:`kumo.Task` to storage.

        Args:
            destination (string): Base directory to save under; the task is
                written to ``<destination>/tasks/<name>`` and the graph to
                ``<destination>/graphs/<name>``. Required — callers pass an
                explicit per-job dir so this never depends on (or mutates) the
                process-global home directory.
            name (string, optional): The name of the folder to store the data.
                If set to :obj:`None`, will use the name of the task
                (default: :obj:`None`)
        """
        name = name or self.name

        # Save graph:
        self.graph.save(destination=destination)

        # If the train table contains a PandasConnector, persist it:
        self.persist_train_table_pandas_connector(destination=destination)

        # Mirror load-time target-encoder inference so saved/loaded configs
        # don't diverge (``ignore_encoders=True`` leaves it unset at construction).
        if (not self.is_pred and not self.task_type.is_link_pred
                and self.train_table.target_col is not None
                and self.train_table.target_col.encoder is None):
            target_col = self.train_table.target_col
            target_col.encoder = self._infer_target_encoder(target_col)

        # Save task-related data:
        root_dir = osp.join(destination, 'tasks', name)
        fsutil.makedirs(root_dir, exist_ok=True)

        task_config: TaskConfig = self.config()
        path = osp.join(root_dir, 'config.json')
        save_json(task_config, path)

        if self.train_table.has_data:
            self.train_table.data.save_data(
                osp.join(root_dir, self.train_table.name))

        return name

    @classmethod
    def load_from_config(
        cls,
        name: str,
        home_dir: str,
    ) -> 'Task':
        path = osp.join(home_dir, 'tasks', name, 'config.json')
        return cls.from_config(load_json(path, cls=TaskConfig))

    @classmethod
    def load(
        cls,
        name: str,
        home_dir: str,
    ) -> 'Task':
        r"""Loads the :class:`kumo.Task` config from storage.

        On a DiskGraph deploy the task is always loaded config-only:
        graph-table features and edges are served from the on-VM diskgraph
        index at training/prediction time, so there is no in-process
        materialization to perform here.

        Args:
            name (str): The name of the task.
            home_dir (str): Base dir the task config is loaded from
                (``<home_dir>/tasks/<name>/config.json``). Required — callers
                pass the per-job dir explicitly (no process-global fallback).
        """
        try:
            task = cls.load_from_config(name, home_dir)
        except ValueError as err:
            logger.error(f"Got error: {err}")
            raise ValueError(f"Task checkpoint '{name}' does not exist. "
                             f"Please check the task name, and make sure "
                             f"you have called `task.save()` when "
                             f"creating your 'kumo.Task'") from err

        return task

    def to_task_config(self):
        from kumo_core.training.task_config import TaskConfig as MLTaskConfig
        return MLTaskConfig(
            task_type=self.task_type,
            name=self.name,
            metrics=self.metrics,
            target_to_entity_paths=self.target_to_entity_paths,
            exclude_cols_dict=self.exclude_cols,
            unix_start_offset=self.unix_start_offset,
            unix_end_offset=self.unix_end_offset,
            unix_step_size=self.unix_step_size,
            forecast_length=self.forecast_length,
            lag_timesteps=self.lag_timesteps,
        )

    def to_resolved_numerical_task(
        self,
        manifest: DiskgraphTaskManifest,
        *,
        sampling: ResolvedTaskSampling | None = None,
    ) -> ResolvedNumericalTask:
        """Drop authoring/services and retain the numerical training task."""

        resolved = ResolvedNumericalTask.from_manifest(
            manifest,
            name=self.name,
            metrics=list(self.metrics),
            forecast_length=self.forecast_length,
            lag_timesteps=self.lag_timesteps,
            unix_start_offset=self.unix_start_offset,
            unix_end_offset=self.unix_end_offset,
            unix_step_size=self.unix_step_size,
            sampling=sampling,
        )
        resolved.target_to_entity_paths = [[
            tuple(edge) for edge in path
        ] for path in self.unqualified_target_to_entity_paths]
        resolved.exclude_cols_dict = {
            str(table): list(columns)
            for table, columns in self.exclude_cols.items()
        }
        return resolved

    # Validation ##############################################################

    def validate(self, raise_on_error: bool = True,
                 ignore_encoders: bool = False) -> bool:
        status = True
        status &= self.graph.validate(raise_on_error, ignore_encoders)
        status &= self.train_table.validate(raise_on_error)

        for fkey in self.train_table.fkeys:
            # 1. For each foreign key, there exists a valid destination:
            if (fkey.dst_table is None
                    or not fkey_destination_exists(self.graph, fkey)):
                status = False
                logging.warn_or_raise(
                    f"The foreign key '{fkey.name}' of the table "
                    f"'{self.train_table.name}' is expected to point to a "
                    f"valid destination table (got {fkey.dst_table})",
                    raise_on_error)

            if fkey.dst_key is None:
                # 2. Each table referenced by a foreign key without a
                # destination column has a primary key:
                if self.graph[fkey.dst_table].pkey is None:
                    status = False
                    logging.warn_or_raise(
                        f"The foreign key '{fkey.name}' of the table "
                        f"'{self.train_table.name}' is expected to point to a "
                        f"destination table with a primary key", raise_on_error)
            else:
                # 3. Each table referenced by a foreign key with a destination
                # column has that destination column as a key:
                if (not self.graph[fkey.dst_table].has_fkey(fkey.dst_key) and
                        not self.graph[fkey.dst_table].is_pkey(fkey.dst_key)):
                    status = False
                    logging.warn_or_raise(
                        f"The foreign key '{fkey.name}' of the table "
                        f"'{self.train_table.name}' points to the "
                        f"non-existent key '{fkey.dst_key}' in table "
                        f"'{fkey.dst_table}'", raise_on_error)

        if self.task_type.is_link_pred and len(self.train_table.fkeys) != 2:
            status = False
            logging.warn_or_raise(f"Link prediction requires exactly two "
                                  f"foreign keys in the training table "
                                  f"(got {len(self.train_table.fkeys)})")

        if self.task_type.is_link_pred and len(self.train_table.cols) > 0:
            status = False
            logging.warn_or_raise(f"Link prediction does not support feature "
                                  f"columns in the training table "
                                  f"(got {len(self.train_table.cols)})")

        return status

    def print_summary(self) -> None:
        r"""Prints a summary of the task, *i.e.* the data types, encoder
        decisions, and number of :obj:`N/A` values per column of the training
        table and all tables in the graph."""
        self.graph.print_summary()
        print()
        self.train_table.print_summary()

    def clear_data(self) -> 'Task':
        super().clear_data()
        self.graph.clear_data()
        self.train_table.clear_data()
        return self

    def _set_forecast_col_encoder(
        self,
        year_over_year: bool,
        include_future_labels: bool = False,
    ) -> 'Task':
        """For forecasting tasks, set the encoder for the
        lag features. This is needed during batch prediction
        so the encoder is stored in the task config."""
        if self.task_type != TaskType.FORECASTING:
            return self

        lag_timesteps = self.lag_timesteps

        if include_future_labels:
            encoder = NumericalList(
                _stats={
                    ColStatType.SEQUENCE_MIN_LENGTH: self.forecast_length,
                    ColStatType.SEQUENCE_MAX_LENGTH: self.forecast_length,
                })
            self.train_table.add_col(
                name=FUTURE_LABELS_COL,
                dtype=Dtype.floatlist,
                stype=Stype.sequence,
                encoder=encoder,
            )
            self.train_table[FUTURE_LABELS_COL].encoder = encoder

        if lag_timesteps > 0:
            encoder = NumericalList(
                _stats={
                    ColStatType.SEQUENCE_MIN_LENGTH: lag_timesteps,
                    ColStatType.SEQUENCE_MAX_LENGTH: lag_timesteps,
                })
            self.train_table.add_col(
                name=AUTOREGRESSION_COL,
                dtype=Dtype.floatlist,
                stype=Stype.sequence,
                encoder=encoder,
            )
            # Explicitly set the encoder since add_col's update() path
            # skips private attributes like _encoder:
            self.train_table[AUTOREGRESSION_COL].encoder = encoder

        if year_over_year:
            encoder = NumericalList(
                _stats={
                    ColStatType.SEQUENCE_MIN_LENGTH:
                    self.forecast_length + 2 * PADDING_SIZE,
                    ColStatType.SEQUENCE_MAX_LENGTH:
                    self.forecast_length + 2 * PADDING_SIZE,
                })
            self.train_table.add_col(
                name=YOY_COL,
                dtype=Dtype.floatlist,
                stype=Stype.sequence,
                encoder=encoder,
            )
            # Explicitly set the encoder since add_col's update() path
            # skips private attributes like _encoder:
            self.train_table[YOY_COL].encoder = encoder

    # Visualization ###########################################################

    def visualize(
        self,
        path: str | BytesIO | None = None,
        show_cols: bool = True,
    ) -> None:
        r"""Visualizes the Kumo graph.

        Args:
            path (str or BytesIO, optional): The path to where the plot is
                saved. Pass in a :obj:`BytesIO` object if you would like the
                image written to an in-memory buffer instead. If set to
                :obj:`None`, will visualize the graph on-the-fly. (default:
                :obj:`None`)
        """
        meta_table_configs = [table.metadata_config() for table in self.tables]
        visualize_graph(meta_table_configs, self.name, path, show_cols)


# Utilities ###################################################################


def fkey_destination_exists(graph: Graph, fkey: ForeignKey) -> bool:
    r"""Since fkey dst_table can be fully qualified or simply table name,
    checking in both."""
    all_table_names = ([table_name for table_name in graph.table_names] +
                       [table.table_name for table in graph.tables])
    return fkey.dst_table in all_table_names


def collect_encoders(task: Task | Graph) -> EncodersConfig:
    r"""Obtains all encoder configurations stored in a task configuration,
    materializes these encoders (by initializing them with statistics computed
    on each column), and returns an object with these initialized encoders.
    This object is intended to be passed to the feature store."""
    logger.info(f"Collecting encoders for Task {task.name}")

    encoders = EncodersConfig()

    # Materialize all encoders:
    for table in task.tables:
        feat_encoders, target_encoder = materialize_encoders(table)
        encoders[table.table_name] = feat_encoders
        if isinstance(table, TrainingTable):
            encoders.target = target_encoder

    return encoders
