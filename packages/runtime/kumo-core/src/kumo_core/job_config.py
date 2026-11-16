# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import os.path as osp
from copy import deepcopy
from dataclasses import field
from datetime import datetime
from typing import Any

from kumo_core.io.filesystem import exists
from kumo_core.io.json import load_json, save_json
from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs

logger = logging.getLogger(__name__)

_DATACLASS_CONFIG = ConfigDict(arbitrary_types_allowed=True)


def kumo_dataclass(cls):
    return dataclass(cls, config=_DATACLASS_CONFIG)


def before_validator_dict(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, ArgsKwargs):
        if data.args:
            raise TypeError('positional arguments are not supported')
        return dict(data.kwargs or {})
    raise TypeError(
        f'cannot normalize {type(data).__name__!r} as configuration values')


def _read_json(path: str) -> dict[str, Any]:
    value = load_json(path, dict)
    assert value is not None
    return value


def _load_dataclass(path: str, cls: type[Any]) -> Any:
    return TypeAdapter(cls).validate_python(_read_json(path))


def _save_dataclass(value: Any, path: str) -> None:
    save_json(value, path)


PRED_TABLE = 'pred_table'


@kumo_dataclass
class BestTrainerConfig:
    best_trainer: str


# This class is to override train dir and model dir during prediction.
# It is used in bin/pred: If there is one persisted `TrainDirOverrides`
# under the prediction job dir, we will use it to override the original
# settings.
@kumo_dataclass
class TrainDirOverrides:
    train_dir: str
    model_dir: str

    def validate(self) -> None:
        """Raise ValueError if the train dir and model dir do not exist
        """
        if not exists(self.train_dir):
            raise ValueError(
                f"`train_dir` override path does not exist, or the server does"
                f" not have read access: {self.train_dir}")
        if not exists(self.model_dir):
            raise ValueError(
                f"`model_dir` override path does not exist, or the server does"
                f" not have read access: {self.model_dir}")


@kumo_dataclass
class JobConfig:
    r"""A lightweight configuration for an AutoML job. Currently we use it
    to hold a task, an AutoML config, and env variables to run the job.

    Ideally, we should be able to come up with an Automl config that contains
    everything we need to run an automl job. Env variables should be used to
    process the data and we directly use the output in the task config.
    """
    home_dir: str
    task_name: str

    # The training job id used for k8s job launching
    training_job_id: str | None = None
    automl_cfg_path: str | None = None
    trainer_cfg_path: str | None = None
    automl_backend: str | None = None  # TODO Can be dropped.

    # List of table names whose feature should be loaded. If provided, only
    # these table's features would be loaded.
    tables_to_load: list[str] | None = None

    # Names of nodes to be loaded into the graphstore.
    nodes_to_load: list[str] | None = None

    disable_explain: bool = False

    # Optional warm start trainer directory to
    # initialize auto trainer model. Used as
    # Trainer.load(task, warm_start_trainer_dir)
    warm_start_trainer_dir: str | None = None
    diskgraph_index_path: str | None = None

    # Path to the DiskGraph-native task-dataset bundle. When non-None,
    # ``AutoTrainer`` dispatches to ``DiskgraphNestedTrainer`` per
    # AutoML trial. The presence of this field is the authoritative
    # TT-NG signal that crosses the materialize/train activity
    # boundary.
    diskgraph_task_dataset_bundle_dir: str | None = None

    @staticmethod
    def load(path: str) -> 'JobConfig':
        # Parameters specified here are going to be (support backward
        # compatibility) ignored if the job config file contains them
        # but the 'JobConfig' class not
        json_ignore_params = [
            'remote_graphengine_address',
            'tables_to_load',
            'nodes_to_load',
            'edges_to_load',
            'automl_backend',
            'disable_explain',
            'num_mutual_information_steps',
            'use_core_v2_api',
            'graph_backend',
            'diskgraph_index_path',
            # Ignored so historical JobConfig JSON blobs still deserialize;
            # the path dicts are no longer consumed.
            'table_names_with_paths',
            'node_names_with_paths',
            'edge_names_with_paths',
        ]
        json_dicts = _load_k8s_job_config(
            _read_json(path),
            JobConfig,
            json_ignore_params=json_ignore_params,
        )
        return TypeAdapter(JobConfig).validate_python(json_dicts)


@kumo_dataclass
class TextEmbeddingSize:
    model_name: str
    num_tokens: int
    embedding_dim: int


@kumo_dataclass
class ModuleSize:
    encoder_num_params: int
    model_num_params: int
    head_num_params: int
    total_num_params: int
    encoder_num_bytes: int
    model_num_bytes: int
    head_num_bytes: int
    total_num_bytes: int
    channels: int
    num_aggrs: int
    num_hops: int
    text_embedding_sizes: list[TextEmbeddingSize] = field(default_factory=list)


@kumo_dataclass
class SubgraphSize:
    num_nodes: int
    num_edges: int
    num_features: int
    num_feature_bytes: int
    batch_size: int
    num_rhs_feature_bytes: int | None = None


@kumo_dataclass
class TrialStatus:
    trial: int
    batch_size: int
    succeed: bool = False
    stacktrace: str | None = None
    module_size: ModuleSize | None = None
    subgraph_size: SubgraphSize | None = None

    @property
    def oomed(self) -> bool:
        return (self.stacktrace is not None
                and 'out of memory' in self.stacktrace)


@kumo_dataclass
class TrainStatus:
    # `trials` can contain retried trials, so the length of :attr:`trials` can
    # be greater than the number of AutoML experiments.
    num_tries: int = 0
    num_test_tries: int = 0
    trials: list[TrialStatus] = field(default_factory=list)
    allow_k8s_retry: bool = True
    error_msg: str | None = None
    stacktrace: str | None = None
    warning_msg: str | None = None
    succeeded: bool = False
    max_val_examples: int | None = None
    max_test_examples: int | None = None

    @staticmethod
    def file_path(home_dir: str) -> str:
        return osp.join(home_dir, 'status.json')

    @staticmethod
    def load(home_dir: str, debug: bool = False) -> 'TrainStatus':
        path = TrainStatus.file_path(home_dir)
        if not exists(path):
            return TrainStatus()

        try:
            return _load_dataclass(path, TrainStatus)
        except Exception as e:
            if debug:
                logger.error("Pass TrainStatus parse error in debug.")
                return TrainStatus()
            else:
                raise e

    def save(self, home_dir: str) -> None:
        path = TrainStatus.file_path(home_dir)
        _save_dataclass(self, path)

    def num_successful_trials(self) -> int:
        return sum([trial.succeed for trial in self.trials])

    def max_successful_batch_size(self) -> int | None:
        max_batch_size = 0
        for trial in self.trials[::-1]:
            if trial.succeed:
                max_batch_size = max(max_batch_size, trial.batch_size)
        return max_batch_size if max_batch_size > 0 else None

    def min_batch_size(self) -> int | None:
        if len(self.trials) == 0:
            return None
        min_batch_size = self.trials[0].batch_size
        for trial in self.trials[1:]:
            min_batch_size = min(min_batch_size, trial.batch_size)
        return min_batch_size

    def add_trial(self, batch_size: int) -> None:
        trial = TrialStatus(trial=len(self.trials), batch_size=batch_size)
        self.trials.append(trial)


@kumo_dataclass
class PredStatus:
    error_msg: str | None = None
    stacktrace: str | None = None

    @staticmethod
    def file_path(home_dir: str) -> str:
        return osp.join(home_dir, 'pred_status.json')

    @staticmethod
    def load(home_dir: str, debug: bool = False) -> 'PredStatus':
        path = PredStatus.file_path(home_dir)
        if not exists(path):
            return PredStatus()

        try:
            return _load_dataclass(path, PredStatus)
        except Exception as e:
            if debug:
                logger.error("Pass PredStatus parse error in debug.")
                return PredStatus()
            else:
                raise e

    def save(self, home_dir: str) -> None:
        path = PredStatus.file_path(home_dir)
        _save_dataclass(self, path)


# Everything related to PredictionConfig and JobConfig will go to
# train/config.py.
@kumo_dataclass
class PredictionConfig:
    r"""A lightweight configuration for a prediction job."""
    # The home directory for saving prediction data.
    home_dir: str
    # The home directory for initial training data loading.
    train_dir: str
    task_name: str
    # The model directory.
    model_dir: str
    output_dir: str
    return_embeddings: bool
    # Persist the requested artifact set explicitly. Do not infer this from
    # top_k: embeddings-only link prediction may carry any top_k value.
    return_predictions: bool
    pred_table_path: str = Field()
    # The persist directory for data backend.
    data_dir: str | None = None
    top_k: int | None = None
    binary_threshold: float | None = 0.5

    # List of table names whose feature should be loaded. If provided, only
    # these table's features would be loaded.
    tables_to_load: list[str] | None = None

    # Names of nodes to be loaded into the graphstore.
    nodes_to_load: list[str] | None = None

    diskgraph_index_path: str | None = None
    diskgraph_prediction_col_stats_path: str | None = None
    # TT-NG inference: predict task-dataset bundle dir, set by
    # materialize_prediction_task.
    diskgraph_task_dataset_bundle_dir: str | None = None
    # TT-NG inference (all None for non-DG / legacy): durable predict staging
    # dir beside the pred table that materialize + each predict shard re-ingest
    # from.
    diskgraph_pred_staging_dir: str | None = None
    # TARGET extension tt_id (``pred_<BP_job_id>``) the consumer reads and each
    # shard re-ingests INTO. The SOURCE (gen-keyed) staging is located by path
    # (``diskgraph_pred_staging_dir``) and its id read from the bundle spec, so
    # no source-id field is carried on the config.
    diskgraph_pred_target_tt_id: str | None = None

    # NOTE: necessary for backwards compatibility with v1.10-
    use_core_v2_api: bool = True

    # Multi-node batch prediction related config #############################
    # No.of partitions to split the prediction table into.
    # `kumo-pred` will split the prediction table into the specified number of
    # partitions and process each partition in a separate batch.
    # TT-NG BP materializes one prediction extension; pool sharded configs use
    # this as the runtime worker count passed to the DiskGraph iterator.
    #
    # 1 partition implies entire prediction table will be processed in a
    # single batch.
    num_partitions: int = 1
    # TT-NG BP only: logical output chunks used for save/flush cadence.
    output_num_partitions: int | None = None
    # The starting index in partitions of the prediction table.
    start_partition_id: int = 0
    # The ending index in partitions of the prediction table, exclusive.
    # E.g. start_partition_id=0, stop_partition_id=3 will run on partitions
    # [0, 1, 2].
    stop_partition_id: int = 1

    # Whether to disable explainers in BP
    disable_explain: bool = True

    # Attributes for customizing BP output format:
    start_time: datetime | None = None
    ignore_time_col: bool = False
    # If provided, the prediction output will use this column name as the
    # prediction time column.
    pred_time_col_name: str | None = None
    # Metadata columns requested for each prediction artifact. The producer
    # uses this to write the final customer-facing schema directly.
    output_extra_fields: dict[str, list[str]] = field(default_factory=dict)
    # Uncompressed size target for each producer-side Parquet file. Warehouse
    # outputs can lower this so upload concurrency remains memory bounded.
    max_prediction_file_size_bytes: int = 1024**3

    output_format: str = ''  # Deprecated.

    @property
    def effective_output_num_partitions(self) -> int:
        return self.output_num_partitions or self.num_partitions

    # v1 @root_validator(pre=True) -> @model_validator(mode='before').
    @model_validator(mode='before')
    @classmethod
    def _get_or_set_pred_table_path(cls, values):
        values = before_validator_dict(values)
        # For legacy, serialized PredictionConfig without `pred_table_path`
        # field, set it to the default value.
        if 'pred_table_path' not in values:
            values['pred_table_path'] = osp.join(
                values['home_dir'],
                PRED_TABLE,
                values['task_name'],
                PRED_TABLE,
            )
        return values

    @staticmethod
    def load(path: str) -> 'PredictionConfig':
        json_ignore_params = [
            'output_dir',
            'use_core_v2_api',
            'remote_graphengine_address',
            # Removed from PredictionConfig (DiskGraph is the only engine
            # backend); legacy JSON blobs may still carry it, so it is ignored.
            'graph_backend',
            # Ignored so historical PredictionConfig JSON blobs still
            # deserialize; the path dicts are no longer consumed.
            'table_names_with_paths',
            'node_names_with_paths',
            'edges_to_load',
            'edge_names_with_paths',
        ]
        config_raise_params = ['output_dir']
        json_dicts = _load_k8s_job_config(
            _read_json(path), PredictionConfig,
            json_ignore_params=json_ignore_params,
            config_raise_params=config_raise_params)
        return TypeAdapter(PredictionConfig).validate_python(json_dicts)

    @property
    def has_top_k(self) -> bool:
        return self.top_k is not None and self.top_k > 0


def _load_k8s_job_config(
    dicts: dict,
    cls: Any,
    json_ignore_params: list | None = None,
    json_raise_params: list | None = None,
    config_ignore_params: list | None = None,
    config_raise_params: list | None = None,
):
    r"""Loading k8s configs with forward and backward compatibility.
    """
    config_params = dict(inspect.signature(cls).parameters)
    json_dicts = deepcopy(dicts)
    # Params that can be ignored if exist in json but not in config class
    json_ignore_params = json_ignore_params or []
    for param in json_ignore_params:
        if param in json_dicts and param not in config_params:
            del json_dicts[param]
            logger.warning(f"Ignore {param} in the json file as "
                           f"it does not exist in {cls.__name__}")
    # Params that raise error if exist in json but not in config class
    json_raise_params = json_raise_params or []
    for param in json_raise_params:
        if param in json_dicts and param not in config_params:
            raise ValueError(f"Raise error for {param} in the json file as "
                             f"it does not exist in {cls.__name__}")
    # Params that use default value if exist in config class but not in json
    config_ignore_params = config_ignore_params or []
    for param in config_ignore_params:
        if param in config_params and param not in json_dicts:
            value = config_params[param]
            logger.warning(f"Use {param}'s default value {value} in"
                           f"{cls.__name__} as it does not exist in "
                           f"the json file ")
    # Params that raise error if exist in config class but not in json
    config_raise_params = config_raise_params or []
    for param in config_raise_params:
        if param in config_params and param not in json_dicts:
            raise ValueError(f"Raise error for {param} in {cls.__name__} "
                             f"as it does not exist in json file")
    return json_dicts
