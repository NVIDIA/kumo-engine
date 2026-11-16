# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
from typing import List, Optional

from kumoapi.train import TrainingStage

from async_utils.sync import blocking_method
from io_utils import fsutil, logging
from io_utils.json import load_json
from service_framework.deploy_env import get_default_home_dir

_home_dir: Optional[str] = None
logger = logging.getLogger(__name__)


def get_home_dir() -> str:
    r"""Get the home directory used for storing all :obj:`kumo` data.

    If :meth:`set_home_dir` is not called, the path defaults to
    :obj:`"<repo>/.cache/kumo"` when running from a source checkout.
    """
    if _home_dir is not None:
        return _home_dir
    return get_default_home_dir()


def get_local_home_dir() -> str:
    r"""Returns a local path to use if `home_dir` is set to an S3 path.

    This path should be used for storing things that are frequently updated
    or doesn't have an S3 support. E.g. Kumo logging, trainer logs etc.

    If local_home_dir is set to some specific path, it'll return that path.
    Else, it returns the default location i.e. "<repo>/.cache/kumo" when
    running from a source checkout.
    """
    home_dir = get_home_dir()
    # TODO(daniel) tech debt, cleanup in #15854
    if not fsutil.is_local(home_dir):
        return get_default_home_dir()
    return home_dir


def get_local_model_dir() -> str:
    return osp.join(get_local_home_dir(), 'models')


def set_home_dir(path: str):
    r"""Set the cache directory used for storing all :obj:`kumo` data.

    Args:
        path (str): Absolute local path or S3 path.
    """
    if fsutil.is_s3(path) or fsutil.is_gcs(path):
        if not fsutil.exists(path):
            raise ValueError(
                f"Provided path: {path} doesn't exist. Please create the "
                "folder before using it as home_dir")

    # Create home_dir if it does not exist:
    fsutil.makedirs(path, exist_ok=True)
    logger.debug("Setting home dir to %s", path)
    global _home_dir
    _home_dir = path


class HomeDirManager:
    """Defines the directory layout of the home dir, which contains the files
    for a single training or batch prediction job in Kumo.

    Going forward, all code (especially code outside the core kumo package)
    should define file paths through this class.  The reason is that we want to
    centralize the logic about file organization into a single place, to (1)
    make it easier for engineers to understand the directory layout and
    discover files (2) make it possible to unit test logic related to the
    directory structure (3) enable safe changes or migrations to the directory
    structure in a backwards-compatible manner
    """
    def __init__(self, home_dir: str) -> None:
        self._home_dir = home_dir
        self._best_trainer_config = None

    @property
    def materialization_task_info_file(self) -> str:
        return osp.join(self.outputs_dir, "materialization_task_info.json.gz")

    @property
    def training_job_input_file(self) -> str:
        return osp.join(self.inputs_dir, 'training_job_input.json')

    @property
    def generate_train_table_job_input_file(self) -> str:
        # TODO(frg100): Create a GenerateTrainTableJobInput class
        return osp.join(self.inputs_dir, 'training_job_input.json')

    @property
    def prediction_job_input_file(self) -> str:
        return osp.join(self.inputs_dir, 'prediction_job_input.json')

    @property
    def ml_job_settings_file(self) -> str:
        return osp.join(self.inputs_dir, 'ml_job_settings.json')

    @property
    def advanced_options_file(self) -> str:
        return osp.join(self.inputs_dir, 'advance_options.json')

    @property
    def home_dir(self) -> str:
        return self._home_dir

    @property
    def outputs_dir(self) -> str:
        """Directory containing outputs of the training/batch prediction job,
        which are intended to consumed by downstream tasks/activities. The
        files in this directory are generally expected to have stable formats,
        as downstream code in other packages (eg. REST API), may directly read
        this data and show it in the UI.
        """
        return osp.join(self._home_dir, "outputs")

    @property
    def inputs_dir(self) -> str:
        """Directory containing persisted inputs for training/batch prediction
        jobs (*JobInput and MLJobSettings) for use to rerun the workflows for
        individual steps.
        """
        return osp.join(self._home_dir, "inputs")

    @property
    def pseudo_time_dir(self) -> str:
        return osp.join(self._home_dir, "pseudo_time")

    @property
    def pred_table_dir(self) -> str:
        """Dir of the prediction table."""
        return osp.join(self._home_dir, "pred_table")

    @property
    def train_table_dir(self) -> str:
        """Dir of the training table."""
        return osp.join(self._home_dir, "training_table")

    @property
    def custom_train_table_path(self) -> str:
        """Dir of the custom training table."""
        return osp.join(self._home_dir, "custom_train_table")

    @property
    def materialized_train_table_dir(self) -> str:
        """Dir of the materialized training table."""
        return osp.join(self.train_table_dir, "materialized")

    @property
    def materialized_train_table_config_path(self) -> str:
        """Path to the materialized training table config."""
        return osp.join(self.materialized_train_table_dir,
                        "training_table_config.json")

    @property
    def pred_table_path(self) -> str:
        """Path to the prediction table."""
        return osp.join(self.pred_table_dir, "pred_table.parquet")

    @property
    def train_table_path(self) -> str:
        """Path to the training table."""
        return osp.join(self.train_table_dir, "train_table.parquet")

    def automl_dir(self, task_name: str) -> str:
        """Directory containing the outputs of the AutoML job."""
        return osp.join(self._home_dir, 'models', task_name, 'automl')

    def best_trainer_config_path(self, task_name: str) -> str:
        """Path to the best trainer config."""
        return osp.join(self.automl_dir(task_name), 'best_trainer.json')

    @property
    def train_dir_overrides_file(self) -> str:
        """Contains TrainDirOverrides data structure"""
        return osp.join(self._home_dir, 'train_dir_overrides.json')

    @property
    def graphs_dir(self) -> str:
        return osp.join(self._home_dir, "graphs")

    @property
    def jobs_dir(self) -> str:
        return osp.join(self._home_dir, "jobs")

    # Path where JobConfig and PredictionConfig data structures are stored.
    # This is applicable for both training and batch prediction jobs.
    def job_config_path(self, job_id: str) -> str:
        return osp.join(self.jobs_dir, f'{job_id}.json')

    @property
    def input_data_stats_path(self) -> str:
        return osp.join(self.jobs_dir, 'input_data_stats.json')

    @property
    def estimated_input_data_stats_path(self) -> str:
        return osp.join(self.jobs_dir, 'estimated_input_data_stats.json')

    ###########################################################################

    # Path where AutoML ModelPlan is stored.
    # This is only applicable for training jobs.
    def job_automl_model_plan(self, job_id: str) -> str:
        return osp.join(self.jobs_dir, f'{job_id}_model_plan.json')

    @property
    def models_dir(self) -> str:
        return osp.join(self._home_dir, "models")

    @property
    def tasks_dir(self) -> str:
        return osp.join(self._home_dir, "tasks")

    @property
    def queries_dir(self) -> str:
        return osp.join(self._home_dir, "queries")

    @property
    def validation_dir(self) -> str:
        return osp.join(self._home_dir, "validation")

    @property
    def prediction_output_dir(self) -> str:
        "Location where batch prediction outputs are written."
        return osp.join(self._home_dir, "prediction_output")

    @property
    def internal_predictions_file(self) -> str:
        """Filepath to the internal raw predictions file"""
        return os.path.join(self.prediction_output_dir, 'predictions.parquet')

    @property
    def internal_embeddings_file(self) -> str:
        """Filepath to the internal raw embeddings file"""
        return os.path.join(self.prediction_output_dir, 'embeddings.parquet')

    @property
    def internal_transformed_predictions_file(self) -> str:
        """Filepath to the internal transformed predictions file"""
        return os.path.join(self.prediction_output_dir, 'transformed.parquet')

    @property
    def internal_transformed_predictions_metadata_file(self) -> str:
        """Filepath to the internal transformed predictions metadata"""
        return os.path.join(self.prediction_output_dir, 'sql_metadata.json')

    @property
    def persist_dir(self) -> str:
        return osp.join(self._home_dir, "persist")

    @property
    def graph_persist_dir(self) -> str:
        return osp.join(self.persist_dir, "edges")

    def validation_path(self, unique_id: str) -> str:
        """ Contains validations from the training or batch prediction job, in
        the data structure ValidationResponse.
        """
        return osp.join(self.validation_dir, f'{unique_id}.json')

    def train_table_mask_dir(self, split_fpid: str) -> str:
        return osp.join(self.persist_dir, 'train_task_mask_index', split_fpid)

    def subgraph_collector_dir(self, stage: TrainingStage,
                               best_trainer_dir: str | None = None) -> str:
        if stage == TrainingStage.PRED:
            return osp.join(self.home_dir, 'xai/subgraph_collector/pred/')
        if best_trainer_dir is None:
            raise ValueError("`best_trainer_dir` must be provided for "
                             "stages other than PRED")
        return osp.join(best_trainer_dir, 'subgraph_collector', stage.value)

    def rfm_dir(self, stage: TrainingStage,
                best_trainer_dir: str | None = None) -> str:
        r"""Returns the directory where the XAI RFMs are stored."""
        path = self.subgraph_collector_dir(stage, best_trainer_dir)
        return osp.join(path, 'rfm')

    def subgraph_json_path(self, stage: TrainingStage, entity_id: str,
                           best_trainer_dir: str | None = None,
                           old: bool = True) -> str:
        r"""Returns the path to the subgraph JSON file.
        Args:
            stage (TrainingStage): The stage of the subgraph.
            entity_id (str): The ID of the XAI entity.
            best_trainer_dir (str | None): The directory of the best trainer.
            old (bool): Whether to use the old nested JSON format.
        Returns:
            The path to the subgraph JSON file.
        """
        path = self.subgraph_collector_dir(stage, best_trainer_dir)
        if old:
            return osp.join(path, 'nested_json', f'{entity_id}.json')
        return osp.join(path, 'subgraph_json', f'{entity_id}.json')

    def subgraph_output_path(self, stage: TrainingStage, entity_id: str,
                             best_trainer_dir: str | None = None) -> str:
        path = self.subgraph_collector_dir(stage, best_trainer_dir)
        return osp.join(path, 'output', f'{entity_id}.json')

    def subgraph_historical_items_path(
            self, stage: TrainingStage, entity_id: str,
            best_trainer_dir: str | None = None) -> str:
        path = self.subgraph_collector_dir(stage, best_trainer_dir)
        return osp.join(path, 'historical_items', f'{entity_id}.json')

    def subgraph_mappings_path(self, stage: TrainingStage,
                               best_trainer_dir: str | None = None) -> str:
        path = self.subgraph_collector_dir(stage, best_trainer_dir)
        return osp.join(path, 'strategy_mappings.json')

    def rfm_path(self, stage: TrainingStage, entity_id: str,
                 best_trainer_dir: str | None = None) -> str:
        path = self.subgraph_collector_dir(stage, best_trainer_dir)
        return osp.join(path, 'rfm', f'{entity_id}.json')

    def column_analysis_dir(self, stage: TrainingStage) -> str:
        if stage == TrainingStage.PRED:
            return osp.join(self.home_dir, 'xai/column_analysis/pred/')
        # TODO (zecheng): Support for other training stages
        raise ValueError(f"Stage {stage} is unsupported")

    def mask_persist_path(self, stage: str, split_fpid: str):
        return osp.join(self.train_table_mask_dir(split_fpid),
                        f'{stage}_mask_index.parquet')

    @blocking_method()
    def touch_init_file(self) -> None:
        """Create the directory and `__init__` file for the home dir."""
        fsutil.makedirs(self._home_dir, exist_ok=True)
        fsutil.touch(osp.join(self._home_dir, '__init__'))

    # Trainer related functions  ##############################################
    def get_best_trainer_config_path(self, task_name: str) -> str:
        return osp.join(self.models_dir, task_name, 'automl',
                        'best_trainer.json')

    # TODO(kiriti): Change the behavior to return Optional[str] if
    # best_trainer doesn't exist
    @blocking_method()
    def get_best_trainer_dir(self, task_name: str) -> str:
        r"""Get the best trainer from the automl train directory."""
        if self._best_trainer_config is None:
            from kumo_core.job_config import BestTrainerConfig

            # NOTE: for V1 objects, when creating a PQ object from the SDK,
            # we lose the PQ name. In V1, the name is used as an ID
            # and is part of the path directory where the best
            # trainer is stored. Because there is only one query per
            # training job, we can just fetch the first directory
            # in the path, regardless of whether we have the query
            # name (directory name) or not.
            first_task_name = self.first_task_name()
            assert first_task_name is not None
            task_name = first_task_name
            self._best_trainer_config = load_json(
                self.get_best_trainer_config_path(task_name),
                BestTrainerConfig,
            )
        # NOTE: Update the stored best trainer path for the same reason listed above.
        self._best_trainer_config.best_trainer = self.make_best_trainer_path_V1_and_v2_compat(
            self._best_trainer_config.best_trainer)
        return self._best_trainer_config.best_trainer

    def get_automl_dir(self, task_name: str) -> str:
        r"""Get the automl directory from the automl train directory"""
        return osp.join(self.models_dir, task_name, "automl")

    def split_train_table_dir(self, split_fpid: str) -> str:
        return osp.join(self.outputs_dir, split_fpid)

    def split_train_table_path(self, split_fpid: str,
                               stage: TrainingStage) -> str:
        return osp.join(
            self.split_train_table_dir(split_fpid),
            'split_tables',
            f'{stage.value}.parquet',
        )

    def split_train_table_stats(self, split_fpid: Optional[str] = None) -> str:
        """Contains dataclass SplitTrainTableStats as json.

        This contains label distribution stats for the target column of
        predictive queries.
        """
        if split_fpid is None:
            # Keep the old path for backwards compatibility
            return osp.join(
                self.outputs_dir,
                "split_train_table_stats.json",
            )
        return osp.join(
            self.split_train_table_dir(split_fpid),
            "split_train_table_stats.json",
        )

    @property
    def train_table_stats(self) -> str:
        """Contains dataclass TrainTableStats as json.

        This contains label distribution stats for the target column of
        predictive queries.
        """
        return osp.join(self.outputs_dir, "pquery_table_stats.json")

    @property
    def pred_table_stats(self) -> str:
        """Contains dataclass PredTableStats as json.
        This contains the anchor time for the prediction table.

        NOTE: right now the batch prediction table stats file has the
        exact same naming as train table stats, so that we can re-use the
        same code for both train and batch prediction table stats.
        """
        return self.train_table_stats

    @blocking_method()
    def get_regression_heatmap_path(self, stage: TrainingStage,
                                    task_name: str) -> str:
        best_trainer_dir = self.get_best_trainer_dir(task_name)
        return os.path.join(
            best_trainer_dir,
            f'{stage.value}_target_prediction_heatmap/heatmap.json')

    @property
    def all_target_regression_stats(self) -> str:
        """Contains dataclass AllTargetRegressionStats as json.

        This only is available for regression predictive queries, and contains
        distribution stats for both the labels and predictions of the target
        column.
        """
        return osp.join(self.outputs_dir, "all_target_regression_stats.json")

    def first_task_name(self) -> Optional[str]:
        """Get the first task under the home directory."""
        task_name = None
        model_folder = self.models_dir
        if not fsutil.exists(model_folder):
            return None

        tasks = fsutil.listdir(model_folder, max_items=1)
        if len(tasks) > 0:
            task_name = osp.basename(tasks[0].rstrip('/\\'))

        return task_name

    def task_config_path(self) -> Optional[str]:
        """
        Get the config path for the first (and only) task under the home
        directory.

        This may be None if we can't yet find a task name (eg the training
        job was just recently created and the directory hasn't been written to)
        """
        task_name = self.first_task_name()
        if task_name is None:
            return None
        return os.path.join(self.tasks_dir, task_name, 'config.json')

    def trials(self) -> List[str]:
        """Get a list of paths corresponding to each automl trial.
        """
        task_name = self.first_task_name()
        if task_name is None:
            return []

        models_path = osp.join(self.models_dir, task_name, "automl")
        model_config_dirs = fsutil.listdir(models_path)

        def _is_trial_path(path: str) -> bool:
            splits = path.strip('/').split('/')
            return splits[-1].startswith('version_')

        return list(filter(_is_trial_path, model_config_dirs))

    def make_best_trainer_path_V1_and_v2_compat(self,
                                                best_trainer_path: str) -> str:
        first_task_name = self.first_task_name()
        prefix = best_trainer_path.rsplit("models/", 1)[0] + 'models/'
        suffix = "/automl" + best_trainer_path.split("/automl")[1]
        final_path = prefix + first_task_name + suffix
        return final_path
