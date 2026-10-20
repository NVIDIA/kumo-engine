# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import gc
import logging
import logging as std_logging
import math
import os
import os.path as osp
import re
import sys
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import IO, Any, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pytorch_lightning as pl
import torch
from kumo_core.callbacks import (
    CudaOOMHandler,
    EarlyStopping,
    PredictionProgressLogger,
    TrainerProgress,
    TrainingProgressLogger,
)
from kumo_core.errors import TrainingValidationError
from kumo_core.explain import ExplainRuntime
from kumo_core.embedding_models import (
    EmbeddingModelStore,
    require_encoder_models,
    resolve_encoder,
)
from kumo_core.instrumentation import set_metrics_job_name, timing, trace
from kumo_core.io import filesystem as fsutil
from kumo_core.io.csv import save_csv
from kumo_core.io.filesystem import fsspec_mangle
from kumo_core.io.json import json_safe_float, load_json, save_json
from kumo_core.job_config import (
    ModuleSize,
    SubgraphSize,
    TextEmbeddingSize,
    TrialStatus,
)
from kumo_core.public_error import PublicTrainingError
from kumo_core.resolved_task import ResolvedNumericalTask
from kumo_core.train.checkpoint import KumoCheckpointIO
from kumo_core.train.formatting import format_embeddings, format_prediction
from kumo_core.train.hybrid_link_gnn_module import HybridLinkGNNLightningModule
from kumo_core.train.hyperedge_gnn_module import HyperedgeGNNLightningModule
from kumo_core.train.id_link_gnn_module import IDLinkGNNLightningModule
from kumo_core.train.link_gnn_module import LinkGNNLightningModule
from kumo_core.train.loss import (
    DistributionLoss,
    MultiQuantileLoss,
    loss_resolver,
)
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumo_core.train.node_gnn_module import NodeGNNLightningModule
from kumo_core.train.output import EvalOutput, PredOutput
from kumo_core.training.config import (
    RHS_NUM_EMB_SIZE_LIMIT,
    FitMode,
    LinkPredModuleKwargs,
    LinkPredModuleType,
    LoaderKwargs,
    RefitConfig,
    SamplerKwargs,
    TrainerConfig,
)
from kumo_core.training.options import allow_array_targets
from kumo_core.training.runtime import (
    DiskgraphTrainingRuntime,
    TrainingArtifactRuntime,
    TrainingDataRuntime,
)
from kumo_core.types import ExplainMethod, TaskType
from kumo_core.utils.config_mixin import PortableConfigMixin
from kumo_core.utils.names import get_table_name
from kumo_core.utils.registry import register
from kumo_core.utils.time import MAX_PSEUDO_TIME
from kumo_core.version import get_kumo_version
from kumoapi.encoder import TextEmbedding
from kumoapi.model_plan import WeightMode
from kumoapi.train import TrainingStage
from kumo_core.encoders.config import EncodersConfig
from kumo_core.features.stores.feature import FeatureStore
from kumo_core.features.stores.graph import GraphStore
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.evaluation.suite import MetricStore
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.encoders.time import TimeEncoder
from kumo_core.encoders.target.multicategorical import MultiCategoricalTargetEncoder
from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.explain.gradient_explainer import GradientExplainer
from kumo_core.explain.subgraph_collector import SubgraphCollector
from kumo_core.encoders.sequence.decomposed import DecomposedSequenceEncoder
from kumo_core.modeling.heads.forecasting import ForecastingHead
from kumo_core.modeling.gnn.transformer import GraphTransformer
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from kumo_core.encoders.sequence.mlp import MLPSequenceEncoder
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.encoders.sequence.normalized import NormalizedSequenceEncoder
from kumo_core.encoders.sequence.base import SequenceEncoder
from kumo_core.encoders.sequence.transformer import TransformerSequenceEncoder
from kumo_core.modeling.heads.year_over_year import YearOverYearEncoder
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_geometric.sampler.base import NumNeighbors, SubgraphType
from torch_geometric.typing import EdgeType, EdgeTypeStr

from .forecasting_gnn_module import ForecastingGNNLightningModule
from .shallow_rhs_link_gnn_module import ShallowRHSLinkGNNLightningModule

logger = logging.getLogger(__name__)

_MAX_PKEY_FETCH_SIZE = 10_000
_MAX_EVAL_TOP_K = 20
_DEFAULT_EVAL_TOP_K = 10
_LOG_EVERY_N_STEPS = 50

_TRAINER_COMPONENT_OWNERS: dict[str, type[Any]] = {
    'kumo_core.modeling.gnn.hetero.HeteroGNN':
    HeteroGNN,
    ('kumo_core.encoders.tensor_frame_encoder.'
     'HeteroTensorFrameEncoder'):
    HeteroTensorFrameEncoder,
    ('kumo_core.modeling.gnn.transformer.'
     'GraphTransformer'):
    GraphTransformer,
    ('kumo_core.modeling.heads.node_prediction.'
     'NodePredHead'):
    NodePredHead,
    ('kumo_core.modeling.heads.link_prediction.'
     'LinkPredHead'):
    LinkPredHead,
    ('kumo_core.modeling.heads.forecasting.'
     'ForecastingHead'):
    ForecastingHead,
    ('kumo_core.encoders.sequence.decomposed.'
     'DecomposedSequenceEncoder'):
    DecomposedSequenceEncoder,
    ('kumo_core.encoders.sequence.normalized.'
     'NormalizedSequenceEncoder'):
    NormalizedSequenceEncoder,
    ('kumo_core.encoders.sequence.mlp.'
     'MLPSequenceEncoder'):
    MLPSequenceEncoder,
    ('kumo_core.encoders.sequence.transformer.'
     'TransformerSequenceEncoder'):
    TransformerSequenceEncoder,
    ('kumo_core.modeling.heads.year_over_year.'
     'YearOverYearEncoder'):
    YearOverYearEncoder,
}


def _trainer_component_from_config(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f'{field_name} config must be an object')
    target = value.get('_target_')
    if not isinstance(target, str):
        raise ValueError(f'{field_name} config requires _target_')
    try:
        owner = _TRAINER_COMPONENT_OWNERS[target]
    except KeyError as error:
        raise ValueError(
            f'unsupported {field_name} config target: {target}') from error
    payload = {key: item for key, item in value.items() if key != '_target_'}
    return owner.from_config(payload)


# Suppress the verbose logging from torch.compile
logging.getLogger('torch._dynamo').setLevel(std_logging.ERROR)
logging.getLogger('torch._inductor').setLevel(std_logging.ERROR)


def _table_encoder_dict_for_model(
    task_type: TaskType,
    encoders: EncodersConfig,
    train_table_name: str,
) -> dict[str, Any]:
    """Return the encoder map used to initialize the model encoder."""
    if task_type != TaskType.FORECASTING:
        return encoders.encoders

    # Forecasting sequence inputs are encoded by dedicated past/yoy encoders,
    # not by the generic train-table TensorFrame encoder.
    table_encoder_dict = dict(encoders.encoders)
    table_encoder_dict[train_table_name] = {}
    return table_encoder_dict


def handle_gpu_oom(func):
    r"""A decorator to decrease the batch size by half in case of GPU OOMs."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        orig_batch_size = batch_size = None

        max_retries = 3
        for i in range(max_retries + 1):
            try:
                out = func(self, *args, **kwargs)
                if orig_batch_size is not None:
                    eval_loader_kwargs = self.datamodule.eval_loader_kwargs
                    eval_loader_kwargs['batch_size'] = orig_batch_size  # Reset
                return out
            except torch.cuda.OutOfMemoryError as e:
                eval_loader_kwargs = self.datamodule.eval_loader_kwargs
                if i == 0:
                    orig_batch_size = eval_loader_kwargs['batch_size']

                if i == max_retries:
                    raise e

                self.pl_model.reset_embedding_hook()
                torch.cuda.empty_cache()
                gc.collect()
                batch_size = math.ceil(eval_loader_kwargs['batch_size'] / 2)
                eval_loader_kwargs['batch_size'] = batch_size
                logger.warning(f"{e}. Temporarily decreasing the batch size "
                               f"from '{orig_batch_size}' to '{batch_size}'")
            except RuntimeError as e:
                # Re-raise the error if it's not a faiss OOM error
                if (not self.task.task_type.is_link_pred
                        or "faiss" not in repr(e)):
                    raise e
                logger.warning(f"{e}. Runtime error interpreted as Faiss OOM, "
                               f"exiting the prediction.")
                rhs_n_limit, rhs_dim_limit = RHS_NUM_EMB_SIZE_LIMIT
                dim = self.head.embedding_dim
                rhs_name = self.feature_store.entity_table_names[-1]
                num_rhs_nodes = self.feature_store.get_tensor_size(rhs_name)[0]
                raise TrainingValidationError(
                    title="GPU out-of-memory error",
                    message=f"Table '{rhs_name}' contains {num_rhs_nodes} "
                    f"rows, and 'output_embedding_dim' is set to "
                    f"{dim}. This configuration ran out of GPU memory. "
                    f"For tables over 5 million rows, we recommend setting "
                    f"'output_embedding_dim' to {rhs_dim_limit} or "
                    f"lower.") from e

    return wrapper


def failed_trial_handler(function: Callable) -> Callable:
    r"""Decorator that performs postprocessing functions on a trial if the
    decorated function fails. In particular, it:
        * Moves the log directory to `failed_trials/log_dir`
        * Deletes the existing log directory under successful trials
    """
    @wraps(function)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        try:
            if self._raise_fake_trial_error_msg is not None:
                # Only included in unit tests:
                raise RuntimeError(self._raise_fake_trial_error_msg)
            return function(self, *args, **kwargs)
        except Exception as e:
            head_tail = osp.split(self.log_dir)
            failed_trial_log_dir = osp.join(head_tail[0], 'failed_trials',
                                            head_tail[1])
            fsutil.rm(failed_trial_log_dir, ignore_errors=True)
            logger.warning(
                f"Trial failed with exception {str(e)}. Moving version_dir"
                f" from {self.log_dir} to {failed_trial_log_dir}")
            fsutil.copy(self.log_dir, failed_trial_log_dir)
            fsutil.rm(self.log_dir)
            raise

    return wrapper


def handle_refit_gpu_oom(function: Callable) -> Callable:
    r"""Decorator that provides a user-friendly error message for CUDA OOM
    errors during refit.

    Unlike normal Trainer.fit() calls for which AutoTrainer has retry logic
    with reduced batch sizes, refit does not retry because it must use the
    exact configuration that achieved the best validation performance. A CUDA
    OOM during refit will therefore fail the entire training job.
    """
    @wraps(function)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return function(self, *args, **kwargs)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e):
                raise e

            logger.exception("GPU out-of-memory error during refit")
            raise PublicTrainingError(
                "GPU out-of-memory error during refit. Because refit runs "
                "with the exact configuration that achieved the best "
                "validation performance, refit does not automatically retry "
                "by reducing batch size on a GPU out-of-memory error. A "
                "workaround is to have your model plan be less GPU-memory "
                "intensive, e.g., by reducing batch size.") from None

    return wrapper


@register(data_cls=TrainerConfig)
class Trainer(PortableConfigMixin):
    r"""Creates a KUMO trainer object.

    The default log directory for fit mode is `{root_dir}/{task_name}`, under
    which `config.json` is also stored.
    If we use refit_trainval or refit_full modes, the log direcotry is shifted
    to `{root_dir}/{task_name}_refit_trainval` and
    `{root_dir}/{task_name}_refit_full`, respectively. All tensorboard and
    checkpoints will be written-in/read-from the respective directory.

    `TensorBoard <https://www.tensorflow.org/tensorboard>`_ support
    available via:

    .. code-block::

        tensorboard --logdir {root_dir}/{task_name}
        open http://localhost:6006

    Args:
        task (TaskConfig): The KUMO task object.
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The GNN model.
        head (torch.nn.Module): The model head.
        past_encoder (torch.nn.Module): A past sequence encoder.
        year_over_year_encoder (torch.nn.Module): A year-over-year sequence
            encoder.
        root_dir (str, optional): Root directory for saving logs and
            checkpoints (default: `"{kumo.get_home_dir()}/models"`)
        checkpoint_path (str, optional): Path to the checkpoint file to load
            a model from.
        module_kwargs (Dict[str, Any], optional): Module-specific arguments.
        sampler_kwargs (Dict[str, Any], optional): Sampler-specific arguments.
            (default: :obj:`None`)
        loader_kwargs (Dict[str, Any], optional): Dataloader-specific
            arguments. (default: :obj:`None`)
        eval_loader_kwargs (Dict[str, Any], optional): Arguments to override
            in the evaluation loaders. (default: :obj:`None`)
        optimizer (str, optional): The optimizer to use.
            (default: :obj:`"Adam"`)
        optimizer_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective optimizer defined by :obj:`optimizer`.
            (default: :obj:`None`)
        lr_scheduler (str, optional): The learning rate scheduler to use.
            Supports LR schedulers from both PyG and PyTorch.
            (default: :obj:`"cosine_with_warmup_restarts"`)
        lr_scheduler_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective learning rate scheduler defined by :obj:`lr_scheduler`.
            (default: :obj:`None`)
        lr_scheduler_interval (str, optional): The scheduling interval of
            :obj:`lr_scheduler` to call :obj:`step` method: :obj:`"step"` or
            :obj:`"epoch"`. (default: :obj:`step`)
        early_stopping (bool, optional): Whether to use :obj:`EarlyStopping`.
            (default: :obj:`True`)
        early_stopping_kwargs (Dict[str, Any], optional): Arguments passed to
            the early stopping callback defined by :obj:`EarlyStopping`.
            (default: :obj:`None`)
        max_epochs (int, optional): The maximum number of epochs to train.
            (default: :obj:`100`)
        max_steps_per_epoch (int): The maximum number of training iterations
            per epoch. (default: :obj:`2000`)
        min_steps_per_epoch (int): The minimum number of training iterations
            per epoch. (default: :obj:`1`)
        max_val_steps (int, optional): The maximum number of
            validation iterations. (default: :obj:`2000`)
        max_test_steps (int, optional): The maximum number of
            test iterations. (default: :obj:`4000`)
        track_version (bool, optional): Whether to track different versions of
            the same experimental name. (default: :obj:`True`)
        version (int, optional): The version number of the experiment (in case
            :obj:`track_version` is set).
            If set to an existing version number, will load the given model
            checkpoint and resume training. (default: :obj:`None`)
        tune_metric (str, optional): Which metric to monitor.
            (default: :obj:`None`)
        loss (str, optional): A custom loss function from
            :obj:`torch.nn.functional`. (default: :obj:`None`)
        distance_measure (Literal['dot_product', 'cosine']): The measure
            method of comparing semantic similarity in link prediction. If
            `'cosine'`, all Shallow RHS vectors are normalized.
            (default: :obj:`'cosine'`)
        binary_threshold (float, optional): Threshold for binary
            classification tasks. (default: :obj:`0.5`)
        metrics_client_job_name (str, optional): Job name to use for the
            `MetricsClient`. This is mainly required in case of `AutoTrainer`
            since the global state doesn't get picked up correctly.
        raise_fake_trial_error_msg (str, optional): A fake error message to
            raise for test purpose.
        enable_compile (bool, optional): If set, the model and head are
            compiled using torch.compile. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`pytorch_lightning.Trainer`.
    """
    @staticmethod
    def _validate_eval_new_visit(task_type: TaskType,
                                 eval_new_visit: bool) -> None:
        r"""Validate ``eval_new_visit`` against task type.

        Static LP never supports it (temporal-only), and TT-NG has no
        past-visit exclusion, so ``eval_new_visit`` is unsupported everywhere.
        """
        if not eval_new_visit:
            return
        if task_type == TaskType.STATIC_LINK_PREDICTION:
            raise ValueError("'eval_new_visit' is only supported for "
                             "temporal tasks")
        raise ValueError("'eval_new_visit' is not supported with TT-NG "
                         "enabled")

    def __init__(
        self,
        task: ResolvedNumericalTask,
        encoder: torch.nn.Module | None = None,
        model: torch.nn.Module | None = None,
        head: torch.nn.Module | None = None,
        past_encoder: torch.nn.Module | None = None,
        year_over_year_encoder: torch.nn.Module | None = None,
        root_dir: str | None = None,
        checkpoint_path: str | None = None,
        module_kwargs: LinkPredModuleKwargs | dict[str, Any] | None = None,
        sampler_kwargs: SamplerKwargs | dict[str, Any] | None = None,
        loader_kwargs: LoaderKwargs | dict[str, Any] | None = None,
        eval_loader_kwargs: dict[str, Any] | None = None,
        optimizer: str = "Adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        lookup_optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: str | None = "cosine_with_warmup_restarts",
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = "step",
        early_stopping: bool = True,
        early_stopping_kwargs: dict[str, Any] | None = None,
        max_epochs: int = 100,
        max_steps_per_epoch: int = 2000,
        min_steps_per_epoch: int = 1,
        max_val_steps: int | None = 2000,
        max_test_steps: int | None = 4000,
        track_version: bool = True,
        version: int | None = None,
        tune_metric: str | None = None,
        loss: str | None = None,
        distance_measure: Literal['dot_product', 'cosine'] = 'dot_product',
        binary_threshold: float = 0.5,
        metrics_client_job_name: str | None = None,
        raise_fake_trial_error_msg: str | None = None,
        trial: TrialStatus | None = None,
        overwrite_train_table_edge: bool | None = True,
        majority_sampling_ratio: float | None = None,
        weight_mode: WeightMode | None = None,
        random_state_seed: int | None = None,
        enable_compile: bool = False,
        is_reverse_edge_enabled: bool = True,
        # `FeatureStore`/`GraphStore` interface:
        feature_store: FeatureStore | None = None,
        graph_store: GraphStore | None = None,
        encoders: EncodersConfig | None = None,
        explain_runtime: ExplainRuntime | None = None,
        data_runtime: TrainingDataRuntime | None = None,
        artifact_runtime: TrainingArtifactRuntime | None = None,
        diskgraph_runtime: DiskgraphTrainingRuntime | None = None,
        embedding_models: EmbeddingModelStore | None = None,
        **kwargs,
    ) -> None:
        self._embedding_models = embedding_models

        # Filter useless PyTorch Lightning warning for `log_every_n_steps`
        # being greater than `steps_per_epoch`:
        warnings.filterwarnings('ignore', '.*value for log_every_n_steps.*')

        # NOTE In some case (e.g., TrainingJobActivities postprocess results of
        # the best trainer), `feature_store`, `graph_store`, and `encoders` are
        # `None`. However, we might want to force those arguments to be
        # non-`None` in the future.

        self.task = task
        self.feature_store = feature_store
        self.graph_store = graph_store
        self.encoders = encoders
        self.explain_runtime = explain_runtime
        self.data_runtime = data_runtime
        self.artifact_runtime = artifact_runtime
        self.diskgraph_runtime = diskgraph_runtime
        self._target_encoder: TargetEncoder | None = None

        if task.task_type.is_link_pred:
            if module_kwargs is None:
                module_kwargs = LinkPredModuleKwargs()
            module_kwargs = LinkPredModuleKwargs.cast(module_kwargs)

            if module_kwargs.module is None:
                if task.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                    module = LinkPredModuleType.HYBRID
                else:
                    module = LinkPredModuleType.SHALLOW_RHS
                module_kwargs.module = module

            if module_kwargs.rhs_emb_init_std is None:
                if module_kwargs.module == LinkPredModuleType.HYBRID:
                    module_kwargs.rhs_emb_init_std = 0.01
                else:
                    module_kwargs.rhs_emb_init_std = 1.0

            module_kwargs.validate()

            self._validate_eval_new_visit(task.task_type,
                                          module_kwargs.eval_new_visit)

        # AutoTrainer passes in a custom job name for the metrics client:
        if metrics_client_job_name is not None:
            set_metrics_job_name(metrics_client_job_name)

        # Refit logic #########################################################

        # Whether we are fitting the model for the first time, or "re-fitting"
        # a model based on data stored in `RefitConfig`, on the full dataset:
        # NOTE: This is required to be set before accessing `self.get_log_dir`
        # which in-turn calls `self.version_dir_name`.
        self.fit_mode: FitMode = FitMode.fit

        # Configurations stored at `fit` time to be re-used at `refit` time:
        self.refit_cfg = RefitConfig()

        # Process inputs ######################################################

        if root_dir is None:
            if self.artifact_runtime is None:
                raise ValueError(
                    "artifact_runtime is required when root_dir is omitted")
            root_dir = osp.join(
                self.artifact_runtime.default_model_root(),
                "models",
            )

        # Save input parameters
        self.root_dir = root_dir
        self.track_version = track_version

        # Get the version to use for this trainer's log directory:
        if self.track_version and version is None:
            path = self.root_dir
            if self.name is not None:
                path = osp.join(self.root_dir, self.name)
            versions = _get_versions(path)
            version = versions[-1] + 1 if len(versions) > 0 else 0
        self.version = version

        # Data Loading Specification ##########################################

        if sampler_kwargs is None:
            sampler_kwargs = SamplerKwargs()
        elif isinstance(sampler_kwargs, dict):
            sampler_kwargs = SamplerKwargs.cast(sampler_kwargs)

        # TODO: Clean up how training job workflow retrieves log_dir, so that
        # we can always require non-None `feature_store` and `graph_store`.
        # NOTE Currently, when loading Trainer in training job workflow via
        # TrainingJobActivities. In such cases, we likely don't need
        # sampler_kwargs anyway:
        if (feature_store is not None
                and feature_store.has_train_table_time_col):
            sampler_kwargs.time_attr = 'time'

        if task.task_type == TaskType.STATIC_LINK_PREDICTION:
            # For static link prediction, we want to sample valid links
            # uniformly, independent of pseudo time:
            sampler_kwargs.temporal_strategy = 'uniform'
            sampler_kwargs.time_attr = 'time'

        if (task.task_type.is_link_pred
                and module_kwargs.module.use_bidirectional_sampling):
            sampler_kwargs.subgraph_type = SubgraphType.bidirectional
        elif (hasattr(model, 'num_mp_layers')
              and model.num_mp_layers > sampler_kwargs.num_neighbors.num_hops):
            sampler_kwargs.subgraph_type = SubgraphType.bidirectional

        if loader_kwargs is None:
            loader_kwargs = LoaderKwargs()
        elif isinstance(loader_kwargs, dict):
            loader_kwargs = LoaderKwargs.cast(loader_kwargs)

        logger.info(f"Graph sampling arguments: {sampler_kwargs}")
        logger.info(f"Data loading arguments: {loader_kwargs}")

        # PyG + PyTorch Lightning Specifications ##############################

        if tune_metric is None:
            if self.task.task_type.is_link_pred:
                tune_metric = 'map@10'
                if tune_metric not in task.metrics:
                    tune_metric = task.metrics[0]
            else:
                tune_metric = 'loss'

        if early_stopping:
            early_stopping_kwargs = copy.copy(early_stopping_kwargs) or {}
            early_stopping_kwargs.pop('monitor', None)
            early_stopping_kwargs.setdefault('patience',
                                             max(max_epochs // 5, 3))

        # Save input arguments ################################################

        self.head = head
        self.model = model
        self.encoder = encoder
        self.past_encoder = past_encoder
        self.year_over_year_encoder = year_over_year_encoder
        self.module_kwargs = module_kwargs
        self.sampler_kwargs = sampler_kwargs
        self.loader_kwargs = loader_kwargs
        self.eval_loader_kwargs = eval_loader_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lookup_optimizer_kwargs = lookup_optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_interval = lr_scheduler_interval
        self.early_stopping = early_stopping
        self.early_stopping_kwargs = early_stopping_kwargs
        self.max_epochs = max_epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.min_steps_per_epoch = min_steps_per_epoch
        self.max_val_steps = max_val_steps
        self.max_test_steps = max_test_steps
        self.tune_metric = tune_metric
        self.distance_measure = distance_measure
        self._raise_fake_trial_error_msg = raise_fake_trial_error_msg
        self.trial = trial
        self.overwrite_train_table_edge = overwrite_train_table_edge
        self.enable_compile = enable_compile
        self.weight_mode = weight_mode
        self.random_state_seed = random_state_seed
        self.is_reverse_edge_enabled = is_reverse_edge_enabled
        self.checkpoint_path = checkpoint_path

        # TODO: Provide a better mapping from :obj:`"loss"` name to
        # `torch.nn.functional`
        self.loss = loss
        self.binary_threshold = binary_threshold
        self.kwargs = kwargs

        # Allow Trainer instance to be created with deferred initialization via
        # later call to self.init()
        self._fully_initialized = False

        # Indicate whether a trainer is used for the prediction_mode. A trainer
        # that is used for the prediction_mode cannot be used for training.
        self._prediction_mode = False
        self._prediction_model_weight_loaded = False

        # Log commit to track the version of the code used by the trainer.
        self.kumo_version = get_kumo_version()

        if (majority_sampling_ratio is not None
                and self.task.task_type != TaskType.BINARY_CLASSIFICATION):
            raise ValueError(f"{majority_sampling_ratio=}, while task "
                             f"type is not binary classification but "
                             f"{self.task.task_type.value} instead.")
        self.majority_sampling_ratio = majority_sampling_ratio
        self.majority_sampling_weights = None

    @property
    def _explain_runtime(self) -> ExplainRuntime:
        if self.explain_runtime is None:
            raise RuntimeError(
                "explain_runtime is required for explainability operations")
        return self.explain_runtime

    @property
    def num_targets(self) -> int:
        assert self.encoders is not None
        assert not self.task.task_type.is_link_pred

        if (self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION
                or self.task.task_type.is_multilabel):
            return self.target_encoder.num_categories

        return 1

    @property
    def val_tune_metric(self) -> str:
        return TrainingStage.VAL.add_stage_prefix(self.tune_metric)

    @property
    def encoders(self) -> EncodersConfig | None:
        return self._encoders

    @encoders.setter
    def encoders(self, encoders: EncodersConfig | None) -> None:
        store = getattr(self, '_embedding_models', None)
        if store is not None and encoders is not None:
            # Text encoders then resolve to this store's models.
            require_encoder_models(encoders, store)
        self._encoders = encoders

    @classmethod
    def customize_from_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name in (
                'encoder',
                'model',
                'head',
                'past_encoder',
                'year_over_year_encoder',
        ):
            values[field_name] = _trainer_component_from_config(
                values[field_name], field_name)
        return values

    def _maybe_compile(self) -> None:
        """No-op if compile not enabled or already compiled."""
        if not self.enable_compile:
            return
        if not isinstance(self.encoder, torch._dynamo.OptimizedModule):
            self.encoder = torch.compile(self.encoder, dynamic=True)
        if not isinstance(self.model, torch._dynamo.OptimizedModule):
            self.model = torch.compile(self.model, dynamic=True)
        if not isinstance(self.head, torch._dynamo.OptimizedModule):
            self.head = torch.compile(self.head, dynamic=True)

    def init(self) -> 'Trainer':
        r"""Initialize the trainer."""

        if self._fully_initialized and not self.fit_mode.is_refit:
            return self

        # Initialize GNN Model ################################################

        num_entities: int | None = None
        if self.task.task_type == TaskType.FORECASTING:
            num_entities = self.feature_store.get_tensor_size(
                self.feature_store.entity_table_names[0], 'tf')[0]

        self.initialize_model(
            node_types=sorted(self.feature_store.table_names),
            edge_types=self.graph_store.edge_types,
            train_table_name=self.feature_store.train_table_name,
            has_train_table_time_col=self.feature_store.has_train_table_time_col,
            time_col_dict=self.feature_store.time_col_dict,
            entity_table_names=self.feature_store.entity_table_names,
            num_entities=num_entities,
        )

        # Instantiate graph samplers ##########################################

        if (self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION
                and self.head.out_channels == 1):
            logger.warning("Setting the task type to binary classification "
                           "since there only exists two classes")
            self.task.task_type = TaskType.BINARY_CLASSIFICATION

        assert self.feature_store is not None
        assert self.graph_store is not None

        with timing("trainer.init.create_graph_sampler"):
            if self.data_runtime is None:
                raise RuntimeError(
                    "data_runtime is required to initialize legacy training")
            self.graph_sampler = self.data_runtime.create_graph_sampler(
                self.feature_store,
                self.graph_store,
                self.sampler_kwargs,
                random_seed=self.random_state_seed,
            )

        # Apply diskgraph auto-configured num_workers to DataLoader.
        _dg_cfg = getattr(self.graph_sampler, '_auto_config', None)
        if _dg_cfg is not None:
            self.loader_kwargs.num_workers = _dg_cfg['num_workers']
            logger.info(
                "DiskGraph auto_config applied: num_workers=%d, "
                "rayon_threads=%d, mode=%s", _dg_cfg['num_workers'],
                _dg_cfg['rayon_threads'], _dg_cfg['mode'])

        # Create PyTorch Lightning Modules ####################################

        entity_table_names = self.feature_store.entity_table_names
        train_table_name = self.feature_store.train_table_name
        with timing("trainer.init.load_train_table_tensors"):
            fkey = self.feature_store[train_table_name, 'fkey', :]
            train_mask = self.feature_store[train_table_name, 'train_mask', :]
            val_mask = self.feature_store[train_table_name, 'val_mask', :]
        logger.info(f"Obtained foreign keys of shape {fkey.shape}")
        logger.info(f"Obtained train mask of shape {train_mask.shape} with "
                    f"{int(train_mask.sum())} positive entries")
        logger.info(f"Obtained validation mask of shape {val_mask.shape} with "
                    f"{int(val_mask.sum())} positive entries")
        test_mask: Tensor | None = None
        if self.feature_store.has_train_table_test_mask:
            test_mask = self.feature_store[train_table_name, 'test_mask', :]
            logger.info(f"Obtained test mask of shape {test_mask.shape} with "
                        f"{int(test_mask.sum())} positive entries")
        time: Tensor | None = None
        if self.feature_store.has_train_table_time_col:
            time = self.feature_store[train_table_name, 'time', :]
            logger.info(f"Obtained time vector of shape {time.shape}")
        weight: Tensor | None = None
        # Prediction does not use `weight` column. Weight column
        # is needed mainly for loss and metrics calculation.
        if (self.feature_store.has_train_table_weight_col
                and not self._prediction_mode):
            weight = self.feature_store[train_table_name, 'weight', :]
            logger.info(f"Obtained weight vector of shape {weight.shape}")
        target: Tensor | None = None
        if (self.task.task_type in {
                TaskType.FORECASTING,
                TaskType.BINARY_CLASSIFICATION,
        } and not self._prediction_mode):
            target = self.feature_store[train_table_name, 'y', :]
            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                target = self.target_encoder.encode_tensor(target)

        # Compute the majority sampling strength for binary classification:
        if (self.majority_sampling_ratio is not None
                and self.task.task_type == TaskType.BINARY_CLASSIFICATION
                and not self._prediction_mode):
            assert target is not None
            train_target = target[train_mask].view(-1).long()
            train_target_count = train_target.bincount(minlength=2)
            assert train_target_count.numel() == 2
            target_distribution = {
                False: int(train_target_count[0]) / train_target.numel(),
                True: int(train_target_count[1]) / train_target.numel(),
            }
            self.majority_sampling_weights = generate_majority_sampling_weight(
                self.majority_sampling_ratio,
                target_distribution,
            )

        if self.fit_mode.is_refit:
            if self.fit_mode == FitMode.refit_full:
                train_mask = torch.ones_like(train_mask)
            elif self.fit_mode == FitMode.refit_trainval:
                if test_mask is not None:
                    train_mask = train_mask | val_mask
                else:
                    logger.warning("Test split is not available. Refit "
                                   "reduces to normal fit")

        # PL data module, containing logic that interfaces with the feature and
        # graph store to load data for the model:
        with timing("trainer.init.create_datamodule"):
            assert self.data_runtime is not None
            self.datamodule = self.data_runtime.create_datamodule(
                task=self.task,
                fkey=fkey,
                entity_table_names=entity_table_names,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                time=time,
                target=target,
                weight=weight,
                feature_store=self.feature_store,
                graph_store=self.graph_store,
                graph_sampler=self.graph_sampler,
                loader_kwargs=self.loader_kwargs,
                module_kwargs=self.module_kwargs,
                eval_loader_kwargs=self.eval_loader_kwargs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                min_steps_per_epoch=self.min_steps_per_epoch,
                max_val_steps=self.max_val_steps,
                max_test_steps=self.max_test_steps,
                majority_sampling_weights=self.majority_sampling_weights,
                weight_mode=self.weight_mode,
                is_prediction=self._prediction_mode,
                fit_mode=self.fit_mode,
            )

        # PL trainer, containing metadata about the training pipeline:
        with timing("trainer.init.create_pl_trainer"):
            self.create_pl_trainer()

        # PL model, containing the actual model (subclass of `base_module.py`):
        rhs_name = entity_table_names[-1]
        with self.feature_store.safe_batch_mode(batch_size=50_000):
            rhs_index = self.feature_store[rhs_name, 'pkey', :]
        rhs_index_map = pd.DataFrame(index=rhs_index)
        rhs_index_map[NODE_INDEX_COLUMN] = range(len(rhs_index_map))

        with timing("trainer.init.create_pl_model"):
            self.create_pl_model(
                entity_table_names=entity_table_names,
                train_table_name=train_table_name,
                rhs_index_map=rhs_index_map,
                ckpt_path=self.checkpoint_path,
            )

        if self.fit_mode == FitMode.fit:
            num_samples = self.datamodule.num_samples_per_epoch
            self.refit_cfg.num_samples_per_epoch = num_samples

        # Save Trainer Configuration ##########################################

        if not self._prediction_mode:
            path = osp.join(self.original_log_dir, 'config.json')
            if not fsutil.exists(path):
                save_json(self.config(), path)

        subgraph_size = None
        if isinstance(self.model,
                      GraphTransformer) and not self.model.is_fully_specified:
            # This block should be inside `initialize_model`. However, to
            # compute the subgraph_size, we need to initialize the `datamodule`
            # first, so we place it here.
            subgraph_size = self.get_subgraph_size()
            self.model.seq_len = GraphTransformer.get_seq_len(
                total_num_nodes=subgraph_size.num_nodes,
                batch_size=subgraph_size.batch_size,
            )

        self._fully_initialized = True

        if self.trial is not None:
            if self.trial.module_size is None:
                self.trial.module_size = self.get_module_size()
            if self.trial.subgraph_size is None:
                self.trial.subgraph_size = (subgraph_size
                                            if subgraph_size is not None else
                                            self.get_subgraph_size())

        return self

    def initialize_model(
        self,
        node_types: list[str],
        edge_types: list[EdgeType],
        train_table_name: str,
        has_train_table_time_col: bool,
        time_col_dict: dict[str, str],
        entity_table_names: list[str],
        num_entities: int | None = None,
    ) -> None:
        r"""Initializes the GNN model. If the model is already fully specified,
        this function is a no-op. Otherwise, the relevant attributes are set
        on the model in preparation for training or checkpoint loading."""
        encoder = self.encoder
        model = self.model
        head = self.head
        past_encoder = self.past_encoder
        year_over_year_encoder = self.year_over_year_encoder

        if not isinstance(model,
                          GraphTransformer) and not model.is_fully_specified:
            model.metadata = (node_types, edge_types)
            model.validate()

        if not encoder.is_fully_specified:
            encoder.table_encoder_dict = _table_encoder_dict_for_model(
                self.task.task_type,
                self.encoders,
                train_table_name,
            )

            encoder.out_channels = model.channels

            if (not has_train_table_time_col
                    or self.task.task_type == TaskType.STATIC_LINK_PREDICTION):
                encoder.time_col_dict = {}
            else:
                encoder.time_col_dict = time_col_dict

            encoder.exclude_cols_dict = {
                get_table_name(k): v
                for k, v in self.task.exclude_cols_dict.items()
            }

            encoder.validate()

        if past_encoder is not None and not past_encoder.is_fully_specified:
            assert isinstance(past_encoder, SequenceEncoder)
            assert self.task.lag_timesteps is not None

            past_encoder.in_channels = 1
            past_encoder.out_channels = model.channels
            past_encoder.length = self.task.lag_timesteps

        if (year_over_year_encoder is not None
                and not year_over_year_encoder.is_fully_specified):
            assert isinstance(year_over_year_encoder, YearOverYearEncoder)

            year_over_year_encoder.in_channels = 1 + TimeEncoder.out_channels
            year_over_year_encoder.out_channels = model.channels

        if not head.is_fully_specified:
            head.in_channels = model.channels

            if self.task.task_type.is_link_pred:
                assert isinstance(head, LinkPredHead)
                head.use_mlp_head = (
                    self.module_kwargs.module.use_bidirectional_sampling)

                if self.module_kwargs.module != LinkPredModuleType.TWO_TOWER:
                    if head.normalize_vectors:
                        raise ValueError("`normalize_vectors=True` is only "
                                         "supported for two-tower models")
                    head.use_rhs_projector = False
                else:
                    head.use_rhs_projector = True

            elif self.task.task_type == TaskType.FORECASTING:
                assert num_entities is not None
                assert isinstance(head, ForecastingHead)
                head.num_entities = num_entities
                if past_encoder is not None:
                    head.past_channels = past_encoder.out_channels
                else:
                    head.past_channels = self.task.lag_timesteps or 0
                if year_over_year_encoder is not None:
                    head.past_year_channels = (
                        year_over_year_encoder.out_channels)
                else:
                    head.past_year_channels = 0
                head.unix_step_size = self.task.unix_step_size

                loss = loss_resolver(self.task.task_type, self.loss)
                if isinstance(loss, MultiQuantileLoss):
                    head.out_channels = len(loss.quantiles)
                else:
                    head.out_channels = self.num_targets

                if isinstance(loss, DistributionLoss):
                    head.param_keys = (
                        loss.distribution.distribution_parameters)
                else:
                    head.param_keys = []

            else:
                assert isinstance(head, NodePredHead)

                loss = loss_resolver(
                    self.task.task_type,
                    self.loss,
                    self.target_encoder,
                )
                if isinstance(loss, MultiQuantileLoss):
                    head.out_channels = len(loss.quantiles)
                else:
                    head.out_channels = self.num_targets

                if isinstance(loss, DistributionLoss):
                    head.param_keys = (
                        loss.distribution.distribution_parameters)
                else:
                    head.param_keys = []

            head.validate()

    def get_module_size(self) -> ModuleSize:
        assert self._fully_initialized

        def get_num_params(module: torch.nn.Module) -> int:
            params = [p for p in module.parameters() if p.requires_grad]
            return sum([p.numel() for p in params])

        def get_num_bytes(module: torch.nn.Module) -> int:
            num_bytes = 0
            for p in module.parameters():
                num_bytes += p.numel() * p.element_size()
            for b in module.buffers():
                num_bytes += b.numel() * b.element_size()
            return num_bytes

        text_models: set[tuple[str, str]] = set()
        text_embedding_sizes: list[TextEmbeddingSize] = []
        for table_encoders in self.encoders.values():
            for encoder in table_encoders.values():
                if not isinstance(encoder, TextEmbedding):
                    continue
                model = resolve_encoder(encoder)
                key = (model.family, model.fingerprint())
                if key in text_models:
                    continue
                text_models.add(key)
                text_embedding_sizes.append(
                    TextEmbeddingSize(
                        model_name=encoder.model,
                        num_tokens=len(model.vocabulary()),
                        embedding_dim=model.dim,
                    ))

        # TODO: add GraphTransformer model stats
        return ModuleSize(
            encoder_num_params=get_num_params(self.pl_model.encoder),
            model_num_params=get_num_params(self.pl_model.model),
            head_num_params=get_num_params(self.pl_model.head),
            total_num_params=get_num_params(self.pl_model),
            encoder_num_bytes=get_num_bytes(self.pl_model.encoder),
            model_num_bytes=get_num_bytes(self.pl_model.model),
            head_num_bytes=get_num_bytes(self.pl_model.head),
            total_num_bytes=get_num_bytes(self.pl_model),
            channels=self.pl_model.model.channels,
            num_aggrs=len(self.pl_model.model.aggr) if hasattr(
                self.pl_model.model, 'aggr') else 0,
            num_hops=self.sampler_kwargs.num_neighbors.num_hops,
            text_embedding_sizes=text_embedding_sizes,
        )

    def get_subgraph_size(self) -> SubgraphSize:
        loader = self.datamodule.val_dataloader()
        batch_size = min(loader.batch_size, len(loader.dataset))
        batches = loader(torch.arange(batch_size))

        if not isinstance(batches, (tuple, list)):
            batches = (batches, )  # `HyperEdgeGNN` returns a tuple.

        num_features = num_feature_bytes = 0
        for batch in batches:
            if batch.num_nodes is None or batch.num_edges is None:
                raise ValueError(
                    'Batch of data was not sampled successfully, most '
                    'likely due to a Graph Engine failure.')
            for tf in batch.tf_dict.values():
                for feat in tf.feat_dict.values():
                    if isinstance(feat, (
                            MultiNestedTensor,
                            MultiEmbeddingTensor,
                    )):
                        feat = feat.values
                    assert isinstance(feat, Tensor)
                    num_features += feat.numel()
                    num_feature_bytes += feat.numel() * feat.element_size()

        num_rhs_feature_bytes = None
        if (isinstance(self.pl_model, ShallowRHSLinkGNNLightningModule)
                and self.module_kwargs.rhs_emb_mode.use_rhs_feature):
            rhs_name = self.feature_store.entity_table_names[-1]
            num_samples = min(1_000, self.pl_model.num_rhs_nodes)
            tf = self.feature_store[rhs_name, 'tf', torch.arange(num_samples)]

            num_rhs_feature_bytes = 0
            for feat in tf.feat_dict.values():
                if isinstance(feat, (
                        MultiNestedTensor,
                        MultiEmbeddingTensor,
                )):
                    feat = feat.values
                assert isinstance(feat, Tensor)
                num_rhs_feature_bytes += feat.numel() * feat.element_size()

            # A featureless RHS table yields an empty feature frame
            # (len(tf) == 0); a zero byte estimate is correct and avoids a
            # divide-by-zero in this OOM-diagnostic-only value.
            if len(tf) == 0:
                num_rhs_feature_bytes = 0
            else:
                num_rhs_feature_bytes = int(
                    num_feature_bytes * self.pl_model.num_rhs_nodes / len(tf))

        assert batch.num_nodes is not None
        assert batch.num_edges is not None

        return SubgraphSize(
            num_nodes=batch.num_nodes,
            num_edges=batch.num_edges,
            num_features=num_features,
            num_feature_bytes=num_feature_bytes,
            num_rhs_feature_bytes=num_rhs_feature_bytes,
            batch_size=batch_size,
        )

    def create_pl_model(
        self,
        entity_table_names: list[str],
        train_table_name: str,
        rhs_index_map: pd.DataFrame | None = None,
        num_rhs_nodes: int | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        r"""Creates a LightningModule from the trainer's GNN specification and
        task attributes.

        Args:
            ckpt_path: Path to the checkpoint to load the model from. If
                :obj:`None`, the model will be randomly initialized.
        """
        self._maybe_compile()

        kwargs = dict(
            encoder=self.encoder,
            model=self.model,
            head=self.head,
            train_table_name=train_table_name,
            metrics=self.task.metrics,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
            # Explicitly pass this so that max_epochs_for_lr_scheduler is
            # consistent between fit and refit modes:
            max_epochs_for_lr_scheduler=self.max_epochs,
            lr_scheduler_interval=self.lr_scheduler_interval,
            enable_compile=self.enable_compile,
        )

        if self.task.task_type.is_link_pred:
            if num_rhs_nodes is None:
                num_rhs_nodes = len(rhs_index_map)

            if self.module_kwargs.module == LinkPredModuleType.SHALLOW_RHS:
                # Learn RHS-aware LHS embeddings via RHS shallow embeddings
                # Support both temporal and static LP.
                self.pl_model = ShallowRHSLinkGNNLightningModule(
                    task_type=self.task.task_type,
                    entity_table_names=entity_table_names,
                    num_rhs_nodes=num_rhs_nodes,
                    rhs_index_map=rhs_index_map,
                    rhs_sample_size=self.module_kwargs.rhs_sample_size,
                    rhs_emb_mode=self.module_kwargs.rhs_emb_mode,
                    rhs_feat_bytes_threshold=self.module_kwargs.
                    rhs_feat_bytes_threshold,
                    rhs_emb_init_std=self.module_kwargs.rhs_emb_init_std,
                    emb_l2_reg_coeff=self.module_kwargs.emb_l2_reg_coeff,
                    eval_new_visit=self.module_kwargs.eval_new_visit,
                    distance_measure=self.distance_measure,
                    loss=self.loss or 'cross_entropy_loss',
                    weight_mode=self.weight_mode,
                    target_fraction=self.module_kwargs.target_fraction,
                    overlap_ratio=self.module_kwargs.overlap_ratio,
                    target_to_entity_paths=self.task.target_to_entity_paths,
                    lookup_optimizer_kwargs=self.lookup_optimizer_kwargs,
                    **kwargs,
                )

            elif self.module_kwargs.module == LinkPredModuleType.ID:
                # Learn pair/ID-aware representations:
                # Only support temporal LP that requires predicting repeated
                # links.
                # Disabled for static LP since ID-GNN is not suitable for
                # predicting never-seen links
                assert self.task.task_type == TaskType.TEMPORAL_LINK_PREDICTION
                self.pl_model = IDLinkGNNLightningModule(
                    entity_table_names=entity_table_names,
                    distance_measure=self.distance_measure,
                    **kwargs,
                )

            elif self.module_kwargs.module == LinkPredModuleType.HYBRID:
                # Hybrid of shallow RHS and ID-GNN:
                # Only support temporal LP that requires predicting repeated
                # links.
                # Disabled for static LP since ID-GNN part is not suitable for
                # predicting never-seen links
                assert self.task.task_type == TaskType.TEMPORAL_LINK_PREDICTION
                self.pl_model = HybridLinkGNNLightningModule(
                    entity_table_names=entity_table_names,
                    num_rhs_nodes=num_rhs_nodes,
                    rhs_index_map=rhs_index_map,
                    rhs_sample_size=self.module_kwargs.rhs_sample_size,
                    rhs_emb_mode=self.module_kwargs.rhs_emb_mode,
                    rhs_feat_bytes_threshold=self.module_kwargs.
                    rhs_feat_bytes_threshold,
                    rhs_emb_init_std=self.module_kwargs.rhs_emb_init_std,
                    emb_l2_reg_coeff=self.module_kwargs.emb_l2_reg_coeff,
                    emb_loss_coeff=self.module_kwargs.hybrid_emb_loss_coeff,
                    eval_new_visit=self.module_kwargs.eval_new_visit,
                    distance_measure=self.distance_measure,
                    loss=self.loss or 'cross_entropy_loss',
                    weight_mode=self.weight_mode,
                    lookup_optimizer_kwargs=self.lookup_optimizer_kwargs,
                    **kwargs,
                )

            else:
                # Otherwise, use regular link prediction code path that learns
                # two independent node representations and combines them:
                neg_amount = self.module_kwargs.rhs_sample_size or 1
                sampled_softmax_factor = math.log(num_rhs_nodes / neg_amount)
                self.pl_model = LinkGNNLightningModule(
                    task_type=self.task.task_type,
                    entity_table_names=entity_table_names,
                    emb_l2_reg_coeff=self.module_kwargs.emb_l2_reg_coeff,
                    loss_fn='margin',
                    sampled_softmax_factor=sampled_softmax_factor,
                    eval_new_visit=self.module_kwargs.eval_new_visit,
                    distance_measure=self.distance_measure,
                    **kwargs,
                )

        elif self.task.task_type == TaskType.FORECASTING:
            self.pl_model = ForecastingGNNLightningModule(
                past_encoder=self.past_encoder,
                year_over_year_encoder=self.year_over_year_encoder,
                forecast_length=self.task.forecast_length,
                unix_step_size=self.task.unix_step_size,
                arl_length=self.task.lag_timesteps or 0,
                target_encoder=self.target_encoder,
                entity_table_name=entity_table_names[0],
                loss=self.loss,
                loss_kwargs=dict(threshold=self.binary_threshold),
                **kwargs,
            )

        elif len(entity_table_names) == 1:
            self.pl_model = NodeGNNLightningModule(
                task_type=self.task.task_type,
                target_encoder=self.target_encoder,
                entity_table_name=entity_table_names[0],
                loss=self.loss,
                loss_kwargs=dict(threshold=self.binary_threshold),
                majority_sampling_weights=self.majority_sampling_weights,
                weight_mode=self.weight_mode,
                **kwargs,
            )

        else:  # len(entity_table_names) > 1:
            self.pl_model = HyperedgeGNNLightningModule(
                task_type=self.task.task_type,
                target_encoder=self.target_encoder,
                entity_table_names=entity_table_names,
                with_identity_awareness=True,
                loss=self.loss,
                loss_kwargs=dict(threshold=self.binary_threshold),
                **kwargs,
            )

        if ckpt_path:
            if (isinstance(self.pl_model, ShallowRHSLinkGNNLightningModule)
                    and self.module_kwargs.rhs_emb_mode.use_rhs_lookup
                    and not self._prediction_mode):
                # This code path should only be used for warm starting a model
                # from a checkpoint for TRAINING.
                self.pl_model.set_rhs_index_map(rhs_index_map)

            # Load model weights (and optimizer states for warm start training)
            load_optimizer_state = not self._prediction_mode

            self.pl_model = self.pl_model.load_checkpoint(
                fsspec_mangle(ckpt_path),
                map_location=torch.device('cpu'),
                load_optimizer_state=load_optimizer_state,
                _new_rhs_order=rhs_index_map
                if self.task.task_type.is_link_pred else None,
            )

            if load_optimizer_state:
                logger.info(
                    f"Initialized model and optimizer from checkpoint {ckpt_path}."
                    f" For task {self.task.name}")
            else:
                logger.info(f"Initialized model from checkpoint {ckpt_path}."
                            f" For task {self.task.name}")

    @property
    def target_encoder(self) -> TargetEncoder | None:
        if self.encoders.target is None or self.encoders.target.encoder is None:
            return None
        if self._target_encoder is not None:
            return self._target_encoder
        self._target_encoder = TargetEncoder.from_encoder(
            self.task.task_type, self.encoders.target.encoder)
        return self._target_encoder

    def create_pl_trainer(self):
        r"""Creates a PyTorch Lightning trainer from the trainer's
        hyperparameters and other metadata that influences the overall training
        pipeline."""

        # Enable Tensor Core optimization for compatible GPUs (A10, A100, etc.)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')

        # Handle callbacks ####################################################

        callbacks: list[pl.callbacks.Callback] = self.kwargs.pop(
            'callbacks', [])

        if self.early_stopping and not self.fit_mode.is_refit:
            early_stopping = EarlyStopping(
                mode=MetricStore.tune_mode(self.tune_metric),
                monitor=self.val_tune_metric,
                **self.early_stopping_kwargs,
            )
            callbacks.append(early_stopping)

        callbacks.append(ProgressBar(refresh_rate=_LOG_EVERY_N_STEPS))
        callbacks.append(CudaOOMHandler())

        if self.fit_mode == FitMode.fit:
            # By default we save the checkpoint with the best validation
            # performance:
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                dirpath=fsspec_mangle(self.ckpt_dir),
                save_top_k=1,
                monitor=self.val_tune_metric,
                mode=MetricStore.tune_mode(self.tune_metric),
                enable_version_counter=False,
            )
        else:
            # We save the last checkpoint in refit mode:
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                dirpath=fsspec_mangle(self.ckpt_dir),
                save_last=True,
                save_on_train_epoch_end=True,
            )
        callbacks.append(ckpt_callback)

        # # Add progress logging callback:
        self.training_progress_logger = TrainingProgressLogger(self)
        callbacks.append(self.training_progress_logger)

        # Set up Lightning trainer instance ###################################

        profiler = self.kwargs.get("profiler", None)
        if profiler == "simple":
            self.kwargs["profiler"] = SimpleProfiler(
                dirpath=self.log_dir,
                filename="simple_pytorch_profiler",
            )
        elif profiler == "advanced":
            self.kwargs["profiler"] = AdvancedProfiler(
                dirpath=self.log_dir,
                filename="advanced_pytorch_profiler",
            )

        max_epochs = self.max_epochs
        if self.fit_mode.is_refit:
            max_epochs = self.refit_cfg.best_val_epoch

        # No validation in refit mode:
        limit_val_batches: int | None = None
        if self.fit_mode.is_refit:
            limit_val_batches = 0

        # TensorBoard stays first: Lightning's ``log_dir``, which the stopper
        # reads, comes from the first logger.
        pl_logger: Any = pl.loggers.TensorBoardLogger(
            save_dir=self.get_log_dir(local=True),
            name='',
            version='',
            sub_dir='tensorboard',
        )
        extra_logger = self.kwargs.pop("logger", None)
        if extra_logger is not None:
            pl_logger = [pl_logger, extra_logger]

        plugins = self.kwargs.pop('plugins', None)
        if plugins is None:
            plugins = []
        elif isinstance(plugins, list):
            plugins = list(plugins)
        else:
            plugins = [plugins]
        plugins.append(KumoCheckpointIO())

        # Constructor kwargs override the placement defaults, e.g. a
        # batch-prediction worker pinned with ``devices=[i]``:
        placement = {
            'accelerator': os.environ.get('KUMO_ACCELERATOR', 'auto'),
            'strategy': 'auto',
            'devices': 1,
            **self.kwargs,
        }

        self.pl_trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=pl_logger,
            callbacks=callbacks,
            plugins=plugins,
            log_every_n_steps=_LOG_EVERY_N_STEPS,
            limit_val_batches=limit_val_batches,
            # TODO (akihironitta): Enable inference_mode for possible better
            # performance. Currently, `Calibration` requires False for
            # backward within its `optimizer.step(closure)`.
            inference_mode=False,
            **placement,
        )

    @property
    def best_ckpt(self) -> str:
        r"""Best checkpoint path based on the current fit mode.

        Raises:
            ValueError: If no best checkpoint is found.
        """
        best_ckpt, ckpt_paths = get_best_ckpt_path(self.ckpt_dir)
        if best_ckpt is None:
            raise ValueError(f"Unable to find the best checkpoint in "
                             f"{self.ckpt_dir}.")

        logger.info(f"Found checkpoints {ckpt_paths}, choosing {best_ckpt}.")
        return best_ckpt

    @property
    def name(self) -> str | None:
        return self.task.name

    @property
    def version_dir_name(self) -> str:
        return f'version_{self.version}'

    def get_log_dir(
        self,
        local: bool = False,
        original: bool = False,
        fit_mode: FitMode | None = None,
    ) -> str:
        r"""Switch the log directory based on `self.fit_mode`.
        `version_xx/`: fit mode.
        `version_xx_refit_trainval/`: refit_trainval mode.
        `version_xx_refit_full/` refit_full mode.
        This way, different modes will write logs in different log directories.

        If `original=True`, then we get the original (non-refitting) log
        directory. This way, we can update `config.json` in the original
        directory. `fit_mode` overrides `self.fit_mode`."""
        fit_mode = self.fit_mode if fit_mode is None else fit_mode
        path = self.root_dir if not local else self.local_root_dir

        # In case the `root_dir` contains a trainer config, it is a log dir
        # for a previous model version.
        if fsutil.exists(osp.join(path, 'config.json')):
            if not original and fit_mode.is_refit:
                if path[-1] == '/':
                    path = path[:-1]
                path = f"{path}_{fit_mode.value}"
                fsutil.makedirs(path, exist_ok=True)
                return path
            else:
                return path

        if self.name is not None:
            path = osp.join(path, self.name)
        if self.track_version:
            path = osp.join(path, self.version_dir_name)
        if not original and fit_mode.is_refit:
            if path[-1] == '/':
                path = path[:-1]
            path = f"{path}_{fit_mode.value}"
        fsutil.makedirs(path, exist_ok=True)
        return path

    @property
    def log_dir(self) -> str:
        return self.get_log_dir()

    @property
    def original_log_dir(self) -> str:
        r"""Original non-refitting log directory that stores config.json"""
        return self.get_log_dir(original=True)

    @property
    def local_root_dir(self) -> str:
        r"""Returns the local root directory of the model."""
        # TODO(daniel) tech debt, cleanup in #15854
        if not fsutil.is_local(self.root_dir):
            if self.artifact_runtime is None:
                raise RuntimeError(
                    "artifact_runtime is required for remote model roots")
            return self.artifact_runtime.local_model_root()

        return self.root_dir

    @property
    def ckpt_dir(self) -> str:
        path = osp.join(self.log_dir, 'checkpoint')
        fsutil.makedirs(path, exist_ok=True)
        return path

    def subgraph_collector_dir(self, root_dir: str | None = None) -> str:
        if root_dir is None:
            path = osp.join(self.log_dir, 'subgraph_collector')
        else:
            path = osp.join(root_dir, 'subgraph_collector')
        fsutil.makedirs(path, exist_ok=True)
        return path

    def column_analysis_dir(self, root_dir: str | None = None) -> str:
        if root_dir is None:
            path = osp.join(self.log_dir, 'column_analysis')
        else:
            path = osp.join(root_dir, 'column_analysis')
        fsutil.makedirs(path, exist_ok=True)
        return path

    def explain_dir(self, stage: str = 'train',
                    root_dir: str | None = None) -> str:
        if root_dir is None:
            path = osp.join(self.log_dir, 'explain_' + stage)
        else:
            path = osp.join(root_dir, 'explain_' + stage)
        fsutil.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def load(
        cls,
        task: ResolvedNumericalTask,
        root_dir: str,
        version: int | None = -1,
        infer_config_path: bool = True,
        feature_store: FeatureStore | None = None,
        graph_store: GraphStore | None = None,
        encoders: EncodersConfig | None = None,
        **kwargs,
    ) -> 'Trainer':
        r"""Loads a (pre-trained) :class:`Trainer` instance from the
        checkpoints in :obj:`root_dir` for a given :class:`TaskConfig`.

        `**kwargs` are passed along to the constructor of Trainer.
        Args:
            version (int, optional): If :obj:`version` is specified to be
                negative, versions will be index from the end, *i.e.*
                :obj:`version == -1` will load the checkpoint of the latest
                trained version.
            infer_config_path (bool): Whether to infer the Trainer config path
                using the version and task name. If set to false, the
                TrainerConfig is assumed to be stored at <root_dir/config.json>
                This is useful for loading a particular version Trainer from
                AutoTrainer generated models.
        """
        path = root_dir
        if infer_config_path:
            if task.name is not None:
                if fsutil.exists(osp.join(path, task.name)):
                    path = osp.join(path, task.name)

            # version can be negative when caller specified -1 to get the
            # latest version.
            if version is not None and version < 0:
                version = _get_versions(path)[version]

            if version is not None:
                path = osp.join(path, f'version_{version}')

        cfg = load_json(osp.join(path, 'config.json'), cls=TrainerConfig)
        logger.info(f"Loaded Trainer of version '{cfg.kumo_version}'")

        # Older trainer config files have FQNs in
        # cfg.sampler_kwargs.num_neighbors.values. Unqualify them for backwards
        # compatibility.
        if cfg.sampler_kwargs is not None:
            old_num_neighbors = cfg.sampler_kwargs.num_neighbors

            if isinstance(old_num_neighbors.values, dict):
                edge_type_by_serialized = {
                    str(EdgeTypeStr(*edge_type)): edge_type
                    for edge_type in (
                        getattr(graph_store, 'edge_types', None) or [])
                }

                def _convert_key(old_key: EdgeTypeStr) -> EdgeTypeStr:
                    serialized_key = str(old_key)
                    matching_edge_type = edge_type_by_serialized.get(
                        serialized_key)
                    if matching_edge_type is None:
                        src, rel, dst = old_key.to_tuple()
                    else:
                        src, rel, dst = matching_edge_type

                    return EdgeTypeStr(get_table_name(src), rel,
                                       get_table_name(dst))

                # Pass tuples (not EdgeTypeStr) to NumNeighbors() so the
                # constructor uses the 3-arg path in EdgeTypeStr.__new__,
                # and does not split on '__' which might break for tables with
                # trailing '_' in the column name.
                new_values = {
                    _convert_key(k).to_tuple(): v
                    for k, v in old_num_neighbors.values.items()
                }
                cfg.sampler_kwargs.num_neighbors = NumNeighbors(
                    new_values, old_num_neighbors.default)

        return cls.from_config(
            cfg,
            task=task,
            root_dir=root_dir,
            version=version,
            feature_store=feature_store,
            graph_store=graph_store,
            encoders=encoders,
            **kwargs,
        )

    def _validate_mask(self, stage: TrainingStage) -> bool:
        r"""Check whether the current training stage contains a valid mask."""
        mask_name = stage.add_stage_prefix('mask')
        train_table_name = self.feature_store.train_table_name

        if (stage is TrainingStage.TEST
                and not self.feature_store.has_train_table_test_mask):
            return False

        mask = self.feature_store[train_table_name, mask_name, :]

        if mask is None or not mask.any():
            msg = (f"No {stage.value} samples found. Check that your data "
                   f"split contains {stage.value} samples.")
            if stage in [TrainingStage.TRAIN, TrainingStage.VAL]:
                raise ValueError(msg)
            elif stage in [TrainingStage.TEST]:
                logger.warning(msg)
            return False

        return True

    @failed_trial_handler
    @trace()
    def fit(self) -> 'Trainer':
        if self._prediction_mode:
            fn_name = 'fit' if self.fit_mode == FitMode.fit else 'refit'
            raise RuntimeError(f"Calling `{fn_name}()` after `predict()` is "
                               f"not allowed")

        if (self.refit_cfg.refitted_full and self.fit_mode == FitMode.refit_full
            ) or (self.refit_cfg.refitted_trainval
                  and self.fit_mode == FitMode.refit_trainval):
            raise RuntimeError("The model is already refitted and cannot be "
                               "refitted again")

        with timing("trainer.fit.init"):
            self.init()
        with timing("trainer.fit.validate_train_mask"):
            self._validate_mask(TrainingStage.TRAIN)
        with timing("trainer.fit.lightning_fit"):
            from kumo_core.diskgraph.lightning_data import (
                suppress_iterable_len_warning,
            )
            with suppress_iterable_len_warning():
                self.pl_trainer.fit(self.pl_model, self.datamodule)

        with timing("trainer.fit.save_train_result"):
            results = {
                k: json_safe_float(v)  # `v` is a zero-dimensional tensor.
                for k, v in self.pl_trainer.logged_metrics.items()
            }
            save_json(results, _train_result_json_path(self.log_dir))

        if self.fit_mode == FitMode.fit:
            # `best_model_path` has the following format:
            # `my/path/epoch=10-step=50.ckpt`
            # Retrieve the best validation epoch from it:
            ckpt_path = self.pl_trainer.checkpoint_callback.best_model_path
            best_val_epoch = int(osp.basename(ckpt_path).split('-')[0][6:]) + 1
            self.refit_cfg.best_val_epoch = best_val_epoch
        elif self.fit_mode == FitMode.refit_trainval:
            self.refit_cfg.refitted_trainval = True
        elif self.fit_mode == FitMode.refit_full:
            self.refit_cfg.refitted_full = True

        # Update local trainer configuration and sync remotely:
        with timing("trainer.fit.save_local_config"):
            local_path = self.get_log_dir(original=True, local=True)
            save_json(self.config(), osp.join(local_path, 'config.json'))

        with timing("trainer.fit.save_remote_config"):
            remote_path = self.get_log_dir(original=True, local=False)
            save_json(self.config(), osp.join(remote_path, 'config.json'))

        if not fsutil.is_local(self.log_dir):  # Copy tensorboard directory.
            local_dir = osp.join(self.get_log_dir(local=True), 'tensorboard')
            remote_dir = osp.join(self.log_dir, 'tensorboard')
            logger.info(f"Copying {local_dir} to {remote_dir}")
            fsutil.copy(local_dir, remote_dir, overwrite=True)

        return self

    def get_train_results(self) -> dict[str, float]:
        """Example JSON:
            {
                'params/learning_rate': 0.0004267767071723938,
                'val_loss': 0.9530508518218994,
                'val_acc': 0.6365422606468201,
                'train_loss': 0.8904240131378174,
                'train_acc': 0.6897398233413696
            }
        """
        fname = _train_result_json_path(self.log_dir)
        if not fsutil.exists(fname):
            raise RuntimeError(
                f'Could not find train results at {fname}. Please call '
                f'`fit()` before getting train results')

        return {
            key: val if val is not None else float('NaN')
            for key, val in load_json(fname, dict[str, float | None]).items()
        }

    def set_explainers(
        self,
        stage: TrainingStage,
        explain_methods: list[ExplainMethod | str],
        root_dir: str | None = None,
    ):
        r"""Set the explainers to running with.

        Args:
            stage (TrainingStage): Stage for explanation.
            explain_methods (list): List of explain methods.
            root_dir (str, optional): If specified, will write all XAI
                results to specified path. This is used for XAI on BP.
                (default: :obj:`None`)
        """
        explain_methods = [
            ExplainMethod(explain_method) for explain_method in explain_methods
        ]
        if explain_methods and self.explain_runtime is None:
            raise RuntimeError("explain_runtime is required for explainability")

        methods: list[ExplainMethod] = []
        for explain_method in explain_methods:
            # Skip non node prediction explainers when explainer is subgraph
            # collector or gradient:
            if (explain_method not in {
                    ExplainMethod.SUBGRAPH_COLLECTOR, ExplainMethod.GRADIENT
            } and not isinstance(self.pl_model, NodeGNNLightningModule)):
                logger.warning(f"Explainability {explain_method} is not yet "
                               f"supported for non node-level prediction "
                               f"tasks")
                continue
            # Skip gradient explainer when lp module is unsupported:
            if (explain_method == ExplainMethod.GRADIENT
                    and self.module_kwargs is not None and
                    self.module_kwargs.module == LinkPredModuleType.TWO_TOWER):
                logger.warning(f"Gradient explainer is not yet supported"
                               f"for link pred {self.module_kwargs.module} "
                               f"module")
                continue
            # Skip gradient explainer if initialized already. This is currently
            # necessary as this method gets run again with reduced batch size
            # on CUDA OOM.
            if explain_method == ExplainMethod.GRADIENT:
                if any(
                        isinstance(explainer, GradientExplainer)
                        for explainer in self.pl_model.explainers):
                    logger.warning("Gradient explainer already exists")
                    continue
            # Skip subgraph collector if initialized already. This is currently
            # necessary as this method gets run again with reduced batch size
            # on CUDA OOM.
            if explain_method == ExplainMethod.SUBGRAPH_COLLECTOR:
                if any(
                        isinstance(explainer, SubgraphCollector)
                        for explainer in self.pl_model.explainers):
                    logger.warning("Subgraph collector already exists")
                    continue
            # Skip column analysis if initialized already. This is currently
            # necessary as this method gets run again with reduced batch size
            # on CUDA OOM.
            if explain_method == ExplainMethod.COLUMN_ANALYSIS:
                if any(
                        isinstance(explainer, ColumnAnalysis)
                        for explainer in self.pl_model.explainers):
                    logger.warning("Column analysis already exists")
                    continue
            methods.append(explain_method)

        task_type = self.task.task_type
        for explain_method in methods:
            if explain_method == ExplainMethod.GRADIENT:
                if task_type in GradientExplainer.supported_task_types:
                    explainer = GradientExplainer(
                        task_type=task_type,
                        root_dir=self.explain_dir(stage.value,
                                                  root_dir=root_dir),
                        formats=['csv'],
                    )
                    self.pl_model.explainers.append(explainer)
                else:
                    logger.warning(f"'GradientExplainer' not supported for "
                                   f"task type '{task_type.value}'")

            elif explain_method == ExplainMethod.COLUMN_ANALYSIS:
                if task_type in ColumnAnalysis.supported_task_types:
                    explainer = ColumnAnalysis(
                        task_type=task_type,
                        root_dir=self.column_analysis_dir(root_dir=root_dir),
                        num_cohorts=10,
                    )
                    self.pl_model.explainers.append(explainer)
                else:
                    logger.warning(f"'ColumnAnalysis' not supported for "
                                   f"task type '{task_type.value}'")

            elif explain_method == ExplainMethod.SUBGRAPH_COLLECTOR:
                if task_type in SubgraphCollector.supported_task_types:
                    formats = ['nested_json', 'output', 'rfm']
                    if task_type.is_link_pred:
                        formats.append('historical_items')
                    local_dir = self.get_log_dir(local=True)
                    explainer = SubgraphCollector(
                        task_type=task_type,
                        root_dir=osp.join(local_dir, 'subgraph_collector'),
                        formats=formats,
                        max_num_facts=self._explain_runtime.max_facts,
                        remote_sync_dir=self.subgraph_collector_dir(root_dir),
                        target_to_entity_paths=self.task.target_to_entity_paths,
                    )
                    self.pl_model.explainers.append(explainer)

    def _eval(
        self,
        stage: TrainingStage,
        return_embeddings: bool = False,
    ) -> EvalOutput | None:
        assert stage in {TrainingStage.VAL, TrainingStage.TEST}

        if self._prediction_mode:
            raise RuntimeError(f"Calling `{stage.value}()` after `predict()` "
                               f"is not allowed")

        with timing("trainer.eval.validate_mask", stage=stage.value):
            if not self._validate_mask(stage):
                return None

        if return_embeddings and self.task.task_type.is_link_pred:
            return_embeddings = False
            logger.warning("Embedding extraction is not supported "
                           "for link prediction")

        if return_embeddings:
            # In case there was there is an OOM, this entire method will be
            # restarted. Therefore, we need to reset the embedding hook before
            # registering a new one.
            if self.pl_model.has_embedding_hook():
                self.pl_model.reset_embedding_hook()
            self.pl_model.register_embedding_hook(stage)

        # TODO(akihironitta): Don't create prediction output on validation set
        # for all tasks. Tracked in #16766.
        eval_top_k: int | None = None
        if ((self.task.task_type.is_ranking and stage == TrainingStage.TEST)
                or self.task.task_type == TaskType.MULTILABEL_RANKING):
            if '@' in self.tune_metric:
                eval_top_k = int(self.tune_metric.split('@')[1])
                eval_top_k = min(eval_top_k, _MAX_EVAL_TOP_K)
            else:
                eval_top_k = _DEFAULT_EVAL_TOP_K
        elif self.task.task_type in {
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
        }:
            eval_top_k = min(self.num_targets, _MAX_EVAL_TOP_K)
        self.pl_model.eval_top_k = eval_top_k

        with timing("trainer.eval.lightning_eval", stage=stage.value):
            if stage == TrainingStage.VAL:
                metrics = self.pl_trainer.validate(
                    model=self.pl_model,
                    datamodule=self.datamodule,
                    ckpt_path=fsspec_mangle(self.best_ckpt),
                )[0]
            else:
                metrics = self.pl_trainer.test(
                    model=self.pl_model,
                    datamodule=self.datamodule,
                    ckpt_path=fsspec_mangle(self.best_ckpt),
                )[0]

        if eval_top_k is not None:  # Reset.
            self.pl_model.eval_top_k = None

        outputs, self.pl_model.outputs = self.pl_model.outputs, None

        with timing("trainer.eval.save_metrics", stage=stage.value):
            path = osp.join(self.log_dir, f'{stage.value}_result.csv')
            save_csv(metrics, path, overwrite=True)

        # Format output #######################################################

        with timing("trainer.eval.format_outputs", stage=stage.value):
            fkey_names = self.feature_store.train_fkey_names
            time_col = self.feature_store.get_time_col_name(
                self.feature_store.train_table_name)

        # TODO Revisit formatting logic to avoid wide to long conversion for
        # *all* task types.
        df: pd.DataFrame | None = None
        if not self.task.task_type.is_link_pred:
            pkey = outputs['pkey']
            pkey = pkey.reshape(1, -1) if pkey.ndim == 1 else pkey.transpose()
            df = pd.DataFrame({
                fkey_name: pkey[i]
                for i, fkey_name in enumerate(fkey_names)
            })

            if outputs.get('time') is not None:
                assert time_col is not None
                df[time_col] = outputs['time'].numpy().astype('datetime64[ns]')

            meta_df = df.copy(deep=False)

            target = self.target_encoder.decode(outputs['target'])
            df[self.encoders.target.col_name] = target

            # TODO Support top-k for other task types:
            if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                assert eval_top_k is not None
                df = create_multiclass_prediction_output(
                    df=df,
                    target_encoder=self.target_encoder,
                    prob=outputs['prob'],
                    top_k=eval_top_k,
                )
            elif self.task.task_type == TaskType.MULTILABEL_RANKING:
                assert eval_top_k is not None
                df = create_multilabel_ranking_prediction_output(
                    df=df,
                    target_encoder=self.target_encoder,
                    prob=outputs['prob'],
                    target=outputs.get('target'),
                    top_k=eval_top_k,
                )
            elif self.task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                # TODO(akihironitta): Don't create unnecessary target column
                df.drop(columns=[self.encoders.target.col_name], inplace=True)

                assert eval_top_k is not None
                df = create_multilabel_classification_prediction_output(
                    df=df,
                    target_encoder=self.target_encoder,
                    prob=outputs['prob'],
                    pred=outputs['pred'],
                    target=outputs.get('target'),
                    top_k=eval_top_k,
                )
            else:
                pred = outputs['pred']
                quantiles = quantile_names = None
                if isinstance(self.pl_model.loss, DistributionLoss):
                    assert self.task.task_type in {
                        TaskType.REGRESSION,
                        TaskType.FORECASTING,
                    }
                    pred = self.pl_model.loss.to_prediction(pred)
                    quantiles = outputs.get('prob')
                    quantile_names = self.pl_model.loss.distribution.quantiles
                    quantile_names = [str(n) for n in quantile_names]
                elif isinstance(self.pl_model.loss, MultiQuantileLoss):
                    assert self.task.task_type in {
                        TaskType.REGRESSION,
                        TaskType.FORECASTING,
                    }
                    quantiles = outputs.get('prob')
                    quantile_names = [
                        str(n) for n in self.pl_model.loss.quantiles
                    ]

                out_timestamps = None
                if self.task.task_type == TaskType.FORECASTING:
                    out_timestamps = self.pl_model.get_future_unix(
                        outputs['time']).flatten()

                df = format_prediction(
                    df=df,
                    target_col_name=self.encoders.target.col_name,
                    target_encoder=self.target_encoder,
                    task_type=self.task.task_type,
                    pred=pred,
                    prob=outputs.get('prob'),
                    quantiles=quantiles,
                    quantile_names=quantile_names,
                    top_k=eval_top_k,
                    time_col_name=time_col,
                    out_timestamps=out_timestamps,
                    array_targets=allow_array_targets(),
                )

        elif self.task.task_type.is_link_pred and stage == TrainingStage.TEST:
            rhs_name = self.feature_store.entity_table_names[-1]
            with self.feature_store.safe_batch_mode(batch_size=50_000):
                rhs_index = self.feature_store[rhs_name, 'pkey', :]
            rhs_index_map = pd.DataFrame(index=rhs_index)
            rhs_index_map[NODE_INDEX_COLUMN] = range(len(rhs_index_map))

            try:
                df = create_lp_prediction_output(
                    outs=outputs,
                    lhs_fkey_name=fkey_names[0],
                    rhs_index_map=rhs_index_map,
                    time=outputs.get('time'),
                    time_col=time_col,
                )
            except Exception as e:
                logger.warn(
                    f"create_lp_prediction_output failure with exception {e}")
                df = pd.DataFrame()

        embedding: pd.DataFrame | None = None
        if return_embeddings:
            embeddings = self.pl_model.get_embeddings()
            assert len(embeddings) == 1
            embedding = list(embeddings.values())[0].numpy()
            embedding = format_embeddings(embedding, meta_df)

        return EvalOutput(metrics, df, embedding)

    @trace()
    def validate(
        self,
        *,
        return_embeddings: bool = False,
    ) -> EvalOutput | None:

        if self.refit_cfg.refitted_trainval or self.refit_cfg.refitted_full:
            # Once refit is called, limit_val_batches = 0 is activated in
            # self.pl_trainer. In fit mode, we do not reconstruct pl_trainer if
            # it is already fully-specified. Therefore, we cannot not call
            # validate() after refit.
            raise RuntimeError("Cannot call validate() after refit")

        self.fit_mode = FitMode.fit
        self.init()

        return self._eval(
            stage=TrainingStage.VAL,
            return_embeddings=return_embeddings,
        )

    def _check_readiness_for_prediction(self, fit_mode: FitMode):
        r"""Whether we can load a model trained with `fit_mode` for prediction
        or not."""
        is_ready = False
        if fit_mode == FitMode.fit:
            is_ready = self.refit_cfg.fitted
        elif fit_mode == FitMode.refit_trainval:
            is_ready = self.refit_cfg.refitted_trainval
        elif fit_mode == FitMode.refit_full:
            is_ready = self.refit_cfg.refitted_full
        if not is_ready:
            raise ValueError(
                f"explain, test or predict(fit_mode={fit_mode}) can be "
                f"only called after {fit_mode.value}() is called.")

    @trace()
    @handle_gpu_oom
    def test(
        self,
        fit_mode_to_use: FitMode = FitMode.fit,
        *,
        return_embeddings: bool = False,
    ) -> EvalOutput | None:
        r"""Get test model performance based on checkpoint trained with the
        given `fit_mode_to_use`"""
        if fit_mode_to_use == FitMode.refit_full:
            raise ValueError(
                f"test(fit_mode={fit_mode_to_use}) is invalid since the model "
                f"is already trained on the test split.")
        self._check_readiness_for_prediction(fit_mode_to_use)
        self.fit_mode = fit_mode_to_use

        self.init()

        return self._eval(
            stage=TrainingStage.TEST,
            return_embeddings=return_embeddings,
        )

    @trace()
    @handle_gpu_oom
    def explain(
        self,
        explain_methods: list[ExplainMethod | str],
        stage: TrainingStage | str = TrainingStage.TEST,
        fit_mode_to_use: FitMode | None = None,
        root_dir: str | None = None,
        top_k: int | None = None,
        valid_rhs_node_idx: Tensor = None,
    ) -> None:
        r"""Runs a set of explainability algorithms outside the
        evaluation/prediction loop for a subset of pre-selected input nodes.

        Args:
            explain_methods (list): List of explain methods in
                :class:`ExplainMethod` or string.
            stage (TrainingStage or str): Training stage to run the XAI.
                (default: :obj:`TrainingStage.TEST`)
            fit_mode_to_use (FitMode, optional): Fit mode to use for XAI.
                By default will use the trainer trained with more data for
                XAI. (default: :obj:`None`)
            root_dir (str, optional): If specified, will write all XAI
                results to specified path. This is used for XAI on BP.
                (default: :obj:`None`)
            top_k (int, optional): Top k for link prediction XAI.
                If not specified tune metric :obj:`top_k` will be used.
                If cannot get tune metric :obj:`top_k`, :obj:`top_k` will
                be set to 10 for either :obj:`eval_top_k` or
                :obj:`pred_top_k`. (default: :obj:`None`)
            valid_rhs_node_idx (Tensor, optional): Valid RHS node indices for
                link prediction, passed from :obj:`kumo-pred`. Used for target
                BP filtering of RHS nodes. If :obj:`None`, we will make
                prediction over the entire RHS nodes.
        """
        if self.explain_runtime is None:
            raise RuntimeError("explain_runtime is required for explainability")
        stage = TrainingStage(stage)

        # Set default FitMode for explainers if not specified
        trainval_stages = {TrainingStage.TEST, TrainingStage.PRED}
        if fit_mode_to_use is None:
            if self.refit_cfg.refitted_trainval and stage in trainval_stages:
                fit_mode_to_use = FitMode.refit_trainval
            elif self.refit_cfg.refitted_full and stage == TrainingStage.PRED:
                fit_mode_to_use = FitMode.refit_full
            else:
                fit_mode_to_use = FitMode.fit

        self._check_readiness_for_prediction(fit_mode_to_use)
        self.fit_mode = fit_mode_to_use
        existing_explainers = self.pl_model.explainers
        if not self._fully_initialized:
            self.init()
        self.pl_model.explainers = existing_explainers

        explain_methods = [
            ExplainMethod(explain_method)
            for explain_method in explain_methods or []
        ]
        column_explain_methods = copy.deepcopy(explain_methods)
        # Remove prediction-level XAI such as subgraph collector during
        # eval column level explanation, which will be handled later:
        if ExplainMethod.SUBGRAPH_COLLECTOR in column_explain_methods:
            column_explain_methods.remove(ExplainMethod.SUBGRAPH_COLLECTOR)
        self.set_explainers(stage, column_explain_methods, root_dir=root_dir)

        # Store current eval and pred XAI for link pred task
        prev_eval_top_k: int | None = None
        prev_pred_top_k: int | None = None
        if self.task.task_type.is_link_pred:
            prev_eval_top_k = self.pl_model.eval_top_k
            prev_pred_top_k = self.pl_model.pred_top_k
            if stage in {TrainingStage.VAL, TrainingStage.TEST}:
                # If top_k is not specified, tune metric top_k will be used
                if top_k is None:
                    if '@' in self.tune_metric:
                        top_k = int(self.tune_metric.split('@')[1])
                        top_k = min(top_k, _MAX_EVAL_TOP_K)
                    else:
                        top_k = 10
                self.pl_model.eval_top_k = top_k
            else:
                top_k = top_k if top_k is not None else 10
                self.pl_model.pred_top_k = top_k

        # Manually set limit_xxx_batches to max_num_batches for XAI
        # which avoids running all evaluation / prediction set
        orig_eval_batch_size = self.datamodule.eval_loader_kwargs['batch_size']

        # To prevent GPU OOM, divide the batch size by four
        eval_batch_size = max(orig_eval_batch_size // 4, 1)
        self.datamodule.eval_loader_kwargs['batch_size'] = eval_batch_size

        max_num_batches = self._explain_runtime.max_batches
        prev_limit_batches: float | int | None = None
        if stage == TrainingStage.VAL:
            eval_fn = self.pl_trainer.validate
            prev_limit_batches = self.pl_trainer.limit_val_batches
            self.pl_trainer.limit_val_batches = max_num_batches
        elif stage == TrainingStage.TEST:
            eval_fn = self.pl_trainer.test
            prev_limit_batches = self.pl_trainer.limit_test_batches
            self.pl_trainer.limit_test_batches = max_num_batches
        elif stage == TrainingStage.PRED:
            eval_fn = self.pl_trainer.predict
            prev_limit_batches = self.pl_trainer.limit_predict_batches
            # Set max_num_batches to at least explaining on
            # the number of nodes for batch prediction XAI
            max_num_batches = max(
                max_num_batches,
                math.ceil(self._explain_runtime.prediction_selection_nodes /
                          eval_batch_size))
            self.pl_trainer.limit_predict_batches = max_num_batches
        else:
            raise ValueError(f'Unsupported stage {stage}')
        assert prev_limit_batches is not None

        explain_capture_datamodule = None
        has_explain_capture = hasattr(self.datamodule,
                                      'begin_explain_selection_capture')
        if has_explain_capture:
            explain_capture_datamodule = self.datamodule
            explain_capture_datamodule.begin_explain_selection_capture(
                stage.value)

        if valid_rhs_node_idx is not None:
            self.pl_model.valid_rhs_node_idx = valid_rhs_node_idx
        from kumo_core.diskgraph.lightning_data import (
            suppress_iterable_len_warning,
        )
        try:
            with suppress_iterable_len_warning():
                step_outputs = eval_fn(
                    model=self.pl_model,
                    datamodule=self.datamodule,
                    ckpt_path=fsspec_mangle(self.best_ckpt),
                )
        except Exception:
            if explain_capture_datamodule is not None:
                explain_capture_datamodule.clear_explain_selection_capture()
            raise

        # Empty the explainers list
        self.pl_model.explainers = []

        task_not_supported = (self.task.task_type
                              not in SubgraphCollector.supported_task_types)
        module_not_supported = (self.task.task_type.is_link_pred
                                and self.module_kwargs.module not in {
                                    LinkPredModuleType.HYBRID,
                                    LinkPredModuleType.ID,
                                    LinkPredModuleType.SHALLOW_RHS,
                                })
        no_eval_output = (stage != TrainingStage.PRED) and getattr(
            self.pl_model, "outputs") is None

        if (ExplainMethod.SUBGRAPH_COLLECTOR not in explain_methods
                or task_not_supported or module_not_supported
                or no_eval_output):
            if explain_capture_datamodule is not None:
                explain_capture_datamodule.clear_explain_selection_capture()
            self.reset_explain(stage, prev_limit_batches, prev_eval_top_k,
                               prev_pred_top_k, orig_eval_batch_size)
            return

        try:
            input_nodes, input_id = self._explain_runtime.get_explain_node_inputs(
                self.datamodule, stage)
        except Exception:
            if explain_capture_datamodule is not None:
                explain_capture_datamodule.clear_explain_selection_capture()
            raise
        explain_capture_datamodule = None
        assert input_nodes is not None

        # Handle outputs from eval or predict
        if stage == TrainingStage.PRED:
            outputs = {}
            for key, value in step_outputs[0].items():
                if value is None:
                    outputs[key] = value
                elif isinstance(value, Tensor):
                    # Concatenate step outputs to single tensors:
                    outputs[key] = torch.cat(
                        [out[key] for out in step_outputs],
                        dim=0,
                    )
                elif isinstance(value, np.ndarray):
                    outputs[key] = np.concatenate(
                        [out[key] for out in step_outputs], axis=0)
        else:
            outputs, self.pl_model.outputs = self.pl_model.outputs, None
        del step_outputs

        # Outputs have max_num_batches * eval_batch_size entities at max
        input_nodes = input_nodes[:max_num_batches * eval_batch_size]
        if input_id is not None:
            input_id = input_id[:max_num_batches * eval_batch_size]
        seed_nodes, strategy_mappings = self._explain_runtime.select_input_nodes(
            self.task.task_type,
            self.feature_store,
            input_nodes,
            input_id,
            outputs,
            stage,
        )
        mapping_dir = osp.join(
            self.subgraph_collector_dir(root_dir),
            stage.value,
        )
        fsutil.makedirs(mapping_dir, exist_ok=True)
        save_json(
            strategy_mappings,
            osp.join(mapping_dir, self._explain_runtime.mapping_file_name),
        )
        explain_methods = [ExplainMethod.SUBGRAPH_COLLECTOR]

        self.set_explainers(stage, explain_methods, root_dir=root_dir)
        # Exit if subgraph collector is not supported
        if len(self.pl_model.explainers) == 0:
            self.reset_explain(stage, prev_limit_batches, prev_eval_top_k,
                               prev_pred_top_k, orig_eval_batch_size)
            return

        # Avoid computing metrics during evaluation:
        self.pl_model.disable_eval_metrics = True
        if stage != TrainingStage.PRED:
            loader = self._explain_runtime.get_eval_dataloader(
                self.datamodule,
                stage,
                seed_nodes,
            )
        else:
            # Directly use the predict dataloader instead of creating the
            # dataloader as there is weird error when creating a new one
            loader = self.datamodule.predict_dataloader()
            # Since we are using the original predict dataloader, need to
            # set limit_predict_batches to just include around first
            # self.explain_runtime.prediction_selection_nodes nodes
            self.pl_trainer.limit_predict_batches = math.ceil(
                self._explain_runtime.prediction_selection_nodes /
                eval_batch_size)

        if valid_rhs_node_idx is not None:
            self.pl_model.valid_rhs_node_idx = valid_rhs_node_idx
        with suppress_iterable_len_warning():
            eval_fn(
                model=self.pl_model,
                dataloaders=loader,
                ckpt_path=fsspec_mangle(self.best_ckpt),
            )

        self.pl_model.disable_eval_metrics = False
        self.reset_explain(stage, prev_limit_batches, prev_eval_top_k,
                           prev_pred_top_k, orig_eval_batch_size)

    def reset_explain(
        self,
        stage: TrainingStage,
        prev_limit_batches: float | int | None = None,
        prev_eval_top_k: int | None = None,
        prev_pred_top_k: int | None = None,
        orig_eval_batch_size: int | None = None,
    ):
        # Empty the explainers list
        self.pl_model.explainers = []
        self.datamodule.eval_loader_kwargs['batch_size'] = orig_eval_batch_size

        # Reset limit_xxx_batches back
        if stage == TrainingStage.VAL:
            self.pl_trainer.limit_val_batches = prev_limit_batches
            if self.task.task_type.is_link_pred:
                self.pl_trainer.eval_top_k = prev_eval_top_k
        elif stage == TrainingStage.TEST:
            self.pl_trainer.limit_test_batches = prev_limit_batches
            if self.task.task_type.is_link_pred:
                self.pl_trainer.eval_top_k = prev_eval_top_k
        elif stage == TrainingStage.PRED:
            self.pl_trainer.limit_predict_batches = prev_limit_batches
            if self.task.task_type.is_link_pred:
                self.pl_trainer.pred_top_k = prev_pred_top_k
        else:
            raise ValueError(f'Unsupported stage {stage}')

    @handle_refit_gpu_oom
    @trace()
    def refit(self, fit_mode: FitMode):
        r"""Refit the model. Set the number of epochs to the best validation
        epoch in `fit()` without early stopping.

        Implementation details:
        Each `Trainer` has the original log directory (like `version_xx/`)
        where `config.json` as well as tensorboard/checkpoints/results for
        `fit()` stage are saved. Then, when we call
        `trainer.refit(fit_mode = ...)`, we spin out a new sibling log
        directory (like `version_xx_refit_trainval/`), where tensorboard/model
        checkpoints, performance results for `refit` are all saved.
        Once `refit` is done, we update `version_xx/config.json` to save the
        the fact that refit is done. So `version_xx/config.json` is always the
        source of truth for the current state of `Trainer`, regardless of
        which fit mode the `Trainer` is in.

        At evaluation/prediction time, we can switch between which model
        checkpoint to load by simply switching fit mode, which will switch
        the active `self.log_dir` in the backend. Note that
        `self.original_log_dir` always points to `version_xx/`, regardless of
        which fit mode we are in.

        Args:
            fit_mode (FitMode): Which fit_mode to use. Either `refit_trainval`
                or `refit_full`. `refit_trainval` will refit on train+val set.
                `refit_full` will refit on the full train+val+test set.
        """
        if fit_mode == FitMode.fit:
            raise ValueError(f'{fit_mode} is invalid for refit.')
        if not self.refit_cfg.fitted:
            raise RuntimeError('fit() needs to be called first.')
        if self.checkpoint_path:
            logger.info(
                f'Refit the model from checkpoint {self.checkpoint_path} '
                f'with {fit_mode} for {self.refit_cfg.best_val_epoch} epochs.')
        else:
            logger.info(f'Refit the model with {fit_mode} from scratch for '
                        f'{self.refit_cfg.best_val_epoch} epochs.')
        self.fit_mode = fit_mode
        self.fit()

    # Batch prediction utilities ##############################################

    def _preprocess_top_k(self, top_k: int | None):

        match self.task.task_type:
            case TaskType.TEMPORAL_LINK_PREDICTION:
                if (self.module_kwargs.module == LinkPredModuleType.ID
                        and top_k is None):
                    # ID-GNN requires `top_k`:
                    raise ValueError(
                        "'top_k' must be specified when using the ID-GNN "
                        "model for link prediction (got None).")
            case TaskType.MULTILABEL_RANKING:
                # TODO(jinu): This is a hack. Until we update the
                # UI to allow specifying topk for multilabel ranking.
                num_categories = self.num_targets
                if top_k is None or top_k > num_categories:
                    top_k = num_categories
            case TaskType.MULTICLASS_CLASSIFICATION:
                if top_k is None:
                    top_k = _MAX_EVAL_TOP_K
                top_k = min(self.num_targets, top_k)

        return top_k

    # Batch prediction ########################################################

    def published_fit_mode(self) -> FitMode | None:
        r"""The fit mode predictions use by default: the model trained on the
        most data, or `None` before fitting."""
        if self.refit_cfg.refitted_full:
            return FitMode.refit_full
        if self.refit_cfg.refitted_trainval:
            return FitMode.refit_trainval
        if self.refit_cfg.fitted:
            return FitMode.fit
        return None

    def _set_fit_mode_to_use(self, fit_mode_to_use: FitMode | None = None):
        if fit_mode_to_use is None:
            fit_mode_to_use = self.published_fit_mode()
            if fit_mode_to_use is None:
                raise RuntimeError("Trainer is not fitted yet.")
        else:
            self._check_readiness_for_prediction(fit_mode_to_use)
        self.fit_mode = fit_mode_to_use

    def predict_setup(self, fit_mode_to_use: FitMode | None = None) -> None:
        r"""Sets up for batch prediction.

        Args:
            fit_mode_to_use (FitMode, optional): We will use the model fitted
                with the specified `fit_mode`. If `None`, we will use the
                trained model fitted with the most data.
        """
        self._prediction_mode = True
        self._set_fit_mode_to_use(fit_mode_to_use=fit_mode_to_use)
        # Prediction table never has a target attribute:
        self.feature_store._target_attr = None

        if not self._fully_initialized:
            self.init()

    def predict_setup_dataloader(
        self,
        pos: int | None,
        length: int | None,
        top_k: int | None,
        valid_rhs_node_idx: Tensor | None,
    ) -> tuple[Tensor | None, bool | None, pd.DataFrame | None, Tensor]:
        r"""Sets a new dataloader to datamodule in-place for batch prediction.

        Args:
            pos: The position of the first row to predict.
            length: The length of the rows to predict.
            top_k: The top k to predict.
            valid_rhs_node_idx: The valid RHS node indices to predict.
        """
        entity_table_names = self.feature_store.entity_table_names
        train_table_name = self.feature_store.train_table_name
        fkeys = self.feature_store[train_table_name, 'fkey', :].t()
        pred_time: Tensor | None = None
        if self.feature_store.has_train_table_time_col:
            pred_time = self.feature_store[train_table_name, 'time', :]

        if self.task.task_type == TaskType.STATIC_LINK_PREDICTION:
            pred_time = torch.ones(fkeys.size(1),
                                   dtype=torch.long) * MAX_PSEUDO_TIME

        if pos is not None:
            assert length is not None
            fkeys = fkeys[:, pos:pos + length]
            if pred_time is not None:
                pred_time = pred_time[pos:pos + length]

        datamodule = self.datamodule
        rhs_index_map: pd.DataFrame | None = None
        use_rhs_lookup: bool | None = None
        if self.task.task_type.is_link_pred:
            # NOTE Due to the current format of the link prediction BP table,
            # either LHS or RHS nodes are padded. We need to de-duplicate here
            # appropriately. Note that `unique()` may not preserve ordering, so
            # we need to gracefully handle this downstream.
            lhs_node_idx = fkeys[0].unique()

            rhs_name = entity_table_names[-1]
            num_rhs_nodes = self.feature_store.get_tensor_size(rhs_name)[0]
            # We require the full RHS index map in two scenarios:
            # 1) We use lookup RHS embeddings
            # 2) We return predictions
            use_rhs_lookup = (isinstance(self.pl_model,
                                         ShallowRHSLinkGNNLightningModule)
                              and self.pl_model.rhs_emb_mode.use_rhs_lookup)
            return_predictions = (top_k is not None and top_k > 0)
            if use_rhs_lookup or return_predictions:
                with self.feature_store.safe_batch_mode(batch_size=50_000):
                    rhs_index = self.feature_store[rhs_name, 'pkey', :]
                rhs_index_map = pd.DataFrame(index=rhs_index)
                rhs_index_map[NODE_INDEX_COLUMN] = range(len(rhs_index_map))

            if self.module_kwargs.module.use_lhs_node_loader:
                if pred_time is not None:
                    if not torch.eq(pred_time[0], pred_time).all():
                        raise ValueError("Link prediction requires predicting "
                                         "a single timestamp")

                def predict_dataloader():
                    return datamodule.dataloader(
                        input_nodes=lhs_node_idx,
                        input_time=pred_time,
                        shuffle=False,
                        **datamodule.eval_loader_kwargs,
                    )

                datamodule.predict_dataloader = predict_dataloader
            else:

                def lhs_dataloader_fn():
                    return datamodule.node_dataloader(
                        input_nodes=(entity_table_names[0], lhs_node_idx),
                        input_time=pred_time,
                    )

                datamodule.predict_dataloader = lhs_dataloader_fn

                def rhs_dataloader_fn():
                    return datamodule.rhs_dataloader(
                        input_nodes=valid_rhs_node_idx,
                        input_time=pred_time,
                    )

                datamodule.rhs_predict_dataloader = rhs_dataloader_fn

            # Set rhs node indices to make BP. Note that we need to
            # explicitly pass it here since :obj:`fkeys` inside this function
            # are partitioned and hence :obj:`fkeys[:,1].unique()` does not
            # give us the entire RHS node indices.
            if valid_rhs_node_idx is not None:
                assert len(valid_rhs_node_idx) <= num_rhs_nodes
                assert valid_rhs_node_idx.min() >= 0
                assert valid_rhs_node_idx.max() < num_rhs_nodes
                assert torch.isin(fkeys[1], valid_rhs_node_idx).all()

                # Only set valid RHS nodes if it is actually a subset:
                if len(valid_rhs_node_idx) < num_rhs_nodes:
                    self.pl_model.valid_rhs_node_idx = valid_rhs_node_idx
                else:
                    self.pl_model.valid_rhs_node_idx = None
            else:
                self.pl_model.valid_rhs_node_idx = None

        elif self.task.task_type == TaskType.FORECASTING:
            datamodule.predict_dataloader = lambda: datamodule.dataloader(
                input_nodes=fkeys.view(-1),
                input_time=pred_time,
                node_sampler=datamodule.eval_graph_sampler,
                shuffle=False,
                **datamodule.eval_loader_kwargs,
            )

        elif len(entity_table_names) == 1:
            input_pred = (entity_table_names[0], fkeys.view(-1))
            datamodule.predict_dataloader = lambda: datamodule.dataloader(
                input_nodes=input_pred,
                input_time=pred_time,
                node_sampler=datamodule.eval_graph_sampler,
                shuffle=False,
                **datamodule.eval_loader_kwargs,
            )

        else:
            assert len(entity_table_names) > 1
            input_fkeys = [
                (entity_table_name, fkey)
                for entity_table_name, fkey, in zip(entity_table_names, fkeys)
            ]
            datamodule.predict_dataloader = lambda: datamodule.dataloader(
                input_fkeys=input_fkeys,
                input_time=pred_time,
                shuffle=False,
                **datamodule.eval_loader_kwargs,
            )

        return pred_time, use_rhs_lookup, rhs_index_map, fkeys

    @handle_gpu_oom
    def predict_v2(
        self,
        top_k: int | None = None,
        valid_rhs_node_idx: Tensor = None,
        progress_log_dir: str = None,
        *,
        return_embeddings: bool = False,
        job_start_time: datetime | None = None,
        ignore_time_col: bool = False,
        pred_time_col_name: str | None = None,
        pos: int | None = None,
        length: int | None = None,
        progress_log_id: int = 0,
    ) -> PredOutput:
        r"""Predicts the output given a prediction table. It supports
        predicting a subset of the prediction table from
        `pos` to `pos` + `length`. To call this function, `predict_setup` must
        be called to materialize the prediction table edges.

        Args:
            top_k (int, optional): If not :obj:`None`, only the :obj:`top_k`
                classes/links with the highest prediction probabilities are
                returned per entity. :obj:`top_k` is only valid for ranking or
                multi-class/multi-label classification tasks. For temporal link
                prediction :obj:`top_k` is required. (default :obj:`None`)
            valid_rhs_node_idx (Tensor, optional): Valid RHS node indices for
                link prediction, passed from :obj:`kumo-pred`. Used for target
                BP filtering of RHS nodes. If :obj:`None`, we will make
                prediction over the entire RHS nodes.
            job_start_time (datetime, optional): If provided, the prediction
                table will contain a new column with name 'JOB_TIMESTAMP' and
                the value will be the provided `job_start_time`.
                NOTE: Applicable only for predictions, not embeddings.
            ignore_time_col (bool, optional): If :obj:`True`, the time column
                will be removed from the prediction output.
                NOTE: Applicable only for predictions, not embeddings.
                (default :obj:`False`)
            pred_time_col_name (str, optional): If provided, the time column
                will be renamed to the provided name in the prediction output.
                NOTE: Applicable only for predictions, not embeddings.
                (default :obj:`None`)
            pos (int, optional): If provided, `length` must be provided as
                will. The prediction will predict the sub data
                [pos: pos + length].
            length (int, optional): It must be provided with `pos`. The
                prediction will predict the sub data [pos: pos + length].
            progress_log_id (int): The prediction progress log id.
        """
        top_k = self._preprocess_top_k(top_k)
        train_table_name = self.feature_store.train_table_name
        time_col = self.feature_store.get_time_col_name(train_table_name)
        fkey_names = self.feature_store.train_fkey_names
        entity_table_names = self.feature_store.entity_table_names
        datamodule = self.datamodule

        # Register a new prediction dataloader ================================

        pred_time, use_rhs_lookup, rhs_index_map, fkeys = self.predict_setup_dataloader(
            pos=pos,
            length=length,
            top_k=top_k,
            valid_rhs_node_idx=valid_rhs_node_idx,
        )

        if self.task.task_type.is_link_pred:
            if return_embeddings:
                if self.module_kwargs.module == LinkPredModuleType.ID:
                    # TODO (matthias) Enable embedding code path for ID-GNN.
                    logger.warning(
                        "Disabling embeddings for ID-GNN: embedding output "
                        "is not supported for the ID link-prediction module.")
                    return_embeddings = False
                if self.module_kwargs.module == LinkPredModuleType.HYBRID:
                    logger.warning(
                        "Embeddings produced by Hybrid GNN do not perfectly "
                        "align with the predicted ranking.")
            else:
                if self.module_kwargs.module == LinkPredModuleType.TWO_TOWER:
                    logger.info("Enabling embeddings for link two-tower GNN")
                    return_embeddings = True

        # Run batch prediction ================================================

        if return_embeddings:
            # In case there was there is an OOM, this entire method will be
            # restarted. Therefore, we need to reset the embedding hook before
            # registering a new one.
            if self.pl_model.has_embedding_hook():
                self.pl_model.reset_embedding_hook()
            self.pl_model.register_embedding_hook(TrainingStage.PRED)

        callbacks = []
        if progress_log_dir is not None:
            progress_callback = PredictionProgressLogger(
                progress_log_dir, progress_log_id,
                total_iterations=len(datamodule.predict_dataloader()))
            callbacks.append(progress_callback)

        if self.task.task_type.is_ranking:
            self.pl_model.pred_top_k = top_k

        # Reorder the shallow embeddings in either shallow RHS or forecasting
        # modules by the new index map, since it may be different from the
        # index map used at training time:
        if self.task.task_type.is_link_pred and use_rhs_lookup:
            self.pl_model.set_rhs_index_map(rhs_index_map)

        if (isinstance(self.pl_model, ForecastingGNNLightningModule)
                and not self.head.handle_new_entities):
            entity_table_name = self.feature_store.entity_table_names[0]
            with self.feature_store.safe_batch_mode(batch_size=50_000):
                entity_index = self.feature_store[entity_table_name, 'pkey', :]
            index_map = pd.DataFrame(index=entity_index)
            index_map[NODE_INDEX_COLUMN] = range(len(index_map))
            self.pl_model.set_index_map(index_map)

        with temporary_callbacks(self.pl_trainer, callbacks):
            step_outputs = self.pl_trainer.predict(
                model=self.pl_model, datamodule=datamodule,
                ckpt_path=(fsspec_mangle(self.best_ckpt) if
                           not self._prediction_model_weight_loaded else None))
            self._prediction_model_weight_loaded = True
        self.pl_trainer.predict_loop.setup_data()

        # Post-processing =====================================================

        outs = {}
        for key, value in step_outputs[0].items():
            if value is None:
                continue
            elif isinstance(value, Tensor):
                outs[key] = torch.cat([out[key] for out in step_outputs], dim=0)
            elif isinstance(value, np.ndarray):
                outs[key] = np.concatenate([out[key] for out in step_outputs],
                                           axis=0)
            else:
                raise NotImplementedError

        del step_outputs

        if ignore_time_col:
            with_time_col = False
        else:
            with_time_col = (pred_time is not None and self.task.task_type
                             != TaskType.STATIC_LINK_PREDICTION)

        if self.task.task_type.is_link_pred:
            embeddings_df: pd.DataFrame | None = None
            predictions_df: pd.DataFrame = pd.DataFrame()

            embeddings: Tensor | None = None
            if return_embeddings:
                embeddings = self.pl_model.get_embeddings()
                lhs_emb = embeddings[self.pl_model.lhs_emb_name]
                rhs_emb = embeddings[self.pl_model.rhs_emb_name]

                # NOTE LHS embeddings already respect partitioning from
                # `kumo-pred`. However, RHS embeddings hold the full set of
                # embeddings based on `valid_rhs_node_idx` for every partition.
                # To (mostly) avoid duplicates in embedding outputs across
                # partitions, we can instead rely on `fkeys[1].unique` when
                # returning embeddings.
                # As such, we mask out any embeddings that are not valid in the
                # given partition, but without modifying their ordering.
                if valid_rhs_node_idx is None:
                    rhs_nodes = fkeys[1].unique(sorted=True)
                    rhs_emb = rhs_emb[rhs_nodes]
                else:
                    mask = torch.isin(valid_rhs_node_idx, fkeys[1].unique())
                    rhs_emb = rhs_emb[mask]
                    rhs_nodes = valid_rhs_node_idx[mask]

                rhs_name = entity_table_names[-1]
                with self.feature_store.safe_batch_mode(batch_size=50_000):
                    rhs_pkey = self.feature_store[rhs_name, 'pkey', rhs_nodes]

                embeddings_df = create_lp_embedding_output(
                    lhs_emb=lhs_emb,
                    rhs_emb=rhs_emb,
                    entity_table_names=entity_table_names,
                    fkey_names=fkey_names,
                    lhs_pkey=outs['pkey'],
                    rhs_pkey=rhs_pkey,
                    time=pred_time if with_time_col else None,
                    time_col=pred_time_col_name or time_col
                    if with_time_col else None,
                )

            return_predictions = (top_k is not None and top_k > 0)
            if return_predictions:
                predictions_df = create_lp_prediction_output(
                    outs=outs,
                    lhs_fkey_name=fkey_names[0],
                    rhs_index_map=rhs_index_map,
                    time=pred_time if with_time_col else None,
                    time_col=pred_time_col_name or time_col
                    if with_time_col else None,
                )

                if job_start_time is not None:
                    start_time = pd.Timestamp(job_start_time).tz_localize(None)
                    predictions_df['JOB_TIMESTAMP'] = start_time

            return PredOutput(predictions_df, embeddings_df)

        # Convert/decode predictions into a `pandas.DataFrame`:

        pred = outs.get('pred')

        quantiles = None
        quantile_names = None
        if isinstance(self.pl_model.loss, DistributionLoss):
            pred = self.pl_model.loss.to_prediction(pred)
            quantiles = outs.get('prob')
            quantile_names = self.pl_model.loss.distribution.quantiles
            quantile_names = [str(n) for n in quantile_names]
        elif isinstance(self.pl_model.loss, MultiQuantileLoss):
            quantiles = outs.get('prob')
            quantile_names = [str(n) for n in self.pl_model.loss.quantiles]

        pkey = outs['pkey']
        pkey = pkey.reshape(1, -1) if pkey.ndim == 1 else pkey.transpose()
        df = pd.DataFrame({
            fkey_name: pkey[i]
            for i, fkey_name in enumerate(fkey_names)
        })

        if pred_time is not None and not ignore_time_col:
            ser = pred_time.numpy().astype('datetime64[ns]')
            if pred_time_col_name is not None:
                df[pred_time_col_name] = ser
            else:
                assert time_col is not None
                df[time_col] = ser

        if job_start_time is not None:
            start_time = pd.Timestamp(job_start_time).tz_localize(None)
            df['JOB_TIMESTAMP'] = start_time

        main_df = df.copy()

        out_timestamps = None
        if self.task.task_type == TaskType.FORECASTING:
            out_timestamps = self.pl_model.get_future_unix(pred_time).flatten()

        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            assert top_k is not None
            df = create_multiclass_prediction_output(
                df=df,
                target_encoder=self.target_encoder,
                prob=outs['prob'],
                top_k=top_k,
            )
        elif self.task.task_type == TaskType.MULTILABEL_RANKING:
            df = create_multilabel_ranking_prediction_output(
                df=df,
                target_encoder=self.target_encoder,
                prob=outs['prob'],
                target=None,
                top_k=top_k,
            )
        elif self.task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            df = create_multilabel_classification_prediction_output(
                df=df,
                target_encoder=self.target_encoder,
                prob=outs['prob'],
                pred=outs['pred'],
                target=None,
                top_k=top_k or self.target_encoder.num_categories,
            )
        else:
            df = format_prediction(
                df=df,
                target_col_name='TARGET',
                target_encoder=self.target_encoder,
                task_type=self.task.task_type,
                pred=pred,
                prob=outs.get('prob'),
                quantiles=quantiles,
                quantile_names=quantile_names,
                top_k=top_k,
                time_col_name=pred_time_col_name or time_col,
                out_timestamps=out_timestamps,
                array_targets=allow_array_targets(),
            )

        pred_output = PredOutput(df)
        if return_embeddings:
            embeddings = self.pl_model.get_embeddings()
            assert len(embeddings) == 1
            embedding = list(embeddings.values())[0].numpy()
            embedding = format_embeddings(embedding, main_df)
            pred_output = PredOutput(df, embedding)

        logger.info(f'Save batch prediction scores to {self.log_dir}.')
        pred_output.save(folder=self.log_dir)

        return pred_output

    def predict(
        self,
        fit_mode_to_use: FitMode | None = None,
        top_k: int | None = None,
        valid_rhs_node_idx: Tensor = None,
        progress_log_dir: str = None,
        *,
        return_embeddings: bool = False,
        job_start_time: datetime | None = None,
        ignore_time_col: bool = False,
        pred_time_col_name: str | None = None,
    ) -> PredOutput:
        r"""Predicts the output of a :class:`kumo.Task` on a given

        Args:
            fit_mode_to_use (FitMode, optional): We will use the model fitted
                with the specified `fit_mode`. If `None`, we will use the
                trained model fitted with the most data.
            top_k (int, optional): If not :obj:`None`, only the :obj:`top_k`
                classes/links with the highest prediction probabilities are
                returned per entity. :obj:`top_k` is only valid for ranking or
                multi-class/multi-label classification tasks. For temporal link
                prediction :obj:`top_k` is required. (default :obj:`None`)
            valid_rhs_node_idx (Tensor, optional): Valid RHS node indices for
                link prediction, passed from :obj:`kumo-pred`. Used for target
                BP filtering of RHS nodes. If :obj:`None`, we will make
                prediction over the entire RHS nodes.
            job_start_time (datetime, optional): If provided, the prediction
                table will contain a new column with name 'JOB_TIMESTAMP' and
                the value will be the provided `job_start_time`.
                NOTE: Applicable only for predictions, not embeddings.
            ignore_time_col (bool, optional): If :obj:`True`, the time column
                will be removed from the prediction output.
                NOTE: Applicable only for predictions, not embeddings.
                (default :obj:`False`)
            pred_time_col_name (str, optional): If provided, the time column
                will be renamed to the provided name in the prediction output.
                NOTE: Applicable only for predictions, not embeddings.
                (default :obj:`None`)
        """
        self.predict_setup(fit_mode_to_use)
        return self.predict_v2(
            top_k=top_k,
            valid_rhs_node_idx=valid_rhs_node_idx,
            progress_log_dir=progress_log_dir,
            return_embeddings=return_embeddings,
            job_start_time=job_start_time,
            ignore_time_col=ignore_time_col,
            pred_time_col_name=pred_time_col_name,
        )

    def print_summary(self, force_init: bool = False):
        r"""Print a summary of the :class:`Trainer` instance.
        """
        if not self._fully_initialized:
            if force_init:
                self.init()
            else:
                logger.warning(
                    "The trainer has not been initialized, and so a summary "
                    "cannot be produced. Please initialize the trainer with "
                    "`trainer.init()` or use `force_init=True`.")
                return

        from tabulate import tabulate
        header = ["name", "shape"]

        print(f'Model task type: {self.task.task_type}')
        print("Encoder parameters:")
        data = []
        for name, param in self.pl_model.encoder.named_parameters():
            data.append([name, param.shape])
        print(tabulate(data, headers=header, tablefmt='psql'))

        print("Model parameters:")
        data = []
        for name, param in self.pl_model.model.named_parameters():
            data.append([name, param.shape])
        print(tabulate(data, headers=header, tablefmt='psql'))

        print("Head parameters:")
        data = []
        for name, param in self.pl_model.head.named_parameters():
            data.append([name, param.shape])
        print(tabulate(data, headers=header, tablefmt='psql'))

    @staticmethod
    def get_progress(log_dir: str) -> TrainerProgress:
        r"""
        Gets the current progress of the trainer by reading the logs in
        :obj:`log_dir`. The progress contains train and validation metrics,
        time taken per epoch etc.
        """
        path = osp.join(log_dir, TrainingProgressLogger.FILE_NAME)

        # If training hasn't started there won't be any progress log.
        if not fsutil.exists(path):
            return TrainerProgress()

        return load_json(path, cls=TrainerProgress)

    # Configuration utilities #################################################

    def compare_config(
        self,
        other: 'Trainer',
        file: Optional[IO] = None,
    ) -> None:
        file = file if file is not None else sys.stdout
        print('Trainer configuration: ', file=file)
        self.diff_config(other, file=file)

    def customize_config(self, values: dict[str, Any]) -> dict[str, Any]:
        # TODO(manan): avoid this explicit removal of encoder statistics from
        # the table encoder dict, which only works since we always call `init`
        # after loading a model from a config. This is weird, but necessary
        # since the UI loads the trainer config...
        values = copy.copy(values)
        encoder = values.get('encoder', None)
        if encoder:
            table_encoder_dict = getattr(encoder, 'table_encoder_dict', None)
            if table_encoder_dict:
                # Delete completely, so the encoder module is no longer
                # fully-specified at prediction time
                del values['encoder'].table_encoder_dict
        return values


# Utilities ###################################################################


def create_lp_embedding_output(
    lhs_emb: Tensor,
    rhs_emb: Tensor,
    entity_table_names: tuple[str, str],
    fkey_names: tuple[str, str],
    lhs_pkey: np.ndarray,
    rhs_pkey: np.ndarray,
    time: Tensor | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    r"""Creates a standardized, formatted dataframe of embeddings assicated
    with the LHS and RHS of a link prediction model."""
    lhs_table, rhs_table = entity_table_names
    lhs_fkey, rhs_fkey = fkey_names

    df = pd.DataFrame()
    df['ID'] = lhs_pkey.tolist() + rhs_pkey.tolist()
    df['TYPE'] = [lhs_table] * len(lhs_pkey) + [rhs_table] * len(rhs_pkey)
    if lhs_table == rhs_table:
        df['NAME'] = [lhs_fkey] * len(lhs_pkey) + [rhs_fkey] * len(rhs_pkey)
    df['EMBEDDING'] = list(torch.cat([lhs_emb, rhs_emb], dim=0).numpy())

    if time is not None:
        assert time_col is not None
        assert len(time.unique()) == 1
        df[time_col] = time[0].numpy().astype('datetime64[ns]')

    return df


def create_lp_prediction_output(
    outs: dict[str, Tensor],
    lhs_fkey_name: str,
    rhs_index_map: pd.DataFrame,
    time: Tensor | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    r"""Creates a standardized, formatted dataframe of predictions associated
    with the links in a link prediction task."""
    # Repeat LHS pkeys k times:
    lhs_ids = np.asarray(outs['pkey']).repeat(outs['target'].size(-1))

    # Convert RHS node IDs to original IDs:
    target = outs['target'].view(-1).cpu().numpy()
    indices = pa.array(target, mask=target == -1)
    values = pa.Array.from_pandas(rhs_index_map.index.to_numpy())
    class_dtype = (pd.UInt64Dtype() if pa.types.is_unsigned_integer(values.type)
                   else pd.Int64Dtype() if pa.types.is_signed_integer(
                       values.type) else pd.ArrowDtype(values.type))
    rhs_ids = values.take(indices).to_pandas(types_mapper=lambda _: class_dtype)

    df = pd.DataFrame({
        lhs_fkey_name: lhs_ids,
        'CLASS': rhs_ids,
        'SCORE': outs['score'].view(-1).numpy(),
    })
    if 'label' in outs:
        df['LABEL'] = outs['label'].view(-1).numpy()

    if time is not None:
        assert time_col is not None
        assert len(time.unique()) == 1
        df[time_col] = time[0].numpy().astype('datetime64[ns]')

    return df


def create_multiclass_prediction_output(
    df: pd.DataFrame,
    target_encoder: TargetEncoder,
    prob: Tensor,
    top_k: int,
) -> pd.DataFrame:
    top_k = min(top_k, prob.size(1))
    assert top_k > 0

    # Repeat pkey/time/target `k` times:
    df = df.loc[df.index.repeat(top_k)].reset_index(drop=True)

    prob, pred = prob.topk(top_k, dim=1)

    predicted = torch.zeros(len(df), dtype=torch.bool)
    predicted[0:len(df):top_k] = True
    df['PREDICTED'] = predicted.numpy()
    df['CLASS'] = target_encoder.decode(pred.view(-1, 1))
    df['SCORE'] = prob.view(-1).cpu().numpy()

    return df


def create_multilabel_ranking_prediction_output(
    df: pd.DataFrame,
    target_encoder: TargetEncoder,
    prob: Tensor,
    target: Tensor | None,
    top_k: int,
) -> pd.DataFrame:
    """
    Args:
        df: The dataframe containing the target values.
        target_encoder: The target encoder used to encode the target values.
        prob: The probability tensor of shape (num_entities, num_classes).
        top_k: The number of top predictions to output. If `top_k` is greater
            than the number of classes, it will be capped at the number of
            classes.

    Returns:
        pd.DataFrame: The dataframe with the predicted classes and scores.

    Raises:
        ValueError: If `top_k` is not greater than 0.
    """
    assert isinstance(target_encoder, MultiCategoricalTargetEncoder)

    num_classes = prob.size(1)
    top_k = min(top_k, num_classes)
    if top_k <= 0:
        raise ValueError(f"'top_k' must be greater than 0. Got top_k={top_k}.")

    # Repeat pkey/time/target `k` times:
    df = df.loc[df.index.repeat(top_k)].reset_index(drop=True)

    prob, pred = prob.topk(top_k, dim=1)

    mapping = target_encoder.tensor_mapper.mapping
    df['CLASS'] = mapping[pred.view(-1).cpu().numpy()].to_numpy()
    df['SCORE'] = prob.view(-1).cpu().numpy()

    if 'TARGET' in df.columns:
        df.drop(columns=['TARGET'], inplace=True)
        assert target is not None
        target = torch.gather(target.to(bool), 1, pred)
        df['LABEL'] = target.view(-1).cpu().numpy()

    return df


def create_multilabel_classification_prediction_output(
    df: pd.DataFrame,
    target_encoder: TargetEncoder,
    prob: Tensor,
    pred: Tensor,
    target: Tensor | None,
    top_k: int,
) -> pd.DataFrame:
    r"""
    Args:
        df: The dataframe containing pkey, optionally with time and target
            columns.
        target_encoder: The target encoder used to encode the target values.
        prob: The probability tensor of size [num_rows, num_classes].
        pred: The prediction tensor of size [num_rows, num_classes].
        target: The target tensor of size [num_rows, num_classes].
        top_k: The number of top predictions to output. If `top_k` is greater
            than the number of classes, it will be capped at the number of
            classes.

    Returns:
        pd.DataFrame: The dataframe with the predicted classes and scores. If
            target is provided, it will also include a LABEL column, indicating
            the ground truth labels.

    Raises:
        ValueError: If ``top_k`` is not greater than 0.
    """
    assert isinstance(target_encoder, MultiCategoricalTargetEncoder)

    num_rows, num_classes = prob.size()
    top_k = min(top_k, num_classes)
    if top_k <= 0:
        raise ValueError(f"'top_k' must be greater than 0. Got top_k={top_k}.")
    if prob.size() != pred.size():
        raise ValueError(
            f"prob and pred must have the same shape. Got {prob.size()} and "
            f"{pred.size()}.")

    # Repeat pkey/time `k` times:
    df = df.loc[df.index.repeat(top_k)].reset_index(drop=True)

    prob, top_k_labels = prob.topk(top_k, dim=1)

    predicted = torch.gather(pred.to(bool), 1, top_k_labels)
    df['PREDICTED'] = predicted.view(-1).cpu().numpy()

    if target is not None:
        target = torch.gather(target.to(bool), 1, top_k_labels)
        df['LABEL'] = target.view(-1).cpu().numpy()

    mapping = target_encoder.tensor_mapper.mapping
    df['CLASS'] = mapping[top_k_labels.view(-1).cpu().numpy()].to_numpy()
    df['SCORE'] = prob.view(-1).cpu().numpy()

    return df


# Helper classes and functions ################################################


class ProgressBar(pl.callbacks.TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # Remove `loss` and `v_num` values from the logger.
        items = super().get_metrics(trainer, model)
        items.pop('loss', None)
        items.pop('v_num', None)
        return items


@contextmanager
def temporary_callbacks(pl_trainer: pl.Trainer, callbacks: list[pl.Callback]):
    r"""A context manager to add temporary callbacks to a :class:`pl.Trainer`
    instance."""
    pl_trainer.callbacks = pl_trainer.callbacks + callbacks
    yield
    for callback in callbacks:
        pl_trainer.callbacks.remove(callback)


def _get_versions(root_dir: str) -> list[int]:
    try:
        versions = fsutil.listdir(root_dir)
    except FileNotFoundError:
        return []
    versions = [osp.basename(v) for v in versions]
    prefix = 'version_'
    versions = [
        int(v[len(prefix):]) for v in versions
        if (v.startswith(prefix) and (not v.endswith(FitMode.refit_full.value))
            and (not v.endswith(FitMode.refit_trainval.value)))
    ]
    return sorted(versions)


def _train_result_json_path(log_dir: str) -> str:
    return osp.join(log_dir, 'train_result.json')


def save_encoders(root_dir: str, encoders: EncodersConfig):
    save_json(encoders, osp.join(root_dir, 'encoders.json'))


def load_encoders(root_dir: str) -> EncodersConfig | None:
    path = osp.join(root_dir, 'encoders.json')
    if fsutil.exists(path):
        logger.info(f"Restoring encoders from file {path}")
        return load_json(path, EncodersConfig)
    return None


def get_best_ckpt_path(ckpt_dir: str) -> tuple[str | None, list[str]]:
    r"""Point the model to the best checkpoint (if present). Note that this
    does not mean the model will execute using parameters from the best
    checkpoint. That is determined by the fit, test, validate and predict
    call parameters.

    Args:
        ckpt_dir (str): Directory path of checkpoints.
    """
    ckpt_paths = fsutil.listdir(ckpt_dir)
    ckpt_paths = [
        path for path in ckpt_paths
        if (path.endswith('.ckpt') and "step=" in path)
    ]
    if len(ckpt_paths) > 0:
        ckpt_steps = [
            int(re.findall(r'step=(.*)\.ckpt', path)[0]) for path in ckpt_paths
        ]
        best_ckpt = [path
                     for _, path in sorted(zip(ckpt_steps, ckpt_paths))][-1]
        return best_ckpt, ckpt_paths
    else:
        return None, ckpt_paths


def generate_majority_sampling_weight(
    majority_sampling_ratio: float,
    target_distribution: dict[bool, float],
) -> dict[bool, float] | None:
    r"""Generates majority sampling weights for binary classification in case
    the ratio of majority vs minority class in the target column is larger than
    `majority_sampling_ratio`. Weights are set in such a way that the ratio
    between majority vs minority class is exactly `majority_sampling_ratio`.

    Args:
        majority_sampling_ratio (float): The desired ratio of majority vs
            minority classes in the target column.
        target_distribution: (Dict[bool, float]): The distribution of
            target values for binary classification problems.
    Returns:
        A dictionary that stores the weighting for majority and minority
        class when the ratio of majority class vs minor class seen during
        training is larger than `majority_sampling_ratio`, `None` otherwise.

    """
    [(minority_value, minority_prob),
     (majority_value, majority_prob)] = sorted(target_distribution.items(),
                                               key=lambda item: item[1])

    majority_sampling_weights = None
    if minority_prob < (1 / (majority_sampling_ratio + 1)):
        majority_sampling_weights = {}
        majority_sampling_weights[majority_value] = 1

        # We upsample the minority class such that the ratio between majority
        # class and minority class is exactly majority_sampling_ratio.
        majority_sampling_weights[minority_value] = (
            majority_prob / (minority_prob * majority_sampling_ratio))

    return majority_sampling_weights
