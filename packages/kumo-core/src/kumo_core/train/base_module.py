# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import inspect
import logging
import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal

import kumo_core.modeling.context as modeling_context
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from kumo_core.train.checkpoint import load_kumo_checkpoint
from kumo_core.types import TaskType, TrainModuleOutput
from kumoapi.train import TrainingStage
from kumo_core.explain.base import Explainer
from kumo_core.evaluation.suite import MetricStore
from kumo_core.encoders.target.base import TargetEncoder
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.hooks import RemovableHandle
from torch_geometric.data import HeteroData
from torch_geometric.nn.resolver import (
    lr_scheduler_resolver,
    optimizer_resolver,
)
from torchmetrics import MetricCollection
from torchmetrics.utilities.data import dim_zero_cat

DEFAULT_OPTIMIZER = 'Adam'
DEFAULT_LR_SCHEDULER = 'cosine_with_warmup_restarts'
DEFAULT_LR_SCHEDULER_INTERVAL = 'step'
_KUMO_BATCH_DUMP_DIR = os.getenv("KUMO_BATCH_DUMP_DIR", "")
_SPARSE_OPTIMIZER_REMAP_CHUNK_SIZE = 65_536

logger = logging.getLogger(__name__)


class BaseLightningModule(pl.LightningModule, ABC):
    r"""A PyTorch Lightning :class:`LightningModule` that handles the execution
    of training, validation, testing and prediction routines.

    PyTorch Lightning builds upon the principles of callbacks to inject custom
    training and evaluation behavior, while taking care of the usual
    boilerplate to train neural networks, such as sanity checking, iterating
    over data loaders, moving mini-batches to the respective devices,
    performing back-propagation, as well as performing optimization steps and
    LR scheduler steps.

    The general flow of a :class:`LightningModule` looks as follows:

    - trainer.fit()
      +-- Create `val_dataloader()`
      +-- Single validation mini-batch sanity checking
      +-- Create `train_dataloader()`
      +-- CALLBACK: configure_optimizers()
      +-- For number of epochs:
          +-- CALLBACK: on_train_epoch_start()
          +-- For number of training steps:
              +-- Move mini-batch to the respective devices
              +-- CALLBACK: train_step() (returns the loss)
              +-- Perform back-propagation and optimization
          +-- CALLBACK: on_validation_start()
          +-- For number of validation steps:
              +-- Disable gradient computation
              +-- Move mini-batch to the respective devices
              +-- CALLBACK: validation_step()
          +-- CALLBACK: on_validation_end()
          +-- CALLBACK: on_train_epoch_end()

    - trainer.validate()
      +-- Create `val_dataloader()` (if not exists yet)
      +-- CALLBACK: on_validation_start()
      +-- For number of validation steps:
          +-- Disable gradient computation
          +-- Move mini-batch to the respective devices
          +-- CALLBACK: validation_step()
      +-- CALLBACK: on_validation_end()

    - trainer.test()
      +-- Create `test_dataloader()`
      +-- CALLBACK: on_test_start()
      +-- For number of test steps:
          +-- Disable gradient computation
          +-- Move mini-batch to the respective devices
          +-- CALLBACK: test_step()
      +-- CALLBACK: on_test_end()

    - trainer.predict()
      +-- Create `predict_dataloader()`
      +-- CALLBACK: on_predict_start()
      +-- For number of prediction steps:
          +-- Disable gradient computation
          +-- Move mini-batch to the respective devices
          +-- CALLBACK: predict_step()
      +-- CALLBACK: on_predict_end()

    See `the PyTorch Lightning docs
    <https://lightning.ai/docs/pytorch/2.1.0/common/lightning_module.html#hooks>`_

    Args:
        task_type (TaskType or str): The type of the task.
        target_encoder (TargetEncoder, optional): The target encoder.
            (default: :obj:`None`)
        metrics (List[str], optional): A list of metrics to track.
            (default: :obj:`None`)
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
        max_epochs_for_lr_scheduler (int): The maximum number of epochs
            we use to decide the learning rate scheduler. In refit mode, this
            may be different from the actual maximum number of epochs used to
            train the model.
        disable_eval_metrics (bool): Whether to disable metrics during
            evaluation step. (default: :obj:`False`)
    """
    #: Optimizer settings for a shallow RHS lookup embedding, which only the
    #: link-prediction subclasses that own such an embedding set. Declared
    #: here because ``configure_optimizers`` reads it for every module, and
    #: ``None`` is how a module without a lookup table says it has none.
    lookup_optimizer_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        task_type: TaskType | str,
        target_encoder: TargetEncoder | None = None,
        metrics: list[str] | None = None,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: str | None = DEFAULT_LR_SCHEDULER,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = DEFAULT_LR_SCHEDULER_INTERVAL,
        max_epochs_for_lr_scheduler: int = 100,
        disable_eval_metrics: bool = False,
        enable_compile: bool = False,
    ) -> None:
        super().__init__()

        self.task_type = TaskType(task_type)
        self.target_encoder = target_encoder
        self.metrics = metrics or []
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.lr_scheduler_interval = lr_scheduler_interval
        self.max_epochs_for_lr_scheduler = max_epochs_for_lr_scheduler
        self.enable_compile = enable_compile
        # Max numel of multilabel preds and target to perform GPU computation
        # over. 100 million * 12 bytes (4 + 8 bytes) = 1.2GB.
        self.max_multilabel_samples = 100_000_000

        # Register metric store ###############################################

        self.metric_store = MetricStore(task_type, target_encoder)

        # Helper attributes ###################################################

        self.explainers: list[Explainer] = []

        self._embedding_handle: list[RemovableHandle] | None = None
        self._embedding_step_outputs: dict[str, list[Tensor]] | None = None
        self._embedding_outputs: dict[str, Tensor] | None = None
        self._validation_step_outputs: list[dict[str, Tensor | None]] = []
        self._test_step_outputs: list[dict[str, Tensor | None]] = []

        self.disable_eval_metrics = disable_eval_metrics
        self._batch_dump_dir = _KUMO_BATCH_DUMP_DIR

        # Optimizer state restored by ``load_checkpoint``, which is the only
        # writer. Declared here so every attribute exists from construction:
        # ``_loaded_optimizer_states`` is the gate the others are read behind,
        # and the RHS reorder trio is set only when the checkpoint carries a
        # lookup embedding to remap.
        self._loaded_optimizer_states: list[dict[str, Any]] | None = None
        self._new_rhs_order: pd.DataFrame | None = None
        self._old_rhs_order: pd.DataFrame | None = None
        self._old_rhs_capacity: int | None = None
        self._new_rhs_capacity: int | None = None

    # Register a `__post_init__` hook that runs after all `__init__` functions
    # (both of parent and children) have been executed:
    def __init_subclass__(cls, **kwargs):
        def decorator(__init__):
            @functools.wraps(__init__)
            def func(self, *args, **kwargs):
                __init__(self, *args, **kwargs)
                if type(self) is cls:
                    self.__post_init__()

            return func

        cls.__init__ = decorator(cls.__init__)

    def __post_init__(self):
        # Save all hyperparameters for PyTorch Lightning. Since we cannot save
        # any `torch.nn.Module` here, we save the config of the model parts.
        # In order to then load a model from checkpoint, we override the
        # PyTorch Lightning `load_from_checkpoint` functionality below that
        # reconstructs the model parts from configuration.

        # Receive arguments passed to the constructor from the current frame:
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        keys = inspect.signature(self.__init__).parameters

        hparams = {key: value for key, value in zip(keys, local_vars['args'])}
        hparams.update(local_vars['kwargs'])

        for key, value in hparams.items():  # Convert modules to their config:
            if isinstance(value, torch.nn.Module) and hasattr(value, 'config'):
                hparams[key] = asdict(value.config())

        self._set_hparams(hparams)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        map_location: torch.device | None = None,
        load_optimizer_state: bool = False,
        _new_rhs_order: pd.DataFrame | None = None,
    ) -> 'BaseLightningModule':
        r"""Loads the model from a PyTorch Lightning checkpoint file.

        This method loads a previously saved model checkpoint, restoring the model's
        state dictionary and optionally the optimizer states.

        Args:
            checkpoint_path (str): Path to the checkpoint file to load.
            map_location (torch.device, optional): Device to map the checkpoint to.
            load_optimizer_state (bool, optional): Whether to load and store optimizer
                states from the checkpoint. If True, optimizer states are stored in
                `self._loaded_optimizer_states` and will be loaded during
                `configure_optimizers` call. (default: :obj:`False`)
            _new_rhs_order (pd.DataFrame, optional): DataFrame used during
                `configure_optimizers` to reorder or add new entries rhs lookup embedding
                when loading the optimizer state. (default: :obj:`None`)
        """
        checkpoint = load_kumo_checkpoint(
            checkpoint_path,
            map_location=map_location,
        )
        checkpoint_state = checkpoint['state_dict']

        # Capture physical capacities before loading replaces the newly
        # constructed model's embedding with the checkpoint embedding.
        optimizer_rhs_state: tuple[pd.DataFrame, int, int] | None = None
        if load_optimizer_state and _new_rhs_order is not None:
            extra_states = [
                key for key in checkpoint_state
                if key.endswith('lookup_embedding._extra_state')
            ]
            if len(extra_states) == 1:
                extra_state_key = extra_states[0]
                weight_key = (extra_state_key.removesuffix('_extra_state') +
                              'weight')
                checkpoint_weight = checkpoint_state.get(weight_key)
                new_rhs_capacity = getattr(self, 'num_rhs_nodes', None)
                if (not isinstance(checkpoint_weight, Tensor)
                        or checkpoint_weight.ndim != 2
                        or not isinstance(new_rhs_capacity, int)):
                    raise RuntimeError(
                        'RHS optimizer restore could not resolve lookup '
                        'embedding capacities')
                optimizer_rhs_state = (
                    checkpoint_state[extra_state_key]['index_map'],
                    int(checkpoint_weight.shape[0]),
                    new_rhs_capacity,
                )

        self.load_state_dict(checkpoint_state)

        # Store optimizer and lr_scheduler states only if requested
        if load_optimizer_state:
            self._loaded_optimizer_states = checkpoint['optimizer_states']
            self._new_rhs_order = _new_rhs_order
            if optimizer_rhs_state is not None:
                (self._old_rhs_order, self._old_rhs_capacity,
                 self._new_rhs_capacity) = optimizer_rhs_state

        return self

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for module in self.children():
            module.reset_parameters()

    @property
    def eval_top_k(self) -> int:
        r"""Returns the top-:math:`k` value used for evaluation output."""
        if not hasattr(self, '_eval_top_k'):
            return 0
        return self._eval_top_k or 0

    @eval_top_k.setter
    def eval_top_k(self, value: int | None):
        r"""Sets the top-:math:`k` value used for evaluation output."""
        self._eval_top_k = value or 0

    @property
    def pred_top_k(self) -> int | None:
        r"""Returns the top-:math:`k` value used for batch prediction."""
        # TODO: Align with eval_top_k
        if not hasattr(self, '_pred_top_k'):
            return None
        return self._pred_top_k

    @pred_top_k.setter
    def pred_top_k(self, value: int | None):
        r"""Sets the top-:math:`k` value used for batch prediction."""
        self._pred_top_k = value

    # Model Execution #########################################################

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        r"""The forward pass of the model."""

    @abstractmethod
    def common_step(self, batch: HeteroData) -> Any:
        r"""A common step that defines how a :class:`HeteroData` object gets
        mapped to the machine learning model."""

    def get_embedding(
        self,
        batch: HeteroData,
        entity_name: str,
        side: Literal['lhs', 'rhs'] | None = None,
    ) -> Tensor:
        r"""Returns the embedding for seed nodes of :obj:`entity_name` in
        :obj:`batch`.

        For link prediction modules, :obj:`side` specifies which projection
        head to apply (``'lhs'`` or ``'rhs'``). For node prediction modules,
        :obj:`side` is unused.
        """
        raise NotImplementedError

    # Training, evaluation and prediction routines ############################

    def on_train_epoch_start(self):
        self.outputs: dict[str, Tensor | None] | None = None

    @abstractmethod
    def train_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        r"""Training step :obj:`batch_idx` for a mini-batch :obj:`batch`.
        Returns the loss value.
        """

    def on_train_epoch_end(self):
        self.compute_metrics(TrainingStage.TRAIN)
        gc.collect()

    def on_evaluation_start(self, stage: TrainingStage):
        self.outputs: dict[str, Tensor | None] | None = None
        if self._trainer is None:
            # Baseline module somehow uses this lightning module without a
            # PyTorch Lightning's trainer.
            num_batches = -1
        elif stage == TrainingStage.VAL:
            num_batches = self.trainer.num_val_batches
        elif stage == TrainingStage.TEST:
            num_batches = self.trainer.num_test_batches
        logger.info(f"Running {stage.value} evaluation across {num_batches} "
                    f"mini-batches with explainers '{self.explainers}'")

    @abstractmethod
    def evaluation_step(
        self,
        batch: HeteroData,
        stage: TrainingStage,
        batch_idx: int,
    ) -> dict[str, Tensor]:
        r"""Evaluation step :obj:`batch_idx` for a mini-batch :obj:`batch`.
        Returns model predictions and targets."""

    def evaluation_epoch_end(
        self,
        stage: TrainingStage,
        step_outputs: list[TrainModuleOutput],
    ) -> TrainModuleOutput:

        self.compute_metrics(stage)

        self.outputs: TrainModuleOutput = {}
        if len(step_outputs) > 0:
            for key, value in step_outputs[0].items():
                if value is None:
                    self.outputs[key] = value
                elif isinstance(value, dict):
                    # Distribution loss has dictionary prob output:
                    self.outputs[key] = {}
                    for dist_key in value:
                        self.outputs[key][dist_key] = torch.cat(
                            [out[key][dist_key] for out in step_outputs],
                            dim=0,
                        )
                elif isinstance(value, Tensor):
                    # Concatenate step outputs to single tensors:
                    self.outputs[key] = torch.cat(
                        [out[key] for out in step_outputs], dim=0)
                elif isinstance(value, np.ndarray):
                    self.outputs[key] = np.concatenate(
                        [out[key] for out in step_outputs], axis=0)

        return self.outputs

    @abstractmethod
    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor]:
        r"""Prediction step :obj:`batch_idx` for a mini-batch :obj:`batch`.
        Returns model predictions."""

    # Embedding routines ######################################################

    def register_embedding_hook(self, stage: TrainingStage):
        r"""Registers necessary hooks to capture embeddings during model
        execution."""
        if (self._embedding_handle is not None
                or self._embedding_step_outputs is not None
                or self._embedding_outputs is not None):
            raise RuntimeError(
                f"Cannot register a new embedding hook without clearing the "
                f"old embedding state. Please call "
                f"'{self.__class__.__name__}'.get_embeddings()' before "
                f"proceeding")

        register_fn = self._get_register_embedding_hook_fn()
        if register_fn is not None:

            def hook(*args, **kwargs):
                step_output = self.embedding_step(stage, *args, **kwargs)
                if self._embedding_step_outputs is None:
                    self._embedding_step_outputs = defaultdict(list)
                for key, value in step_output.items():
                    value = value.detach().cpu()
                    self._embedding_step_outputs[key].append(value)

            self._embedding_handle = register_fn(hook)

        else:
            warnings.warn(f"Trying to register an embedding hook, but no "
                          f"embedding logic implemented in "
                          f"'{self.__class__.__name__}'")

    def has_embedding_hook(self) -> bool:
        r"""Returns whether the module has an embedding hook."""
        return self._embedding_handle is not None

    def reset_embedding_hook(self):
        r"""Resets the embedding hook."""
        self._embedding_handle = None
        self._embedding_step_outputs = None
        self._embedding_outputs = None

    def _get_register_embedding_hook_fn(self) -> Callable | None:
        r"""Returns a hook function of a child module to serve as input into
        :meth:`embedding_step`."""
        return None

    def embedding_step(
        self,
        stage: TrainingStage,
        *args,
        **kwargs,
    ) -> dict[str, Tensor]:
        r"""Returns the embeddings per mini-batch and node type from the hook
        outputs."""
        return {}

    def _embedding_epoch_end(self, stage: TrainingStage):
        r"""Fuses the embeddings obtained from all mini-batches together.
        Customizable via the :meth:`embedding_epoch_end` callback."""
        outputs = {
            key: torch.cat(values, dim=0) if len(values) > 1 else values[0]
            for key, values in self._embedding_step_outputs.items()
        }

        assert self._embedding_handle is not None
        self._embedding_handle.remove()  # Clear old state.
        self._embedding_handle = None
        self._embedding_step_outputs = None

        self._embedding_outputs = self.embedding_epoch_end(stage, outputs)

    def embedding_epoch_end(
        self,
        stage: TrainingStage,
        outputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        r"""Callback for post-processing embedding outputs."""
        return outputs

    def get_embeddings(self) -> dict[str, Tensor]:
        r"""Returns the final embeddings per node type."""
        if self._embedding_outputs is None:
            raise RuntimeError(
                f"Cannot provide embeddings. You either forgot to register an "
                f"embedding hook via "
                f"'{self.__class__.__name__}.register_embedding_hook(stage)' "
                f"or called '{self.__class__.__name__}.get_embeddings()' "
                f"multiple times")

        out = self._embedding_outputs
        self._embedding_outputs = None  # Clear old state.
        return out

    # Metric Computation and Logging ##########################################

    def log_epoch(self, name: str, value: Tensor, **kwargs):
        r"""Logs a metric with name :obj:`"name"` with value :obj:`value` to
        to the logger and progress bar to show per epoch."""
        kwargs.setdefault('prog_bar', True)
        kwargs.setdefault('on_step', False)
        kwargs.setdefault('on_epoch', True)
        kwargs.setdefault('metric_attribute', name)
        return super().log(name, value, **kwargs)

    def log_step(self, name: str, value: Tensor, **kwargs):
        r"""Logs a metric with name :obj:`"name"` with value :obj:`value` to
        the logger and progress bar to show every `log_every_n_steps` steps."""
        kwargs.setdefault('prog_bar', False)
        kwargs.setdefault('on_step', True)
        kwargs.setdefault('on_epoch', False)
        kwargs.setdefault('metric_attribute', name)
        return super().log(name, value, **kwargs)

    def log_learning_rate(self):
        # No-op if PL trainer is not attached
        if self._trainer is None:
            return
        # No-op if no optimizer is registered:
        if not hasattr(self.trainer, 'optimizers'):
            return

        return self.log_step(
            name='params/learning_rate',
            value=self.trainer.optimizers[0].param_groups[0]['lr'],
        )

    @property
    def is_fitting(self) -> bool:
        r"""Indicates whether in the fitting stage or not."""
        return (self._trainer is not None
                and self.trainer.state.fn == TrainerFn.FITTING)

    @torch.no_grad()
    def update_metrics(
        self,
        stage: TrainingStage,
        pred: Tensor,
        target: Tensor | tuple[Tensor, Tensor],
        **kwargs,
    ) -> None:
        r"""Accumulates all defined metrics for a given mini-batch prediction.

        Args:
            stage (TrainingStage): The split for which metrics are updated.
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The target.
            kwargs: Any additional information passed to the metric, *e.g.*,
                weighting coefficients.
        """
        metric = self.metric_store.get_metric_collection(stage)

        if self.task_type == TaskType.MULTILABEL_RANKING:
            # TaskType.MULTILABEL_RANKING is a node prediction task, but
            # we use the same metric as for link prediction tasks, since
            # the prediction requires the same top-k evaluation as link
            # prediction.
            target = target.nonzero().t()
            _, pred = pred.topk(min(metric.max_k, pred.size(-1)), dim=-1)

        metric.update(pred, target, **kwargs)  # type: ignore

    @torch.no_grad()
    def compute_metrics(self, stage: TrainingStage) -> dict[str, Tensor]:
        r"""Computes the final value for all defined metrics over all
        accumulated mini-batches.

        Args:
            stage (TrainingStage): The stage for which metrics are computed.
        """
        metric = self.metric_store.get_metric_collection(stage)

        if len(metric) > 0:
            if isinstance(metric, MetricCollection):
                # No need to deepcopy state just to check `_update_count`
                m = next(iter(metric.values(copy_state=False)))
            else:
                m = next(iter(metric.values()))

            if m._update_count == 0:
                return {}

        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            # NOTE: This code path is not explicitly tested by the unit test.
            # Sample a representative key. We assume ap_macro is always present
            key = stage.add_stage_prefix('ap_macro')
            assert key in metric
            preds = metric.__getitem__(key, copy_state=False).preds
            target = metric.__getitem__(key, copy_state=False).target
            num_data = sum([pred.numel() for pred in preds])

            if num_data > self.max_multilabel_samples:
                logger.info(f"{num_data} data found in multi-label metric "
                            f"computation. Subsampling down to "
                            f"{self.max_multilabel_samples}.")
                pred_cat = dim_zero_cat(preds)
                target_cat = dim_zero_cat(target)

                num_rows = len(pred_cat)
                num_rows_to_sample = None
                if pred_cat.ndim == 1:
                    # single-label scenario
                    num_rows_to_sample = self.max_multilabel_samples
                else:
                    # multi-label scenario
                    num_labels = len(pred_cat[0])
                    num_rows_to_sample = (self.max_multilabel_samples //
                                          num_labels)

                assert num_rows_to_sample is not None
                sampled_idx = torch.randperm(num_rows)[:num_rows_to_sample]

                # It is sufficient to update the state of a single metric.
                # `MetricCollection` shares this state across other metrics.
                metric.__getitem__(key, False).preds = [pred_cat[sampled_idx]]
                metric.__getitem__(key,
                                   False).target = [target_cat[sampled_idx]]

            # Use GPU for fast computation
            metric.to(self.device)

        out_dict = metric.compute()
        metric.reset()

        out_dict_processed: dict[str, Tensor] = {}
        for name, out in out_dict.items():
            # For multi-label per_label metric, `out` is a 1-dim tensor.
            # We log each element of `out`, corresponding to the metric
            # for each binary label.
            # We do not log per-label metrics at fit() stage to avoid
            # pollution of the training progress bar and tensorboard.
            if "_per_label" in name:
                if not self.is_fitting:
                    if out.dim() == 0:  # Single label:
                        out = out.unsqueeze(0)
                    assert self.target_encoder is not None
                    mapping = self.target_encoder.tensor_mapper.mapping
                    assert len(mapping) == len(out)
                    for i, value in enumerate(out):
                        name_processed = f'{name}_{mapping[i]}'
                        out_dict_processed[name_processed] = value
                        self.log_epoch(name_processed, value, prog_bar=False)
            else:
                out_dict_processed[name] = out
                self.log_epoch(name, out)

        return out_dict_processed

    # Optimizer and LR Schedulers #############################################

    def configure_optimizers(
            self) -> tuple[dict[str, Optimizer | LRScheduler]] | None:
        r"""Initializes optimizers and LR schedulers."""
        if sum([param.numel() for param in self.parameters()]) == 0:
            return None

        def get_optimizer_config(
            optimizer: Optimizer,
            idx: int,
        ) -> dict[str, Optimizer | LRScheduler]:
            optimizer_config = dict(optimizer=optimizer)

            # Load optimizer state if available:
            if self._loaded_optimizer_states is not None:
                logger.info(f"Loading optimizer state for optimizer {idx}")

                if idx == 1:
                    assert self._new_rhs_order is not None
                    assert self._old_rhs_order is not None
                    assert self._old_rhs_capacity is not None
                    assert self._new_rhs_capacity is not None
                    state = self._loaded_optimizer_states[idx]['state'][0]
                    for key, value in state.items():  # Reorder the state:
                        if (isinstance(value, Tensor) and value.dim() > 0
                                and value.size(0) == self._old_rhs_capacity):
                            state[key] = order_by(
                                self._new_rhs_order,
                                self._old_rhs_order,
                                value,
                                new_num_embeddings=self._new_rhs_capacity,
                            )
                    self._new_rhs_order = self._old_rhs_order = None
                    self._new_rhs_capacity = self._old_rhs_capacity = None

                for new, old in zip(  # Do not inherit LR/weight decay:
                        self._loaded_optimizer_states[idx]['param_groups'],
                        optimizer.param_groups,
                ):
                    new['lr'] = old['lr']
                    new['initial_lr'] = old['lr']
                    if 'weight_decay' in old:
                        new['weight_decay'] = old['weight_decay']

                optimizer.load_state_dict(self._loaded_optimizer_states[idx])

            if self.lr_scheduler is not None:
                steps_per_epoch = len(
                    self.trainer.datamodule.train_dataloader())
                lr_scheduler = lr_scheduler_resolver(
                    self.lr_scheduler,
                    optimizer=optimizer,
                    num_training_steps=(steps_per_epoch *
                                        self.max_epochs_for_lr_scheduler),
                    **self.lr_scheduler_kwargs,
                )
                optimizer_config.update(dict(lr_scheduler=lr_scheduler))

            if self.enable_compile:
                optimizer.step = torch.compile(optimizer.step)

            return optimizer_config

        # We create two optimizers here: One for GNN/feature transformations
        # and one for the RHS lookup embedding (in case it exists):
        named_lookup_parameters: list[tuple[str, Tensor]] = []
        RHS_EMBEDDING_NAME = 'rhs_embedding.lookup_embedding.weight'
        if (hasattr(self, 'rhs_embedding')
                and self.rhs_embedding.lookup_embedding is not None):
            named_lookup_parameters = [(
                RHS_EMBEDDING_NAME,
                self.rhs_embedding.lookup_embedding.weight,
            )]
        named_parameters = [(name, w) for name, w in self.named_parameters()
                            if name != RHS_EMBEDDING_NAME]

        optimizer = optimizer_resolver(
            self.optimizer,
            params=[w for _, w in named_parameters],
            **self.optimizer_kwargs,
        )
        out = (get_optimizer_config(optimizer, idx=0), )

        if len(named_lookup_parameters) > 0:
            is_sparse = self.rhs_embedding.lookup_embedding.sparse
            optimizer_kwargs = self.lookup_optimizer_kwargs
            if optimizer_kwargs is None:
                optimizer_kwargs = {
                    'optimizer': self.optimizer,
                    **self.optimizer_kwargs,
                }
            optimizer_kwargs = optimizer_kwargs.copy()
            optimizer_kwargs.pop('weight_decay', None)
            name = optimizer_kwargs.pop('optimizer', 'Adam')
            if is_sparse and name.lower() not in {'adam', 'adagrad'}:
                name = 'Adam'  # Only Adam and Adagrad support sparse grads.

            if name.lower() == 'adam' and is_sparse:
                optimizer = torch.optim.SparseAdam(
                    params=[w for _, w in named_lookup_parameters],
                    **optimizer_kwargs,
                )
            else:
                optimizer = optimizer_resolver(
                    name,
                    params=[w for _, w in named_lookup_parameters],
                    weight_decay=0.0,
                    **optimizer_kwargs,
                )
            out += (get_optimizer_config(optimizer, idx=1), )

        self._loaded_optimizer_states = None

        return out

    def lr_scheduler_step(self, *args, **kwargs):
        # Explicit passthrough retained for the pinned Lightning/Torch scheduler
        # callback contract.
        return super().lr_scheduler_step(*args, **kwargs)

    # Training Callbacks ####################################################

    def setup(self, stage: TrainerFn) -> None:
        if not self.metric_store.is_initialized and stage in {
                TrainerFn.FITTING,
                TrainerFn.VALIDATING,
                TrainerFn.TESTING,
        }:
            self.init_metrics()

    def init_metrics(self) -> None:
        self.metric_store.init_metrics(self.metrics, self)

    @property
    def automatic_optimization(self) -> bool:
        # Disable automatic optimization in orderto handle multiple optimizers.
        return False

    def training_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        if self._batch_dump_dir:
            self._maybe_dump_batch(batch, "train")
        # Perform manual optimization:
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = self.train_step(batch, batch_idx)

        if loss is not None:
            self.manual_backward(loss)

        for optimizer in optimizers:
            optimizer.step()

        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is None:
            lr_schedulers = []
        elif not isinstance(lr_schedulers, (list, tuple)):
            lr_schedulers = [lr_schedulers]

        for lr_scheduler in lr_schedulers:
            if self.lr_scheduler_interval == 'step':
                lr_scheduler.step()
            elif self.trainer.is_last_batch:
                lr_scheduler.step()

        return loss

    # Validation Callbacks ####################################################

    def is_explaining(self) -> bool:
        return len(self.explainers) > 0

    def on_validation_start(self):
        if (getattr(self, '_trainer', None) is not None
                and self.trainer.sanity_checking):
            self.on_sanity_check_start()

        self.on_evaluation_start(TrainingStage.VAL)

        for explainer in self.explainers:
            explainer.explain_start(
                module=self,
                datamodule=self.trainer.datamodule,
                stage=TrainingStage.VAL,
            )

    def validation_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:
        if self._batch_dump_dir:
            self._maybe_dump_batch(batch, "val")
        if self.is_explaining():
            seed_table_name = batch.input_type
            assert seed_table_name is not None
            batch_size = batch[seed_table_name].batch_size

            # Only log when XAI is enabled to avoid log flooding.
            logger.info(f"Running validation step {batch_idx} with batch size "
                        f"{batch_size}")

        requires_grad = False
        for explainer in self.explainers:
            requires_grad |= explainer.requires_grad
            explainer.explain_pre_step(
                batch=batch,
                module=self,
                datamodule=self.trainer.datamodule,
                stage=TrainingStage.VAL,
                batch_idx=batch_idx,
            )

        with (modeling_context.explaining(self.is_explaining()),
              torch.set_grad_enabled(requires_grad)):
            out = self.evaluation_step(batch, TrainingStage.VAL, batch_idx)

        for explainer in self.explainers:
            explainer.explain_post_step(
                batch=batch,
                module=self,
                datamodule=self.trainer.datamodule,
                step_output=out,
                stage=TrainingStage.VAL,
                batch_idx=batch_idx,
            )

        self._validation_step_outputs.append({
            key:
            value.detach().cpu() if isinstance(value, Tensor) else value
            for key, value in out.items()
        })

    def on_validation_epoch_end(self):
        outputs = self.evaluation_epoch_end(
            TrainingStage.VAL,
            self._validation_step_outputs,
        )
        self._validation_step_outputs.clear()

        for explainer in self.explainers:
            explainer.explain_end(
                module=self,
                datamodule=self.trainer.datamodule,
                outputs=outputs,
                stage=TrainingStage.VAL,
            )

        if self._embedding_step_outputs is not None:
            self._embedding_epoch_end(TrainingStage.VAL)

        if (getattr(self, '_trainer', None) is not None
                and self.trainer.sanity_checking):
            self.on_sanity_check_end()

    # Testing Callbacks #######################################################

    def on_test_start(self):
        self.on_evaluation_start(TrainingStage.TEST)

        for explainer in self.explainers:
            explainer.explain_start(
                module=self,
                datamodule=self.trainer.datamodule,
                stage=TrainingStage.TEST,
            )

    def test_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:
        if batch is None:
            # The TT-NG explain dataloader can yield an empty/None tail batch
            # when the test split is small; there is nothing to explain for it.
            return {}
        if self._batch_dump_dir:
            self._maybe_dump_batch(batch, "test")
        if self.is_explaining():
            seed_table_name = batch.input_type
            assert seed_table_name is not None
            batch_size = batch[seed_table_name].batch_size

            # Only log when XAI is enabled to avoid log flooding.
            logger.info(
                f"Running test step {batch_idx} with batch size {batch_size}")

        requires_grad = False
        for explainer in self.explainers:
            requires_grad |= explainer.requires_grad
            explainer.explain_pre_step(
                batch=batch,
                module=self,
                datamodule=self.trainer.datamodule,
                stage=TrainingStage.TEST,
                batch_idx=batch_idx,
            )

        with (modeling_context.explaining(self.is_explaining()),
              torch.set_grad_enabled(requires_grad)):
            out = self.evaluation_step(batch, TrainingStage.TEST, batch_idx)

        for explainer in self.explainers:
            explainer.explain_post_step(
                batch=batch,
                module=self,
                datamodule=self.trainer.datamodule,
                step_output=out,
                stage=TrainingStage.TEST,
                batch_idx=batch_idx,
            )

        self._test_step_outputs.append({
            key:
            value.detach().cpu() if isinstance(value, Tensor) else value
            for key, value in out.items()
        })

    def on_test_epoch_end(self):
        outputs = self.evaluation_epoch_end(
            TrainingStage.TEST,
            self._test_step_outputs,
        )
        self._test_step_outputs.clear()

        for explainer in self.explainers:
            explainer.explain_end(
                module=self,
                datamodule=self.trainer.datamodule,
                stage=TrainingStage.TEST,
                outputs=outputs,
            )

        if self._embedding_step_outputs is not None:
            self._embedding_epoch_end(TrainingStage.TEST)

    # Prediction Callbacks ####################################################

    def on_predict_start(self):
        if self._trainer is None:
            # Baseline module somehow uses this lightning module without a
            # PyTorch Lightning's trainer.
            num_batches = -1
        else:
            num_batches = self.trainer.num_predict_batches
        logger.info(f"Running predict across {num_batches} mini-batches with "
                    f"explainers '{self.explainers}'")

        for explainer in self.explainers:
            explainer.explain_start(
                module=self,
                datamodule=self.trainer.datamodule,
                stage=TrainingStage.PRED,
            )

    def on_predict_batch_start(
        self,
        batch: HeteroData,
        batch_idx: int,
        dataloader_idx=0,
    ):
        if self.is_explaining():
            seed_table_name = batch.input_type
            assert seed_table_name is not None
            batch_size = batch[seed_table_name].batch_size

            # Only log when XAI is enabled to avoid log flooding.
            logger.info(f"Running predict step {batch_idx} with batch size "
                        f"{batch_size}")

        requires_grad = False
        for explainer in self.explainers:
            requires_grad |= explainer.requires_grad
            explainer.explain_pre_step(
                batch=batch,
                module=self,
                datamodule=self.trainer.datamodule,
                stage=TrainingStage.PRED,
                batch_idx=batch_idx,
            )

        modeling_context.set_explaining(self.is_explaining())
        # TODO: Use a context manager instead so that we can always restore the
        # gradient context even when an exception is raised.
        torch.set_grad_enabled(requires_grad)

    def on_predict_batch_end(
        self,
        out: dict[str, Tensor | None],
        batch: HeteroData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        modeling_context.set_explaining(False)

        for explainer in self.explainers:
            explainer.explain_post_step(
                batch=batch,
                module=self,
                datamodule=self.trainer.datamodule,
                step_output=out,
                stage=TrainingStage.PRED,
                batch_idx=batch_idx,
            )

        torch.set_grad_enabled(False)

    def on_predict_end(self):
        if len(self.explainers) > 0:
            step_outputs = self.trainer.predict_loop._predictions[0]
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

            for explainer in self.explainers:
                explainer.explain_end(
                    module=self,
                    datamodule=self.trainer.datamodule,
                    outputs=outputs,
                    stage=TrainingStage.PRED,
                )

        if self._embedding_step_outputs is not None:
            self._embedding_epoch_end(TrainingStage.PRED)

    def on_sanity_check_start(self) -> None:
        modeling_context.set_debugging(True)

    def on_sanity_check_end(self) -> None:
        modeling_context.set_debugging(False)

    def _batch_dump_extra_payload(
        self,
        batch: Any,
        stage: str,
    ) -> dict[str, Any] | None:
        return None

    def _maybe_dump_batch(self, batch: Any, stage: str) -> None:
        dump_dir = self._batch_dump_dir
        if not dump_dir:
            return

        try:
            import hashlib
            import os

            import torch
            from torch_geometric.data import HeteroData

            if not isinstance(batch, HeteroData):
                return

            def _values_to_list(
                    values: Any,
                    batch_size: int | None = None) -> list[Any] | None:
                if values is None:
                    return None
                if batch_size is not None:
                    try:
                        values = values[:batch_size]
                    except Exception:
                        pass
                if hasattr(values, "detach"):
                    values = values.detach().cpu()
                if hasattr(values, "tolist"):
                    out = values.tolist()
                elif hasattr(values, "to_numpy"):
                    out = values.to_numpy().tolist()
                elif isinstance(values, (list, tuple)):
                    out = list(values)
                else:
                    out = [values]
                if not isinstance(out, list):
                    out = [out]
                return out

            def _normalize_identity_value(value: Any) -> str:
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass
                if isinstance(value, (list, tuple)):
                    inner = ",".join(
                        _normalize_identity_value(item) for item in value)
                    return f"[{inner}]"
                return str(value)

            def _row_tokens(columns: list[tuple[str, list[Any]]]) -> list[str]:
                usable = [(name, values) for name, values in columns if values]
                if not usable:
                    return []
                num_rows = min(len(values) for _, values in usable)
                out = []
                for idx in range(num_rows):
                    out.append("|".join(
                        f"{name}={_normalize_identity_value(values[idx])}"
                        for name, values in usable))
                return out

            def _store_value(store: Any, name: str) -> Any:
                if hasattr(store, "get"):
                    try:
                        value = store.get(name)
                    except Exception:
                        value = None
                    if value is not None:
                        return value
                return getattr(store, name, None)

            seed_nt = None

            if hasattr(self, "lhs_name") and self.lhs_name in batch.node_types:
                seed_nt = self.lhs_name
            elif hasattr(
                    self,
                    "entity_name") and self.entity_name in batch.node_types:
                seed_nt = self.entity_name

            if seed_nt is None:
                input_type = getattr(batch, "input_type", None)
                if input_type is not None and input_type in batch.node_types:
                    seed_nt = input_type

            if seed_nt is None:
                for nt in batch.node_types:
                    store = batch[nt]
                    if getattr(store, "input_id", None) is not None:
                        seed_nt = nt
                        break

            table_name = "batch"
            dm = (getattr(self.trainer, "datamodule", None) if getattr(
                self, "trainer", None) is not None else None)
            if dm is not None:
                if getattr(dm, "_train_table_name", None) is not None:
                    table_name = dm._train_table_name
                elif getattr(dm, "train_table_name", None) is not None:
                    table_name = dm.train_table_name

            if table_name == "batch":
                for nt in batch.node_types:
                    if getattr(batch[nt], "y", None) is not None:
                        table_name = nt
                        break

            train_store = (batch[table_name]
                           if table_name in batch.node_types else None)
            seed_store = batch[seed_nt] if seed_nt is not None else None
            batch_size = (getattr(seed_store, "batch_size", None)
                          if seed_store is not None else None)

            seed_tokens: list[str] = []
            if seed_store is not None:
                pkeys = _values_to_list(
                    getattr(seed_store, "pkey", None),
                    batch_size,
                )
                time_values = None
                if train_store is not None:
                    time_values = _values_to_list(
                        _store_value(train_store, "time"),
                        batch_size,
                    )
                if time_values is None:
                    seed_time = _store_value(seed_store, "time")
                    if seed_time is None:
                        seed_time = _store_value(seed_store, "seed_time")
                    time_values = _values_to_list(
                        seed_time,
                        batch_size,
                    )
                seed_tokens = _row_tokens([
                    ("pkey", pkeys or []),
                    ("time", time_values or []),
                ])

                if not seed_tokens:
                    for attr in ("input_id", "n_id"):
                        values = _values_to_list(
                            getattr(seed_store, attr, None),
                            batch_size,
                        )
                        seed_tokens = _row_tokens([(attr, values or [])])
                        if seed_tokens:
                            break

            if not seed_tokens and train_store is not None:
                train_batch_size = getattr(train_store, "batch_size", None)
                for attr in ("n_id", "input_id"):
                    values = _values_to_list(
                        getattr(train_store, attr, None),
                        train_batch_size,
                    )
                    seed_tokens = _row_tokens([(attr, values or [])])
                    if seed_tokens:
                        break

            if not seed_tokens:
                return

            try:
                sorted_list = sorted(seed_tokens)
            except Exception:
                sorted_list = seed_tokens

            seed_str = f"{stage}:" + ",".join(sorted_list)
            seed_hash = hashlib.md5(seed_str.encode("utf-8")).hexdigest()

            os.makedirs(dump_dir, exist_ok=True)
            filename = os.path.join(
                dump_dir, f"batch_{stage}_{table_name}_{seed_hash}.pt")
            extra_payload = self._batch_dump_extra_payload(batch, stage)
            if extra_payload:
                batch.kumo_parity_extra = extra_payload
            try:
                torch.save(batch, filename)
            finally:
                if extra_payload and hasattr(batch, "kumo_parity_extra"):
                    del batch.kumo_parity_extra
        except Exception as e:
            logger.warning("Failed to dump batch: %s", e)


def order_by(
    new_index_map: pd.DataFrame,
    old_index_map: pd.DataFrame,
    old_tensor: torch.Tensor,
    *,
    new_num_embeddings: int,
) -> torch.Tensor:
    r"""Re-orders the old_tensor according to the new_index_map.
    New indices that are not in the old_index_map are set to zeros.

    Args:
        new_index_map: Unique raw keys mapped to new physical row IDs.
        old_index_map: Unique raw keys mapped to old physical row IDs.
        old_tensor: Tensor with old physical-capacity rows.
        new_num_embeddings: New physical row capacity, including holes.

    Returns:
        Tensor with ``new_num_embeddings`` rows.
    """
    if old_tensor.ndim != 2:
        raise ValueError(
            f'old_tensor must be a 2D tensor but got {old_tensor.ndim}')
    if new_num_embeddings < 0:
        raise ValueError('new_num_embeddings must be non-negative')

    old_index_map = _normalize_optimizer_index_map(
        old_index_map,
        old_tensor.shape[0],
        context='old optimizer index map',
    )
    new_index_map = _normalize_optimizer_index_map(
        new_index_map,
        new_num_embeddings,
        context='new optimizer index map',
    )
    device = old_tensor.device
    new_tensor = torch.zeros(
        (new_num_embeddings, old_tensor.shape[1]),
        device=device,
        dtype=old_tensor.dtype,
    )

    old_index = old_index_map.index
    new_index = new_index_map.index
    old_node_ids = old_index_map['arange'].to_numpy(copy=False)
    new_node_ids = new_index_map['arange'].to_numpy(copy=False)
    for start in range(0, len(old_index), _SPARSE_OPTIMIZER_REMAP_CHUNK_SIZE):
        end = min(start + _SPARSE_OPTIMIZER_REMAP_CHUNK_SIZE, len(old_index))
        new_positions = new_index.get_indexer(old_index[start:end])
        matched = new_positions >= 0
        if not matched.any():
            continue
        old_idx = torch.from_numpy(old_node_ids[start:end][matched]).to(device)
        new_idx = torch.from_numpy(
            new_node_ids[new_positions[matched]]).to(device)
        new_tensor.index_copy_(0, new_idx, old_tensor.index_select(0, old_idx))

    return new_tensor


def _normalize_optimizer_index_map(
    index_map: pd.DataFrame,
    capacity: int,
    *,
    context: str,
) -> pd.DataFrame:
    if len(index_map.columns) != 1:
        raise ValueError(f'{context} must contain exactly one column')
    if index_map.index.hasnans:
        raise ValueError(f'{context} contains null primary keys')
    if not index_map.index.is_unique:
        raise ValueError(f'{context} contains duplicate primary keys')

    normalized = index_map.copy(deep=False)
    normalized.columns = ['arange']
    node_ids = normalized['arange']
    if node_ids.hasnans:
        raise ValueError(f'{context} contains null physical node IDs')
    if not pd.api.types.is_integer_dtype(node_ids.dtype):
        raise ValueError(f'{context} physical node IDs must be integral')
    if not node_ids.is_unique:
        raise ValueError(f'{context} contains duplicate physical node IDs')
    if len(node_ids) and int(node_ids.min()) < 0:
        raise ValueError(f'{context} contains negative physical node IDs')
    if len(node_ids) and int(node_ids.max()) >= capacity:
        raise ValueError(f'{context} physical node IDs exceed capacity '
                         f'{capacity}')
    normalized['arange'] = node_ids.astype(np.int64, copy=False)
    return normalized
