# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import copy
import gc
import glob
import logging
import math
import os
import os.path as osp
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable

import pandas as pd
import torch
from kumo_core.automl.error import (
    check_gpu_ready,
    construct_gpu_oom_message,
    construct_reduce_batch_msg,
)
from kumo_core.automl.optuna_searcher import TPESearcher
from kumo_core.automl.runtime import (
    AutoTrainingRuntime,
    apply_encoder_overrides,
    apply_na_strategy,
    resolve_warm_start,
    with_na_strategy,
)
from kumo_core.callbacks.stopper import TopKEarlyStopping
from kumo_core.embedding_models import require_encoder_models
from kumo_core.callbacks.train_progress_logger import (
    AutoTrainerProgress,
    AutoTrainingProgressLogger,
)
from kumo_core.instrumentation import TrainingInstrumentation, format_error
from kumo_core.io import filesystem
from kumo_core.io.json import load_json, save_json
from kumo_core.job_config import JobConfig, TrainStatus
from kumo_core.model_plan import ModelPlan, _is_inferred, _parse_hop_key
from kumo_core.resolved_task import ResolvedNumericalTask, ResolvedTaskGraph
from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
from kumo_core.train.shallow_rhs_link_gnn_module import MAX_RHS_SAMPLE_SIZE
from kumo_core.train.trainer import Trainer, load_encoders, save_encoders
from kumo_core.training.config import LinkPredModuleKwargs, LinkPredModuleType
from kumo_core.types import TaskType
from kumoapi.model_plan import (
    DistanceMeasureType,
    GraphTransformerModelPlan,
    LinkPredOutputType,
    MissingType,
    NumNeighborsConfig,
    OptimizerConfig,
    PastEncoderType,
    PruningConfig,
    RHSEmbeddingMode,
)
from kumoapi.train import TrainingStage
from kumo_core.encoders.config import EncodersConfig
from kumo_core.evaluation.suite import MetricStore
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
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
from kumo_core.modeling.gnn.positional import positional_encoding_resolver
from pytorch_lightning.callbacks import Callback
from torch_geometric import seed_everything
from torch_geometric.sampler import NumNeighbors

logger = logging.getLogger(__name__)
_eval_fn = {'min': min, 'max': max}


def _failed_trials_error_message(status: TrainStatus) -> str:
    message = "Failed to run any AutoML trial"
    trial_errors = [
        trial.stacktrace.rstrip().splitlines()[-1] for trial in status.trials
        if trial.stacktrace
    ]
    if trial_errors:
        message += f". Last trial error: {trial_errors[-1]}"
    return message


def _resolved_hop_edge_type(
    graph: ResolvedTaskGraph,
    src_table_name: str,
    src_col_name: str,
    dst_table_name: str,
    dst_col_name: str,
) -> tuple[str, str, str]:
    candidates = (
        (dst_table_name, f'rev_{src_col_name}', src_table_name),
        (dst_table_name, dst_col_name, src_table_name),
    )
    matches = [edge for edge in candidates if edge in graph.edge_types]
    if len(matches) == 1:
        return matches[0]

    # edge_types lists foreign keys as authored, with no reverse entries.
    if not matches and (src_table_name, src_col_name,
                        dst_table_name) in graph.edge_types:
        return (dst_table_name, f'rev_{src_col_name}', src_table_name)

    raise ValueError(
        f'hop edge must resolve exactly once for {src_table_name}.'
        f'{src_col_name}->{dst_table_name}.{dst_col_name}; got {matches}. '
        f'Tried {list(candidates)} and the authored foreign key '
        f'{(src_table_name, src_col_name, dst_table_name)} against edge types '
        f'{sorted(graph.edge_types)}')


RHS_FEAT_BYTES_THRESHOLD_RATIO = float(
    os.getenv('RHS_FEAT_BYTES_THRESHOLD_RATIO', '0.3'))


def infer_link_pred_module_type(
    task: Any,
    output_type: LinkPredOutputType,
) -> LinkPredModuleType:
    assert task.task_type.is_link_pred
    direct_map = {
        LinkPredOutputType.SHALLOW_RHS: LinkPredModuleType.SHALLOW_RHS,
        LinkPredOutputType.CONTEXT: LinkPredModuleType.HYBRID,
        LinkPredOutputType.TWO_TOWER: LinkPredModuleType.TWO_TOWER,
        LinkPredOutputType.PAIRWISE: LinkPredModuleType.ID,
    }
    if output_type in direct_map:
        return direct_map[output_type]
    if output_type == LinkPredOutputType.EMBEDDING:
        return (LinkPredModuleType.SHALLOW_RHS
                if task.is_temporal else LinkPredModuleType.TWO_TOWER)
    assert output_type == LinkPredOutputType.RANKING
    return (LinkPredModuleType.HYBRID
            if task.is_temporal else LinkPredModuleType.SHALLOW_RHS)


def construct_object_string_from_config(obj: Any) -> str:
    """Return the compact constructor representation used in model plans."""
    if isinstance(obj, dict):
        obj_fields = obj
        target = obj['_target_']
        if not isinstance(target, str) or not target:
            raise ValueError("Config '_target_' must be a non-empty string")
        obj_cls_name = target.rsplit('.', maxsplit=1)[-1]
    elif is_dataclass(obj):
        obj_fields = asdict(obj)
        for field in fields(obj):
            if not field.repr:
                del obj_fields[field.name]
        obj_cls_name = obj.__class__.__name__
    else:
        obj_fields = asdict(obj.config())
        obj_cls_name = obj.__class__.__name__

    parameters = []
    for key, value in obj_fields.items():
        if key.startswith('_'):
            continue
        if hasattr(obj, '_external_args') and key not in obj._external_args:
            continue
        if isinstance(value, Enum):
            value = value.value
        parameters.append(
            f"{key}='{value}'" if isinstance(value, str) else f'{key}={value}')
    return f"{obj_cls_name}({', '.join(parameters)})"


@dataclass
class TrialResults:
    name: str
    logdir: str
    result: float
    config: ModelPlan


class AutoTrainer:
    r"""Creates a KUMO trainer object with built-in hyperparameter
    optimization.

    Args:
        space: The search space.
        task: A :class:`~kumo.task.Task` object.
        runtime: Environment-specific task, encoder, and trainer capabilities.
        instrumentation: Explicit tracing and timing hooks.
        max_tries: The maximum number of trials to re-run an AutoML experiment
            due to a GPU OOM error.
        min_retry_batch_size: The minimum batch size to retry.
        checkpoint_path: If provided start autotrainer runs from this
            checkpoint.
            Also loads the encoders from this checkpoint. Usually this is the
            best checkpoint from another run `Trainer.best_ckpt`.
            The `space` object must be compatible with the checkpoint
            i.e has the same exact model architecture as the checkpoint.
        home_dir: The home directory used to resolve ``root_dir`` and to
            load/save :class:`TrainStatus`. If ``None``, the runtime supplies
            its default.
        trial_logger: Called with ``(trial, attempt)`` for each search trial;
            a returned Lightning logger (or list of them) also receives that
            trial's metrics.
    """
    def __init__(
        self,
        space: ModelPlan,
        task: ResolvedNumericalTask,
        runtime: AutoTrainingRuntime,
        instrumentation: TrainingInstrumentation,
        max_tries: int = 4,
        min_retry_batch_size: int = 16,
        checkpoint_path: str | None = None,
        home_dir: str | None = None,
        checkpoint_encoder_dir: str | None = None,
        trial_logger: Callable[[int, int], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(task, ResolvedNumericalTask):
            raise TypeError("AutoTrainer requires ResolvedNumericalTask")
        self.space = space
        self.task = task
        self.runtime = runtime
        self.instrumentation = instrumentation
        self.trial_logger = trial_logger
        self.checkpoint_path = checkpoint_path
        self.checkpoint_encoder_dir = checkpoint_encoder_dir
        self.max_tries = max_tries
        self.min_retry_batch_size = min_retry_batch_size
        # Resolve the process-global home dir once at the boundary, then
        # thread the concrete value through the instance. Production callers
        # pass an explicit ``home_dir``; direct test constructions fall back
        # to the global.
        self.home_dir = home_dir or runtime.default_home_dir()
        self.root_dir = osp.join(self.home_dir, 'models')

        self._raise_fake_trial_error_msg: str | None = kwargs.pop(
            'raise_fake_trial_error_msg', None)

        self.random_state_seed = kwargs.pop('random_state_seed', None)

        # TT-NG: when set, ``_train`` dispatches to
        # ``DiskgraphNestedTrainer`` per AutoML trial.
        self.diskgraph_task_dataset_bundle_dir: str | None = kwargs.pop(
            'diskgraph_task_dataset_bundle_dir', None)

        # Seed fs/gs as None: TT-NG always seeds them so consumers that
        # forward them as kwargs don't AttributeError. The ``hasattr``
        # guards in ``fit()`` / ``get_best_trainer`` then see them present.
        self.feature_store = None
        self.graph_store = None

        # AutoTune-related variables:
        self.tune_metric = space.training_job.tune_metric
        self.num_experiments = space.training_job.num_experiments
        self.max_epochs = space.optimization.max_epochs
        self.min_steps_per_epoch = space.optimization.min_steps_per_epoch
        self.max_steps_per_epoch = space.optimization.max_steps_per_epoch
        self.max_val_steps = space.optimization.max_val_steps
        self.max_test_steps = space.optimization.max_test_steps
        self.refit_trainval = space.training_job.refit_trainval
        self.refit_full = space.training_job.refit_full
        self.disable_compilation = space.training_job.disable_compilation
        if self.random_state_seed is None:
            self.random_state_seed = space.training_job.manual_seed

        self.lp_module_type: LinkPredModuleType | None = None
        if self.task.task_type.is_link_pred:
            self.lp_module_type = infer_link_pred_module_type(
                task, space.model_architecture.module)

        if self.random_state_seed is not None:
            seed_everything(self.random_state_seed)

        # Resolve the `inferred` target-edge fanout. The number itself is
        # derived from the data before the task is built -- by the monolith's
        # `infer_authoring_max_target_neighbors_per_entity` or portable
        # training's `infer_max_target_neighbors_per_entity`, both over the
        # one degree policy in `kumo_core.automl.target_fanout`.
        inferred_target_neighbors = task.sampling.max_target_neighbors_per_entity
        for num_neighbors in space.neighbor_sampling.num_neighbors:
            if (isinstance(num_neighbors, NumNeighborsConfig)
                    and num_neighbors.num_hops() > 0):
                for k, v in num_neighbors.hop1.__pydantic_extra__.items():
                    if _is_inferred(v):
                        num_neighbors.hop1[k] = inferred_target_neighbors

        # Backward compatibility: Also handle
        # `max_target_neighbors_per_entity`
        max_targets = space.neighbor_sampling.max_target_neighbors_per_entity
        if isinstance(max_targets, list):
            for i, max_target in enumerate(max_targets):
                if _is_inferred(max_target):
                    assert isinstance(inferred_target_neighbors, int)
                    max_targets[i] = inferred_target_neighbors

        # Clean leftover shared memory files. This is necessary as previous
        # runs of AutoML may have exited prematurely without properly cleaning
        # themselves up (e.g. via SIGKILL, a segmentation fault, an internal
        # os.exit, or something else entirely). In these caes, the torch shared
        # memory manager will not properly clean up leftover files, thus
        # leaking shared memory. Since we do not support multiple concurrent
        # invocations of AutoTrainer, it is safe to simply check and remove
        # any torch_* shm files upon each invocation here.
        #
        # TODO(manan): this is not a permanent solution. A much better one
        # would be our own cleanup daemon.
        leaked_files = glob.glob(r"/dev/shm/torch_[0-9]+_[0-9]+")
        if len(leaked_files) > 0:
            logger.info(
                f"Found shared memory files {leaked_files} that have not "
                "properly been cleaned up. Cleaning up now...")
            for f in leaked_files:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(f)

    @property
    def val_tune_metric(self) -> str:
        return TrainingStage.VAL.add_stage_prefix(self.tune_metric)

    # Core ####################################################################

    def _get_encoder(
        self,
        cfg: ModelPlan,
    ) -> HeteroTensorFrameEncoder:
        arch_cfg = cfg.model_architecture
        return HeteroTensorFrameEncoder(
            act=arch_cfg.activation[0],
            use_seq_id=arch_cfg.use_seq_id[0],
        )

    def _get_model(self, cfg: ModelPlan) -> HeteroGNN | GraphTransformer:
        arch_cfg = cfg.model_architecture
        model_cfg = cfg.model_architecture.model[0]

        if isinstance(cfg.neighbor_sampling.num_neighbors[0], list):
            num_hops = len(cfg.neighbor_sampling.num_neighbors[0])
        else:
            assert isinstance(cfg.neighbor_sampling.num_neighbors[0],
                              NumNeighborsConfig)
            num_hops = cfg.neighbor_sampling.num_neighbors[0].num_hops()

        if isinstance(model_cfg, GraphTransformerModelPlan):
            positional_encodings = []
            if isinstance(model_cfg.positional_encodings, list):
                positional_encodings = [
                    positional_encoding_resolver(
                        pos,
                        channels=model_cfg.channels[0],
                        num_layers=num_hops,
                        num_hops=num_hops,
                        edge_types=self.task.graph.edge_types,
                    ) for pos in model_cfg.positional_encodings[0]
                ]

            return GraphTransformer(
                channels=model_cfg.channels[0],
                num_heads=model_cfg.num_heads[0],
                num_layers=model_cfg.num_layers[0],
                dropout=model_cfg.dropout[0],
                act=arch_cfg.activation[0],
                positional_encodings=positional_encodings,
            )

        return HeteroGNN(
            channels=model_cfg.channels[0],
            num_mp_layers=num_hops,
            num_pre_mp_layers=arch_cfg.num_pre_message_passing_layers[0],
            aggr=[aggr for aggr in model_cfg.aggregation[0]],
            dropout=model_cfg.dropout[0],
            act=arch_cfg.activation[0],
            norm=arch_cfg.normalization[0],
        )

    def _get_head(
        self,
        cfg: ModelPlan,
    ) -> NodePredHead | LinkPredHead:
        arch_cfg = cfg.model_architecture

        prediction_time_encodings = arch_cfg.prediction_time_encodings
        if isinstance(prediction_time_encodings, list):
            prediction_time_encodings = prediction_time_encodings[0]
        else:
            prediction_time_encodings = False

        if self.task.task_type.is_link_pred:
            return LinkPredHead(
                embedding_dim=arch_cfg.output_embedding_dim[0],
                num_layers=arch_cfg.num_post_message_passing_layers[0],
                act=arch_cfg.activation[0],
                norm=arch_cfg.normalization[0],
            )

        elif self.task.task_type == TaskType.FORECASTING:
            return ForecastingHead(
                pos_enc_channels=arch_cfg.model[0].channels[0],
                num_layers=arch_cfg.num_post_message_passing_layers[0],
                act=arch_cfg.activation[0],
                norm=arch_cfg.normalization[0],
                prediction_time_encodings=prediction_time_encodings,
                handle_new_entities=arch_cfg.handle_new_entities,
            )

        return NodePredHead(
            num_layers=arch_cfg.num_post_message_passing_layers[0],
            act=arch_cfg.activation[0],
            norm=arch_cfg.normalization[0],
            prediction_time_encodings=prediction_time_encodings,
        )

    def _get_past_encoder(self, cfg: ModelPlan) -> SequenceEncoder:
        if (self.task.task_type == TaskType.FORECASTING
                and self.task.lag_timesteps > 0):

            arch_cfg = cfg.model_architecture
            act = arch_cfg.activation[0]
            norm = arch_cfg.normalization[0]

            if arch_cfg.past_encoder[0] == PastEncoderType.DECOMPOSED:
                return DecomposedSequenceEncoder()
            if arch_cfg.past_encoder[0] == PastEncoderType.NORMALIZED:
                return NormalizedSequenceEncoder()
            if arch_cfg.past_encoder[0] == PastEncoderType.MLP:
                return MLPSequenceEncoder(act=act, norm=norm)
            if arch_cfg.past_encoder[0] == PastEncoderType.TRANSFORMER:
                return TransformerSequenceEncoder(
                    num_layers=2,
                    num_heads=4,
                    act=act,
                )

        return None

    def _get_year_over_year_encoder(self) -> YearOverYearEncoder | None:
        """Encode year-over-year labels whenever the train table has them.

        The model plan only ever *requests* the feature, and the request is
        not what got built: an unset request compiles to off, and only a
        forecasting train table is given the column at all. The task carries
        what materialization actually emitted, so reading it here keeps the
        encoder and the column it consumes to a single source of truth.
        """
        if not self.task.has_year_over_year_labels:
            return None
        return YearOverYearEncoder()

    def _train(
        self,
        cfg: ModelPlan,
        metrics_client_job_name: str,
        pruning: Callback | None = None,
        status: TrainStatus | None = None,
        trial: int | None = None,
        attempt: int | None = None,
    ) -> list[float]:
        r"""Config-driven training function.

        Args:
            cfg: The config object to be executed.
            pruning: A stopper can be used to early stop a trial based on
                previous trials.
        """
        # Define loader parameters:
        default_num_neighbors = []
        num_neighbors_per_edge_type = {}

        sampling_opts = cfg.neighbor_sampling
        num_neighbors = sampling_opts.num_neighbors[0]

        if isinstance(num_neighbors, list):  # Backward compatibility
            default_num_neighbors = num_neighbors

            # In case `num_neighbors` is passed as a list, we also need to
            # parse `max_target_neighbors_per_entity`
            target_neighbors = sampling_opts.max_target_neighbors_per_entity
            if isinstance(target_neighbors, list):
                target_to_entity_paths = (
                    self.task.unqualified_target_to_entity_paths)

                if len(num_neighbors) > 0 and len(target_to_entity_paths) > 0:
                    target_to_entity_path = target_to_entity_paths[0]
                    assert len(target_to_entity_path) == 1
                    for edge_type in target_to_entity_path:
                        num_neighbors_per_edge_type[edge_type] = (
                            [target_neighbors[0]] + default_num_neighbors[1:])

        else:
            assert isinstance(num_neighbors, NumNeighborsConfig)

            default_num_neighbors = [0] * num_neighbors.num_hops()

            for i, hop_cfg in num_neighbors.to_dict().items():
                default_num_neighbors[i] = hop_cfg.default

            for i, hop_cfg in num_neighbors.to_dict().items():
                for key, value in hop_cfg.__pydantic_extra__.items():
                    assert isinstance(value, int)
                    (
                        src_table_name,
                        src_col_name,
                        dst_table_name,
                        dst_col_name,
                    ) = _parse_hop_key(key)

                    edge_type = _resolved_hop_edge_type(
                        self.task.graph,
                        src_table_name,
                        src_col_name,
                        dst_table_name,
                        dst_col_name,
                    )

                    if edge_type not in num_neighbors_per_edge_type:
                        values = copy.copy(default_num_neighbors)
                    else:
                        values = num_neighbors_per_edge_type[edge_type]

                    values[i] = value
                    num_neighbors_per_edge_type[edge_type] = values

        # Define optimization parameters:
        sampler_kwargs = dict(
            num_neighbors=NumNeighbors(values=num_neighbors_per_edge_type,
                                       default=default_num_neighbors),
            adaptive_sampling=cfg.neighbor_sampling.adaptive_sampling,
        )

        loader_kwargs = dict(batch_size=cfg.optimization.batch_size[0])

        optimizer = cfg.optimization.optimizer[0]
        if isinstance(optimizer, MissingType):
            optimizer = 'Adam'
        optimizer_kwargs = dict(
            lr=cfg.optimization.base_lr[0],
            weight_decay=cfg.optimization.weight_decay[0],
        )

        lookup_optimizer_kwargs: dict[str, Any] = None
        if self.task.task_type.is_link_pred:
            if isinstance(cfg.optimization.lookup_optimization, list):
                lookup_optimization = cfg.optimization.lookup_optimization[0]
                if isinstance(lookup_optimization, OptimizerConfig):
                    lookup_optimizer_kwargs = dict(
                        optimizer=lookup_optimization.optimizer,
                        lr=lookup_optimization.base_lr,
                    )

        loss = cfg.optimization.loss[0]
        if not is_dataclass(cfg.optimization.loss[0]):
            loss = f'{loss.value}_loss'

        # Define scheduler parameters:
        lr_scheduler_cfg = cfg.optimization.lr_scheduler[0]
        if lr_scheduler_cfg is not None:
            lr_scheduler = lr_scheduler_cfg.name
            lr_scheduler_interval = lr_scheduler_cfg.interval
            lr_scheduler_kwargs = lr_scheduler_cfg.kwargs
        else:
            lr_scheduler = lr_scheduler_interval = None
            lr_scheduler_kwargs = {}

        early_stopping_cfg = cfg.optimization.early_stopping[0]
        if early_stopping_cfg is not None:
            early_stopping = True
            early_stopping_kwargs = dict(
                min_delta=early_stopping_cfg.min_delta,
                patience=early_stopping_cfg.patience,
            )
        else:
            early_stopping = False
            early_stopping_kwargs = {}

        majority_sampling_ratio = None
        if isinstance(cfg.optimization.majority_sampling_ratio, list):
            majority_sampling_ratio = (
                cfg.optimization.majority_sampling_ratio[0])

        weight_mode = None  # Dummy.
        if isinstance(cfg.optimization.weight_mode, list):
            weight_mode = cfg.optimization.weight_mode[0]

        target_fraction = 1.0
        if isinstance(cfg.optimization.target_fraction, list):
            target_fraction = cfg.optimization.target_fraction[0]

        overlap_ratio = 1.0
        if isinstance(cfg.optimization.overlap_ratio, list):
            overlap_ratio = cfg.optimization.overlap_ratio[0]

        loss_coeff = cfg.model_architecture.ranking_embedding_loss_coeff
        lp_hybrid_emb_loss_coeff = 0.0
        if isinstance(loss_coeff, list):
            lp_hybrid_emb_loss_coeff = loss_coeff[0]

        emb_l2_reg_coeff = cfg.model_architecture.embedding_norm_reg_coeff
        if isinstance(emb_l2_reg_coeff, list):
            emb_l2_reg_coeff = emb_l2_reg_coeff[0]

        rhs_emb_mode = RHSEmbeddingMode.FUSION
        if isinstance(cfg.model_architecture.target_embedding_mode, list):
            rhs_emb_mode = cfg.model_architecture.target_embedding_mode[0]

        distance_measure = 'dot_product'
        if isinstance(cfg.model_architecture.distance_measure[0],
                      DistanceMeasureType):
            distance_measure = cfg.model_architecture.distance_measure[0]

        # Override to feature mode if handle_new_target_entities is True.
        if (cfg.model_architecture.handle_new_target_entities
                and not rhs_emb_mode.only_use_rhs_feature):
            rhs_emb_mode = RHSEmbeddingMode.FEATURE

        # Create Trainer instance:
        # NOTE The output directory structure will look as follows:
        # {self.root_dir}/{self.task.name}/automl/{version_*}
        # DiskgraphNestedTrainer omits the redundant task-name directory.
        module_kwargs = None
        if self.task.task_type.is_link_pred:
            module_kwargs = LinkPredModuleKwargs(
                module=self.lp_module_type,
                emb_l2_reg_coeff=emb_l2_reg_coeff,
                hybrid_emb_loss_coeff=lp_hybrid_emb_loss_coeff,
                rhs_emb_mode=rhs_emb_mode,
                target_fraction=target_fraction,
                overlap_ratio=overlap_ratio,
            )

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.mem_get_info()[1]
                if (status is not None
                        and any([trial.oomed for trial in status.trials])):
                    module_kwargs.rhs_sample_size = MAX_RHS_SAMPLE_SIZE // 2
                    module_kwargs.rhs_feat_bytes_threshold = int(
                        RHS_FEAT_BYTES_THRESHOLD_RATIO / 2 * gpu_mem)
                else:
                    module_kwargs.rhs_feat_bytes_threshold = int(
                        RHS_FEAT_BYTES_THRESHOLD_RATIO * gpu_mem)

        year_over_year_encoder = self._get_year_over_year_encoder()
        unified_cfg = cfg.unify()

        # TODO Pass in `ModelArchitecturePlan` once migration is complete.
        encoder = self._get_encoder(unified_cfg)
        model = self._get_model(unified_cfg)
        head = self._get_head(unified_cfg)
        past_encoder = self._get_past_encoder(unified_cfg)

        extra_trainer_kwargs: dict[str, Any] = {
            'random_state_seed': self.random_state_seed,
        }
        assert self.diskgraph_task_dataset_bundle_dir is not None, (
            "TT-NG requires a diskgraph task-dataset bundle")
        extra_trainer_kwargs['diskgraph_task_dataset_bundle_dir'] = (
            self.diskgraph_task_dataset_bundle_dir)
        if (self.trial_logger is not None and trial is not None
                and attempt is not None):
            pl_logger = self.trial_logger(trial, attempt)
            if pl_logger is not None:
                extra_trainer_kwargs['logger'] = pl_logger

        with self.instrumentation.timing(
                "auto_trainer.train.construct_trainer"):
            trainer = DiskgraphNestedTrainer(
                self.task.to_task_config(),
                # Instance-specific arguments:
                encoder=encoder,
                model=model,
                head=head,
                past_encoder=past_encoder,
                year_over_year_encoder=year_over_year_encoder,
                root_dir=self.log_dir,
                module_kwargs=module_kwargs,
                track_version=True,
                sampler_kwargs=sampler_kwargs,
                loader_kwargs=loader_kwargs,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                lookup_optimizer_kwargs=lookup_optimizer_kwargs,
                loss=loss,
                lr_scheduler=lr_scheduler,
                lr_scheduler_interval=lr_scheduler_interval,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                early_stopping=early_stopping,
                early_stopping_kwargs=early_stopping_kwargs,
                majority_sampling_ratio=majority_sampling_ratio,
                weight_mode=weight_mode,
                distance_measure=distance_measure,
                # Space-specific arguments:
                tune_metric=self.tune_metric,
                max_epochs=self.max_epochs,
                min_steps_per_epoch=self.min_steps_per_epoch,
                max_steps_per_epoch=self.max_steps_per_epoch,
                max_val_steps=self.max_val_steps,
                max_test_steps=self.max_test_steps,
                # General arguments:
                metrics_client_job_name=metrics_client_job_name,
                raise_fake_trial_error_msg=self._raise_fake_trial_error_msg,
                feature_store=self.feature_store,
                graph_store=self.graph_store,
                encoders=self.encoders,
                enable_compile=not self.disable_compilation,
                trial=status.trials[-1] if status is not None else None,
                checkpoint_path=self.checkpoint_path,
                **self.runtime.trainer_runtime_kwargs(),
                **extra_trainer_kwargs,
            )

        with self.instrumentation.timing("auto_trainer.train.trainer_init"):
            trainer.init()

        if pruning is not None:
            trainer.pl_trainer.callbacks.append(pruning)

        # Add progress logging callback:
        self.training_progress_logger.set_current_trainer(trainer, trial)
        trainer.pl_trainer.callbacks.append(self.training_progress_logger)

        # Save AutoTrainer YAML (Trainer YAML will be saved within
        # Trainer itself). The AutoTrainer configuration contains relevant
        # fields for the construction of the encoder/model/head as well as
        # relevant learning parameters. The trainer configuration contains
        # all relevant parameters for a particular trainer run:
        with self.instrumentation.timing("auto_trainer.train.save_model_plan"):
            save_json(cfg, osp.join(trainer.log_dir, 'model_plan.json'))

        with self.instrumentation.timing("auto_trainer.train.fit"):
            trainer.fit()

        with self.instrumentation.timing("auto_trainer.train.validate"):
            eval_output = trainer.validate()
        with self.instrumentation.timing("auto_trainer.train.save_eval_output"):
            eval_output.save_all(
                root_dir=trainer.log_dir,
                stage=TrainingStage.VAL,
                task=self.task,
            )

        # Prevent keeping old trainer state around.
        self.training_progress_logger.set_current_trainer(None)
        del self.training_progress_logger.log
        del self.training_progress_logger.log_dict
        trainer.pl_trainer.callbacks.remove(self.training_progress_logger)

        if pruning is not None:
            # Prevent keeping old trainer state around.
            del pruning.log
            del pruning.log_dict
            trainer.pl_trainer.callbacks.remove(pruning)

        progress = trainer.training_progress_logger.progress
        val_performance = [
            epoch_metric.validation_metrics[self.val_tune_metric]
            for epoch_metric in progress.metrics.values()
        ]

        return val_performance

    def run(
        self,
        init_best_trainer: bool = True,
        debug: bool = False,
    ) -> Trainer:
        r"""Public API to execute an AutoTrainer search."""
        # Verify GPU is actually usable before starting training.
        # This catches NVML initialization issues that nvidia-smi doesn't detect.
        with self.instrumentation.span("auto_trainer.run"):
            return self._run(init_best_trainer=init_best_trainer, debug=debug)

    def _make_pruning_callback(self) -> 'TopKEarlyStopping | None':
        r"""Builds the cross-trial pruning callback from
        ``self.space.training_job.pruning``, or :obj:`None` if pruning is
        disabled.

        Used both when actually running the search (:meth:`_run`, where the
        same instance is shared across trials to accumulate top-k scores)
        and when reloading the winning trial's :class:`Trainer` in
        :meth:`get_best_trainer` (where the fresh instance never sees other
        trials, but only needs a matching ``monitor``/``mode`` for Lightning
        to recognize it and restore that trial's saved callback state from
        its checkpoint -- ``EarlyStopping``-family callbacks are no-ops
        outside of ``fit()`` anyway, so the freshly-built instance never
        actually runs any stopping logic).
        """
        pruning_cfg = self.space.training_job.pruning
        if not isinstance(pruning_cfg, PruningConfig):
            return None
        return TopKEarlyStopping(
            monitor=self.val_tune_metric,
            min_epochs=pruning_cfg.min_epochs,
            k=pruning_cfg.k,
            min_delta=pruning_cfg.min_delta,
            patience=pruning_cfg.patience,
            mode=MetricStore.tune_mode(self.tune_metric),
        )

    def _run(
        self,
        init_best_trainer: bool = True,
        debug: bool = False,
    ) -> Trainer:
        with self.instrumentation.timing("auto_trainer.run.check_gpu_ready"):
            check_gpu_ready()
        # A missing word-embedding model fails the job, not each trial.
        _ = self.encoders

        # TT-NG reads from the bundle, so no legacy fs/gs construction:
        # ``Task.load`` is config-only and the TT-NG trainer ignores fs/gs
        # anyway.
        with self.instrumentation.timing("auto_trainer.run.load_status"):
            status = TrainStatus.load(self.home_dir, debug)
        # If a previous AutoML run failed because of system interruption
        # (e.g., a CPU OOM), we will re-run `kumo-auto` and re-start training.
        # In this case, we try to prevent any further CPU OOMs by capping the
        # space of batch sizes. If there exists successful previous trials, we
        # cap by their maximum successful batch size:
        max_batch_size = status.max_successful_batch_size()
        if max_batch_size is None and status.min_batch_size() is not None:
            # If no previous run was successful, we divide the previous minimum
            # batch size by half and try again:
            max_batch_size = status.min_batch_size() // 2
            max_batch_size = max(max_batch_size, self.min_retry_batch_size)

        if max_batch_size is not None:
            self.space.optimization.batch_size = [
                min(batch_size, max_batch_size)
                for batch_size in self.space.optimization.batch_size
            ]
            logger.info(f"Capping maximum batch size to {max_batch_size} due "
                        f"to previously encountered system interruption")

        # If previous AutoML runs succeeded, we just need to train for the
        # number of remaining experiments:
        start = 0 if debug else status.num_successful_trials()
        end = self.num_experiments

        logger.info(f"Starting AutoML engine with "
                    f"{end - start} experiments...")

        with self.instrumentation.timing(
                "auto_trainer.run.initialize_searcher"):
            self.searcher = TPESearcher(
                self.space,
                n_startup_trials=max(4, math.ceil(0.75 * self.num_experiments)),
                seed=self.random_state_seed,
            )
        logger.info(f"Initialized '{self.searcher}'")

        self.training_progress_logger = AutoTrainingProgressLogger(
            self.log_dir, self.num_experiments, on_event=self._emit)

        tune_mode = MetricStore.tune_mode(self.tune_metric)
        metrics_job_name = self.runtime.metrics_job_name()
        self.instrumentation.record_metrics_job_name(metrics_job_name)

        pruning = self._make_pruning_callback()

        num_prev_trials = len(status.trials)
        failed_trials = []
        for i in range(start, end):
            # TODO Inherit previous trial configuration for `Searcher` in case
            # of `kumo-auto` re-start.
            with self.instrumentation.timing(
                    "auto_trainer.run.searcher_suggest", trial=i + 1):
                cfg = self.searcher.suggest()

            # Set `num_experiments` to one for individual model plans:
            cfg.training_job.num_experiments = 1

            for j in range(self.max_tries):
                try:
                    logger.info(f"Starting AutoML experiment "
                                f"{i + 1}/{self.num_experiments}")

                    self._emit('trial_started', trial=i + 1, params={
                        **self.searcher.sampled_params,
                        'attempt': j + 1,
                    })
                    batch_size = cfg.optimization.batch_size[0]
                    with self.instrumentation.timing(
                            "auto_trainer.run.status_add_trial", trial=i + 1):
                        status.add_trial(batch_size)
                        status.save(self.home_dir)

                    trial_start = time.monotonic()
                    with self.instrumentation.timing(
                            "auto_trainer.run.train_trial", trial=i + 1):
                        val_performance = self._train(
                            cfg=cfg,
                            metrics_client_job_name=metrics_job_name,
                            pruning=pruning,
                            status=status,
                            trial=i + 1,
                            attempt=j + 1,
                        )
                    trial_seconds = time.monotonic() - trial_start
                    best_performance = _eval_fn[tune_mode](val_performance)
                    logger.info(f"AutoML experiment "
                                f"{i + 1}/{self.num_experiments} finished "
                                f"with validation performance: "
                                f"{self.tune_metric}={best_performance}")
                    with self.instrumentation.timing(
                            "auto_trainer.run.searcher_update", trial=i + 1):
                        self.searcher.update(cfg, performance=val_performance)

                    status.trials[-1].succeed = True
                    if j > 0:
                        status.warning_msg = construct_reduce_batch_msg()
                    with self.instrumentation.timing(
                            "auto_trainer.run.status_save_success",
                            trial=i + 1):
                        status.save(self.home_dir)
                    self._emit('trial_ended', trial=i + 1, metrics={
                        self.val_tune_metric: best_performance,
                        'seconds': trial_seconds,
                    }, params={
                        'status': 'succeeded',
                        'attempt': j + 1,
                    })
                    break

                except RuntimeError as e:
                    self._emit('trial_ended', trial=i + 1, params={
                        'status': 'failed',
                        'attempt': j + 1,
                        'error': format_error(e),
                    })
                    stacktrace = ''.join(
                        traceback.format_exception(e, e, e.__traceback__))
                    logger.warning(stacktrace)

                    status.trials[-1].stacktrace = stacktrace
                    status.save(self.home_dir)

                    if "out of memory" in str(stacktrace):
                        cfg = self._handle_oom_retry(cfg, trial=i, try_idx=j)
                        if cfg is None or j == self.max_tries - 1:
                            # Cannot reduce batch size anymore
                            failed_trials.append(status.trials[-1])
                            break
                    elif "Unable to locate credentials" in str(stacktrace):
                        # Raise the unable to locate credentials exception.
                        # It will be catch by kumo-auto and retry from the k8s
                        # container level.
                        raise e
                    elif ("Error(s) in loading state_dict" in str(stacktrace)
                          and self.checkpoint_path):
                        # Raise an error. The caller is expected to provide a
                        # `space` object that is compatible with the checkpoint.
                        raise e
                    else:
                        logger.warning(f"Skipping AutoML experiment "
                                       f"{i + 1}/{self.num_experiments}.")
                        break

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        if len(failed_trials) > 0:
            status.error_msg = construct_gpu_oom_message(
                num_experiments=self.num_experiments,
                failed_trials=failed_trials,
                task_type=self.task.task_type,
            )
            status.save(self.home_dir)
            if len(failed_trials) == len(status.trials[num_prev_trials:]):
                logger.warning("Failed to run any AutoML trial in this run")
            else:
                logger.warning(f"Failed {len(failed_trials)} AutoML trials")

        if not any(t.succeed for t in status.trials):
            raise RuntimeError(_failed_trials_error_message(status))

        with self.instrumentation.timing("auto_trainer.run.get_best_trainer",
                                         should_init=init_best_trainer):
            return self.get_best_trainer(should_init=init_best_trainer)

    def _emit(self, name: str, **fields: Any) -> None:
        event = getattr(self.instrumentation, 'event', None)
        if event is None:
            return
        try:
            event(name, **fields)
        except Exception:
            logger.warning(f"Could not report progress event '{name}'",
                           exc_info=True)

    def _handle_oom_retry(
        self,
        cfg: ModelPlan,
        trial: int,
        try_idx: int,
    ) -> ModelPlan | None:
        r"""Handles a CUDA OOM exception that occurred during training
        by reducing the batch size and re-trying."""

        msg = (f"AutoML experiment {trial + 1}/{self.num_experiments} "
               f"failed due to an CUDA OOM error after "
               f"{try_idx + 1}/{self.max_tries} tries")

        if try_idx < self.max_tries:
            batch_size = cfg.optimization.batch_size[0] // 2
            if batch_size < self.min_retry_batch_size:
                logger.warning(f"{msg}. Cannot re-run by reducing the batch "
                               f"size to {batch_size}. The minimum retry "
                               f"batch size is {self.min_retry_batch_size}.")
                return None

            cfg.optimization.batch_size = [batch_size]
            logger.warning(f"{msg}. Trying to re-run by "
                           f"reducing the batch size by half.")

        elif try_idx == self.max_tries:
            logger.warning(f"{msg}. Giving up.")

        return cfg

    # Properties ##############################################################
    @property
    def log_dir(self) -> str:
        r"""Returns the log directory of the AutoTrainer."""
        return osp.join(
            self.root_dir,  # Root dir (aligned w/Trainer)
            self.task.name,  # Task name (aligned w/Trainer)
            'automl',  # New sub-directory.
        )

    @property
    def encoders(self) -> EncodersConfig:
        if getattr(self, '_encoders', None) is None:
            encoders = self._resolve_encoders()
            store = self.runtime.trainer_runtime_kwargs().get(
                'embedding_models')
            if require_encoder_models(encoders, store):
                # Record the identities of the text embedding models.
                save_encoders(self.log_dir, encoders)
            self._encoders = encoders
        return self._encoders

    def _resolve_encoders(self) -> EncodersConfig:
        if self.checkpoint_path is not None:
            checkpoint_dir = osp.dirname(self.checkpoint_path)
            # Go up one level to get the automl directory
            # (checkpoint_dir is '.../automl/version_X/checkpoint/')
            automl_dir = (self.checkpoint_encoder_dir
                          or osp.dirname(osp.dirname(checkpoint_dir)))
            encoders = load_encoders(automl_dir)
            if encoders is None:
                raise RuntimeError(f"Checkpoint is set to {checkpoint_dir}. "
                                   f"But encoders not found in {automl_dir}")
            logger.info(f"Copying encoders.json from '{automl_dir}' "
                        f"to '{self.log_dir}'.")
            save_encoders(self.log_dir, encoders)
            return encoders

        encoders = load_encoders(self.log_dir)
        if encoders is None:
            # collect_encoders is fairly expensive, so we only do it once.
            encoders = self.runtime.collect_encoders(self.task)
        processing = self.space.column_processing
        overrides = processing.get_encoder_overrides(
            processing.encoder_overrides)
        apply_na_strategy(self.task, encoders, processing.na_strategy,
                          overridden=overrides.keys())
        apply_encoder_overrides(self.task, encoders, overrides)
        save_encoders(self.log_dir, encoders)

        return encoders

    @property
    def training_progress_file_path(self) -> str:
        r"""Returns the path to the training progress JSON file."""
        return osp.join(self.log_dir, AutoTrainingProgressLogger.FILE_NAME)

    # Trainer Utilities #######################################################

    def get_best_trials(self, k: int) -> list[TrialResults]:
        r"""Obtains the top-k trials from an experiment analysis."""
        run_dirs = filesystem.listdir(self.log_dir)
        trials = []
        for run_dir in run_dirs:
            ckpt_dir = osp.join(run_dir, 'checkpoint')
            if not filesystem.exists(ckpt_dir):
                continue
            paths = filesystem.listdir(ckpt_dir)
            paths = [path for path in paths if path.endswith('.ckpt')]
            if len(paths) == 0:
                continue
            name = os.path.basename(os.path.normpath(run_dir))
            try:
                df = pd.read_csv(osp.join(run_dir, "val_result.csv"))

                result = df[self.val_tune_metric].values[0]
                if pd.isna(result):
                    continue

                path = osp.join(run_dir, 'model_plan.json')
                cfg = load_json(path, ModelPlan)

                trial = TrialResults(name, run_dir, result, cfg)
                trials.append(trial)
            except FileNotFoundError:
                logger.info(f"Skip reading existing results from '{run_dir}' "
                            f"since it was not complete.")

        reverse = MetricStore.tune_mode(self.tune_metric) == 'max'
        trials = sorted(trials, key=lambda d: d.result, reverse=reverse)

        if len(trials) == 0:
            raise RuntimeError("No valid trials found")
        elif len(trials) < k:
            logger.warning(f"Expected {k} valid trials (got {len(trials)})")

        return trials[:k]

    def get_best_trial(self) -> TrialResults:
        r"""Obtains the best trial from an experiment analysis."""
        return self.get_best_trials(k=1)[0]

    def get_best_trainer(self, should_init: bool = True) -> Trainer:
        r"""Get the best :class:`Trainer` from the :class:`AutoTrainer` search.
        """
        best_trial = self.get_best_trial()
        version = int(best_trial.name[len('version_'):])
        # ``random_state_seed`` is set in ``__init__``; tolerate stubbed
        # instances (e.g. unit tests) that bypass it.
        random_state_seed = getattr(self, 'random_state_seed', None)

        # On TT-NG, load through the subclass so ``trainer.init()``
        # resolves to the bundle-driven init (no fs/gs reads).
        assert self.diskgraph_task_dataset_bundle_dir is not None, (
            "TT-NG requires a diskgraph task-dataset bundle")
        trainer = DiskgraphNestedTrainer.load(
            self.task.to_task_config(),
            self.log_dir,
            version,
            **self.runtime.trainer_runtime_kwargs(),
            encoders=self.encoders,
            diskgraph_task_dataset_bundle_dir=(
                self.diskgraph_task_dataset_bundle_dir),
            random_state_seed=random_state_seed,
        )
        if should_init:
            trainer.init()
            # Reattach the pruning callback so its state_key matches what
            # this trial's checkpoint recorded (it was present, appended
            # post-`create_pl_trainer()`, during the actual search run) --
            # otherwise Lightning warns that ckpt_path callback state has
            # nowhere to restore to whenever this reloaded trainer later
            # calls test()/validate()/predict() with ckpt_path set. See
            # `_make_pruning_callback` for why the fresh instance is safe
            # despite having no cross-trial history of its own.
            pruning = self._make_pruning_callback()
            if pruning is not None:
                trainer.pl_trainer.callbacks.append(pruning)
        return trainer

    @staticmethod
    def get_training_progress(log_dir: str) -> AutoTrainerProgress | None:
        path = osp.join(log_dir, AutoTrainingProgressLogger.FILE_NAME)

        ret = load_json(path, AutoTrainerProgress, default=None,
                        raise_on_missing=False)
        if ret is None:
            logger.warning("AutoTrainer progress file not found at %s", path)
        return ret

    # Predictive Query ########################################################

    @classmethod
    def parse_pquery(
        cls,
        pquery: Any,
        model_plan: ModelPlan,
        train_table_path: str | None = None,
        job_id: str | None = None,
        job_dir: str | None = None,
        ignore_encoders: bool = False,
        weight_col_name: str | None = None,
        train_table_name: str | None = None,
    ) -> tuple[Any, ModelPlan]:
        r"""Parse the predictive query into a :class:`Task` object and a search
        space consumed by an :class:`AutoTrainer`.

        Args:
            pquery: The predictive query.
            plan: The model plan.
            train_table_path: Path where the training table is located.
            job_id: Job id for retrieving table table stats.
            job_dir: Directory for storing training-job specific artifacts such
                as split table stats. Sometimes known as :obj:`home_dir`.
            weight_col_name: Name of the weight column in the training table.
        """
        # Split Inferral ##################################################
        train_table_opts = model_plan.training_table_generation
        if weight_col_name is None and train_table_opts.weight_col is not None:
            weight_col_name = 'WEIGHT'

        # Parsing logic
        if pquery.split is None or is_dataclass(pquery.split):
            # PQueryV2 should already have its split set at init
            assert False

        task = pquery.create_task(
            train_table_path=train_table_path,
            task_path_override=train_table_opts.task_path_override,
            job_id=job_id,
            job_dir=job_dir,
            ignore_encoders=ignore_encoders,
            weight_col_name=weight_col_name,
            train_table_name=train_table_name,
        )

        if _is_inferred(train_table_opts.split):
            split_repr = construct_object_string_from_config(
                task.train_table.split)
            train_table_opts.split = split_repr

        # Metric Validation ###############################################
        task._metrics = model_plan.training_job.metrics
        if model_plan.training_job.tune_metric not in task._metrics:
            model_plan.training_job.tune_metric = task._metrics[0]

        # Applying Encoder Overrides ######################################
        na_strategy = model_plan.column_processing.na_strategy
        for table in task.tables:
            for col in table.cols:
                updated = with_na_strategy(col.stype, col.encoder, na_strategy)
                if updated is not None:
                    col.encoder = updated

        encoder_dict = model_plan.column_processing.get_encoder_overrides(
            model_plan.column_processing.encoder_overrides)
        for key, encoder in encoder_dict.items():
            table_name, col_name = key.rsplit('.', maxsplit=1)

            try:
                table = task[table_name]
            except Exception:
                warnings.warn(f"'{table_name}' is not a valid table name. "
                              f"Skipping encoder override.")
                continue

            try:
                table[col_name].encoder = encoder
            except Exception:
                warnings.warn(f"'{col_name}' is not a valid column name "
                              f"for table '{table_name}'. "
                              f"Skipping encoder override.")
                continue

        return task, model_plan

    @classmethod
    def from_pquery(
        cls,
        pquery: Any,
        model_plan: ModelPlan,
        runtime: AutoTrainingRuntime,
        instrumentation: TrainingInstrumentation,
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> 'AutoTrainer':
        r"""Creates an :class:`AutoTrainer` instance from a
        :class:`~kumo.pquery.PredictiveQuery`.

        Args:
            pquery: The predictive query.
            model_plan: The model plan.
        """
        task, space = cls.parse_pquery(pquery, model_plan, **kwargs)
        return cls(
            task=task,
            space=space,
            runtime=runtime,
            instrumentation=instrumentation,
            checkpoint_path=checkpoint_path,
        )

    @classmethod
    def from_job_config(
        cls,
        job_config: JobConfig,
        runtime: AutoTrainingRuntime,
        instrumentation: TrainingInstrumentation,
        task: ResolvedNumericalTask | None = None,
        random_state_seed: int | None = None,
        overwrite_train_table_edge: bool | None = True,
    ) -> 'AutoTrainer':
        r"""Creates an :class:`AutoTrainer` instance from a job config.

        Args:
            job_config: The auto trainer job config.
            random_state_seed: Random seed for reproducible results.
            task: Fully resolved numerical task to train. If omitted, the
                runtime resolves the task named by the job config.
            overwrite_train_table_edge: Determines if train table edges should
                be overwritten even if they exist.
        """
        # Construct the task and auto trainer config.
        if task is not None and task.name != job_config.task_name:
            raise RuntimeError(
                f"The task name '{job_config.task_name}' in the job config "
                f"does not match the task '{task.name}' passed in.")
        if task is None:
            task = runtime.load_task(
                job_config.task_name,
                job_config.home_dir,
            )

        space = runtime.load_automl_config(job_config)
        checkpoint_path, checkpoint_encoder_dir = resolve_warm_start(
            space, task, runtime, trainer_dir=job_config.warm_start_trainer_dir)

        # The auto trainer reporter is in pytest if the home dir is not on s3.
        return cls(
            task=task,
            space=space,
            runtime=runtime,
            instrumentation=instrumentation,
            random_state_seed=random_state_seed,
            overwrite_train_table_edge=overwrite_train_table_edge,
            checkpoint_path=checkpoint_path,
            checkpoint_encoder_dir=checkpoint_encoder_dir,
            home_dir=job_config.home_dir,
            diskgraph_task_dataset_bundle_dir=(
                job_config.diskgraph_task_dataset_bundle_dir),
        )
