# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torchmetrics
from kumoapi.encoder import Index, Numerical
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from kumoapi.typing import ColStatType, Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric.metrics import (
    LinkPredAveragePopularity,
    LinkPredCoverage,
    LinkPredDiversity,
    LinkPredF1,
    LinkPredHitRatio,
    LinkPredMAP,
    LinkPredMetricCollection,
    LinkPredMRR,
    LinkPredNDCG,
    LinkPredPersonalization,
    LinkPredPrecision,
    LinkPredRecall,
)
from torch_geometric.metrics.link_pred import _LinkPredMetric
from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)
from torchmetrics.retrieval.base import RetrievalMetric
from typing_extensions import Self

import logging
from kumo_core.features.stores.feature import FeatureStore
from kumo_core.evaluation.classification import BinaryAUPRC, BinaryAUROC, BinaryAveragePrecision, MultilabelAUPRC, MultilabelAUROC, MultilabelAveragePrecision
from kumo_core.evaluation.distribution import LogNormalDistribution, NegativeBinomialDistribution, NormalDistribution
from kumo_core.evaluation.ranking import MulticlassMRR
from kumo_core.encoders.target.base import TargetEncoder

logger = logging.getLogger(__name__)


def _to_torchmetrics_task(task_type: TaskType, num_classes: int) -> str:
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return 'multiclass'
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return 'binary'
    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return 'multilabel' if num_classes > 1 else 'binary'

    raise RuntimeError(f"Unsupported task '{task_type}' in `torchmetrics`")


@dataclass
class _MetricType:
    cls: type
    pred_transform: Callable | None = None
    target_transform: Callable | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def init_metric(self) -> torchmetrics.Metric:
        metric = self.cls(**self.kwargs)

        # Patch `metric.update()` to inject custom logic on-the-fly:
        def decorator(compute_fn: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Tensor:
                pred, target, args = args[0], args[1], args[2:]

                # Transform prediction and target:
                if self.pred_transform is not None:
                    pred = self.pred_transform(pred)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                # All `RetrievalMetric` instances need an extra `indexes`
                # argument to decide to which group the prediction belongs to.
                # In our case, all predictions belong to the same group:
                if isinstance(metric, RetrievalMetric):
                    indexes = torch.zeros_like(target, dtype=torch.long)
                    kwargs['indexes'] = indexes

                return compute_fn(pred, target, *args, **kwargs)

            return wrapper

        metric.update = decorator(metric.update)

        return metric


class MetricStore(torch.nn.Module):
    r"""A metric store to register and compute metrics.

    Args:
        task_type: The type of the task.
        target_encoder: The target encoder used for decoding normalized targets
            back to their original raw values.
    """
    @classmethod
    def _parse_full_name(
        cls,
        full_name: str,
    ) -> tuple[str, TrainingStage | None]:
        r"""Parses the full name, and returns metric name and stage."""
        metric_name = full_name
        stage: TrainingStage | None = None
        if metric_name.startswith('train_'):
            metric_name = metric_name[len('train_'):]
            stage = TrainingStage.TRAIN
        elif metric_name.startswith('val_'):
            metric_name = metric_name[len('val_'):]
            stage = TrainingStage.VAL
        elif metric_name.startswith('test_'):
            metric_name = metric_name[len('test_'):]
            stage = TrainingStage.TEST
        return metric_name, stage

    @classmethod
    def tune_mode(cls, metric_name: str) -> Literal['min', 'max']:
        r"""Returns the mode of the metric (``"min"`` or ``"max"``)."""
        metric_name, _ = cls._parse_full_name(metric_name)
        if '@' in metric_name:
            metric_name = metric_name[:metric_name.find('@') + 1]

        if metric_name == 'loss':
            return 'min'

        # TODO (matthias) This is a hacky way to get the tune mode of a metric
        # independent of the task type. Think about how to fix more elegantly.
        for task_type in TaskType:
            target_encoder: TargetEncoder | None = None
            if task_type.is_classification:
                # We just use a dummy index encoder here since tune_mode
                # is not relevant to the actual number of classes anyway.
                # Here `target_encoder` is just defined locally and the
                # resulting `metric_store` is never used by the actual
                # lightning module. The correct `target_encoder` is set inside
                # lightning module, so the actual metric computation will use
                # the correct `target_encoder`.
                encoder = Index()
                encoder._stats = {
                    ColStatType.CATEGORY_COUNTS: [['A', 1], ['B', 1]],
                }
                target_encoder = TargetEncoder.from_encoder(task_type, encoder)
            elif task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
                encoder = Numerical()
                target_encoder = TargetEncoder.from_encoder(task_type, encoder)
            metric_store = cls(task_type, target_encoder)
            if metric_name in metric_store._metric_types:
                metric_type = metric_store._metric_types[metric_name]

                # Note: In the current implementation of these metrics,
                # `higher_is_better` is hardcoded to be `None`. Here we
                # manually adjust `higher_is_better` to `True`
                higher_is_better: bool | None
                if metric_type.cls in {
                        BinaryAUROC,
                        BinaryAUPRC,
                        BinaryAveragePrecision,
                        MultilabelAUROC,
                        MultilabelAUPRC,
                        MultilabelAveragePrecision,
                        torchmetrics.Accuracy,
                        torchmetrics.F1Score,
                        torchmetrics.Precision,
                        torchmetrics.Recall,
                }:
                    higher_is_better = True
                elif hasattr(metric_type.cls, 'higher_is_better'):
                    higher_is_better = bool(metric_type.cls.higher_is_better)
                else:
                    metric = metric_type.init_metric()
                    higher_is_better = metric.higher_is_better

                if higher_is_better is None:
                    raise ValueError(f"Failed to infer the tuning direction "
                                     f"of {metric_name}.")
                return 'max' if higher_is_better else 'min'

        raise ValueError(f"Unknown metric '{metric_name}'")

    def __init__(
        self,
        task_type: TaskType,
        target_encoder: TargetEncoder | None = None,
    ) -> None:
        super().__init__()

        self.task_type = task_type
        self.target_encoder = target_encoder

        # Add available metric signatures to the store:
        self._metric_types: dict[str, _MetricType] = {}

        # Initialized metrics:
        self._metric_collections: torch.nn.ModuleDict[
            TrainingStage,
            torchmetrics.MetricCollection | LinkPredMetricCollection,
        ] = torch.nn.ModuleDict()

        # Classification metrics ##############################################

        if task_type.is_classification:
            average: str | None = 'macro'
            if task_type == TaskType.BINARY_CLASSIFICATION:
                num_categories = 2
            else:
                assert target_encoder is not None
                num_categories = target_encoder.num_categories

            self._metric_types['acc'] = _MetricType(
                cls=torchmetrics.Accuracy,
                target_transform=lambda target: target.view(-1)
                if task_type == TaskType.MULTICLASS_CLASSIFICATION else target,
                kwargs=dict(
                    task=_to_torchmetrics_task(
                        task_type=task_type,
                        num_classes=num_categories,
                    ),
                    num_classes=num_categories,
                    num_labels=num_categories,
                ),
            )

            self._metric_types['precision'] = _MetricType(
                cls=torchmetrics.Precision,
                target_transform=lambda target: target.view(-1)
                if task_type == TaskType.MULTICLASS_CLASSIFICATION else target,
                kwargs=dict(
                    task=_to_torchmetrics_task(
                        task_type=task_type,
                        num_classes=num_categories,
                    ),
                    num_classes=num_categories,
                    num_labels=num_categories,
                    average=average,
                ),
            )

            self._metric_types['recall'] = _MetricType(
                cls=torchmetrics.Recall,
                target_transform=lambda target: target.view(-1)
                if task_type == TaskType.MULTICLASS_CLASSIFICATION else target,
                kwargs=dict(
                    task=_to_torchmetrics_task(
                        task_type=task_type,
                        num_classes=num_categories,
                    ),
                    num_classes=num_categories,
                    num_labels=num_categories,
                    average=average,
                ),
            )

            self._metric_types['f1'] = _MetricType(
                cls=torchmetrics.F1Score,
                target_transform=lambda target: target.view(-1)
                if task_type == TaskType.MULTICLASS_CLASSIFICATION else target,
                kwargs=dict(
                    task=_to_torchmetrics_task(
                        task_type=task_type,
                        num_classes=num_categories,
                    ),
                    num_classes=num_categories,
                    num_labels=num_categories,
                    average=average,
                ),
            )

        if task_type == TaskType.BINARY_CLASSIFICATION:
            warnings.filterwarnings('ignore', '.*Metric `AUROC` will.*')
            self._metric_types['auroc'] = _MetricType(
                cls=BinaryAUROC,
                pred_transform=lambda pred: pred.sigmoid(),
            )

            warnings.filterwarnings('ignore', '.*Metric `PrecisionRecall*')

            # TODO Migrate from AUPRC to Average Precision
            self._metric_types['auprc'] = _MetricType(
                cls=BinaryAUPRC,
                pred_transform=lambda pred: pred.sigmoid(),
            )

            self._metric_types['ap'] = _MetricType(
                cls=BinaryAveragePrecision,
                target_transform=lambda target: target.long(),
                pred_transform=lambda pred: pred.sigmoid(),
            )

            self._metric_types['recall@'] = _MetricType(cls=RetrievalRecall)

            self._metric_types['precision@'] = _MetricType(
                cls=RetrievalPrecision)

            self._metric_types['ndcg'] = _MetricType(
                cls=RetrievalNormalizedDCG)

            self._metric_types['ndcg@'] = _MetricType(
                cls=RetrievalNormalizedDCG)

        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self._metric_types['mrr'] = _MetricType(cls=MulticlassMRR)

        if task_type == TaskType.MULTILABEL_CLASSIFICATION:
            assert target_encoder is not None

            # All computation for multilabel classification will be performed
            # on CPU, to avoid memory overflow. This is because it requires
            # storing the entire predictions/targets across all labels to
            # compute AUROC/AP/AUPRC metrics.
            if target_encoder.num_categories == 1:
                # In case num_labels is 1, we fall back to binary AUROC/AP
                # since multi-label AUROC/AP does not support num_labels is 1.
                # In this case, both macro and micro average fall back to the
                # ordinary binary AUROC/AP
                for name in ["macro", "micro", "per_label"]:
                    self._metric_types[f'auroc_{name}'] = _MetricType(
                        cls=BinaryAUROC,
                        target_transform=lambda target: target.cpu(),
                        pred_transform=lambda pred: pred.sigmoid().cpu(),
                    )
                    self._metric_types[f'ap_{name}'] = _MetricType(
                        cls=BinaryAveragePrecision,
                        target_transform=lambda target: target.long().cpu(),
                        pred_transform=lambda pred: pred.sigmoid().cpu(),
                    )
                    self._metric_types[f'auprc_{name}'] = _MetricType(
                        cls=BinaryAUPRC,
                        target_transform=lambda target: target.long().cpu(),
                        pred_transform=lambda pred: pred.sigmoid().cpu(),
                    )
            else:
                for name in ["macro", "micro", "per_label"]:
                    average = None if name == "per_label" else name
                    self._metric_types[f'auroc_{name}'] = _MetricType(
                        cls=MultilabelAUROC,
                        target_transform=lambda target: target.long().cpu(),
                        pred_transform=lambda pred: pred.sigmoid().cpu(),
                        kwargs=dict(
                            num_labels=target_encoder.num_categories,
                            average=average,
                        ),
                    )
                    self._metric_types[f'ap_{name}'] = _MetricType(
                        cls=MultilabelAveragePrecision,
                        target_transform=lambda target: target.long().cpu(),
                        pred_transform=lambda pred: pred.sigmoid().cpu(),
                        kwargs=dict(
                            num_labels=target_encoder.num_categories,
                            average=average,
                        ),
                    )
                    self._metric_types[f'auprc_{name}'] = _MetricType(
                        cls=MultilabelAUPRC,
                        target_transform=lambda target: target.long().cpu(),
                        pred_transform=lambda pred: pred.sigmoid().cpu(),
                        kwargs=dict(
                            num_labels=target_encoder.num_categories,
                            average=average,
                        ),
                    )

        # Regression metrics ##################################################

        if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
            assert target_encoder is not None
            self._metric_types['mae'] = _MetricType(
                cls=torchmetrics.MeanAbsoluteError,
                pred_transform=target_encoder.decode_tensor,
                target_transform=target_encoder.decode_tensor,
            )

            self._metric_types['mape'] = _MetricType(
                cls=torchmetrics.MeanAbsolutePercentageError,
                pred_transform=target_encoder.decode_tensor,
                target_transform=target_encoder.decode_tensor,
            )

            self._metric_types['mse'] = _MetricType(
                cls=torchmetrics.MeanSquaredError,
                pred_transform=target_encoder.decode_tensor,
                target_transform=target_encoder.decode_tensor,
            )

            self._metric_types['rmse'] = _MetricType(
                cls=torchmetrics.MeanSquaredError,
                pred_transform=target_encoder.decode_tensor,
                target_transform=target_encoder.decode_tensor,
                kwargs=dict(squared=False),
            )

            self._metric_types['smape'] = _MetricType(
                cls=torchmetrics.SymmetricMeanAbsolutePercentageError,
                pred_transform=target_encoder.decode_tensor,
                target_transform=target_encoder.decode_tensor,
            )

            self._metric_types['wape'] = _MetricType(
                cls=torchmetrics.WeightedMeanAbsolutePercentageError,
                pred_transform=target_encoder.decode_tensor,
                target_transform=target_encoder.decode_tensor,
            )

            self._metric_types['r2'] = _MetricType(
                cls=torchmetrics.R2Score,
                pred_transform=target_encoder.decode_tensor,
                target_transform=target_encoder.decode_tensor,
            )

        # Link prediction metrics #############################################

        if task_type.is_ranking:
            self._metric_types['map@'] = _MetricType(cls=LinkPredMAP)
            self._metric_types['ndcg@'] = _MetricType(cls=LinkPredNDCG)
            self._metric_types['mrr@'] = _MetricType(cls=LinkPredMRR)
            self._metric_types['precision@'] = _MetricType(
                cls=LinkPredPrecision)
            self._metric_types['recall@'] = _MetricType(cls=LinkPredRecall)
            self._metric_types['f1@'] = _MetricType(cls=LinkPredF1)
            self._metric_types['hit_ratio@'] = _MetricType(
                cls=LinkPredHitRatio)

        if task_type.is_link_pred:
            self._metric_types['weighted_recall@'] = _MetricType(
                cls=LinkPredRecall, kwargs=dict(weighted=True))
            self._metric_types['weighted_ndcg@'] = _MetricType(
                cls=LinkPredNDCG, kwargs=dict(weighted=True))
            self._metric_types['personalization@'] = _MetricType(
                cls=LinkPredPersonalization)
            self._metric_types['coverage@'] = _MetricType(cls=LinkPredCoverage)
            self._metric_types['avg_popularity@'] = _MetricType(
                cls=LinkPredAveragePopularity)
            self._metric_types['diversity@'] = _MetricType(
                cls=LinkPredDiversity)

        # Forecasting metrics #################################################

        # TODO (matthias) Add support for decoding once `FORECASTING` has been
        # properly integrated.
        if task_type == TaskType.FORECASTING:
            self._metric_types['neg_binomial'] = _MetricType(
                cls=NegativeBinomialDistribution,
                kwargs=dict(squared=False),
            )

            self._metric_types['normal'] = _MetricType(
                cls=NormalDistribution,
                kwargs=dict(squared=False),
            )

            self._metric_types['lognormal'] = _MetricType(
                cls=LogNormalDistribution,
                kwargs=dict(squared=False),
            )

    def _init_metric(
        self,
        name: str,
        stage: TrainingStage,
        kwargs: dict[str, Any],
    ) -> torchmetrics.Metric:
        top_k: int | None = None
        if '@' in name:
            name, _top_k = name.split('@')
            top_k = int(_top_k)
            name = f'{name}@'

        # Extract column name specified in brackets:
        match = re.search(r"\[(.*?)\]", name)
        col_name = match.group(1) if match else None
        name = re.sub(r"\[.*?\]", "", name)

        # Deep-copy metric type so that we don't override `kwargs`:
        metric_type = copy.deepcopy(self._metric_types[name])

        if top_k is not None:
            if issubclass(metric_type.cls, _LinkPredMetric):
                metric_type.kwargs['k'] = top_k
            else:
                metric_type.kwargs['top_k'] = top_k

        if issubclass(metric_type.cls, LinkPredCoverage):
            metric_type.kwargs['num_dst_nodes'] = kwargs['num_dst_nodes']

        if issubclass(metric_type.cls, LinkPredAveragePopularity):
            metric_type.kwargs['popularity'] = kwargs['popularity']

        if issubclass(metric_type.cls, LinkPredDiversity):
            metric_type.kwargs['category'] = kwargs[f'category_{col_name}']

        return metric_type.init_metric()

    @property
    def is_initialized(self) -> bool:
        return len(self._metric_collections) > 0

    def init_metrics(
        self,
        metrics: list[str],
        module: torch.nn.Module,
    ) -> Self:

        # Pre-compute data-aware metric arguments:
        kwargs: dict[TrainingStage, dict[str, Any]] = defaultdict(dict)
        rhs_tf: TensorFrame | None = None
        for name in copy.copy(metrics):
            key = f'{name.split("@")[0]}@' if '@' in name else name
            key = re.sub(r"\[.*?\]", "", key)
            metric_type = self._metric_types[key]

            # Compute number of valid RHS nodes per stage:
            if (issubclass(metric_type.cls, LinkPredCoverage)
                    and 'num_dst_nodes' not in kwargs.get(
                        TrainingStage.TRAIN, {})):
                if not hasattr(module, 'lp_train_table'):  # Baseline fallback.
                    metrics.remove(name)
                    warnings.warn(
                        f"Cannot compute metric '{name}' for module "
                        f"'{module.__class__.__name__}' since it "
                        f"does not have a train table registered",
                        stacklevel=2)
                    continue
                for stage in TrainingStage.list_stages():
                    kwargs[stage]['num_dst_nodes'] = (
                        module.lp_train_table.get_valid_rhs_idx(stage).numel())

            # Compute popularity of RHS nodes on the training set:
            if (issubclass(metric_type.cls, LinkPredAveragePopularity) and
                    'popularity' not in kwargs.get(TrainingStage.TRAIN, {})):
                if not hasattr(module, 'lp_train_table'):  # Baseline fallback.
                    metrics.remove(name)
                    warnings.warn(
                        f"Cannot compute metric '{name}' for module "
                        f"'{module.__class__.__name__}' since it "
                        f"does not have a train table registered",
                        stacklevel=2)
                    continue
                popularity = module.lp_train_table.get_popularity(
                    TrainingStage.TRAIN)
                for stage in TrainingStage.list_stages():
                    kwargs[stage]['popularity'] = popularity

            # Retrieve RHS features:
            if (issubclass(metric_type.cls, LinkPredDiversity)
                    and rhs_tf is None):
                if (not hasattr(module, 'datamodule')
                        and not hasattr(module.datamodule, 'feature_store')):
                    metrics.remove(name)
                    warnings.warn(
                        f"Cannot compute metric '{name}' for module "
                        f"'{module.__class__.__name__}' since it "
                        f"does not have a feature store registered",
                        stacklevel=2)
                    continue
                feature_store = module.datamodule.feature_store
                assert isinstance(feature_store, FeatureStore)
                assert isinstance(module.entity_table_names, tuple | list)
                table_name = module.entity_table_names[1]
                batch_size = 16_000
                # NOTE This may be expensive to run :(
                logger.info(f"Fetching '{table_name}' features with a "
                            f"mini-batch size of {batch_size}")
                with feature_store.safe_batch_mode(batch_size=batch_size):
                    rhs_tf = feature_store[table_name, 'tf', :]

            # Compute RHS category that is used as diversity measure:
            if (issubclass(metric_type.cls, LinkPredDiversity)
                    and rhs_tf is not None):
                match = re.search(r"\[(.*?)\]", name)
                assert match is not None
                col_name = match.group(1)
                if f'category_{col_name}' in kwargs[TrainingStage.TRAIN]:
                    continue
                stype = Stype.categorical.value
                index = rhs_tf.col_names_dict[stype].index(col_name)
                category = rhs_tf.feat_dict[stype][:, index]
                for stage in TrainingStage.list_stages():
                    kwargs[stage][f'category_{col_name}'] = category

        for stage in TrainingStage.list_stages():
            metric_dict = {}
            for name in metrics:
                staged_name = stage.add_stage_prefix(name)
                metric_dict[staged_name] = self._init_metric(
                    name, stage, kwargs[stage])

            if self.task_type.is_ranking:
                metric_collection = LinkPredMetricCollection(metric_dict)
            else:
                metric_collection = torchmetrics.MetricCollection(metric_dict)

            self._metric_collections[f'{stage}_metrics'] = metric_collection

        return self

    def get_metric_collection(
        self,
        stage: TrainingStage,
    ) -> torchmetrics.MetricCollection | LinkPredMetricCollection:
        metric_collection = self._metric_collections[f'{stage}_metrics']
        if isinstance(metric_collection, torchmetrics.MetricCollection):
            for metric in metric_collection.values(copy_state=False):
                if metric.device.type == 'mps':
                    # On macOS 26.7.1 / PyTorch 2.12.1, MPSGraph constant
                    # padding corrupts 1D tensors when the padded length
                    # reaches 65,536. TorchMetrics uses F.pad to append a
                    # threshold index for AP/AUPRC/AUROC, causing invalid
                    # metrics or out-of-bounds errors on large epochs.
                    # Keep accumulated prediction/target lists and their
                    # computation on CPU; model training stays on MPS.
                    # Check the actual metric device after model placement.
                    metric.compute_on_cpu = True
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            metric_collection.cpu()  # Use CPU for multilabel classification.
        return metric_collection

    @staticmethod
    def validate_metrics(
        task_type: TaskType,
        metrics: list[str],
        stype_dict: dict[str, dict[str, Stype]] | None = None,
        entity_table_names: list[str] | None = None,
    ) -> None:
        r"""Validate the compatiblity between a list of metrics and
        certain task_type.

        Args:
            task_type: The task type to check metrics compatiblity
            for.
            metrics: A list of metrics.
            stype_dict: The semantic types of columns for each table in the
                graph.
            entity_table_names: The entity table names to predict on.
        """
        target_encoder: TargetEncoder | None = None
        if task_type.is_classification:
            # We just use a dummy index encoder here for metric validation.
            # Here `target_encoder` is just defined locally and the
            # resulting `metric_store` is never used by the actual
            # lightning module. The correct `target_encoder` is set inside
            # lightning module, so the actual metric computation will use
            # the correct `target_encoder`.
            encoder = Index()
            encoder._stats = {
                ColStatType.CATEGORY_COUNTS: [['A', 1], ['B', 1]],
            }
            target_encoder = TargetEncoder.from_encoder(task_type, encoder)
        elif task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
            target_encoder = TargetEncoder.from_encoder(task_type, Numerical())

        metric_store = MetricStore(task_type, target_encoder)
        supported_metrics = list(metric_store._metric_types.keys())

        for name in metrics:
            if '@' in name:
                name_split = name.split('@')
                if len(name_split) != 2:
                    raise ValueError(f"Unsupported metric '{name}' for "
                                     f"'{task_type}' problems. Available "
                                     f"metrics are {supported_metrics}.")

                name, top_k = f'{name_split[0]}@', name_split[1]

                if not top_k.isdigit() or int(top_k) <= 0:
                    raise ValueError(f"Metric '{name}' does not define a "
                                     f"valid 'top_k' value (got '{top_k}').")

                if task_type.is_link_pred and name.startswith('diversity'):
                    match = re.match(r"(.*?)\[(.*?)\]@", name)
                    if match is None:
                        raise ValueError(f"Metric '{name}' does not point to "
                                         f"a column name.")

                    _name, col_name = match.groups()
                    if stype_dict is not None and entity_table_names:
                        rhs_table_name = entity_table_names[-1]
                        stype = stype_dict[rhs_table_name].get(col_name, None)
                        if stype not in {Stype.categorical, Stype.ID}:
                            raise ValueError(f"Metric '{name}' does not point "
                                             f"to a categorical column name.")

                    name = f'{_name}@'

            if name not in supported_metrics:
                raise ValueError(f"Unsupported metric '{name}' for "
                                 f"'{task_type}' problems. Available metrics "
                                 f"are {supported_metrics}.")
