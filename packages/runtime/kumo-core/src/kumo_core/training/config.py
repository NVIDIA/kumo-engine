# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from dataclasses import field
from enum import Enum
from typing import Any, Literal

from kumo_core.utils.mixins import CastMixin
from kumo_core.utils.names import get_table_name
from kumoapi.model_plan import (
    FocalLossConfig,
    HuberLossConfig,
    OrdinalLogLossConfig,
    QuantileLossConfig,
    RHSEmbeddingMode,
    WeightMode,
)
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs
from torch_geometric.sampler.base import NumNeighbors, SubgraphType
from torch_geometric.typing import EdgeTypeStr
from typing_extensions import Self

_DATACLASS_CONFIG = ConfigDict(arbitrary_types_allowed=True)
RHS_NUM_EMB_SIZE_LIMIT = (5_000_000, 64)
_KUMO_EDGE_KEY = '__KUMO_EDGE_KEY__'


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


def _normalize_edge_type_key(edge_type: Any) -> tuple[str, str, str] | None:
    """Normalize PyG/serialized edge-type keys to bare-name triples."""
    try:
        if hasattr(edge_type, 'to_tuple'):
            src, rel, dst = edge_type.to_tuple()
        elif isinstance(edge_type, (tuple, list)) and len(edge_type) == 3:
            src, rel, dst = edge_type
        elif isinstance(edge_type, str):
            if _KUMO_EDGE_KEY in edge_type:
                src, rel, dst = edge_type.split(_KUMO_EDGE_KEY, 2)
            else:
                src, rel, dst = EdgeTypeStr(edge_type).to_tuple()
        else:
            return None
    except Exception:
        return None
    if not all(isinstance(v, str) for v in (src, rel, dst)):
        return None
    return (get_table_name(src), rel, get_table_name(dst))


def _unglue_edge_type_key(key: Any) -> Any:
    # Models trained before this fix persisted edge types in a format that
    # PyG's EdgeTypeStr rejects. Other valid keys must remain unchanged, or
    # get_table_name() would strip their data-source prefixes.
    if isinstance(key, str) and key.count(_KUMO_EDGE_KEY) == 2:
        return _normalize_edge_type_key(key) or key
    return key


class LinkPredModuleType(Enum):
    r"""The different link prediction modules used internally."""
    # Shallow-RHS based link prediction:
    SHALLOW_RHS = 'shallow_rhs'
    # ID-based/repeated link prediction:
    ID = 'id'
    # Combination of ID and Shallow-RHS link prediction:
    HYBRID = 'hybrid'
    # Two tower link prediction:
    TWO_TOWER = 'two_tower'

    @property
    def use_bidirectional_sampling(self) -> bool:
        return self in [
            LinkPredModuleType.ID,
            LinkPredModuleType.HYBRID,
        ]

    @property
    def use_lhs_node_loader(self) -> bool:
        return self in [
            LinkPredModuleType.SHALLOW_RHS,
            LinkPredModuleType.ID,
            LinkPredModuleType.HYBRID,
        ]


class FitMode(Enum):
    r"""Specifies the fitting mode of a trainer."""
    # Single fitting:
    fit = 'fit'
    # Refit on the training/validation set:
    refit_trainval = 'refit_trainval'
    # Refit on the entire training/validation/test set:
    refit_full = 'refit_full'

    @property
    def is_refit(self) -> bool:
        return self != FitMode.fit


@kumo_dataclass
class RefitConfig:
    r"""Configuration for refitting models."""
    # Best validation epoch founde (set in `fit()`):
    best_val_epoch: int = -1
    # Number of samples seen per epoch (set in `Trainer.__init__`):
    num_samples_per_epoch: int = -1
    # Indicate whether a trainer is already full-refitted:
    refitted_full: bool = False
    # Indicate whether a trainer is already trainval-refitted:
    refitted_trainval: bool = False

    # Indicate whether a trainer is already fitted.
    @property
    def fitted(self) -> bool:
        r"""Indicates whether a trainer is already fitted."""
        return self.best_val_epoch != -1


@kumo_dataclass
class LinkPredModuleKwargs(CastMixin):
    # The link prediction module to use during training:
    module: LinkPredModuleType | None = None
    # L2 regularization on the final embeddings:
    emb_l2_reg_coeff: float = 0.0
    # The number of RHS nodes to sample:
    rhs_sample_size: int | None = None
    # How to model RHS features:
    rhs_emb_mode: RHSEmbeddingMode = RHSEmbeddingMode.LOOKUP
    # The standard deviation of newly initialized RHS embeddings.
    # If `None`, it will be set based on `module`:
    rhs_emb_init_std: float | None = None
    # The coefficient for the embedding-based loss in Hybrid-GNN:
    hybrid_emb_loss_coeff: float = 0.0
    # Whether to focus on evaluating on new visits:
    eval_new_visit: bool = False
    # Device handling of RHS features:
    rhs_feat_bytes_threshold: int | None = None
    # The fraction of ground-truth links to be used for supervision.
    target_fraction: float = 1.0
    # The overlap ratio between message passing edges and supervision edges.
    overlap_ratio: float = 1.0

    def validate(self) -> Self:
        if self.module is None:
            raise ValueError("'module' is undefined")

        if self.emb_l2_reg_coeff < 0:
            raise ValueError(f"'emb_l2_reg_coeff' needs to be non-negative "
                             f"(got '{self.emb_l2_reg_coeff}')")

        if self.rhs_sample_size is not None and self.rhs_sample_size <= 0:
            raise ValueError(f"'rhs_sample_size' needs to be positive "
                             f"(got '{self.rhs_sample_size}')")

        if self.eval_new_visit and self.module == LinkPredModuleType.ID:
            raise ValueError("'eval_new_visit' is not supported for ID-GNN")

        if self.rhs_emb_init_std is None:
            raise ValueError("'rhs_emb_init_std' is undefined")

        if self.rhs_emb_init_std <= 0.0:
            raise ValueError(f"'rhs_emb_init_std' needs to be positive "
                             f"(got '{self.rhs_emb_init_std}')")

        if self.hybrid_emb_loss_coeff < 0.0:
            raise ValueError(f"'hybrid_emb_loss_coeff' needs to be non-"
                             f"negative (got '{self.hybrid_emb_loss_coeff}')")

        if (self.hybrid_emb_loss_coeff > 0.0
                and self.module != LinkPredModuleType.HYBRID):
            warnings.warn("'hybrid_emb_loss_coeff' is not used for non-"
                          "hybrid GNN models")

        return self

    @property
    def supports_unlimited_rhs_nodes_in_bp(self) -> bool:
        return self.module in {
            LinkPredModuleType.SHALLOW_RHS,
            LinkPredModuleType.HYBRID,
        } and self.rhs_emb_mode.only_use_rhs_feature


@kumo_dataclass
class SamplerKwargs(CastMixin):
    num_neighbors: NumNeighbors = field(
        default_factory=lambda: NumNeighbors([-1, -1]))
    disjoint: bool = True
    replace: bool = False
    time_attr: str | None = None
    temporal_strategy: Literal['uniform', 'last'] = 'last'
    subgraph_type: SubgraphType = SubgraphType.directional
    adaptive_sampling: bool = False

    @model_validator(mode='before')
    @classmethod
    def _coerce_num_neighbors(cls, data: Any) -> Any:
        data = before_validator_dict(data)
        if not isinstance(data, dict):
            return data
        num_neighbors = data.get('num_neighbors')
        if num_neighbors is None:
            data['num_neighbors'] = NumNeighbors([-1, -1])
        elif isinstance(num_neighbors, list):
            data['num_neighbors'] = NumNeighbors(num_neighbors)
        elif isinstance(num_neighbors, dict):
            # ``NumNeighbors`` is a PyG arbitrary type whose pydantic-core
            # schema validates ``values`` against ``EdgeTypeStr`` keys (a str
            # subclass) and rejects the plain ``(src, rel, dst)`` tuples that
            # ``from_json`` restores from serialized config. Reconstruct a
            # ``NumNeighbors`` instance so it is accepted as-is. The serialized
            # form is the ``{'values': ..., 'default': ...}`` envelope; older
            # configs may store the bare per-edge-type mapping directly.
            if 'values' in num_neighbors:
                values = num_neighbors['values']
                default = num_neighbors.get('default')
            else:
                values, default = num_neighbors, None
            if isinstance(values, dict):
                values = {
                    _unglue_edge_type_key(key): value
                    for key, value in values.items()
                }
            data['num_neighbors'] = NumNeighbors(values, default)
        return data


_SHM_BYTES_PER_WORKER = 256 * 1024**2


def _dev_shm_bytes() -> int | None:
    try:
        stat = os.statvfs('/dev/shm')
    except OSError:
        return None
    return stat.f_blocks * stat.f_frsize


def shm_safe_num_workers(num_workers: int) -> int:
    """Reduce ``num_workers`` to what ``/dev/shm`` can hold.

    DataLoader workers pass batches through shared memory, so too little
    ``/dev/shm`` kills them mid-epoch (kumo-ai/kumo#29547). 0 means the main
    process loads inline and needs none.
    """
    shm_bytes = _dev_shm_bytes()
    if num_workers <= 0 or shm_bytes is None:
        return num_workers
    affordable = shm_bytes // _SHM_BYTES_PER_WORKER
    if affordable >= num_workers:
        return num_workers
    resolved = affordable if affordable >= 2 else 0
    warnings.warn(
        f"/dev/shm is only {shm_bytes // 1024**2} MiB; reducing DataLoader "
        f"num_workers {num_workers} -> {resolved}. Start the container with "
        f"--ipc=host or --shm-size=8g to restore full parallel loading.",
        RuntimeWarning, stacklevel=2)
    return resolved


@kumo_dataclass
class LoaderKwargs(CastMixin):
    batch_size: int = 1024
    num_workers: int = -1
    filter_per_worker: bool = True
    # Pairs with the non-blocking transfer in BaseLightningModule.
    pin_memory: bool = False
    # Keeps workers and their sampler state alive across epochs.
    persistent_workers: bool = False
    # Batches each worker runs ahead. None leaves PyTorch's default of 2.
    prefetch_factor: int | None = None

    @model_validator(mode='before')
    @classmethod
    def _resolve_num_workers(cls, data: Any) -> Any:
        data = before_validator_dict(data)
        if not isinstance(data, dict):
            return data
        if data.get('num_workers') is None:
            if 'PYTEST_CURRENT_TEST' in os.environ:
                warnings.filterwarnings('ignore', '.*not have many workers.*')
                num_workers = 0
            elif os.getenv('GITHUB_ACTIONS') == 'true':
                warnings.filterwarnings('ignore', '.*not have many workers.*')
                num_workers = 0
            elif os.environ.get('SYNC_DATA_LOADER', 0):
                # Enforce synchronous data loading. Useful for debugging or for
                # systems with limited shared memory.
                warnings.filterwarnings('ignore', '.*not have many workers.*')
                num_workers = 0
            else:
                import torch
                num_workers = min(torch.multiprocessing.cpu_count(), 16)
                # Sample in main thread if only one worker is available.
                num_workers = num_workers if num_workers > 1 else 0
                num_workers = shm_safe_num_workers(num_workers)
            data['num_workers'] = num_workers
        return data


@kumo_dataclass
class TrainerConfig:
    encoder: Any
    model: Any
    head: Any
    past_encoder: Any | None = None
    year_over_year_encoder: Any | None = None
    root_dir: str | None = None
    module_kwargs: LinkPredModuleKwargs | None = None
    sampler_kwargs: SamplerKwargs | None = None
    loader_kwargs: LoaderKwargs | None = None
    eval_loader_kwargs: dict[str, Any] | None = None
    optimizer: str = 'Adam'
    optimizer_kwargs: dict[str, Any] | None = None
    lr_scheduler: str | None = None
    lr_scheduler_kwargs: dict[str, Any] | None = None
    lr_scheduler_interval: str | None = None
    early_stopping: bool = True
    early_stopping_kwargs: dict[str, Any] | None = None
    max_epochs: int = 100
    min_steps_per_epoch: int = 1
    max_steps_per_epoch: int = 2000
    max_val_steps: int | None = 2000
    max_test_steps: int | None = 4000
    track_version: bool = True
    version: int | None = None
    tune_metric: str | None = None
    loss: (str | FocalLossConfig | HuberLossConfig | QuantileLossConfig
           | OrdinalLogLossConfig | None) = None
    distance_measure: Literal['dot_product', 'cosine'] = 'dot_product'
    binary_threshold: float = 0.5
    kumo_version: str | None = None
    refit_cfg: RefitConfig = field(default_factory=RefitConfig)
    majority_sampling_ratio: float | None = None
    weight_mode: WeightMode | None = None
    random_state_seed: int | None = None
    enable_compile: bool = False
    # By default, is reverse edge is diabled for models trained in the past.
    # It is True by default for new models.
    is_reverse_edge_enabled: bool = False

    @property
    def table_names(self) -> set[str]:
        r"""Returns the table names the trainer is operating on."""
        return set(self.encoder['table_encoder_dict'].keys())


@kumo_dataclass
class DiskgraphNestedTrainerConfig(TrainerConfig):
    """Trainer config for ``DiskgraphNestedTrainer``.

    Empty subclass — the bundle dir is threaded via ``JobConfig``, not
    persisted here. Lives in this module (next to its parent) so
    pydantic resolves the inherited annotations' forward references
    against the namespace where the referenced types are defined.
    Distinct from ``TrainerConfig`` because
    ``torch_geometric.config_store`` rejects two classes sharing the
    same data_cls registration.
    """
