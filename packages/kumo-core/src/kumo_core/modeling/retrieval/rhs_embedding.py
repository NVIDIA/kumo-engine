# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Literal

import pandas as pd
import torch
import torch.nn.functional as F
import torch_frame
from kumoapi.encoder import Encoder
from kumoapi.model_plan import RHSEmbeddingMode
from torch import Tensor
from torch_frame.nn.models.resnet import FCResidualBlock
from torch_frame.typing import TensorData
from torch_frame.utils import num_bytes
from torch_geometric.data import FeatureStore
from typing_extensions import Self

import kumo_core.modeling.context as modeling_context
import logging
from kumo_core.encoders.tensor_frame_encoder import TensorFrameEncoder
from kumo_core.modeling.retrieval.mips import MIPSFaiss, require_faiss
from kumo_core.modeling.embedding import Embedding

logger = logging.getLogger(__name__)

FS_BATCH_SIZE = 16_000
MAX_BATCH_SIZE = 500_000


class RHSEmbedding(torch.nn.Module):
    r"""A module to receive embeddings from RHS nodes based on a given
    embedding mode.

    Args:
        emb_mode: How to model RHS features (one of ``"lookup"``,
            ``"feature"``, ``"shallow_feature"``, or ``"fusion"``).
        num_nodes: The number of RHS nodes.
        embedding_dim: Size of each embedding vector.
        encoder_dict: The column-wise encoders to use for encoding RHS feature
            columns.
        sparse: If set to ``True``, gradient w.r.t. lookup embeddings will
            be sparse.
        init_std: The standard deviation for parameter initialization.
        index_map: The index map from raw RHS primary keys to node indices.
        distance_measure: The measure method of comparing semantic similarity.
            If ``'cosine'``, all output vectors are normalized.
        rhs_feat_bytes_threshold: Defines a threshold (in bytes) at which RHS
            features are stored in CPU memory. Useful to set in order to avoid
            RHS features taking up too much GPU memory at the cost of runtime
            efficiency.
    """
    def __init__(
        self,
        emb_mode: RHSEmbeddingMode | str,
        num_nodes: int,
        embedding_dim: int,
        encoder_dict: dict[str, Encoder] | None = None,
        sparse: bool = False,
        init_std: float = 1.0,
        index_map: pd.DataFrame | None = None,
        distance_measure: Literal['dot_product', 'cosine'] = 'dot_product',
        rhs_feat_bytes_threshold: int | None = None,
    ) -> None:
        super().__init__()

        self.emb_mode = RHSEmbeddingMode(emb_mode)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.init_std = init_std
        self.distance_measure = distance_measure
        self.rhs_feat_bytes_threshold = rhs_feat_bytes_threshold

        self.lookup_embedding: Embedding | None = None
        if self.emb_mode.use_rhs_lookup:
            self.lookup_embedding = Embedding(
                num_embeddings=num_nodes,
                embedding_dim=embedding_dim,
                init_std=init_std,
                index_map=index_map,
                sparse=sparse,
            )

        self.encoder: TensorFrameEncoder | None = None
        self.projector: torch.nn.Module | None = None
        if self.emb_mode.use_rhs_feature:
            if encoder_dict is None:
                raise ValueError(f"Embedding mode '{self.emb_mode.value}' "
                                 f"requires 'encoder_dict' to be specified")

            self.encoder = TensorFrameEncoder(
                name='rhs_table',
                col_encoder_dict=encoder_dict,
                time_col=None,
                out_channels=embedding_dim,
                act=None,
            )

            seqs: list[torch.nn.Module] = []
            if self.emb_mode == RHSEmbeddingMode.FEATURE:
                seqs += [  # Apply deep RHS projector for pure feature mode.
                    FCResidualBlock(embedding_dim, embedding_dim),
                    FCResidualBlock(embedding_dim, embedding_dim),
                ]
            seqs += [torch.nn.LayerNorm(embedding_dim, eps=1e-7)]
            self.projector = torch.nn.Sequential(*seqs)

        # Filtered index mapping:
        self._filtered_index: Tensor | None = None
        self._filtered_index_map: Tensor | None = None

        # Feature cache:
        self._feats: list[TensorData] | None = None
        self._filtered_feats: bool = False

        # Output cache:
        self._out_cache: Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.lookup_embedding is not None:
            self.lookup_embedding.reset_parameters()
        if self.encoder is not None:
            self.encoder.reset_parameters()
            for param in self.encoder.parameters():
                param.data *= self.init_std
        if self.projector is not None:
            for child in self.projector.children():
                child.reset_parameters()
                if isinstance(child, torch.nn.LayerNorm):
                    child.weight.data *= self.init_std

        self.clear_output_cache()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"emb_mode={self.emb_mode.value}, "
                f"num_nodes={self.num_nodes}, "
                f"embedding_dim={self.embedding_dim}, "
                f"distance_measure={self.distance_measure})")

    def order_by(
        self,
        index_map: pd.DataFrame,
        default_init: Literal['zero', 'mean'] = 'mean',
    ) -> Self:
        r"""Re-orders the lookup embeddings according to ``index_map``.

        See :meth:`~kumo_core.modeling.embedding.Embedding.order_by` for details.
        """
        if self.lookup_embedding is not None:
            self.lookup_embedding.order_by(index_map, default_init,
                                           num_embeddings=self.num_nodes)
        return self

    # Device Transfer #########################################################

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def _apply_device_to_cache(self, device: torch.device) -> None:
        if self._filtered_index is not None:
            self._filtered_index = self._filtered_index.to(device)
        if self._filtered_index_map is not None:
            self._filtered_index_map = self._filtered_index_map.to(device)
        if self._feats is not None:
            if (device.type != 'cuda' or self.rhs_feat_bytes_threshold is None
                    or num_bytes(self._feats) < self.rhs_feat_bytes_threshold):
                self._feats = [feat.to(device) for feat in self._feats]
        if self._out_cache is not None:
            self._out_cache = self._out_cache.to(device)

    def _apply(self, fn: Callable, recurse: bool = True) -> Self:
        super()._apply(fn, recurse)
        # Don't apply dtype casting here as cache tensors have different dtypes
        # (e.g., int64, float32) depending on column stypes, and we don't want
        # to change dtypes of caches on a, e.g., `RHSEmbedding.float()` call.
        self._apply_device_to_cache(self.device)
        return self

    # Filter Logic ############################################################

    def filter(self, index: Tensor | None) -> None:
        r"""Filters the valid set of RHS nodes to operate on."""
        self.clear_output_cache()

        # If we have only fetched a subset of RHS features, reset them:
        if self._feats is not None and self._filtered_feats:
            self._feats = None
            self._filtered_feats = False

        if index is None:
            self._filtered_index = None
            self._filtered_index_map = None
            return

        assert index.dtype == torch.int64
        assert index.device == self.device

        self._filtered_index = index
        if index.numel() < 2 or bool((index[1:] > index[:-1]).all()):
            # Canonical and temporal RHS catalogs are ordered physical node
            # IDs. Keep their compact mapping implicit instead of allocating
            # an additional num_nodes-sized inverse map.
            self._filtered_index_map = None
        else:
            self._filtered_index_map = index.new_full((self.num_nodes, ), -1)
            self._filtered_index_map[index] = torch.arange(
                start=0,
                end=index.numel(),
                device=index.device,
            )

    def _map_filtered_index(self, index: Tensor) -> Tensor:
        if self._filtered_index is None:
            return index
        if self._filtered_index_map is not None:
            return map_index(index, self._filtered_index_map)

        local_index = torch.searchsorted(self._filtered_index, index)
        in_range = local_index < self._filtered_index.numel()
        if self._filtered_index.numel():
            probe = local_index.clamp_max(self._filtered_index.numel() - 1)
            in_range &= self._filtered_index[probe] == index
        if not bool(in_range.all()):
            raise IndexError("Tried to access invalid RHS nodes")
        return local_index

    # Feature Fetching ########################################################

    def fetch_feats(
        self,
        feature_store: FeatureStore,
        table_name: str,
    ) -> None:
        r"""Fetches raw RHS features from a :class:`FeatureStore` instance."""
        if not self.emb_mode.use_rhs_feature:
            return  # Nothing to do.

        if self._feats is not None and not self._filtered_feats:
            return  # Nothing to do.

        if self._filtered_index is None:
            index = torch.arange(self.num_nodes)
            self._filtered_feats = False
        else:
            index = self._filtered_index.cpu()
            self._filtered_feats = True

        logger.info(f"Fetching '{table_name}' features with a mini-batch size "
                    f"of {FS_BATCH_SIZE}")
        feat_device = self.device
        batches: list[list[TensorData]] = []
        for i, _index in enumerate(index.split(FS_BATCH_SIZE)):
            tf = feature_store[table_name, 'tf', _index].to(self.device)
            assert self.encoder is not None
            feats = self.encoder.forward_fixed(tf)

            if i == 0 and self.rhs_feat_bytes_threshold is not None:
                # Approximately calculate bytes to store RHS features:
                factor = index.numel() / _index.numel()
                rhs_feat_bytes = int(factor * num_bytes(feats))
                if rhs_feat_bytes >= self.rhs_feat_bytes_threshold:
                    feat_device = torch.device('cpu')
                    if self.device.type == 'cuda':
                        total_mem = torch.cuda.mem_get_info()[1]
                        logger.info(
                            f"Storing RHS features on CPU since they consume "
                            f"approximately {rhs_feat_bytes/1024**3:.2f}GB "
                            f"out of the {total_mem/1024**3:.2f}GB of "
                            f"available GPU memory, but only "
                            f"{self.rhs_feat_bytes_threshold/1024**3:.2f}GB "
                            f"were allowed")

            feats = [feat.to(feat_device) for feat in feats]
            batches.append(feats)

        self._feats = [
            torch_frame.cat([feats[i] for feats in batches], dim=0)
            for i in range(len(batches[0]))
        ]

    # Caching Logic ###########################################################

    @torch.no_grad()
    def cache_output(
        self,
        feature_store: FeatureStore | None = None,
        table_name: str | None = None,
    ) -> None:
        r"""Caches the RHS output embeddings for evaluation and prediction."""
        if not self.emb_mode.use_rhs_feature:
            return  # Nothing to do.

        if self._out_cache is not None:
            raise RuntimeError("Cannot cache RHS embeddings twice")

        if self._filtered_index is None:
            index = torch.arange(self.num_nodes, device=self.device)
        else:
            index = self._filtered_index.to(self.device)

        cache = torch.empty(
            (index.numel(), self.embedding_dim),
            device=self.device,
        )

        start = 0
        for i in index.split(MAX_BATCH_SIZE):
            cache[start:start + i.numel()] = self(i, feature_store, table_name)
            start += i.numel()

        self._out_cache = cache

    def clear_output_cache(self) -> None:
        r"""Clears any cached RHS output embeddings."""
        self._out_cache = None

    @torch.no_grad()
    def get_mips_index(
        self,
        feature_store: FeatureStore | None = None,
        table_name: str | None = None,
    ) -> MIPSFaiss:
        r"""Returns a MIPS index from RHS output embeddings."""
        require_faiss(self.device)
        if self._out_cache is not None:
            raise RuntimeError("Creating a MIPS index while holding cached "
                               "RHS outputs in memory will lead to "
                               "oversubscription of memory")

        if self._filtered_index is None:
            index = torch.arange(self.num_nodes, device=self.device)
        else:
            index = self._filtered_index.to(self.device)

        cache = torch.empty(
            (index.numel(), self.embedding_dim),
            device=self.device,
        )

        start = 0
        for i in index.split(MAX_BATCH_SIZE):
            cache[start:start + i.numel()] = self(i, feature_store, table_name)
            start += i.numel()

        return MIPSFaiss(cache)

    # Retrieval ###############################################################

    def forward(
        self,
        index: Tensor | None = None,
        feature_store: FeatureStore | None = None,
        table_name: str | None = None,
    ) -> Tensor:
        r"""Returns the RHS output embeddings.

        Concretely, it behaves as follows depending on ``emb_mode``:
        - ``"lookup"``: Lookup embeddings.
        - ``"shallow_feature"``: Linear transformation of node features.
        - ``"feature"``: Use deep transformation of node features.
        - ``"hybrid"``: Add lookup and features together.

        It also avoids the redundant computation of RHS embeddings at the
        evaluation and prediction stages via caching.
        """
        if (self._out_cache is not None and torch.is_grad_enabled()
                and not modeling_context.is_explaining()):
            raise RuntimeError("Cached RHS outputs are only available in "
                               "inference mode. Please clear cache first via "
                               "`clear_outputs()` before proceeding.")

        if index is not None:
            assert index.dtype == torch.int64
            assert index.device == self.device

        if self._out_cache is not None:
            if index is None:
                return self._out_cache
            elif self._filtered_index is None:
                return self._out_cache[index]
            else:
                local_index = self._map_filtered_index(index)
                return self._out_cache[local_index]

        out: Tensor | None = None

        if self.lookup_embedding is not None:
            if index is None:
                out = self.lookup_embedding(self._filtered_index)
            elif self._filtered_index is None:
                out = self.lookup_embedding(index)
            else:
                # Ensure that indices are valid:
                if not self.emb_mode.use_rhs_feature:
                    self._map_filtered_index(index)
                out = self.lookup_embedding(index)

        if self.encoder is not None and self.projector is not None:
            if self._feats is None:
                if torch.is_grad_enabled():
                    raise RuntimeError(
                        "Raw RHS features are not yet available. Please call "
                        "`fetch_feats(...)` first before proceeding.")

                if index is None:
                    raise RuntimeError(
                        "Raw RHS features can only be requested on-the-fly "
                        "in batch mode. Please call `fetch_feats(...)` first "
                        "before proceeding.")

                if feature_store is None or table_name is None:
                    raise ValueError("Both 'feature_store' and 'table_name' "
                                     "needs to be provided to compute RHS "
                                     "on-the-fly")

                batches: list[list[TensorData]] = []
                for _index in index.cpu().split(FS_BATCH_SIZE):
                    tf = feature_store[table_name, 'tf', _index]
                    tf = tf.to(self.device)
                    assert self.encoder is not None
                    batches.append(self.encoder.forward_fixed(tf))

                feats = [
                    torch_frame.cat([feats[i] for feats in batches], dim=0)
                    for i in range(len(batches[0]))
                ]
            elif index is None and self._filtered_index is None:
                feats = self._feats
            elif index is None and self._filtered_feats:
                feats = self._feats
            elif index is None and self._filtered_index is not None:
                assert not self._filtered_feats
                index = self._filtered_index
                if len(self._feats) > 0:
                    index = index.to(self._feats[0].device)
                feats = [feat[index] for feat in self._feats]
            elif not self._filtered_feats:
                assert index is not None
                # Ensure that indices are valid:
                if self._filtered_index is not None:
                    self._map_filtered_index(index)
                if len(self._feats) > 0:
                    index = index.to(self._feats[0].device)
                feats = [feat[index] for feat in self._feats]
            else:
                assert index is not None
                local_index = self._map_filtered_index(index)
                if len(self._feats) > 0:
                    local_index = local_index.to(self._feats[0].device)
                feats = [feat[local_index] for feat in self._feats]

            if len(feats) > 0:
                num_nodes = feats[0].size(0)
            elif index is not None:
                num_nodes = index.numel()
            else:
                num_nodes = self.num_nodes

            if torch.is_grad_enabled():
                # During training, we assume that we only train against an
                # upper bound of RHS nodes - no need to batch anything:
                batch = torch.arange(num_nodes, device=self.device)
                feats = [feat.to(batch.device) for feat in feats]
                tmp = self.encoder.forward_post_fused_encoders(
                    self.encoder.forward_learnable(feats),
                    batch,
                )
                tmp = self.projector(tmp)
            else:  # Run batch-wise:
                tmps: list[Tensor] = []
                # Since we train against a maximum number of 500k RHS in
                # full-batch mode, it is safe to assume we can handle similar
                # scale during prediction:
                for start in range(0, num_nodes, MAX_BATCH_SIZE):
                    end = min(start + MAX_BATCH_SIZE, num_nodes)
                    batch = torch.arange(end - start, device=self.device)
                    _feats = [
                        feat[start:end].to(batch.device) for feat in feats
                    ]
                    tmp = self.encoder.forward_post_fused_encoders(
                        self.encoder.forward_learnable(_feats),
                        batch,
                    )
                    tmp = self.projector(tmp)
                    tmps.append(tmp)

                if len(tmps) == 0:
                    size = (0, self.embedding_dim)
                    tmp = torch.empty(size, device=self.device)
                elif len(tmps) == 1:
                    tmp = tmps[0]
                else:
                    tmp = torch.cat(tmps, dim=0)

            out = tmp if out is None else out + tmp

        assert out is not None

        if self.distance_measure == 'cosine':
            out = F.normalize(out, dim=1)

        return out


def map_index(index: Tensor, index_map: Tensor) -> Tensor:
    mapped_index = index_map[index]
    if (mapped_index < 0).any():
        raise IndexError("Tried to access invalid RHS nodes")
    return mapped_index
