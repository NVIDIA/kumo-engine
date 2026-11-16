# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import Any, Literal

import pandas as pd
import torch
from torch import Tensor

_SPARSE_REMAP_CHUNK_SIZE = 65_536


class Embedding(torch.nn.Embedding):
    r"""A simple lookup table that stores embeddings.

    Invalid indices will return a zero vector.

    Args:
        num_embeddings: Size of the dictionary of embeddings.
        embedding_dim: Size of each embedding vector.
        init_std: The standard deviation for parameter initialization.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float = 1.0,
        index_map: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        self.init_std = init_std
        self._invalid_index_found = False

        super().__init__(
            num_embeddings,
            embedding_dim,
            **kwargs,
        )

        # The index map stores a mapping from primary key to node index used
        # to index into this embedding. The mask stores invalid indices
        # according to this map (1 = invalid).
        self.index_map = None
        mask = torch.zeros(num_embeddings, dtype=torch.bool)
        if index_map is not None:
            self.index_map = self._normalize_index_map(
                index_map,
                num_embeddings,
                context='index_map',
            )
            mask.fill_(True)
            node_ids = torch.from_numpy(
                self.index_map['arange'].to_numpy(copy=False))
            mask[node_ids] = False
        self.register_buffer('mask', mask)
        self.mask: Tensor

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, std=self.init_std)

    @staticmethod
    def _normalize_index_map(
        index_map: pd.DataFrame,
        num_embeddings: int,
        *,
        context: str,
    ) -> pd.DataFrame:
        columns = list(index_map.columns)
        if len(columns) != 1:
            raise ValueError(f'{context} must contain exactly one column')
        if index_map.index.hasnans:
            raise ValueError(f'{context} contains null primary keys')
        if not index_map.index.is_unique:
            raise ValueError(
                f'{context} contains non-unique primary-key mappings; '
                'retrain the model')

        normalized = index_map.copy(deep=False)
        normalized.columns = ['arange']
        node_ids = normalized['arange']
        if node_ids.hasnans:
            raise ValueError(f'{context} contains null physical node IDs')
        if not pd.api.types.is_integer_dtype(node_ids.dtype):
            raise ValueError(f'{context} physical node IDs must be integral')
        if not node_ids.is_unique:
            raise ValueError(f'{context} contains duplicate physical node IDs')
        if len(node_ids) and node_ids.min() < 0:
            raise ValueError(f'{context} contains negative physical node IDs')
        if len(node_ids) and node_ids.max() >= num_embeddings:
            raise ValueError(
                f'{context} physical node IDs exceed embedding capacity '
                f'{num_embeddings}')
        normalized['arange'] = node_ids.astype('int64', copy=False)
        return normalized

    @torch.no_grad()
    def order_by(
        self,
        index_map: pd.DataFrame,
        default_init: Literal['zero', 'mean'] = 'mean',
        *,
        num_embeddings: int,
    ) -> 'Embedding':
        r"""Re-orders this embedding according to `index_map`, which defines a
        new mapping from primary keys to node indices. Primary keys that are
        no longer present and new primary keys are both dropped, and their
        embeddings are zeroed. In the former case, these primary keys will not
        be requested, and in the latter case, we expect zero embeddings until
        we define a better strategy for handling new embeddings.

        This is necessary since Kumo does not guarantee consistent index maps
        between training and batch prediction, so we need to ensure that our
        embedding is agnostic to node indices.

        Index values are physical node IDs. They may be sparse when ingest
        assigned rows that were not admitted by the primary-key map.
        """
        if self.index_map is None:
            return self

        w = self.weight.data
        device = w.device
        old_index_map = self._normalize_index_map(
            self.index_map,
            w.shape[0],
            context='checkpoint index_map',
        )
        index_map = self._normalize_index_map(
            index_map,
            num_embeddings,
            context='new index_map',
        )

        # Initialize bad indices before copying matched rows. Broadcasting the
        # mean avoids materializing the invalid-row index on the accelerator.
        if default_init not in {'zero', 'mean'}:
            raise ValueError(f'unsupported default_init: {default_init}')
        if default_init == "mean":
            old_node_ids = old_index_map['arange'].to_numpy(copy=False)
            acc_dtype = (torch.float32 if w.dtype
                         in {torch.float16, torch.bfloat16} else w.dtype)
            if (len(old_node_ids) and len(old_node_ids) == w.shape[0]
                    and old_node_ids.min() == 0
                    and old_node_ids.max() == w.shape[0] - 1):
                mean_tensor = w.mean(dim=0, dtype=acc_dtype).to(w.dtype)
            elif len(old_node_ids):
                mean_tensor = torch.zeros(w.shape[1], device=device,
                                          dtype=acc_dtype)
                for start in range(0, len(old_node_ids),
                                   _SPARSE_REMAP_CHUNK_SIZE):
                    chunk = torch.from_numpy(
                        old_node_ids[start:start +
                                     _SPARSE_REMAP_CHUNK_SIZE]).to(device)
                    rows = w.index_select(0, chunk)
                    mean_tensor.add_(rows.sum(dim=0, dtype=acc_dtype))
                mean_tensor.div_(len(old_node_ids))
                mean_tensor = mean_tensor.to(w.dtype)
            else:
                mean_tensor = w.new_zeros(w.shape[1])
            new_w = mean_tensor.expand(num_embeddings, -1).clone()
        else:
            new_w = w.new_zeros((num_embeddings, w.shape[1]))

        # Match and copy rows in bounded chunks. ``get_indexer`` reuses the
        # destination Index's lookup engine without constructing a third,
        # catalog-sized merged DataFrame.
        old_index = old_index_map.index
        new_index = index_map.index
        old_node_ids = old_index_map['arange'].to_numpy(copy=False)
        new_node_ids = index_map['arange'].to_numpy(copy=False)
        self.mask = torch.ones(num_embeddings, device=device, dtype=torch.bool)
        for start in range(0, len(old_index), _SPARSE_REMAP_CHUNK_SIZE):
            end = min(start + _SPARSE_REMAP_CHUNK_SIZE, len(old_index))
            new_positions = new_index.get_indexer(old_index[start:end])
            matched = new_positions >= 0
            if not matched.any():
                continue
            matched_old_ids = old_node_ids[start:end][matched]
            matched_new_ids = new_node_ids[new_positions[matched]]

            for copy_start in range(0, len(matched_old_ids),
                                    _SPARSE_REMAP_CHUNK_SIZE):
                copy_end = copy_start + _SPARSE_REMAP_CHUNK_SIZE
                w_idx = torch.from_numpy(
                    matched_old_ids[copy_start:copy_end]).to(device)
                new_w_idx = torch.from_numpy(
                    matched_new_ids[copy_start:copy_end]).to(device)
                new_w.index_copy_(0, new_w_idx, w.index_select(0, w_idx))
                self.mask[new_w_idx] = False

        # Update weight data:
        self.weight.data = new_w
        self.num_embeddings = num_embeddings
        self.index_map = index_map

        return self

    def forward(self, input: Tensor | None = None) -> Tensor:
        r"""Returns the embedding vectors for a given :obj:`index` vector.
        If :obj:`index` is not specified, returns the full embedding matrix.
        """
        if input is None:
            return self.weight  # Just return the dense embedding matrix here.

        if (not self._invalid_index_found and self.mask[input].sum() > 0):
            self._invalid_index_found = True
            warnings.warn(
                "Requested embeddings of invalid rows. This "
                "suggests that the embedding module was trained "
                "on an earlier snapshot of data. Returning default "
                "vectors for these rows. It is recommended to "
                "re-train the embedding module to include the "
                "new but currently missing rows.", stacklevel=2)

        return super().forward(input)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_embeddings}, '
                f'{self.embedding_dim})')

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        r"""Loads this module's state from a state dictionary. A couple weird
        things to call out:
            * For some reason, the native PyTorch lightning approach to load
                from a checkpoint does not call load_state_dict here, but only
                calls _load_from_state_dict. That's why we've overridden this
                method.
            * The `get_extra_state` and `set_extra_state` methods are called
                as part of the superclass method, and set the extra state on
                the class appropriately. All we do here is make sure we do not
                get compatibility errors when loading a state dict of a module
                that was initialized on a different number of input embeddings.
        """
        # Explicitly set, we will adjust dimensions in the orderby later, if
        # they are incompatible. TODO(manan): raise an exception if `orderby`
        # is not called and dimensions are incompatible here
        with torch.no_grad():
            for param in ['weight', 'mask']:
                key = f'{prefix}{param}'
                param_attr = getattr(self, param)
                param_attr.data = state_dict[key]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def get_extra_state(self) -> Any:
        return {'index_map': self.index_map}

    def set_extra_state(self, state: Any) -> None:
        index_map = state['index_map']
        self.index_map = (None
                          if index_map is None else self._normalize_index_map(
                              index_map,
                              self.weight.shape[0],
                              context='checkpoint index_map',
                          ))
