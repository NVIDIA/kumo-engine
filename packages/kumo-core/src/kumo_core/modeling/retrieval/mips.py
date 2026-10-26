# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
import warnings
from abc import ABC, abstractmethod
from types import ModuleType

import torch
from kumo_core.errors import TrainingValidationError
from torch import Tensor
from torch_geometric.utils import scatter

import logging

logger = logging.getLogger(__name__)

# TODO: Change this back to 2048 when upgrading faiss
# as newer versions fixed the validation, e.g., see:
# https://github.com/facebookresearch/faiss/blob/v1.12.0/faiss/gpu/impl/IndexUtils.cu#L30  # noqa: E501
MAX_K = 2047  # Upper `faiss` limit


class FaissUnavailableError(TrainingValidationError):
    """The selected retrieval device lacks a compatible FAISS backend."""


def require_faiss(device: torch.device | None = None) -> ModuleType:
    """Load optional FAISS without hiding errors in an installed backend."""
    try:
        faiss = importlib.import_module('faiss')
    except ModuleNotFoundError as error:
        if error.name != 'faiss':
            raise
        raise FaissUnavailableError(
            title="FAISS is required for link prediction retrieval",
            message="Install faiss-cpu for CPU execution or a compatible "
            "faiss-gpu build for CUDA execution. A compatible source-built "
            "FAISS installation is also supported.",
        ) from error

    if device is not None and device.type == 'cuda' and (
            not hasattr(faiss, 'StandardGpuResources')
            or not hasattr(faiss, 'knn_gpu')):
        raise FaissUnavailableError(
            title="GPU FAISS is required for CUDA link prediction",
            message="The installed FAISS backend lacks GPU support. Install "
            "a compatible GPU FAISS build (including a source build on ARM) "
            "or explicitly select CPU execution with faiss-cpu.",
        )
    # This patches FAISS search for Torch tensors only when retrieval is used.
    importlib.import_module('faiss.contrib.torch_utils')
    return faiss


class MIPSStructure(ABC):
    r"""Interface for Maximum Inner Product Search (MIPS).

    Args:
        key: The key embeddings.
    """
    def __init__(self, key: Tensor) -> None:
        if key.dim() != 2:
            raise ValueError(f"'key' needs to be two-dimensional "
                             f"(got {key.dim()} dimensions)")

        self._key = key.detach()

    def search(
        self,
        query: Tensor,
        k: int,
        exclude: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Gets the nearest neighbors of query embeddings.

        Returns the similarities of the nearest neighbors and their indices.

        Args:
            query: The data points to search.
            k: The number of nearest neighbors to return.
            exclude: Specifies pairs of vectors ``(query_index, key_index)`` to
                exclude from the search result.
        """
        if query.dim() != 2:
            raise ValueError(f"'query' needs to be two-dimensional "
                             f"(got {query.dim()} dimensions)")

        if k < 1:
            raise ValueError(f"'k' needs to be greater than 0 (got {k})")

        if k > MAX_K:
            raise ValueError(f"'k' exceeds upper limit of {MAX_K} (got {k})")

        if self.embedding_dim != query.size(1):
            raise ValueError(f"'key' and 'query' needs have have "
                             f"dimensionality (got {self.embedding_dim} and "
                             f"{query.size(1)})")

        if self.dtype != query.dtype:
            raise ValueError(f"'key' and 'query' needs have same data type "
                             f"(got {self.dtype} and {query.dtype})")

        if self.device != query.device:
            raise ValueError(f"'key' and 'query' needs to be on the same "
                             f"device (got {self.device} and {query.device})")

        k = min(k, self.num_keys)

        if k == 0 or query.numel() == 0:
            score = query.new_empty(query.size(0), k)
            index = query.new_empty(query.size(0), k, dtype=torch.long)
        else:
            score, index = self._search(query.detach(), k, exclude)

        score = score.contiguous()
        index = index.contiguous()

        if query.requires_grad:
            query = query.repeat_interleave(index.size(1), dim=0)
            key = self._key[index.view(-1)]
            mask = score.isinf()
            score = (query * key).sum(dim=-1).view_as(score)
            score = score.masked_fill(mask, float('-inf'))

        return score, index

    @property
    def num_keys(self) -> int:
        r"""The number of keys."""
        return self._key.size(0)

    @property
    def embedding_dim(self) -> int:
        r"""The dimensionality of key embeddings."""
        return self._key.size(1)

    @property
    def device(self) -> torch.device:
        r"""The device of key embeddings."""
        return self._key.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The data type of key embeddings."""
        return self._key.dtype

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self._key.size(0)}, '
                f'{self._key.size(1)})')

    @abstractmethod
    def _search(
        self,
        query: Tensor,
        k: int,
        exclude: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        pass


class MIPSFaiss(MIPSStructure):
    r"""MIPS using PyTorch on macOS CPU and Faiss for other backends.

    Metal's Python API accepts host arrays, so inputs and search results
    pass through CPU memory. Keys are copied into the Metal index once.
    """
    def __init__(self, key: Tensor) -> None:
        super().__init__(key=key)

        faiss = require_faiss(key.device)
        self._cpu_index: MIPSBruteForce | None = None
        self._res = None
        self._metal_index = None
        if sys.platform == 'darwin' and self.device.type == 'cpu':
            self._cpu_index = MIPSBruteForce(self._key)
            return

        if self._key.is_cuda:
            self._res = faiss.StandardGpuResources()
            self._res.setDefaultNullStreamAllDevices()
            self._res.setTempMemory(0)
        elif self.device.type == 'mps' and self.num_keys > 0:
            if ('MAC_METAL' not in faiss.get_compile_options()
                    or faiss.get_num_gpus() == 0):
                raise RuntimeError(
                    'MPS retrieval requires a Metal-enabled Faiss build and '
                    'an available Metal device. Install faiss-cpu==1.14.3 '
                    'on Apple Silicon or use CPU tensors.')
            self._res = faiss.StandardGpuResources()
            self._metal_index = faiss.index_cpu_to_gpu(
                self._res, 0, faiss.IndexFlatIP(self.embedding_dim))
            self._metal_index.add(
                self._key.to(device='cpu', dtype=torch.float32).numpy())

    def _search(
        self,
        query: Tensor,
        k: int,
        exclude: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:

        if self._cpu_index is not None:
            return self._cpu_index.search(query, k, exclude)

        query_k = k
        if exclude is not None:
            # Over-retrieve `k` to compensate for exclusions:
            query_k += int(
                scatter(
                    torch.ones_like(exclude[0]),
                    exclude[0],
                    dim_size=query.size(0),
                    reduce='sum',
                ).max().item())
            query_k = min(query_k, self.num_keys)
            if query_k > MAX_K:
                logger.warning(
                    f"Capping 'query_k' ({query_k}) to faiss' upper limit "
                    f"({MAX_K}). This may cause some relevant items to not be "
                    f"retrieved.")
                query_k = MAX_K

        faiss = require_faiss(query.device)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='TypedStorage')
            if self._metal_index is not None:
                score, index = self._metal_index.search(
                    query.to(device='cpu', dtype=torch.float32).numpy(),
                    query_k)
                score = torch.from_numpy(score).to(
                    device=self.device, dtype=self.dtype)
                index = torch.from_numpy(index).to(self.device)
            elif self._res is not None:
                score, index = faiss.knn_gpu(self._res, query, self._key,
                                             query_k,
                                             metric=faiss.METRIC_INNER_PRODUCT)
            else:
                score, index = faiss.knn(query.numpy(), self._key.numpy(),
                                         query_k,
                                         metric=faiss.METRIC_INNER_PRODUCT)
                score = torch.from_numpy(score)
                index = torch.from_numpy(index)

        if exclude is not None:
            # Drop indices to exclude by converting to 1D-array:
            flat_exclude = self.num_keys * exclude[0] + exclude[1]
            offset = torch.arange(
                start=0,
                end=self.num_keys * index.size(0),
                step=self.num_keys,
                device=index.device,
            ).view(-1, 1)
            flat_index = (index + offset).view(-1)

            mask = torch.isin(flat_index, flat_exclude).view_as(score)

            if mask.any():
                score[mask] = float('-inf')
                score, perm = score.sort(dim=-1, descending=True)
                index = index.gather(dim=-1, index=perm)

            score = score[:, :k]
            index = index[:, :k]

        return score, index


class MIPSBruteForce(MIPSStructure):
    r"""MIPS based on brute-force search."""
    def _search(
        self,
        query: Tensor,
        k: int,
        exclude: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:

        score = query @ self._key.t()

        if exclude is not None:  # Do not search over excluded indices:
            score[exclude[0], exclude[1]] = float('-inf')

        score, index = score.topk(k, dim=-1)

        return score, index
