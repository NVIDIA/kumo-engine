# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import weakref
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    from kumo_core.embedding_models import TokenEmbeddingModel


class GloveDisabledError(RuntimeError):
    pass


class VocabularyIndex:
    def __init__(self, words: pd.Index) -> None:
        # Lazy: kumo_core.encoders imports this module.
        from kumo_core.encoders.index_mapping import Mapper

        self.words = words
        keep = words.notna() & ~words.duplicated(keep="last")
        self._mapper = Mapper(words[keep])
        self._rows = torch.from_numpy(np.flatnonzero(keep))

    def __len__(self) -> int:
        return len(self.words)

    def map_index(self, values, return_mask=False):
        codes = self._mapper.map(values)
        mask = codes >= 0
        rows = self._rows[codes[mask]]
        return (rows, mask) if return_mask else rows

    def tokenize(self, values: pd.Series) -> tuple[Tensor, Tensor]:
        rowptr, codes = self._mapper.tokenize(values)
        return rowptr, self._rows[codes]


class TokenEmbeddingTable(torch.nn.Module):
    # One table per model identity and process, shared by the encoders using
    # it, which own it: it goes with the last of them.
    instances: "weakref.WeakValueDictionary[tuple[str, str], TokenEmbeddingTable]" = (
        weakref.WeakValueDictionary())
    _emb: Tensor

    def __init__(self, model: "TokenEmbeddingModel") -> None:
        super().__init__()
        self.model = model

    @classmethod
    def instance(cls, model: "TokenEmbeddingModel") -> "TokenEmbeddingTable":
        key = (model.family, model.fingerprint())
        table = cls.instances.get(key)
        if table is None:
            table = cls.instances[key] = cls(model)
        return table

    @property
    def emb(self) -> Tensor:
        if not hasattr(self, "_emb"):
            raise RuntimeError("Text embeddings have not been initialized")
        return self._emb

    @property
    def is_initialized(self) -> bool:
        return hasattr(self, "_emb")

    def initialize(self, device: torch.device | None = None) -> "TokenEmbeddingTable":
        if not hasattr(self, "_emb"):
            # Row-major, or every lookup copies the table.
            vectors = np.require(self.model.vectors(), np.float32, ["C", "W"])
            shape = (len(self.model.vocabulary()), self.model.dim)
            if vectors.shape != shape:
                raise ValueError(
                    f"{self.model!r}.vectors() has shape {vectors.shape}, but "
                    f"it must be {shape}: one row of dim={shape[1]} values "
                    f"per word of vocabulary().")
            self.register_buffer("_emb", torch.from_numpy(vectors),
                                 persistent=False)
        return self.to(device)

    def cleanup(self) -> "TokenEmbeddingTable":
        if hasattr(self, "_emb"):
            delattr(self, "_emb")
        return self
