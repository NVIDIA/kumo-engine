# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import torch

from kumo_core.embedding_models import _fingerprint, _VocabularyModel

WORDS = ["the", ",", ".", "of", "to", "and", "in", "a"]


class TinyModel(_VocabularyModel):
    """Random vectors for a few words, registered as "glove.test-<dim>d"."""
    family = "glove.test/1"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @property
    def label(self) -> str:
        return f"glove.test {self.dim}d"

    def _read_vocabulary(self) -> pd.Index:
        return pd.Index(WORDS, dtype=object, name="index")

    def vectors(self) -> np.ndarray:
        generator = torch.Generator().manual_seed(0)
        return torch.randn(len(WORDS), self.dim, generator=generator).numpy()

    def _compute_fingerprint(self) -> str:
        return _fingerprint(self.vocabulary(), self.vectors())
