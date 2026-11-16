# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Any

from kumo_core.automl.searcher import PlanT, Searcher


class RandomSearcher(Searcher[PlanT]):
    """Sample random experiment instantiations from a search plan."""
    def sample_value(self, key: str, value: list[Any]) -> Any:
        return random.choice(value)

    def analyze(self, path: str) -> None:
        pass
