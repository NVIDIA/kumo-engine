# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.pquery.stats_manager import StatsManager


class GenerationValidator:
    """Runs validations during training table or prediction table generation.
    This is a base class for shared validations

    Args:
        stats_manager (StatsManager): `StatsManager` that has been used to
            cache stats.

    """
    def __init__(
        self,
        stats_manager: StatsManager,
    ):
        self.stats_manager = stats_manager
