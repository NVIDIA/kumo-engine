# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# isort: off
from kumo.pquery.pquery_validators.generation_validator import (
    GenerationValidator, )
# isort: on
from kumo.pquery.stats_manager import StatsManager


class PredTableValidator(GenerationValidator):
    """Runs validations during prediction table generation

    Args:
        stats_manager (StatsManager): `StatsManager` that has been used to
            cache stats.
    """
    def __init__(
        self,
        stats_manager: StatsManager,
    ):
        super(PredTableValidator, self).__init__(stats_manager)
