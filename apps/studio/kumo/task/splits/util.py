# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from kumoapi.train import TrainingStage

SPLIT_STAGES = [TrainingStage.TRAIN, TrainingStage.VAL, TrainingStage.TEST]
SMALLEST_OFFSET_UNIT = pd.DateOffset(nanoseconds=1)
