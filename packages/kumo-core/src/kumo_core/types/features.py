# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import numpy as np
import pandas as pd
import torch_frame
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor

TensorData = Tensor | MultiNestedTensor | MultiEmbeddingTensor
FeatureStoreData = Tensor | TensorFrame | np.ndarray | pd.DataFrame

TimeUnit = Literal['seconds', 'minutes', 'hours', 'days', 'weeks', 'months']

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24
SECONDS_IN_WEEK = SECONDS_IN_DAY * 7
SECONDS_IN_MONTH = SECONDS_IN_DAY * 30

NANOSECONDS_IN_SECONDS = 1000**3
NANOSECONDS_IN_MINUTE = NANOSECONDS_IN_SECONDS * SECONDS_IN_MINUTE
NANOSECONDS_IN_HOUR = NANOSECONDS_IN_SECONDS * SECONDS_IN_HOUR
NANOSECONDS_IN_DAY = NANOSECONDS_IN_SECONDS * SECONDS_IN_DAY
NANOSECONDS_IN_WEEK = NANOSECONDS_IN_SECONDS * SECONDS_IN_WEEK
NANOSECONDS_IN_MONTHS = NANOSECONDS_IN_SECONDS * SECONDS_IN_MONTH

# Used for serializing timestamps as strings.
# TODO Ideally, we want to never import this anywhere since json serialization
# should take care of it. Revisit as soon as possible.
TIMESTAMP_PREFIX = 'kumo_timestamp_'

FROM_TORCH_FRAME_STYPE = {
    torch_frame.numerical: Stype.numerical,
    torch_frame.categorical: Stype.categorical,
    torch_frame.text_embedded: Stype.text,
    torch_frame.text_tokenized: Stype.text,
    torch_frame.multicategorical: Stype.multicategorical,
    torch_frame.sequence_numerical: Stype.sequence,
    torch_frame.timestamp: Stype.timestamp,
    torch_frame.image_embedded: Stype.sequence,
    torch_frame.embedding: Stype.sequence,
}
