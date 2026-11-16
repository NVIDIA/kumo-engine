# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

LAG_TIMESTEPS: int = 1
AUTOREGRESSION_SUFFIX: str = '_prev_kumo_suffix_'
AUTOREGRESSION_COL: str = 'autoregression_labels'
FUTURE_LABELS_COL: str = 'future_labels'

YOY_COL = 'year_over_year_labels'
PADDING_SIZE = 10
ONE_YEAR_AGO_OFFSET_DAYS = 364
