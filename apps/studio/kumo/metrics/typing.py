# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Union

from prometheus_client import Counter, Gauge, Histogram, Summary

MetricType = Union[Summary, Histogram, Counter, Gauge]


class Metrics(Enum):
    TABLE_MATERIALIZATION_TIME = "table_materialization_time"
    TABLE_ENCODING_TIME = "table_encoding_time"
    TABLE_READ_TIME = "table_read_time"
    COLUMN_ENCODING_TIME = "column_encoding_time"
    GRAPH_ENGINE_DF_TO_SST_TIME = "graph_engine_df_to_sst_time"
    GRAPH_ENGINE_SST_LOAD_TIME = "graph_engine_sst_load_time"

    # DataLoader Metrics #####################################################
    FEATURE_STORE_GET_TIME = "feature_store_get_time"
    FEATURE_STORE_BATCH_SIZE = "feature_store_batch_size"
    FEATURE_STORE_BATCH_ENCODING_TIME = "feature_store_batch_encoding_time"
