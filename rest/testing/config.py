# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils import kumo_dataclass


@kumo_dataclass
class TestFunctionCleanupOptions:
    should_reset_rest_db: bool = True
    should_reset_ml_db: bool = True
    should_reset_data_backend_cache: bool = False
    should_delete_data_persist_dir: bool = False
